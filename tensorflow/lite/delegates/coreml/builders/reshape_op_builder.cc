#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/delegates/coreml/builders/reshape_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ReshapeOpBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::DebugName");

  if (debug_name_.empty()) {
    SetDebugName("ReshapeOpBuilder", node_id_);
  }
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ReshapeOpBuilder::Build() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::Build");

  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());
  for (int dim : shape_) {
    layer_->mutable_reshape()->add_targetshape(dim);
  }
  if (need_transpose_)
    layer_->mutable_reshape()->set_mode(
        CoreML::Specification::ReshapeLayerParams::CHANNEL_LAST);
  return layer_.release();
}

void ReshapeOpBuilder::SetShapeFromTensor(const TfLiteTensor* output_shape,
                                          const TfLiteIntArray* input_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::SetShapeFromTensor");

  TfLiteIntArray* shape = TfLiteIntArrayCreate(output_shape->dims->data[0]);
  std::memcpy(shape->data, GetTensorData<int>(output_shape),
              shape->size * sizeof(int));

  SetShapeFromIntArray(shape, input_shape);
  TfLiteIntArrayFree(shape);
}

void ReshapeOpBuilder::SetShapeFromIntArray(const TfLiteIntArray* output_shape,
                                            const TfLiteIntArray* input_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_3(mht_3_v, 239, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::SetShapeFromIntArray");

  // ignore first dimension (batch)
  std::copy(output_shape->data + 1, output_shape->data + output_shape->size,
            std::back_inserter(shape_));

  int64_t reshape_size = 1;
  int negative_index = -1;
  for (int i = 0; i < shape_.size(); ++i) {
    if (shape_[i] == -1) {
      negative_index = i;
    } else {
      reshape_size *= shape_[i];
    }
  }
  if (negative_index >= 0) {
    int64_t input_size = NumElements(input_shape);
    shape_[negative_index] = input_size / reshape_size;
  }

  if (shape_.size() == 2) {
    shape_ = {shape_[1], 1, shape_[0]};
  } else if (shape_.size() == 3) {
    shape_ = {shape_[2], shape_[0], shape_[1]};
  }
  // When channel dimension is changed, reshape should be done with HWC layout.
  if (shape_[0] != input_shape->data[input_shape->size - 1]) {
    need_transpose_ = true;
  }
}

TfLiteStatus ReshapeOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                              TfLiteContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_4(mht_4_v, 273, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::RegisterInputs");

  AddInput(inputs->data[0]);

  if (inputs->size == 2) {
    SetShapeFromTensor(&context->tensors[inputs->data[1]],
                       context->tensors[inputs->data[0]].dims);
  } else {
    const auto* params = reinterpret_cast<TfLiteReshapeParams*>(builtin_data_);
    TfLiteIntArray* output_shape = TfLiteIntArrayCreate(params->num_dimensions);
    std::memcpy(output_shape->data, params->shape,
                params->num_dimensions * sizeof(int));

    SetShapeFromIntArray(output_shape, context->tensors[inputs->data[0]].dims);
    TfLiteIntArrayFree(output_shape);
  }
  return kTfLiteOk;
}

TfLiteStatus ReshapeOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_5(mht_5_v, 295, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "ReshapeOpBuilder::RegisterOutputs");

  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

bool IsReshapeOpSupported(const TfLiteRegistration* registration,
                          const TfLiteNode* node, TfLiteContext* context,
                          int coreml_version) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_6(mht_6_v, 305, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "IsReshapeOpSupported");

  if (coreml_version >= 3) {
    return false;
  }
  if (node->inputs->size == 1) {
    const auto* params =
        reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);
    return params->num_dimensions == 3 || params->num_dimensions == 4;
  }

  const int kShapeTensor = 1;
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShapeTensor, &shape));
  if (shape->allocation_type != kTfLiteMmapRo) {
    TF_LITE_KERNEL_LOG(context, "Reshape has non-const shape.");
    return false;
  }
  const bool is_shape_tensor =
      shape->dims->size == 1 && shape->type == kTfLiteInt32;
  return is_shape_tensor &&
         (shape->dims->data[0] == 3 || shape->dims->data[0] == 4);
}

OpBuilder* CreateReshapeOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSreshape_op_builderDTcc mht_7(mht_7_v, 331, "", "./tensorflow/lite/delegates/coreml/builders/reshape_op_builder.cc", "CreateReshapeOpBuilder");

  return new ReshapeOpBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
