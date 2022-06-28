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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/hexagon/builders/reshape_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

void PopulateOutputShapeFromTensor(const TfLiteTensor* shape_tensor,
                                   std::vector<int>* output_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "PopulateOutputShapeFromTensor");

  for (int i = 0; i < shape_tensor->dims->data[0]; ++i) {
    output_shape->push_back(shape_tensor->data.i32[i]);
  }
}

void PopulateShapeFromParam(const TfLiteReshapeParams* params,
                            std::vector<int>* output_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "PopulateShapeFromParam");

  // The function is returned above this line if the shape tensor is usable.
  // Now fallback to the shape parameter in `TfLiteReshapeParams`.
  int num_dimensions = params->num_dimensions;
  if (num_dimensions == 1 && params->shape[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    num_dimensions = 0;
  }
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->push_back(params->shape[i]);
  }
}
}  // namespace

TfLiteStatus ReshapeOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                const TfLiteIntArray* outputs,
                                                TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "ReshapeOpBuilder::PopulateSubGraph");

  // Input data tensor.
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));

  // Output shape.
  TfLiteTensor* shape_tensor = nullptr;
  bool output_shape_is_dynamic = false;
  if (inputs->size == 2) {
    shape_tensor = &context->tensors[inputs->data[1]];
    bool is_shape_tensor =
        (shape_tensor->dims->size == 1 && shape_tensor->type == kTfLiteInt32);
    // If tensor shape is dynamic, pass it along directly.
    if (shape_tensor->allocation_type != kTfLiteMmapRo && is_shape_tensor) {
      output_shape_is_dynamic = true;
      AddInput(graph_builder_->GetHexagonTensorId(inputs->data[1]));
    }
    if (!is_shape_tensor) {
      shape_tensor = nullptr;
    }
  }
  if (!output_shape_is_dynamic) {
    if (shape_tensor) {
      PopulateOutputShapeFromTensor(shape_tensor, &output_shape_);
    } else {
      const TfLiteReshapeParams* reshape_params =
          reinterpret_cast<const TfLiteReshapeParams*>(builtin_data_);
      PopulateShapeFromParam(reshape_params, &output_shape_);
    }
    int num_elements_in_shape = static_cast<int>(output_shape_.size());
    output_shape_shape_ = {1, 1, 1, num_elements_in_shape};
    auto* shape_node = graph_builder_->AddConstNodeWithData(
        output_shape_shape_.data(),
        reinterpret_cast<char*>(output_shape_.data()),
        sizeof(int) * num_elements_in_shape);
    AddInput(TensorID(shape_node->GetID(), 0));
  }

  // Hexagon output for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});

  return kTfLiteOk;
}

TfLiteStatus ReshapeOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_3(mht_3_v, 284, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "ReshapeOpBuilder::RegisterOutputs");

  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ReshapeOpBuilder::~ReshapeOpBuilder() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_4(mht_4_v, 294, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "ReshapeOpBuilder::~ReshapeOpBuilder");
}

OpBuilder* CreateReshapeBuilder(GraphBuilder* graph_builder, int op_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPSreshape_builderDTcc mht_5(mht_5_v, 299, "", "./tensorflow/lite/delegates/hexagon/builders/reshape_builder.cc", "CreateReshapeBuilder");

  return new ReshapeOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
