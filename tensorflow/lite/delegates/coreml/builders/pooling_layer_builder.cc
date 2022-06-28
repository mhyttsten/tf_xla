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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& PoolingLayerBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "PoolingLayerBuilder::DebugName");

  if (!debug_name_.empty()) return debug_name_;
  switch (pooling_type_) {
    case kTfLiteBuiltinAveragePool2d:
      SetDebugName("PoolingLayerBuilder (AVERAGE)", node_id_);
      break;
    case kTfLiteBuiltinMaxPool2d:
      SetDebugName("PoolingLayerBuilder (MAX)", node_id_);
      break;
    case kTfLiteBuiltinL2Pool2d:
      SetDebugName("PoolingLayerBuilder (L2, unsupported)", node_id_);
      break;
    case kTfLiteBuiltinMean:
      SetDebugName("PoolingLayerBuilder (MEAN)", node_id_);
      break;
    default:
      SetDebugName("PoolingLayerBuilder (ERROR)", node_id_);
  }
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* PoolingLayerBuilder::Build() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "PoolingLayerBuilder::Build");

  layer_->set_name(DebugName());
  auto* pooling_params = layer_->mutable_pooling();

  if (pooling_type_ == kTfLiteBuiltinMean) {
    pooling_params->set_type(
        CoreML::Specification::PoolingLayerParams::AVERAGE);
    pooling_params->set_globalpooling(true);
    return layer_.release();
  }

  const TfLitePoolParams* params =
      reinterpret_cast<const TfLitePoolParams*>(builtin_data_);
  pooling_params->mutable_stride()->Add(params->stride_height);
  pooling_params->mutable_stride()->Add(params->stride_width);
  pooling_params->mutable_kernelsize()->Add(params->filter_height);
  pooling_params->mutable_kernelsize()->Add(params->filter_width);

  if (params->padding == kTfLitePaddingSame) {
    pooling_params->mutable_same();
  } else {
    pooling_params->mutable_valid();
  }

  switch (pooling_type_) {
    case kTfLiteBuiltinAveragePool2d:
      pooling_params->set_type(
          CoreML::Specification::PoolingLayerParams::AVERAGE);
      pooling_params->set_avgpoolexcludepadding(true);
      break;
    case kTfLiteBuiltinMaxPool2d:
      pooling_params->set_type(CoreML::Specification::PoolingLayerParams::MAX);
      break;
    case kTfLiteBuiltinL2Pool2d:
      // TODO(b/145873272) implement L2 pooling
      // NOLINTNEXTLINE: minimize absl usage
      fprintf(stderr, "L2 pooling is not supported yet.\n");
      return nullptr;
    default:
      // NOLINTNEXTLINE: minimize absl usage
      fprintf(stderr, "Unexpected pooling type.\n");  // Should not reach here.
      return nullptr;
  }

  // TODO(b/145582958): Add padding values.
  // TODO(b/145582958): Handle fused activation function.
  return layer_.release();
}

TfLiteStatus PoolingLayerBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                 TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_2(mht_2_v, 274, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "PoolingLayerBuilder::RegisterInputs");

  if (pooling_type_ == kTfLiteBuiltinMean) {
    if (inputs->size != 2) {
      TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to Mean!.");
      return kTfLiteError;
    }
  } else if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to Pooling!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus PoolingLayerBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_3(mht_3_v, 292, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "PoolingLayerBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to Pooling!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateAveragePool2dOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_4(mht_4_v, 304, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "CreateAveragePool2dOpBuilder");

  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinAveragePool2d);
}

OpBuilder* CreateMaxPool2dOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_5(mht_5_v, 311, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "CreateMaxPool2dOpBuilder");

  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinMaxPool2d);
}

OpBuilder* CreateMeanOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_6(mht_6_v, 318, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "CreateMeanOpBuilder");

  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinMean);
}

// Only supports averaging over H and W dimensions, as
bool IsMeanOpSupported(const TfLiteRegistration* registration,
                       const TfLiteNode* node, TfLiteContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpooling_layer_builderDTcc mht_7(mht_7_v, 327, "", "./tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc", "IsMeanOpSupported");

  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* axis = GetInput(context, node, 1);
  const auto* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);

  if (!params->keep_dims) {
    TF_LITE_KERNEL_LOG(context, "keep_dims should be true for Mean op.");
    return false;
  }
  if (input->dims->size != 4) {
    TF_LITE_KERNEL_LOG(context, "Mean op is only supported for 4D input.");
    return false;
  }
  const int* axis_data = GetTensorData<int>(axis);
  std::vector<bool> axis_mask = {false, true, true, false};
  for (int i = 0; i < axis->dims->data[0]; ++i) {
    if (!axis_mask[(axis_data[i] + 4) % 4]) {
      TF_LITE_KERNEL_LOG(context,
                         "Mean op should reduce for H and W dimensions.");
      return false;
    }
  }
  return true;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
