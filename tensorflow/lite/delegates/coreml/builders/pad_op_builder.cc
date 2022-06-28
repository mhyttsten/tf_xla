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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/pad_op_builder.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& PadOpBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::DebugName");

  if (!debug_name_.empty()) return debug_name_;
  SetDebugName(padding_type_ == PadType::kPad ? "PadOpBuilder (PAD)"
                                              : "PadOpBuilder (MIRROR_PAD)",
               node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* PadOpBuilder::Build() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::Build");

  layer_->set_name(DebugName());
  if (padding_type_ == PadType::kPad) {
    layer_->mutable_padding()->mutable_constant();
  } else if (padding_type_ == PadType::kMirrorPad) {
    layer_->mutable_padding()->mutable_reflection();
  }
  return layer_.release();
}

// padding is d x 2 tensor, where d is the dimension of input.
// only paddings for width and height are considered.
void PadOpBuilder::SetPadding(const TfLiteTensor* padding) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::SetPadding");

  const int32_t* padding_data = GetTensorData<int32_t>(padding);
  for (int i = 1; i <= 2; ++i) {
    auto* borderamount = layer_->mutable_padding()
                             ->mutable_paddingamounts()
                             ->add_borderamounts();
    borderamount->set_startedgesize(padding_data[i * 2]);
    borderamount->set_endedgesize(padding_data[i * 2 + 1]);
  }
}

void PadOpBuilder::SetConstantValue(const TfLiteTensor* constant_value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::SetConstantValue");

  layer_->mutable_padding()->mutable_constant()->set_value(
      GetTensorData<float>(constant_value)[0]);
}

TfLiteStatus PadOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                          TfLiteContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_4(mht_4_v, 246, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::RegisterInputs");

  if (!(inputs->size == 2 || inputs->size == 3)) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to Padding!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  SetPadding(GetInput(context, tflite_node_, 1));
  if (inputs->size == 3) {
    SetConstantValue(GetInput(context, tflite_node_, 2));
  }

  return kTfLiteOk;
}

TfLiteStatus PadOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                           TfLiteContext* context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "PadOpBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to Padding!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreatePadOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_6(mht_6_v, 276, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "CreatePadOpBuilder");

  return new PadOpBuilder(graph_builder, PadType::kPad);
}

OpBuilder* CreateMirrorPadOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_7(mht_7_v, 283, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "CreateMirrorPadOpBuilder");

  return new PadOpBuilder(graph_builder, PadType::kMirrorPad);
}

bool IsPadOpSupported(const TfLiteRegistration* registration,
                      const TfLiteNode* node, TfLiteContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_8(mht_8_v, 291, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "IsPadOpSupported");

  // padding is d x 2 tensor, where d is the dimension of input.
  const TfLiteTensor* padding;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &padding));
  if (!IsConstantTensor(padding)) {
    TF_LITE_KERNEL_LOG(context,
                       "%s: Only constant padding is supported for PAD.",
                       padding->name);
    return false;
  }
  if (padding->dims->data[0] != 4 || padding->dims->data[1] != 2) {
    TF_LITE_KERNEL_LOG(context, "%s: Only 4D inputs are supported for PAD.",
                       padding->name);
    return false;
  }
  const int32_t* padding_data = GetTensorData<int32_t>(padding);
  if (!(padding_data[0] == 0 && padding_data[1] == 0)) {
    TF_LITE_KERNEL_LOG(
        context, "%s: Padding for batch dimension is not supported in PAD.",
        padding->name);
    return false;
  }

  if (!(padding_data[6] == 0 && padding_data[7] == 0)) {
    TF_LITE_KERNEL_LOG(
        context, "%s: Padding for channel dimension is not supported in PAD.",
        padding->name);
    return false;
  }
  return true;
}

bool IsMirrorPadOpSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSpad_op_builderDTcc mht_9(mht_9_v, 327, "", "./tensorflow/lite/delegates/coreml/builders/pad_op_builder.cc", "IsMirrorPadOpSupported");

  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);
  if (params->mode != kTfLiteMirrorPaddingReflect) {
    TF_LITE_KERNEL_LOG(context,
                       "Only REFLECT mode is supported for MIRROR_PAD.");
    return false;
  }
  return IsPadOpSupported(registration, node, context);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
