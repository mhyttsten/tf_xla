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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/activation_layer_builder.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/threshold_layer_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ActivationLayerBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "ActivationLayerBuilder::DebugName");

  if (debug_name_.empty()) SetDebugName("ActivationLayerBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ActivationLayerBuilder::Build() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "ActivationLayerBuilder::Build");

  layer_->set_name(DebugName());
  switch (activation_) {
    // ActNone is used for sclalar multiplication (linear activation)
    case kTfLiteActNone:
      layer_->mutable_activation()->mutable_linear()->set_alpha(alpha_);
      break;
    case kTfLiteActRelu:
      layer_->mutable_activation()->mutable_relu();
      break;
    // Relu1 and Relu6 layers are fully composed in PopulateSubgraph().
    case kTfLiteActReluN1To1:  // clip(-1, 1)
      layer_->mutable_unary()->set_alpha(-1);
      layer_->mutable_unary()->set_type(
          CoreML::Specification::UnaryFunctionLayerParams::THRESHOLD);
      break;
    case kTfLiteActRelu6:  // clip(0, 6)
      layer_->mutable_activation()->mutable_relu();
      break;
    case kTfLiteActTanh:
      layer_->mutable_activation()->mutable_tanh();
      break;
    case kTfLiteActSigmoid:
      layer_->mutable_activation()->mutable_sigmoid();
      break;
    // TODO(taeheej): signbit is not implemented.
    default:
      fprintf(stderr, "Activation %d is not supported.\n", activation_);
      break;
  }
  return layer_.release();
}

TfLiteStatus ActivationLayerBuilder::PopulateSubgraph(TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "ActivationLayerBuilder::PopulateSubgraph");

  if (!(activation_ == kTfLiteActRelu6 || activation_ == kTfLiteActReluN1To1)) {
    builder_output_ = AddOutput();
    return kTfLiteOk;
  }

  // Relu1: Threshold(-1) -> Threshold(-1) with scale: -1 -> Negation
  // Relu6: ReLU -> Threshold(-6) with scale: -1 -> Negation
  const int relu_threshold = activation_ == kTfLiteActRelu6 ? 6 : 1;
  ThresholdLayerBuilder* threshold_builder =
      reinterpret_cast<ThresholdLayerBuilder*>(
          graph_builder_->AddBuilder(CreateThresholdLayerBuilder, nullptr));

  threshold_builder->SetAlpha(-relu_threshold);
  threshold_builder->SetScale(-1);

  threshold_builder->AddInput(AddOutput());

  ActivationLayerBuilder* negation_builder =
      reinterpret_cast<ActivationLayerBuilder*>(
          graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
  negation_builder->SetActivation(kTfLiteActNone);
  negation_builder->SetAlpha(-1);

  negation_builder->AddInput(threshold_builder->AddOutput());
  builder_output_ = negation_builder->AddOutput();
  return kTfLiteOk;
}

TfLiteStatus ActivationLayerBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_3(mht_3_v, 273, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "ActivationLayerBuilder::RegisterInputs");

  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Activation: Wrong # of inputs!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus ActivationLayerBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_4(mht_4_v, 286, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "ActivationLayerBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Activation: Wrong # of outputs!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateActivationLayerBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_5(mht_5_v, 298, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateActivationLayerBuilder");

  return new ActivationLayerBuilder(graph_builder);
}

OpBuilder* CreateLogisticOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_6(mht_6_v, 305, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateLogisticOpBuilder");

  return new ActivationLayerBuilder(graph_builder, kTfLiteActSigmoid);
}

OpBuilder* CreateReluOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_7(mht_7_v, 312, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateReluOpBuilder");

  return new ActivationLayerBuilder(graph_builder, kTfLiteActRelu);
}

OpBuilder* CreateReluN1To1OpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_8(mht_8_v, 319, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateReluN1To1OpBuilder");

  return new ActivationLayerBuilder(graph_builder, kTfLiteActReluN1To1);
}

OpBuilder* CreateRelu6OpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_9(mht_9_v, 326, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateRelu6OpBuilder");

  return new ActivationLayerBuilder(graph_builder, kTfLiteActRelu6);
}

OpBuilder* CreateTanhOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSactivation_layer_builderDTcc mht_10(mht_10_v, 333, "", "./tensorflow/lite/delegates/coreml/builders/activation_layer_builder.cc", "CreateTanhOpBuilder");

  return new ActivationLayerBuilder(graph_builder, kTfLiteActTanh);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
