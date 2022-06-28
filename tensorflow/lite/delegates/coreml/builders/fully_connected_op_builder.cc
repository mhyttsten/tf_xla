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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/activation_layer_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
const std::string& FullyConnectedOpBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::DebugName");

  if (debug_name_.empty()) SetDebugName("FullyConnectedOpBuilder", node_id_);
  return debug_name_;
}

void FullyConnectedOpBuilder::SetWeights(TfLiteTensor* weights) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::SetWeights");

  weights_ = weights;
}

void FullyConnectedOpBuilder::SetBias(TfLiteTensor* bias) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_2(mht_2_v, 211, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::SetBias");
 bias_ = bias; }

CoreML::Specification::NeuralNetworkLayer* FullyConnectedOpBuilder::Build() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_3(mht_3_v, 216, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::Build");

  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());

  FillCoreMLWeights();
  FillCoreMLBias();

  return layer_.release();
}

void FullyConnectedOpBuilder::FillCoreMLWeights() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_4(mht_4_v, 231, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::FillCoreMLWeights");

  layer_->mutable_innerproduct()->set_inputchannels(weights_->dims->data[1]);
  layer_->mutable_innerproduct()->set_outputchannels(weights_->dims->data[0]);
  if (weights_->type == kTfLiteFloat32) {
    const float* weights_data = GetTensorData<float>(weights_);
    std::copy(weights_data, weights_data + NumElements(weights_),
              google::protobuf::RepeatedFieldBackInserter(layer_->mutable_innerproduct()
                                                    ->mutable_weights()
                                                    ->mutable_floatvalue()));
  } else if (weights_->type == kTfLiteFloat16) {
    // float16value has type of bytes (std::string)
    layer_->mutable_innerproduct()
        ->mutable_weights()
        ->mutable_float16value()
        ->assign(weights_->data.raw, weights_->bytes);
  }
}

void FullyConnectedOpBuilder::FillCoreMLBias() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_5(mht_5_v, 252, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::FillCoreMLBias");

  if (bias_ != nullptr) {
    layer_->mutable_innerproduct()->set_hasbias(true);
    if (bias_->type == kTfLiteFloat32) {
      const float* bias_data = GetTensorData<float>(bias_);
      std::copy(bias_data, bias_data + NumElements(bias_),
                google::protobuf::RepeatedFieldBackInserter(layer_->mutable_innerproduct()
                                                      ->mutable_bias()
                                                      ->mutable_floatvalue()));
    } else if (bias_->type == kTfLiteFloat16) {
      // float16value has type of bytes (std::string)
      layer_->mutable_innerproduct()
          ->mutable_bias()
          ->mutable_float16value()
          ->assign(bias_->data.raw, bias_->bytes);
    }
  }
}

TfLiteStatus FullyConnectedOpBuilder::PopulateSubgraph(TfLiteContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_6(mht_6_v, 274, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::PopulateSubgraph");

  const auto* fc_params =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(builtin_data_);
  TfLiteFusedActivation activation = fc_params->activation;

  if (activation == kTfLiteActNone) {
    builder_output_ = AddOutput();
  } else {
    ActivationLayerBuilder* activation_builder =
        reinterpret_cast<ActivationLayerBuilder*>(
            graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
    activation_builder->SetActivation(activation);
    activation_builder->AddInput(AddOutput());
    activation_builder->PopulateSubgraph(context);
    builder_output_ = activation_builder->GetOutput(context);
  }
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedOpBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_7(mht_7_v, 297, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::RegisterInputs");

  const int kInput = 0;
  const int kWeights = 1;
  const int kBias = 2;
  AddInput(inputs->data[kInput]);
  SetWeights(&context->tensors[inputs->data[kWeights]]);
  if (inputs->size > 2) {
    SetBias(&context->tensors[inputs->data[kBias]]);
  }
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_8(mht_8_v, 313, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "FullyConnectedOpBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs!.");
    return kTfLiteError;
  }
  TensorID output_tensor = GetOutput(context);
  if (output_tensor.NodeID() == -1) {
    TF_LITE_KERNEL_LOG(context, "Failed to build output tensor.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], output_tensor);
  return kTfLiteOk;
}

OpBuilder* CreateFullyConnectedOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_9(mht_9_v, 330, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "CreateFullyConnectedOpBuilder");

  return new FullyConnectedOpBuilder(graph_builder);
}

bool IsFloatType(TfLiteType type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_10(mht_10_v, 337, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "IsFloatType");

  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

bool IsFullyConnectedOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSfully_connected_op_builderDTcc mht_11(mht_11_v, 346, "", "./tensorflow/lite/delegates/coreml/builders/fully_connected_op_builder.cc", "IsFullyConnectedOpSupported");

  if (node->builtin_data == nullptr) return false;
  const auto* fc_params =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  const int kInput = 0;
  const int kWeights = 1;
  const int kBias = 2;

  if (fc_params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
    return false;
  }
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInput, &input));
  const TfLiteTensor* weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kWeights, &weights));

  if (!IsFloatType(input->type)) {
    return false;
  }
  if (!IsFloatType(weights->type) || !IsConstantTensor(weights)) {
    return false;
  }
  // Core ML 2 only supports single-batch fully connected layer, thus dimensions
  // except the last one should be 1.
  if (input->dims->data[input->dims->size - 1] != NumElements(input)) {
    return false;
  }

  if (node->inputs->size > 2) {
    const TfLiteTensor* bias;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBias, &bias));
    if (!IsFloatType(bias->type) || !IsConstantTensor(bias)) {
      return false;
    }
  }

  TfLiteFusedActivation activation = fc_params->activation;
  if (activation == kTfLiteActSignBit) {
    return false;
  }
  return true;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
