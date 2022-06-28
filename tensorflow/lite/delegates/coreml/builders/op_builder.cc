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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

std::string TensorID::ToString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "TensorID::ToString");

  return std::to_string(node_) + "_" + std::to_string(output_id_);
}

int TensorID::NodeID() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "TensorID::NodeID");
 return node_; }

int TensorID::OutputID() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_2(mht_2_v, 209, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "TensorID::OutputID");
 return output_id_; }

OpBuilder* GraphBuilder::AddBuilder(int builtin_code, const TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_3(mht_3_v, 214, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::AddBuilder");

  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
      return AddBuilder(CreateAddOpBuilder, node);
    case kTfLiteBuiltinAveragePool2d:
      return AddBuilder(CreateAveragePool2dOpBuilder, node);
    case kTfLiteBuiltinConcatenation:
      return AddBuilder(CreateConcatenationOpBuilder, node);
    case kTfLiteBuiltinConv2d:
      return AddBuilder(CreateConvolutionOpBuilder, node);
    case kTfLiteBuiltinDepthwiseConv2d:
      return AddBuilder(CreateDepthwiseConvolutionOpBuilder, node);
    // TODO(b/141490853): Add proper dequantize OpBuilder for int8/uint8 inputs.
    case kTfLiteBuiltinDequantize:
      // FP16 dequantize is claimed by the delegate to prevent them from running
      // on CPU, but don't need to be excuted on the Core ML delegate either.
      return AddBuilder(CreateDummyOpBuilder, node);
    case kTfLiteBuiltinFullyConnected:
      return AddBuilder(CreateFullyConnectedOpBuilder, node);
    case kTfLiteBuiltinLogistic:
      return AddBuilder(CreateLogisticOpBuilder, node);
    case kTfLiteBuiltinMaxPool2d:
      return AddBuilder(CreateMaxPool2dOpBuilder, node);
    case kTfLiteBuiltinMean:
      return AddBuilder(CreateMeanOpBuilder, node);
    case kTfLiteBuiltinMirrorPad:
      return AddBuilder(CreateMirrorPadOpBuilder, node);
    case kTfLiteBuiltinMul:
      return AddBuilder(CreateMulOpBuilder, node);
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
      return AddBuilder(CreatePadOpBuilder, node);
    case kTfLiteBuiltinRelu:
      return AddBuilder(CreateReluOpBuilder, node);
    case kTfLiteBuiltinReluN1To1:
      return AddBuilder(CreateReluN1To1OpBuilder, node);
    case kTfLiteBuiltinRelu6:
      return AddBuilder(CreateRelu6OpBuilder, node);
    case kTfLiteBuiltinReshape:
      return AddBuilder(CreateReshapeOpBuilder, node);
    case kTfLiteBuiltinResizeBilinear:
      return AddBuilder(CreateResizeBilinearOpBuilder, node);
    case kTfLiteBuiltinSoftmax:
      return AddBuilder(CreateSoftmaxOpBuilder, node);
    case kTfLiteBuiltinTanh:
      return AddBuilder(CreateTanhOpBuilder, node);
    case kTfLiteBuiltinTransposeConv:
      return AddBuilder(CreateTransposeConvolutionOpBuilder, node);
    case kTfLiteBuiltinHardSwish:
      return AddBuilder(CreateHardSwishOpBuilder, node);
    default:
      return nullptr;
  }
}

OpBuilder* GraphBuilder::AddBuilder(
    const std::function<OpBuilder*(GraphBuilder*)>& builder,
    const TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_4(mht_4_v, 274, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::AddBuilder");

  if (builder == nullptr) {
    fprintf(stderr, "builder should be set.\n");
    return nullptr;
  }
  OpBuilder* op = builder(this);

  builders_.emplace_back(op);
  op->SetNodeID(builders_.size());
  if (node != nullptr) {
    op->SetBuiltinData(node->builtin_data);
    op->SetTfLiteNode(node);
  }
  return builders_.back().get();
}

CoreML::Specification::Model* GraphBuilder::BuildModel() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_5(mht_5_v, 293, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::BuildModel");

  CoreML::Specification::Model* model = new CoreML::Specification::Model();
  if (coreml_version_ == 2) {  // Core ML 2, iOS >= 12.0
    model->set_specificationversion(3);
  } else if (coreml_version_ == 3) {  // Core ML 3, iOS >= 13.0
    model->set_specificationversion(4);
    model->mutable_neuralnetwork()->set_arrayinputshapemapping(
        CoreML::Specification::EXACT_ARRAY_MAPPING);
  } else {
    fprintf(stderr, "Unsupported Core ML version: %d\n", coreml_version_);
    delete model;
    return nullptr;
  }
  auto* neural_network = model->mutable_neuralnetwork();
  for (auto& builder : builders_) {
    CoreML::Specification::NeuralNetworkLayer* layer = builder->Build();
    if (layer == nullptr) {
      fprintf(stderr, "Null layer returned from builder: %s\n",
              builder->DebugName().c_str());
      continue;
    }
    neural_network->mutable_layers()->AddAllocated(layer);
  }
  return model;
}

void GraphBuilder::AddTensorWithID(int tf_tensor_id,
                                   const TensorID& tensor_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_6(mht_6_v, 323, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::AddTensorWithID");

  if (tensors_.size() <= tf_tensor_id) {
    tensors_.resize(tf_tensor_id + 1);
    used_tensor_.resize(tf_tensor_id + 1);
  }
  tensors_[tf_tensor_id] = tensor_id;
}

std::string GraphBuilder::GetTensorName(int tensor_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_7(mht_7_v, 334, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::GetTensorName");

  return GetTensorID(tensor_id).ToString();
}

const TensorID GraphBuilder::GetTensorID(int tensor_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_8(mht_8_v, 341, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::GetTensorID");

  if (!HasTensor(tensor_id)) {
    // TODO(karimnosseir): Double check if this happened, if we are
    // adding in execution order it shouldn't happen.
    fprintf(stderr, "index out of range...!!! Requested index %d , size %d\n",
            tensor_id, static_cast<int>(tensors_.size()));
    // Return invalid ID.
    return TensorID(-1, -1);
  }
  used_tensor_[tensor_id] = true;
  return tensors_[tensor_id];
}

bool GraphBuilder::HasTensor(int tflite_tensor_index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_9(mht_9_v, 357, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::HasTensor");

  if (tensors_.size() <= tflite_tensor_index) {
    return false;
  }
  return tensors_[tflite_tensor_index].NodeID() != -1;
}

bool GraphBuilder::IsTensorUsed(int tflite_tensor_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_10(mht_10_v, 367, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "GraphBuilder::IsTensorUsed");

  if (!HasTensor(tflite_tensor_index)) return false;
  return used_tensor_[tflite_tensor_index];
}

CoreML::Specification::NeuralNetworkLayer* OpBuilder::Build() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_11(mht_11_v, 375, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::Build");

  layer_->set_name(DebugName());
  return layer_.release();
}

TfLiteStatus OpBuilder::PopulateSubgraph(TfLiteContext* context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_12(mht_12_v, 383, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::PopulateSubgraph");

  builder_output_ = AddOutput();
  return kTfLiteOk;
}

void OpBuilder::SetBuiltinData(void* builtin_data) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_13(mht_13_v, 391, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::SetBuiltinData");

  builtin_data_ = builtin_data;
}

void OpBuilder::SetNodeID(int id) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_14(mht_14_v, 398, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::SetNodeID");
 node_id_ = id; }

void OpBuilder::SetTfLiteNode(const TfLiteNode* node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_15(mht_15_v, 403, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::SetTfLiteNode");
 tflite_node_ = node; }

int OpBuilder::GetID() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_16(mht_16_v, 408, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::GetID");
 return node_id_; }

TensorID OpBuilder::GetOutput(TfLiteContext* context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_17(mht_17_v, 413, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::GetOutput");

  if (builder_output_.NodeID() != -1) {
    return builder_output_;
  }
  // builder_output_ is not set when PopulateSubgraph is not called.
  builder_output_ = AddOutput();
  return builder_output_;
}

void OpBuilder::AddInput(const std::string& input_name) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_18(mht_18_v, 426, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::AddInput");

  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  *layer_->mutable_input()->Add() = input_name;
}

void OpBuilder::AddInput(const TensorID& input_id) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_19(mht_19_v, 436, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::AddInput");

  AddInput(input_id.ToString());
}

void OpBuilder::AddInput(int tf_input_id) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_20(mht_20_v, 443, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::AddInput");

  AddInput(graph_builder_->GetTensorName(tf_input_id));
}

TensorID OpBuilder::AddOutput() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_21(mht_21_v, 450, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::AddOutput");

  auto tensor_id = TensorID(GetID(), num_outputs_++);
  *layer_->mutable_output()->Add() = tensor_id.ToString();
  return tensor_id;
}

void OpBuilder::SetDebugName(const char* name, int id) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSop_builderDTcc mht_22(mht_22_v, 460, "", "./tensorflow/lite/delegates/coreml/builders/op_builder.cc", "OpBuilder::SetDebugName");

  debug_name_ = std::string(name) + "_" + std::to_string(id);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
