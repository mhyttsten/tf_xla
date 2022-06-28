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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc() {
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
#include "tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/add_op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/mul_op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace coreml {
const std::string& HardSwishOpBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "HardSwishOpBuilder::DebugName");

  if (debug_name_.empty()) SetDebugName("HardSwishOpBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* HardSwishOpBuilder::Build() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "HardSwishOpBuilder::Build");

  layer_->set_name(DebugName());
  layer_->mutable_multiply()->set_alpha(1.0f / 6.0f);

  return layer_.release();
}

TfLiteStatus HardSwishOpBuilder::PopulateSubgraph(TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_2(mht_2_v, 213, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "HardSwishOpBuilder::PopulateSubgraph");

  // hswish(x) = (x/6) * ReLU6(x+3). main layer_ contains the first part, x/6.
  // ReLU6(x +3) constructed as add op with fused ReLU6 activation.
  AddOpBuilder* add_builder = reinterpret_cast<AddOpBuilder*>(
      graph_builder_->AddBuilder(CreateAddOpBuilder, nullptr));
  TfLiteAddParams add_param{kTfLiteActRelu6};
  add_builder->SetBuiltinData(&add_param);
  add_builder->SetAlpha(3.0f);
  add_builder->AddInput(layer_->input(0));
  add_builder->PopulateSubgraph(context);

  // multiplies (x/6) from main layer_ and ReLU6(x+3) from the above code.
  MulOpBuilder* mul_builder = reinterpret_cast<MulOpBuilder*>(
      graph_builder_->AddBuilder(CreateMulOpBuilder, nullptr));
  mul_builder->AddInput(AddOutput());
  mul_builder->AddInput(add_builder->GetOutput(context));
  builder_output_ = mul_builder->AddOutput();
  return kTfLiteOk;
}

TfLiteStatus HardSwishOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                TfLiteContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "HardSwishOpBuilder::RegisterInputs");

  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to hardswish!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus HardSwishOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_4(mht_4_v, 250, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "HardSwishOpBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to hardswish!.");
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

OpBuilder* CreateHardSwishOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPShardswish_op_builderDTcc mht_5(mht_5_v, 267, "", "./tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.cc", "CreateHardSwishOpBuilder");

  return new HardSwishOpBuilder(graph_builder);
}
}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
