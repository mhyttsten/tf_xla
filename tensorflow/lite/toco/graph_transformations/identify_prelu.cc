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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_preluDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_preluDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_preluDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

// This transformation rule tries to identify the PRelu structure generated by
// Keras, and convert it to a single op.
//
// The formula of PReLU is:
// f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
//
// `x` is the input, and `alpha` is a trainable tensor which can be broadcasted
// to the shape of `x`.
//
// There's no native PRelu op in TensorFlow, so Keras generates the following
// structure which does the equivalent calculation:
// f(x) = Relu(x) + (-alpha * Relu(-x))
//
// Practically, alpha is always a constant in the inference graph, and Toco have
// other graph transformations which fold the activation functions to other ops.
// Therefore, we're looking for the structure:
//
// f(x) = Relu(x) + (negative_alpha * Neg(x, activation=Relu))

namespace toco {

::tensorflow::Status IdentifyPRelu::Run(Model* model, std::size_t op_index,
                                        bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_preluDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/toco/graph_transformations/identify_prelu.cc", "IdentifyPRelu::Run");

  *modified = false;
  const auto add_op_it = model->operators.begin() + op_index;
  const auto* add_op = add_op_it->get();
  if (add_op == nullptr || add_op->type != OperatorType::kAdd ||
      add_op->inputs.size() != 2 ||
      add_op->fused_activation_function != FusedActivationFunctionType::kNone) {
    return ::tensorflow::Status::OK();
  }

  const auto* relu_input_op = GetOpWithOutput(*model, add_op->inputs[0]);
  if (relu_input_op == nullptr || relu_input_op->type != OperatorType::kRelu ||
      relu_input_op->inputs.size() != 1 ||
      relu_input_op->fused_activation_function !=
          FusedActivationFunctionType::kNone) {
    return ::tensorflow::Status::OK();
  }

  // TODO(ycling): Both Add and Mul are commutative. Support the case where
  // the position of operands are exchanged.
  const auto* mul_op = GetOpWithOutput(*model, add_op->inputs[1]);
  if (mul_op == nullptr || mul_op->type != OperatorType::kMul ||
      mul_op->inputs.size() != 2 ||
      mul_op->fused_activation_function != FusedActivationFunctionType::kNone) {
    return ::tensorflow::Status::OK();
  }

  const auto neg_alpha_tensor_name = mul_op->inputs[0];

  const auto* relu_neg_input_op = GetOpWithOutput(*model, mul_op->inputs[1]);

  if (relu_neg_input_op == nullptr ||
      relu_neg_input_op->inputs.size() != 1) {
    return ::tensorflow::Status::OK();
  }

  const Operator* final_input_op;
  if (relu_neg_input_op->type == OperatorType::kNeg &&
      relu_neg_input_op->fused_activation_function ==
          FusedActivationFunctionType::kRelu) {
    // This detects a Neg op with fused Relu activation function.
    final_input_op = relu_neg_input_op;
  } else {
    // This detects a Neg op followed by a separated Relu op.
    const auto* neg_input_op =
        GetOpWithOutput(*model, relu_neg_input_op->inputs[0]);
    if (neg_input_op == nullptr || neg_input_op->inputs.size() != 1 ||
        relu_neg_input_op->type != OperatorType::kRelu ||
        relu_neg_input_op->fused_activation_function !=
            FusedActivationFunctionType::kNone) {
      return ::tensorflow::Status::OK();
    }
    final_input_op = neg_input_op;
  }

  if (relu_input_op->inputs[0] != final_input_op->inputs[0]) {
    return ::tensorflow::Status::OK();
  }

  const auto input_tensor_name = relu_input_op->inputs[0];
  const auto output_tensor_name = add_op->outputs[0];

  // Construct a tensor for positive alpha (double negative).
  const auto alpha_tensor_name =
      AvailableArrayName(*model, neg_alpha_tensor_name + "_neg");
  model->GetOrCreateArray(alpha_tensor_name);

  auto* neg_neg_alpha_op = new NegOperator;
  neg_neg_alpha_op->inputs = {neg_alpha_tensor_name};
  neg_neg_alpha_op->outputs = {alpha_tensor_name};
  model->operators.emplace(add_op_it, neg_neg_alpha_op);

  auto* prelu_op = new PReluOperator;
  prelu_op->inputs = {input_tensor_name, alpha_tensor_name};
  prelu_op->outputs = {output_tensor_name};
  model->operators.emplace(add_op_it, prelu_op);
  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*prelu_op));

  DeleteArrayIfUnusedOutsideOfOp(neg_alpha_tensor_name, neg_neg_alpha_op,
                                 model);
  DeleteArrayIfUnusedOutsideOfOp(mul_op->inputs[1], mul_op, model);
  DeleteOpAndArrays(model, add_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
