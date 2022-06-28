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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_hardswishDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_hardswishDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_hardswishDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/identify_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"

// This transformation rule tries to identify the HardSwish structure generated
// by tensorflow.
// The formula of hardswish is:
// f(x) = x * relu6((x+3))/6
//
// We look for the following tensorflow subgraph:
// x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)
namespace toco {

using util::IsBinaryOp;

::tensorflow::Status IdentifyHardSwish::Run(Model* model, std::size_t op_index,
                                            bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_hardswishDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/toco/graph_transformations/identify_hardswish.cc", "IdentifyHardSwish::Run");

  *modified = false;
  const auto add_with_relu6_op_it = (model->operators.begin() + op_index);
  const auto add_with_relu6_op = add_with_relu6_op_it->get();
  if (!util::IsBinaryOp(add_with_relu6_op, OperatorType::kAdd,
                        FusedActivationFunctionType::kRelu6)) {
    return ::tensorflow::Status::OK();
  }
  std::vector<const Operator*> ops;
  ops.push_back(add_with_relu6_op);
  const auto* mul_op = GetOpWithInput(*model, add_with_relu6_op->outputs[0]);
  ops.push_back(mul_op);

  if (mul_op->type == OperatorType::kFakeQuant) {
    mul_op = GetOpWithInput(*model, mul_op->outputs[0]);
    ops.push_back(mul_op);
  }
  if (!IsBinaryOp(mul_op, OperatorType::kMul)) {
    return ::tensorflow::Status::OK();
  }

  const auto* output_op = GetOpWithInput(*model, mul_op->outputs[0]);
  ops.push_back(output_op);
  if (output_op->type == OperatorType::kFakeQuant) {
    output_op = GetOpWithInput(*model, output_op->outputs[0]);
    ops.push_back(output_op);
  }
  if (!IsBinaryOp(output_op, OperatorType::kMul)) {
    return ::tensorflow::Status::OK();
  }
  const auto add_3_tensor =
      util::GetSingleScalarInputIndexOfBinaryOp(model, add_with_relu6_op, 3.0f);
  if (add_3_tensor < 0) {
    // Expected 3.0f got something else.;
    return ::tensorflow::Status::OK();
  }
  const auto input_tensor_name = add_with_relu6_op->inputs[1 - add_3_tensor];

  // Now we verify that the 3 mul arguments are respectively:
  // 1. non-constant input of add_with_relu6_op
  // 2. 1/6
  // 3. (and add_with_relu6_op[0].outputs[0] - which we already know!)
  std::vector<std::string> mul_inputs = mul_op->inputs;
  mul_inputs.insert(mul_inputs.end(), output_op->inputs.begin(),
                    output_op->inputs.end());

  // 1. Check that we have the input tensor as one of the multiplicants
  if (std::find(mul_inputs.begin(), mul_inputs.end(), input_tensor_name) ==
      mul_inputs.end()) {
    // Input tensor not found! << input_tensor_name << std::endl;
    return ::tensorflow::Status::OK();
  }
  // 2. Find 1/6
  bool found = false;
  for (const auto& input : mul_inputs) {
    found |= util::CheckArrayIsScalarFloat(model, input, 1.f / 6.f);
  }
  if (!found) {
    // Input tensor is not divided by 6!.";
    return ::tensorflow::Status::OK();
  }
  //  Success! Now delete the subgraph and instert new one
  const auto output_tensor_name = output_op->outputs[0];
  auto* hardswish_op = new HardSwishOperator;
  hardswish_op->inputs = {input_tensor_name};
  hardswish_op->outputs = {output_tensor_name};
  model->operators.emplace(add_with_relu6_op_it, hardswish_op);
  AddMessageF("Creating hardswish op (%s) replacing equivalent subgraph",
              LogName(*hardswish_op));
  while (!ops.empty()) {
    DeleteOpAndArrays(model, ops.back());
    ops.pop_back();
  }
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
