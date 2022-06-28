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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_l2_normalizationDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_l2_normalizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_l2_normalizationDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status IdentifyL2Normalization::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_l2_normalizationDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/identify_l2_normalization.cc", "IdentifyL2Normalization::Run");

  *modified = false;
  const auto div_it = model->operators.begin() + op_index;
  const auto* div_or_mul_op = div_it->get();
  OperatorType expected_op_type_producing_div_or_mul_input;
  if (div_or_mul_op->type == OperatorType::kDiv) {
    expected_op_type_producing_div_or_mul_input = OperatorType::kSqrt;
  } else if (div_or_mul_op->type == OperatorType::kMul) {
    expected_op_type_producing_div_or_mul_input = OperatorType::kRsqrt;
  } else {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(div_or_mul_op->inputs.size(), 2);
  Operator* op_producing_div_or_mul_input[2] = {
      GetOpWithOutput(*model, div_or_mul_op->inputs[0]),
      GetOpWithOutput(*model, div_or_mul_op->inputs[1]),
  };
  if (!op_producing_div_or_mul_input[1] ||
      op_producing_div_or_mul_input[1]->type !=
          expected_op_type_producing_div_or_mul_input) {
    return ::tensorflow::Status::OK();
  }
  Operator* sqrt_or_rsqrt_op = op_producing_div_or_mul_input[1];
  CHECK_EQ(sqrt_or_rsqrt_op->inputs.size(), 1);
  Operator* op_producing_sqrt_or_rsqrt_input =
      GetOpWithOutput(*model, sqrt_or_rsqrt_op->inputs[0]);
  if (!op_producing_sqrt_or_rsqrt_input) {
    return ::tensorflow::Status::OK();
  }

  // There may be an Add or a Maximum here, adding or clamping to a "small"
  // constant scalar.
  // Reported bug: b/29395854
  Operator* add_op = nullptr;
  Operator* op_producing_add_input = nullptr;
  if (op_producing_sqrt_or_rsqrt_input->type == OperatorType::kAdd ||
      op_producing_sqrt_or_rsqrt_input->type == OperatorType::kMaximum) {
    add_op = op_producing_sqrt_or_rsqrt_input;
    bool add_can_be_removed = false;
    CHECK_EQ(op_producing_sqrt_or_rsqrt_input->inputs.size(), 2);
    for (int i = 0; i < 2; i++) {
      const auto& input_array =
          model->GetArray(op_producing_sqrt_or_rsqrt_input->inputs[i]);
      if (!input_array.buffer) {
        continue;
      }
      if (input_array.buffer->type != ArrayDataType::kFloat) {
        continue;
      }
      if (RequiredBufferSizeForShape(input_array.shape()) != 1) {
        continue;
      }
      const auto& input_float_data =
          input_array.GetBuffer<ArrayDataType::kFloat>().data;
      if (std::abs(input_float_data[0]) > 1e-3f) {
        continue;
      }
      add_can_be_removed = true;
      op_producing_add_input = GetOpWithOutput(*model, add_op->inputs[1 - i]);
      break;
    }
    if (!add_can_be_removed) {
      AddMessageF(
          "Giving up trying to identify L2Normalization subgraph "
          " because the operator producing the input to the square root, %s,"
          ", does not match the expected pattern",
          LogName(*op_producing_sqrt_or_rsqrt_input));
      return ::tensorflow::Status::OK();
    }
  }

  Operator* sum_op =
      add_op ? op_producing_add_input : op_producing_sqrt_or_rsqrt_input;
  if (sum_op->type != OperatorType::kSum) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: "
        "expected Sum op, got %s",
        LogName(*sum_op));
    return ::tensorflow::Status::OK();
  }

  Operator* square_op = GetOpWithOutput(*model, sum_op->inputs[0]);
  if (square_op->type != OperatorType::kSquare) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: "
        "expected Square op, got %s",
        LogName(*square_op));
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(square_op->inputs.size(), 1);

  if (square_op->inputs[0] != div_or_mul_op->inputs[0]) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: %s does not "
        "take the same input as the Mul/Div node",
        LogName(*square_op));
    return ::tensorflow::Status::OK();
  }

  // Create and emplace the new L2Normalization
  auto* l2norm_op = new L2NormalizationOperator;
  l2norm_op->inputs = {div_or_mul_op->inputs[0]};
  l2norm_op->outputs = div_or_mul_op->outputs;
  model->operators.emplace(div_it, l2norm_op);

  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*l2norm_op));

  // Erase the subgraph that is now replaced by L2Normalization
  DeleteOpAndArrays(model, square_op);
  DeleteOpAndArrays(model, sum_op);
  if (add_op) {
    DeleteOpAndArrays(model, add_op);
  }
  DeleteOpAndArrays(model, sqrt_or_rsqrt_op);
  DeleteOpAndArrays(model, div_or_mul_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
