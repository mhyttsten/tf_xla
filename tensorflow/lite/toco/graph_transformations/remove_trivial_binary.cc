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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_binaryDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_binaryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_binaryDTcc() {
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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <typename Scalar>
bool AreAllBufferElementsEqualTo(const std::vector<Scalar>& buffer_data,
                                 Scalar value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_binaryDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/remove_trivial_binary.cc", "AreAllBufferElementsEqualTo");

  for (const auto& x : buffer_data) {
    if (x != value) {
      return false;
    }
  }
  return true;
}
}  // namespace

// A binary operator is called trivial when exactly one of its operands is
// a constant and is such that the binary operation is equivalent to
// the identity operation on its other input.
// For example, an Add operator is trivial if
// one of its operands is constant 0, a Mul operator is trivial
// if one of its operands is constant 1, etc.
::tensorflow::Status RemoveTrivialBinaryOperator::Run(Model* model,
                                                      std::size_t op_index,
                                                      bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_binaryDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/toco/graph_transformations/remove_trivial_binary.cc", "RemoveTrivialBinaryOperator::Run");

  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(binary_op->inputs.size(), 2);

  // This graph transformation is only concerned with the case
  // when one input is constant and the other is not constant.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can resolve here.
    return ::tensorflow::Status::OK();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants
    // propagation, not for us to handle here.
    return ::tensorflow::Status::OK();
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  // If this was a broadcasting op we can't remove it as we need the broadcast.
  // It's possible we could replace it with a cheaper op, though.
  const auto& input_array_0 = model->GetArray(binary_op->inputs[0]);
  const auto& input_array_1 = model->GetArray(binary_op->inputs[1]);
  if (!input_array_0.has_shape() || !input_array_1.has_shape()) {
    // Both input shapes must be known.
    return ::tensorflow::Status::OK();
  }
  if (input_array_0.shape().dimensions_count() ==
          input_array_1.shape().dimensions_count() &&
      input_array_0.shape() != input_array_1.shape()) {
    AddMessageF(
        "Preserving %s even though it's trivial as we need to broadcast "
        "(lhs %s, rhs %s)",
        LogName(*binary_op), ShapeToString(input_array_0.shape()),
        ShapeToString(input_array_1.shape()));
    return ::tensorflow::Status::OK();
  }

  // Now check if the constant operand makes this binary
  // operator trivial.
  const auto& constant_input_array =
      model->GetArray(binary_op->inputs[index_of_constant_input]);
  // For now, we only handle floats here.
  if (constant_input_array.data_type != ArrayDataType::kFloat) {
    return ::tensorflow::Status::OK();
  }
  const auto& constant_input_float_data =
      constant_input_array.GetBuffer<ArrayDataType::kFloat>().data;
  bool is_trivial = false;
  if (binary_op->type == OperatorType::kAdd) {
    is_trivial = AreAllBufferElementsEqualTo(constant_input_float_data, 0.f);
  } else if (binary_op->type == OperatorType::kSub) {
    is_trivial = index_of_constant_input == 1 &&
                 AreAllBufferElementsEqualTo(constant_input_float_data, 0.f);
  } else if (binary_op->type == OperatorType::kMul) {
    is_trivial = AreAllBufferElementsEqualTo(constant_input_float_data, 1.f);
  } else if (binary_op->type == OperatorType::kDiv) {
    is_trivial = index_of_constant_input == 1 &&
                 AreAllBufferElementsEqualTo(constant_input_float_data, 1.f);
  }

  is_trivial = is_trivial && binary_op->fused_activation_function ==
                                 FusedActivationFunctionType::kNone;

  if (!is_trivial) {
    return ::tensorflow::Status::OK();
  }

  // Now we know that this node is trivial, so we can remove it.
  AddMessageF("Removing trivial %s", LogName(*binary_op));
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return ::tensorflow::Status::OK();
}

}  // namespace toco
