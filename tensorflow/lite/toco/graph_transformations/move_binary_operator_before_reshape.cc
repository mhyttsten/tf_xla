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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmove_binary_operator_before_reshapeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmove_binary_operator_before_reshapeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmove_binary_operator_before_reshapeDTcc() {
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
#include <algorithm>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool IsTailOfShape(const Shape& tail, const Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmove_binary_operator_before_reshapeDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/toco/graph_transformations/move_binary_operator_before_reshape.cc", "IsTailOfShape");

  // Return true if 'tail' dimensions are the same as the ending dimensions of
  // 'shape'.

  int shape_end = shape.dimensions_count() - 1;
  int tail_end = tail.dimensions_count() - 1;

  if (tail_end > shape_end) {
    // tail cannot be longer than shape.
    return false;
  }

  // Walk dimensions back to front and compare
  for (int i = 0; i <= tail_end; i++) {
    if (shape.dims(shape_end - i) != tail.dims(tail_end - i)) {
      return false;
    }
  }
  return true;
}

}  // namespace

// If a binary operator is doing a broadcast operation from a constant array,
// and the constant array shape is the tail of both the other input shape, and a
// subsequent reshape op's output shape, we can swap their order. Since we
// prefer to have reshape ops after mathematic ops, this can allow for the
// collapsing of some reshapes. The WaveNet model in particular benefits from
// this transformation.
//
// Note we are testing for one particular case of a broader set of possible
// binary-reshape op transformations. This transformation could be generalized.
::tensorflow::Status MoveBinaryOperatorBeforeReshape::Run(Model* model,
                                                          std::size_t op_index,
                                                          bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmove_binary_operator_before_reshapeDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/toco/graph_transformations/move_binary_operator_before_reshape.cc", "MoveBinaryOperatorBeforeReshape::Run");

  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  Operator* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv &&
      binary_op->type != OperatorType::kFloorDiv &&
      binary_op->type != OperatorType::kFloorMod &&
      binary_op->type != OperatorType::kMinimum &&
      binary_op->type != OperatorType::kMaximum &&
      binary_op->type != OperatorType::kLess &&
      binary_op->type != OperatorType::kLessEqual &&
      binary_op->type != OperatorType::kGreater &&
      binary_op->type != OperatorType::kGreaterEqual) {
    return ::tensorflow::Status::OK();
  }

  // BINARY OP INPUT CHECKS
  CHECK_EQ(binary_op->inputs.size(), 2);
  const bool input_is_const[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!input_is_const[0] && !input_is_const[1]) {
    // To limit our scope, we require one constant input. Though there's no
    // reason this transformation wouldn't work with all variable inputs.
    return ::tensorflow::Status::OK();
  }
  if (input_is_const[0] && input_is_const[1]) {
    // Both inputs are constants. Leave this for constants propagation.
    return ::tensorflow::Status::OK();
  }
  const int constant_input_idx = input_is_const[0] ? 0 : 1;
  const int variable_input_idx = input_is_const[0] ? 1 : 0;
  CHECK(input_is_const[constant_input_idx]);
  CHECK(!input_is_const[variable_input_idx]);

  const auto& variable_input_array =
      model->GetArray(binary_op->inputs[variable_input_idx]);
  if (!variable_input_array.has_shape()) {
    AddMessageF(
        "Not moving %s because it's non-constant input shape is not resolved.",
        LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  if (!IsTailOfShape(
          model->GetArray(binary_op->inputs[constant_input_idx]).shape(),
          model->GetArray(binary_op->inputs[variable_input_idx]).shape())) {
    // Constant array shape must be the latter part of the variable shape.
    return ::tensorflow::Status::OK();
  }

  // RESHAPE OP CHECKS
  auto reshape_it =
      FindOpWithOutput(*model, binary_op->inputs[variable_input_idx]);
  if (reshape_it == model->operators.end()) {
    AddMessageF("Not moving %s because it's variable input is not connected.",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  Operator* reshape_op = reshape_it->get();
  if (reshape_op->type != OperatorType::kReshape) {
    AddMessageF("Not moving %s because the preceding %s is not a reshape op",
                LogName(*binary_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }
  const auto& reshape_input_array = model->GetArray(reshape_op->inputs[0]);
  if (!reshape_input_array.has_shape()) {
    AddMessageF(
        "Not moving %s because it's non-constant input shape is not resolved "
        "yet",
        LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  if (!IsTailOfShape(
          model->GetArray(binary_op->inputs[constant_input_idx]).shape(),
          model->GetArray(reshape_op->outputs[0]).shape())) {
    // Constant array shape must be the latter part of the binary op output
    // shape.
    return ::tensorflow::Status::OK();
  }

  // EXTRA CHECKS ON CONNECTING ARRAY
  for (const std::string& output_array : model->flags.output_arrays()) {
    if (binary_op->inputs[variable_input_idx] == output_array) {
      AddMessageF(
          "Not moving %s because the output of reshape op %s is an output op.",
          LogName(*binary_op), LogName(*reshape_op));
      return ::tensorflow::Status::OK();
    }
  }
  int count_ops_consuming_output =
      CountOpsWithInput(*model, binary_op->inputs[variable_input_idx]);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not moving %s because the output of reshape op %s is consumed by "
        "another op",
        LogName(*binary_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }

  // SWAP ORDER OF BINARY AND RESHAPE OPS
  AddMessageF("Moving op %s before reshape op %s", LogName(*binary_op),
              LogName(*reshape_op));

  // Swap op input and outputs
  std::iter_swap(reshape_op->inputs.begin(),
                 binary_op->inputs.begin() + variable_input_idx);
  std::iter_swap(reshape_op->outputs.begin(), binary_op->outputs.begin());

  // Swap operator ordering
  std::iter_swap(binary_it, reshape_it);

  // Clear binary output shape so it will be re-propagated
  model->GetArray(binary_op->outputs[0]).clear_shape();

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
