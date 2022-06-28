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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

TransposeOperator* FindTransposeOpWithInput(const Model& model,
                                            const std::string& array_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/toco/graph_transformations/resolve_tensorflow_matmul.cc", "FindTransposeOpWithInput");

  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    Operator* op = it->get();
    if (op->type != OperatorType::kTranspose) {
      continue;
    }
    if (op->inputs[0] != array_name) {
      continue;
    }
    const auto& permutation_array = model.GetArray(op->inputs[1]);
    if (permutation_array.data_type != ArrayDataType::kInt32) {
      continue;
    }
    const auto& permutation_data =
        permutation_array.GetBuffer<ArrayDataType::kInt32>().data;
    if (permutation_data.size() != 2) {
      continue;
    }
    if (permutation_data[0] != 1 || permutation_data[1] != 0) {
      continue;
    }
    return static_cast<TransposeOperator*>(op);
  }
  return nullptr;
}

}  // namespace

::tensorflow::Status ResolveTensorFlowMatMul::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/toco/graph_transformations/resolve_tensorflow_matmul.cc", "ResolveTensorFlowMatMul::Run");

  *modified = false;
  auto matmul_it = model->operators.begin() + op_index;
  if (matmul_it->get()->type != OperatorType::kMatMul) {
    return ::tensorflow::Status::OK();
  }
  const auto* matmul_op =
      static_cast<const TensorFlowMatMulOperator*>(matmul_it->get());

  auto refresh_matmul_iterator = [&model, &matmul_it, &matmul_op]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_matmulDTcc mht_2(mht_2_v, 245, "", "./tensorflow/lite/toco/graph_transformations/resolve_tensorflow_matmul.cc", "lambda");

    matmul_it = std::find_if(model->operators.begin(), model->operators.end(),
                             [matmul_op](const std::unique_ptr<Operator>& op) {
                               return op.get() == matmul_op;
                             });
    DCHECK_EQ(matmul_it->get(), matmul_op);
  };

  std::string input_lhs = matmul_op->inputs[0];
  std::string input_rhs = matmul_op->inputs[1];

  // Handle `transpose_a` with best effort: If the dimension of lhs is known,
  // insert a `Transpose` op.
  if (matmul_op->transpose_a) {
    Array& lhs_array = model->GetArray(input_lhs);
    if (!lhs_array.has_shape()) {
      AddMessageF(
          "Not replacing %s by a FullyConnected operator, because it has "
          "the transpose_a attribute and LHS has no shape",
          LogName(*matmul_op));
      return ::tensorflow::Status::OK();
    }

    int dimensions_count = lhs_array.shape().dimensions_count();
    if (dimensions_count < 2) {
      return ::tensorflow::errors::InvalidArgument(
          "Inputs of MatMul should have dimension >= 2. Got %d dimensions",
          dimensions_count);
    }

    // Create a permutation vector to exchange the last 2 dimensions.
    // E.g. For 4D, create [0, 1, 3, 2].
    std::vector<int> perm;
    perm.reserve(dimensions_count);
    for (int i = 0; i < dimensions_count; ++i) {
      perm.push_back(i);
    }
    std::swap(perm[dimensions_count - 1], perm[dimensions_count - 2]);

    auto* transpose_op = new TransposeOperator;
    transpose_op->inputs = {
        input_lhs,
        CreateInt32Array(
            model, AvailableArrayName(*model, input_lhs + "/transpose/perm"),
            perm)};
    transpose_op->outputs = {
        AvailableArrayName(*model, input_lhs + "/transpose")};
    model->GetOrCreateArray(transpose_op->outputs[0]);
    model->operators.emplace(matmul_it, transpose_op);
    // Sanity check
    DCHECK_EQ(transpose_op, FindTransposeOpWithInput(*model, input_lhs));
    input_lhs = transpose_op->outputs[0];

    refresh_matmul_iterator();
  }

  // TODO(b/138662017): The following code assumes that RHS is 2D. This isn't
  // always true in TensorFlow.
  //
  // Reorder the axes on the second input. TensorFlow uses row-major ordering
  // on both inputs, however this is inefficient for the FullyConnected
  // operator. We'll transpose the second input to be in column-major order now
  // and let constant propagation optimize things (if possible).
  if (!matmul_op->transpose_b) {
    // Need to transpose input_rhs, by inserting a TransposeOperator.
    // First, check if there already is a TransposeOperator transposing that
    // array, so we can just reuse it.
    auto* transpose_op = FindTransposeOpWithInput(*model, input_rhs);
    if (!transpose_op) {
      AddMessageF(
          "While replacing %s by a FullyConnected operator, created new "
          "Transpose op wrapping RHS input array %s",
          LogName(*matmul_op), input_rhs);
      // No such TransposeOperator found. Create one now.
      transpose_op = new TransposeOperator;
      transpose_op->inputs = {
          input_rhs,
          CreateInt32Array(
              model, AvailableArrayName(*model, input_rhs + "/transpose/perm"),
              {1, 0})};
      transpose_op->outputs = {
          AvailableArrayName(*model, input_rhs + "/transpose")};
      model->GetOrCreateArray(transpose_op->outputs[0]);
      model->operators.emplace(matmul_it, transpose_op);
      // Sanity check
      DCHECK_EQ(transpose_op, FindTransposeOpWithInput(*model, input_rhs));
      refresh_matmul_iterator();
    } else {
      AddMessageF(
          "While replacing %s by a FullyConnected operator, reused existing "
          "Transpose op wrapping RHS input array %s",
          LogName(*matmul_op), input_rhs);
    }
    // Re-wire: have the matmul consume the transposed array.
    input_rhs = transpose_op->outputs[0];
  }

  // Construct the new FullyConnectedOperator.
  auto* fc_op = new FullyConnectedOperator;
  fc_op->inputs = {input_lhs, input_rhs};
  fc_op->outputs = matmul_op->outputs;

  // Insert the newly constructed FullyConnectedOperator.
  model->operators.emplace(matmul_it, fc_op) + 1;

  // Find the op producing the array passed to this MatMul
  auto previous_op_it = model->operators.begin();
  bool found = false;
  for (; previous_op_it != model->operators.end(); ++previous_op_it) {
    for (const auto& output : (*previous_op_it)->outputs) {
      if (output == matmul_op->inputs[0]) {
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  Operator* previous_op = (found) ? previous_op_it->get() : nullptr;

  // Refresh iterator.
  matmul_it = model->operators.begin();
  for (; matmul_it != model->operators.end(); ++matmul_it) {
    if (matmul_it->get() == matmul_op) {
      break;
    }
  }
  DCHECK_EQ(matmul_it->get(), matmul_op);

  // The way that TensorFlow encodes FullyConnected ops is as a pair
  // (Reshape, MatMul), so we want to remove the Reshape op and rewrite the
  // MatMul op as a FullyConnected. However, TensorFlow skips the Reshape ops if
  // the input doesn't need reshaping, so we can't just match (Reshape, MatMul)
  // pairs.
  if (previous_op && previous_op->type == OperatorType::kReshape) {
    AddMessageF("Combining %s and %s into %s", LogName(*previous_op),
                LogName(*matmul_op), LogName(*fc_op));
    const auto& previous_op_output = previous_op->outputs[0];
    if (CountOpsWithInput(*model, previous_op_output) == 1) {
      model->EraseArray(previous_op_output);
    }
    CHECK_EQ(previous_op->inputs.size(), 2);
    input_lhs = previous_op->inputs[0];
    fc_op->inputs = {input_lhs, input_rhs};
    // Only remove Reshape node if no other node uses its output.
    if (CountOpsWithInput(*model, previous_op_output) == 1) {
      DeleteOpAndArrays(model, previous_op);
    }

    // We may have just invalidated matmul_it, so let's refresh it now.
    refresh_matmul_iterator();
  } else {
    AddMessageF("Replacing %s by a FullyConnected operator",
                LogName(*matmul_op));
  }


  // erase the MatMul operator
  model->operators.erase(matmul_it);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
