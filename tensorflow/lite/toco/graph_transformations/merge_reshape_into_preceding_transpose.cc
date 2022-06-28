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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmerge_reshape_into_preceding_transposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmerge_reshape_into_preceding_transposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmerge_reshape_into_preceding_transposeDTcc() {
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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool OperatorReady(const Model& model, const Operator* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmerge_reshape_into_preceding_transposeDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/toco/graph_transformations/merge_reshape_into_preceding_transpose.cc", "OperatorReady");

  if (!model.HasArray(op->inputs[0]) || !model.HasArray(op->inputs[1]) ||
      !model.HasArray(op->outputs[0])) {
    // Arrays are missing.
    return false;
  }

  if (!model.GetArray(op->inputs[0]).has_shape() ||
      !model.GetArray(op->outputs[0]).has_shape()) {
    // Input and output needs the shape.
    return false;
  }

  if (!model.GetArray(op->inputs[1]).buffer) {
    // Buffer needs to be a constant.
    return false;
  }

  return true;
}

// Returns whether the reshape could be a transpose.
std::vector<int32> ReshapeToTranspose(const Model& model,
                                      const TensorFlowReshapeOperator* op) {
  CHECK(!op->shape.empty());
  CHECK(model.HasArray(op->inputs[0]));
  CHECK(model.HasArray(op->outputs[0]));

  const auto& input_array = model.GetArray(op->inputs[0]);
  const auto& output_array = model.GetArray(op->outputs[0]);

  CHECK(input_array.has_shape());
  CHECK(output_array.has_shape());

  std::vector<int> in_shape = input_array.shape().dims();
  std::vector<int> out_shape = output_array.shape().dims();

  std::vector<int> one_indices;
  std::vector<int> not_one_indices;

  // Separate into one indices and not one indices.
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (in_shape[i] == 1) {
      one_indices.push_back(i);
    } else {
      not_one_indices.push_back(i);
    }
  }

  // Reorder the vertices.
  std::vector<int> perm;
  perm.reserve(in_shape.size());
  int one_index = 0;
  int not_one_index = 0;
  for (const auto val : out_shape) {
    if (val == 1) {
      perm.push_back(one_indices[one_index]);
      one_index++;
    } else {
      perm.push_back(not_one_indices[not_one_index]);
      not_one_index++;
    }
  }

  return perm;
}

}  // namespace

// When a transpose is fed into a reshape, it is possible for the two operators
// to be merged if the reshape does not affect memory ordering and does not
// affects the number of dimensions. This only occurs when only unary dimensions
// are shifting position.
::tensorflow::Status MergeReshapeIntoPrecedingTranspose::Run(
    Model* model, std::size_t op_index, bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmerge_reshape_into_preceding_transposeDTcc mht_1(mht_1_v, 278, "", "./tensorflow/lite/toco/graph_transformations/merge_reshape_into_preceding_transpose.cc", "MergeReshapeIntoPrecedingTranspose::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* reshape_op = ConvertOperator<TensorFlowReshapeOperator*>(
      it->get(), OperatorType::kReshape);

  if (reshape_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (!OperatorReady(*model, reshape_op) || reshape_op->shape.empty()) {
    return ::tensorflow::Status::OK();
  }

  const std::string intermediate_name = reshape_op->inputs[0];
  const std::string output_name = reshape_op->outputs[0];

  // Guarantee the input is only consume by the reshape.
  if (CountOpsWithInput(*model, intermediate_name) != 1) {
    return ::tensorflow::Status::OK();
  }

  // Check for the parent operator.
  const auto& transpose_it = FindOpWithOutput(*model, intermediate_name);
  if (transpose_it == model->operators.end()) {
    return ::tensorflow::Status::OK();
  }

  // Find the parent operator and guarantee it is a transpose.
  TransposeOperator* transpose_op = ConvertOperator<TransposeOperator*>(
      transpose_it->get(), OperatorType::kTranspose);

  if (transpose_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (!OperatorReady(*model, transpose_op) || transpose_op->perm.empty()) {
    return ::tensorflow::Status::OK();
  }

  if (!ReshapeIsEquivalentToTranspose(*model, reshape_op,
                                      false /*allow_extra_unary_dimensions*/)) {
    return ::tensorflow::Status::OK();
  }

  // Check that the intermediate is not an output array.
  if (!IsDiscardableArray(*model, intermediate_name)) {
    AddMessageF(
        "Cannot fuse %s and %s as it would invalidate the transpose "
        "output array.",
        LogName(*transpose_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }

  AddMessageF("Merging operations %s and %s", LogName(*transpose_op),
              LogName(*reshape_op));

  // const auto& intermediate_array = model->GetArray(intermediate_name);
  // const auto& output_array = model->GetArray(output_name);

  auto merged_perm = ReshapeToTranspose(*model, reshape_op);

  // Combine the permutations.
  const auto& transpose_perm = transpose_op->perm;
  for (size_t i = 0; i < merged_perm.size(); i++) {
    merged_perm[i] = transpose_perm[merged_perm[i]];
  }

  // Remove the reshape as passthrough operation.
  if (!RemoveTrivialPassthroughOp(this, model, op_index)) {
    return ::tensorflow::Status::OK();
  }

  // Update transpose_op's constant buffer to contain the new permutation.
  model->GetArray(transpose_op->inputs[1])
      .GetMutableBuffer<ArrayDataType::kInt32>()
      .data = merged_perm;
  transpose_op->perm = merged_perm;

  // transpose_ops's shape will likely has changed.
  model->GetArray(transpose_op->outputs[0]).clear_shape();

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
