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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc() {
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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool OperatorReady(const Model& model, const Operator* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/reorder_reshape_transpose.cc", "OperatorReady");

  if (!model.HasArray(op->inputs[0]) || !model.HasArray(op->inputs[1]) ||
      !model.HasArray(op->outputs[0])) {
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

// Utility function to filter out a value.
void Filter(std::vector<int>* vec, int value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/toco/graph_transformations/reorder_reshape_transpose.cc", "Filter");

  vec->erase(std::remove(vec->begin(), vec->end(), value), vec->end());
}

// Computes a new permutation used to swap a reshape-transpose to a
// transpose-reshape. In this case the permutation operates on the intermediate
// shape.
std::vector<int> ComputeNewPerm(std::vector<int> input_dims,
                                std::vector<int> intermediate_dims,
                                std::vector<int> perm) {
  // These are the major axis of the input.
  std::vector<int> input_indices;
  for (size_t i = 0; i < input_dims.size(); i++) {
    if (input_dims[i] != 1) {
      input_indices.push_back(i);
    }
  }

  // This maps which indices of the input produced the intermediate indices for
  // non-unary dimensions.
  std::unordered_map<int, int> intermediate_to_input_indices_map;
  for (size_t i = 0; i < intermediate_dims.size(); i++) {
    if (intermediate_dims[i] != 1) {
      intermediate_to_input_indices_map[i] =
          input_indices[intermediate_to_input_indices_map.size()];
    }
  }

  // Translate the transpose permutation to a new permutation starting with the
  // major indices.
  std::vector<int> new_perm;
  new_perm.reserve(input_dims.size());
  for (size_t i = 0; i < perm.size(); i++) {
    if (intermediate_dims[perm[i]] == 1) continue;

    new_perm.push_back(intermediate_to_input_indices_map[perm[i]]);
  }

  // Fill the rest of the transpose in with the ones.
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] == 1) {
      new_perm.push_back(index);
    }
  }

  CHECK_EQ(new_perm.size(), input_dims.size());
  return new_perm;
}

}  // namespace

// Swaps reshape-transpose to transpose-reshape whenever possible. This is
// possible when the reshape does not affect memory ordering.
::tensorflow::Status ReorderReshapeTranspose::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSreorder_reshape_transposeDTcc mht_2(mht_2_v, 281, "", "./tensorflow/lite/toco/graph_transformations/reorder_reshape_transpose.cc", "ReorderReshapeTranspose::Run");

  *modified = false;
  auto transpose_it = model->operators.begin() + op_index;

  TransposeOperator* transpose_op = ConvertOperator<TransposeOperator*>(
      transpose_it->get(), OperatorType::kTranspose);

  if (transpose_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (!OperatorReady(*model, transpose_op) || transpose_op->perm.empty()) {
    // Wait for values to propagate.
    return ::tensorflow::Status::OK();
  }

  // Find the operator that produces the transpose op.
  auto reshape_it = FindOpWithOutput(*model, transpose_op->inputs[0]);
  if (reshape_it == model->operators.end()) {
    return ::tensorflow::Status::OK();
  }

  TensorFlowReshapeOperator* reshape_op =
      ConvertOperator<TensorFlowReshapeOperator*>(reshape_it->get(),
                                                  OperatorType::kReshape);
  if (reshape_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  // Ignore if the reshape is uninitialized.
  if (!OperatorReady(*model, reshape_op) || reshape_op->shape.empty()) {
    return ::tensorflow::Status::OK();
  }

  // Need to copy to keep static if permutated.
  const std::string input_name = reshape_op->inputs[0];
  const std::string intermediate_name = reshape_op->outputs[0];
  const std::string output_name = transpose_op->outputs[0];

  // Intermediate should not be consumed by any other operators.
  if (CountOpsWithInput(*model, intermediate_name) != 1) {
    AddMessageF("Input %s used elsewhere", intermediate_name);
    return ::tensorflow::Status::OK();
  }

  // Check that the intermediate is not an output array.
  if (!IsDiscardableArray(*model, intermediate_name)) {
    AddMessageF(
        "Cannot reorder reshape-transpose as it would invalidate %s which is "
        "an output array.",
        intermediate_name);
    return ::tensorflow::Status::OK();
  }

  // Get the arrays.
  const auto& input_array = model->GetArray(input_name);
  const auto& intermediate_array = model->GetArray(intermediate_name);
  const auto& output_array = model->GetArray(output_name);

  // Get the shapes of each array.
  Shape input_shape = input_array.shape();
  Shape intermediate_shape = intermediate_array.shape();
  Shape output_shape = output_array.shape();

  // Assign ids to non-unary indices.
  std::vector<int> input_dims = input_shape.dims();
  std::vector<int> intermediate_dims = intermediate_shape.dims();
  std::vector<int> output_dims = output_shape.dims();

  // If the reshape is equivalent to a transpose with fewer/more unary
  // dimensions then it can be moved between the transpose.
  if (!ReshapeIsEquivalentToTranspose(*model, reshape_op,
                                      true /*allow_extra_unary_dims*/)) {
    return ::tensorflow::Status::OK();
  }

  if (!IsDiscardableArray(*model, output_name)) {
    // The output name of the sequence needs to stay static, so create a new
    // array new use for the intermediate.
    const auto new_intermediate_name =
        AvailableArrayName(*model, transpose_op->outputs[0] + "_exchange");
    AddMessageF("Adding new array %s to preserve output array name %s",
                new_intermediate_name, transpose_op->outputs[0]);
    transpose_op->inputs[0] = input_name;
    transpose_op->outputs[0] = new_intermediate_name;
    reshape_op->inputs[0] = new_intermediate_name;
    reshape_op->outputs[0] = output_name;
    DeleteArrayIfUnused(intermediate_name, model);
  } else {
    // The intermediate array is now the output array.
    for (size_t i = 0; i < model->operators.size(); i++) {
      Operator* consumer = model->operators[i].get();
      for (size_t j = 0; j < consumer->inputs.size(); j++) {
        if (consumer->inputs[j] == output_name) {
          consumer->inputs[j] = intermediate_name;
        }
      }
    }

    transpose_op->inputs[0] = input_name;
    reshape_op->inputs[0] = output_name;
  }

  // If transposes constant buffer is used elsewhere, make a new copy.
  if (CountOpsWithInput(*model, transpose_op->inputs[1]) != 1) {
    transpose_op->inputs[1] =
        AvailableArrayName(*model, transpose_op->inputs[1] + "_copy");
  }

  // Make the new transpose permutation.
  const std::vector<int> new_perm =
      ComputeNewPerm(input_dims, intermediate_dims, transpose_op->perm);
  CHECK_EQ(input_dims.size(), new_perm.size());

  auto& transpose_array = model->GetOrCreateArray(transpose_op->inputs[1]);
  transpose_array.data_type = ArrayDataType::kInt32;
  transpose_array.GetMutableBuffer<ArrayDataType::kInt32>().data = new_perm;
  *(transpose_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(new_perm.size())};
  transpose_op->perm = new_perm;

  // If the reshape's constant buffer is reused, create a new one.
  if (CountOpsWithInput(*model, reshape_op->inputs[1]) != 1) {
    reshape_op->inputs[1] =
        AvailableArrayName(*model, reshape_op->inputs[1] + "_copy");
  }

  // We need to modify the reshape input array to target the new output size.
  auto& reshape_array = model->GetOrCreateArray(reshape_op->inputs[1]);
  reshape_array.GetMutableBuffer<ArrayDataType::kInt32>().data = output_dims;
  *(reshape_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(output_shape.dimensions_count())};
  reshape_op->shape.clear();

  AddMessageF("Swapping around operators between %s and %s", input_name,
              output_name);

  model->GetOrCreateArray(transpose_op->outputs[0]).clear_shape();
  model->GetOrCreateArray(reshape_op->outputs[0]).clear_shape();

  // Swap the order of the operators.
  transpose_it->swap(*reshape_it);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
