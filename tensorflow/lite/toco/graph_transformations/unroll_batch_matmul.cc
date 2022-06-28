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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSunroll_batch_matmulDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSunroll_batch_matmulDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSunroll_batch_matmulDTcc() {
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
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {
namespace {

absl::InlinedVector<int64_t, 4> ToInlinedVector(const std::vector<int>& vec) {
  return absl::InlinedVector<int64_t, 4>(vec.begin(), vec.end());
}

std::vector<std::string> SliceInput(
    const std::string& input, const std::string& base_name,
    const std::string& input_name, const int batch_size,
    const Array& input_array, Model* model,
    std::vector<std::unique_ptr<Operator>>::iterator* tail_it) {
  int rank = input_array.shape().dimensions_count();
  int num_rows = input_array.shape().dims(rank - 2);
  int num_cols = input_array.shape().dims(rank - 1);
  // Reshape to rank-3 Tensor with first dimension as the batch size.
  auto* reshape_op = new TensorFlowReshapeOperator;
  reshape_op->inputs = {
      input,
      CreateInt32Array(model, absl::StrCat(base_name, "/reshape_a/shape"),
                       {batch_size, num_rows, num_cols})};
  reshape_op->outputs = {AvailableArrayName(
      *model, absl::StrCat(base_name, "/reshape_", input_name, "/reshape"))};
  auto& reshape_op_output = model->GetOrCreateArray(reshape_op->outputs[0]);
  reshape_op_output.data_type = input_array.data_type;
  *tail_it = model->operators.emplace(*tail_it, reshape_op) + 1;

  // Slice along each batch index and remember the slice output for future use.
  std::vector<std::string> slice_outputs;
  slice_outputs.reserve(batch_size);
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    std::string batch_name =
        absl::StrCat(base_name, "_b", batch_idx, "/slice_", input_name);
    auto* slice_op = new SliceOperator;
    slice_op->inputs = {
        reshape_op->outputs[0],
        CreateInt32Array(model, absl::StrCat(batch_name, "/slice/begin"),
                         {batch_idx, 0, 0}),
        CreateInt32Array(model, absl::StrCat(batch_name, "/slice/size"),
                         {1, num_rows, num_cols})};
    slice_op->outputs = {
        AvailableArrayName(*model, absl::StrCat(batch_name, "/slice"))};
    auto& slice_op_output = model->GetOrCreateArray(slice_op->outputs[0]);
    slice_op_output.data_type = input_array.data_type;
    *tail_it = model->operators.emplace(*tail_it, slice_op) + 1;

    // Reshape to rank-2: [1, num_rows, num_cols] -> [num_rows, num_cols].
    auto* slice_reshape_op = new TensorFlowReshapeOperator;
    slice_reshape_op->inputs = {
        slice_op->outputs[0],
        CreateInt32Array(model, absl::StrCat(batch_name, "/reshape/shape"),
                         {num_rows, num_cols})};
    slice_reshape_op->outputs = {
        AvailableArrayName(*model, absl::StrCat(batch_name, "/reshape"))};
    auto& slice_reshape_op_output =
        model->GetOrCreateArray(slice_reshape_op->outputs[0]);
    slice_reshape_op_output.data_type = input_array.data_type;
    *tail_it = model->operators.emplace(*tail_it, slice_reshape_op) + 1;

    slice_outputs.push_back(slice_reshape_op->outputs[0]);
  }
  return slice_outputs;
}

std::vector<int32> GetTransposePerm(const Array& input_array) {
  const int32_t dims = input_array.shape().dimensions_count();
  std::vector<int32> perm_array_val(dims);
  for (int32_t i = 0; i < dims; ++i) {
    perm_array_val[i] = i;
  }
  perm_array_val[dims - 2] = dims - 1;
  perm_array_val[dims - 1] = dims - 2;
  return perm_array_val;
}

std::vector<int32> GetTransposeShape(const Shape& input_shape,
                                     const std::vector<int32>& perm_array_val) {
  const int32_t dims = input_shape.dimensions_count();
  std::vector<int32> output_shape(dims);
  for (int32_t i = 0; i < dims; ++i) {
    output_shape[i] = input_shape.dims(perm_array_val[i]);
  }
  return output_shape;
}

TransposeOperator* TransposeInput(const std::string& input, Model* model) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSunroll_batch_matmulDTcc mht_0(mht_0_v, 285, "", "./tensorflow/lite/toco/graph_transformations/unroll_batch_matmul.cc", "TransposeInput");

  const auto& input_array = model->GetArray(input);
  const auto perm_array = GetTransposePerm(input_array);
  const std::string perm_array_name = CreateInt32Array(
      model, AvailableArrayName(*model, input + "/transpose/perm"), perm_array);
  auto* transpose_op = new TransposeOperator;
  transpose_op->inputs = {input, perm_array_name};
  transpose_op->outputs = {AvailableArrayName(*model, input + "/transpose")};
  auto& transpose_array = model->GetOrCreateArray(transpose_op->outputs[0]);
  *transpose_array.mutable_shape()->mutable_dims() =
      GetTransposeShape(input_array.shape(), perm_array);
  model->GetOrCreateArray(transpose_op->outputs[0]);
  return transpose_op;
}

}  // namespace

// Unrolls a BatchMatMul on the batch dimension.
// We need to slice each batch out of the inputs, matmul them individually, then
// stack them all back together at the end.
::tensorflow::Status UnrollBatchMatMul::Run(Model* model, std::size_t op_index,
                                            bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSunroll_batch_matmulDTcc mht_1(mht_1_v, 309, "", "./tensorflow/lite/toco/graph_transformations/unroll_batch_matmul.cc", "UnrollBatchMatMul::Run");

  *modified = false;
  auto batch_op_it = model->operators.begin() + op_index;
  if (batch_op_it->get()->type != OperatorType::kBatchMatMul) {
    return ::tensorflow::Status::OK();
  }
  const auto* batch_op =
      static_cast<const BatchMatMulOperator*>(batch_op_it->get());
  auto& tail_it = batch_op_it;

  std::string input_lhs = batch_op->inputs[0];
  std::string input_rhs = batch_op->inputs[1];
  const auto& input_lhs_array = model->GetArray(input_lhs);
  const auto& input_rhs_array = model->GetArray(input_rhs);
  if (!input_lhs_array.has_shape() || !input_rhs_array.has_shape())
    return ::tensorflow::Status::OK();

  // Transpose LHS input if necessary.
  if (batch_op->adj_x) {
    TransposeOperator* transpose_op = TransposeInput(input_lhs, model);
    tail_it = model->operators.emplace(tail_it, transpose_op) + 1;
    input_lhs = transpose_op->outputs[0];
  }
  const auto& input_array_a = model->GetArray(input_lhs);

  // Transpose RHS input if necessary.
  if (batch_op->adj_y) {
    TransposeOperator* transpose_op = TransposeInput(input_rhs, model);
    tail_it = model->operators.emplace(tail_it, transpose_op) + 1;
    input_rhs = transpose_op->outputs[0];
  }
  const auto& input_array_b = model->GetArray(input_rhs);

  // Ensure that input ranks are at least 2 and batch shapes are broadcastable.
  const int dims_a = input_array_a.shape().dimensions_count();
  const int dims_b = input_array_b.shape().dimensions_count();
  CHECK_GE(dims_a, 2) << "First input must have rank >= 2";
  CHECK_GE(dims_b, 2) << "Second input must have rank >= 2";

  ::tensorflow::MatMulBCast bcast(
      ToInlinedVector(input_array_a.shape().dims()),
      ToInlinedVector(input_array_b.shape().dims()));
  CHECK(bcast.IsValid()) << "Input batch dimensions must be broadcastable";

  CHECK_EQ(input_array_a.shape().dims(dims_a - 1),
           input_array_b.shape().dims(dims_b - 2))
      << "Input dimensions must be compatible for multiplication. shape a = ["
      << absl::StrJoin(input_array_a.shape().dims(), ", ") << "], shape b = ["
      << absl::StrJoin(input_array_b.shape().dims(), ", ") << "]";

  if (dims_a == 2 && dims_b == 2) {
    // This is really just a MatMul.
    AddMessageF("Replacing non-batch BatchMatMul %s by a MatMul operator",
                LogName(*batch_op));
    auto* matmul_op = new TensorFlowMatMulOperator;
    matmul_op->inputs = {input_lhs, input_rhs};
    matmul_op->outputs = batch_op->outputs;
    model->operators.emplace(tail_it, matmul_op);
    DeleteOpAndArrays(model, batch_op);
    *modified = true;
    return ::tensorflow::Status::OK();
  }
  AddMessageF("Unrolling BatchMatMul %s %d times", LogName(*batch_op),
              bcast.output_batch_size());
  std::string base_name = std::string(batch_op->outputs[0]);

  // Compute slices for each batch in the LHS and RHS.
  std::vector<std::string> slice_a_outputs =
      SliceInput(input_lhs, base_name, "a", bcast.x_batch_size(), input_array_a,
                 model, &tail_it);
  std::vector<std::string> slice_b_outputs =
      SliceInput(input_rhs, base_name, "b", bcast.y_batch_size(), input_array_b,
                 model, &tail_it);

  // Compute (single batch) MatMul for each output batch. The MatMul outputs are
  // then packed together into one output Tensor.
  std::vector<std::string> pack_inputs;
  for (int64_t batch_idx = 0; batch_idx < bcast.output_batch_size();
       ++batch_idx) {
    std::string batch_name =
        absl::StrCat(batch_op->outputs[0], "_b", batch_idx);
    const int a_batch_idx = bcast.IsBroadcastingRequired()
                                ? bcast.x_batch_indices()[batch_idx]
                                : batch_idx;
    const int b_batch_idx = bcast.IsBroadcastingRequired()
                                ? bcast.y_batch_indices()[batch_idx]
                                : batch_idx;
    auto* matmul_op = new TensorFlowMatMulOperator;
    matmul_op->inputs = {slice_a_outputs[a_batch_idx],
                         slice_b_outputs[b_batch_idx]};
    matmul_op->outputs = {AvailableArrayName(*model, batch_name)};
    auto& matmul_op_output = model->GetOrCreateArray(matmul_op->outputs[0]);
    matmul_op_output.data_type = input_array_a.data_type;
    tail_it = model->operators.emplace(tail_it, matmul_op) + 1;

    // Add to stack.
    pack_inputs.push_back(matmul_op->outputs[0]);
  }

  // Combine the result of each individual MatMul into a rank-3 Tensor.
  auto* pack_op = new PackOperator;
  pack_op->inputs = pack_inputs;
  pack_op->outputs = {AvailableArrayName(*model, base_name + "/pack")};
  auto& pack_op_output = model->GetOrCreateArray(pack_op->outputs[0]);
  pack_op_output.data_type = input_array_a.data_type;
  pack_op->axis = 0;
  pack_op->values_count = pack_inputs.size();
  tail_it = model->operators.emplace(tail_it, pack_op) + 1;

  // Reshape the rank-3 Tensor into the correct output shape.
  const auto& result_batch_shape = bcast.output_batch_shape().dim_sizes();
  std::vector<int> result_shape;
  // Explicitly cast 64-bit sizes to int in order to avoid MSVC warnings.
  std::transform(result_batch_shape.begin(), result_batch_shape.end(),
                 std::back_inserter(result_shape),
                 [](const int64_t dim) { return static_cast<int>(dim); });
  result_shape.push_back(input_array_a.shape().dims(dims_a - 2));
  result_shape.push_back(input_array_b.shape().dims(dims_b - 1));

  auto* reshape_result_op = new TensorFlowReshapeOperator;
  reshape_result_op->inputs = {
      pack_op->outputs[0],
      CreateInt32Array(model, base_name + "/reshape_out/shape", result_shape)};
  reshape_result_op->outputs = {batch_op->outputs[0]};
  model->operators.emplace(tail_it, reshape_result_op);

  DeleteOpAndArrays(model, batch_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
