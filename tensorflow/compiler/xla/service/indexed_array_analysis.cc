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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {
using Analysis = IndexedArrayAnalysis;
using UnknownArray = Analysis::UnknownArray;
using ConstantArray = Analysis::ConstantArray;
using ReshapedArray = Analysis::ReshapedArray;
using ScalarIndexedArray = Analysis::ScalarIndexedArray;
using absl::StrJoin;
}  // namespace

std::string IndexedArrayAnalysis::ToString(Array* root, bool print_constants) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ToString");

  switch (root->kind()) {
    case Array::kUnknown: {
      auto* unknown_tensor = root->as<UnknownArray>();
      return absl::StrCat("%", unknown_tensor->instruction().name());
    }

    case Array::kConstant: {
      if (print_constants) {
        std::string contents = root->as<ConstantArray>()->literal()->ToString();
        return absl::StrCat("(constant ", ShapeUtil::HumanString(root->shape()),
                            " ", contents, ")");
      }
      return absl::StrCat("(constant ", ShapeUtil::HumanString(root->shape()),
                          ")");
    }

    case Array::kReshaped: {
      ReshapedArray* reshaped_array = root->as<ReshapedArray>();
      return absl::StrCat(
          "(reshape ", ToString(reshaped_array->operand(), print_constants),
          " to ", ShapeUtil::HumanString(reshaped_array->shape()), ")");
    }

    case Array::kScalarIndexedConstant:
    case Array::kScalarIndexed: {
      auto* indexed_array = root->as<ScalarIndexedArray>();
      std::string name = root->kind() == Array::kScalarIndexedConstant
                             ? "scalar-indexed-const"
                             : "scalar-indexed";
      return absl::StrCat(
          "(", name, " ", ToString(indexed_array->source(), print_constants),
          " ", ToString(indexed_array->indices(), print_constants), " ",
          indexed_array->source_dim(), "->[",
          StrJoin(indexed_array->output_dims(), ","), "])");
    }
  }
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::GetArrayFor(
    const HloInstruction* instr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::GetArrayFor");

  auto it = cache_.find(instr);
  if (it != cache_.end()) {
    return it->second;
  }

  TF_RETURN_IF_ERROR(TraverseAndPopulateCache(instr));
  return FindOrDie(cache_, instr);
}

Status IndexedArrayAnalysis::TraverseAndPopulateCache(
    const HloInstruction* root) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::TraverseAndPopulateCache");

  // Depth first search over the DAG, invoking ComputeArrayFor in post order.
  // The HLO instructions already in the cache are considered leaves.

  absl::InlinedVector<const HloInstruction*, 4> stack;

  enum DfsState { kDiscovered, kVisited };
  absl::flat_hash_map<const HloInstruction*, DfsState> dfs_state_map;

  stack.push_back(root);
  InsertOrDie(&dfs_state_map, root, kDiscovered);

  do {
    const HloInstruction* instr = stack.back();
    if (cache_.contains(instr)) {
      stack.pop_back();
      continue;
    }

    switch (FindOrDie(dfs_state_map, instr)) {
      case kDiscovered: {
        for (const HloInstruction* operand : instr->operands()) {
          if (!cache_.contains(operand)) {
            stack.push_back(operand);
            CHECK(!dfs_state_map.contains(operand) ||
                  dfs_state_map[operand] == kDiscovered);
            dfs_state_map[operand] = kDiscovered;
          }
        }
        dfs_state_map[instr] = kVisited;
        break;
      }

      case kVisited:
        stack.pop_back();
        TF_ASSIGN_OR_RETURN(Array * array, ComputeArrayFor(instr));
        InsertOrDie(&cache_, instr, array);
        break;
    }
  } while (!stack.empty());

  return Status::OK();
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayFor(
    const HloInstruction* instr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_3(mht_3_v, 319, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayFor");

  Array* computed_array;
  if (instr->IsElementwise() && instr->operand_count() == 1) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForElementwiseUnaryOp(
            instr->opcode(), FindOrDie(cache_, instr->operand(0))));
  } else if (instr->IsElementwise() && instr->operand_count() == 2) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForElementwiseBinaryOp(
            instr->opcode(), FindOrDie(cache_, instr->operand(0)),
            FindOrDie(cache_, instr->operand(1))));
  } else if (instr->opcode() == HloOpcode::kConstant) {
    TF_ASSIGN_OR_RETURN(computed_array,
                        ComputeArrayForConstant(instr->literal()));
  } else if (instr->opcode() == HloOpcode::kGather) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForGather(instr->shape(), instr->gather_dimension_numbers(),
                              instr->gather_slice_sizes(),
                              FindOrDie(cache_, instr->operand(0)),
                              FindOrDie(cache_, instr->operand(1))));
  } else if (instr->opcode() == HloOpcode::kReshape) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForReshape(instr->shape(),
                               FindOrDie(cache_, instr->operand(0))));
  } else if (instr->opcode() == HloOpcode::kDot) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForDot(instr->shape(), instr->dot_dimension_numbers(),
                           instr->precision_config(),
                           FindOrDie(cache_, instr->operand(0)),
                           FindOrDie(cache_, instr->operand(1))));
  } else {
    computed_array = nullptr;
  }

  if (!computed_array) {
    computed_array = Construct<UnknownArray>(instr);
  }

  return computed_array;
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForConstant(
    const Literal& literal) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_4(mht_4_v, 369, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForConstant");

  return Construct<ConstantArray>(&literal);
}

StatusOr<ScalarIndexedArray*> IndexedArrayAnalysis::FoldGatherOfGather(
    ScalarIndexedArray* source, Array* indices, int64_t source_dim,
    absl::Span<const int64_t> output_dims, Shape shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_5(mht_5_v, 378, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::FoldGatherOfGather");

  // We want to transform Gather(Gather(A, X), Y) => Gather(A, Gather(X, Y)).
  // `source` is the inner Gather(A, X).

  Array* a = source->source();
  Array* x = source->indices();
  Array* y = indices;

  // This bit is slightly tricky, so we do a naive "simulation" of the two
  // consecutive gather operations to infer what the composed gather should look
  // like.

  enum class IndexComponent { Ungathered, GatheredFirst, GatheredSecond };

  std::vector<IndexComponent> simulated_index(a->shape().dimensions_size(),
                                              IndexComponent::Ungathered);

  // Simulate the first gather.
  EraseAt(&simulated_index, source->source_dim());
  for (int64_t gather_dim : source->output_dims()) {
    simulated_index.insert(simulated_index.begin() + gather_dim,
                           IndexComponent::GatheredFirst);
  }

  // Simulate the second gather.
  EraseAt(&simulated_index, source_dim);
  for (int64_t output_dim : output_dims) {
    simulated_index.insert(simulated_index.begin() + output_dim,
                           IndexComponent::GatheredSecond);
  }

  int64_t source_dim_for_index_array =
      FindIndex(source->output_dims(), source_dim);
  CHECK_NE(source_dim_for_index_array, source->output_dims().size());

  std::vector<int64_t> output_dims_for_index_array;
  int64_t gathered_index_components_seen = 0;
  for (IndexComponent simulation_dim : simulated_index) {
    if (simulation_dim == IndexComponent::GatheredSecond) {
      output_dims_for_index_array.push_back(gathered_index_components_seen);
    }
    if (simulation_dim != IndexComponent::Ungathered) {
      gathered_index_components_seen++;
    }
  }

  std::vector<int64_t> dim_sizes_for_composed_index;
  std::vector<int64_t> output_dims_for_new_gather;
  for (int64_t i = 0, e = simulated_index.size(); i < e; i++) {
    if (simulated_index[i] != IndexComponent::Ungathered) {
      dim_sizes_for_composed_index.push_back(shape.dimensions(i));
      output_dims_for_new_gather.push_back(i);
    }
  }

  Array* inner_indices = ConstructScalarIndexedArray(
      x, y, source_dim_for_index_array, output_dims_for_index_array,
      ShapeUtil::MakeShape(x->shape().element_type(),
                           dim_sizes_for_composed_index));
  return ConstructScalarIndexedArray(a, inner_indices, source->source_dim(),
                                     output_dims_for_new_gather,
                                     std::move(shape));
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForGather(
    const Shape& shape, const GatherDimensionNumbers& dim_numbers,
    absl::Span<const int64_t> slice_sizes, Array* source, Array* indices) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_6(mht_6_v, 447, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForGather");

  if (dim_numbers.index_vector_dim() != indices->shape().dimensions_size()) {
    VLOG(3) << "ComputeArrayForGather: indices are not scalar";
    return nullptr;
  }

  CHECK_EQ(dim_numbers.start_index_map_size(), 1);

  // We can also handle dim_numbers.collapsed_slice_dims_size() == 0 here,
  // should it become relevant.

  if (dim_numbers.collapsed_slice_dims_size() != 1 ||
      dim_numbers.collapsed_slice_dims(0) != dim_numbers.start_index_map(0)) {
    VLOG(3) << "ComputeArrayForGather: gather operations must elide "
               "start_index_map[0] and "
               "start_index_map[0] only";
    return nullptr;
  }

  // ScalarIndexedArray cannot represent gathers that "slice" along some
  // dimensions -- for instance it cannot represent a gather that picks 5 [2,3]
  // arrays from an array of size [7,4,6].  We check that condition down below:

  for (int64_t i = 0, e = source->shape().dimensions_size(); i < e; i++) {
    if (i != dim_numbers.collapsed_slice_dims(0) &&
        source->shape().dimensions(i) != slice_sizes[i]) {
      VLOG(3) << "ComputeArrayForGather: slice_sizes[" << i
              << "] != source->shape().dimensions(" << i << ") -- "
              << source->shape().dimensions(i) << " vs. " << slice_sizes[i]
              << " with dim_numbers.collapsed_slice_dims(0) = "
              << dim_numbers.collapsed_slice_dims(0);
      return nullptr;
    }
  }

  int64_t source_dim = dim_numbers.start_index_map(0);
  std::vector<int64_t> output_dims;
  for (int64_t i = 0, e = shape.dimensions_size(); i < e; i++) {
    if (!absl::c_binary_search(dim_numbers.offset_dims(), i)) {
      output_dims.push_back(i);
    }
  }

  if (auto* indexed = dynamic_cast<ScalarIndexedArray*>(source)) {
    if (absl::c_linear_search(indexed->output_dims(), source_dim)) {
      return FoldGatherOfGather(indexed, indices, source_dim, output_dims,
                                shape);
    }
  } else if (auto* constant = dynamic_cast<ConstantArray*>(source)) {
    return Construct<ScalarIndexedConstantArray>(constant, indices, source_dim,
                                                 output_dims, shape);
  }

  return Construct<ScalarIndexedArray>(source, indices, source_dim, output_dims,
                                       shape);
}

namespace {
// Returns an index into `values` such that the product of the range
// [values.begin()+index, values.end()) is equal to `product`.  If there is no
// such index, return -1.  All integers in `values` must be positive.
int64_t FindSuffixWithProduct(absl::Span<const int64_t> values,
                              int64_t product) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_7(mht_7_v, 512, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "FindSuffixWithProduct");

  DCHECK(absl::c_all_of(values, [](int64_t value) { return value > 0; }));

  int64_t current_product = 1;
  int64_t i;
  for (i = values.size() - 1; i >= 0 && product > current_product; --i) {
    current_product *= values[i];
  }

  if (product == current_product) {
    return i + 1;
  }

  return -1;
}

struct ReshapePassthroughDimPair {
  int64_t result_dim;
  int64_t operand_dim;
};

// Returns a set of dimension pairs such for all (result_dim, operand_dim) in
// the set:
//
// output_index[result_dim] = SourceIndexOfReshape(output_index)[operand_dim]
//
// The returned vector of pairs is sorted in both the result_dim and the
// operand_dim components.
std::vector<ReshapePassthroughDimPair> ComputeReshapePassthroughDimPairs(
    absl::Span<const int64_t> operand_shape,
    absl::Span<const int64_t> result_shape) {
  // A reshape can be seen as an index mapping from output index to input index:
  //
  // (i_0, ..., i_n) = f(o_0, ..., o_m)
  //
  // This function returns the pairs (j, k) for which the following invariant
  // holds for all indices in the shape:
  //
  //   o_j == i_k
  //
  // And this occurs when:
  //
  //    O_{j+1} * ... * O_n == I_{k+1} * ...  * I_m
  //
  // (where O_x are the sizes of the output shape and I_x are the sizes of the
  // input shape) and the size of the dimension j of the result is the same as
  // the size of dimension k in the operand.
  //
  // These conditions are sufficient because the Reshape HLO is spec'ed such
  // that the rightmost dimensions are always minor in the flattening and refine
  // operation.

  std::vector<ReshapePassthroughDimPair> result;
  int64_t result_subarray_size = 1;
  for (int64_t result_dim = result_shape.size() - 1; result_dim >= 0;
       --result_dim) {
    int64_t candidate_operand_dim =
        FindSuffixWithProduct(operand_shape, result_subarray_size);

    // result_subarray_size does not include the elements in the current
    // `result_dim` dimension (we multiply in result_shape[result_dim] at the
    // end of loop body) so candidate_operand_dim can never be zero.
    CHECK_NE(candidate_operand_dim, 0)
        << "result_dim = " << result_dim
        << ", result_subarray_size = " << result_subarray_size
        << ", result_shape = [" << StrJoin(result_shape, ",") << "]"
        << ", operand_shape = [" << StrJoin(operand_shape, ",") << "]";

    if (candidate_operand_dim != -1 &&
        result_shape[result_dim] == operand_shape[candidate_operand_dim - 1]) {
      result.push_back({/*result_dim=*/result_dim,
                        /*operand_dim=*/candidate_operand_dim - 1});
    }
    result_subarray_size *= result_shape[result_dim];
  }

  absl::c_reverse(result);

  if (VLOG_IS_ON(3)) {
    std::vector<std::string> result_strings;
    absl::c_transform(result, std::back_inserter(result_strings),
                      [](ReshapePassthroughDimPair value) {
                        return absl::StrCat(value.result_dim, "->",
                                            value.operand_dim);
                      });
    VLOG(3) << "For a reshape from [" << StrJoin(operand_shape, ",") << "] to ["
            << StrJoin(result_shape, ",") << "] passthrough indices are ["
            << StrJoin(result_strings, ",")
            << "] (legend: `result`->`operand`)";
  }

  DCHECK(absl::c_is_sorted(
      result, [](ReshapePassthroughDimPair lhs, ReshapePassthroughDimPair rhs) {
        return lhs.result_dim < rhs.result_dim;
      }));

  DCHECK(absl::c_is_sorted(
      result, [](ReshapePassthroughDimPair lhs, ReshapePassthroughDimPair rhs) {
        return lhs.operand_dim < rhs.operand_dim;
      }));

  return result;
}

// Return true if `dim` is stated as an passthrough operand dim in
// `passthrough_dims`.
bool IsReshapePassthroughOperandDim(
    absl::Span<const ReshapePassthroughDimPair> passthrough_dims, int64_t dim) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_8(mht_8_v, 622, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IsReshapePassthroughOperandDim");

  return absl::c_any_of(passthrough_dims,
                        [&](ReshapePassthroughDimPair passthrough_dim_pair) {
                          return passthrough_dim_pair.operand_dim == dim;
                        });
}

// Maps `operand_dim` which must be an passthrough operand dimension to its
// corresponding passthrough result dimension based on `passthrough_dims`.
int64_t MapPassthroughOperandDimToResultDim(
    absl::Span<const ReshapePassthroughDimPair> passthrough_dims,
    int64_t operand_dim) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_9(mht_9_v, 636, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "MapPassthroughOperandDimToResultDim");

  auto it = absl::c_find_if(
      passthrough_dims, [&](ReshapePassthroughDimPair passthrough_dim_pair) {
        return passthrough_dim_pair.operand_dim == operand_dim;
      });
  CHECK(it != passthrough_dims.end());
  return it->result_dim;
}

int64_t FindSourcePositionForPassthroughResultDim(
    absl::Span<const int64_t> operand_shape,
    absl::Span<const int64_t> result_shape, int64_t source_passthrough_dim) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_10(mht_10_v, 650, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "FindSourcePositionForPassthroughResultDim");

  VLOG(3) << "FindSourcePositionForPassthroughResultDim(["
          << StrJoin(operand_shape, ",") << "], [" << StrJoin(result_shape, ",")
          << "], " << source_passthrough_dim << ")";

  int64_t indexed_source_subarray_size =
      std::accumulate(operand_shape.begin() + source_passthrough_dim + 1,
                      operand_shape.end(), 1LL, std::multiplies<int64_t>());

  return FindSuffixWithProduct(result_shape, indexed_source_subarray_size);
}

Shape StripDegenerateDimensions(const Shape& shape) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_11(mht_11_v, 665, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "StripDegenerateDimensions");

  DimensionVector new_dims;
  absl::c_copy_if(shape.dimensions(), std::back_inserter(new_dims),
                  [](int64_t dim) { return dim != 1; });
  return ShapeUtil::MakeShape(shape.element_type(), new_dims);
}
};  // namespace

StatusOr<ScalarIndexedArray*>
IndexedArrayAnalysis::ReshapeToRemoveDegenerateDims(
    ScalarIndexedArray* operand) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_12(mht_12_v, 678, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ReshapeToRemoveDegenerateDims");

  const Shape& shape = operand->shape();
  if (!ShapeUtil::HasDegenerateDimensions(shape)) {
    return operand;
  }

  // We only need to reshape out the degenerate dims from the indices and the
  // source (except the source dim).

  const Shape& source_shape = operand->source()->shape();
  DimensionVector new_source_shape_dims;
  for (int64_t i = 0, e = source_shape.dimensions_size(); i < e; i++) {
    if (i == operand->source_dim() || source_shape.dimensions(i) != 1) {
      new_source_shape_dims.push_back(source_shape.dimensions(i));
    }
  }

  Shape new_source_shape =
      ShapeUtil::MakeShape(shape.element_type(), new_source_shape_dims);
  Shape new_indices_shape =
      StripDegenerateDimensions(operand->indices()->shape());

  TF_ASSIGN_OR_RETURN(
      Array* const new_source,
      ComputeArrayForReshape(new_source_shape, operand->source()));
  TF_ASSIGN_OR_RETURN(
      Array* const new_indices,
      ComputeArrayForReshape(new_indices_shape, operand->indices()));

  // Build the new output dims while keeping track of the degenerate dims that
  // will no longer be present.
  DimensionVector new_output_dims;
  int64_t degenerate_dims_seen = 0;
  for (int64_t i = 0, e = shape.dimensions_size(); i < e; i++) {
    if (shape.dimensions(i) == 1) {
      degenerate_dims_seen++;
    } else if (absl::c_linear_search(operand->output_dims(), i)) {
      new_output_dims.push_back(i - degenerate_dims_seen);
    }
  }

  // Similarly, build the new source dim while keeping track of the degenerate
  // dims that will no longer be present.
  int64_t degenerate_dims_before_source_dim =
      std::count(source_shape.dimensions().begin(),
                 source_shape.dimensions().begin() + operand->source_dim(), 1);
  int64_t new_source_dim =
      operand->source_dim() - degenerate_dims_before_source_dim;

  return ConstructScalarIndexedArray(
      new_source, new_indices, new_source_dim,
      InlinedVectorToVector(new_output_dims),
      StripDegenerateDimensions(operand->shape()));
}

StatusOr<ScalarIndexedArray*> IndexedArrayAnalysis::ReshapeToAddDegenerateDims(
    ScalarIndexedArray* operand, absl::Span<const int64_t> degenerate_dims) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_13(mht_13_v, 737, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ReshapeToAddDegenerateDims");

  if (degenerate_dims.empty()) {
    return operand;
  }

  CHECK(!ShapeUtil::HasDegenerateDimensions(operand->shape()));

  DimensionVector new_output_dims = [&]() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_14(mht_14_v, 747, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "lambda");

    // To make things easy we use a "scratch" buffer of bools where the i'th
    // element is true iff the i'th component of the result index is an output
    // index.

    absl::InlinedVector<bool, 6> output_dims_bitvector(
        operand->shape().dimensions_size());
    for (int64_t output_dim : operand->output_dims()) {
      output_dims_bitvector[output_dim] = true;
    }

    for (int64_t degenerate_dim : degenerate_dims) {
      InsertAt(&output_dims_bitvector, degenerate_dim, false);
    }

    DimensionVector result;
    result.reserve(operand->output_dims().size());
    for (int64_t i = 0, e = output_dims_bitvector.size(); i < e; i++) {
      if (output_dims_bitvector[i]) {
        result.push_back(i);
      }
    }

    return result;
  }();

  DimensionVector new_result_shape_dims;
  absl::c_copy(operand->shape().dimensions(),
               std::back_inserter(new_result_shape_dims));
  for (int64_t degenerate_dim : degenerate_dims) {
    InsertAt(&new_result_shape_dims, degenerate_dim, 1);
  }

  DimensionVector new_source_shape_dims = new_result_shape_dims;
  for (int64_t output_dim : new_output_dims) {
    EraseAt(&new_source_shape_dims, output_dim);
  }

  int64_t new_source_dim = [&]() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_15(mht_15_v, 788, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "lambda");

    for (int i = 0, e = new_source_shape_dims.size(); i < e; i++) {
      int64_t non_degenerate_dims_seen = 0;
      if (non_degenerate_dims_seen == operand->source_dim()) {
        return i;
      }
      if (new_source_shape_dims[new_source_dim] != 1) {
        non_degenerate_dims_seen++;
      }
    }
    LOG(FATAL) << "Did not find source dim in " << ToString(operand);
  }();

  int64_t source_dim_size =
      operand->source()->shape().dimensions(operand->source_dim());
  InsertAt(&new_source_shape_dims, /*index=*/new_source_dim,
           /*value=*/source_dim_size);

  Shape new_source_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                                new_source_shape_dims);
  Shape new_result_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                                new_result_shape_dims);

  TF_ASSIGN_OR_RETURN(
      Array* const new_source,
      ComputeArrayForReshape(new_source_shape, operand->source()));
  return ConstructScalarIndexedArray(
      new_source, operand->indices(), new_source_dim,
      InlinedVectorToVector(new_output_dims), new_result_shape);
}

StatusOr<ScalarIndexedArray*> IndexedArrayAnalysis::FoldReshapeOfGather(
    const Shape& shape, ScalarIndexedConstantArray* operand) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_16(mht_16_v, 823, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::FoldReshapeOfGather");

  VLOG(3) << "FoldReshapeOfGather(" << ToString(operand) << ")";

  // To make things easier on ourselves, instead of directly trying to fold the
  // reshape of `operand` to `shape`, we call
  // `FoldReshapeOfGatherNoDegenerateDims` on shapes without degenerate dims and
  // handle the degenerate dimensions here by inserting reshapes.

  TF_ASSIGN_OR_RETURN(ScalarIndexedArray* const operand_without_degenerate_dims,
                      ReshapeToRemoveDegenerateDims(operand));

  Shape output_shape_without_degenerate_dims = StripDegenerateDimensions(shape);
  TF_ASSIGN_OR_RETURN(
      ScalarIndexedArray* const folded_reshape_without_degenerate_dims,
      FoldReshapeOfGatherNoDegenerateDims(
          output_shape_without_degenerate_dims,
          operand_without_degenerate_dims->as<ScalarIndexedConstantArray>()));

  if (folded_reshape_without_degenerate_dims == nullptr) {
    return nullptr;
  }

  DimensionVector degenerate_result_dims;
  for (int64_t i = 0, e = shape.dimensions_size(); i < e; i++) {
    if (shape.dimensions(i) == 1) {
      degenerate_result_dims.push_back(i);
    }
  }

  return ReshapeToAddDegenerateDims(folded_reshape_without_degenerate_dims,
                                    degenerate_result_dims);
}

StatusOr<ScalarIndexedArray*>
IndexedArrayAnalysis::FoldReshapeOfGatherNoDegenerateDims(
    const Shape& shape, ScalarIndexedConstantArray* scalar_indexed) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_17(mht_17_v, 861, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::FoldReshapeOfGatherNoDegenerateDims");

  VLOG(3) << "FoldReshapeOfGatherNoDegenerateDims(" << ToString(scalar_indexed)
          << ")";
  CHECK(!ShapeUtil::HasDegenerateDimensions(shape));
  CHECK(!ShapeUtil::HasDegenerateDimensions(scalar_indexed->shape()));

  // Try to fold Reshape(ScalarIndexed(Const, Indices))
  //          => ScalarIndexed(Const', Indices)
  //
  // We can view the reshape and the scalar-indexed operations as functions that
  // map an output index (i.e. an index into the result) to an input index
  // (i.e. an index into the operand).  The key idea used here is that the
  // output-to-input mapping for some reshape operations may "pass through" some
  // output dimensions into the input space unchanged -- i.e. there may exist
  // output dimension "O" and input dimension "I" such that OutputIndex[O] is
  // always == InputIndexForReshape(OutputIndex)[I].  If these pass-through
  // dimensions in the input space of the reshape happen to be include all the
  // output dimensions for the scalar-indexed node then, roughly, the following
  // holds:
  //
  //    SourceIndexOfScalarIndexed(SourceIndexOfReshape(Idx))
  // == SourceIndexOfScalarIndexed(SourceIndexOfReshape(Ps ++ Qs))
  //
  //      Where Ps are the set of the pass-through components of Idx that are
  //      also the output dims of the scalar-indexed node, and Qs are the rest.
  //      For brevity, we're playing fast and loose with the notation here -- we
  //      don't literally require Idx to be a concatenation of Ps and Qs, as
  //      suggested by the "++".
  //
  // == SourceIndexOfScalarIndexed(Ps ++ SourceIndexOfReshape(Qs))
  //
  //      Again, we're playing fast and loose with the notation around "++".
  //      Generally this ++ will be a different function that the ++ in the
  //      previous step.
  //
  // If the scalar-indexed node has a constant as the source then the
  // SourceIndexOfReshape function can be "folded into" the constant itself by
  // reshaping it, leaving us with:
  //
  // == SourceIndexOfScalarIndexed(Ps ++ Qs)
  // == SourceIndexOfScalarIndexed(Idx)
  //
  // which is just a scalar-indexed node (with parameters different from the
  // scalar-indexed node we started with) with a reshaped constant as the
  // source.
  //
  // We can't fold SourceIndexOfReshape into the constant without introducing
  // another precondition: since the new scalar-indexed node will have a
  // reshaped (constant) array as its source it will, in general, have a
  // different source dimension than the original scalar-indexed node.  This
  // source dimension will have to be a passthrough dimension of the
  // SourceIndexOfReshape indexing function that is folded into the source. And
  // such a dimension need not exist so this is a non-trivial precondition.

  std::vector<ReshapePassthroughDimPair> reshape_passthrough_dims =
      ComputeReshapePassthroughDimPairs(
          /*operand_shape=*/scalar_indexed->shape().dimensions(),
          /*result_shape=*/shape.dimensions());

  auto is_reshape_passthrough_operand_dim = [&](int64_t operand_dim) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_18(mht_18_v, 923, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "lambda");

    return IsReshapePassthroughOperandDim(reshape_passthrough_dims,
                                          operand_dim);
  };

  if (!absl::c_all_of(scalar_indexed->output_dims(),
                      is_reshape_passthrough_operand_dim)) {
    VLOG(3) << "Not all output dims are passthrough dims "
            << ToString(scalar_indexed);
    return nullptr;
  }

  // To compute the shape of the source for the new scalar-indexed node we're
  // going to create, we first "undo" the scalar-indexed operation.
  std::vector<int64_t> new_scalar_indexed_source_shape(
      shape.dimensions().begin(), shape.dimensions().end());
  for (int64_t i = scalar_indexed->output_dims().size() - 1; i >= 0; i--) {
    int64_t output_dim = scalar_indexed->output_dims()[i];
    int64_t output_dim_after_reshape = MapPassthroughOperandDimToResultDim(
        reshape_passthrough_dims, output_dim);
    EraseAt(&new_scalar_indexed_source_shape, output_dim_after_reshape);
  }

  // After this, we need to add in the dimension that will be the source
  // dimension for the new scalar-indexed node.  A scalar-indexed node "removes"
  // the source dimensions and "adds" the output dimensions, so to get back to
  // the shape for the *source* of the scalar-indexed node we need to remove the
  // output dims (which we did above) and then add back the source dim (which we
  // are about to do below):

  const Shape& scalar_indexed_source_shape = scalar_indexed->source()->shape();

  int64_t source_dim_for_new_scalar_indexed_node =
      FindSourcePositionForPassthroughResultDim(
          /*operand_shape=*/scalar_indexed_source_shape.dimensions(),
          /*result_shape=*/new_scalar_indexed_source_shape,
          scalar_indexed->source_dim());

  // We may not be able to find a source dim for the new scalar-indexed node.
  // For instance consider:
  //
  //   operand = s32[3,5,2] constant({...})
  //   indices = s32[7] parameter(0)
  //   gather = s32[3,2,7] gather(operand, indices),
  //       offset_dims={0,1},
  //       collapsed_slice_dims={1},
  //       start_index_map={1},
  //       index_vector_dim=1,
  //       slice_sizes={3,1,2}
  //   reshape = s32[6,7] reshape(gather)
  //
  // In this case the gather maps to:
  //    (scalar-indexed-const (constant s32[3,5,2]) %indices 1->[2])
  //
  // and the reshape passes through dimension 2 from its input into dimension 1
  // in its output.  However, we can't rewrite the reshape as a scalar-indexed
  // node because then we'd have to reshape the [3,5,2] `operand` array to
  // [6,5], but then dimension 1 of the reshaped [6,5] array indexes differently
  // (a.k.a. isn't pass-through) than the [3,5,2] array.

  if (source_dim_for_new_scalar_indexed_node == -1) {
    VLOG(3) << "Could not compute the source dim for the new scalar indexed "
               "node: scalar_indexed_source_shape = ["
            << StrJoin(scalar_indexed_source_shape.dimensions(), ",")
            << "] and new_scalar_indexed_source_shape = ["
            << StrJoin(new_scalar_indexed_source_shape, ",") << "]";
    return nullptr;
  }

  InsertAt(
      &new_scalar_indexed_source_shape, source_dim_for_new_scalar_indexed_node,
      scalar_indexed_source_shape.dimensions(scalar_indexed->source_dim()));

  CHECK_EQ(absl::c_accumulate(new_scalar_indexed_source_shape, 1LL,
                              std::multiplies<int64_t>()),
           ShapeUtil::ElementsIn(scalar_indexed_source_shape));

  CHECK(IsReshapePassthroughOperandDim(
      ComputeReshapePassthroughDimPairs(
          /*operand_shape=*/scalar_indexed_source_shape.dimensions(),
          /*result_shape=*/new_scalar_indexed_source_shape),
      scalar_indexed->source_dim()));

  auto map_passthrough_operand_dim_to_result_dim = [&](int64_t result_dim) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_19(mht_19_v, 1009, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "lambda");

    return MapPassthroughOperandDimToResultDim(reshape_passthrough_dims,
                                               result_dim);
  };

  std::vector<int64_t> output_dims_for_new_scalar_indexed_node;
  absl::c_transform(scalar_indexed->output_dims(),
                    std::back_inserter(output_dims_for_new_scalar_indexed_node),
                    map_passthrough_operand_dim_to_result_dim);

  TF_ASSIGN_OR_RETURN(const Literal* new_scalar_indexed_source_literal,
                      TakeOwnership(scalar_indexed->literal().Reshape(
                          new_scalar_indexed_source_shape)));
  TF_ASSIGN_OR_RETURN(
      Array * new_scalar_indexed_source,
      ComputeArrayForConstant(*new_scalar_indexed_source_literal));

  return ConstructScalarIndexedArray(
      new_scalar_indexed_source, scalar_indexed->indices(),
      source_dim_for_new_scalar_indexed_node,
      output_dims_for_new_scalar_indexed_node, shape);
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForReshape(
    const Shape& shape, Array* operand) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_20(mht_20_v, 1036, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForReshape");

  if (ShapeUtil::Compatible(operand->shape(), shape)) {
    return operand;
  }

  if (auto* scalar_indexed =
          dynamic_cast<ScalarIndexedConstantArray*>(operand)) {
    TF_ASSIGN_OR_RETURN(Analysis::Array * reshape_folded_into_gather,
                        FoldReshapeOfGather(shape, scalar_indexed));
    if (reshape_folded_into_gather) {
      return reshape_folded_into_gather;
    }
  }

  if (auto* constant_array = dynamic_cast<ConstantArray*>(operand)) {
    TF_ASSIGN_OR_RETURN(
        Literal* const new_literal,
        TakeOwnership(constant_array->literal()->Reshape(shape.dimensions())));
    return Construct<ConstantArray>(new_literal);
  }

  return Construct<ReshapedArray>(operand, shape);
}

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForElementwiseBinaryOp(HloOpcode opcode,
                                                         Array* lhs,
                                                         Array* rhs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_21(mht_21_v, 1066, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForElementwiseBinaryOp");

  // Try to fold BinaryOp(Broadcast(Const0), ScalarIndexed(Const1, Indices))
  //          => ScalarIndexed(BinaryOp(Broadcast'(Const0), Const1), Indices)
  //
  // We can do this if every output dimension from the scalar-indexed node is a
  // broadcasted dimension for the broadcast node.  Informally, the precondition
  // means Broadcast(Const0)[IDX] is solely a function of the components of IDX
  // that are not output-dims for the scalar-indexed node. In other words, for
  // every assignment to the non-output dims in IDX we have a "constant" LHS to
  // the BinaryOp.  This transform propagates this "constant" to the source for
  // the scalar-indexed node.

  ScalarIndexedConstantArray* lhs_scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(lhs);
  ScalarIndexedConstantArray* rhs_scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(rhs);

  bool lhs_is_indexed;

  // One of the operands must be scalar-indexed and the other must be a
  // broadcast of a constant.
  if (lhs_scalar_indexed_const && !rhs_scalar_indexed_const) {
    lhs_is_indexed = true;
  } else if (rhs_scalar_indexed_const && !lhs_scalar_indexed_const) {
    lhs_is_indexed = false;
  } else {
    return nullptr;
  }

  ScalarIndexedConstantArray* scalar_indexed_const =
      lhs_is_indexed ? lhs_scalar_indexed_const : rhs_scalar_indexed_const;
  UnknownArray* candidate_broadcast_array =
      dynamic_cast<UnknownArray*>(lhs_is_indexed ? rhs : lhs);
  if (!candidate_broadcast_array ||
      candidate_broadcast_array->instruction().opcode() !=
          HloOpcode::kBroadcast) {
    return nullptr;
  }

  const HloInstruction* broadcast_instr =
      &candidate_broadcast_array->instruction();
  const HloInstruction* broadcast_const_operand = broadcast_instr->operand(0);
  if (broadcast_const_operand->opcode() != HloOpcode::kConstant) {
    return nullptr;
  }

  absl::Span<const int64_t> broadcast_dims = broadcast_instr->dimensions();
  auto is_broadcasted_dim = [&](int64_t output_dim) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_22(mht_22_v, 1116, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "lambda");

    return absl::c_find(broadcast_dims, output_dim) == broadcast_dims.end();
  };

  // All of the output dims must be "broadcasted" dims for the other operand.
  if (!absl::c_all_of(scalar_indexed_const->output_dims(),
                      is_broadcasted_dim)) {
    return nullptr;
  }

  // To figure out the broadcast dimensions for the (constant) source for the
  // scalar-indexed node, we "simulate" the index transformation done by the
  // existing broadcast:
  enum class IndexComponent { Broadcasted, NotBroadcasted };
  std::vector<IndexComponent> simulated_index(
      broadcast_instr->shape().dimensions_size(), IndexComponent::Broadcasted);
  for (int64_t broadcast_dim : broadcast_dims) {
    simulated_index[broadcast_dim] = IndexComponent::NotBroadcasted;
  }

  // The scalar-indexed node "removes" the source dim and "inserts" the output
  // dims.  We do the opposite here to undo the scalar-indexed operation.
  absl::Span<const int64_t> output_dims = scalar_indexed_const->output_dims();
  for (int64_t i = output_dims.size() - 1; i >= 0; --i) {
    CHECK(simulated_index[output_dims[i]] == IndexComponent::Broadcasted);
    EraseAt(&simulated_index, output_dims[i]);
  }

  InsertAt(&simulated_index, scalar_indexed_const->source_dim(),
           IndexComponent::Broadcasted);

  // new_inner_broadcast_dims holds the broadcast dimensions for the inner
  // BinaryOp(Broadcast'(Const0), Const1).  We now translate simulated_index to
  // new_inner_broadcast_dims.
  std::vector<int64_t> new_inner_broadcast_dims;
  for (int64_t i = 0; i < simulated_index.size(); i++) {
    if (simulated_index[i] == IndexComponent::NotBroadcasted) {
      new_inner_broadcast_dims.push_back(i);
    }
  }

  // inner_broadcast_result is the Broadcast'(Const0) bit in
  // BinaryOp(Broadcast'(Const0), Const1)
  TF_ASSIGN_OR_RETURN(
      Literal inner_broadcast_result,
      broadcast_const_operand->literal().Broadcast(
          scalar_indexed_const->source()->shape(), new_inner_broadcast_dims));

  // literal_for_new_source is BinaryOp(Broadcast'(Const0), Const1)
  const Literal* literal_for_new_source;
  if (lhs_is_indexed) {
    TF_ASSIGN_OR_RETURN(
        literal_for_new_source,
        TakeOwnership(HloEvaluator{}.EvaluateElementwiseBinaryOp(
            opcode, scalar_indexed_const->literal(), inner_broadcast_result)));
  } else {
    TF_ASSIGN_OR_RETURN(
        literal_for_new_source,
        TakeOwnership(HloEvaluator{}.EvaluateElementwiseBinaryOp(
            opcode, inner_broadcast_result, scalar_indexed_const->literal())));
  }

  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, scalar_indexed_const->indices(),
      scalar_indexed_const->source_dim(),
      std::vector<int64_t>(scalar_indexed_const->output_dims().begin(),
                           scalar_indexed_const->output_dims().end()),
      scalar_indexed_const->shape());
}

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForElementwiseUnaryOp(HloOpcode opcode,
                                                        Array* operand) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_23(mht_23_v, 1192, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForElementwiseUnaryOp");

  auto* scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(operand);
  if (scalar_indexed_const == nullptr) {
    return nullptr;
  }

  // Fold UnaryOp(ScalarIndexed(Const, Indices))
  //   => ScalarIndexed(UnaryOp(Const), Indices)

  TF_ASSIGN_OR_RETURN(Literal * literal_for_new_source,
                      TakeOwnership(HloEvaluator{}.EvaluateElementwiseUnaryOp(
                          opcode, scalar_indexed_const->literal())));
  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, scalar_indexed_const->indices(),
      scalar_indexed_const->source_dim(),
      SpanToVector(scalar_indexed_const->output_dims()),
      scalar_indexed_const->shape());
}

namespace {

// Returns the non-contracting non-batch dimension (as per `contracting_dims`
// and `batch_dims`) if there is exactly one, otherwise returns nullopt.
absl::optional<int64_t> GetOnlyNonContractingNonBatchDim(
    int64_t rank, absl::Span<const int64_t> contracting_dims,
    absl::Span<const int64_t> batch_dims) {
  absl::optional<int64_t> result;
  for (int64_t dim = 0; dim < rank; dim++) {
    if (!absl::c_linear_search(contracting_dims, dim) &&
        !absl::c_linear_search(batch_dims, dim)) {
      if (result.has_value()) {
        return absl::nullopt;
      }
      result = dim;
    }
  }
  return result;
}

// Returns true if `indexed_array`, which is either the LHS or the RHS of a Dot
// HLO, can be folded into the dot operation.  For now these conditions are both
// necessary and sufficient.
//
// `tag` describes the caller.  Used only for logging.
//
// `contracting_dims` and `batch_dims` are the contracting and batch dimensions
// of whatever operand `indexed_array` is to the dot (LHS or RHS).
bool CanFoldDotIntoIndexedArray(
    absl::string_view tag, Analysis::ScalarIndexedConstantArray* indexed_array,
    absl::Span<const int64_t> contracting_dims,
    absl::Span<const int64_t> batch_dims) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("tag: \"" + std::string(tag.data(), tag.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_24(mht_24_v, 1248, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "CanFoldDotIntoIndexedArray");

  absl::optional<int64_t> non_contracting_non_batch_dim =
      GetOnlyNonContractingNonBatchDim(indexed_array->shape().rank(),
                                       contracting_dims, batch_dims);
  if (!non_contracting_non_batch_dim.has_value()) {
    VLOG(3) << tag << ": multiple or no non-contracting non-batch dimensions";
    return false;
  }

  if (indexed_array->output_dims().size() != 1 ||
      indexed_array->output_dims()[0] != *non_contracting_non_batch_dim) {
    VLOG(3) << tag << ": output dims != the lhs non-contracting non-batch dim";
    return false;
  }

  int64_t indexed_array_rank = indexed_array->shape().rank();
  if (indexed_array->source_dim() < (indexed_array_rank - 2)) {
    // This restriction can be lifted by inserting reshape nodes.
    VLOG(3) << tag
            << ": source dim is not in the low two dims, won't be able to form "
               "a matmul";
    return false;
  }

  return true;
}

}  // namespace

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForDotWithIndexedLhs(
    const Shape& shape, const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config, ScalarIndexedConstantArray* lhs,
    ConstantArray* rhs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_25(mht_25_v, 1284, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForDotWithIndexedLhs");

  VLOG(3) << "ComputeArrayForDotWithIndexedLhs(" << ToString(lhs) << " "
          << ToString(rhs);
  if (!CanFoldDotIntoIndexedArray(
          "ComputeArrayForDotWithIndexedLhs", lhs, /*contracting_dims=*/
          dim_numbers.lhs_contracting_dimensions(),
          /*batch_dims=*/dim_numbers.lhs_batch_dimensions())) {
    return nullptr;
  }

  int64_t lhs_rank = lhs->shape().rank();
  DotDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.set_lhs_contracting_dimensions(
      0, lhs->source_dim() == (lhs_rank - 1) ? (lhs_rank - 2) : (lhs_rank - 1));

  TF_ASSIGN_OR_RETURN(
      Literal * literal_for_new_source,
      TakeOwnership(HloEvaluator{}.EvaluateDotOp(
          new_dim_numbers, precision_config, lhs->literal(), *rhs->literal())));

  // The new source dimension is wherever the non-batch non-contracting LHS
  // dimension "went".
  int64_t new_source_dim = dim_numbers.lhs_batch_dimensions_size() +
                           dim_numbers.rhs_batch_dimensions_size();

  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, lhs->indices(), new_source_dim,
      SpanToVector(lhs->output_dims()), shape);
}

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForDotWithIndexedRhs(
    const Shape& shape, const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config, ConstantArray* lhs,
    ScalarIndexedConstantArray* rhs) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_26(mht_26_v, 1322, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForDotWithIndexedRhs");

  VLOG(3) << "ComputeArrayForDotWithIndexedRhs(" << ToString(lhs) << " "
          << ToString(rhs);
  if (!CanFoldDotIntoIndexedArray(
          "ComputeArrayForDotWithIndexedRhs", rhs, /*contracting_dims=*/
          dim_numbers.rhs_contracting_dimensions(),
          /*batch_dims=*/dim_numbers.rhs_batch_dimensions())) {
    return nullptr;
  }

  int64_t rhs_rank = rhs->shape().rank();

  DotDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.set_rhs_contracting_dimensions(
      0, rhs->source_dim() == (rhs_rank - 1) ? (rhs_rank - 2) : (rhs_rank - 1));

  TF_ASSIGN_OR_RETURN(
      Literal * literal_for_new_source,
      TakeOwnership(HloEvaluator{}.EvaluateDotOp(
          new_dim_numbers, precision_config, *lhs->literal(), rhs->literal())));

  // The new source dimension is wherever the non-batch non-contracting RHS
  // dimension "went".
  int64_t new_source_dim = dim_numbers.lhs_batch_dimensions_size() +
                           dim_numbers.rhs_batch_dimensions_size() + 1;

  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, rhs->indices(), new_source_dim,
      SpanToVector(rhs->output_dims()), shape);
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForDot(
    const Shape& shape, const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config, Array* lhs, Array* rhs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_27(mht_27_v, 1359, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysis::ComputeArrayForDot");

  // Intuitively, if
  //
  //  - The LHS of a dot product is a gathered sequence of rows from a constant
  //    array (i.e. LHS[I,J] = Const[Indices[I],J]) and the RHS is a constant
  //
  //  OR
  //
  //  - If the RHS of a dot product is a gathered sequence of columns from a
  //    constant array (i.e. RHS[I,J] = Const[I, Indices[J]]) and the LHS is a
  //    constant
  //
  // then the result of the dot product itself is a gather from a constant
  // array.  E.g. Dot(LHS, ConstRhs) where LHS[I,J] = Const[Indices[I],J] can be
  // rewritten as Result where Result[I,J] = Dot(Const, ConstRhs)[Indices[I],
  // J].
  //
  // We do a general version of this rewrite here.
  VLOG(3) << "ComputeArrayForDot(" << ToString(lhs) << " " << ToString(rhs);
  if (auto* lhs_indexed_array =
          dynamic_cast<ScalarIndexedConstantArray*>(lhs)) {
    if (auto* rhs_constant = dynamic_cast<ConstantArray*>(rhs)) {
      return ComputeArrayForDotWithIndexedLhs(shape, dim_numbers,
                                              precision_config,
                                              lhs_indexed_array, rhs_constant);
    }
  }

  if (auto* rhs_indexed_array =
          dynamic_cast<ScalarIndexedConstantArray*>(rhs)) {
    if (auto* lhs_constant = dynamic_cast<ConstantArray*>(lhs)) {
      return ComputeArrayForDotWithIndexedRhs(shape, dim_numbers,
                                              precision_config, lhs_constant,
                                              rhs_indexed_array);
    }
  }

  return nullptr;
}

absl::string_view IndexedArrayAnalysisPrinterPass::name() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_28(mht_28_v, 1402, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysisPrinterPass::name");

  return "indexed-array-analysis-printer-pass";
}

StatusOr<bool> IndexedArrayAnalysisPrinterPass::Run(HloModule* module) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTcc mht_29(mht_29_v, 1409, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.cc", "IndexedArrayAnalysisPrinterPass::Run");

  if (!VLOG_IS_ON(2)) {
    return false;
  }

  IndexedArrayAnalysis analysis;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(Analysis::Array * t, analysis.GetArrayFor(instr));
      if (!dynamic_cast<UnknownArray*>(t) && !dynamic_cast<ConstantArray*>(t)) {
        VLOG(2) << instr->ToString() << "   ->   " << analysis.ToString(t);
      }
    }
  }

  return false;
}

}  // namespace xla
