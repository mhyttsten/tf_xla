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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_merger.h"

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace {

// Determines if this dot is canonical according to DotDecomposer's rules.  The
// LHS and RHS must have
//
//  - batch dimensions at the beginning, followed by
//  - one non-contracting dimension and one contracting dimension, in either
//    order.
//
// (Note: DotDecomposer doesn't guarantee that the LHS contracting dim is the
// last dim or the RHS contracting dim is the second-to-last.)
bool IsCanonicalDot(HloInstruction* dot) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/dot_merger.cc", "IsCanonicalDot");

  if (dot->opcode() != HloOpcode::kDot) {
    return false;
  }

  // Checks that the given list is a permutation of [0, 1, ..., n].
  auto is_permutation_of_iota =
      [](const tensorflow::protobuf::RepeatedField<int64_t>& vals) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/dot_merger.cc", "lambda");

        absl::InlinedVector<int64_t, 8> copy(vals.begin(), vals.end());
        absl::c_sort(copy);
        for (int i = 0; i < copy.size(); i++) {
          if (copy[i] != i) {
            return false;
          }
        }
        return true;
      };

  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  int64_t batch_size = dnums.lhs_batch_dimensions_size();
  return is_permutation_of_iota(dnums.lhs_batch_dimensions()) &&
         dnums.lhs_contracting_dimensions_size() == 1 &&
         dot->operand(0)->shape().dimensions_size() == batch_size + 2 &&
         is_permutation_of_iota(dnums.rhs_batch_dimensions()) &&
         dnums.rhs_contracting_dimensions_size() == 1 &&
         dot->operand(1)->shape().dimensions_size() == batch_size + 2;
}

// Tries to merge dot instructions a and b if they share an operand.  Example:
//
//   lhs = f32[200,100] parameter(0)
//   rhs0 = f32[100,10] parameter(1)
//   rhs1 = f32[100,50] parameter(2)
//   dot0 = f32[200,10] dot(lhs, rhs0), lhs_contracting_dims={1},
//   rhs_contracting_dims={0} dot1 = f32[200,50] dot(lhs, rhs1),
//   lhs_contracting_dims={1}, rhs_contracting_dims={0}
//
// can be merged to
//
//   dot = f32[200,60] dot(lhs, concat(rhs0, lhs1))
//   dot0 = slice(dot)
//   dot1 = slice(dot)
//
// Preconditions:
//  - `a` and `b` are canonical dots.
//  - `a` does not transitively depend on the value of `b`, and `b` does not
//    transitively depend on the value of `a`.
//
StatusOr<HloInstruction*> TryMergeSameOperand(HloInstruction* a,
                                              HloInstruction* b) {
  if (a->operand(0) != b->operand(0) && a->operand(1) != b->operand(1)) {
    VLOG(4) << "Can't merge dots because they don't share an operand.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  if (a->operand(0)->shape().element_type() !=
          b->operand(0)->shape().element_type() ||
      a->operand(1)->shape().element_type() !=
          b->operand(1)->shape().element_type() ||
      a->shape().element_type() != b->shape().element_type()) {
    VLOG(3)
        << "Can't merge dots because their lhs/rhs/return-types don't match.\n"
        << "\t" << a->ToString() << "\n"
        << "\t" << b->ToString();
    return nullptr;
  }

  const DotDimensionNumbers& dnums_a = a->dot_dimension_numbers();
  const DotDimensionNumbers& dnums_b = b->dot_dimension_numbers();
  if (!absl::c_equal(dnums_a.lhs_batch_dimensions(),
                     dnums_b.lhs_batch_dimensions()) ||
      !absl::c_equal(dnums_a.rhs_batch_dimensions(),
                     dnums_b.rhs_batch_dimensions()) ||
      !absl::c_equal(dnums_a.lhs_contracting_dimensions(),
                     dnums_b.lhs_contracting_dimensions()) ||
      !absl::c_equal(dnums_a.rhs_contracting_dimensions(),
                     dnums_b.rhs_contracting_dimensions())) {
    VLOG(3) << "Can't merge dots because they have mismatching dnums.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString() << "\n"
            << absl::c_equal(dnums_a.lhs_batch_dimensions(),
                             dnums_b.lhs_batch_dimensions())
            << ", "
            << absl::c_equal(dnums_a.rhs_batch_dimensions(),
                             dnums_b.rhs_batch_dimensions())
            << ", "
            << absl::c_equal(dnums_a.lhs_contracting_dimensions(),
                             dnums_b.lhs_contracting_dimensions())
            << ", "
            << absl::c_equal(dnums_a.rhs_contracting_dimensions(),
                             dnums_b.rhs_contracting_dimensions());
    return nullptr;
  }

  if (!absl::c_equal(a->precision_config().operand_precision(),
                     b->precision_config().operand_precision())) {
    VLOG(3) << "Can't merge dots because they have mismatching operand "
               "precisions:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  VLOG(2) << "Merging dots sharing an operand:\n"
          << "\t" << a->ToString() << "\n"
          << "\t" << b->ToString();

  // At this point we have dnums_a == dnums_b.  Rename to just `dnums`.
  const DotDimensionNumbers& dnums = a->dot_dimension_numbers();

  // At this point, either the LHS'es are the same, or the RHS'es are the same.
  bool lhs_same = a->operand(0) == b->operand(0);
  HloInstruction* shared_op = a->mutable_operand(lhs_same ? 0 : 1);
  HloInstruction* diff_op_a = a->mutable_operand(lhs_same ? 1 : 0);
  HloInstruction* diff_op_b = b->mutable_operand(lhs_same ? 1 : 0);

  // Dimension along which we're going to concatenate diff_op_a and diff_op_b.
  // This is the outer (i.e. non-contracing) dim.  Because the dot is canonical,
  // we know that the dimensions are
  //
  //  [batch_dims ..., outer/contracting dim, contracting/outer dim].
  //
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  int64_t contracting_dim = (lhs_same ? dnums.rhs_contracting_dimensions()
                                      : dnums.lhs_contracting_dimensions())[0];
  int64_t outer_dim = contracting_dim == dnums.lhs_batch_dimensions_size()
                          ? contracting_dim + 1
                          : contracting_dim - 1;

  HloComputation* comp = a->parent();
  TF_ASSIGN_OR_RETURN(
      Shape concat_shape,
      ShapeInference::InferConcatOpShape(
          {&diff_op_a->shape(), &diff_op_b->shape()}, outer_dim));
  HloInstruction* concat_op =
      comp->AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {diff_op_a, diff_op_b}, outer_dim));

  HloInstruction* dot_lhs = lhs_same ? shared_op : concat_op;
  HloInstruction* dot_rhs = lhs_same ? concat_op : shared_op;
  TF_ASSIGN_OR_RETURN(
      Shape new_dot_shape,
      ShapeInference::InferDotOpShape(
          dot_lhs->shape(), dot_rhs->shape(), dnums,
          /*preferred_element_type=*/a->shape().element_type()));
  HloInstruction* new_dot = comp->AddInstruction(HloInstruction::CreateDot(
      new_dot_shape, dot_lhs, dot_rhs, dnums, a->precision_config()));

  // Slice the outputs.
  absl::InlinedVector<int64_t, 6> start_indices(new_dot_shape.dimensions_size(),
                                                0);
  absl::InlinedVector<int64_t, 6> limit_indices(
      new_dot_shape.dimensions().begin(), new_dot_shape.dimensions().end());
  absl::InlinedVector<int64_t, 6> strides(new_dot_shape.dimensions_size(), 1);

  int64_t slice_dim = new_dot_shape.dimensions_size() - (lhs_same ? 1 : 2);
  limit_indices[slice_dim] = a->shape().dimensions(slice_dim);
  // Important: We do RAUW, not ReplaceInstruction, because the old instruction
  // must live until the end of the pass.
  HloInstruction* new_a = comp->AddInstruction(HloInstruction::CreateSlice(
      a->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));
  new_a->set_metadata(a->metadata());

  start_indices[slice_dim] = limit_indices[slice_dim];
  limit_indices[slice_dim] = new_dot_shape.dimensions(slice_dim);
  HloInstruction* new_b = comp->AddInstruction(HloInstruction::CreateSlice(
      b->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));
  new_b->set_metadata(b->metadata());

  return new_dot;
}

StatusOr<bool> MergeDots(HloComputation* comp, int64_t max_size_to_merge) {
  auto is_merge_candidate = [&](HloInstruction* instr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc mht_2(mht_2_v, 392, "", "./tensorflow/compiler/xla/service/dot_merger.cc", "lambda");

    int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
    for (const HloInstruction* operand : instr->operands()) {
      bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
    }
    return bytes <= max_size_to_merge;
  };

  // Collect equivalence classes.  Specifically, create the map
  //
  //   instruction -> [canonical dots that use the instruction].
  //
  // We'll then try to merge dots within each equivalence class.  A dot will be
  // a member of two equivalence classes (because it has two operands), but if
  // it's merged with a dot from one equivalence class, it won't also be merged
  // in another class.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
      equivalence_classes;
  for (HloInstruction* instr : comp->instructions()) {
    // Cowardly skip instructions with control dependencies.
    if (!IsCanonicalDot(instr) || !instr->control_predecessors().empty() ||
        !instr->control_successors().empty()) {
      continue;
    }
    for (HloInstruction* operand : instr->operands()) {
      equivalence_classes[operand].insert(instr);
    }
  }

  // Remove "uninteresting" equivalence classes where either
  //
  //  - there's just one instruction (nothing to merge!), or
  //  - there are zero instructions marked as mergeable.  (Our contract is that
  //    at least one instruction of the pair needs to be mergeable in order for
  //    us to merge.)
  absl::erase_if(
      equivalence_classes,
      [&](const std::pair<const HloInstruction*,
                          absl::flat_hash_set<HloInstruction*>>& kv) {
        const auto& v = kv.second;
        return v.size() < 2 || absl::c_none_of(v, is_merge_candidate);
      });

  // Are there any possible optimization opportunities?
  if (equivalence_classes.empty()) {
    return false;
  }

  // Build a dependency graph representing the whole computation.
  //
  // TODO(jlebar): If this is slow to create or use, could we make it faster by
  // collapsing elements of the graph that don't correspond to dots, or
  // otherwise not adding them to the graph in the first place?
  tensorflow::GraphCycles graph;

  absl::flat_hash_map<HloInstruction*, int32_t> graph_ids_map;
  auto graph_id = [&](HloInstruction* instr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc mht_3(mht_3_v, 451, "", "./tensorflow/compiler/xla/service/dot_merger.cc", "lambda");

    auto it_and_inserted = graph_ids_map.emplace(instr, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      it->second = graph.NewNode();
    }
    return it->second;
  };

  for (HloInstruction* instr : comp->instructions()) {
    int32_t id = graph_id(instr);
    for (HloInstruction* operand : instr->operands()) {
      CHECK(graph.InsertEdge(graph_id(operand), id));
    }
    for (HloInstruction* control_pred : instr->control_predecessors()) {
      CHECK(graph.InsertEdge(graph_id(control_pred), id));
    }
  }

  // Merge within equivalence classes.  We keep a set of all instructions that
  // have been merged so we don't try to merge an instruction twice.  We'll
  // remove these dead instructions at the end of the pass.  (We can't remove
  // them earlier because removing an instruction deletes it; we'd then have
  // dangling pointers in our hashtable!)
  absl::flat_hash_set<HloInstruction*> dead_instrs;
  for (auto& kv : equivalence_classes) {
    // For determinism, iterate in order of the instructions' IDs.
    absl::InlinedVector<HloInstruction*, 16> dots(kv.second.begin(),
                                                  kv.second.end());
    absl::c_sort(dots, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });

    // Try merging all pairs of dots in this equivalence class.
    for (int64_t i = 0; i < dots.size(); i++) {
      HloInstruction*& a = dots[i];
      if (a == nullptr) {
        continue;
      }
      for (int64_t j = i + 1; j < dots.size(); j++) {
        HloInstruction* b = dots[j];
        if (b == nullptr) {
          continue;
        }
        int32_t a_id = graph_id(a);
        int32_t b_id = graph_id(b);

        if (dead_instrs.contains(a) || dead_instrs.contains(b) ||
            graph.IsReachableNonConst(a_id, b_id) ||
            graph.IsReachableNonConst(b_id, a_id) ||
            (!is_merge_candidate(a) && !is_merge_candidate(b))) {
          continue;
        }

        TF_ASSIGN_OR_RETURN(HloInstruction * merged, TryMergeSameOperand(a, b));
        if (merged != nullptr) {
          int32_t merged_id = graph_id(merged);
          graph.InsertEdge(a_id, merged_id);
          graph.InsertEdge(b_id, merged_id);
          for (int32_t succ : graph.SuccessorsCopy(a_id)) {
            graph.InsertEdge(merged_id, succ);
          }
          for (int32_t succ : graph.SuccessorsCopy(b_id)) {
            graph.InsertEdge(merged_id, succ);
          }

          dead_instrs.insert(a);
          dead_instrs.insert(b);
          dots[i] = merged;
          dots[j] = nullptr;
        }
      }
    }
  }

  // Now it's finally safe to delete the old instructions from the graph.
  for (HloInstruction* instr : dead_instrs) {
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(instr));
  }

  return !dead_instrs.empty();
}

}  // anonymous namespace

StatusOr<bool> DotMerger::Run(HloModule* module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdot_mergerDTcc mht_4(mht_4_v, 540, "", "./tensorflow/compiler/xla/service/dot_merger.cc", "DotMerger::Run");

  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MergeDots(comp, max_size_to_merge_));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
