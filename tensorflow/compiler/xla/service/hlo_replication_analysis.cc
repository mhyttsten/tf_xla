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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"

#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

// Determines whether an HLO instruction is replicated at index based on current
// knowledge in hlo_replication.
bool DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<bool>>&
        hlo_replication) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "DetermineHloInstructionIsReplicated");

  // Returns true if all operands are known to be replicated.
  const auto all_operands_replicated =
      [&hlo_replication](const HloInstruction* inst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "lambda");

        for (auto operand : inst->operands()) {
          auto operand_it = hlo_replication.find(operand);
          if (operand_it == hlo_replication.end() ||
              !operand_it->second.element({})) {
            return false;
          }
        }
        return true;
      };

  if (hlo->opcode() == HloOpcode::kAllReduce ||
      hlo->opcode() == HloOpcode::kAllGather) {
    // All-reduce/all-gather returns same values across partitions/replicas as
    // long as its operands are replicated.
    if (all_operands_replicated(hlo)) {
      return true;
    }
    if (!hlo->channel_id().has_value()) {
      // This is cross-replica-only.
      if (cross_partition_spmd) {
        return false;
      }
      // Only all-reduce/all-gather across all cores are replicated, which means
      // there is only one subgroup.
      return hlo->replica_groups().empty() || hlo->replica_groups().size() == 1;
    } else {
      bool global_id;
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        global_id = Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids();
      } else {
        global_id = Cast<HloAllGatherInstruction>(hlo)->use_global_device_ids();
      }
      if (global_id) {
        bool replicated_across_partitions = true;
        bool replicated_across_replicas = true;
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        for (const auto& group : hlo->replica_groups()) {
          absl::flat_hash_set<int64_t> visited_partitions;
          absl::flat_hash_set<int64_t> visited_replicas;
          for (int64_t id : group.replica_ids()) {
            int64_t rid = id / num_partitions;
            int64_t pid = id % num_partitions;
            visited_partitions.insert(pid);
            visited_replicas.insert(rid);
          }
          replicated_across_partitions &=
              visited_partitions.size() == num_partitions;
          replicated_across_replicas &=
              visited_replicas.size() ==
              hlo->GetModule()->config().replica_count();
        }
        return cross_partition_spmd ? replicated_across_partitions
                                    : replicated_across_replicas;
      }
      return cross_partition_spmd ? true
                                  : hlo->replica_groups().empty() ||
                                        hlo->replica_groups().size() == 1;
    }
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return false;
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    // ReplicaId returns the same value for all partitions in each replica.
    return cross_partition_spmd;
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    // PartitionId returns the same value for all replicas in each partition.
    return !cross_partition_spmd;
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    // Parameters should have been processed.
    return it != hlo_replication.end() && it->second.element(index);
  }
  if (it != hlo_replication.end() && !it->second.element(index)) {
    // The HLO is already marked as non-replicated.
    return false;
  }
  if (hlo->opcode() == HloOpcode::kConstant) {
    return true;
  }

  if (hlo->opcode() == HloOpcode::kCustomCall &&
      (hlo->custom_call_target() == "X64SplitLow" ||
       hlo->custom_call_target() == "X64SplitHigh" ||
       hlo->custom_call_target() == "X64Combine")) {
    return all_operands_replicated(hlo);
  }

  if (hlo->IsElementwise() ||                             //
      hlo->opcode() == HloOpcode::kConcatenate ||         //
      hlo->opcode() == HloOpcode::kConvolution ||         //
      hlo->opcode() == HloOpcode::kDot ||                 //
      hlo->opcode() == HloOpcode::kReduce ||              //
      hlo->opcode() == HloOpcode::kBroadcast ||           //
      hlo->opcode() == HloOpcode::kTranspose ||           //
      hlo->opcode() == HloOpcode::kReshape ||             //
      hlo->opcode() == HloOpcode::kBitcast ||             //
      hlo->opcode() == HloOpcode::kReverse ||             //
      hlo->opcode() == HloOpcode::kGather ||              //
      hlo->opcode() == HloOpcode::kScatter ||             //
      hlo->opcode() == HloOpcode::kIota ||                //
      hlo->opcode() == HloOpcode::kPad ||                 //
      hlo->opcode() == HloOpcode::kSlice ||               //
      hlo->opcode() == HloOpcode::kDynamicSlice ||        //
      hlo->opcode() == HloOpcode::kDynamicUpdateSlice ||  //
      hlo->opcode() == HloOpcode::kReduceWindow ||        //
      hlo->opcode() == HloOpcode::kCopy) {
    return all_operands_replicated(hlo);
  }
  return false;
}

}  // namespace

bool HloReplicationAnalysis::ComputeHloReplicationOnComputation(
    const HloComputation* computation, bool mark_everything_not_replicated) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_2(mht_2_v, 341, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "HloReplicationAnalysis::ComputeHloReplicationOnComputation");

  bool changed = false;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    // Assigns the shape tree to dest if dest doesn't have one yet, or combines
    // it with the existing one by and'ing them. Returns if anything is updated.
    auto assign_or_combine_shapetree = [&](ShapeTree<bool>&& to_combine,
                                           const HloInstruction* dest) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_3(mht_3_v, 350, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "lambda");

      auto it = hlo_replication_.find(dest);
      if (it == hlo_replication_.end()) {
        hlo_replication_[dest] = std::move(to_combine);
        return true;
      }
      bool updated = false;
      it->second.ForEachMutableElement(
          [&](const ShapeIndex& index, bool* element) {
            if (*element && !to_combine.element(index)) {
              *element = false;
              updated = true;
            }
          });
      return updated;
    };
    // Assigns or combines source's shape tree to dest. Returns if anything is
    // updated.
    auto propagate_shapetree = [&](const HloInstruction* source,
                                   const HloInstruction* dest) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_4(mht_4_v, 372, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "lambda");

      auto source_it = hlo_replication_.find(source);
      if (source_it == hlo_replication_.end()) {
        return false;
      }
      return assign_or_combine_shapetree(ShapeTree<bool>(source_it->second),
                                         dest);
    };
    // For the opcodes below that we do special handling, we don't need to
    // explicitly check mark_everything_not_replicated because if it is set, the
    // operands should already be marked as not replicated.
    if (inst->opcode() == HloOpcode::kWhile) {
      // Since while body's input and output alias each other, we need to run it
      // multiple times until a fixed point is reached.
      while (true) {
        // First, propagate the input's and body root's shape trees to the
        // parameters of the body and condition.
        bool updated = propagate_shapetree(
            inst->operand(0),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->while_body()->root_instruction(),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->operand(0), inst->while_body()->parameter_instruction(0));
        updated |=
            propagate_shapetree(inst->while_body()->root_instruction(),
                                inst->while_body()->parameter_instruction(0));
        // Compute the condition.
        updated |= ComputeHloReplicationOnComputation(
            inst->while_condition(), mark_everything_not_replicated);
        // Compute the body. If the condition is not replicated, the while body
        // should be different across replicas.
        if (!ContainsKey(loops_known_with_same_iterations_, inst) &&
            !hlo_replication_[inst->while_condition()->root_instruction()]
                 .element({})) {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), /*mark_everything_not_replicated=*/true);
        } else {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), mark_everything_not_replicated);
        }
        if (!updated) {
          break;
        }
        changed = true;
      }
      // Propagate the input's and body root's shape trees to the while HLO.
      changed |= propagate_shapetree(inst->operand(0), inst);
      changed |=
          propagate_shapetree(inst->while_body()->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kCall ||
               inst->opcode() == HloOpcode::kFusion) {
      auto called = inst->called_computations().front();
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        changed |= propagate_shapetree(inst->operand(i),
                                       called->parameter_instruction(i));
      }
      changed |= ComputeHloReplicationOnComputation(
          called, mark_everything_not_replicated);
      changed |= propagate_shapetree(called->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kConditional) {
      // Propagate inputs' shape trees to the called computations' parameters.
      for (int64_t i = 0; i < inst->called_computations().size(); ++i) {
        changed |= propagate_shapetree(
            inst->operand(i + 1),
            inst->called_computations()[i]->parameter_instruction(0));
      }
      // If the condition is not replicated, the conditional result should be
      // different across replicas.
      if (!hlo_replication_[inst->operand(0)].element({})) {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called,
              /*mark_everything_not_replicated=*/true);
        }
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called, mark_everything_not_replicated);
          changed |= propagate_shapetree(called->root_instruction(), inst);
        }
      }
    } else if (inst->opcode() == HloOpcode::kTupleSelect) {
      if (!hlo_replication_[inst->operand(0)].element({})) {
        // The predicate is not replicated, so the result is different across
        // replicas.
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        changed |= propagate_shapetree(inst->operand(1), inst);
        changed |= propagate_shapetree(inst->operand(2), inst);
      }
    } else if (inst->opcode() == HloOpcode::kTuple) {
      ShapeTree<bool> shape_tree(inst->shape(), true);
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(i)], {}, {i});
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      ShapeTree<bool> shape_tree(inst->shape(), true);
      shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(0)],
                                 {inst->tuple_index()}, {});
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kInfeed && cross_partition_spmd_) {
      ShapeTree<bool> shape_tree(inst->shape(), false);
      if (inst->has_sharding()) {
        auto sharding = inst->sharding().GetAsShapeTree(inst->shape());
        shape_tree.ForEachMutableElement(
            [&sharding](const ShapeIndex& index, bool* data) {
              *data = sharding.element(index).IsReplicated();
            });
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else {
      if (mark_everything_not_replicated) {
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        ShapeTree<bool> shape_tree(inst->shape(), true);
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              *shape_tree.mutable_element(index) =
                  DetermineHloInstructionIsReplicated(
                      inst, index, cross_partition_spmd_, hlo_replication_);
              return Status::OK();
            });
        changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
      }
    }
  }
  return changed;
}

void HloReplicationAnalysis::ComputeHloReplication() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_5(mht_5_v, 511, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "HloReplicationAnalysis::ComputeHloReplication");

  // Add entry parameters to the above sets according to user annotation.
  // Replicated modules read from `parameter_replicated_at_leaf_buffers` whereas
  // SPMD partitioned modules read from HloSharding attributes.
  auto entry = module_->entry_computation();
  for (int i = 0; i < entry->num_parameters(); ++i) {
    auto param = entry->parameter_instruction(i);
    ShapeTree<bool> shape_tree(param->shape(), false);
    if (cross_partition_spmd_ && param->has_sharding()) {
      auto sharding_tree =
          param->sharding().AsShapeTree(param->shape()).ValueOrDie();
      ShapeUtil::ForEachSubshape(
          param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
            if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
              return Status::OK();
            }
            *shape_tree.mutable_element(index) =
                sharding_tree.element(index).IsReplicated();
            return Status::OK();
          });
    } else if (!cross_partition_spmd_) {
      const auto& replication = param->parameter_replicated_at_leaf_buffers();
      int leaf_index = 0;
      ShapeUtil::ForEachSubshape(
          param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
            if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
              return Status::OK();
            }
            if (replication && replication->at(leaf_index)) {
              *shape_tree.mutable_element(index) = true;
            }
            ++leaf_index;
            return Status::OK();
          });
    }
    hlo_replication_[param] = std::move(shape_tree);
  }
  ComputeHloReplicationOnComputation(entry,
                                     /*mark_everything_not_replicated=*/false);
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_6(mht_6_v, 556, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "HloReplicationAnalysis::HloInstructionIsReplicatedAt");

  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  return it->second.element(index);
}

/* static */ StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module,
                            bool cross_partition_spmd) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_7(mht_7_v, 569, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "HloReplicationAnalysis::Run");

  const absl::flat_hash_set<const HloInstruction*> empty;
  return Run(module, cross_partition_spmd, &empty);
}

/* static */ StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module, bool cross_partition_spmd,
                            const absl::flat_hash_set<const HloInstruction*>*
                                loops_known_with_same_iterations) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_replication_analysisDTcc mht_8(mht_8_v, 580, "", "./tensorflow/compiler/xla/service/hlo_replication_analysis.cc", "HloReplicationAnalysis::Run");

  auto analysis = absl::WrapUnique(new HloReplicationAnalysis(
      module, cross_partition_spmd, loops_known_with_same_iterations));
  analysis->ComputeHloReplication();
  return analysis;
}

}  // namespace xla
