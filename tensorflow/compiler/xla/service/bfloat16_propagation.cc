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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc() {
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

#include "tensorflow/compiler/xla/service/bfloat16_propagation.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

BFloat16Propagation::BFloat16Propagation(
    const BFloat16Support* bfloat16_support)
    : bfloat16_support_(bfloat16_support) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::BFloat16Propagation");
}

void BFloat16Propagation::DetermineFusionComputationPrecision(
    HloInstruction* fusion) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::DetermineFusionComputationPrecision");

  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!bfloat16_support_->SupportsMixedPrecisions(*fusion)) {
    return;
  }

  // We are depending on the fusion node itself having already been analyzed
  // for whether it can output BF16 and this has been adjusted in the output
  // shape, and now we're looking to update the interior of the fusion node to
  // match the new output shape, as well as recursively process the whole fusion
  // node even if the output shape was not modified.
  auto root = fusion->fused_instructions_computation()->root_instruction();

  // Adjust root's element types according to the fusion's output shape.
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(fusion, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
          VLOG(2) << "Fused root " << root->ToString() << " at shape index "
                  << index << " changed to BF16 precision for fusion "
                  << fusion->ToString();
        }
      });

  // Propagate BF16 in the fusion computation.
  auto insts =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(
      fusion->fused_instructions_computation());

  RevertIfFusionInternalBF16Changes(fusion);
}

void BFloat16Propagation::RevertIfFusionInternalBF16Changes(
    HloInstruction* fusion) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::RevertIfFusionInternalBF16Changes");

  auto has_changes = [this](HloInstruction* inst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

    auto it = changes_to_bf16_.find(inst);
    return it != changes_to_bf16_.end() && !it->second.empty();
  };

  auto root = fusion->fused_instructions_computation()->root_instruction();
  absl::flat_hash_set<const HloValue*> changed_root_buffers;

  auto root_changes_it = changes_to_bf16_.find(root);
  if (root_changes_it != changes_to_bf16_.end()) {
    for (const auto& entry : root_changes_it->second) {
      for (const HloValue* value :
           dataflow_->GetValueSet(root, entry.second).values()) {
        changed_root_buffers.insert(value);
      }
    }
  }

  auto aliases_changed_root_buffer =
      [this, &changed_root_buffers](const HloInstruction* inst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

        bool aliasing = false;
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              if (aliasing) {
                // Skip if aliasing is already found.
                return;
              }
              // Only F32 buffers are considered for changing to BF16 in this
              // pass.
              if (subshape.element_type() != F32) {
                return;
              }

              aliasing =
                  absl::c_any_of(dataflow_->GetValueSet(inst, index).values(),
                                 IsValueIn(changed_root_buffers));
            });
        return aliasing;
      };

  for (auto inst :
       fusion->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (aliases_changed_root_buffer(inst)) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kFusion) {
      bool parameter_reverted = false;
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        if (has_changes(inst->mutable_operand(i))) {
          // Changes on the operand have not been reverted.
          continue;
        }
        auto* fused_parameter = inst->fused_parameter(i);
        if (has_changes(fused_parameter)) {
          changes_to_bf16_.erase(fused_parameter);
          parameter_reverted = true;
        }
      }
      if (parameter_reverted) {
        RevertIfFusionInternalBF16Changes(inst);
      }
    }
    if (!has_changes(inst)) {
      continue;
    }
    bool revert_changes = true;
    for (auto operand : inst->operands()) {
      if (has_changes(operand)) {
        revert_changes = false;
        break;
      }
    }
    if (revert_changes) {
      changes_to_bf16_.erase(inst);
    }
  }
}

void BFloat16Propagation::DetermineWhileComputationsPrecision(
    HloInstruction* while_hlo) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_5(mht_5_v, 347, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::DetermineWhileComputationsPrecision");

  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);

  // We are depending on the while node itself having already been analyzed for
  // whether it can output BF16 and this has been adjusted in the output shape,
  // and now we're looking to update the body and condition computations to
  // match the new output shape, as well as recursively process the whole while
  // node even if the output shape was not modified.
  HloComputation* body = while_hlo->while_body();
  auto body_root = body->root_instruction();
  HloComputation* condition = while_hlo->while_condition();

  ShapeUtil::ForEachSubshape(
      body_root->shape(), [this, while_hlo, body_root](
                              const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(while_hlo, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(body_root, index, BF16);
          VLOG(2) << "While body root " << body_root->ToString()
                  << " at shape index " << index
                  << " changed to BF16 precision for while "
                  << while_hlo->ToString();
        }
      });

  auto body_insts = body->MakeInstructionPostOrder();
  for (auto inst_it = body_insts.rbegin(); inst_it != body_insts.rend();
       ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(body);

  auto condition_insts = condition->MakeInstructionPostOrder();
  for (auto inst_it = condition_insts.rbegin();
       inst_it != condition_insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(condition);
}

void BFloat16Propagation::DetermineConditionalComputationsPrecision(
    HloInstruction* cond) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_6(mht_6_v, 393, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::DetermineConditionalComputationsPrecision");

  CHECK_EQ(cond->opcode(), HloOpcode::kConditional);
  for (int64_t i = 0; i < cond->branch_count(); ++i) {
    auto branch = cond->branch_computation(i);
    auto root = branch->root_instruction();
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.element_type() != F32) {
            return;
          }
          if (OutputTypeAfterChange(cond, index) == BF16) {
            AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
            VLOG(2) << "Conditional branch " << i << " root "
                    << root->ToString() << " at shape index " << index
                    << " changed to BF16 precision for conditional "
                    << cond->ToString();
          }
        });
    auto insts = branch->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
    }
    computations_visited_in_backward_pass_.insert(branch);
  }
}

bool BFloat16Propagation::AllUsersConsumeBF16(const HloInstruction& hlo,
                                              const ShapeIndex& index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_7(mht_7_v, 423, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::AllUsersConsumeBF16");

  // If the subshape isn't floating point then none of the users will be BF16.
  const Shape& subshape = ShapeUtil::GetSubshape(hlo.shape(), index);
  if (subshape.element_type() != BF16 && subshape.element_type() != F32) {
    return false;
  }

  auto& value_set = dataflow_->GetValueSet(&hlo, index);
  for (const HloValue* value : value_set.values()) {
    if (ContainsKey(values_that_must_be_kept_as_f32_, value)) {
      return false;
    }
    // We use the original type for the value because we are going to examine
    // the uses of it, instead of the value itself. If ValueTypeAfterChange()
    // were used, it would cause problems when there are aliasing buffers, i.e.,
    // ResolveInconsistencyOfAliasingBuffers() would fail to revert the
    // tentative change to BF16 even if the uses require F32.
    if (value->shape().element_type() == BF16) {
      continue;
    }
    for (const HloUse& use : value->GetUses()) {
      if (!ContainsKey(instructions_visited_in_backward_pass_,
                       use.instruction)) {
        // We don't know yet whether use.instruction will consume BF16 since it
        // hasn't been visited. Although we visit instructions in reverse
        // topological order, this is still possible because there may be
        // unvisited instruction that alias the same buffer. In this case, we
        // aggressively skip this use, and if this causes inconsistency (e.g.,
        // one use is in BF16 but another use is in F32), it will be resolved at
        // the end of the BFloat16Propagation pass.
        continue;
      }
      if (use.instruction->HasSideEffectNoRecurse()) {
        // Keep side-effecting instruction's operands unchanged.
        return false;
      }
      // Any visited user that can accept BF16 has already been updated if
      // necessary, e.g., the output has been changed to BF16 if it propagates
      // precision, or a called computation's parameters have been changed to
      // BF16 for fusions or whiles.
      if (use.instruction->opcode() == HloOpcode::kFusion) {
        auto* fused_parameter =
            use.instruction->fused_parameter(use.operand_number);
        if (OutputTypeAfterChange(fused_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kWhile) {
        auto* cond_parameter =
            use.instruction->while_condition()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        auto* body_parameter =
            use.instruction->while_body()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(body_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kConditional) {
        auto* cond_parameter =
            use.instruction->branch_computation(use.operand_number - 1)
                ->parameter_instruction(0);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsBF16(
              *use.instruction, use.operand_number)) {
        continue;
      }
      // If the op propagates precision and it outputs a BF16, then it's OK to
      // supply BF16 also as the input. In the backward pass, the users shapes
      // should have already been processed.
      if (bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(
              *use.instruction, use.operand_number)) {
        if (use.instruction->opcode() == HloOpcode::kTuple ||
            (use.instruction->opcode() == HloOpcode::kAllReduce &&
             use.instruction->shape().IsTuple())) {
          ShapeIndex use_output_index{use.operand_number};
          for (int64_t i : use.operand_index) {
            use_output_index.push_back(i);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
          ShapeIndex use_output_index;
          for (int64_t i = 1; i < use.operand_index.size(); ++i) {
            use_output_index.push_back(use.operand_index[i]);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else {
          if (OutputTypeAfterChange(use.instruction, use.operand_index) ==
              BF16) {
            continue;
          }
        }
      }
      return false;
    }
  }
  return true;
}

bool BFloat16Propagation::ShouldKeepPrecisionUnchanged(
    const HloInstruction* inst) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_8(mht_8_v, 539, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ShouldKeepPrecisionUnchanged");

  if (inst->opcode() == HloOpcode::kFusion &&
      inst->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return ShouldKeepPrecisionUnchanged(
        inst->fused_instructions_computation()->root_instruction());
  }
  // Do not change precision for side-effecting instructions, control flow, and
  // bitcast-convert, because this pass might break the interfaces or
  // assumptions for them.
  return inst->opcode() == HloOpcode::kCustomCall ||
         inst->opcode() == HloOpcode::kCall ||
         inst->opcode() == HloOpcode::kBitcastConvert ||
         inst->HasSideEffectNoRecurse();
}

void BFloat16Propagation::DetermineInstructionPrecision(HloInstruction* hlo,
                                                        bool skip_parameters) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_9(mht_9_v, 558, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::DetermineInstructionPrecision");

  // We handle any fusion computation, while body/condition or conditional
  // branches after the instruction is handled, because we need to know the
  // output shape of a fusion or while before propagating inside its
  // computations.
  bool postpone_processing_called_computations = false;
  auto cleaner = tensorflow::gtl::MakeCleanup(
      [this, hlo, &postpone_processing_called_computations] {
        if (!postpone_processing_called_computations) {
          if (hlo->opcode() == HloOpcode::kFusion) {
            DetermineFusionComputationPrecision(hlo);
          } else if (hlo->opcode() == HloOpcode::kWhile) {
            DetermineWhileComputationsPrecision(hlo);
          } else if (hlo->opcode() == HloOpcode::kConditional) {
            DetermineConditionalComputationsPrecision(hlo);
          }
        }
        instructions_visited_in_backward_pass_.insert(hlo);
      });

  if (hlo->opcode() == HloOpcode::kWhile &&
      (caller_counts_[hlo->while_condition()] > 1 ||
       caller_counts_[hlo->while_body()] > 1)) {
    postpone_processing_called_computations = true;
    return;
  }

  if (hlo->opcode() == HloOpcode::kConditional &&
      absl::c_any_of(hlo->branch_computations(), [&](const HloComputation* c) {
        return caller_counts_[c] > 1;
      })) {
    postpone_processing_called_computations = true;
    return;
  }

  // Prevent root instructions from having their output modified by recording
  // all F32 output values as needing to stay as F32.
  CHECK(hlo->parent() != nullptr);
  if (hlo == hlo->parent()->root_instruction()) {
    if (!hlo->parent()->IsFusionComputation()) {
      ShapeUtil::ForEachSubshape(hlo->shape(), [&](const Shape& /* subshape */,
                                                   const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) != F32) {
          return;
        }
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          // Since we use HloValues from the dataflow analysis, this can also
          // affect HLO instructions beyond the root, e.g., if the root is a
          // Tuple HLO, then its operands are also affected.
          values_that_must_be_kept_as_f32_.insert(value);
        }
      });
    }
    return;
  }

  if (ShouldKeepPrecisionUnchanged(hlo) ||
      (hlo->opcode() == HloOpcode::kParameter && skip_parameters)) {
    return;
  }

  if (!ContainsKey(consider_using_bfloat16_, hlo)) {
    return;
  }

  if (!bfloat16_support_->SupportsBF16Output(*hlo)) {
    return;
  }

  ShapeUtil::ForEachSubshape(
      hlo->shape(),
      [hlo, this](const Shape& /* subshape */, const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) == F32 &&
            AllUsersConsumeBF16(*hlo, index)) {
          AddToOrRemoveFromBF16ChangeSet(hlo, index, BF16);
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " changed to BF16 precision: " << hlo->ToString();
        }
      });
}

bool BFloat16Propagation::InstructionIsCandidateForBF16Output(
    HloInstruction* hlo) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_10(mht_10_v, 643, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::InstructionIsCandidateForBF16Output");

  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo) &&
      hlo->opcode() != HloOpcode::kTuple &&
      hlo->opcode() != HloOpcode::kGetTupleElement &&
      hlo->opcode() != HloOpcode::kDomain &&
      hlo->shape().element_type() != BF16) {
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      if (!bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                         i) ||
          !ContainsKey(consider_using_bfloat16_, hlo->operand(i))) {
        return false;
      }
    }
  }
  return true;
}

void BFloat16Propagation::AdjustCalledComputationParameters(
    HloInstruction* hlo) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_11(mht_11_v, 664, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::AdjustCalledComputationParameters");

  auto adjust_computation =
      [this, hlo](HloComputation* computation,
                  absl::Span<HloInstruction* const> operands) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_12(mht_12_v, 670, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

        // Adjust parameters.
        CHECK_EQ(operands.size(), computation->num_parameters());
        for (int64_t i = 0; i < operands.size(); ++i) {
          auto parameter = computation->parameter_instruction(i);
          ShapeUtil::ForEachSubshape(
              parameter->shape(),
              [this, i, hlo, &operands, parameter](const Shape& /* subshape */,
                                                   const ShapeIndex& index) {
                if (!ShapeUtil::IsLeafIndex(parameter->shape(), index)) {
                  return;
                }
                PrimitiveType operand_type =
                    OutputTypeAfterChange(operands[i], index);
                if (OutputTypeAfterChange(parameter, index) == operand_type) {
                  return;
                }
                AddToOrRemoveFromBF16ChangeSet(parameter, index, operand_type);
                VLOG(2) << "Called computation parameter "
                        << parameter->ToString() << " at shape index " << index
                        << " adjusted to "
                        << (operand_type == BF16 ? "BF16" : "F32")
                        << " to match operand in HLO " << hlo->ToString();
              });
        }
      };

  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(),
                         hlo->operands());
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_condition(), hlo->operands());
      adjust_computation(hlo->while_body(), hlo->operands());
      break;
    case HloOpcode::kConditional:
      for (int64_t i = 0; i < hlo->branch_count(); ++i) {
        adjust_computation(hlo->branch_computation(i),
                           {hlo->mutable_operand(i + 1)});
      }
      break;
    default:
      break;
  }
}

void BFloat16Propagation::AdjustCalledComputationRoot(HloInstruction* hlo) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_13(mht_13_v, 720, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::AdjustCalledComputationRoot");

  auto adjust_computation = [this, hlo](HloComputation* computation,
                                        HloInstruction* output) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_14(mht_14_v, 725, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

    // Adjust root.
    HloInstruction* root = computation->root_instruction();
    ShapeUtil::ForEachSubshape(root->shape(), [this, hlo, root, output](
                                                  const Shape& /* subshape */,
                                                  const ShapeIndex& index) {
      if (!ShapeUtil::IsLeafIndex(hlo->shape(), index)) {
        return;
      }
      const PrimitiveType output_type = OutputTypeAfterChange(output, index);
      if (OutputTypeAfterChange(root, index) == output_type) {
        return;
      }
      AddToOrRemoveFromBF16ChangeSet(root, index, output_type);
      // It's possible that output_type is F32, but the root instruction's
      // type is BF16; e.g., a fusion node's output was changed to BF16
      // initially but then adjusted back to F32, and the fusion computation
      // is now being adjusted after the fusion node.
      if (output_type == F32) {
        for (const auto* value : dataflow_->GetValueSet(root, index).values()) {
          // We rely on the fact that this adjustment works in reverse
          // topological order so that called computation will be
          // processed later. Adding the value to
          // values_that_must_be_kept_as_f32_ will ensure the
          // correctness of the adjustment for HLOs that will be
          // processed later.
          values_that_must_be_kept_as_f32_.insert(value);
        }
      }
      VLOG(2) << "Called computation root " << root->ToString()
              << " at shape index " << index << " adjusted to "
              << (output_type == BF16 ? "BF16" : "F32")
              << " to match output shape of " << hlo->ToString();
    });
  };

  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(), hlo);
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_body(), hlo);
      break;
    case HloOpcode::kConditional:
      for (auto* branch : hlo->branch_computations()) {
        adjust_computation(branch, hlo);
      }
      break;
    default:
      break;
  }
}

bool BFloat16Propagation::ResolveInconsistencyOfAliasingBuffersHelper(
    HloComputation* computation,
    absl::flat_hash_set<const HloComputation*>* visited_computations) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_15(mht_15_v, 783, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ResolveInconsistencyOfAliasingBuffersHelper");

  bool parameter_changed = false;
  auto insts = computation->MakeInstructionPostOrder();
  // Do the adjustment on each instruction in the computation in reverse
  // topological order.
  while (true) {
    bool any_change = false;
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      auto adjust_hlo_output = [&](const Shape& /* subshape */,
                                   const ShapeIndex& index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_16(mht_16_v, 796, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

        auto output_type = OutputTypeAfterChange(hlo, index);
        VLOG(2) << "output_type is " << ((output_type == BF16) ? "BF16" : "F32")
                << " for :" << hlo->ToString() << "\n";
        if (output_type != F32 && output_type != BF16) {
          return;
        }
        PrimitiveType type = BF16;
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          auto value_type = ValueTypeAfterChange(value);
          if (value_type == BF16) {
            continue;
          }
          VLOG(2) << "Adjust to F32 due to aliased dataflow value: "
                  << value->ToString() << "\n";
          CHECK_EQ(value_type, F32);
          type = F32;
          break;
        }
        // In order to find aliases due to in-place operations, use
        // GetInPlaceInputOutputPairs. Ideally, we'd use HloAliasAnalysis here,
        // but this code works with HloModules that aren't ready yet to use
        // HloAliasAnalysis (e.g., their computation graphs may not have been
        // flattened yet).
        for (const auto& operand_and_output_index :
             HloDataflowAnalysis::GetInPlaceInputOutputPairs(hlo)) {
          if (operand_and_output_index.second == index) {
            const HloUse& operand = operand_and_output_index.first;
            for (const auto* value :
                 dataflow_
                     ->GetValueSet(hlo->operand(operand.operand_number),
                                   operand.operand_index)
                     .values()) {
              auto value_type = ValueTypeAfterChange(value);
              if (value_type == BF16) {
                continue;
              }
              VLOG(2) << "Adjust to F32 due to InputOutPair: "
                      << value->ToString() << "\n";
              CHECK_EQ(value_type, F32);
              type = F32;
              break;
            }
          }
        }

        // It's possible that a user has been changed from BF16 to F32
        // during this final adjustment pass, so we need to check
        // AllUsersConsumeBF16() again.
        if (type == BF16 && !AllUsersConsumeBF16(*hlo, index)) {
          VLOG(2) << "Adjust to F32 due to All user consumeBF16 fail\n";
          type = F32;
        }
        if (type == F32) {
          for (const auto* value :
               dataflow_->GetValueSet(hlo, index).values()) {
            // We rely on the fact that this adjustment works in reverse
            // topological order. Adding the value to
            // values_that_must_be_kept_as_f32_ will ensure the correctness
            // of the adjustment for HLOs that will be processed later.
            values_that_must_be_kept_as_f32_.insert(value);
          }
        }
        if (type != output_type) {
          any_change = true;
          AddToOrRemoveFromBF16ChangeSet(hlo, index, type);
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " adjusted to " << (type == BF16 ? "BF16" : "F32") << ": "
                  << hlo->ToString();
          if (hlo->opcode() == HloOpcode::kParameter) {
            parameter_changed = true;
          }
        }
      };
      ShapeUtil::ForEachSubshape(hlo->shape(), adjust_hlo_output);
      AdjustCalledComputationRoot(hlo);
      if (hlo->opcode() == HloOpcode::kWhile) {
        // We need to run on the while body and condition repeatedly until a
        // fixed point is reached, i.e., the parameters do not change any more.
        // We may need more than one iteration because the while input and
        // output alias each other, so changing one input parameter requires
        // changing the corresponding output element and thus may transitively
        // require changing another input parameter. A fixed point will be
        // reached because the parameters can only be changed from BF16 to F32,
        // not the other way around.
        absl::flat_hash_set<const HloComputation*> visited_in_while;
        while (ResolveInconsistencyOfAliasingBuffersHelper(
                   hlo->while_condition(), &visited_in_while) ||
               ResolveInconsistencyOfAliasingBuffersHelper(hlo->while_body(),
                                                           &visited_in_while)) {
          visited_in_while.clear();
          ShapeUtil::ForEachSubshape(hlo->shape(), adjust_hlo_output);
          AdjustCalledComputationRoot(hlo);
        }
        visited_computations->insert(visited_in_while.begin(),
                                     visited_in_while.end());
      } else if (hlo->opcode() == HloOpcode::kFusion) {
        ResolveInconsistencyOfAliasingBuffersHelper(
            hlo->fused_instructions_computation(), visited_computations);
      } else if (hlo->opcode() == HloOpcode::kConditional) {
        for (auto* branch : hlo->branch_computations()) {
          ResolveInconsistencyOfAliasingBuffersHelper(branch,
                                                      visited_computations);
        }
      }
    }
    if (!any_change) {
      break;
    }
  }
  // Now adjust parameters of called computations.
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    AdjustCalledComputationParameters(*inst_it);
  }
  return parameter_changed;
}

void BFloat16Propagation::ResolveInconsistencyOfAliasingBuffers(
    HloModule* module) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_17(mht_17_v, 917, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ResolveInconsistencyOfAliasingBuffers");

  const auto& computations_topological_order =
      module->MakeComputationPostOrder();
  absl::flat_hash_set<const HloComputation*> resolved;
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(resolved, *comp_it)) {
      continue;
    }
    ResolveInconsistencyOfAliasingBuffersHelper(*comp_it, &resolved);
  }
}

Status BFloat16Propagation::ResolveInconsistentFusions(HloModule* module) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_18(mht_18_v, 933, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ResolveInconsistentFusions");

  // We could have changed a fusion computation's root shape to have a different
  // precision than the fusion node's output, if the fusion root does not
  // define a buffer (e.g., a tuple). Now we add conversions after such fusion
  // roots to make them match the fusion output. If the fusion output is a
  // (possibly nested) tuple, we first create get-tuple-elements, then convert
  // the unmatching leaf nodes, and finally create a new tuple as the fusion
  // computation's root. If tuples and get-tuple-elements are created, we will
  // run tuple simplifier and dead code elimination at the end (dead code is not
  // allowed in fusion computation). E.g.,
  //
  // (1)             (2)             (3)
  // a  b            a  b            a  b
  // |\ |            |\ |            |\ |
  // \ add   ->      |add    ->      | add
  //  \ |            \ |        convert |
  //  tuple         tuple             \ |
  //                 / \              tuple
  //               gte gte
  //                |   |
  //           convert  |
  //                 \  /
  //                 tuple
  // (1) a is F32 but tuple is BF16
  // (2) after adding conversion
  // (3) after tuple simplifier and DCE.
  for (auto computation : module->MakeComputationPostOrder()) {
    auto insts = computation->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      if (hlo->opcode() != HloOpcode::kFusion) {
        continue;
      }
      auto fusion_computation = hlo->fused_instructions_computation();
      auto fusion_root = fusion_computation->root_instruction();
      if (ShapeUtil::Compatible(fusion_root->shape(), hlo->shape())) {
        continue;
      }
      ShapeTree<HloInstruction*> converted_outputs(hlo->shape());
      // Deep copy the fusion root, and convert a leaf node only if its shape
      // does not match the fusion output.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          fusion_computation->DeepCopyInstructionWithCustomCopier(
              fusion_root,
              [hlo](HloInstruction* leaf, const ShapeIndex& leaf_index,
                    HloComputation* comp) {
                const Shape& hlo_subshape =
                    ShapeUtil::GetSubshape(hlo->shape(), leaf_index);
                if (ShapeUtil::Compatible(leaf->shape(), hlo_subshape)) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(hlo_subshape, leaf));
              }));
      fusion_computation->set_root_instruction(copy);
    }
  }
  return Status::OK();
}

Status BFloat16Propagation::ResolveConvertedConstants(HloModule* module) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_19(mht_19_v, 997, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ResolveConvertedConstants");

  // We may have converted some constants from F32 to BF16, so adjust the
  // constant literals in such cases. We do this here instead of when the
  // constant node's is changed because 1) the HloInstruction interface does not
  // allow resetting the literal so we have to create a new kConstant
  // instruction to replace the old one, which invalidates dataflow analysis,
  // and 2) it's possible that a kConstant's output gets changed to BF16 at the
  // beginning but later on adjusted back to F32, so converting literals here
  // can avoid repeated conversions.
  //
  // TODO(b/73833576): Consider resetting literal in HloInstruction.
  for (auto computation : module->MakeComputationPostOrder()) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (!Shape::Equal().MinorToMajorOnlyInLayout()(hlo->literal().shape(),
                                                     hlo->shape())) {
        TF_ASSIGN_OR_RETURN(auto converted_literal,
                            hlo->literal().ConvertToShape(hlo->shape()));
        auto new_constant = computation->AddInstruction(
            HloInstruction::CreateConstant(std::move(converted_literal)));
        UpdateLayout(new_constant->mutable_shape());
        TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_constant));
      }
    }
  }
  return Status::OK();
}

Status BFloat16Propagation::SkipNoopConversions(HloModule* module) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_20(mht_20_v, 1030, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::SkipNoopConversions");

  for (auto computation : module->computations()) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConvert) {
        continue;
      }
      auto source = hlo->mutable_operand(0);
      if (!ShapeUtil::Equal(source->shape(), hlo->shape())) {
        continue;
      }
      const bool is_root = hlo == computation->root_instruction();
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(source));
      if (is_root) {
        computation->set_root_instruction(source);
      }
    }
  }
  return Status::OK();
}

// The algorithm first does a forward pass (parameters to root) to determine a
// set of instructions to consider using bfloat16, then does a backward pass to
// determine the precisions of those instructions according to the need of
// their users. During the backward pass, the potential changes are stored in
// changes_to_bf16_ which are subject to further adjustments then applied to the
// HLOs.
StatusOr<bool> BFloat16Propagation::Run(HloModule* module) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_21(mht_21_v, 1059, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::Run");

  consider_using_bfloat16_.clear();
  instructions_visited_in_backward_pass_.clear();
  computations_visited_in_backward_pass_.clear();
  values_that_must_be_kept_as_f32_.clear();
  caller_counts_.clear();
  changes_to_bf16_.clear();
  changed_ = false;

  auto computations_topological_order = module->MakeComputationPostOrder();

  // Before running the propagation pass, we insert copies (kConvert to the same
  // type) of F32 inputs to while loops. This prevents other uses of the same
  // input from aliasing the while loop input/output, so that there's greater
  // chance to use BF16 inside the loop. If some of these added copies do not
  // help, they will remain F32 after BF16 propagation and will be removed since
  // they are no-ops.
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }

      auto operand = inst->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          computation->DeepCopyInstructionWithCustomCopier(
              operand, [](HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* comp) {
                if (leaf->shape().element_type() != F32) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(leaf->shape(), leaf));
              }));
      TF_RETURN_IF_ERROR(operand->ReplaceUseWith(inst, copy));
    }
  }

  TF_ASSIGN_OR_RETURN(dataflow_, HloDataflowAnalysis::Run(*module));

  // The first step is a forward pass (parameters to root), where we determine
  // the potential candidate instructions to use bfloat16 in the outputs that
  // are not likely to cause overhead from extra explicit conversions. This is
  // done forwardly because we determine whether an HLO is a candidate partially
  // based on whether its operands are candidates.
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (InstructionIsCandidateForBF16Output(inst)) {
        consider_using_bfloat16_.insert(inst);
      }
    }
  }

  // The second step is a backward pass (root to parameters), where we modify
  // the precisions of the instructions identified in the first step when
  // feasible. This is done backwardly because we determine the precision of an
  // HLO's output based on how it is later used.
  //
  // The precision of an instruction is determined by its users, so we do the
  // propagation in reverse topological order.
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(computations_visited_in_backward_pass_, *comp_it)) {
      continue;
    }
    auto insts = (*comp_it)->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it,
                                    /*skip_parameters=*/true);
    }
    computations_visited_in_backward_pass_.insert(*comp_it);
  }

  // It's possible that an instruction does not define a buffer, but the
  // defining instruction's shape has changed. So we need to adjust the output
  // shapes of instructions according to the HLO values they refer to.
  ResolveInconsistencyOfAliasingBuffers(module);

  // Apply the changes in changes_to_bf16_.
  for (auto& change : changes_to_bf16_) {
    auto inst = change.first;
    // It is possible that we marked inst to change precision even if it is an
    // unsupported change, when inst is the root of a fusion computation and it
    // has to match the fusion node's output precision. We do a convert instead
    // of in-place change for such cases.
    if (ShouldKeepPrecisionUnchanged(inst)) {
      auto users = inst->users();
      bool is_root = inst == inst->parent()->root_instruction();
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          inst->parent()->DeepCopyInstructionWithCustomCopier(
              inst, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* comp) {
                if (!ContainsKey(change.second,
                                 ShapeUtil::GetMutableSubshape(
                                     inst->mutable_shape(), leaf_index))) {
                  return leaf;
                }
                auto converted_shape =
                    ShapeUtil::ChangeElementType(leaf->shape(), BF16);
                UpdateLayout(&converted_shape);
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(converted_shape, leaf));
              }));
      for (auto user : users) {
        TF_RETURN_IF_ERROR(inst->ReplaceUseWithDifferentShape(user, copy));
      }
      if (is_root) {
        inst->parent()->set_root_instruction(copy,
                                             /*accept_different_shape=*/true);
      }
      continue;
    }
    for (const auto& entry : change.second) {
      auto subshape = entry.first;
      CHECK_EQ(subshape->element_type(), F32);
      subshape->set_element_type(BF16);
      UpdateLayout(subshape);
      changed_ = true;
    }
  }

  // Removes redundant HLOs added by this pass, either when inserting
  // de-aliasing copies to while loop inputs, or later when converting output
  // types.
  auto clean_up = [this, module]() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_22(mht_22_v, 1188, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "lambda");

    TF_RETURN_IF_ERROR(SkipNoopConversions(module));
    TupleSimplifier tuple_simplifier;
    TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
    return Status::OK();
  };

  if (!changed_) {
    TF_RETURN_IF_ERROR(clean_up());
    return false;
  }

  TF_RETURN_IF_ERROR(ResolveInconsistentFusions(module));
  TF_RETURN_IF_ERROR(ResolveConvertedConstants(module));

  TF_RETURN_IF_ERROR(clean_up());
  return true;
}

PrimitiveType BFloat16Propagation::OutputTypeAfterChange(
    HloInstruction* hlo, const ShapeIndex& index) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_23(mht_23_v, 1213, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::OutputTypeAfterChange");

  Shape* subshape = ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index);
  const PrimitiveType type_on_hlo = subshape->element_type();
  if (type_on_hlo != F32) {
    return type_on_hlo;
  }
  auto it = changes_to_bf16_.find(hlo);
  if (it == changes_to_bf16_.end()) {
    return type_on_hlo;
  }
  return ContainsKey(it->second, subshape) ? BF16 : F32;
}

PrimitiveType BFloat16Propagation::ValueTypeAfterChange(
    const HloValue* value) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_24(mht_24_v, 1230, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::ValueTypeAfterChange");

  auto hlo = value->defining_instruction();
  const auto& position = value->defining_position();
  return OutputTypeAfterChange(hlo, position.index);
}

void BFloat16Propagation::AddToOrRemoveFromBF16ChangeSet(
    HloInstruction* hlo, const ShapeIndex& index, PrimitiveType target_type) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_propagationDTcc mht_25(mht_25_v, 1240, "", "./tensorflow/compiler/xla/service/bfloat16_propagation.cc", "BFloat16Propagation::AddToOrRemoveFromBF16ChangeSet");

  if (target_type == BF16) {
    auto& entry = changes_to_bf16_[hlo];
    entry.emplace(ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index),
                  index);
  } else {
    CHECK_EQ(target_type, F32);
    auto it = changes_to_bf16_.find(hlo);
    if (it == changes_to_bf16_.end()) {
      return;
    }
    it->second.erase(
        ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index));
  }
}

}  // namespace xla
