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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc() {
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

#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/compile_time_cap.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;

// Copies `to_hoist` to the computation containing `while_instr`, hoisting its
// operands as needed.  All of its transitive operands are expected to be either
// in `hoisted_instructions` or `unhoisted_invariant_instructions`.  This
// function hoists the operands in `unhoisted_invariant_instructions` and moves
// them into `hoisted_instructions`.
static void CreateLoopInvariantCopy(
    flat_hash_map<HloInstruction*, HloInstruction*>* hoisted_instructions,
    flat_hash_set<HloInstruction*>* unhoisted_invariant_instructions,
    HloInstruction* while_instr, HloInstruction* to_hoist) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "CreateLoopInvariantCopy");

  HloComputation* parent_of_while = while_instr->parent();
  HloComputation* while_body = while_instr->while_body();

  struct DFSFrame {
    HloInstruction* instruction;
    int64_t operand_index;
  };

  InlinedVector<DFSFrame, 8> dfs_stack;
  dfs_stack.push_back({to_hoist, 0});

  HloInstruction* while_body_param = while_body->parameter_instruction(0);
  HloInstruction* while_operand = while_instr->mutable_operand(0);

  do {
    DFSFrame* frame = &dfs_stack.back();
    if (frame->operand_index == frame->instruction->operand_count()) {
      HloInstruction* old_instruction = frame->instruction;

      // All of the operands for old_instruction have been cloned, so it is
      // time to clone old_instruction itself.

      auto get_new_operand = [&](HloInstruction* old_operand) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "lambda");

        return old_operand == while_body_param
                   ? while_operand
                   : FindOrDie(*hoisted_instructions, old_operand);
      };

      InlinedVector<HloInstruction*, 4> new_operands;
      absl::c_transform(old_instruction->operands(),
                        std::back_inserter(new_operands), get_new_operand);

      HloInstruction* new_instruction =
          parent_of_while->AddInstruction(old_instruction->CloneWithNewOperands(
              old_instruction->shape(), new_operands));

      InsertOrDie(hoisted_instructions, old_instruction, new_instruction);

      // Approximately half of the instructions that would normally be present
      // in unhoisted_invariant_instructions are constants.  We save a bit of
      // compile time by not putting these in the hashtable.
      CHECK_EQ(unhoisted_invariant_instructions->erase(old_instruction),
               to_hoist != old_instruction &&
                   old_instruction->opcode() != HloOpcode::kConstant);
      dfs_stack.pop_back();
      continue;
    }

    HloInstruction* next_operand =
        frame->instruction->mutable_operand(frame->operand_index++);
    if (hoisted_instructions->contains(next_operand) ||
        next_operand == while_body_param) {
      continue;
    }

    dfs_stack.push_back({next_operand, 0});
  } while (!dfs_stack.empty());
}

// Returns true if `instruction` is worth hoisting only if it lets us hoist some
// instruction using it.  The rationale is that hoisting these instructions will
// prevent simplification and fusion in the while body.
bool WhileLoopInvariantCodeMotion::NotWorthHoistingIndividually(
    const HloInstruction& instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_2(mht_2_v, 283, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "WhileLoopInvariantCodeMotion::NotWorthHoistingIndividually");

  switch (instruction.opcode()) {
    default:
      return false;

    case HloOpcode::kConstant:
      return !hoist_constants_;

    case HloOpcode::kReshape:
      return !hoist_reshapes_;

    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kIota:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return true;
  }
}

StatusOr<bool>
WhileLoopInvariantCodeMotion::TryHoistingInvariantInstructionsFromWhileBody(
    HloInstruction* while_instr, BoundNonLinearCompilerAnalysis* allowance) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "WhileLoopInvariantCodeMotion::TryHoistingInvariantInstructionsFromWhileBody");

  auto print_no_metadata = HloPrintOptions{}.set_print_metadata(false);

  if (!while_instr->shape().IsTuple()) {
    // This restriction leaves one interesting pattern on the table:
    //
    //  while_body(f32[1024, 1024] %param) {
    //    %value = expensive_op(%param)
    //    outfeed(%value)
    //    ROOT = %param
    //  }
    //
    // If we see that pattern in the while, instead of generalizing this
    // algorithm to work with non-tuples, we should instead add a pass that
    // canonicalizes while loops like the above to use a tuple state.
    return false;
  }

  std::string while_instr_name = while_instr->ToString(print_no_metadata);
  VLOG(2) << "Trying to hoist from " << while_instr_name;

  auto maybe_upper_bound = ComputeWhileLoopTripCountUpperBound(while_instr);
  if (maybe_upper_bound && *maybe_upper_bound <= 1) {
    VLOG(2) << "Loop has a trip count of at most 1, skipping.";
    return false;
  }

  HloComputation* while_body = while_instr->while_body();

  // Maps instructions in the while body to instructions hoisted outside the
  // while that compute the same value.
  flat_hash_map<HloInstruction*, HloInstruction*> hoisted_instructions;

  // Contains instructions that can be legally hoisted, but were deemed to be
  // unprofitable to be hoisted alone by NotWorthHoistingIndividually.  When we
  // hoist an instruction in this set, we move it from
  // unhoisted_invariant_instructions to hoisted_instructions.
  flat_hash_set<HloInstruction*> unhoisted_invariant_instructions;

  // Invariant GTE's axiomatically satisfy the constraints for
  // unhoisted_invariant_instructions -- they can be legally hoisted, but there
  // is no benefit to hoisting them unless something that uses it is also
  // hoisted.
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    if (instr->shape().IsArray()) {
      // TODO(b/79147885): We should try to generalize this to tuples for
      // uniformity's sake, if nothing else.
      InsertOrDie(&unhoisted_invariant_instructions, instr);
    }
  }

  if (unhoisted_invariant_instructions.empty() && !hoist_constants_) {
    // There are no obviously loop invariant elements in the state being
    // threaded through the while loop so give up.  In theory this precondition
    // is too strong -- we could have code that e.g. permutes the elements in
    // the while state but uses a select to pick the same value on every
    // iteration.
    //
    // If we were asked to hoist constants, we need to scan the while body for
    // constants even if we didn't find any loop invariant values in the while
    // state tuple.
    return false;
  }

  // LICM in the presence of domain instructions is complex, bail.
  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kDomain) {
      return false;
    }
  }

  // instructions_to_replace[i] is hoisted into a loop invariant instruction
  // replacement_instructions[i].
  std::vector<HloInstruction*> instructions_to_replace;
  std::vector<HloInstruction*> replacement_instructions;

  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    allowance->DeductCost(1);
    if (!allowance->ContinueAnalysis()) {
      return false;
    }

    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty()) {
      continue;
    }

    if (!hoist_other_ && instruction->opcode() != HloOpcode::kConstant &&
        instruction->opcode() != HloOpcode::kReshape) {
      continue;
    }
    // Constants don't inflate, so size inflation check doesn't make sense for
    // constants.
    if (hoist_size_inflation_ratio_ &&
        instruction->opcode() != HloOpcode::kConstant) {
      // Check that hoisting the instruction doesn't cause a significant memory
      // blow-up. LICM extends the live-range of the output of the hoisted
      // instruction to be the entire while loop, which may be problematic on
      // platforms where memory is limited. This can be especially harmful if
      // the instruction has a significantly larger output than its input, e.g.
      // kIota, kBroadcast or kConstant.
      int64_t input_size = 0, output_size = 0;

      for (auto* operand : instruction->operands()) {
        ShapeUtil::ForEachSubshape(
            operand->shape(), [&input_size, this](const Shape& subshape,
                                                  const ShapeIndex& /*index*/) {
              if (subshape.IsArray()) {
                input_size += shape_size_function_(subshape);
              }
            });
      }
      ShapeUtil::ForEachSubshape(
          instruction->shape(),
          [&output_size, this](const Shape& subshape,
                               const ShapeIndex& /*index*/) {
            if (subshape.IsArray()) {
              output_size += shape_size_function_(subshape);
            }
          });

      if (output_size > input_size * *hoist_size_inflation_ratio_) {
        continue;
      }
    }

    auto is_invariant = [&](HloInstruction* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_4(mht_4_v, 441, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "lambda");

      return hoisted_instructions.find(op) != hoisted_instructions.end() ||
             unhoisted_invariant_instructions.contains(op) ||
             op->opcode() == HloOpcode::kConstant;
    };

    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }

    if (NotWorthHoistingIndividually(*instruction)) {
      VLOG(2) << "Adding " << instruction->ToString(print_no_metadata)
              << " to unhoisted invariant set.";
      // Approximately half of the instructions that reach this point are
      // constants.  We save a bit of compile time by not putting these in the
      // hashtable.
      if (instruction->opcode() != HloOpcode::kConstant) {
        InsertOrDie(&unhoisted_invariant_instructions, instruction);
      }
      continue;
    }

    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);

    CreateLoopInvariantCopy(&hoisted_instructions,
                            &unhoisted_invariant_instructions, while_instr,
                            instruction);

    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(
        FindOrDie(hoisted_instructions, instruction));
  }

  if (instructions_to_replace.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      WhileUtil::MakeInstructionsLiveInResult live_in_instructions_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, replacement_instructions));

  HloComputation* new_while_body =
      live_in_instructions_result.new_while_instr->while_body();

  for (int i = 0; i < instructions_to_replace.size(); i++) {
    HloInstruction* instruction_to_replace_in_new_while =
        FindOrDie(live_in_instructions_result.while_body_instruction_map,
                  instructions_to_replace[i]);
    TF_RETURN_IF_ERROR(new_while_body->ReplaceInstruction(
        instruction_to_replace_in_new_while,
        live_in_instructions_result.while_body_live_in_values[i]));
  }

  VLOG(1) << "Hoisted " << instructions_to_replace.size()
          << " instructions from " << while_instr_name;

  return true;
}

StatusOr<bool> WhileLoopInvariantCodeMotion::Run(HloModule* module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_invariant_code_motionDTcc mht_5(mht_5_v, 503, "", "./tensorflow/compiler/xla/service/while_loop_invariant_code_motion.cc", "WhileLoopInvariantCodeMotion::Run");

  VLOG(2) << "HLO module before WhileLoopInvariantCodeMotion:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kWhile;
                    });
  }
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);

  for (HloInstruction* while_instr : while_instrs) {
    // Right now we only hoist computations from the while body, but
    // TryHoistingInvariantInstructionsFromWhileBody can be generalized to
    // optimize the condition computation too, if needed.
    //
    // The transform we do here is a pessimization for while loops that execute
    // zero times*, but at this time we expect those to be rare.  If this
    // becomes a problem we can consider using the conditional HLO to avoid
    // doing extra work for while loops with zero trip count.
    //
    // * We delete while loops that have a zero trip count, so this would have
    //   to be a while loop with a somewhat opaque condition expression.

    if (!allowance.ContinueAnalysis()) {
      break;
    }
    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr, &allowance));
    changed |= result;
  }

  if (changed) {
    // Run DCE if changed. This pass may create new while loops with new
    // computations and if we don't delete the old ones, we can have spurious
    // verification failures (e.g., the verifier may see multiple channel
    // instructions that have the same channel ids).
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopInvariantCodeMotion:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopInvariantCodeMotion";
  }

  return changed;
}
}  // namespace xla
