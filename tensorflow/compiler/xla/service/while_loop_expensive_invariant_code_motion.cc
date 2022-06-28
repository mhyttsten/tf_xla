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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc() {
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

#include "tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {
using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;

struct InvariantInfo {
  explicit InvariantInfo(int64_t user_count)
      : remaining_user_count(user_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "InvariantInfo");
}
  // The transitive input size of all input operands, traced up to the while
  // loop parameter or leaf invariant ops.
  int64_t transitive_input_size = 0;
  // The remaining users count that remain in the body after all hoistable
  // invariant users are hoisted. This number excludes the root instruction.
  int64_t remaining_user_count;
  // If this instruction is hoisted, this stores the copy outside the body.
  HloInstruction* hoisted_copy = nullptr;
  // Hoistable instructions depending on this op to be hoisted.
  InlinedVector<HloInstruction*, 2> blocked_users;
};

// Copies `to_hoist` to the computation containing `while_instr`, hoisting its
// operands as needed.  All of its transitive operands are expected to be in
// `invariant_instructions`. This function hoists the operands in
// `invariant_instructions` and sets the entry's hoisted_copy to the hoisted
// instruction.
static void CreateLoopInvariantCopy(
    flat_hash_map<HloInstruction*, InvariantInfo>* invariant_instructions,
    HloInstruction* while_instr, HloInstruction* to_hoist) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "CreateLoopInvariantCopy");

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
    // All of the operands for old_instruction have been cloned, so it is time
    // to clone old_instruction itself.
    if (frame->operand_index == frame->instruction->operand_count()) {
      HloInstruction* old_instruction = frame->instruction;
      InvariantInfo& info = FindOrDie(*invariant_instructions, old_instruction);

      // Check if this instruction might have already been hoisted.
      if (info.hoisted_copy == nullptr) {
        auto get_new_operand = [&](HloInstruction* old_operand) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_2(mht_2_v, 256, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "lambda");

          return old_operand == while_body_param
                     ? while_operand
                     : FindOrDie(*invariant_instructions, old_operand)
                           .hoisted_copy;
        };

        InlinedVector<HloInstruction*, 4> new_operands;
        absl::c_transform(old_instruction->operands(),
                          std::back_inserter(new_operands), get_new_operand);

        HloInstruction* new_instruction = parent_of_while->AddInstruction(
            old_instruction->CloneWithNewOperands(old_instruction->shape(),
                                                  new_operands));

        info.hoisted_copy = new_instruction;
      }

      dfs_stack.pop_back();
      continue;
    }

    HloInstruction* next_operand =
        frame->instruction->mutable_operand(frame->operand_index++);
    if (next_operand == while_body_param ||
        FindOrDie(*invariant_instructions, next_operand).hoisted_copy !=
            nullptr) {
      continue;
    }

    dfs_stack.push_back({next_operand, 0});
  } while (!dfs_stack.empty());
}
}  // namespace

StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::
    TryHoistingInvariantInstructionsFromWhileBody(HloInstruction* while_instr) {
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

  // Contains the information for all invariant instructions that can be legally
  // hoisted. When we hoist an instruction in this set, we set its hoisted_copy
  // field to the hoisted instruction.
  flat_hash_map<HloInstruction*, InvariantInfo> invariant_instructions;

  // Map from an invariant instruction to the number of remaining unresolved
  // operands, i.e. operands used by unvisited instructions. If all these
  // operands are used by other invariant instructions, then hoisting out that
  // operand won't leave a copy of itself in the body and it's free to hoist.
  flat_hash_map<HloInstruction*, int64_t> to_hoist_when_ready;

  // Identify invariant GTE instructions so that we can identify its users that
  // are also invariants.
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    // TODO(b/79147885): We should try to generalize this to tuples for
    // uniformity's sake, if nothing else.
    if (instr->shape().IsArray()) {
      // We subtract 1 from user_count because we know one of the users is root.
      auto emplace_result = invariant_instructions.emplace(
          instr, InvariantInfo(/*user_count=*/instr->user_count() - 1));
      CHECK(emplace_result.second);
      InvariantInfo& info = emplace_result.first->second;
      info.transitive_input_size = shape_size_function_(instr->shape());
    }
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

  auto hoist = [&](HloInstruction* instruction, const InvariantInfo& info) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_3(mht_3_v, 362, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "lambda");

    if (info.hoisted_copy) {
      // Already hoisted.
      return;
    }
    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);

    CreateLoopInvariantCopy(&invariant_instructions, while_instr, instruction);

    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(info.hoisted_copy);
  };

  // Temporary helper container for marking a operand as checked when
  // decrementing its remaining_user_count counter. Cleared after each
  // iteration.
  flat_hash_set<HloInstruction*> checked_operands;

  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty() ||
        instruction == while_body->root_instruction()) {
      continue;
    }

    auto is_invariant = [&](HloInstruction* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_4(mht_4_v, 392, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "lambda");

      return invariant_instructions.find(op) != invariant_instructions.end();
    };

    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }

    auto emplace_result = invariant_instructions.emplace(
        instruction, InvariantInfo(/*user_count=*/instruction->user_count()));
    CHECK(emplace_result.second);
    InvariantInfo& instr_info = emplace_result.first->second;
    // If root is a users of it, substract 1 from remaining user count as we
    // don't want root to be blocking other users from being hoisted. Note that
    // for invariant parameter GTEs, they will skip the iteration because their
    // operand parameter(0) is not invariant, and they are put into
    // invariant_instructions before this loop.
    for (auto* user : instruction->users()) {
      if (user == while_body->root_instruction()) {
        --instr_info.remaining_user_count;
        break;
      }
    }

    int64_t num_blocking_operands = 0;
    // Check that hoisting the instruction doesn't cause a significant memory
    // blow-up. LICM extends the live-range of the output of the hoisted
    // instruction to be the entire while loop, which may be problematic on
    // platforms where memory is limited. This can be especially harmful if
    // the instruction has a significantly larger output than its input, e.g.
    // kIota, kBroadcast or kConstant.
    int64_t output_size = 0;

    for (auto* operand : instruction->operands()) {
      auto& operand_info = invariant_instructions.at(operand);
      if (!checked_operands.contains(operand)) {
        instr_info.transitive_input_size += operand_info.transitive_input_size;
        --operand_info.remaining_user_count;
        checked_operands.insert(operand);
      }
      if (operand_info.remaining_user_count == 0) {
        // All users are hoistable invariants, unblock held off users.
        for (auto* user : operand_info.blocked_users) {
          auto it = to_hoist_when_ready.find(user);
          if (it != to_hoist_when_ready.end()) {
            auto& num_blocking = it->second;
            CHECK_GT(num_blocking, 0);
            --num_blocking;
            // Hoist a previously held off instruction now that there are no
            // more blocking operands.
            if (num_blocking == 0) {
              hoist(user, invariant_instructions.at(user));
              to_hoist_when_ready.erase(it);
            }
          }
        }
        operand_info.blocked_users.clear();
      } else if (operand_info.remaining_user_count > 0) {
        ++num_blocking_operands;
        if (operand_info.blocked_users.empty() ||
            operand_info.blocked_users.back() != instruction) {
          operand_info.blocked_users.push_back(instruction);
        }
      } else {
        LOG(FATAL)
            << "An instruction should not have number of negative users.";
      }
    }
    checked_operands.erase(checked_operands.begin(), checked_operands.end());
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&output_size, this](const Shape& subshape,
                             const ShapeIndex& /*index*/) {
          if (subshape.IsArray()) {
            output_size += shape_size_function_(subshape);
          }
        });
    // If it is size-inflating, we leave it as is and potentially will still
    // hoist it out if we later found a group of ops that are worth hoisting
    // as a whole.
    if (output_size > instr_info.transitive_input_size) {
      continue;
    }

    if (!worth_hoisting_individually_(*instruction)) {
      continue;
    }

    // Need to wait until we inspected the users of some operands until we can
    // finally decide whether to hoist this instruction.
    if (num_blocking_operands > 0) {
      to_hoist_when_ready.emplace(instruction, num_blocking_operands);
      continue;
    }

    hoist(instruction, instr_info);
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

StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::Run(HloModule* module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_expensive_invariant_code_motionDTcc mht_5(mht_5_v, 519, "", "./tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.cc", "WhileLoopExpensiveInvariantCodeMotion::Run");

  VLOG(2) << "HLO module before WhileLoopExpensiveInvariantCodeMotion:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->computations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kWhile;
                    });
  }

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

    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopExpensiveInvariantCodeMotion:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2)
        << "HLO module unchanged after WhileLoopExpensiveInvariantCodeMotion";
  }

  return changed;
}
}  // namespace xla
