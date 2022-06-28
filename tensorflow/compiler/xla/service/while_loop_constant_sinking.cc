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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc() {
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

#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

// Replaces all uses of old_instr with new_instr except the use at
// `while_body_root` (which must be a tuple instruction) at index `tuple_index`.
// This utility helps us replace an instruction in the while body with a
// constant while still keeping it trivially loop invariant.
static Status ReplaceUsesWhileKeepingLoopInvariance(
    HloInstruction* old_instr, HloInstruction* new_instr,
    HloInstruction* while_body_root, int64_t tuple_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/while_loop_constant_sinking.cc", "ReplaceUsesWhileKeepingLoopInvariance");

  CHECK_EQ(while_body_root->opcode(), HloOpcode::kTuple);

  std::vector<HloInstruction*> users;
  users.reserve(old_instr->user_count());
  absl::c_copy(old_instr->users(), std::back_inserter(users));

  for (auto* user : users) {
    for (int64_t i = 0, e = user->operand_count(); i < e; i++) {
      if (user->operand(i) == old_instr &&
          !(user == while_body_root && i == tuple_index)) {
        TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, new_instr));
      }
    }
  }

  return Status::OK();
}

StatusOr<bool> WhileLoopConstantSinking::TrySinkingConstantsIntoWhileLoop(
    HloInstruction* while_instr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/service/while_loop_constant_sinking.cc", "WhileLoopConstantSinking::TrySinkingConstantsIntoWhileLoop");

  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();

  const HloInstruction& init_value = *while_instr->operand(0);
  if (init_value.opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;

  absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>
      conditional_gte_index_to_insts =
          WhileUtil::GetGTEsMapForWhileConditional(*while_cond);
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    const HloInstruction& invariant_value = *init_value.operand(index);

    // Original value should be a constant.
    if (invariant_value.opcode() != HloOpcode::kConstant) {
      continue;
    }

    // Sink into the while_body.
    // Should have at least one user that's not while_body_root.
    if (invariant_body_gte->user_count() > 1) {
      HloInstruction* constant_instr =
          while_body->AddInstruction(invariant_value.Clone(/*suffix=*/".sunk"));
      TF_RETURN_IF_ERROR(ReplaceUsesWhileKeepingLoopInvariance(
          invariant_body_gte, constant_instr, while_body->root_instruction(),
          index));
      changed = true;
    }

    // Check if there is a corresponding GTE in while_conditional.
    auto it = conditional_gte_index_to_insts.find(index);
    if (it == conditional_gte_index_to_insts.end()) {
      continue;
    }

    for (HloInstruction* invariant_cond_gte : it->second) {
      // Should have at least one user.
      if (invariant_cond_gte->user_count() > 0) {
        HloInstruction* constant_instr = while_cond->AddInstruction(
            invariant_value.Clone(/*suffix=*/".sunk"));
        TF_RETURN_IF_ERROR(
            invariant_cond_gte->ReplaceAllUsesWith(constant_instr));
        changed = true;
      }
    }
  }

  return changed;
}

StatusOr<bool> WhileLoopConstantSinking::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_constant_sinkingDTcc mht_2(mht_2_v, 283, "", "./tensorflow/compiler/xla/service/while_loop_constant_sinking.cc", "WhileLoopConstantSinking::Run");

  VLOG(2) << "HLO module before WhileLoopConstantSinking:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeNonfusionComputations()) {
    // Right now we don't particularly care about optimizing while-of-while
    // patterns.  If/When we do, we'll want to visit the outer while (while_0)
    // before we visit the inner while (while_1):
    //
    // while_1_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   use(val)
    // }
    //
    // while_0_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   while_1 = while(init=tuple(val, ...), body=while_1_body, ...)
    //   ...
    // }
    //
    // main {
    //   while_0 = while(init=(constant, ...), body=while_0_body, ...)
    // }
    //
    // This will let us sink the constant into the outer while first and then
    // into the inner while in a single run of this pass.
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kWhile;
                    });
  }

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(bool result,
                        TrySinkingConstantsIntoWhileLoop(while_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopConstantSinking:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopConstantSinking";
  }

  return changed;
}
}  // namespace xla
