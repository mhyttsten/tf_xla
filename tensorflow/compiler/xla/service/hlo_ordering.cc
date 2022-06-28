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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

bool HloOrdering::ExecutesBefore(const HloInstruction* a,
                                 const HloInstruction* b) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::ExecutesBefore");

  switch (GetExecutionConstraint(a, b)) {
    case ExecutionConstraint::kIsSame:  // a and b are the same instruction;
      return false;
    case ExecutionConstraint::kRunBeforeStart:
    case ExecutionConstraint::kRunBeforeEnd:
    case ExecutionConstraint::kRunExclusiveBefore:
      return true;
    case ExecutionConstraint::kRunExclusiveAfter:
    case ExecutionConstraint::kRunAfter:
    case ExecutionConstraint::kUnordered:
      return false;
  }
}

HloOrdering::ExecutionConstraint HloOrdering::GetExecutionConstraint(
    const HloInstruction* a, const HloInstruction* b) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::GetExecutionConstraint");

  // 'a' and 'b' may be in different computations. In this case, find the
  // callgraph ancestor instructions which call (potentially transitively) the
  // computations containing 'a' and 'b' and use these ancestor instructions to
  // compare order.
  if (a == b) {
    return ExecutionConstraint::kIsSame;
  }
  const HloInstruction* a_ancestor;
  const HloInstruction* b_ancestor;
  std::tie(a_ancestor, b_ancestor) =
      call_graph_->NearestAncestorsInSameComputation(
          const_cast<HloInstruction*>(a), const_cast<HloInstruction*>(b));

  if (a_ancestor == nullptr) {
    VLOG(4) << "Ancestors in a common computation could not be found between"
            << a->ToString() << "\n and \n"
            << b->ToString() << "\n so consider them to be unordered.\n";
    return ExecutionConstraint::kUnordered;
  }
  // a_ancestor and b_ancestor must be either both null or both non-null.
  CHECK_NE(b_ancestor, nullptr);
  CHECK_EQ(a_ancestor->parent(), b_ancestor->parent());

  // If the common ancestor is a while instruction there is an additional
  // ordering criteria which may apply. The condition computation is considered
  // to execute before the body computation so if 'a' is in the condition and
  // 'b' is in the body, then 'a' executes before 'b'.
  if (a_ancestor == b_ancestor && a_ancestor->opcode() == HloOpcode::kWhile) {
    const HloComputation* body = a_ancestor->while_body();
    const HloComputation* condition = a_ancestor->while_condition();
    if (call_graph_->InstructionIsNestedIn(a, condition) &&
        call_graph_->InstructionIsNestedIn(b, body)) {
      return ExecutionConstraint::kRunBeforeEnd;
    }
  }

  // If the common ancestor is a conditional instruction, even though the branch
  // computations are not really ordered per-se, we define the 0th branch
  // computation to be ordered before the 1st one, before the 2nd and so forth.
  // This ensures that buffers can still be shared among branch computations
  // as they will forcibly have disjoint liveness.
  if (a_ancestor == b_ancestor &&
      (a_ancestor->opcode() == HloOpcode::kConditional)) {
    int a_branch = -1;
    int b_branch = -1;
    for (int j = 0; j < a_ancestor->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a, a_ancestor->branch_computation(j))) {
        a_branch = j;
      }
      if (call_graph_->InstructionIsNestedIn(
              b, a_ancestor->branch_computation(j))) {
        b_branch = j;
      }
    }
    // If neither a nor b is inside the branches they both are the ancestor.
    if (a_branch == -1 && b_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      CHECK_EQ(b, b_ancestor);
      CHECK_EQ(a, b);
      return ExecutionConstraint::kIsSame;
    }
    // If 'b' is the conditional ancestor, and 'a' is within a branch
    // computation, 'a' executes before 'b'.
    if (b_branch == -1) {
      CHECK_EQ(b, a_ancestor);
      return ExecutionConstraint::kRunBeforeEnd;
    }
    if (a_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      return ExecutionConstraint::kRunAfter;
    }
    if (a_branch < b_branch) {
      return ExecutionConstraint::kRunExclusiveBefore;
    }
    if (b_branch < a_branch) {
      return ExecutionConstraint::kRunExclusiveAfter;
    }
  }

  if (ExecutesBeforeInSameComputation(a_ancestor, b_ancestor)) {
    return ExecutionConstraint::kRunBeforeStart;
  }
  if (ExecutesBeforeInSameComputation(b_ancestor, a_ancestor)) {
    return ExecutionConstraint::kRunAfter;
  }
  VLOG(1) << "Cannot determine order between:" << a->ToString() << "\n"
          << "and " << b->ToString() << " which are in the same computation\n";
  return ExecutionConstraint::kUnordered;
}

bool HloOrdering::IsDefinedBefore(const HloValue& a, const HloValue& b) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_2(mht_2_v, 319, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::IsDefinedBefore");

  // Entry parameter should always be defined before other instructions.
  const HloModule* module = b.defining_instruction()->parent()->parent();
  if (b.defining_instruction()->parent() == module->entry_computation() &&
      b.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return false;
  }

  if (a.defining_instruction()->parent() == module->entry_computation() &&
      a.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return true;
  }

  // Phi values require special handling. Because XLA does not have a phi
  // instruction, the definition instruction of the phis values are
  // placeholders: either the subcomputation parameter (body or condition) or
  // the while instruction. However, the program point where these values are
  // logically defined does not necessarily coincide exactly with program point
  // of these place-holder instructions. So we explicitly define the following
  // order for phi values:
  //
  //   body/condition parameter phi:
  //     Defined before all values defined in its computation excepting other
  //     phis.
  //
  //   while phi:
  //     defined after all values defined in the condition or body.
  //
  auto is_body_or_condition_phi = [](const HloValue& v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_3(mht_3_v, 350, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "lambda");

    return v.is_phi() &&
           v.defining_instruction()->opcode() == HloOpcode::kParameter;
  };
  if (is_body_or_condition_phi(a) && !is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(b.defining_instruction(),
                                         a.defining_instruction()->parent())) {
    return true;
  }
  if (is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(a.defining_instruction(),
                                         b.defining_instruction()->parent())) {
    return false;
  }

  // If 'b' is a while phi and 'a' is in the body or condition, then 'a'
  // executes before 'b'.
  if (b.is_phi() && b.defining_instruction()->opcode() == HloOpcode::kWhile &&
      (call_graph_->InstructionIsNestedIn(
           a.defining_instruction(), b.defining_instruction()->while_body()) ||
       call_graph_->InstructionIsNestedIn(
           a.defining_instruction(),
           b.defining_instruction()->while_condition()))) {
    return true;
  }
  // If 'b' is a conditional phi and 'a' is in some branch computation, then 'a'
  // executes before 'b'.
  if (b.is_phi() &&
      b.defining_instruction()->opcode() == HloOpcode::kConditional) {
    for (int j = 0; j < b.defining_instruction()->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a.defining_instruction(),
              b.defining_instruction()->branch_computation(j))) {
        return true;
      }
    }
  }
  return ExecutesBefore(a.defining_instruction(), b.defining_instruction());
}

/* static */
bool HloOrdering::UsesBeforeValueDefinition(
    absl::Span<const HloUse* const> uses, const HloValue& value,
    const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_4(mht_4_v, 397, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::UsesBeforeValueDefinition");

  bool has_use_in_exclusive_branches = false;
  bool has_escaped_use_in_conditional = false;
  auto UseIsBeforeValueDefinition = [&](const HloUse& use) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_5(mht_5_v, 403, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "lambda");

    VLOG(4) << "UseIsBeforeValueDefinition(use=" << use
            << ", value=" << value.ToShortString() << ")";
    switch (
        GetExecutionConstraint(use.instruction, value.defining_instruction())) {
      case HloOrdering::ExecutionConstraint::kIsSame:
        // If the use is at the instruction where the value is defined, then the
        // use is before the def if the instruction allows buffer sharing (in
        // place computation).
        if (use_is_always_before_def_in_same_instr ||
            dataflow.CanShareOperandBufferWithUser(
                use.instruction->mutable_operand(use.operand_number),
                use.operand_index, value.defining_instruction(),
                value.defining_index())) {
          VLOG(4)
              << "  use is value def, and instruction can share use buffer.";
          return true;
        }
        break;
      case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
        // If the use is located in a branch that is exclusive to the branch
        // where value is located, in order for them to interfere, there must be
        // an execution path where the value's definition can reach the use, so
        // that the wrong value would reach use if their live ranges are merged.
        // If there is such a path, it would have to pass through the point
        // where the two exclusive branches are joined --- specifically the end
        // of the conditional operation. For the join point to reach back to the
        // use at the other exclusive branch, there has to be a be a surrounding
        // loop, where the result of the conditional is passed back inside the
        // conditional through one of its parameters. This use-def conflict
        // between the parameter of a conditional and one of its branches is
        // caught in the has_escaped_use_in_conditinoal variable.
        VLOG(4) << " use and value def are in exclusive branches.";
        if (!has_escaped_use_in_conditional) {
          has_use_in_exclusive_branches = true;
          VLOG(4) << "Allowing them to share buffer.\n";
          return true;
        }
        VLOG(4) << "value def has escaped use in conditional. \n";
        break;
      case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
      case HloOrdering::ExecutionConstraint::kRunBeforeStart:
      case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
        VLOG(4)
            << "  use instruction executes before value-defining instruction";
        return true;
      case HloOrdering::ExecutionConstraint::kRunAfter:
        // Treat CollectivePermuteDone as a special case as it shares the buffer
        // from its operand (CollectivePermuteStart).
        if (use_is_always_before_def_in_same_instr &&
            use.instruction->opcode() == HloOpcode::kCollectivePermuteDone &&
            use.instruction->operand(0) == value.instruction()) {
          return true;
        }
        break;
      case HloOrdering::ExecutionConstraint::kUnordered:
        break;
    }

    // The use at a while is an input to a phi, and logically occurs before
    // values are defined in the body. Note that the use is *not* before the
    // value if the value is defined in the condition and is not the condition
    // parameter, since the input of a while's live range is only ended at the
    // start the body.
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      const HloInstruction* xla_while = use.instruction;
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_body())) {
        VLOG(4) << "  use is while " << use.instruction->name()
                << " and def is in body";
        return true;
      }
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_condition())) {
        if (value.defining_instruction() !=
            xla_while->while_condition()->parameter_instruction(0)) {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is not the parameter";
          return false;
        } else {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is the parameter";
          return true;
        }
      }
    }
    // Similarly if the value is defined at a while, it logically occurs after
    // any uses in the body or condition computations.
    if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
      CHECK(value.is_phi());
      const HloInstruction* xla_while = value.defining_instruction();
      if (call_graph_->InstructionIsNestedIn(use.instruction,
                                             xla_while->while_body()) ||
          call_graph_->InstructionIsNestedIn(use.instruction,
                                             xla_while->while_condition())) {
        VLOG(4) << "  value is while " << value.defining_instruction()->name()
                << " and use is in condition or body";
        return true;
      }
    }
    // The use at a call occurs before values that are defined in the called
    // computation.
    if (use.instruction->opcode() == HloOpcode::kCall) {
      const HloInstruction* call = use.instruction;
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             call->to_apply())) {
        VLOG(4) << "  use is call " << use.instruction->name()
                << " and def is in called computation";
        return true;
      }
    }
    if (use.instruction->opcode() == HloOpcode::kConditional) {
      const HloInstruction* conditional = use.instruction;
      // In general the use of a value in the conditional parameter should be
      // considered to be before a definition in one of its branches, and
      // therefore allowed in live range merging, if there is no
      // surrounding loop that creates a backward control flow path that
      // allows the definition in the branch to have its value flow backward
      // into the conditional and then flow into another branch in the
      // conditional that uses the value. This is reflected by checking that
      // the use-def in exclusive branches has not been already allowed.
      // Further, if the def value escapes its branch, we conservatively
      // assume a backward control flow path could exist, and set
      // has_escaped_use_in_conditinoal to disallow any later uses in
      // exclusive branches.
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (call_graph_->InstructionIsNestedIn(
                value.defining_instruction(),
                conditional->branch_computation(j))) {
          // If the use operand does not create a new value, and the value def
          // is returned by as part of the result of the conditional, it
          // is possible for the branch definition to flow backward through a
          // surrounding loop and then back into the conditional parameter.
          if (!dataflow.ValueIsDefinedAt(
                  use.instruction->operand(use.operand_number), {})) {
            for (auto value_use : value.GetUses()) {
              VLOG(4) << "def have use:" << value_use << "\n";
              if (value_use.instruction ==
                  value_use.instruction->parent()->root_instruction()) {
                VLOG(4) << "def use is conditional root \n";
                has_escaped_use_in_conditional = true;
                break;
              }
            }
          }
          if (!has_use_in_exclusive_branches) {
            VLOG(4) << "  use is conditional " << use.instruction->name()
                    << " and def is in " << j << "th branch computation";
            return true;
          }
        }
      }
      if (value.defining_instruction() == use.instruction) {
        VLOG(4) << "  use is conditional " << use << " and def is "
                << value.ToShortString();
        return true;
      }
    }

    VLOG(4) << "  use is not before value definition";
    return false;
  };
  for (auto* use : uses) {
    if (!UseIsBeforeValueDefinition(*use)) {
      return false;
    }
  }
  return true;
}

bool HloOrdering::LiveRangeStrictlyBefore(
    const HloValue& a, const HloValue& b, const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_6(mht_6_v, 578, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::LiveRangeStrictlyBefore");

  VLOG(4) << "LiveRangeStrictlyBefore(a = " << a.ToShortString()
          << ", b = " << b.ToShortString() << ")";
  VLOG(4) << "Parent:" << a.instruction()->parent()->ToString() << "\n";
  if (!IsDefinedBefore(a, b)) {
    VLOG(4) << a << " not defined before " << b;
    return false;
  }

  if (a.live_out_of_module()) {
    VLOG(4) << a << " is live out of module and not defined before " << b;
    return false;
  }

  // If the root instruction aliases the buffer 'a', the live range of 'a' is
  // until the end of the computation and can never be strictly before another
  // buffer nested in the same computation. This is needed to prevent the root
  // instruction's buffers from being reused by later instructions even when
  // the root is not the last instruction in the schedule.
  for (const HloPosition& pos : a.positions()) {
    if (pos.instruction->parent()->root_instruction() == pos.instruction &&
        call_graph().InstructionIsNestedIn(b.instruction(),
                                           pos.instruction->parent())) {
      return false;
    }
  }

  // All uses of 'a' must be before 'b' is defined.
  std::vector<const HloUse*> uses;
  for (const HloUse& use : a.GetUses()) {
    if (dataflow.DoesNotUseOperandBuffer(a.instruction(), a.index(),
                                         use.instruction)) {
      continue;
    }
    uses.push_back(&use);
  }
  if (!UsesBeforeValueDefinition(uses, b, dataflow,
                                 use_is_always_before_def_in_same_instr)) {
    VLOG(4) << "uses of " << a << "not before " << b << " is defined";
    return false;
  }

  if (a.IsRootOf(b.instruction()->parent())) {
    VLOG(4) << a << " is live out of computation and defined before " << b
            << " which is in same computation";
    return false;
  }

  return true;
}

bool HloOrdering::MayInterfere(const HloValue& a, const HloValue& b,
                               const HloDataflowAnalysis& dataflow) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_7(mht_7_v, 633, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "HloOrdering::MayInterfere");

  // Buffers without disjoint liveness may interfere.
  return !LiveRangeStrictlyBefore(a, b, dataflow) &&
         !LiveRangeStrictlyBefore(b, a, dataflow);
}

PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : HloOrdering(module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_8(mht_8_v, 643, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "PredecessorHloOrdering::PredecessorHloOrdering");
}

bool PredecessorHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_9(mht_9_v, 649, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "PredecessorHloOrdering::ExecutesBeforeInSameComputation");

  CHECK_EQ(a->parent(), b->parent());

  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b'.
  return a != b && predecessors_.at(a->parent())->IsReachable(a, b);
}

std::string PredecessorHloOrdering::ToStringHelper(
    const std::string& name) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_10(mht_10_v, 661, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "PredecessorHloOrdering::ToStringHelper");

  std::vector<std::string> pieces;
  pieces.push_back(name);
  for (auto* computation : module_->MakeNonfusionComputations()) {
    pieces.push_back(absl::StrFormat("computation %s:", computation->name()));
    const auto all = computation->MakeInstructionPostOrder();
    for (auto instruction : all) {
      pieces.push_back(
          absl::StrFormat("  %s predecessors:", instruction->name()));
      for (auto predecessor : all) {
        if (predecessors_.at(computation)
                ->IsReachable(predecessor, instruction)) {
          pieces.push_back(absl::StrFormat("    %s", predecessor->name()));
        }
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}

DependencyHloOrdering::DependencyHloOrdering(const HloModule* module)
    : PredecessorHloOrdering(module) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_11(mht_11_v, 685, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "DependencyHloOrdering::DependencyHloOrdering");

  // Compute predecessor relationships between all instructions to determine
  // ordering based on dependencies. ExecutesBefore will return true iff there
  // exists a path in the HLO computation graph from 'a' to 'b'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    predecessors_.emplace(computation, HloReachabilityMap::Build(computation));
  }
}

std::string DependencyHloOrdering::ToString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_12(mht_12_v, 697, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "DependencyHloOrdering::ToString");

  return ToStringHelper("DependencyHloOrdering");
}

SequentialHloOrdering::SequentialHloOrdering(const HloSchedule& schedule)
    : HloOrdering(schedule.module()), schedule_(schedule) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_13(mht_13_v, 705, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::SequentialHloOrdering");

  Initialize();
}

SequentialHloOrdering::SequentialHloOrdering(HloSchedule&& schedule)
    : HloOrdering(schedule.module()), schedule_(std::move(schedule)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_14(mht_14_v, 713, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::SequentialHloOrdering");

  Initialize();
}

void SequentialHloOrdering::Initialize() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_15(mht_15_v, 720, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::Initialize");

  // Create a map from instruction to its order position.
  TF_DCHECK_OK(schedule_.Verify());
  for (const auto& computation_sequence : schedule_.sequences()) {
    const auto& order = computation_sequence.second.instructions();
    for (int i = 0; i < order.size(); ++i) {
      InsertOrDie(&order_position_, order[i], i);
    }
  }
}

bool SequentialHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_16(mht_16_v, 735, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::ExecutesBeforeInSameComputation");

  CHECK_EQ(a->parent(), b->parent());
  // If either instruction is not in the order, then 'a' and 'b' are unordered.
  if (!order_position_.contains(a) || !order_position_.contains(b)) {
    return false;
  }
  if (a->parent()->root_instruction() == a) {
    // 'a' is the root instruction of the computation, which lives out. So
    // 'a' cannot execute before 'b'.
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}

const HloInstructionSequence* SequentialHloOrdering::SequentialOrder(
    const HloComputation& computation) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_17(mht_17_v, 753, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::SequentialOrder");

  return schedule_.is_computation_scheduled(&computation)
             ? &schedule_.sequence(&computation)
             : nullptr;
}

std::string SequentialHloOrdering::ToString() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTcc mht_18(mht_18_v, 762, "", "./tensorflow/compiler/xla/service/hlo_ordering.cc", "SequentialHloOrdering::ToString");

  return absl::StrCat("SequentialHloOrdering\n", schedule_.ToString());
}

}  // namespace xla
