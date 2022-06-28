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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc() {
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

#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"

namespace xla {
namespace {

// Create a constant (recursively for tuples) of the given shape and add it to
// the computation.
HloInstruction* CreateConstant(const Shape& shape,
                               HloComputation* computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "CreateConstant");

  if (shape.IsTuple()) {
    std::vector<HloInstruction*> tuple_arguments(shape.tuple_shapes_size());
    for (int index = 0; index < shape.tuple_shapes_size(); ++index) {
      tuple_arguments[index] =
          CreateConstant(shape.tuple_shapes(index), computation);
    }
    return computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_arguments));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
  }
}

// Extracts an instruction that satisfies filter from a fusion instruction.
// Returns nullptr if the fusion doesn't contain any instruction that satisfies
// filter.
const HloInstruction* ExtractInstruction(
    const HloInstruction* hlo,
    const std::function<bool(const HloInstruction*)>& filter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "ExtractInstruction");

  if (filter(hlo)) {
    return hlo;
  }
  if (hlo->opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  for (HloInstruction* inst :
       hlo->fused_instructions_computation()->instructions()) {
    if (filter(inst)) {
      return inst;
    }
  }
  return nullptr;
}

// Returns true if instruction is a collective op.
bool IsCollective(const HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_2(mht_2_v, 245, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "IsCollective");

  switch (instruction->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
      return true;
    default:
      return false;
  }
}

// Prints sub-expression rooted at inst for a given depth.
void PrintSubexpression(HloInstruction* inst, int depth) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "PrintSubexpression");

  if (depth == 0) {
    return;
  }
  for (auto* operand : inst->operands()) {
    PrintSubexpression(operand, depth - 1);
  }
  VLOG(2) << inst->ToString();
}

bool IsConstantScalarInt(const HloInstruction* inst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_4(mht_4_v, 280, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "IsConstantScalarInt");

  return inst->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(inst->shape()) &&
         inst->shape().IsInteger();
}

bool IsNotContainedInLoop(const HloInstruction& while_hlo,
                          const CallGraph& call_graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_5(mht_5_v, 290, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "IsNotContainedInLoop");

  const HloComputation* computation = while_hlo.parent();
  while (!computation->IsEntryComputation()) {
    auto& node = call_graph.GetNode(computation);
    CHECK_EQ(node.caller_callsites().size(), 1)
        << "The module is not flattened!";
    auto& callsite = node.caller_callsites()[0];
    if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      // Another while loop has been found traversing up the call tree.
      return false;
    }
    computation = callsite.instruction()->parent();
  }
  // No calling while loops were found.
  return true;
}

int GetLoopBoundWithOuterLoopMax(const HloInstruction& while_hlo,
                                 const CallGraph& call_graph,
                                 const int default_loop_count,
                                 const int max_outer_loop_count,
                                 const int max_loop_count) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_6(mht_6_v, 314, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "GetLoopBoundWithOuterLoopMax");

  int loop_bound = GetLoopBound(while_hlo, default_loop_count, max_loop_count);
  if (loop_bound > max_outer_loop_count) {
    // First does the inexpensive loop bound check to avoid as many
    // expensive graph traversals in IsNotContainedInLoop as possible.
    if (IsNotContainedInLoop(while_hlo, call_graph)) {
      return max_outer_loop_count;
    }
  }
  return loop_bound;
}

}  // namespace

int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_7(mht_7_v, 332, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "GetLoopBound");

  HloInstruction* condition = while_hlo.while_condition()->root_instruction();
  if (condition->opcode() == HloOpcode::kCompare) {
    int64_t value = 0;
    Comparison::Direction cmp = condition->comparison_direction();
    if ((cmp == Comparison::Direction::kLt ||
         cmp == Comparison::Direction::kLe ||
         cmp == Comparison::Direction::kNe) &&
        IsConstantScalarInt(condition->operand(1))) {
      value = *condition->operand(1)->literal().GetFirstInteger();
    } else if ((cmp == Comparison::Direction::kGt ||
                cmp == Comparison::Direction::kGe ||
                cmp == Comparison::Direction::kNe) &&
               IsConstantScalarInt(condition->operand(0))) {
      value = *condition->operand(0)->literal().GetFirstInteger();
    }
    if (value > 0) {
      // Caps to a max loop count to avoid long execution times.
      return std::min(value, static_cast<int64_t>(max_loop_count));
    }
  }
  return default_loop_count;
}

Status HloControlFlowFlattening::FlattenWhileLoop(
    HloInstruction* while_hlo, const CallGraph& call_graph) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_8(mht_8_v, 360, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::FlattenWhileLoop");

  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);
  HloComputation* computation = while_hlo->parent();
  // Add a new induction variable.
  HloInstruction* initialization = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)));
  // Create a new while operand with the induction variable added.
  HloInstruction* old_tuple = while_hlo->mutable_operand(0);
  HloInstruction* new_tuple =
      TupleUtil::AppendSuffix(old_tuple, {initialization});
  int new_tuple_size = new_tuple->shape().tuple_shapes().size();
  TF_RETURN_IF_ERROR(while_hlo->ReplaceOperandWithDifferentShape(0, new_tuple));

  auto change_op_shape = [&](HloInstruction* instruction) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_9(mht_9_v, 376, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "lambda");

    Shape* shape = instruction->mutable_shape();
    CHECK(shape->IsTuple());
    CHECK_EQ(shape->tuple_shapes().size(), new_tuple_size - 1);
    Shape* subshape = shape->add_tuple_shapes();
    return ShapeUtil::PopulateShape(S32, {}, subshape);
  };

  // Replace the given tuple-shaped instruction of size N in each of its
  // non-get-tuple-element users with a new tuple instruction which has the
  // first N - 1 elements.
  auto replace_non_gte_users =
      [](HloInstruction* new_tuple) -> StatusOr<HloInstruction*> {
    CHECK(new_tuple->shape().IsTuple());
    HloInstruction* prefix = nullptr;
    std::vector<HloInstruction*> users(new_tuple->users());
    for (HloInstruction* user : users) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      // Lazily extract the prefix on demand, reuse it as needed.
      if (prefix == nullptr) {
        prefix = TupleUtil::ExtractPrefix(
            new_tuple, new_tuple->shape().tuple_shapes_size() - 1);
      }
      TF_RETURN_IF_ERROR(new_tuple->ReplaceUseWithDifferentShape(user, prefix));
    }
    return prefix;
  };

  {
    // Add the new variable to the while loop condition.
    HloComputation* condition = while_hlo->while_condition();
    TF_RETURN_IF_ERROR(change_op_shape(condition->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(condition->parameter_instruction(0)).status());
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Loop condition in " << while_hlo->parent()->name();
      PrintSubexpression(condition->root_instruction(), /*depth=*/3);
    }
    const int loop_bound = GetLoopBoundWithOuterLoopMax(
        *while_hlo, call_graph, while_execution_count_, max_outer_loop_count_,
        max_loop_count_);

    VLOG(1) << "loop_bound = " << loop_bound;

    HloInstruction* limit = condition->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(loop_bound)));
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        condition->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, condition->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* compare =
        condition->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), induction_variable, limit,
            ComparisonDirection::kLt));
    TF_RETURN_IF_ERROR(
        condition->ReplaceInstruction(condition->root_instruction(), compare));
  }

  {
    // Add the new variable to the while loop body.
    HloComputation* body = while_hlo->while_body();
    TF_RETURN_IF_ERROR(change_op_shape(body->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(body->parameter_instruction(0)).status());
    HloInstruction* old_root = body->root_instruction();
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, body->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* increment = body->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    induction_variable = body->AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, induction_variable, increment));
    HloInstruction* new_root =
        TupleUtil::AppendSuffix(old_root, {induction_variable});
    body->set_root_instruction(new_root, /*accept_different_shape=*/true);
  }

  // Snapshot the users of while hlo before we add new users.
  std::vector<HloInstruction*> while_users(while_hlo->users().begin(),
                                           while_hlo->users().end());

  // Take care of the users of this while loop.
  TF_RETURN_IF_ERROR(change_op_shape(while_hlo));
  TF_ASSIGN_OR_RETURN(HloInstruction * prefix,
                      replace_non_gte_users(while_hlo));

  // If the while loop had been the root of its computation, make the prefix new
  // root.
  if (while_hlo->parent()->root_instruction() == while_hlo) {
    // We need to set accept_different_shape=true to reset the root shape to the
    // original, because we have already changed the shape of the old root
    // (while).
    if (prefix == nullptr) {
      prefix = TupleUtil::ExtractPrefix(while_hlo, new_tuple_size - 1);
    }
    while_hlo->parent()->set_root_instruction(prefix,
                                              /*accept_different_shape=*/true);
  }

  return Status::OK();
}

constexpr char kAllocateBuffer[] = "AllocateBuffer";

Status HloControlFlowFlattening::RemoveInfeed(
    HloInstruction* infeed_hlo) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_10(mht_10_v, 487, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemoveInfeed");

  CHECK_EQ(infeed_hlo->opcode(), HloOpcode::kInfeed);
  HloComputation* computation = infeed_hlo->parent();
  CHECK_EQ(infeed_hlo->shape().tuple_shapes_size(), 2);
  const Shape& infeed_shape = ShapeUtil::GetSubshape(infeed_hlo->shape(), {0});

  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(infeed_shape, {}, kAllocateBuffer));

  // Create a new tuple consisting op the constant and the token that was
  // originally the operand of infeed, and replace the infeed operation.
  auto new_tuple = HloInstruction::CreateTuple(
      {custom_call, infeed_hlo->mutable_operand(0)});
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(infeed_hlo, std::move(new_tuple)));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveRecvDone(
    HloInstruction* recv_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_11(mht_11_v, 511, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemoveRecvDone");

  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  CHECK_EQ(recv_done->operand_count(), 1);
  HloInstruction* recv = recv_done->mutable_operand(0);
  CHECK_EQ(recv->opcode(), HloOpcode::kRecv);

  HloComputation* computation = recv_done->parent();
  CHECK_EQ(recv_done->shape().tuple_shapes_size(), 2);
  const Shape& recv_shape = ShapeUtil::GetSubshape(recv_done->shape(), {0});

  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(recv_shape, {}, kAllocateBuffer));

  // Create a new tuple consisting op the constant and the token that was
  // originally the operand of recv, and replace the recv operation.
  auto new_tuple =
      HloInstruction::CreateTuple({custom_call, recv->mutable_operand(0)});
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(recv_done, std::move(new_tuple)));
  additional_removed->insert(recv);
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(recv));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveOutfeed(
    HloInstruction* outfeed_hlo) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_12(mht_12_v, 540, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemoveOutfeed");

  CHECK_EQ(outfeed_hlo->opcode(), HloOpcode::kOutfeed);
  HloComputation* computation = outfeed_hlo->parent();
  // Replace the outfeed with a no-op custom call with side effect to ensure the
  // operands aren't DCE'd.
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          outfeed_hlo->shape(), outfeed_hlo->operands(), "NopReturnToken"));
  Cast<HloCustomCallInstruction>(custom_call)
      ->set_custom_call_has_side_effect(true);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(outfeed_hlo, custom_call));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveSendDone(
    HloInstruction* send_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_13(mht_13_v, 560, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemoveSendDone");

  CHECK_EQ(send_done->opcode(), HloOpcode::kSendDone);
  CHECK_EQ(send_done->operand_count(), 1);
  HloInstruction* send = send_done->mutable_operand(0);
  CHECK_EQ(send->opcode(), HloOpcode::kSend);

  HloComputation* computation = send_done->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          send_done->shape(), send_done->operand(0)->operands(),
          "NopReturnToken"));
  Cast<HloCustomCallInstruction>(custom_call)
      ->set_custom_call_has_side_effect(true);

  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(send_done, custom_call));
  additional_removed->insert(send);
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(send));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveCollective(HloInstruction* hlo) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_14(mht_14_v, 584, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemoveCollective");

  HloComputation* computation = hlo->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          hlo->shape(), hlo->operands(), kAllocateBuffer));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, custom_call));
  return Status::OK();
}

Status HloControlFlowFlattening::RemovePartitionOrReplicaId(
    HloInstruction* hlo) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_15(mht_15_v, 597, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::RemovePartitionOrReplicaId");

  HloComputation* computation = hlo->parent();
  HloInstruction* zero = CreateConstant(hlo->shape(), computation);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, zero));
  return Status::OK();
}

StatusOr<bool> HloControlFlowFlattening::Run(HloModule* module) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTcc mht_16(mht_16_v, 607, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.cc", "HloControlFlowFlattening::Run");

  auto call_graph = CallGraph::Build(module);
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> removed;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (removed.contains(instruction)) {
        // Skip the instruction if it is already removed.
        continue;
      }
      if (flatten_while_loop_ && instruction->opcode() == HloOpcode::kWhile) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(FlattenWhileLoop(instruction, *call_graph));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kInfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveInfeed(instruction));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kOutfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveOutfeed(instruction));
        changed = true;
      } else if (instruction->opcode() == HloOpcode::kSendDone) {
        auto send_done_instruction =
            DynCast<HloSendDoneInstruction>(instruction);
        CHECK(send_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             send_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(RemoveSendDone(instruction, &removed));
          changed = true;
        }
      } else if (instruction->opcode() == HloOpcode::kRecvDone) {
        auto recv_done_instruction =
            DynCast<HloRecvDoneInstruction>(instruction);
        CHECK(recv_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             recv_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(RemoveRecvDone(instruction, &removed));
          changed = true;
        }
      } else if (remove_comm_ && IsCollective(instruction)) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveCollective(instruction));
        changed = true;
      } else if (remove_comm_ &&
                 (instruction->opcode() == HloOpcode::kPartitionId ||
                  instruction->opcode() == HloOpcode::kReplicaId)) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemovePartitionOrReplicaId(instruction));
      }
    }
  }

  // Fix the schedule if the module was scheduled.
  if (changed && module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

}  // namespace xla
