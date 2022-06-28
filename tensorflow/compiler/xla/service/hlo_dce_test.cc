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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dce_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dce_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dce_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloDceTest : public HloTestBase {
 protected:
  HloDceTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dce_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/hlo_dce_test.cc", "HloDceTest");
}

  // Returns whether the given instruction exists in the given computation.
  bool HasInstruction(const HloComputation& computation,
                      const HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dce_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/hlo_dce_test.cc", "HasInstruction");

    return absl::c_linear_search(computation.instructions(), instruction);
  }
};

TEST_F(HloDceTest, NoDeadCode) {
  // Verify that no dead code is removed from a computation with no dead code.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
}

TEST_F(HloDceTest, InstructionsWithSideEffect) {
  // Verify that side-effect instructions (Send in this test) are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(
      HloInstruction::CreateSend(constant, token, /*channel_id=*/0));
  builder.AddInstruction(HloInstruction::CreateSendDone(send));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(5, computation->instruction_count());
}

TEST_F(HloDceTest, CustomCallInstructionsWithSideEffect) {
  // Verify that custom call instruction with side-effect is not removed.
  auto builder = HloComputation::Builder(TestName());
  auto instr = Cast<HloCustomCallInstruction>(builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo")));
  instr->set_custom_call_has_side_effect(true);
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloDceTest, CustomCallInstructionsWithoutSideEffect) {
  // Verify that custom call instruction without side-effect is removed.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo"));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(HloDceTest, DeadParameters) {
  // Verify that dead parameters are not removed, but use of the dead parameters
  // are.
  auto builder = HloComputation::Builder(TestName());
  auto live_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "live_param"));
  auto dead_param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "dead_param1"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(F32, {}), "dead_param2"));

  // This is a dead negate instruction.
  builder.AddInstruction(HloInstruction::CreateUnary(
      dead_param1->shape(), HloOpcode::kNegate, dead_param1));

  // This negate is not dead because it is the root.
  builder.AddInstruction(HloInstruction::CreateUnary(
      live_param->shape(), HloOpcode::kNegate, live_param));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_EQ(1, dead_param1->user_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(0, dead_param1->user_count());
}

TEST_F(HloDceTest, ControlDependencies) {
  // Verify that instructions with control dependencies are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));

  // Create two dead instructions: a negate and an add.
  auto dead_negate = builder.AddInstruction(HloInstruction::CreateUnary(
      constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create the same two instructions again, but these will have a control
  // dependency added.
  auto dead_negate_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateUnary(
          constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateBinary(
          constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create a root so the previously added instruction is dead.
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Add a control dependency between two instructions.
  TF_ASSERT_OK(dead_negate_with_control_dep->AddControlDependencyTo(
      dead_add_with_control_dep));

  EXPECT_EQ(7, computation->instruction_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_negate));
  EXPECT_TRUE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_negate));
  EXPECT_FALSE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));
}

// Tests that a dead call instruction is removed.
TEST_F(HloDceTest, DeadInstructionWithCalledComputation) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Called computation for the call instruction.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto dead_call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {param}, called_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, dead_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_call));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_call));
}

// Tests that a while instruction with an infeed (effectul instruction) in its
// body is not removed, even its user count is 0.
TEST_F(HloDceTest, CalledComputationWithSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Condition computation of a while instruction.
  auto cond_builder = HloComputation::Builder(TestName() + "-cond");
  {
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "cond_param"));
    auto constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
    cond_builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param,
                                      constant, ComparisonDirection::kLt));
  }
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  // Body computation of a while instruction.
  auto body_builder = HloComputation::Builder(TestName() + "-body");
  {
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token = body_builder.AddInstruction(HloInstruction::CreateToken());
    auto infeed = body_builder.AddInstruction(
        HloInstruction::CreateInfeed(shape, token, ""));
    auto infeed_data = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, infeed, 0));
    body_builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, param, infeed_data));
  }
  auto body_computation = module->AddEmbeddedComputation(body_builder.Build());

  // Entry computation with a while instruction and a negate on the parameter.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_while = builder.AddInstruction(HloInstruction::CreateWhile(
      shape, cond_computation, body_computation, param));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  // Check the while instruction is not removed even if its user count is 0.
  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));
}

// Tests that a nested call instruction with a side effect is not removed.
TEST_F(HloDceTest, CalledComputationWithNestedSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Nested called computation with a side effect.
  auto nested_callee_builder =
      HloComputation::Builder(TestName() + "-nested_callee");
  {
    auto param = nested_callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token =
        nested_callee_builder.AddInstruction(HloInstruction::CreateToken());
    nested_callee_builder.AddInstruction(
        HloInstruction::CreateOutfeed(shape, param, token, ""));
  }
  auto nested_called_computation =
      module->AddEmbeddedComputation(nested_callee_builder.Build());

  // Outer called computation that calls the nested computation.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(HloInstruction::CreateCall(
        ShapeUtil::MakeTokenShape(), {param}, nested_called_computation));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_call = builder.AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeTokenShape(), {param}, called_computation));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));
}

TEST_F(HloDceTest, RemoveDeadSubcomputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto* param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "param0"));
    auto* param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {1}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  // We should have DCE'ed the reduction computation along with the reduction
  // instruction.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, KeepUsedSubcomputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto* param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "param0"));
    auto* param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation that also uses
  // reduce_subcomp.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {100}), "param1")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  // We shouldn't have DCE'ed reduce_subcomp, even though we removed one of
  // its users.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(HloDceTest, RemovedNestedDeadComputations) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  HloComputation::Builder called_subcomp_builder("called_dead_add");
  {
    auto* param0 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, shape, "param0"));
    auto* param1 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, shape, "param1"));
    called_subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto called_subcomp =
      module->AddEmbeddedComputation(called_subcomp_builder.Build());

  // Creates a module with unflattened control flow with two dead computations
  // that both call the same subcomputation, which becomes dead after the two
  // callers are removed.
  {
    HloComputation::Builder dead_subcomp_builder("dead_caller0");
    auto* param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto* param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  {
    HloComputation::Builder dead_subcomp_builder("dead_caller1");
    auto* param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto* param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  HloComputation::Builder builder(TestName());

  // Adds a constant instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 4);

  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  // Only the entry computation should be left after eliminating the dead caller
  // and callee subcomputations.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

}  // namespace
}  // namespace xla
