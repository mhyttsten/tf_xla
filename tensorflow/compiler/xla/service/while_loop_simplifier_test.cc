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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc() {
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

#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::_;
namespace op = xla::testing::opcode_matchers;

// Returns the first kWhile instruction within m's entry computation.
HloInstruction* FindFirstWhile(HloModule* m) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/while_loop_simplifier_test.cc", "FindFirstWhile");

  const auto& instrs = m->entry_computation()->instructions();
  return *absl::c_find_if(instrs, [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kWhile;
  });
}

class WhileLoopSimplifierTest : public HloTestBase {
 protected:
  // Makes an HloModule that contains a loop with `num_iters` iteration.
  ABSL_MUST_USE_RESULT std::unique_ptr<VerifiedHloModule>
  MakeModuleWithSimpleLoop(int num_iters);

  // Similar to MakeModuleWithSimpleLoop except that the loop bound is passed to
  // the loop-condition through an element of a tuple which is the
  // loop-condition parameter.
  ABSL_MUST_USE_RESULT std::unique_ptr<VerifiedHloModule>
  MakeModuleWithSimpleLoopTupleElementLoopBound(int num_iters);
};

std::unique_ptr<VerifiedHloModule>
WhileLoopSimplifierTest::MakeModuleWithSimpleLoop(int num_iters) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/service/while_loop_simplifier_test.cc", "WhileLoopSimplifierTest::MakeModuleWithSimpleLoop");

  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(42 + num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopSimplifierTest::MakeModuleWithSimpleLoopTupleElementLoopBound(
    int num_iters) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_loop_simplifier_testDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/xla/service/while_loop_simplifier_test.cc", "WhileLoopSimplifierTest::MakeModuleWithSimpleLoopTupleElementLoopBound");

  std::string hlo_string_template = R"(
  HloModule SimpleLoopWithIndirectLoopBound
  SimpleLoopWithIndirectLoopBound.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    limit = s32[] get-tuple-element(loop_var.1), index=2
    ROOT tuple = (s32[], s32[3]{0}, s32[]) tuple(add, multiply, limit)
  }
  SimpleLoopWithIndirectLoopBound.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=2
    ROOT less-than = pred[] compare(get-tuple-element.3, get-tuple-element.4), direction=LT
  }
  ENTRY SimpleLoopWithIndirectLoopBound {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.2 = s32[] constant({{LOOP_BOUND}})
    tuple.1 = (s32[], s32[3]{0}, s32[]) tuple(constant.3, constant.4,
      constant.2)
    ROOT while = (s32[], s32[3]{0}, s32[]) while(tuple.1),
      condition=SimpleLoopWithIndirectLoopBound.condition,
      body=SimpleLoopWithIndirectLoopBound.body
  }
  )";

  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(42 + num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
}

TEST_F(WhileLoopSimplifierTest, LoopWithZeroIterationSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/0);
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithZeroIterationTupleElementLoopBoundSimplified) {
  auto m = MakeModuleWithSimpleLoopTupleElementLoopBound(/*num_iters=*/0);
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(op::Constant(), op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, LoopWithOneIterationSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(op::Add(), op::Multiply()));
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithOneIterationTupleELementLoopBoundSimplified) {
  auto m = MakeModuleWithSimpleLoopTupleElementLoopBound(/*num_iters=*/1);
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(op::Add(), op::Multiply(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, LoopWithTwoIterationsNotSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/2);
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithControlDependencySimplifiedDependencyPreserved) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloComputation* computation = m->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* true_op = while_op->while_body()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSERT_OK(true_op->AddControlDependencyTo(
      while_op->while_body()->root_instruction()));
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction()->control_predecessors(),
              ElementsAre(op::Constant()))
      << computation->ToString();
}

// Loops that contain send/recv nodes can't be simplified; the loop structure
// around send/recv nodes must be preserved.
TEST_F(WhileLoopSimplifierTest, LoopWithSendNotSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloComputation* computation = m->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* token = while_body->AddInstruction(HloInstruction::CreateToken());
  auto* send = while_body->AddInstruction(HloInstruction::CreateSend(
      while_body->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true))),
      token,
      /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateSendDone(send));
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, LoopWithRecvNotSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloComputation* computation = m->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* token = while_body->AddInstruction(HloInstruction::CreateToken());
  auto* recv = while_body->AddInstruction(
      HloInstruction::CreateRecv(ShapeUtil::MakeShape(F32, {1}), token,
                                 /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateRecvDone(recv));
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// We can't simplify loops whose bodies contain infeed or other side-effecting
// instructions.
TEST_F(WhileLoopSimplifierTest, LoopWithInfeedSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloComputation* computation = m->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto token = while_body->AddInstruction(HloInstruction::CreateToken());
  while_body->AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(F32, {1}), token, "config"));
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// We don't simplify trip-count-1 loops whose *conditions* contain infeed or
// other side-effecting instructions, because simplifying such a loop always
// removes its condition!
TEST_F(WhileLoopSimplifierTest, LoopWithInfeedInCondNotSimplified) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloComputation* computation = m->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_cond = while_op->while_condition();
  auto token = while_cond->AddInstruction(HloInstruction::CreateToken());
  while_cond->AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(F32, {1}), token, "config"));
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// A non-tuple shaped loop shouldn't be simplified or crash the compiler.
TEST_F(WhileLoopSimplifierTest, NonTupleShapedLoopNotSimplified) {
  const std::string hlo_string = R"(
 HloModule NonTupleShapedLoop
 NonTupleShapedLoop.body {
   loop_var.1 = s32[] parameter(0)
   constant.1 = s32[] constant(-1)
   ROOT add = s32[] add(s32[] loop_var.1, s32[] constant.1)
 }
 NonTupleShapedLoop.condition {
   loop_var = s32[] parameter(0)
   constant = s32[] constant(100)
   ROOT less-than = pred[] compare(s32[] loop_var, s32[] constant), direction=LT
 }
 ENTRY INonTupleShapedLoop {
   constant.2 = s32[] constant(42)
   ROOT while = s32[] while(s32[] constant.2),
     condition=NonTupleShapedLoop.condition,
     body=NonTupleShapedLoop.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// A while loop that does nothing else besides swapping tuple elements
// can't be simplified as the result of the swapping is visible to users of the
// loop.
TEST_F(WhileLoopSimplifierTest, LoopSwappingTupleElementsNotSimplified) {
  const std::string hlo_string = R"(
  HloModule SwappingTupleElements
  SwappingTupleElements.body {
    loop_var = (s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[]) loop_var),index=1
    get-tuple-element.1 = s32[] get-tuple-element((s32[], s32[]) loop_var),
      index=0
    ROOT tuple = (s32[], s32[]) tuple(s32[] get-tuple-element,
      s32[] get-tuple-element.1)
  }
  SwappingTupleElements.always_true {
   param = (s32[], s32[]) parameter(0)
   ROOT constant = pred[] constant(true)
  }
  ENTRY SwappingTupleElements {
   x = s32[] parameter(0)
   y = s32[] parameter(1)
   tuple.1 = (s32[], s32[]) tuple(s32[] x, s32[] y)
   ROOT while = (s32[], s32[]) while((s32[], s32[]) tuple.1),
     condition=SwappingTupleElements.always_true,
     body=SwappingTupleElements.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// Construct a loop where we assign a constant to tuple element 0 in each
// iteration.  We can't eliminate tuple element 0, even though we never use its
// value.
TEST_F(WhileLoopSimplifierTest,
       LoopWithUnusedButModifiedTupleElementNotSimplified) {
  const std::string hlo_string = R"(
  HloModule UnusedButModifiedTupleElement
  UnusedButModifiedTupleElement.body {
    loop_var = (s32[]) parameter(0)
    constant.1 = s32[] constant(1)
    ROOT tuple = (s32[]) tuple(s32[] constant.1)
  }
  UnusedButModifiedTupleElement.always_true {
    param = (s32[]) parameter(0)
   ROOT  constant = pred[] constant(true)
  }
  ENTRY  UnusedButModifiedTupleElement {
    constant.2 = s32[] constant(0)
    tuple.1 = (s32[]) tuple(s32[]  constant.2)
    ROOT while = (s32[]) while((s32[]) tuple.1),
      condition=UnusedButModifiedTupleElement.always_true,
      body=UnusedButModifiedTupleElement.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// Construct a loop where we assign a constant to tuple element 1 in each
// iteration.  We can eliminate tuple element 1 if it's unused both inside and
// outside the loop.
TEST_F(WhileLoopSimplifierTest,
       LoopWithUnusedOutsideLoopButModifiedTupleElementSimplified) {
  const std::string hlo_string = R"(
  HloModule UnusedButModifiedTupleElement
  UnusedButModifiedTupleElement.body {
    loop_var = (s32[], s32[]) parameter(0)
    constant.1 = s32[] constant(1)
    ROOT tuple = (s32[], s32[]) tuple(s32[] constant.1, constant.1)
  }
  UnusedButModifiedTupleElement.cond {
    param = (s32[], s32[]) parameter(0)
    gte.cond = s32[] get-tuple-element(param), index=0
    constant.3 = s32[] constant(1)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  UnusedButModifiedTupleElement {
    constant.2 = s32[] constant(0)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.2)
    while = (s32[], s32[]) while(tuple.1),
      condition=UnusedButModifiedTupleElement.cond,
      body=UnusedButModifiedTupleElement.body
    ROOT gte = s32[] get-tuple-element(while), index=0
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  auto m_while = AllOf(op::While(), op::Shape("(s32[])"));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::GetTupleElement(m_while));
}

// Nothing to simplify in a while loop whose tuple has 0 elements.
TEST_F(WhileLoopSimplifierTest, LoopWithEmptyTupleNotSimplified) {
  const std::string hlo_string = R"(
  HloModule EmptyTuple
  EmptyTuple.body {
    loop_var = () parameter(0)
    ROOT  tuple = () tuple()
  }
  EmptyTuple.always_true {
   param = () parameter(0)
   ROOT constant = pred[] constant(true)
  }
  ENTRY EmptyTuple {
   tuple.1 = () tuple()
   ROOT while = () while(() tuple.1), condition=EmptyTuple.always_true,
     body=EmptyTuple.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// While loop where one tuple element is used twice in the body, and thus can't
// be simplified away.
TEST_F(WhileLoopSimplifierTest, LoopWithElemUsedTwiceNotSimplified) {
  const std::string hlo_string = R"(
  HloModule ElemUsedTwice
  ElemUsedTwice.body {
    param0 = (s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[]) param0), index=0
    ROOT tuple = (s32[], s32[]) tuple(s32[] get-tuple-element,
      s32[] get-tuple-element)
  }
  ElemUsedTwice.always_true {
    param = (s32[], s32[]) parameter(0)
    ROOT constant = pred[] constant(true)
  }
  ENTRY ElemUsedTwice {
   x = s32[] parameter(0)
   y = s32[] parameter(1)
   tuple.1 = (s32[], s32[]) tuple(s32[] x, s32[] y)
   ROOT while = (s32[], s32[]) while((s32[], s32[]) tuple.1),
     condition=ElemUsedTwice.always_true, body=ElemUsedTwice.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

// This while loop has three tuple elements.  Element 0 is unused and should be
// removed. Element 1 is used by the loop body, and element 2 is used by the
// loop condition; these two should stay.
TEST_F(WhileLoopSimplifierTest, RemoveUnusedLoopOperands) {
  const std::string hlo_string = R"(
  HloModule RemoveUnusedOperands
  RemoveUnusedOperands.body {
    loop_var = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element((s32[], s32[],
      s32[]) loop_var), index=0
    get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[],
      s32[]) loop_var), index=1
    constant.1 = s32[] constant(1)
    add = s32[] add(s32[] get-tuple-element.2, s32[] constant.1)
    get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[])
      loop_var), index=2
    ROOT tuple = (s32[], s32[], s32[]) tuple(s32[] get-tuple-element.1,
      s32[] add, s32[] get-tuple-element.3)
  }
  RemoveUnusedOperands.loop_condition {
    constant.2 = s32[] constant(0)
    param0 = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[], s32[]) param0),
      index=2
    ROOT equal-to = pred[] compare(s32[] constant.2, s32[] get-tuple-element), direction=EQ
  }
  ENTRY RemoveUnusedOperands {
    x = s32[] parameter(0)
    constant.3 = s32[] constant(0)
    y = s32[] parameter(1)
    tuple.1 = (s32[], s32[], s32[]) tuple(s32[] x, s32[] constant.3,
      s32[] y)
    ROOT while = (s32[], s32[], s32[]) while((s32[], s32[], s32[]) tuple.1),
      condition=RemoveUnusedOperands.loop_condition,
      body=RemoveUnusedOperands.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());

  // The original while instruction is still left in the module as a dead
  // instruction, find a while instruction with a different name as the new
  // while instruction.
  const auto& instrs = m->entry_computation()->instructions();
  HloInstruction* new_while_op =
      *absl::c_find_if(instrs, [&](const HloInstruction* instr) {
        return (instr->opcode() == HloOpcode::kWhile &&
                instr->name() != "while");
      });

  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  EXPECT_TRUE(
      ShapeUtil::Equal(new_while_op->shape(),
                       ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32})))
      << ShapeUtil::HumanString(new_while_op->shape());
  EXPECT_THAT(
      new_while_op->while_body()->root_instruction(),
      op::Tuple(
          op::Add(op::GetTupleElement(op::Parameter(0), /*tuple_index=*/0),
                  op::Constant()),
          op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1)));

  EXPECT_THAT(new_while_op->while_condition()->root_instruction(),
              op::Eq(op::Constant(),
                     op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1)));
}

// Check that we can remove unused loop operands even if the loop contains a
// side-effecting instruction.
TEST_F(WhileLoopSimplifierTest,
       RemoveUnusedLoopOperandsDespiteSideEffectingOps) {
  const std::string hlo_string = R"(
  HloModule RemoveUnusedOperands
  body {
    loop_var = (s32[]) parameter(0)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    token0 = token[] after-all()
    unused = ((s32[], pred[]), token[]) infeed(token0)
    ROOT tuple = (s32[]) tuple(gte0)
  }
  cond {
    loop_var = (s32[]) parameter(0)
    ROOT constant = pred[] constant(true)
  }
  ENTRY RemoveUnusedOperands {
    x = s32[] parameter(0)
    tuple.1 = (s32[]) tuple(s32[] x)
    ROOT while = (s32[]) while((s32[]) tuple.1),
      condition=cond, body=body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());

  // The original while instruction is still left in the module as a dead
  // instruction, find a while instruction with a different name as the new
  // while instruction.
  const auto& instrs = m->entry_computation()->instructions();
  HloInstruction* new_while_op =
      *absl::c_find_if(instrs, [&](const HloInstruction* instr) {
        return (instr->opcode() == HloOpcode::kWhile &&
                instr->name() != "while");
      });
  EXPECT_TRUE(ShapeUtil::IsEmptyTuple(new_while_op->shape()))
      << new_while_op->shape().ToString();
}

TEST_F(WhileLoopSimplifierTest, LoopWithNonTupleBodyShapeNotSimplified) {
  const std::string hlo_string = R"(
  HloModule BodyHasNonTupleRoot
  BodyHasNonTupleRoot.passthrough {
    ROOT param = (s32[], s32[]) parameter(0)
  }
  BodyHasNonTupleRoot.always_true {
    param.1 = (s32[], s32[]) parameter(0)
    ROOT constant = pred[] constant(true)
  }
  ENTRY BodyHasNonTupleRoot {
    init_value = (s32[], s32[]) parameter(0)
    ROOT while = (s32[], s32[]) while((s32[], s32[]) init_value),
      condition=BodyHasNonTupleRoot.always_true,
      body=BodyHasNonTupleRoot.passthrough
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithNonTupleBodyRootInstructionNotSimplified) {
  const std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT custom-call = (s32[], s32[3]{0}) custom-call(add, multiply),
      custom_call_target="x"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(44)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, LoopWithArrayConstantNotSimplified) {
  const std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    add.2 = s32[3]{0} add(get-tuple-element.2, get-tuple-element.3)
    ROOT tuple = (s32[], s32[3]{0}, s32[3]{0}) tuple(add, add.2, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(47)
    ROOT less-than = pred[] compare(get-tuple-element.4, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}, s32[3]{0}) tuple(constant.3, constant.4, constant.4)
    ROOT while = (s32[], s32[3]{0}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_FALSE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, FlattenNestedTuple) {
  const std::string hlo_string = R"(
  HloModule Test
  Body {
    param = ((s32[1]), (s32[2], s32[3], (s32[4]))) parameter(0)
    ta = (s32[1]) get-tuple-element(param), index=0
    a = s32[1] get-tuple-element(ta), index=0
    a.1 = s32[1] add(a, a)
    tbcd = (s32[2], s32[3], (s32[4])) get-tuple-element(param), index=1
    ROOT tuple = ((s32[1]), (s32[2], s32[3], (s32[4]))) tuple(ta, tbcd)
  }
  Cond {
    param = ((s32[1]), (s32[2], s32[3], (s32[4]))) parameter(0)
    ROOT cond = pred[] constant(true)
  }
  ENTRY Loop {
    a = s32[1] constant({0})
    b = s32[2] constant({0,1})
    c = s32[3] constant({0,1,2})
    d = s32[4] constant({0,1,2,3})
    ta = (s32[1]) tuple(a)
    td = (s32[4]) tuple(d)
    tbcd = (s32[2], s32[3], (s32[4])) tuple(b, c, td)
    init = ((s32[1]), (s32[2], s32[3], (s32[4]))) tuple(ta, tbcd)
    ROOT while = ((s32[1]), (s32[2], s32[3], (s32[4]))) while(init),
      condition=Cond, body=Body
  })";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  // DCE away the old loop so there's just one while loop in the module, making
  // it easy to find.
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());

  HloInstruction* new_while = FindFirstWhile(m.get());
  Shape flat_tuple =
      ParseShape("(s32[1], s32[2], s32[3], s32[4])").ValueOrDie();
  SCOPED_TRACE(m->ToString());
  EXPECT_TRUE(ShapeUtil::Equal(new_while->shape(), flat_tuple));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->root_instruction()->shape(), flat_tuple));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->parameter_instruction(0)->shape(), flat_tuple));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_condition()->parameter_instruction(0)->shape(),
      flat_tuple));
  EXPECT_TRUE(ShapeUtil::Equal(
      m->entry_computation()->root_instruction()->shape(),
      ParseShape("((s32[1]), (s32[2], s32[3], (s32[4])))").ValueOrDie()));
}

// Edge-case: All elements of the loop carry are constants which can be removed,
// leaving us with a nullary loop.  This is a special case, we just replace the
// loop with its init.
TEST_F(WhileLoopSimplifierTest, OnlyConstantsInLoopCarry) {
  const std::string hlo_string = R"(
  HloModule Test
  Body {
    param = (s32[1]) parameter(0)
    a = s32[1] constant({0})
    ROOT tuple = (s32[1]) tuple(a)
  }
  Cond {
    param = (s32[1]) parameter(0)
    ROOT cond = pred[] constant(true)
  }
  ENTRY Loop {
    a = s32[1] constant({0})
    init = (s32[1]) tuple(a)
    ROOT while = (s32[1]) while(init), condition=Cond, body=Body
  })";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, RemoveConstantFromLoopCarry) {
  const std::string hlo_string = R"(
  HloModule Test
  Body {
    param = (s32[1], s32[2], s32[3]) parameter(0)
    a = s32[1] get-tuple-element(param), index=0
    a.1 = s32[1] add(a, a)
    b = s32[2] constant({1,1})
    c = s32[3] constant({10,10,10})
    ROOT tuple = (s32[1], s32[2], s32[3]) tuple(a.1, b, c)
  }
  Cond {
    param = (s32[1], s32[2], s32[3]) parameter(0)
    /* Use each tuple element.  The verifier will then ensure that if any of
     * these get modified, they're replaced with values of the correct shape. */
    a = s32[1] get-tuple-element(param), index=0
    b = s32[2] get-tuple-element(param), index=1
    c = s32[3] get-tuple-element(param), index=2
    ROOT cond = pred[] constant(true)
  }
  ENTRY Loop {
    /* Only `b` should be simplified away.  `a` is not a constant within the
     * loop, and `c`'s value changes depending on whether we run 0 or 1
     * iterations of the loop. */
    a = s32[1] constant({0})
    b = s32[2] constant({1,1})
    c = s32[3] constant({2,2,2})
    init = (s32[1], s32[2], s32[3]) tuple(a,b,c)
    ROOT while = (s32[1], s32[2], s32[3]) while(init),
      condition=Cond, body=Body
  })";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  // DCE away the old loop so there's just one while loop in the module, making
  // it easy to find.
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  // Run the tuple simplifier to make the resulting HLO a bit easier to check.
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());

  HloInstruction* new_while = FindFirstWhile(m.get());
  Shape new_while_shape = ParseShape("(s32[1], s32[3])").ValueOrDie();
  EXPECT_TRUE(ShapeUtil::Equal(new_while->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->root_instruction()->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->parameter_instruction(0)->shape(),
      new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_condition()->parameter_instruction(0)->shape(),
      new_while_shape));
  EXPECT_TRUE(
      ShapeUtil::Equal(m->entry_computation()->root_instruction()->shape(),
                       ParseShape("(s32[1], s32[2], s32[3])").ValueOrDie()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::Tuple(_, op::Constant(), _));
}

const char* const kSimpleMergeInductionVariablesModule = R"(
  HloModule Test
  Body {
    param = (TYPE[], TYPE[], TYPE[]) parameter(0)

    a = TYPE[] get-tuple-element(param), index=0
    one = TYPE[] constant(1)
    a1 = TYPE[] add(a, one)

    b = TYPE[] get-tuple-element(param), index=1
    negone = TYPE[] constant(-1)
    b1 = TYPE[] add(b, negone)

    c = TYPE[] add(a, b)

    ROOT tuple = (TYPE[], TYPE[], TYPE[]) tuple(a1,b1,c)
  }
  Cond {
    param = (TYPE[], TYPE[], TYPE[]) parameter(0)
    a = TYPE[] get-tuple-element(param), index=0
    b = TYPE[] get-tuple-element(param), index=1
    sum = TYPE[] power(a, b)
    ten = TYPE[] constant(10)
    ROOT cond = pred[] compare(sum, ten), direction=LT
  }
  ENTRY Loop {
    a = TYPE[] constant(10)
    b = TYPE[] constant(100)
    c = TYPE[] constant(0)
    init = (TYPE[], TYPE[], TYPE[]) tuple(a,b,c)
    while = (TYPE[], TYPE[], TYPE[]) while(init), condition=Cond, body=Body

    a1 = TYPE[] get-tuple-element(while), index=0
    b1 = TYPE[] get-tuple-element(while), index=1
    c1 = TYPE[] get-tuple-element(while), index=2
    sum = TYPE[] add(a1, b1)
    ROOT sum.1 = TYPE[] add(sum, c1)
  })";

TEST_F(WhileLoopSimplifierTest, MergeInductionVariables_Simple) {
  std::string hlo_string = absl::StrReplaceAll(
      kSimpleMergeInductionVariablesModule, {{"TYPE", "s32"}});

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  // DCE away the old loop so there's just one while loop in the module, making
  // it easy to find, and run the tuple simplifier to make the resulting HLO
  // easier to check.
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());

  HloInstruction* new_while = FindFirstWhile(m.get());
  // We should have added a new loop counter for s32[] to the end of the tuple.
  SCOPED_TRACE(m->ToString());
  Shape new_while_shape =
      ParseShape("(s32[], s32[], s32[], s32[])").ValueOrDie();
  EXPECT_TRUE(ShapeUtil::Equal(new_while->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->root_instruction()->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->parameter_instruction(0)->shape(),
      new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_condition()->parameter_instruction(0)->shape(),
      new_while_shape));

  EXPECT_THAT(new_while->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(), 0),
                        op::GetTupleElement(op::Parameter(), 1), op::Add(),
                        op::Add(op::GetTupleElement(op::Parameter(), 3),
                                op::Constant())));
  EXPECT_THAT(new_while->while_condition()->root_instruction(),
              op::Lt(op::Power(op::Add(), op::Add()), op::Constant()));
}

// We shouldn't merge S16 induction variables; we can't create constants of this
// type because S16 literals are not implemented.
TEST_F(WhileLoopSimplifierTest, MergeInductionVariables_SkipS16) {
  std::string hlo_string = absl::StrReplaceAll(
      kSimpleMergeInductionVariablesModule, {{"TYPE", "s16"}});
  EXPECT_FALSE(
      WhileLoopSimplifier()
          .Run(ParseAndReturnVerifiedModule(hlo_string).ValueOrDie().get())
          .ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, RemoveRepeatedParams) {
  const std::string hlo_string = R"(
  HloModule SwappingTupleElements

  SwappingTupleElements.body {
    loop_var = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element(loop_var), index=0
    get-tuple-element.1 = s32[] get-tuple-element(loop_var), index=1
    get-tuple-element.2 = s32[] get-tuple-element(loop_var), index=2
    y = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[]) tuple(s32[] get-tuple-element, y,
      s32[] get-tuple-element.2)
  }

  SwappingTupleElements.always_true {
   param = (s32[], s32[], s32[]) parameter(0)
   get-tuple-element = s32[] get-tuple-element(param), index=0
   get-tuple-element.1 = s32[] get-tuple-element(param), index=1
   ROOT less-than = pred[] compare(get-tuple-element, get-tuple-element.1), direction=LT
  }

  ENTRY SwappingTupleElements {
   x = s32[] parameter(0)
   y = s32[] parameter(1)
   tuple.1 = (s32[], s32[], s32[]) tuple(s32[] x, s32[] y, s32[] x)
   ROOT while = (s32[], s32[], s32[]) while(tuple.1),
     condition=SwappingTupleElements.always_true,
     body=SwappingTupleElements.body
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  EXPECT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  HloInstruction* new_while = FindFirstWhile(m.get());
  Shape new_while_shape = ParseShape("(s32[], s32[])").ValueOrDie();
  EXPECT_TRUE(ShapeUtil::Equal(new_while->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->root_instruction()->shape(), new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_body()->parameter_instruction(0)->shape(),
      new_while_shape));
  EXPECT_TRUE(ShapeUtil::Equal(
      new_while->while_condition()->parameter_instruction(0)->shape(),
      new_while_shape));
}

// A group of elements are inter-dependent, but unused as by the output.
TEST_F(WhileLoopSimplifierTest, LoopWithUnusedGroupSimplified) {
  const std::string hlo_string = R"(
  HloModule LoopWithUnusedGroupSimplified
  LoopWithUnusedGroupSimplified.body {
    loop_var = (s32[], s32[], s32[]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=1
    gte1 = s32[] get-tuple-element(loop_var), index=2
    add = s32[] add(gte0, gte1)
    ROOT tuple = (s32[], s32[], s32[]) tuple(constant.1, add, add)
  }
  LoopWithUnusedGroupSimplified.cond {
    param = (s32[], s32[], s32[]) parameter(0)
    gte.cond = s32[] get-tuple-element(param), index=0
    constant.3 = s32[] constant(1)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  LoopWithUnusedGroupSimplified {
    constant.2 = s32[] constant(0)
    tuple.1 = (s32[], s32[], s32[]) tuple(constant.2, constant.2, constant.2)
    while = (s32[], s32[], s32[]) while(tuple.1),
      condition=LoopWithUnusedGroupSimplified.cond,
      body=LoopWithUnusedGroupSimplified.body
    ROOT gte = s32[] get-tuple-element(while), index=0
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  auto m_while = AllOf(op::While(), op::Shape("(s32[])"));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              op::GetTupleElement(m_while));
}

// An element is not a passthrough, but it's not used by other elements.
TEST_F(WhileLoopSimplifierTest, LoopWithUnusedNonPassthroughElementSimplified) {
  const std::string hlo_string = R"(
  HloModule LoopWithUnusedNonPassthroughElementSimplified
  LoopWithUnusedNonPassthroughElementSimplified.body {
    loop_var = (s32[], s32[], s32[]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=1
    gte1 = s32[] get-tuple-element(loop_var), index=2
    add = s32[] add(gte0, gte1)
    add2 = s32[] add(gte0, gte0)
    ROOT tuple = (s32[], s32[], s32[]) tuple(constant.1, add2, add)
  }
  LoopWithUnusedNonPassthroughElementSimplified.cond {
    param = (s32[], s32[], s32[]) parameter(0)
    gte.cond = s32[] get-tuple-element(param), index=0
    constant.3 = s32[] constant(1)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  LoopWithUnusedNonPassthroughElementSimplified {
    constant.2 = s32[] constant(0)
    tuple.1 = (s32[], s32[], s32[]) tuple(constant.2, constant.2, constant.2)
    while = (s32[], s32[], s32[]) while(tuple.1),
      condition=LoopWithUnusedNonPassthroughElementSimplified.cond,
      body=LoopWithUnusedNonPassthroughElementSimplified.body
    gte2 = s32[] get-tuple-element(while), index=0
    gte3 = s32[] get-tuple-element(while), index=1
    ROOT tuple.2 = (s32[], s32[]) tuple(gte2, gte3)
  }
  )";

  auto m = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  ASSERT_TRUE(WhileLoopSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(TupleSimplifier().Run(m.get()).ok());
  EXPECT_TRUE(HloDCE().Run(m.get()).ok());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              AllOf(op::While(), op::Shape("(s32[], s32[])")));
}

}  // namespace
}  // namespace xla
