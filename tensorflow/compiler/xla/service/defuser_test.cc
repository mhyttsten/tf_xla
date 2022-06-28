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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdefuser_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdefuser_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdefuser_testDTcc() {
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

#include "tensorflow/compiler/xla/service/defuser.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class DefuserTest : public HloTestBase {
 protected:
  // Returns the number of fusion instructions in the module.
  int FusionCount(const HloModule* m) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdefuser_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/defuser_test.cc", "FusionCount");

    int count = 0;
    for (HloComputation* computation : m->computations()) {
      if (computation->IsFusionComputation()) {
        count++;
      }
    }
    return count;
  }

  Defuser defuser_;
  const Shape shape_ = ShapeUtil::MakeShape(F32, {2, 2});
};

TEST_F(DefuserTest, NoFusionInstruction) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  m->AddEntryComputation(builder.Build());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_FALSE(defuser_.Run(m.get()).ValueOrDie());
}

TEST_F(DefuserTest, TrivialFusionInstructionAsRoot) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).ValueOrDie());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Parameter()));
}

TEST_F(DefuserTest, TrivialFusionInstructionNotAsRoot) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Fusion()));

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).ValueOrDie());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Negate(op::Add(op::Parameter(), op::Parameter())));
}

TEST_F(DefuserTest, NonTrivialFusionInstruction) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto param3 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape_, "p2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kSubtract, add, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kMultiply, sub, param3));
  auto div = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kDivide, mul, param3));
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction(
      {add2, constant, div, mul, sub, negate, add},
      HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).ValueOrDie());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, MultipleFusionInstructions) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto param3 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape_, "p2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kSubtract, add, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kMultiply, sub, param3));
  auto div = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kDivide, mul, param3));
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add2, constant, div, mul},
                                       HloInstruction::FusionKind::kLoop);
  computation->CreateFusionInstruction({sub, negate, add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).ValueOrDie());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, NestedFusionInstructions) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = m->AddEntryComputation(builder.Build());
  auto outer_fusion = computation->CreateFusionInstruction(
      {negate, add}, HloInstruction::FusionKind::kLoop);
  HloInstruction* fused_negate = outer_fusion->fused_expression_root();
  ASSERT_EQ(fused_negate->opcode(), HloOpcode::kNegate);
  outer_fusion->fused_instructions_computation()->CreateFusionInstruction(
      {fused_negate}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).ValueOrDie());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Add()));
}

}  // namespace
}  // namespace xla
