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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_subcomputation_unification_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_subcomputation_unification_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_subcomputation_unification_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {

class HloSubcomputationUnificationTest : public HloTestBase {
 protected:
  HloSubcomputationUnificationTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_subcomputation_unification_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/hlo_subcomputation_unification_test.cc", "HloSubcomputationUnificationTest");
}

  std::unique_ptr<HloComputation> CreateR0S32IdentityComputation() {
    auto builder = HloComputation::Builder("Identity");
    builder.AddInstruction(HloInstruction::CreateParameter(0, r0s32_, "x"));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> CreateR0S32AdditionComputation() {
    auto builder = HloComputation::Builder("Addition");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, r0s32_, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, r0s32_, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> CreateR1S32AdditionComputation(
      const Shape& shape) {
    auto builder = HloComputation::Builder("Addition");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, x, y));
    return builder.Build();
  }

  Shape r0s32_ = ShapeUtil::MakeShape(S32, {});
  Shape r0f32_ = ShapeUtil::MakeShape(S32, {});
  Shape r1s32_5_ = ShapeUtil::MakeShape(S32, {5});
  Shape r1s32_3_ = ShapeUtil::MakeShape(S32, {3});
};

TEST_F(HloSubcomputationUnificationTest, UnifyIdentities) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR0S32IdentityComputation());
  auto callee2 =
      module->AddEmbeddedComputation(CreateR0S32IdentityComputation());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(5)));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant}, callee2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  EXPECT_EQ(2, module->computation_count());
  EXPECT_EQ(x->to_apply(), y->to_apply());
}

TEST_F(HloSubcomputationUnificationTest, UnifyAdditions) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR0S32AdditionComputation());
  auto callee2 =
      module->AddEmbeddedComputation(CreateR0S32AdditionComputation());

  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(5)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(3)));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant1, constant2}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant1, constant2}, callee2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  EXPECT_EQ(2, module->computation_count());
  EXPECT_EQ(x->to_apply(), y->to_apply());
}

// Do not unify subcomputations with different parameter shapes.
TEST_F(HloSubcomputationUnificationTest, DifferentParameterShapes) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR1S32AdditionComputation(r1s32_5_));
  auto callee2 =
      module->AddEmbeddedComputation(CreateR1S32AdditionComputation(r1s32_3_));

  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1s32_5_, "param1"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1s32_3_, "param2"));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r1s32_5_, {param1, param1}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r1s32_3_, {param2, param2}, callee2));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {8}), {x, y}, 0));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  EXPECT_FALSE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
}

// Regression test for b/31466798. Checks that entry_computation is still valid
// after unification.
TEST_F(HloSubcomputationUnificationTest, TwoIdenticalComputations) {
  auto module = CreateNewVerifiedModule();
  for (int i = 0; i < 2; ++i) {
    HloComputation::Builder builder("pow");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0f32_, HloOpcode::kPower, x, y));
    if (i == 0) {
      module->AddEmbeddedComputation(builder.Build());
    } else {
      module->AddEntryComputation(builder.Build());
    }
  }

  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  EXPECT_EQ(1, module->computation_count());
  EXPECT_EQ(*module->computations().begin(), module->entry_computation());
}

}  // namespace xla
