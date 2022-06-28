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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_schedule_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_schedule_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_schedule_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_schedule.h"

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloScheduleTest : public HloTestBase {};

TEST_F(HloScheduleTest, UpdateScheduleUnchangedModule) {
  // Updating the schedule of an unchanged HLO module should not affect the
  // schedule at all.
  const std::string module_str = R"(
HloModule UpdateScheduleUnchanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));
  const auto& entry_schedule =
      schedule.sequence(module->entry_computation()).instructions();

  EXPECT_EQ(entry_schedule.size(), 6);

  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(entry_schedule,
            schedule.sequence(module->entry_computation()).instructions());
}

TEST_F(HloScheduleTest, UpdateScheduleWithNewInstructions) {
  // Add some additional instructions to a module and verify the schedule can be
  // updated.
  const std::string module_str = R"(
HloModule UpdateScheduleWithNewInstructions

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  HloComputation* entry = module->entry_computation();
  const Shape shape = entry->root_instruction()->shape();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  HloInstruction* sub = entry->AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, constant, entry->root_instruction()));
  entry->set_root_instruction(sub);

  auto in_schedule = [&](const HloInstruction* hlo) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_schedule_testDTcc mht_0(mht_0_v, 273, "", "./tensorflow/compiler/xla/service/hlo_schedule_test.cc", "lambda");

    return absl::c_linear_search(schedule.sequence(entry).instructions(), hlo);
  };

  EXPECT_EQ(schedule.sequence(entry).size(), 6);
  EXPECT_FALSE(in_schedule(constant));
  EXPECT_FALSE(in_schedule(sub));

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 8);
  EXPECT_TRUE(in_schedule(constant));
  EXPECT_TRUE(in_schedule(sub));
}

TEST_F(HloScheduleTest, UpdateScheduleWithAddedAndDeletedInstruction) {
  // Add and delete some instructions from a module and verify that the schedule
  // can be updated successfully.
  const std::string module_str = R"(
HloModule UpdateScheduleWithAddedAndDeletedInstruction

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  // Set the entry root to some expression containing just a parameter and a
  // constant.
  HloComputation* entry = module->entry_computation();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  HloInstruction* new_root = entry->AddInstruction(
      HloInstruction::CreateBinary(constant->shape(), HloOpcode::kSubtract,
                                   constant, entry->parameter_instruction(0)));
  entry->set_root_instruction(new_root);

  // DCE should remove everything but the parameters and the newly added code.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(entry).size(), 6);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 4);
}

TEST_F(HloScheduleTest, UpdateScheduleWithCompletelyReplacedModule) {
  // Completely replace a module with an entirely new set of instructions and
  // verify that the schedule can be updated successfully.
  const std::string module_str = R"(
HloModule UpdateScheduleWithCompletelyReplacedModule

ENTRY main {
  a = f32[] constant(42.0)
  b = f32[] constant(123.0)
  ROOT sum = f32[] add(a, b)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  // Replace the entry computation with the negation of a constant.
  HloComputation* entry = module->entry_computation();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* new_root = entry->AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  entry->set_root_instruction(new_root);

  // DCE the old instructions.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(entry).size(), 3);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 2);
}

TEST_F(HloScheduleTest, UpdateScheduleWithMultipleComputations) {
  // Create changes to more than one computation in an HLO module and verify
  // that the schedule can be updated.
  const std::string module_str = R"(
HloModule UpdateScheduleWithMultipleComputations

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %WhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(),
                                     /*pointer_size=*/sizeof(void*));
      }));

  const HloInstruction* xla_while =
      module->entry_computation()->root_instruction()->operand(0);
  HloComputation* body = xla_while->while_body();
  HloComputation* cond = xla_while->while_condition();

  // Negate the root of the cond.
  cond->set_root_instruction(cond->AddInstruction(
      HloInstruction::CreateUnary(ShapeUtil::MakeShape(PRED, {}),
                                  HloOpcode::kNot, cond->root_instruction())));

  // Replace the body with a computation which just passes through its
  // parameter.
  body->set_root_instruction(body->parameter_instruction(0));

  // DCE the dead code in the body.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(body).size(), 7);
  EXPECT_EQ(schedule.sequence(cond).size(), 4);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(body).size(), 1);
  EXPECT_EQ(schedule.sequence(cond).size(), 5);
}

TEST_F(HloScheduleTest, UpdateScheduleComputationRemoved) {
  // Remove computations from a module and verify the schedule can be updated.
  const std::string module_str = R"(
HloModule UpdateScheduleWithMultipleComputations

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %WhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(),
                                     /*pointer_size=*/sizeof(void*));
      }));

  HloInstruction* xla_while =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloInstruction* init = xla_while->mutable_operand(0);

  // Replace the while with its init value. The conditional and body
  // computations should then be dead.
  TF_ASSERT_OK(xla_while->ReplaceAllUsesWith(init));

  // DCE the dead code in the body.
  HloDCE dce;
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK(dce.Run(module.get()).status());
  ASSERT_EQ(module->computation_count(), 1);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());
}

}  // namespace
}  // namespace xla
