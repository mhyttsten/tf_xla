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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_util_testDTcc() {
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

#include "tensorflow/compiler/xla/service/while_util.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class WhileUtilTest : public HloTestBase {
 protected:
  StatusOr<std::unique_ptr<VerifiedHloModule>> GetParsedModule(
      HloComputation** entry_computation, HloInstruction** param0,
      HloInstruction** param1, HloInstruction** param2) {
    const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
}

while_condition {
  p_cond = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = f32[32,32]{1,0} parameter(0)
  p_entry_1 = s32[32,32]{1,0} parameter(1)
  p_entry_2 = s64[32,32]{1,0} parameter(2)
  while_init = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p_entry_0, p_entry_0)
  ROOT while = (f32[32,32]{1,0}, f32[32,32]{1,0}) while(while_init), condition=while_condition, body=while_body
}
)";

    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));

    *entry_computation = module->entry_computation();
    *param0 = (*entry_computation)->parameter_instruction(0);
    *param1 = (*entry_computation)->parameter_instruction(1);
    *param2 = (*entry_computation)->parameter_instruction(2);

    return std::move(module);
  }
};

TEST_F(WhileUtilTest, MakeZeroInstructionsLiveOp) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetParsedModule(&entry_computation, &param0, &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  TF_ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, /*instructions=*/{}));

  HloInstruction* new_while_instr = make_live_in_result.new_while_instr;

  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::Tuple(op::GetTupleElement(::testing::Eq(new_while_instr), 0),
                op::GetTupleElement(::testing::Eq(new_while_instr), 1)));

  auto param_reconstructed =
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::GetTupleElement(op::Parameter(0), 1));

  EXPECT_THAT(new_while_instr->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(param_reconstructed, 0),
                        op::GetTupleElement(param_reconstructed, 1)));
}

TEST_F(WhileUtilTest, MakeTwoInstructionsLive) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetParsedModule(&entry_computation, &param0, &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  TF_ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr,
                                        /*instructions=*/{param0, param1}));

  HloInstruction* new_while_instr = make_live_in_result.new_while_instr;

  XLA_VLOG_LINES(3, module->ToString());

  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::Tuple(op::GetTupleElement(::testing::Eq(new_while_instr), 0),
                op::GetTupleElement(::testing::Eq(new_while_instr), 1)));

  auto first_half_param_reconstructed =
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::GetTupleElement(op::Parameter(0), 1));

  EXPECT_THAT(new_while_instr->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(first_half_param_reconstructed, 0),
                        op::GetTupleElement(first_half_param_reconstructed, 1),
                        op::GetTupleElement(op::Parameter(0), 2),
                        op::GetTupleElement(op::Parameter(0), 3)));
}

TEST_F(WhileUtilTest, GetInvariantGTEsForWhileBody) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  ROOT constant = pred[] constant(true)
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* while_body = module->GetComputationWithName("body");

  ASSERT_NE(while_body, nullptr)
      << "Expected exactly one while_body computation";

  std::vector<HloInstruction*> gte_list =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  ASSERT_EQ(gte_list.size(), 1);
  EXPECT_EQ((*gte_list.begin())->name(), "gte.0");
}

TEST_F(WhileUtilTest, AlwaysRemovePreviousWhileBody) {
  const char* const hlo_string = R"(
HloModule WhileWithSideEffects

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  token0 = token[] after-all()
  infeed = (pred[], token[]) infeed(token0)
  ROOT condition = pred[] get-tuple-element(infeed), index=0
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  to_make_live_in = f32[100] parameter(1)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* main = module->GetComputationWithName("main");
  HloInstruction* while_instr = main->root_instruction();
  HloInstruction* to_make_live_in = main->parameter_instruction(1);

  TF_ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr,
                                        /*instructions=*/{to_make_live_in}));

  auto is_while = [](const HloInstruction* instr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSwhile_util_testDTcc mht_0(mht_0_v, 381, "", "./tensorflow/compiler/xla/service/while_util_test.cc", "lambda");

    return instr->opcode() == HloOpcode::kWhile;
  };
  EXPECT_EQ(absl::c_count_if(main->instructions(), is_while), 1);
}
}  // namespace
}  // namespace xla
