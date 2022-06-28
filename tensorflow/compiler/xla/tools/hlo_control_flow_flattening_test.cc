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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flattening_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flattening_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flattening_testDTcc() {
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

#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using HloControlFlowFlatteningTest = HloTestBase;

constexpr int kDefaultMaxLoopCount = 1000;

TEST_F(HloControlFlowFlatteningTest, WhileRoot) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));

  auto body = while_op->while_body();
  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::Add(op::GetTupleElement(op::Parameter(0), 2),
                                op::Constant())));
}

TEST_F(HloControlFlowFlatteningTest, WhileConditionCallComputation) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition.called {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] custom-call(), custom_call_target="AllocateBuffer", custom_call_has_side_effect=true
    less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
    ROOT tuple.2 = (pred[]) tuple(less-than)
  }
  While.condition {
    loop_var.3 = (s32[], s32[3]{0}) parameter(0)
    call = (pred[]) call(loop_var.3), to_apply=While.condition.called
    ROOT get-tuple-element.4 = pred[] get-tuple-element(call), index=0
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  XLA_VLOG_LINES(3, "Loaded HLO module: " + module->ToString());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));

  auto body = while_op->while_body();
  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::Add(op::GetTupleElement(op::Parameter(0), 2),
                                op::Constant())));
}

TEST_F(HloControlFlowFlatteningTest, WhileRootScheduled) {
  absl::string_view hlo_string = R"(
  HloModule While, is_scheduled=true
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));
}

TEST_F(HloControlFlowFlatteningTest, WhileUser) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  FusedComputation {
    param = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(param), index=0
    get-tuple-element.5 = s32[3]{0} get-tuple-element(param), index=1
    broadcast = s32[3]{0} broadcast(get-tuple-element.4), dimensions={}
    ROOT add = s32[3]{0} add(broadcast, get-tuple-element.5)
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
    ROOT fusion = s32[3]{0} fusion(while), kind=kLoop, calls=FusedComputation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto fusion = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(fusion, op::Fusion(op::Tuple(op::GetTupleElement(while_op, 0),
                                           op::GetTupleElement(while_op, 1))));
}

TEST_F(HloControlFlowFlatteningTest, Infeed) {
  absl::string_view hlo_string = R"(
  HloModule Infeed
  ENTRY Infeed {
    after-all = token[] after-all()
    ROOT infeed = ((bf16[3]{0}, s32[12,5]{0,1}), token[]) infeed(after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::CustomCall(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, InfeedPreserveLayout) {
  absl::string_view hlo_string = R"(
  HloModule Infeed
  ENTRY Infeed {
    after-all = token[] after-all()
    ROOT infeed = ((bf16[3]{0}, s32[12,5]{0,1:T(8,128)}), token[]) infeed(after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Shape root_shape = module->entry_computation()->root_instruction()->shape();
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::CustomCall(), op::AfterAll()));
  EXPECT_EQ(tuple->shape(), root_shape);
}

TEST_F(HloControlFlowFlatteningTest, Outfeed) {
  absl::string_view hlo_string = R"(
  HloModule Outfeed
  ENTRY Outfeed {
    param = (bf16[3]{0}, s32[12,5]{0,1}) parameter(0)
    after-all = token[] after-all()
    ROOT outfeed = token[] outfeed(param, after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto custom_call = module->entry_computation()->root_instruction();
  EXPECT_THAT(custom_call, op::CustomCall(op::Parameter(0), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, AllReduce) {
  absl::string_view hlo_string = R"(
  HloModule AllReduce
  sum {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY AllReduce {
    param0 = f32[3]{0} parameter(0)
    param1 = f32[12,5]{0,1} parameter(1)
    ROOT all-reduce = (bf16[3]{0}, bf16[12,5]{0,1}) all-reduce(param0, param1), to_apply=sum, replica_groups={}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Parameter(1)));
}

TEST_F(HloControlFlowFlatteningTest, AllReduceStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule CRS

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    crs = f32[8]{0} all-reduce-start(input), replica_groups={}, to_apply=add
    ROOT done = f32[8]{0} all-reduce-done(crs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

TEST_F(HloControlFlowFlatteningTest, AllGather) {
  absl::string_view hlo_string = R"(
  HloModule AllGather

  ENTRY AllGather {
    input = f32[128,32]{0,1} parameter(0)
    ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, dimensions={1}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
}

TEST_F(HloControlFlowFlatteningTest, AllToAll) {
  absl::string_view hlo_string = R"(
  HloModule AllToAll

  ENTRY AllToAll {
    input = f32[128,32]{0,1} parameter(0)
    ROOT a2a = (f32[128,32]{0,1}) all-to-all(input), replica_groups={}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermute) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermute

  ENTRY CollectivePermute {
    input = f32[128,32]{0,1} parameter(0)
    ROOT root = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermuteInPlaceUpdate) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermuteInPlaceUpdate

  ENTRY CollectivePermuteInPlaceUpdate {
    input = f32[128,32]{0,1} parameter(0)
    constant = f32[] constant(1)
    output = f32[128,128]{0,1} broadcast(constant), dimensions={}
    constant.1 = s32[] constant(0)
    tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
    constant.2 = s32[] constant(64)
    tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
    ROOT root = f32[128,128]{0,1} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,32}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Broadcast(op::Constant()),
                             op::Tuple(op::Constant(), op::Constant()),
                             op::Tuple(op::Constant(), op::Constant())));
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermuteStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermuteStartAndDone

  ENTRY CollectivePermuteStartAndDone {
    input = f32[128,32]{0,1} parameter(0)
    collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,32]{0,1}, u32[], u32[]) collective-permute-start(input), source_target_pairs={{0,1},{1,2},{2,3}}
    ROOT collective-permute-done.1 = f32[128,32]{0,1} collective-permute-done(collective-permute-start.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

TEST_F(HloControlFlowFlatteningTest, Recv) {
  absl::string_view hlo_string = R"(
  HloModule Recv

  ENTRY %Recv () -> (f32[], token[]) {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
    ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
    %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::CustomCall(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, RecvHostTransfer) {
  absl::string_view hlo_string = R"(
  HloModule Recv

  ENTRY %Recv () -> (f32[], token[]) {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, is_host_transfer=true, sharding={maximal device=1}
    ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, is_host_transfer=true, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
    %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/true});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::CustomCall(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, Send) {
  absl::string_view hlo_string = R"(
  HloModule Send

  ENTRY %Send () -> token[] {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
    %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
    ROOT %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Constant(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, SendHostTransfer) {
  absl::string_view hlo_string = R"(
  HloModule Send

  ENTRY %Send () -> token[] {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
    %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, is_host_transfer=true, sharding={maximal device=0}, control-predecessors={%recv}
    ROOT %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, is_host_transfer=true, sharding={maximal device=0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/true});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Constant(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, AllGatherStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule AllGatherStartAndDone

  ENTRY AllGatherStartAndDone {
    %input = f32[8,256,256] parameter(0)
    %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
      f32[8,256,256] %input), replica_groups={{0,1}}, dimensions={0},
      metadata={op_type="AllGather" op_name="ag0"}
    ROOT %ag-done = f32[16,256,256] all-gather-done(
      (f32[8,256,256], f32[16,256,256]) %ag-start),
      metadata={op_type="AllGather" op_name="ag0"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

void CheckWhileBound(HloInstruction* while_op, int expected_bound) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flattening_testDTcc mht_0(mht_0_v, 811, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening_test.cc", "CheckWhileBound");

  auto* cond = while_op->while_condition();
  ASSERT_NE(cond, nullptr);
  auto* hlo_bound = cond->root_instruction()->operand(1);
  EXPECT_TRUE(hlo_bound->IsConstant());
  if (hlo_bound->IsConstant()) {
    EXPECT_TRUE(hlo_bound->literal().IsAll(expected_bound));
  }
}

TEST_F(HloControlFlowFlatteningTest, MaxOuterLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule NestedWhileComp

  InnerBody {
    constant.8 = pred[] constant(false)
    parameter.5 = (s32[], s32[]) parameter(0)
    get-tuple-element.6 = s32[] get-tuple-element(parameter.5), index=0
    constant.9 = s32[] constant(1)
    add.10 = s32[] add(get-tuple-element.6, constant.9)
    get-tuple-element.7 = s32[] get-tuple-element(parameter.5), index=1
    constant.11 = s32[] constant(1)
    add.12 = s32[] add(get-tuple-element.7, constant.11)
    ROOT tuple.13 = (s32[], s32[]) tuple(add.10, add.12)
  }

  InnerCond {
    parameter.15 = (s32[], s32[]) parameter(0)
    get-tuple-element.17 = s32[] get-tuple-element(parameter.15), index=1
    constant.18 = pred[] constant(false)
    get-tuple-element.16 = s32[] get-tuple-element(parameter.15), index=0
    inner_bound = s32[] constant(100)
    ROOT compare.20 = pred[] compare(get-tuple-element.16, inner_bound), direction=LT
  }

  OuterBody {
    constant.24 = pred[] constant(false)
    constant.25 = s32[] constant(0)
    parameter.22 = (s32[]) parameter(0)
    get-tuple-element.23 = s32[] get-tuple-element(parameter.22), index=0
    tuple.26 = (s32[], s32[]) tuple(constant.25, get-tuple-element.23)
    inner_while = (s32[], s32[]) while(tuple.26), condition=InnerCond, body=InnerBody
    get-tuple-element.28 = s32[] get-tuple-element(inner_while), index=0
    get-tuple-element.29 = s32[] get-tuple-element(inner_while), index=1
    tuple.30 = (s32[], s32[]) tuple(get-tuple-element.28, get-tuple-element.29)
    get-tuple-element.31 = s32[] get-tuple-element(tuple.30), index=0
    get-tuple-element.32 = s32[] get-tuple-element(tuple.30), index=1
    ROOT tuple.33 = (s32[]) tuple(get-tuple-element.32)
  }

  OuterCond {
    constant.37 = pred[] constant(false)
    parameter.35 = (s32[]) parameter(0)
    get-tuple-element.36 = s32[] get-tuple-element(parameter.35), index=0
    outer_bound = s32[] constant(1000)
    ROOT compare.39 = pred[] compare(get-tuple-element.36, outer_bound), direction=LT
  }

  ENTRY NestedWhileComp {
    constant.1 = pred[] constant(false)
    constant.2 = s32[] constant(0)
    tuple.3 = (s32[]) tuple(constant.2)
    outer_while = (s32[]) while(tuple.3), condition=OuterCond, body=OuterBody
    get-tuple-element.41 = s32[] get-tuple-element(outer_while), index=0
    tuple.42 = (s32[]) tuple(get-tuple-element.41)
    get-tuple-element.43 = s32[] get-tuple-element(tuple.42), index=0
    ROOT tuple.44 = (s32[]) tuple(get-tuple-element.43)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  constexpr int kWhileExecutionCount = 5;
  constexpr int kExistingInnerLoopCount = 100;
  constexpr int kMaxLoopCount = 10;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/kWhileExecutionCount,
      /*max_outer_loop_count=*/kMaxLoopCount});
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();

  auto* outer_while =
      module->entry_computation()->GetInstructionWithName("outer_while");
  ASSERT_NE(outer_while, nullptr);
  // Checks that the outer while loop has changed its loop bound.
  CheckWhileBound(outer_while, kMaxLoopCount);
  auto* while_body = outer_while->while_body();
  ASSERT_NE(while_body, nullptr);

  auto* inner_while = while_body->GetInstructionWithName("inner_while");
  ASSERT_NE(inner_while, nullptr);
  // Checks that the inner loop bound has not changed.
  CheckWhileBound(inner_while, kExistingInnerLoopCount);
}

TEST_F(HloControlFlowFlatteningTest, MatchLtUseInferedLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            100);
}

TEST_F(HloControlFlowFlatteningTest, MatchGtUseInferedLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(50)
    ROOT greater-than = pred[] compare(constant.2, get-tuple-element.3), direction=GT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            50);
}

TEST_F(HloControlFlowFlatteningTest, NotMatchEqUseDefaultLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT equal = pred[] compare(get-tuple-element.3, constant.2), direction=EQ
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            123);
}

}  // namespace
}  // namespace xla
