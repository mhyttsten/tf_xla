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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/topk_rewriter.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using TopkRewriterTest = HloTestBase;

std::string getComparator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/topk_rewriter_test.cc", "getComparator");

  return R"(
%compare {
  %p.1.lhs.8 = s32[] parameter(2)
  %p.1.rhs.9 = s32[] parameter(3)
  %p.0.lhs.6 = f32[] parameter(0)
  %bitcast-convert.11 = s32[] bitcast-convert(%p.0.lhs.6)
  %constant.15 = s32[] constant(0)
  %compare.16 = pred[] compare(%bitcast-convert.11, %constant.15), direction=LT
  %constant.10 = u32[] constant(2147483647)
  %bitcast-convert.12 = u32[] bitcast-convert(%p.0.lhs.6)
  %subtract.13 = u32[] subtract(%constant.10, %bitcast-convert.12)
  %bitcast-convert.14 = s32[] bitcast-convert(%subtract.13)
  %select.17 = s32[] select(%compare.16, %bitcast-convert.14,
                            %bitcast-convert.11)
  %p.0.rhs.7 = f32[] parameter(1)
  %bitcast-convert.19 = s32[] bitcast-convert(%p.0.rhs.7)
  %constant.23 = s32[] constant(0)
  %compare.24 = pred[] compare(%bitcast-convert.19, %constant.23), direction=LT
  %constant.18 = u32[] constant(2147483647)
  %bitcast-convert.20 = u32[] bitcast-convert(%p.0.rhs.7)
  %subtract.21 = u32[] subtract(%constant.18, %bitcast-convert.20)
  %bitcast-convert.22 = s32[] bitcast-convert(%subtract.21)
  %select.25 = s32[] select(%compare.24, %bitcast-convert.22,
                            %bitcast-convert.19)
  ROOT %compare.26 = pred[] compare(%select.17, %select.25), direction=GT
})";
}

std::string getConvertMaxComparator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/xla/service/topk_rewriter_test.cc", "getConvertMaxComparator");

  return R"(
%compare {
  %p.1.lhs.6 = s32[] parameter(2)
  %p.1.rhs.7 = s32[] parameter(3)
  %p.0.lhs.4 = f32[] parameter(0)
  %bitcast-convert = s32[] bitcast-convert(f32[] %p.0.lhs.4)
  %constant = s32[] constant(0)
  %compare = pred[] compare(s32[] %bitcast-convert, s32[] %constant), direction=LT
  %constant.1 = s32[] constant(2147483647)
  %convert = u32[] convert(s32[] %constant.1)
  %bitcast-convert.1 = u32[] bitcast-convert(f32[] %p.0.lhs.4)
  %subtract = u32[] subtract(u32[] %convert, u32[] %bitcast-convert.1)
  %bitcast-convert.2 = s32[] bitcast-convert(u32[] %subtract)
  %select = s32[] select(pred[] %compare, s32[] %bitcast-convert.2, s32[] %bitcast-convert)
  %p.0.rhs.5 = f32[] parameter(1)
  %bitcast-convert.3 = s32[] bitcast-convert(f32[] %p.0.rhs.5)
  %compare.1 = pred[] compare(s32[] %bitcast-convert.3, s32[] %constant), direction=LT
  %bitcast-convert.4 = u32[] bitcast-convert(f32[] %p.0.rhs.5)
  %subtract.1 = u32[] subtract(u32[] %convert, u32[] %bitcast-convert.4)
  %bitcast-convert.5 = s32[] bitcast-convert(u32[] %subtract.1)
  %select.1 = s32[] select(pred[] %compare.1, s32[] %bitcast-convert.5, s32[] %bitcast-convert.3)
  ROOT %compare.2 = pred[] compare(s32[] %select, s32[] %select.1), direction=GT
})";
}

std::string getComparatorNoIota() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriter_testDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/topk_rewriter_test.cc", "getComparatorNoIota");

  return R"(
%compare {
  %p.0.lhs.6 = f32[] parameter(0)
  %bitcast-convert.11 = s32[] bitcast-convert(%p.0.lhs.6)
  %constant.15 = s32[] constant(0)
  %compare.16 = pred[] compare(%bitcast-convert.11, %constant.15), direction=LT
  %constant.10 = u32[] constant(2147483647)
  %bitcast-convert.12 = u32[] bitcast-convert(%p.0.lhs.6)
  %subtract.13 = u32[] subtract(%constant.10, %bitcast-convert.12)
  %bitcast-convert.14 = s32[] bitcast-convert(%subtract.13)
  %select.17 = s32[] select(%compare.16, %bitcast-convert.14,
                            %bitcast-convert.11)
  %p.0.rhs.7 = f32[] parameter(1)
  %bitcast-convert.19 = s32[] bitcast-convert(%p.0.rhs.7)
  %constant.23 = s32[] constant(0)
  %compare.24 = pred[] compare(%bitcast-convert.19, %constant.23), direction=LT
  %constant.18 = u32[] constant(2147483647)
  %bitcast-convert.20 = u32[] bitcast-convert(%p.0.rhs.7)
  %subtract.21 = u32[] subtract(%constant.18, %bitcast-convert.20)
  %bitcast-convert.22 = s32[] bitcast-convert(%subtract.21)
  %select.25 = s32[] select(%compare.24, %bitcast-convert.22,
                            %bitcast-convert.19)
  ROOT %compare.26 = pred[] compare(%select.17, %select.25), direction=GT
})";
}

TEST_F(TopkRewriterTest, Rewrite) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[8,1234567] iota(), iota_dimension=1
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[8,1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
  %get-tuple-element.30 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[8,5] slice(%get-tuple-element.30), slice={[0:8], [0:5]}
  ROOT %tuple.32 = (f32[8,5], s32[8,5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteWithConvertMaxComparator) {
  const std::string hlo_string = R"(
HloModule module
)" + getConvertMaxComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[8,1234567] iota(), iota_dimension=1
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[8,1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
  %get-tuple-element.30 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[8,5] slice(%get-tuple-element.30), slice={[0:8], [0:5]}
  ROOT %tuple.32 = (f32[8,5], s32[8,5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteUnbatched) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234567] parameter(0)
  %iota.4 = s32[1234567] iota(), iota_dimension=0
  %sort.27 = (f32[1234567], s32[1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5] slice(%get-tuple-element.28), slice={[0:5]}
  %get-tuple-element.30 = s32[1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5] slice(%get-tuple-element.30), slice={[0:5]}
  ROOT %tuple.32 = (f32[5], s32[5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteTranspose) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234567,8] parameter(0)
  %iota.4 = s32[1234567,8] iota(), iota_dimension=0
  %sort.27 = (f32[1234567,8], s32[1234567,8]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234567,8] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5,8] slice(%get-tuple-element.28), slice={[0:5], [0:8]}
  %get-tuple-element.30 = s32[1234567,8] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5,8] slice(%get-tuple-element.30), slice={[0:5], [0:8]}
  ROOT %tuple.32 = (f32[5,8], s32[5,8]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  LOG(INFO) << module->entry_computation()->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Transpose(op::GetTupleElement(
                    op::CustomCall(op::Transpose(op::Parameter(0))), 0)),
                op::Transpose(op::GetTupleElement(
                    op::CustomCall(op::Transpose(op::Parameter(0))), 1))));
  const HloInstruction* cc = module->entry_computation()
                                 ->root_instruction()
                                 ->operand(0)
                                 ->operand(0)
                                 ->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteNoIota) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparatorNoIota() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %sort.27 = f32[8,1234567] sort(%arg_tuple.1), dimensions={1}, is_stable=true, to_apply=%compare
  ROOT %slice.29 = f32[8,5] slice(%sort.27), slice={[0:8], [0:5]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

}  // namespace
}  // namespace xla
