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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combiner_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combiner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combiner_testDTcc() {
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

#include "tensorflow/compiler/xla/service/ar_crs_combiner.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ArCrsCombinerTest : public HloTestBase {};

TEST_F(ArCrsCombinerTest, SameValueTestBasecase) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32.1 = f32[2,2] constant({{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant({{1, 2}, {3, 4}})
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(
      i1, module->entry_computation()->parameter_instruction(0)));
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestBasecase2) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (x: f32[]) -> (f32[], f32[]) {
  %x = f32[] parameter(0)
  ROOT %tuple = (f32[], f32[]) tuple(%x, %x)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestBasecase3) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (x: f32[], y: f32[]) -> (f32[], f32[]) {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%x, %y)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestNumOperands) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> ((f32[2,2]), (f32[2,2], f32[2,2])) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %tuple1 = (f32[2,2]) tuple(%constant.f32)
  %tuple2 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %tuple = ((f32[2,2]), (f32[2,2], f32[2,2])) tuple(%tuple1, %tuple2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestSliceIndicesMatch) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2]) -> (f32[1], f32[1]) {
  %p = f32[2] parameter(0)
  %slice.1 = f32[1] slice(f32[2] %p), slice={[0:1]}
  %slice.2 = f32[1] slice(f32[2] %p), slice={[0:1]}
  ROOT %tuple = (f32[1], f32[1]) tuple(%slice.1, %slice.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestSliceIndicesDontMatch) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2]) -> (f32[1], f32[1]) {
  %p = f32[2] parameter(0)
  %slice.1 = f32[1] slice(f32[2] %p), slice={[0:1]}
  %slice.2 = f32[1] slice(f32[2] %p), slice={[1:2]}
  ROOT %tuple = (f32[1], f32[1]) tuple(%slice.1, %slice.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementSameIndex) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=0
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementDifferentIndex1) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=1
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementDifferentIndex2) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32.1 = f32[2,2] constant({{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant({{2, 3}, {4, 5}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=1
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile1) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %constant.0), direction=GT
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32 = f32[2,2] constant({{3, 4}, {5, 6}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0];
  auto i2 = body_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile2) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %constant.0), direction=GT
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32.1 = f32[2,2] constant({{3, 4}, {5, 6}})
  %constant.f32.2 = f32[2,2] constant({{3, 4}, {7, 8}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0];
  auto i2 = body_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile3) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %constant.0), direction=GT
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32.1 = f32[2,2] constant({{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant({{3, 4}, {1, 2}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32.1)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32.2)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32 = f32[2,2] constant({{3, 4}, {5, 6}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0]->operands()[0];  // %get-tuple-element.1
  auto i2 = body_tuple->operands()[1]->operands()[0];  // %get-tuple-element.2
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestNestedWhile) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  ROOT %t = pred[] constant(true)
}

%body_inner (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32 = f32[2,2] constant({{1, 2}, {3, 4}})
  %gte.1 = f32[2,2] get-tuple-element(%x), index=0
  %gte.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%gte.1, %constant.f32)
  %add.2 = f32[2,2] add(%gte.2, %constant.f32)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

%body_outer (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %gte.1 = f32[2,2] get-tuple-element(%x), index=0
  %gte.2 = f32[2,2] get-tuple-element(%x), index=1
  %init = (f32[2,2], f32[2,2]) tuple(%gte.1, %gte.2)
  ROOT %while.1 = (f32[2,2], f32[2,2]) while(%init), condition=%condition,
    body=%body_inner
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32 = f32[2,2] constant({{3, 4}, {5, 6}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition,
    body=%body_outer
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto root_while = module->entry_computation()->root_instruction();
  auto inner_while = root_while->while_body()->root_instruction();
  auto i1 = inner_while->while_body()->root_instruction()->operands()[0];
  auto i2 = inner_while->while_body()->root_instruction()->operands()[1];
  // They are the same because the same constant {{3, 4}, {5, 6}} flows to both,
  // and we add the same number {{1, 2}, {3, 4}} to both in each iteration
  // of the inner while.
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

void CompareReplicaGroups(absl::Span<const ReplicaGroup> groups_before,
                          absl::Span<const ReplicaGroup> groups_after) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combiner_testDTcc mht_0(mht_0_v, 549, "", "./tensorflow/compiler/xla/service/ar_crs_combiner_test.cc", "CompareReplicaGroups");

  ASSERT_EQ(groups_before.size(), groups_after.size());
  for (int i = 0; i < groups_before.size(); ++i) {
    // Somewhat verbose way to compare the replica_ids, because EqualsProto
    // is not available in the open-source build.
    auto group_before = groups_before[i];
    std::vector<int64_t> ids_before(group_before.replica_ids().begin(),
                                    group_before.replica_ids().end());
    auto group_after = groups_after[i];
    std::vector<int64_t> ids_after(group_after.replica_ids().begin(),
                                   group_after.replica_ids().end());
    EXPECT_EQ(ids_before, ids_after);
  }
}

TEST_F(ArCrsCombinerTest, RewriteArConvertCrs) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[], f32[]) {
  %p = bf16[] parameter(0)
  %constant.bf16 = bf16[] constant(1)

  %all-reduce.ar.1 = bf16[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=0}
  %convert.1 = f32[]
      convert(%all-reduce.ar.1),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%convert.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=1}
  %convert.2 = f32[]
      convert(%all-reduce.ar.2),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%convert.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Convert(op::Parameter())),
                        op::AllReduce(op::Convert(op::Constant()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArConvertCrsSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[]) {
  %p = bf16[] parameter(0)
  %all-reduce.ar.1 = bf16[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16
  %convert.1 = f32[] convert(%all-reduce.ar.1)
  %all-reduce.1 = f32[]
      all-reduce(%convert.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2, true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Convert(op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArBitcastCrs) {
  const char* module_str = R"(
HloModule foobar

%sum.1 (a: f32[2,1], b: f32[2,1]) -> f32[2,1] {
  %a = f32[2,1] parameter(0)
  %b = f32[2,1] parameter(1)
  ROOT %add = f32[2,1] add(%a, %b)
}

%sum.2 (x: f32[2], y: f32[2]) -> f32[2] {
  %x = f32[2] parameter(0)
  %y = f32[2] parameter(1)
  ROOT %add = f32[2] add(%x, %y)
}

ENTRY %entrycomp (p: f32[2,1]) -> (f32[2], f32[2]) {
  %p = f32[2,1] parameter(0)

  %all-reduce.ar.1 = f32[2,1]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.1,
      sharding={maximal device=0}
  %bitcast.1 = f32[2]{0} bitcast(f32[2,1]{1,0} %all-reduce.ar.1)
  %all-reduce.1 = f32[2]
      all-reduce(%bitcast.1),
      replica_groups={{0,1}},
      to_apply=%sum.2,
      sharding={maximal device=0}

  %all-reduce.ar.2 = f32[2,1]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.1,
      sharding={maximal device=1}
  %bitcast.2 = f32[2]{0} bitcast(f32[2,1]{1,0} %all-reduce.ar.2)
  %all-reduce.2 = f32[2]
      all-reduce(%bitcast.2),
      replica_groups={{0,1}},
      to_apply=%sum.2,
      sharding={maximal device=1}

  ROOT %tuple = (f32[2], f32[2])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Bitcast(op::Parameter())),
                        op::AllReduce(op::Bitcast(op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArMultiplyCrs) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.f32 = f32[] constant(123)

  %all-reduce.ar.1 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.f32,
      sharding={maximal device=0}
  %multiply.1 = f32[]
      multiply(%all-reduce.ar.1, %constant.f32),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%multiply.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.f32,
      sharding={maximal device=1}
  %multiply.2 = f32[]
      multiply(%all-reduce.ar.2, %constant.f32),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%multiply.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllReduce(op::Multiply(op::Parameter(), op::Constant())),
                op::AllReduce(op::Multiply(op::Parameter(), op::Constant()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArMultiplyCrsSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %constant.f32 = f32[] constant(123)

  %all-reduce.ar.1 = f32[] all-reduce(%p), replica_groups={{0},{1}},
      channel_id=1, to_apply=%sum.f32
  %multiply.1 = f32[] multiply(%all-reduce.ar.1, %constant.f32)
  %all-reduce.1 = f32[] all-reduce(%multiply.1), replica_groups={{0,1}},
      to_apply=%sum.f32, sharding={maximal device=0}
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllReduce(op::Multiply(op::Parameter(), op::Constant()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArConvertAddCrs) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.bf16 = bf16[] constant(1)
  %constant.f32 = f32[] constant(2)

  %all-reduce.ar.1 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=0}
  %convert.1 = f32[]
      convert(%all-reduce.ar.1),
      sharding={maximal device=0}
  %add.1 = f32[]
      add(%constant.f32, %convert.1),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%add.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=1}
  %convert.2 = f32[]
      convert(%all-reduce.ar.2),
      sharding={maximal device=1}
  %add.2 = f32[]
      add(%constant.f32, %convert.2),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%add.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::AllReduce(op::Add(op::Divide(op::Constant(), op::Constant()),
                                op::Convert())),
          op::AllReduce(op::Add(op::Divide(op::Constant(), op::Constant()),
                                op::Convert()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArConvertAddCrsSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %constant.bf16 = bf16[] constant(1)
  %constant.f32 = f32[] constant(2)

  %all-reduce.ar.1 = bf16[] all-reduce(%constant.bf16), replica_groups={{0},{1}},
      channel_id=1, to_apply=%sum.bf16
  %convert.1 = f32[] convert(%all-reduce.ar.1), sharding={maximal device=0}
  %add.1 = f32[] add(%constant.f32, %convert.1)
  %all-reduce.1 = f32[] all-reduce(%add.1), replica_groups={{0,1}},
      to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Add(
                  op::Divide(op::Constant(), op::Constant()), op::Convert()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, OtherSummandNotTheSameDontRewrite) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.bf16 = bf16[] constant(1)
  %constant.f32.1 = f32[] constant(2)
  %constant.f32.2 = f32[] constant(3)

  %all-reduce.ar.1 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=0}
  %convert.1 = f32[]
      convert(%all-reduce.ar.1),
      sharding={maximal device=0}
  %add.1 = f32[]
      add(%constant.f32.1, %convert.1),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%add.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=1}
  %convert.2 = f32[]
      convert(%all-reduce.ar.2),
      sharding={maximal device=1}
  %add.2 = f32[]
      add(%constant.f32.2, %convert.2),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%add.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, OtherSummandNotTheSameDontRewriteSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %constant.bf16 = bf16[] constant(1)
  %constant.f32.1 = f32[] constant(2)

  %all-reduce.ar.1 = bf16[] all-reduce(%constant.bf16), replica_groups={{0},{1}},
      channel_id=1, to_apply=%sum.bf16
  %convert.1 = f32[] convert(%all-reduce.ar.1)
  %add.1 = f32[] add(%p, %convert.1)
  %all-reduce.1 = f32[] all-reduce(%add.1), replica_groups={{0,1}}, to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, ArThenCrsDontCrash) {
  const char* module_str = R"(
HloModule foobar

%sum.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.f32 = f32[] constant(123)

  %all-reduce.ar.1 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.1,
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%all-reduce.ar.1),
      replica_groups={{0,1}},
      to_apply=%sum.1,
      sharding={maximal device=0}
  %multiply.1 = f32[]
      multiply(%all-reduce.1, %constant.f32),
      sharding={maximal device=0}

  %all-reduce.ar.2 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.1,
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%all-reduce.ar.2),
      replica_groups={{0,1}},
      to_apply=%sum.1,
      sharding={maximal device=1}
  %multiply.2 = f32[]
      multiply(%all-reduce.2, %constant.f32),
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Parameter()),
                        op::AllReduce(op::Parameter())));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleAdds) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.1 = f32[] constant(1)
  %constant.2 = f32[] constant(2)

  %all-reduce.ar.1 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=0}
  %add.11 = f32[]
      add(%constant.1, %all-reduce.ar.1),
      sharding={maximal device=0}
  %add.12 = f32[]
      add(%constant.2, %add.11),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%add.12),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=0}

  %all-reduce.ar.2 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=0}
  %add.21 = f32[]
      add(%constant.1, %all-reduce.ar.2),
      sharding={maximal device=0}
  %add.22 = f32[]
      add(%constant.2, %add.21),
      sharding={maximal device=0}
  %all-reduce.2 = f32[]
      all-reduce(%add.22),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=0}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Add(
                            op::Divide(op::Constant(), op::Constant()),
                            op::Add(op::Divide(op::Constant(), op::Constant()),
                                    op::Parameter()))),
                        op::AllReduce(op::Add(
                            op::Divide(op::Constant(), op::Constant()),
                            op::Add(op::Divide(op::Constant(), op::Constant()),
                                    op::Parameter())))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleAddsSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %constant.1 = f32[] constant(1)
  %constant.2 = f32[] constant(2)

  %all-reduce.ar.1 = f32[] all-reduce(%p), replica_groups={{0},{1}},
      channel_id=1, to_apply=%sum
  %add.11 = f32[] add(%constant.1, %all-reduce.ar.1)
  %add.12 = f32[] add(%constant.2, %add.11)
  %all-reduce.1 = f32[] all-reduce(%add.12), replica_groups={{0,1}}, to_apply=%sum
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(
                  op::Add(op::Divide(op::Constant(), op::Constant()),
                          op::Add(op::Divide(op::Constant(), op::Constant()),
                                  op::Parameter())))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArSubtractCrs) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %constant.f32 = f32[] constant(123)

  %all-reduce.ar.1 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.f32,
      sharding={maximal device=0}
  %sub.1 = f32[]
      subtract(%constant.f32, %all-reduce.ar.1),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%sub.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum.f32,
      sharding={maximal device=1}
  %sub.2 = f32[]
      subtract(%constant.f32, %all-reduce.ar.2),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%sub.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::AllReduce(op::Subtract(op::Divide(op::Constant(), op::Constant()),
                                     op::Parameter())),
          op::AllReduce(op::Subtract(op::Divide(op::Constant(), op::Constant()),
                                     op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteArSubtractCrsSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %constant.f32 = f32[] constant(123)
  %all-reduce.ar.1 = f32[] all-reduce(%p), replica_groups={{0},{1}},
      channel_id=1, to_apply=%sum.f32
  %sub.1 = f32[] subtract(%constant.f32, %all-reduce.ar.1)
  %all-reduce.1 = f32[] all-reduce(%sub.1), replica_groups={{0,1}},
      to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllReduce(op::Subtract(
          op::Divide(op::Constant(), op::Constant()), op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleARsLeft) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %const1 = f32[] constant(1)
  %const2 = f32[] constant(2)

  %ar11 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=0}
  %add11 = f32[]
      add(%ar11, %const1),
      sharding={maximal device=0}
  %ar12 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=2,
      to_apply=%sum,
      sharding={maximal device=0}
  %add12 = f32[]
      add(%add11, %ar12),
      sharding={maximal device=0}
  %crs1 = f32[]
      all-reduce(%add12),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=0}

  %ar21 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=1}
  %add21 = f32[]
      add(%ar21, %const1),
      sharding={maximal device=1}
  %ar22 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=2,
      to_apply=%sum,
      sharding={maximal device=1}
  %add22 = f32[]
      add(%add21, %ar22),
      sharding={maximal device=1}
  %crs2 = f32[]
      all-reduce(%add22),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%crs1, %crs2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Add(
                            op::Add(op::Parameter(),
                                    op::Divide(op::Constant(), op::Constant())),
                            op::Parameter())),
                        op::AllReduce(op::Add(
                            op::Add(op::Parameter(),
                                    op::Divide(op::Constant(), op::Constant())),
                            op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleARsLeftSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %const1 = f32[] constant(1)
  %const2 = f32[] constant(2)

  %ar11 = f32[] all-reduce(%p), replica_groups={{0},{1}}, channel_id=1,
      to_apply=%sum
  %add11 = f32[] add(%ar11, %const1)
  %ar12 = f32[] all-reduce(%p), replica_groups={{0},{1}}, channel_id=2,
      to_apply=%sum
  %add12 = f32[] add(%add11, %ar12)
  %crs1 = f32[] all-reduce(%add12), replica_groups={{0,1}},
      to_apply=%sum
  ROOT %tuple = (f32[]) tuple(%crs1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllReduce(op::Add(
          op::Add(op::Parameter(), op::Divide(op::Constant(), op::Constant())),
          op::Parameter()))));
  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleARsRight) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[], f32[]) {
  %p = f32[] parameter(0)
  %const1 = f32[] constant(1)
  %const2 = f32[] constant(2)

  %ar11 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=0}
  %ar12 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=2,
      to_apply=%sum,
      sharding={maximal device=0}
  %add11 = f32[]
      add(%ar12, %const1),
      sharding={maximal device=0}
  %add12 = f32[]
      add(%ar11, %add11),
      sharding={maximal device=0}
  %crs1 = f32[]
      all-reduce(%add12),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=0}

  %ar21 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=1,
      to_apply=%sum,
      sharding={maximal device=1}
  %ar22 = f32[]
      all-reduce(%p),
      replica_groups={{0},{1}},
      channel_id=2,
      to_apply=%sum,
      sharding={maximal device=1}
  %add21 = f32[]
      add(%ar22, %const1),
      sharding={maximal device=1}
  %add22 = f32[]
      add(%ar21, %add21),
      sharding={maximal device=1}
  %crs2 = f32[]
      all-reduce(%add22),
      replica_groups={{0,1}},
      to_apply=%sum,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%crs1, %crs2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllReduce(op::Add(
                    op::Parameter(),
                    op::Add(op::Parameter(),
                            op::Divide(op::Constant(), op::Constant())))),
                op::AllReduce(op::Add(
                    op::Parameter(),
                    op::Add(op::Parameter(),
                            op::Divide(op::Constant(), op::Constant()))))));

  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, RewriteMultipleARsRightSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[]) -> (f32[]) {
  %p = f32[] parameter(0)
  %const1 = f32[] constant(1)
  %const2 = f32[] constant(2)

  %ar11 = f32[] all-reduce(%p), replica_groups={{0},{1}}, channel_id=1, to_apply=%sum
  %ar12 = f32[] all-reduce(%p), replica_groups={{0},{1}}, channel_id=2, to_apply=%sum
  %add11 = f32[] add(%ar12, %const1)
  %add12 = f32[] add(%ar11, %add11)
  %crs1 = f32[] all-reduce(%add12), replica_groups={{0,1}}, to_apply=%sum
  ROOT %tuple = (f32[]) tuple(%crs1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::Add(
                  op::Parameter(),
                  op::Add(op::Parameter(),
                          op::Divide(op::Constant(), op::Constant()))))));

  auto crs_after =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  CompareReplicaGroups(replica_groups_before, replica_groups_after);
}

TEST_F(ArCrsCombinerTest, OneReplicaDontRewrite) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[], f32[]) {
  %p = bf16[] parameter(0)
  %constant.bf16 = bf16[] constant(1)

  %all-reduce.ar.1 = bf16[]
      all-reduce(%p),
      replica_groups={{0}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=0}
  %convert.1 = f32[]
      convert(%all-reduce.ar.1),
      sharding={maximal device=0}
  %all-reduce.1 = f32[]
      all-reduce(%convert.1),
      replica_groups={{0}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %all-reduce.ar.2 = bf16[]
      all-reduce(%constant.bf16),
      replica_groups={{0}},
      channel_id=1,
      to_apply=%sum.bf16,
      sharding={maximal device=1}
  %convert.2 = f32[]
      convert(%all-reduce.ar.2),
      sharding={maximal device=1}
  %all-reduce.2 = f32[]
      all-reduce(%convert.2),
      replica_groups={{0}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[], f32[])
      tuple(%all-reduce.1, %all-reduce.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, OneReplicaDontRewriteSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.bf16 (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[]) {
  %p = bf16[] parameter(0)
  %constant.bf16 = bf16[] constant(1)

  %all-reduce.ar.1 = bf16[] all-reduce(%p), replica_groups={{0}},
      channel_id=1, to_apply=%sum.bf16
  %convert.1 = f32[] convert(%all-reduce.ar.1)
  %all-reduce.1 = f32[] all-reduce(%convert.1),
      replica_groups={{0}}, to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, SameValueTestConditional) {
  const char* module_str = R"(
HloModule foobar

branch_true {
  pt = (f32[2,4], f32[2,4]) parameter(0)
  gte.0 = f32[2,4] get-tuple-element(pt), index=0
  gte.1 = f32[2,4] get-tuple-element(pt), index=1
  ROOT tuple.t = (f32[2,4], f32[2,4]) tuple(gte.1, gte.0)
}

branch_false {
  pf = (f32[2,4], f32[2,4]) parameter(0)
  gte.0 = f32[2,4] get-tuple-element(pf), index=0
  gte.1 = f32[2,4] get-tuple-element(pf), index=1
  add = f32[2,4] add(gte.1, gte.1)
  ROOT tuple.f = (f32[2,4], f32[2,4]) tuple(gte.0, add)
}

ENTRY Parameters1.v4 {
  constant = pred[] constant(true)
  p = f32[2,4] parameter(0)
  tuple = (f32[2,4], f32[2,4]) tuple(p, p)
  ROOT conditional = (f32[2,4], f32[2,4]) conditional(constant, tuple, tuple), true_computation=branch_true, false_computation=branch_false
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto cond = module->entry_computation()->root_instruction();

  auto branch_true = cond->branch_computation(0)->root_instruction();
  auto t0 = branch_true->mutable_operand(0);
  auto t1 = branch_true->mutable_operand(1);
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(t0, t1));

  auto branch_false = cond->branch_computation(1)->root_instruction();
  auto f0 = branch_false->mutable_operand(0);
  auto f1 = branch_false->mutable_operand(1);
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(f0, f1));
}

TEST_F(ArCrsCombinerTest, AllReduceWithReplicas) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[], f32[]) {
  %p = bf16[] parameter(0)
  %all-reduce.0 = f32[] all-reduce(%p), channel_id=1, replica_groups={{0,1}},
    to_apply=%sum.f32, sharding={maximal device=0}
  %all-reduce.1 = f32[] all-reduce(%p), channel_id=1, replica_groups={{0,1}},
    to_apply=%sum.f32, sharding={maximal device=1}
  %all-reduce.2 = f32[] all-reduce(%all-reduce.0), replica_groups={{0,1}},
    to_apply=%sum.f32, sharding={maximal device=0}
  %all-reduce.3 = f32[] all-reduce(%all-reduce.1), replica_groups={{0,1}},
    to_apply=%sum.f32, sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[]) tuple(%all-reduce.2, %all-reduce.3),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, AllReduceWithReplicasSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[]) {
  %p = bf16[] parameter(0)
  %all-reduce.0 = f32[] all-reduce(%p), channel_id=1, replica_groups={{0},{1}},
    to_apply=%sum.f32
  %all-reduce.2 = f32[] all-reduce(%all-reduce.0), replica_groups={{0},{1}},
    to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

TEST_F(ArCrsCombinerTest, ReplaceReplicatedAllReduceSPMD) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[2,4]) -> f32[2,4] {
  %p = f32[2,4] parameter(0), sharding={replicated}
  ROOT %all-reduce = f32[2,4] all-reduce(%p), to_apply=%sum.f32,
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}}
}
)";

  // Replacing replicated all-reduce is only triggered when there are enough
  // replicas (currently > num_partitions * 8).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/32));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Divide(op::AllReduce(op::Parameter()),
                               op::Broadcast(op::Constant())));

  auto ar = root->operand(0);
  auto divisor = root->operand(1)->operand(0);
  EXPECT_TRUE(ar->channel_id());
  EXPECT_TRUE(divisor->literal().IsAllFloat(2));
}

TEST_F(ArCrsCombinerTest, AllReduceWithGlobalIdReplicaGroups) {
  const char* module_str = R"(
HloModule foobar

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: bf16[]) -> (f32[]) {
  %p = bf16[] parameter(0)
  %all-reduce.0 = f32[] all-reduce(%p), channel_id=1,
    replica_groups={{0,1,2,3},{4,5,6,7}}, use_global_device_ids=true,
    to_apply=%sum.f32
  %all-reduce.2 = f32[] all-reduce(%all-reduce.0), replica_groups={{0,1}},
    to_apply=%sum.f32
  ROOT %tuple = (f32[]) tuple(%all-reduce.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2,
                                   /*num_partitions=*/4));
  ArCrsCombiner combiner(/*num_spatial_partitions=*/4,
                         /*spmd_partition=*/true);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla
