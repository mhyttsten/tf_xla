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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_element_type_converter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_element_type_converter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_element_type_converter_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Not;
using ::testing::ResultOf;

using HloElementTypeConverterTest = HloTestBase;

TEST_F(HloElementTypeConverterTest, CustomCallsNotConverted) {
  const std::string& hlo_string = R"(
    HloModule custom_call
    ENTRY CustomCall {
      constant = bf16[1]{0} constant({12345})
      ROOT custom-call = bf16[1,2,3]{0,2,1} custom-call(constant),
           custom_call_target="foo"
    }
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_FALSE(converted);
}

TEST_F(HloElementTypeConverterTest, InfeedsOutfeedsNotConverted) {
  const std::string& hlo_string = R"(
    HloModule InfeedOutfeed
    ENTRY RoundTrip16MiBR1.v2 {
      token0 = token[] after-all()
      infeed = (bf16[4]{0}, token[]) infeed(token0)
      ROOT infeed.data = bf16[4]{0} get-tuple-element(infeed), index=0
      outfeed = token[] outfeed(infeed.data, token0)
    }
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_FALSE(converted);
}

TEST_F(HloElementTypeConverterTest, OperationsInNestedTuplesConverted) {
  const std::string& hlo_string = R"(
    HloModule NestedTuples
    ENTRY NestedTuples.v5 {
      constant.2 = f32[2]{0} constant({1, 2})
      constant.3 = bf16[2]{0} constant({42, 42})
      add = bf16[2]{0} add(constant.2, constant.3)
      tuple = (f32[2]{0}, bf16[2]{0}) tuple(constant.2, add)
      constant.5 = bf16[2]{0} constant({22, 44})
      ROOT tuple.1 = ((f32[2]{0}, bf16[2]{0}), bf16[2]{0}) tuple(tuple, constant.5)
    }
  )";

  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);
  const HloInstruction* bf16_op =
      module->entry_computation()->root_instruction()->operand(0)->operand(1);
  EXPECT_THAT(bf16_op, op::Convert(op::Add(op::Constant(), op::Convert())));
}

TEST_F(HloElementTypeConverterTest, BatchNormGradBF16Converted) {
  const std::string& hlo_string = R"(
    HloModule BatchNormGrad
    ENTRY BatchNormGrad.v6 {
      constant.4 = bf16[2,2,2,1]{3,2,1,0} constant({ { /*i0=0*/
      { /*i1=0*/ {0}, {0} }, { /*i1=1*/ {0}, {0} } }, { /*i0=1*/ { /*i1=0*/ {0},
      {0} }, { /*i1=1*/ {0}, {0} } } })
      constant.5 = bf16[2]{0} constant({1, 1})
      constant.6 = bf16[2]{0} constant({0, 0})
      constant.7 = bf16[2]{0} constant({1, 1})
      constant.8 = bf16[2,2,2,1]{3,2,1,0} constant({ { /*i0=0*/
      { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} } }, { /*i0=1*/ { /*i1=0*/
      {5}, {6} }, { /*i1=1*/ {7}, {8} } } })
      ROOT batch-norm-grad = (bf16[2,2,2,1]{3,2,1,0}, bf16[2]{0}, bf16[2]{0})
      batch-norm-grad(constant.4, constant.5, constant.6, constant.7,
      constant.8), epsilon=0, feature_index=2
    }
  )";

  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);
  const HloInstruction* tuple_instr =
      module->entry_computation()->root_instruction();
  ::testing::Matcher<const ::xla::HloInstruction*> batch_norm =
      op::BatchNormGrad();
  EXPECT_THAT(tuple_instr,
              op::Tuple(op::Convert(op::GetTupleElement(batch_norm, 0)),
                        op::Convert(op::GetTupleElement(batch_norm, 1)),
                        op::Convert(op::GetTupleElement(batch_norm, 2))));
}

TEST_F(HloElementTypeConverterTest, RngIsRemoved) {
  const std::string& hlo_string = R"(
HloModule RngIsRemoved

ENTRY main {
  constant.3 = bf16[] constant(0)
  constant.4 = bf16[] constant(1)
  ROOT rng = bf16[1,1000,20]{2,1,0} rng(constant.3, constant.4), distribution=rng_uniform
}
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);

  std::function<bool(const HloInstruction*)> is_bf16_rng =
      [](const HloInstruction* inst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_element_type_converter_testDTcc mht_0(mht_0_v, 305, "", "./tensorflow/compiler/xla/service/hlo_element_type_converter_test.cc", "lambda");

        return inst->shape().element_type() == BF16 &&
               inst->opcode() == HloOpcode::kRng;
      };

  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(ResultOf(is_bf16_rng, Eq(true)))));
}

TEST_F(HloElementTypeConverterTest, RngCtrlDep) {
  const std::string& hlo_string = R"(
HloModule RngIsRemoved

ENTRY main {
  constant.3 = bf16[] constant(0)
  constant.4 = bf16[] constant(1)
  rng0 = bf16[1,2000,20]{2,1,0} rng(constant.3, constant.4), distribution=rng_uniform
  ROOT rng1 = bf16[1,1000,20]{2,1,0} rng(constant.3, constant.4), control-predecessors={%rng0}, distribution=rng_uniform
}
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();

  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);

  HloInstruction *rng0, *rng1;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kRng) {
      const Shape& shape = inst->shape();
      ASSERT_EQ(shape.dimensions_size(), 3);
      ASSERT_TRUE(shape.dimensions(1) == 2000 || shape.dimensions(1) == 1000);
      if (shape.dimensions(1) == 2000) {
        rng0 = inst;
      } else {
        rng1 = inst;
      }
    }
  }

  EXPECT_THAT(rng0->control_successors(), ElementsAre(rng1));
  EXPECT_THAT(rng1->control_predecessors(), ElementsAre(rng0));
}

TEST_F(HloElementTypeConverterTest, BitcastConvertIsUnmodified) {
  const std::string& hlo_string = R"(
  HloModule test

  ENTRY test {
    p = bf16[] parameter(0)
    ROOT c = u16[] bitcast-convert(p)
  })";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ValueOrDie();
  HloElementTypeConverter converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, RunHloPass(&converter, module.get()));
  EXPECT_FALSE(converted);
}

}  // namespace
}  // namespace xla
