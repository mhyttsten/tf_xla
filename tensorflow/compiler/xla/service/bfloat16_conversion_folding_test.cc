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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc() {
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

#include "tensorflow/compiler/xla/service/bfloat16_conversion_folding.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

class TestBFloat16Support : public BFloat16Support {
 public:
  TestBFloat16Support() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "TestBFloat16Support");
}
  ~TestBFloat16Support() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_1(mht_1_v, 205, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "~TestBFloat16Support");
}

  bool SupportsBF16Operand(const HloInstruction& hlo,
                           int64_t operand_index) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_2(mht_2_v, 211, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "SupportsBF16Operand");

    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }

  bool SupportsBF16Output(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_3(mht_3_v, 225, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "SupportsBF16Output");

    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_4(mht_4_v, 239, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "SupportsMixedPrecisions");

    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }
};

class BFloat16ConversionFoldingTest : public HloTestBase {
 protected:
  BFloat16ConversionFoldingTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/true) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_5(mht_5_v, 256, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "BFloat16ConversionFoldingTest");
}

  bool FoldConversions(HloModule* module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_folding_testDTcc mht_6(mht_6_v, 261, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding_test.cc", "FoldConversions");

    TestBFloat16Support bfloat16_support_;
    BFloat16ConversionFolding fold(&bfloat16_support_);
    StatusOr<bool> result = fold.Run(module);
    EXPECT_IS_OK(result.status());
    return result.ValueOrDie();
  }
};

TEST_F(BFloat16ConversionFoldingTest, FoldIfSupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, add0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, convert1, c));
  builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, add1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), add1);
  EXPECT_EQ(add0->shape().element_type(), BF16);
  EXPECT_EQ(add1->shape().element_type(), BF16);
  EXPECT_EQ(add1->operand(0), add0);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldIfUnsupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* mul0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kMultiply, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, mul0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* mul1 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_shape, HloOpcode::kMultiply, convert1, c));
  HloInstruction* convert2 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, mul1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert2);
  EXPECT_EQ(mul0->shape().element_type(), F32);
  EXPECT_EQ(mul1->shape().element_type(), F32);
  EXPECT_EQ(mul1->operand(0), convert1);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldUnsupportedMixedPrecision) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* sub0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kSubtract, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, sub0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* sub1 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_shape, HloOpcode::kSubtract, convert1, c));
  HloInstruction* convert2 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, sub1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert2);
  EXPECT_EQ(sub0->shape().element_type(), F32);
  EXPECT_EQ(sub1->shape().element_type(), F32);
  EXPECT_EQ(sub1->operand(0), convert1);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldTuple) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, b));

  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({a, convert0}));
  HloInstruction* gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, tuple, 0));
  HloInstruction* convert1 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert1);
  EXPECT_EQ(gte->shape().element_type(), F32);
  EXPECT_EQ(tuple->operand(1), convert0);
}

TEST_F(BFloat16ConversionFoldingTest, FoldAllReduceTupleOutput) {
  auto builder = HloComputation::Builder(TestName());

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("add");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, x, y));
  HloComputation* sum = module->AddEmbeddedComputation(sum_builder.Build());

  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape, "a"));
  HloInstruction* convert_a =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, a));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));

  HloInstruction* crs = builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape({f32_shape, f32_shape}), {convert_a, b}, sum,
      /*replica_groups=*/{},
      /*constrain_layout=*/false,
      /*channel_id=*/absl::nullopt, /*use_global_device_ids=*/false));
  HloInstruction* gte_a = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, crs, 0));
  HloInstruction* gte_b = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, crs, 1));
  HloInstruction* convert_gte_b =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte_b));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({gte_a, convert_gte_b}));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), tuple);
  EXPECT_EQ(tuple->operand(0), gte_a);
  EXPECT_EQ(tuple->operand(1), gte_b);
  EXPECT_EQ(gte_a->shape().element_type(), F32);
  EXPECT_EQ(gte_b->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(0), a);
  EXPECT_EQ(crs->operand(1), b);
  EXPECT_EQ(a->shape().element_type(), BF16);
  EXPECT_EQ(b->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {0}).element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {1}).element_type(), BF16);
}

}  // namespace xla
