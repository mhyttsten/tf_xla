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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc() {
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

#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "TestBFloat16Support");
}
  ~TestBFloat16Support() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "~TestBFloat16Support");
}

  bool SupportsBF16Operand(const HloInstruction& hlo,
                           int64_t operand_index) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_2(mht_2_v, 213, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "SupportsBF16Operand");

    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kReduce ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll) {
      return true;
    }
    if (hlo.opcode() == HloOpcode::kDot) {
      // Test that only the first operand of kDot supports BF16.
      return operand_index == 0;
    }
    return false;
  }

  bool SupportsBF16Output(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_3(mht_3_v, 232, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "SupportsBF16Output");

    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kReduce ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kDot || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll) {
      return true;
    }
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "SupportsMixedPrecisions");

    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement) {
      return true;
    }
    return false;
  }
};

class BFloat16NormalizationTest : public HloTestBase {
 protected:
  BFloat16NormalizationTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/true) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "BFloat16NormalizationTest");
}

  bool Normalize(HloModule* module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalization_testDTcc mht_6(mht_6_v, 267, "", "./tensorflow/compiler/xla/service/bfloat16_normalization_test.cc", "Normalize");

    TestBFloat16Support bfloat16_support_;
    BFloat16Normalization normalization(&bfloat16_support_);
    StatusOr<bool> result = normalization.Run(module);
    EXPECT_IS_OK(result.status());

    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    EXPECT_IS_OK(verifier.Run(module).status());

    return result.ValueOrDie();
  }
};

TEST_F(BFloat16NormalizationTest, NoopIfSupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kAdd, a, b));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, add0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), add1);
  EXPECT_EQ(add0->shape().element_type(), BF16);
  EXPECT_EQ(add1->shape().element_type(), F32);
}

TEST_F(BFloat16NormalizationTest, ResolveIfUnsupportedBF16) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* mul0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kMultiply, a, b));

  HloInstruction* mul1 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kMultiply, mul0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(computation->root_instruction()->operand(0), mul1);
  EXPECT_EQ(mul0->shape().element_type(), F32);
  EXPECT_EQ(mul1->shape().element_type(), F32);
  EXPECT_EQ(mul1->operand(0)->opcode(), HloOpcode::kConvert);
}

TEST_F(BFloat16NormalizationTest, ResolveUnsupportedMixedPrecisionSubtraction) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* sub0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kSubtract, a, b));

  HloInstruction* sub1 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kSubtract, sub0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(computation->root_instruction()->operand(0), sub1);
  EXPECT_EQ(sub0->shape().element_type(), F32);
  EXPECT_EQ(sub1->shape().element_type(), F32);
  EXPECT_EQ(sub1->operand(0)->opcode(), HloOpcode::kConvert);
}

TEST_F(BFloat16NormalizationTest, ResolveUnsupportedMixedPrecisionReduce) {
  Shape f32_input_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape f32_output_shape = ShapeUtil::MakeShape(F32, {4});

  Shape bf16_scalar_shape = ShapeUtil::MakeShape(BF16, {});

  auto reduce_comp_builder = HloComputation::Builder("reduce_comp");
  auto reduce_comp_param0 = reduce_comp_builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_scalar_shape, "param0"));
  auto reduce_comp_param1 = reduce_comp_builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_scalar_shape, "param1"));
  reduce_comp_builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_scalar_shape, HloOpcode::kAdd,
                                   reduce_comp_param0, reduce_comp_param1));

  auto module = CreateNewVerifiedModule();
  auto reduce_computation =
      module->AddEmbeddedComputation(reduce_comp_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_input_shape, "a"));
  HloInstruction* init = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_scalar_shape, "init"));
  HloInstruction* reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      f32_output_shape, input, init, {0}, reduce_computation));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), reduce);
  EXPECT_EQ(reduce->called_computations().size(), 1);
  EXPECT_EQ(reduce->called_computations()[0]->num_parameters(), 2);
  EXPECT_EQ(reduce->called_computations()[0]
                ->parameter_instruction(0)
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->called_computations()[0]
                ->parameter_instruction(1)
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->called_computations()[0]
                ->root_instruction()
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->shape().element_type(), F32);
  EXPECT_EQ(reduce->operand(0), input);
  EXPECT_EQ(input->shape().element_type(), F32);
  EXPECT_EQ(reduce->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(reduce->operand(1)->shape().element_type(), F32);
}

TEST_F(BFloat16NormalizationTest, ResolveMixedPrecisionTupleAllReduce) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));

  HloInstruction* crs = builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape({f32_shape, bf16_shape}), {a, b}, reduction,
      /*replica_groups=*/{},
      /*constrain_layout=*/false,
      /*channel_id=*/absl::nullopt,
      /*use_global_device_ids=*/false));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(bf16_shape, crs, 1));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(1)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {1}).element_type(), F32);
}

TEST_F(BFloat16NormalizationTest, ResolveMixedPrecisionTupleAllToAllToBF16) {
  auto module = CreateNewVerifiedModule(TestName(), /*replica_count=*/2);

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));

  std::vector<ReplicaGroup> replica_groups(1);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(1);
  HloInstruction* a2a = builder.AddInstruction(HloInstruction::CreateAllToAll(
      ShapeUtil::MakeTupleShape({bf16_shape, bf16_shape}), {a, a},
      replica_groups, /*constrain_layout=*/false, absl::nullopt));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), a2a);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {0}).element_type(), BF16);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {1}).element_type(), BF16);
  EXPECT_EQ(a2a->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(a2a->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(a2a->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(a2a->operand(1)->shape().element_type(), BF16);
}

TEST_F(BFloat16NormalizationTest, ResolveMixedPrecisionTupleAllToAllToF32) {
  auto module = CreateNewVerifiedModule(TestName(), /*replica_count=*/2);

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));

  std::vector<ReplicaGroup> replica_groups(1);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(1);
  HloInstruction* a2a = builder.AddInstruction(HloInstruction::CreateAllToAll(
      ShapeUtil::MakeTupleShape({bf16_shape, f32_shape}), {a, a},
      replica_groups, /*constrain_layout=*/false, absl::nullopt));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {0}).element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {1}).element_type(), F32);
  EXPECT_EQ(a2a->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(a2a->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(a2a->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(a2a->operand(1)->shape().element_type(), F32);
}

TEST_F(BFloat16NormalizationTest, ResolveMixedPrecisionTupleSort) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {1024});
  Shape s32_shape = ShapeUtil::MakeShape(BF16, {1024});

  HloInstruction* key = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "key"));
  HloInstruction* value = builder.AddInstruction(
      HloInstruction::CreateParameter(1, s32_shape, "value"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort,
      MakeSortHlo(ShapeUtil::MakeTupleShape({bf16_shape, s32_shape}),
                  {key, value}, 0, /*is_stable=*/false, &builder,
                  module.get()));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(bf16_shape, sort, 0));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(sort->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(sort->shape(), {0}).element_type(), F32);
}

TEST_F(BFloat16NormalizationTest, ResolveMixedPrecisionTupleSortRoot) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {1024});

  HloInstruction* key = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "key"));
  HloInstruction* value = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "value"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort,
      MakeSortHlo(ShapeUtil::MakeTupleShape({bf16_shape, f32_shape}),
                  {key, value}, 0, /*is_stable=*/false, &builder,
                  module.get()));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(sort->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(sort->shape(), {0}).element_type(), F32);
  EXPECT_NE(computation->root_instruction(), sort);
  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(sort->to_apply()->parameter_instruction(1)->shape().element_type(),
            F32);
  // Make sure that no convert to BF16 was added to the 'to_apply' comparison
  // computation.
  auto users = sort->to_apply()->parameter_instruction(1)->users();
  for (auto user : users) {
    EXPECT_NE(user->opcode(), HloOpcode::kConvert);
  }
}

// Tests that the normalization should not cause unsupported mixed precision due
// to resolving unsupported BF16 operand.
TEST_F(BFloat16NormalizationTest, DoNotAddUnsupportedMixedPrecision) {
  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {4, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateDot(bf16_shape, a, b, dot_dnums, precision_config));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->operand(1)->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kConvert);
}

TEST_F(BFloat16NormalizationTest, DoNotChangeBitcastConvert) {
  auto builder = HloComputation::Builder(TestName());
  Shape u16_shape = ShapeUtil::MakeShape(U16, {4, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {4, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, u16_shape, "a"));

  builder.AddInstruction(HloInstruction::CreateBitcastConvert(bf16_shape, a));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(Normalize(module.get()));
  auto root = computation->root_instruction();

  EXPECT_EQ(root->opcode(), HloOpcode::kBitcastConvert);
  EXPECT_EQ(root->shape().element_type(), BF16);
  EXPECT_EQ(root->operand(0)->shape().element_type(), U16);
}

}  // namespace xla
