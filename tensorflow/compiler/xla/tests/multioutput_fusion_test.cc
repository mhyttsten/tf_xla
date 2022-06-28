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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc() {
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

#include <math.h>

#include <algorithm>
#include <memory>
#include <new>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace {

class MultiOutputFusionTest : public HloTestBase {
 protected:
  MultiOutputFusionTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/tests/multioutput_fusion_test.cc", "MultiOutputFusionTest");
 error_spec_ = ErrorSpec{0.0001, 1e-2}; }

  // Layout assignment assumes that there are no fusions in the input graph.
  // Since the purpose of this test is to send pre-fused graphs to XLA, we have
  // to do layout assignment ourselves.
  DebugOptions GetDebugOptionsForTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/xla/tests/multioutput_fusion_test.cc", "GetDebugOptionsForTest");

    auto opts = HloTestBase::GetDebugOptionsForTest();
    opts.add_xla_disable_hlo_passes("layout-assignment");
    return opts;
  }

  void RunTest2D(bool manual_fusion, int64_t size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/tests/multioutput_fusion_test.cc", "RunTest2D");

    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewVerifiedModule();

    const Shape elem_shape0 = ShapeUtil::MakeShapeWithLayout(F32, {}, {});
    const Shape elem_shape2 =
        ShapeUtil::MakeShapeWithLayout(F32, {size, size}, {1, 0});

    auto const0 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(8.0f)));
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape0, "0"));

    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape0, HloOpcode::kAdd, param0, const0));

    HloInstruction* broadcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(elem_shape2, add1, {}));

    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape2, "1"));

    HloInstruction* add2 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kAdd, broadcast, param1));
    HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kSubtract, param1, broadcast));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
        elem_shape2, sub, add2, dot_dnums, DefaultPrecisionConfig(2)));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple =
          computation->AddInstruction(HloInstruction::CreateTuple({sub, add2}));
      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 1));
      TF_CHECK_OK(dot->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(dot->ReplaceOperandWith(1, gte1));

      CHECK_NE(
          computation->CreateFusionInstruction(
              {tuple, sub, add2, broadcast}, HloInstruction::FusionKind::kLoop),
          nullptr);
    }

    Literal arg1(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, size}));
    arg1.PopulateWithValue<float>(2.5f);

    Literal expect(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, size}));
    expect.PopulateWithValue<float>(size * 1.5f * 3.5f);
    Literal literal_r0 = LiteralUtil::CreateR0<float>(-9.0f);
    auto actual =
        ExecuteAndTransfer(std::move(hlo_module), {&literal_r0, &arg1});
    EXPECT_TRUE(LiteralTestUtil::Near(expect, actual, error_spec_));
  }

  void RunTest1D(bool manual_fusion, int size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmultioutput_fusion_testDTcc mht_3(mht_3_v, 301, "", "./tensorflow/compiler/xla/tests/multioutput_fusion_test.cc", "RunTest1D");

    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewVerifiedModule();

    const Shape elem_shape_F32 =
        ShapeUtil::MakeShapeWithDescendingLayout(F32, {size});
    const Shape elem_shape_U8 =
        ShapeUtil::MakeShapeWithDescendingLayout(F64, {size});
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape_F32, "0"));
    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape_U8, "1"));

    HloInstruction* param0_U8 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_U8, param0));
    HloInstruction* param1_F32 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, param1));
    HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape_F32, HloOpcode::kAdd, param0, param1_F32));
    HloInstruction* sub_U8 =
        builder.AddInstruction(HloInstruction::CreateBinary(
            elem_shape_U8, HloOpcode::kSubtract, param0_U8, param1));
    HloInstruction* sub = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, sub_U8));

    HloInstruction* reshape =
        builder.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, 1}), add));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(0);
    dot_dnums.add_rhs_contracting_dimensions(0);
    HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
        ShapeUtil::MakeShapeWithDescendingLayout(F32, {1}), sub, reshape,
        dot_dnums, DefaultPrecisionConfig(2)));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple = computation->AddInstruction(
          HloInstruction::CreateTuple({sub_U8, add}));

      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_U8, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_F32, tuple, 1));
      TF_CHECK_OK(sub->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(reshape->ReplaceOperandWith(0, gte1));

      CHECK_NE(computation->CreateFusionInstruction(
                   {tuple, sub_U8, add, param0_U8, param1_F32},
                   HloInstruction::FusionKind::kLoop),
               nullptr);
    }

    Literal input0(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size}));
    input0.PopulateWithValue(2.5f);
    Literal input1(ShapeUtil::MakeShapeWithDescendingLayout(F64, {size}));
    input1.PopulateWithValue(1.);

    Literal expect = LiteralUtil::CreateR1<float>({size * 1.5f * 3.5f});
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {&input0, &input1});
    EXPECT_TRUE(LiteralTestUtil::Near(expect, actual, error_spec_));
  }
};

XLA_TEST_F(MultiOutputFusionTest, 2DNofusion) { RunTest2D(false, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusion) { RunTest2D(true, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusionSize129) { RunTest2D(true, 129); }
XLA_TEST_F(MultiOutputFusionTest, DifferentTypesNoFusion) {
  RunTest1D(false, 8);
}
XLA_TEST_F(MultiOutputFusionTest, DifferentTypesFusion) { RunTest1D(true, 8); }

XLA_TEST_F(MultiOutputFusionTest, FusionNodeIsRoot) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      x.param_0 = (((s32[]), f32[]), (f32[], s32[])) parameter(0)
      gte.3 = ((s32[]), f32[]) get-tuple-element(x.param_0), index=0
      gte.2 = (s32[]) get-tuple-element(gte.3), index=0
      gte.4 = s32[] get-tuple-element(gte.2), index=0
      copy = s32[] copy(gte.4)
      ROOT tuple = (s32[]) tuple(copy)
    }

    ENTRY thing.v3 {
      x = (((s32[]), f32[]), (f32[], s32[])) parameter(0)
      ROOT fusion = (s32[]) fusion(x), kind=kLoop, calls=fused_computation
    }
  )";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::MakeTupleOwned(
      LiteralUtil::MakeTupleOwned(
          LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<int32_t>(42)),
          LiteralUtil::CreateR0<float>(1.0)),
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<float>(3.0),
                                  LiteralUtil::CreateR0<int32_t>(4)));
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<int32_t>(42)), result));
}

XLA_TEST_F(MultiOutputFusionTest, MultiOutputLoopFusion) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      p = f32[4] parameter(0)
      multiply = f32[4] multiply(p, p)
      less-than = pred[4] compare(p, multiply), direction=LT
      ROOT tuple = (pred[4], f32[4]) tuple(less-than, multiply)
    }

    ENTRY PredFloatMOF {
      p0 = f32[4] parameter(0)
      fusion = (pred[4], f32[4]) fusion(p0), kind=kLoop, calls=fused_computation
      gte0 = pred[4] get-tuple-element(fusion), index=0
      gte1 = f32[4] get-tuple-element(fusion), index=1
      const = f32[4] constant({0, 0, 0, 0})
      ROOT select = f32[4] select(gte0, gte1, const)
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, -1.0});
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  LiteralTestUtil::ExpectR1Equal<float>({0.0, 4.0, 9.0, 1.0}, result);
}

XLA_TEST_F(MultiOutputFusionTest, MultiOutputLoopFeedingMap) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      p = f32[] parameter(0)
      multiply = f32[] multiply(p, p)
      less-than = pred[] compare(p, multiply), direction=LT
      ROOT tuple = (pred[], f32[]) tuple(less-than, multiply)
    }

    map_computation {
      p0 = f32[] parameter(0)
      fusion = (pred[], f32[]) fusion(p0), kind=kLoop, calls=fused_computation
      gte0 = pred[] get-tuple-element(fusion), index=0
      gte1 = f32[] get-tuple-element(fusion), index=1
      const = f32[] constant(0)
      ROOT select = f32[] select(gte0, gte1, const)
    }

    ENTRY MapMOF {
      p1 = f32[3] parameter(0)
      ROOT map = f32[3] map(p1), to_apply=map_computation
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  LiteralTestUtil::ExpectR1Equal<float>({0.0, 4.0, 9.0}, result);
}

const char* const kScalarOps = R"(
    HloModule m, is_scheduled=true

    Add {
      lhsadd = f32[] parameter(0)
      rhsadd = f32[] parameter(1)
      ROOT add = f32[] add(lhsadd, rhsadd)
    }

    Max {
      lhsmax = f32[] parameter(0)
      rhsmax = f32[] parameter(1)
      ROOT max = f32[] maximum(lhsmax, rhsmax)
    }
)";

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMinor)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0, c0), dimensions={2}, to_apply=Add
      mul = f32[32,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[32,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMajor)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0, c0), dimensions={0}, to_apply=Add
      mul = f32[32,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={0}, to_apply=Max
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[32,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionScalar)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32]{0} reduce(p0, c0), dimensions={0,2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(1.17549e-38)
      r2 = f32[32]{0} reduce(mul, c1), dimensions={0,2}, to_apply=Max
      r3 = f32[32]{0} reduce(mul, c0), dimensions={0,2}, to_apply=Add
      ROOT tuple = (f32[32]{0}, f32[32]{0}, f32[32]{0}) tuple(r1, r2, r3)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32]{0}, f32[32]{0}, f32[32]{0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMinorWithExtraOutput)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[2,32]{1,0} reduce(p0, c0), dimensions={2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[2,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32,32]{2,1,0}, f32[2,32]{1,0}, f32[2,32]{1,0})
                     tuple(p0, r1, r2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[2,32,32]{2,1,0}, f32[2,32]{1,0}, f32[2,32]{1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMajorWithExtraOutput)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,2]{1,0} reduce(p0, c0), dimensions={0}, to_apply=Add
      mul = f32[32,32,2]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,2]{1,0} reduce(mul, c1), dimensions={0}, to_apply=Max
      ROOT tuple = (f32[32,2]{1,0}, f32[32,32,2]{2,1,0}, f32[32,2]{1,0})
                     tuple(r1, mul, r2)
    }

    ENTRY reduce {
      p = f32[32,32,2]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,2]{1,0}, f32[32,32,2]{2,1,0}, f32[32,2]{1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionScalarWithExtraOutput)) {
  const std::string testcase = R"(
    HloModule m, is_scheduled=true

    Add {
      lhsadd = f32[] parameter(0)
      rhsadd = f32[] parameter(1)
      ROOT add = f32[] add(lhsadd, rhsadd)
    }
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32]{0} reduce(p0, c0), dimensions={0,2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      b1 = f32[2,32,32]{2,1,0} broadcast(c1), dimensions={}
      mul2 = f32[2,32,32]{2,1,0} multiply(p0, b1)
      ROOT tuple = (f32[32]{0}, f32[2,32,32]{2,1,0}, f32[2,32,32]{2,1,0})
        tuple(r1, mul, mul2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32]{0}, f32[2,32,32]{2,1,0}, f32[2,32,32]{2,1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionNonConstInit)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      init1 = f32[] parameter(1)
      init2 = f32[] parameter(2)
      r1 = f32[2,32]{1,0} reduce(p0, init1), dimensions={2}, to_apply=Add
      r2 = f32[2,32]{1,0} reduce(p0, init2), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32]{1,0}, f32[2,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      i = f32[] parameter(1)
      j = f32[] parameter(2)
      ROOT fusion = (f32[2,32]{1,0}, f32[2,32]{1,0}) fusion(p, i, j),
       kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionDifferentElementTypes)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce (p0: f16[2,32,32]) -> (f32[2,32], f32[2,32], f16[2,32,32]) {
      p0 = f16[2,32,32]{2,1,0} parameter(0)
      convert = f32[2,32,32]{2,1,0} convert(p0)
      c0 = f32[] constant(0)
      r1 = f32[2,32]{1,0} reduce(convert, c0), dimensions={2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(convert, convert)
      c1 = f32[] constant(5)
      r2 = f32[2,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32]{1,0}, f32[2,32]{1,0}, f16[2,32,32]{2,1,0})
                   tuple(r1, r2, p0)
    }

    ENTRY reduce {
      p = f16[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[2,32]{1,0}, f32[2,32]{1,0}, f16[2,32,32]{2,1,0}) fusion(p),
                    kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

}  // namespace
}  // namespace xla
