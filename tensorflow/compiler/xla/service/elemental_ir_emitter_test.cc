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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitter_testDTcc() {
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

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using absl::nullopt;

class ElementalIrEmitterExecutionTest : public HloTestBase {
 protected:
  void RunTest(const std::string& hlo_text, absl::Span<Literal* const> args) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hlo_text: \"" + hlo_text + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitter_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter_test.cc", "RunTest");

    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), args, nullopt));
  }

  void RunTypeConversionTest(absl::string_view hlo_text) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSelemental_ir_emitter_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/service/elemental_ir_emitter_test.cc", "RunTypeConversionTest");

    HloModuleConfig config;
    auto debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_cpu_fast_math_honor_nans(true);
    debug_options.set_xla_cpu_fast_math_honor_infs(true);
    config.set_debug_options(debug_options);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
  }
};

XLA_TEST_F(ElementalIrEmitterExecutionTest, DotFusion) {
  const std::string hlo_text = R"(
HloModule FusedDot

fused_computation {
  arg0 = s32[1,2,1]{2,1,0} parameter(0)
  reshape.lhs = s32[2,1]{1,0} reshape(arg0)
  arg1 = s32[1,2,1]{2,1,0} parameter(1)
  reshape.rhs = s32[2,1]{1,0} reshape(arg1)
  ROOT dot = s32[1,1]{1,0} dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[1,2,1]{2,1,0} parameter(0)
  entry_arg1 = s32[1,2,1]{2,1,0} parameter(1)
  ROOT fusion = s32[1,1]{1,0} fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR3<int32_t>({{{1}, {2}}});
  Literal rhs = LiteralUtil::CreateR3<int32_t>({{{3}, {4}}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ScalarDotFusion) {
  const char* hlo_text = R"(
HloModule ScalarDotFusion

fused_computation {
  arg0 = s32[2,2]{1,0} parameter(0)
  reshape.lhs = s32[4]{0} reshape(arg0)
  arg1 = s32[2,2]{1,0} parameter(1)
  reshape.rhs = s32[4]{0} reshape(arg1)
  ROOT dot = s32[] dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[2,2]{1,0} parameter(0)
  entry_arg1 = s32[2,2]{1,0} parameter(1)
  ROOT fusion = s32[] fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal rhs = LiteralUtil::CreateR2<int32_t>({{10, 20}, {30, 40}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, BatchDot) {
  const char* hlo_text = R"(
HloModule BatchDot

fused_computation.1 {
  param_0 = f64[1,1,8]{2,1,0} parameter(0)
  r.1 = f64[2,4]{1,0} reshape(param_0)
  param_1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  r.2 = f64[2,4,1]{2,1,0} reshape(param_1)
  ROOT dot = f64[2,1]{1,0} dot(r.1, r.2), lhs_batch_dims={0},
                                          lhs_contracting_dims={1},
                                          rhs_batch_dims={0},
                                          rhs_contracting_dims={1}
}

ENTRY resampler_Resampler.49 {
  p0 = f64[1,1,8]{2,1,0} parameter(0)
  p1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  ROOT f = f64[2,1]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_computation.1
}
)";

  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  // Disable the layout assignment pass because it would throw away the layouts
  // in the fusion computation, but not recreate them.
  debug_options.add_xla_disable_hlo_passes("layout-assignment");
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithInfiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[8]{0} constant({
        (1, 1),     (1, inf),   (1, inf),   (nan, 1),
        (inf, inf), (inf, nan), (nan, nan), (1, 2)})
      real = f32[8]{0} constant({nan, nan, inf, inf, inf, 1, inf, 3})
      imag = f32[8]{0} constant({inf, inf, inf, inf, 1, inf, inf, 4})
      complex.2 = c64[8]{0} complex(real, imag)
      ROOT divide.1 = c64[8]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithFiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[5]{0} constant({
        (1, inf), (inf, 1), (inf, nan), (inf, inf), (nan, inf)})
      real = f32[5]{0} constant({1, 1, 1, 1, 1})
      imag = f32[5]{0} constant({1, 1, 1, 1, 1})
      complex.2 = c64[5]{0} complex(real, imag)
      ROOT divide.1 = c64[5]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithZeroNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[9]{0} constant({
        (1, 1),     (1, nan), (1, inf),   (inf, inf), (inf, 1),
        (inf, nan), (nan, 1), (nan, inf), (nan, nan)})
      real = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      imag = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      complex.2 = c64[9]{0} complex(real, imag)
      ROOT divide.1 = c64[9]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertFloatsToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16
        (f16_ f16[], f32_ f32[], f64_ f64[]) -> (bf16[], bf16[], bf16[]) {
      f16_ = f16[] parameter(0)
      f32_ = f32[] parameter(1)
      f64_ = f64[] parameter(2)
      converted_f16 = bf16[] convert(f16[] f16_)
      converted_f32 = bf16[] convert(f32[] f32_)
      converted_f64 = bf16[] convert(f64[] f64_)
      ROOT tuple = (bf16[], bf16[], bf16[]) tuple(converted_f16, converted_f32,
                                                  converted_f64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertSignedToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16 (s8_ s8[], s16_ s16[], s32_ s32[], s64_ s64[]) ->
        (bf16[], bf16[], bf16[], bf16[]) {
      s8_ = s8[] parameter(0)
      s16_ = s16[] parameter(1)
      s32_ = s32[] parameter(2)
      s64_ = s64[] parameter(3)
      converted_s8 = bf16[] convert(s8[] s8_)
      converted_s16 = bf16[] convert(s16[] s16_)
      converted_s32 = bf16[] convert(s32[] s32_)
      converted_s64 = bf16[] convert(s64[] s64_)
      ROOT tuple = (bf16[], bf16[], bf16[], bf16[]) tuple(
          converted_s8, converted_s16, converted_s32, converted_s64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertUnsignedToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16 (u8_ u8[], u16_ u16[], u32_ u32[], u64_ u64[]) ->
        (bf16[], bf16[], bf16[], bf16[]) {
      u8_ = u8[] parameter(0)
      u16_ = u16[] parameter(1)
      u32_ = u32[] parameter(2)
      u64_ = u64[] parameter(3)
      converted_u8 = bf16[] convert(u8[] u8_)
      converted_u16 = bf16[] convert(u16[] u16_)
      converted_u32 = bf16[] convert(u32[] u32_)
      converted_u64 = bf16[] convert(u64[] u64_)
      ROOT tuple = (bf16[], bf16[], bf16[], bf16[]) tuple(
          converted_u8, converted_u16, converted_u32, converted_u64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToFloat) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16
        (to_f16 bf16[], to_f32 bf16[], to_f64 bf16[]) -> (f16[], f32[], f64[]) {
      to_f16 = bf16[] parameter(0)
      to_f32 = bf16[] parameter(1)
      to_f64 = bf16[] parameter(2)
      f16_ = f16[] convert(bf16[] to_f16)
      f32_ = f32[] convert(bf16[] to_f32)
      f64_ = f64[] convert(bf16[] to_f64)
      ROOT tuple = (f16[], f32[], f64[]) tuple(f16_, f32_, f64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToSigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16(to_s8 bf16[], to_s16 bf16[], to_s32 bf16[],
                          to_s64 bf16[]) -> (s8[], s16[], s32[], s64[]) {
      to_s8 = bf16[] parameter(0)
      to_s16 = bf16[] parameter(1)
      to_s32 = bf16[] parameter(2)
      to_s64 = bf16[] parameter(3)
      s8_ = s8[] convert(bf16[] to_s8)
      s16_ = s16[] convert(bf16[] to_s16)
      s32_ = s32[] convert(bf16[] to_s32)
      s64_ = s64[] convert(bf16[] to_s64)
      ROOT tuple = (s8[], s16[], s32[], s64[]) tuple(s8_, s16_, s32_, s64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToUnsigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16(to_u8 bf16[], to_u16 bf16[], to_u32 bf16[],
                          to_u64 bf16[]) -> (u8[], u16[], u32[], u64[]) {
      to_u8 = bf16[] parameter(0)
      to_u16 = bf16[] parameter(1)
      to_u32 = bf16[] parameter(2)
      to_u64 = bf16[] parameter(3)
      u8_ = u8[] convert(bf16[] to_u8)
      u16_ = u16[] convert(bf16[] to_u16)
      u32_ = u32[] convert(bf16[] to_u32)
      u64_ = u64[] convert(bf16[] to_u64)
      ROOT tuple = (u8[], u16[], u32[], u64[]) tuple(u8_, u16_, u32_, u64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToComplex) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16
        (to_c64 bf16[], to_c128 bf16[]) -> (c64[], c128[]) {
      to_c64 = bf16[] parameter(0)
      to_c128 = bf16[] parameter(1)
      c64_ = c64[] convert(bf16[] to_c64)
      c128_ = c128[] convert(bf16[] to_c128)
      ROOT tuple = (c64[], c128[]) tuple(c64_, c128_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, CompareBF16) {
  constexpr char hlo_text[] = R"(
  HloModule compareBF16
  ENTRY main {
    p0 = bf16[4] parameter(0)
    p1 = bf16[4] parameter(1)
    ROOT cmp = pred[4] compare(p0, p1), direction=LT
})";

  Literal lhs = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal rhs = LiteralUtil::CreateR1<float>({4, 3, 2, 1});
  lhs = LiteralUtil::ConvertF32ToBF16(lhs);
  rhs = LiteralUtil::ConvertF32ToBF16(rhs);
  RunTest(hlo_text, {&lhs, &rhs});
}

}  // namespace
}  // namespace xla
