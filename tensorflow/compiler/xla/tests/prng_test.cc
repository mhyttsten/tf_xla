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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc() {
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

#include <limits>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class PrngTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  Literal UniformTest(T a, T b, absl::Span<const int64_t> dims,
                      int64_t seed = 42);

  // Computes the χ² statistic of a sample of the discrete uniform distribution
  // of the given range size. `expected_count` is the number of times each
  // possible value is expected to be generated. Thus, the sample size is
  // `range_size * expected_count`.
  double UniformChiSquared(int32_t range_size, int32_t expected_count,
                           int64_t seed = 42);
};

template <typename T>
Literal PrngTest::UniformTest(T a, T b, absl::Span<const int64_t> dims,
                              int64_t seed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "PrngTest::UniformTest");

  XlaBuilder builder(TestName());
  RngUniform(
      ConstantR0<T>(&builder, a), ConstantR0<T>(&builder, b),
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(), dims));

  SetSeed(seed);
  auto actual =
      ExecuteAndTransfer(&builder, /*arguments=*/{}).ConsumeValueOrDie();
  EXPECT_THAT(dims, ::testing::ElementsAreArray(actual.shape().dimensions()));
  actual.EachCell<T>([=](absl::Span<const int64_t>, T value) {
    EXPECT_LE(a, value);
    EXPECT_LT(value, b);
  });
  return actual;
}

// Uniform random number generation tests
XLA_TEST_F(PrngTest, ScalarU01) { UniformTest<float>(0, 1, {}); }
XLA_TEST_F(PrngTest, ScalarU01limits) {
  UniformTest<float>(std::numeric_limits<float>::min(),
                     std::numeric_limits<float>::max(), {});
}
XLA_TEST_F(PrngTest, ZeroValuesU01) { UniformTest<float>(0, 1, {0}); }
XLA_TEST_F(PrngTest, TenValuesU01) { UniformTest<float>(0, 1, {10}); }
XLA_TEST_F(PrngTest, TenValuesU37) { UniformTest<float>(3, 7, {10}); }
XLA_TEST_F(PrngTest, ZeroValuesR2) { UniformTest<float>(0, 1, {0, 20}); }
XLA_TEST_F(PrngTest, LargeU01) { UniformTest<float>(0, 1, {0x100, 0x100}); }
XLA_TEST_F(PrngTest, TwelveValuesU524) { UniformTest<int32_t>(5, 24, {12}); }

// TODO(b/71543667): Fix Rng ops on LLVM backends.
// TODO(b/122047800): Interpreter does not support BF16 for RNG ops.
using ScalarBF16TestCase = std::tuple<int64_t, std::pair<float, float>>;

class ScalarBF16Test
    : public PrngTest,
      public ::testing::WithParamInterface<ScalarBF16TestCase> {};

XLA_TEST_P(ScalarBF16Test,
           DISABLED_ON_INTERPRETER(DISABLED_ON_GPU(DISABLED_ON_CPU(DoIt)))) {
  auto test_params = GetParam();
  UniformTest<bfloat16>(static_cast<bfloat16>(std::get<1>(test_params).first),
                        static_cast<bfloat16>(std::get<1>(test_params).second),
                        {},
                        /*seed=*/std::get<0>(test_params));
}

INSTANTIATE_TEST_SUITE_P(
    ScalarBF16TestInstance, ScalarBF16Test,
    ::testing::Combine(
        ::testing::Range<int64_t>(0, 100),
        ::testing::Values(
            // The largest negative number smaller than zero in bf16 that's not
            // denormalized.
            std::make_pair(static_cast<float>(
                               -std::numeric_limits<Eigen::bfloat16>::min()),
                           0.0f),
            // Test odd and even values.
            std::make_pair(32.75f, 33.00f), std::make_pair(32.50f, 32.75f),
            std::make_pair(-33.00f, -32.75f),
            std::make_pair(-32.75f, -32.50f))));

// TODO(b/71543667): Fix Rng ops on LLVM backends.
// TODO(b/122047800): Interpreter does not support BF16 for RNG ops.
XLA_TEST_F(PrngTest, DISABLED_ON_INTERPRETER(DISABLED_ON_GPU(
                         DISABLED_ON_CPU(ScalarBF16CountTests)))) {
  // There are 3 BF16 values in the range of [32.25, 33): 32.25, 32.5, 32.75,
  // they should get similar counts.
  bfloat16 low = static_cast<bfloat16>(32.25);
  bfloat16 high = static_cast<bfloat16>(33);
  bfloat16 interval = static_cast<bfloat16>(0.25);
  std::vector<int32_t> counts(static_cast<int64_t>((high - low) / interval), 0);

  constexpr int64_t count = 1000;
  for (int64_t seed = 0; seed < count; ++seed) {
    auto result = UniformTest<bfloat16>(low, high, {}, /*seed=*/seed);
    result.EachCell<bfloat16>([&](absl::Span<const int64_t>, bfloat16 value) {
      int64_t index = static_cast<int64_t>((value - low) / interval);
      counts[index]++;
    });
  }
  // Each bucket should have similar amount of counts. That is, not more than
  // 10% of total counts. This mostly tests that we don't fall into a 1:2:2
  // distribution, which yields 20% expected difference.
  EXPECT_LT(std::abs(counts[0] - counts[1]), count * 0.1);
  EXPECT_LT(std::abs(counts[1] - counts[2]), count * 0.1);
}

namespace {
template <typename T>
T Square(T x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_1(mht_1_v, 314, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "Square");

  return x * x;
}
}  // namespace

double PrngTest::UniformChiSquared(int32_t range_size, int32_t expected_count,
                                   int64_t seed) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_2(mht_2_v, 323, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "PrngTest::UniformChiSquared");

  int32_t sample_size = range_size * expected_count;

  XlaBuilder builder(TestName());
  RngUniform(ConstantR0<int32_t>(&builder, 0),
             ConstantR0<int32_t>(&builder, range_size),
             ShapeUtil::MakeShape(S32, {sample_size}));

  SetSeed(seed);
  auto actual =
      ExecuteAndTransfer(&builder, /*arguments=*/{}).ConsumeValueOrDie();
  std::vector<int32_t> counts(range_size, 0);
  actual.EachCell<int32_t>(
      [&counts](absl::Span<const int64_t>, int32_t value) { ++counts[value]; });
  int64_t sum = 0;
  for (int32_t i = 0; i < range_size; ++i) {
    sum += Square(static_cast<int64_t>(counts[i] - expected_count));
  }
  return static_cast<double>(sum) / expected_count;
}

// We only test distribution of uniform discrete PRNG as other types are based
// on it.
// These range sizes are arbitrary but include prime numbers, powers of 2, and
// other composite numbers.
// The level of significance in all these cases is 1/20.
// TODO(b/35723038): Use parametrized tests where possible.
XLA_TEST_F(PrngTest, Uniformity7) {
  EXPECT_LT(UniformChiSquared(7, 256), 12.5916);
}
XLA_TEST_F(PrngTest, Uniformity61) {
  EXPECT_LT(UniformChiSquared(61, 256), 79.0819);
}
XLA_TEST_F(PrngTest, Uniformity64) {
  EXPECT_LT(UniformChiSquared(64, 256), 82.5287);
}
XLA_TEST_F(PrngTest, Uniformity108) {
  EXPECT_LT(UniformChiSquared(108, 256), 132.144);
}
XLA_TEST_F(PrngTest, Uniformity256) {
  EXPECT_LT(UniformChiSquared(256, 512), 293.248);
}

// TODO(b/134770669): May remove this test if we decide not to support map
//                    computations with kRng instructions.
XLA_TEST_F(PrngTest, DISABLED_ON_GPU(DISABLED_ON_CPU(MapUsingRng))) {
  // Build a x -> (x + U[0,1)) computation.
  auto build_sum_rng = [](XlaBuilder& builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_3(mht_3_v, 373, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "lambda");

    auto b = builder.CreateSubBuilder("sum_with_rng");
    auto x = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "input");
    Add(x,
        RngUniform(ConstantR0<float>(b.get(), 0), ConstantR0<float>(b.get(), 1),
                   ShapeUtil::MakeShape(F32, {})));
    return b->BuildAndNoteError();
  };

  XlaBuilder builder(TestName());
  Literal param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 5.3f, 4.4f, 5.5f});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> param0_data,
                          client_->TransferToServer(param0_literal));

  auto param0 = Parameter(&builder, 0, param0_literal.shape(), "param0");
  auto fn = build_sum_rng(builder);
  Map(&builder, {param0}, fn, {0});

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  ExecutionOptions execution_options = execution_options_;
  execution_options.set_seed(125);
  TF_ASSERT_OK_AND_ASSIGN(
      auto actual, client_->ExecuteAndTransfer(
                       computation,
                       /*arguments=*/{param0_data.get()}, &execution_options));

  EXPECT_EQ(ShapeUtil::ElementsIn(actual.shape()),
            ShapeUtil::ElementsIn(param0_literal.shape()));
  for (int i = 0; i < ShapeUtil::ElementsIn(actual.shape()); ++i) {
    EXPECT_GE(actual.data<float>()[i], param0_literal.data<float>()[i]);
    EXPECT_LT(actual.data<float>()[i], param0_literal.data<float>()[i] + 1.0f);
  }
}

// This tests demonstrates the global seeding behavior.
// * If a seed is passed in via Execute (ExecuteAndTransfer) then the output
// is
//   fixed (i.e., there is a single output for a given seed);
// * If no seed is passed in then the output of every call can be different;
XLA_TEST_F(PrngTest, PassInGlobalRngSeed) {
  // Build a U[0,1) computation.
  auto build_computation = [this]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_4(mht_4_v, 419, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "lambda");

    XlaBuilder builder(TestName());
    RngUniform(ConstantR0<float>(&builder, 0), ConstantR0<float>(&builder, 1),
               ShapeUtil::MakeShape(F32, {10}));
    return builder.Build();
  };

  ExecutionOptions execution_options1 = execution_options_;
  execution_options1.set_seed(42);

  ExecutionOptions execution_options2 = execution_options_;
  execution_options2.set_seed(65);

  Literal result1;
  {
    TF_ASSERT_OK_AND_ASSIGN(auto computation, build_computation());
    TF_ASSERT_OK_AND_ASSIGN(
        result1, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options1));
  }
  Literal result2;
  Literal result3;
  {
    TF_ASSERT_OK_AND_ASSIGN(auto computation, build_computation());
    TF_ASSERT_OK_AND_ASSIGN(
        result2, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options1));
    TF_ASSERT_OK_AND_ASSIGN(
        result3, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options1));
  }

  Literal result4;
  Literal result5;
  Literal result6;
  {
    TF_ASSERT_OK_AND_ASSIGN(auto computation, build_computation());
    TF_ASSERT_OK_AND_ASSIGN(
        result4, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options2));
    TF_ASSERT_OK_AND_ASSIGN(
        result5, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options_));
    TF_ASSERT_OK_AND_ASSIGN(
        result6, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                             &execution_options_));
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(result1, result2));
  EXPECT_TRUE(LiteralTestUtil::Equal(result1, result3));
  EXPECT_FALSE(LiteralTestUtil::Equal(result1, result4));
  EXPECT_FALSE(LiteralTestUtil::Equal(result4, result5));
  EXPECT_FALSE(LiteralTestUtil::Equal(result5, result6));
}

// This test verifies that the two RNG instructions with the same parameters
// in the same HloComputation produces different values.
XLA_TEST_F(PrngTest, DifferentValuesForIdenticalRngNodesInSameComputation) {
  // Build a U[0,1) computation.
  auto build_computation = [this]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSprng_testDTcc mht_5(mht_5_v, 481, "", "./tensorflow/compiler/xla/tests/prng_test.cc", "lambda");

    XlaBuilder builder(TestName());
    auto a = RngUniform(ConstantR0<int32_t>(&builder, 0),
                        ConstantR0<int32_t>(&builder, 100),
                        ShapeUtil::MakeShape(S32, {10}));
    auto b = RngUniform(ConstantR0<int32_t>(&builder, 0),
                        ConstantR0<int32_t>(&builder, 100),
                        ShapeUtil::MakeShape(S32, {10}));
    Tuple(&builder, {a, b});
    return builder.Build();
  };

  ExecutionOptions execution_options = execution_options_;
  execution_options.set_seed(42);

  Literal result_tuple;
  {
    TF_ASSERT_OK_AND_ASSIGN(auto computation, build_computation());
    TF_ASSERT_OK_AND_ASSIGN(
        result_tuple, client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                                  &execution_options));
  }

  auto results = result_tuple.DecomposeTuple();
  ASSERT_EQ(results.size(), 2);

  EXPECT_FALSE(LiteralTestUtil::Equal(results[0], results[1]));
}

XLA_TEST_F(PrngTest, TenValuesN01) {
  XlaBuilder builder(TestName());
  RngNormal(ConstantR0<float>(&builder, 0), ConstantR0<float>(&builder, 1),
            ShapeUtil::MakeShape(F32, {10}));

  SetSeed(42);
  ExecuteAndTransfer(&builder, /*arguments=*/{}).ConsumeValueOrDie();
  // TODO(b/25995601): Test that resultant values are reasonable
}

XLA_TEST_F(PrngTest, RngUniformCrash) {
  XlaBuilder builder(TestName());

  // This used to crash XLA during LLVM IR generation for CPUs.
  RngUniform(ConstantR0<int32_t>(&builder, 0),
             ConstantR0<int32_t>(&builder, 1000 * 1000),
             ShapeUtil::MakeShape(S32, {}));
  SetSeed(0);
  ExecuteAndTransfer(&builder, /*arguments=*/{}).ConsumeValueOrDie();
}

}  // namespace
}  // namespace xla
