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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc() {
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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"
#include "tensorflow/compiler/xla/util.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {

// T is the Primitive Type of the complex number
// Test parameter is a tuple containing
//   - primitive type under test,
//   - two FpValues representing the values for the real and imaginary
//     components. The complex numbers for the test input is the cartesian
//     product of the values represented by the two FpValues.
template <PrimitiveType T>
class ExhaustiveComplexUnaryTestBase
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

  void SetParamsForTanh() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "SetParamsForTanh");

    // TODO(b/138126045): Current libc++ implementation of the complex tanh
    //                    function returns (NaN, NaN) when the imaginary
    //                    component is more than half of the max value.
    // TODO(b/138750327): Current libc++ implementation of the complex tanh
    //                    function returns (1, 0) when the real component is
    //                    negative infinity, when it should return (-1, 0).
    // We only need to set the former as incorrect values for C128 because when
    // testing with C64, we first cast our input to a C128 value.
    this->known_incorrect_fn_ = [&](int64_t v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

      double f = this->ConvertValue(v);
      return (T == C128 &&
              std::abs(f) > std::numeric_limits<double>::max() / 2) ||
             f == -std::numeric_limits<double>::infinity();
    };
  }

 private:
  // Generates the input complex literal given the FpValues representation for
  // the real and imaginary components.
  void FillInput(std::array<Literal, 1>* input_literal) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "FillInput");

    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());

    VLOG(2) << " testing input total "
            << real_values.GetTotalNumValues() * imag_values.GetTotalNumValues()
            << ", range " << real_values.ToString() << " "
            << imag_values.ToString();

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();

    uint64_t i = 0;
    for (auto real : real_values) {
      for (auto imag : imag_values) {
        input_arr[i] =
            NativeT(this->ConvertAndReplaceKnownIncorrectValueWith(real, 1),
                    this->ConvertAndReplaceKnownIncorrectValueWith(imag, 1));

        ++i;
      }
    }
  }

  int64_t GetInputSize() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "GetInputSize");

    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());
    return real_values.GetTotalNumValues() * imag_values.GetTotalNumValues();
  }
};

using ExhaustiveC64UnaryTest = ExhaustiveComplexUnaryTestBase<C64>;

using ExhaustiveC128UnaryTest = ExhaustiveComplexUnaryTestBase<C128>;

#define UNARY_TEST_COMPLEX_64(test_name, ...)   \
  XLA_TEST_P(ExhaustiveC64UnaryTest, test_name) \
  __VA_ARGS__

// TODO(b/138578594): Enable the test for the CPU backend after fixing the bug.
UNARY_TEST_COMPLEX_64(DISABLED_ON_CPU(Log), {
  // TODO(timshen): see b/162664705.
  known_incorrect_fn_ = [this](int64_t val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_4(mht_4_v, 282, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

    return std::isnan(this->ConvertValue(val));
  };
  Run(Log, [](complex64 x) { return std::log<float>(x); });
})

UNARY_TEST_COMPLEX_64(Sqrt, {
  Run(Sqrt, [](complex64 x) {
    return static_cast<complex64>(
        std::sqrt<double>(static_cast<complex128>(x)));
  });
})

UNARY_TEST_COMPLEX_64(Rsqrt, {
  Run(Rsqrt, [](complex64 x) {
    return static_cast<complex64>(
        complex128(1, 0) / std::sqrt<double>(static_cast<complex128>(x)));
  });
})

// The current libc++ implementation of the complex tanh function provides
// less accurate results when the denomenator of a complex tanh is small, due
// to floating point precision loss. To avoid this issue for complex64 numbers,
// we cast it to and from a complex128 when computing tanh.
UNARY_TEST_COMPLEX_64(Tanh, {
  SetParamsForTanh();
  ErrorSpecGen error_spec_gen = +[](complex64 x) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_5(mht_5_v, 311, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

    // This implementation of Tanh becomes less accurate when the denominator
    // is small.
    if (std::cosh(2 * x.real()) + std::cos(2 * x.imag()) < 1e-4) {
      return ErrorSpec{5e-2, 5e-2};
    }

    return GetDefaultSpecGenerator()(x);
  };
  Run(
      Tanh,
      +[](complex64 x) {
        return static_cast<complex64>(std::tanh(static_cast<complex128>(x)));
      },
      error_spec_gen);
})

INSTANTIATE_TEST_SUITE_P(
    F32SpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32SpecialAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(10000))));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndSpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(10000)),
                       ::testing::Values(GetNormals<float>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 4000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    F32LargeAndSmallMagnitudeNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         4000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 4000))));

#define UNARY_TEST_COMPLEX_128(test_name, ...)   \
  XLA_TEST_P(ExhaustiveC128UnaryTest, test_name) \
  __VA_ARGS__

UNARY_TEST_COMPLEX_128(Log, {
  // TODO(b/138578313): Enable the test for all values after fixing the bug.
  known_incorrect_fn_ = [&](int64_t v) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_6(mht_6_v, 370, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

    double f = this->ConvertValue(v);
    return std::fpclassify(f) == FP_NAN || std::abs(f) > 1.0e+300 ||
           std::abs(f) < 1.0e-300;
  };
  Run(Log, [](complex128 x) { return std::log<double>(x); });
})

UNARY_TEST_COMPLEX_128(Sqrt, {
  // Similar to the Tanh bug.
  known_incorrect_fn_ = [&](int64_t v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_7(mht_7_v, 383, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

    double f = this->ConvertValue(v);
    return std::abs(f) > std::numeric_limits<double>::max() / 2;
  };
  Run(Sqrt, [](complex128 x) { return std::sqrt<double>(x); });
})

UNARY_TEST_COMPLEX_128(Rsqrt, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    // Edge case on CUDA backend where the Log of a complex number made up of
    // the smallest denormals is more accurate than the interpreter backend.
    error_spec_gen = [](complex128 x) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_complexDTcc mht_8(mht_8_v, 398, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_complex.cc", "lambda");

      constexpr double denorm_min = std::numeric_limits<double>::denorm_min();
      if (std::abs(x.real()) == denorm_min &&
          std::abs(x.imag()) == denorm_min) {
        return ErrorSpec(0.5, 0.5);
      }
      return GetDefaultSpecGenerator()(x);
    };
  }
  Run(
      Rsqrt,
      [](complex128 x) { return complex128(1, 0) / std::sqrt<double>(x); },
      error_spec_gen);
})

UNARY_TEST_COMPLEX_128(Tanh, {
  SetParamsForTanh();
  Run(
      Tanh, +[](complex128 x) { return std::tanh(x); });
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(10000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(10000)),
                       ::testing::Values(GetNormals<double>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));

}  // namespace exhaustive_op_test
}  // namespace xla
