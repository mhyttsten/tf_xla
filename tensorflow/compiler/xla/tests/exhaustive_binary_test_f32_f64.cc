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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

// Exhaustive test for binary operations for float and double.
//
// Test parameter is a tuple of (FpValues, FpValues) describing the possible
// values for each operand. The inputs for the test are the Cartesian product
// of the possible values for the two operands.
template <PrimitiveType T>
class Exhaustive32BitOrMoreBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  using typename ExhaustiveBinaryTest<T>::NativeT;
  using ExhaustiveBinaryTest<T>::ConvertAndReplaceKnownIncorrectValueWith;

 private:
  int64_t GetInputSize() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "GetInputSize");

    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();
    return values_0.GetTotalNumValues() * values_1.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 2>* input_literals) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "FillInput");

    int64_t input_size = GetInputSize();
    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();

    VLOG(2) << " testing " << values_0.ToString() << " " << values_1.ToString()
            << "total values " << input_size;
    CHECK(input_size == (*input_literals)[0].element_count() &&
          input_size == (*input_literals)[1].element_count());

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();

    uint64_t i = 0;
    for (auto src0 : values_0) {
      for (auto src1 : values_1) {
        input_arr_0[i] = ConvertAndReplaceKnownIncorrectValueWith(src0, 1);
        input_arr_1[i] = ConvertAndReplaceKnownIncorrectValueWith(src1, 1);
        ++i;
      }
    }
    CHECK_EQ(i, input_size);
  }
};

using ExhaustiveF32BinaryTest = Exhaustive32BitOrMoreBinaryTest<F32>;

#define BINARY_TEST_FLOAT_32(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF32BinaryTest, test_name) \
  __VA_ARGS__

BINARY_TEST_FLOAT_32(Add, {
  auto host_add = [](float x, float y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_FLOAT_32(Sub, {
  auto host_sub = [](float x, float y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](float x, float y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(Div), {
  auto host_div = [](float x, float y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_5(mht_5_v, 281, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_FLOAT_32(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>);
})

BINARY_TEST_FLOAT_32(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>);
})

// It is more convenient to implement Abs(complex) as a binary op than a unary
// op, as the operations we currently support all have the same data type for
// the source operands and the results.
// TODO(bixia): May want to move this test to unary test if we will be able to
// implement Abs(complex) as unary conveniently.
//
// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(AbsComplex), {
  // TODO(timshen): see b/162664705.
  known_incorrect_fn_ = [this](int64_t val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_6(mht_6_v, 305, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");

    return std::isnan(this->ConvertValue(val));
  };
  auto host_abs_complex = [](float x, float y) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_7(mht_7_v, 311, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");

    return std::abs(std::complex<float>(x, y));
  };
  auto device_abs_complex = [](XlaOp x, XlaOp y) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_8(mht_8_v, 317, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return Abs(Complex(x, y)); };

  Run(device_abs_complex, host_abs_complex);
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(2000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(2000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(2000)),
                       ::testing::Values(GetNormals<float>(2000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test.
// Comparing with the unary tests, the binary tests use a smaller set of inputs
// for each sub-test to avoid timeout because the implementation of ExpectNear
// more than 2x slower for binary test.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 2000))));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
using ExhaustiveF64BinaryTest = Exhaustive32BitOrMoreBinaryTest<F64>;
#define BINARY_TEST_FLOAT_64(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF64BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_FLOAT_64(test_name, ...)
#endif

BINARY_TEST_FLOAT_64(Add, {
  auto host_add = [](double x, double y) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_9(mht_9_v, 370, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_FLOAT_64(Sub, {
  auto host_sub = [](double x, double y) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_10(mht_10_v, 378, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](double x, double y) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_11(mht_11_v, 387, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(Div), {
  auto host_div = [](double x, double y) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_12(mht_12_v, 396, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_FLOAT_64(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<double>);
})

BINARY_TEST_FLOAT_64(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<double>);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(AbsComplex), {
  // TODO(timshen): see b/162664705.
  known_incorrect_fn_ = [this](int64_t val) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_13(mht_13_v, 414, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");

    return std::isnan(this->ConvertValue(val));
  };
  auto host_abs_complex = [](double x, double y) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_14(mht_14_v, 420, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");

    return std::abs(std::complex<double>(x, y));
  };
  auto device_abs_complex = [](XlaOp x, XlaOp y) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_test_f32_f64DTcc mht_15(mht_15_v, 426, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_test_f32_f64.cc", "lambda");
 return Abs(Complex(x, y)); };

  Run(device_abs_complex, host_abs_complex);
})

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(1000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(1000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(1000)),
                       ::testing::Values(GetNormals<double>(1000))));

// Tests a total of 40000 ^ 2 inputs, with 1000 ^ 2 inputs in each sub-test.
// Similar to ExhaustiveF64BinaryTest, we use a smaller set of inputs for each
// for each sub-test comparing with the unary test to avoid timeout.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));
#endif  // !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
