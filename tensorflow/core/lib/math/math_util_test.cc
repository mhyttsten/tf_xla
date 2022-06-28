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
class MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/math/math_util.h"

#include <cmath>
#include <limits>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Number of arguments for each test of the CeilOrRatio method
const int kNumTestArguments = 4;

template <typename IntegralType, typename TestDataType>
void TestCeilOfRatio(const TestDataType test_data[][kNumTestArguments],
                     int num_tests) {
  for (int i = 0; i < num_tests; ++i) {
    const IntegralType numerator = test_data[i][0];
    const IntegralType denominator = test_data[i][1];
    const IntegralType expected_floor = test_data[i][2];
    const IntegralType expected_ceil = test_data[i][3];
    // Make sure the two ways to compute the floor return the same thing.
    IntegralType floor_1 = MathUtil::FloorOfRatio(numerator, denominator);
    IntegralType floor_2 = MathUtil::CeilOrFloorOfRatio<IntegralType, false>(
        numerator, denominator);
    EXPECT_EQ(floor_1, floor_2);
    EXPECT_EQ(expected_floor, floor_1)
        << "FloorOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator << " "
        << (8 * sizeof(IntegralType)) << " bits";
    IntegralType ceil_1 = MathUtil::CeilOfRatio(numerator, denominator);
    IntegralType ceil_2 = MathUtil::CeilOrFloorOfRatio<IntegralType, true>(
        numerator, denominator);
    EXPECT_EQ(ceil_1, ceil_2);
    EXPECT_EQ(expected_ceil, ceil_1)
        << "CeilOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator << " "
        << (8 * sizeof(IntegralType)) << " bits";
  }
}

template <typename UnsignedIntegralType>
void TestCeilOfRatioUnsigned(uint64 kMax) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestCeilOfRatioUnsigned");

  const int kNumTests = 12;
  const uint64 kTestData[kNumTests][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, 2, 0, 0},
      {0, kMax, 0, 0},
      // Try some non-extreme cases
      {1, 1, 1, 1},
      {5, 2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge numerator and a huge denominator
      {kMax, kMax, 1, 1},
  };
  TestCeilOfRatio<UnsignedIntegralType, uint64>(kTestData, kNumTests);
}

template <typename SignedInteger>
void TestCeilOfRatioSigned(int64_t kMin, int64_t kMax) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_1(mht_1_v, 261, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestCeilOfRatioSigned");

  const int kNumTests = 30;
  const int64_t kTestData[kNumTests][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, -1, 0, 0},
      {0, 2, 0, 0},
      {0, kMin, 0, 0},
      {0, kMax, 0, 0},
      // Try all four combinations of 1 and -1
      {1, 1, 1, 1},
      {-1, 1, -1, -1},
      {1, -1, -1, -1},
      {-1, -1, 1, 1},
      // Try all four combinations of +/-5 divided by +/- 2
      {5, 2, 2, 3},
      {-5, 2, -3, -2},
      {5, -2, -3, -2},
      {-5, -2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, -1, -kMax, -kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with huge negative numerator
      {kMin, 1, kMin, kMin},
      {kMin, 2, kMin / 2 - ((kMin % 2 != 0) ? 1 : 0), kMin / 2},
      {kMin, 3, kMin / 3 - ((kMin % 3 != 0) ? 1 : 0), kMin / 3},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge negative denominator
      {1, kMin, -1, 0},
      {2, kMin, -1, 0},
      {3, kMin, -1, 0},
      // Try with a huge numerator and a huge denominator
      {kMin, kMin, 1, 1},
      {kMin, kMax, -2, -1},
      {kMax, kMin, -1, 0},
      {kMax, kMax, 1, 1},
  };
  TestCeilOfRatio<SignedInteger, int64_t>(kTestData, kNumTests);
}

// ------------------------------------------------------------------------ //
// Benchmarking CeilOrFloorOfRatio
//
// We compare with other implementations that are unsafe in general.
// ------------------------------------------------------------------------ //

// An implementation of CeilOfRatio that is correct for small enough values,
// and provided that the numerator and denominator are both positive
template <typename IntegralType>
static IntegralType CeilOfRatioDenomMinusOne(IntegralType numerator,
                                             IntegralType denominator) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_2(mht_2_v, 321, "", "./tensorflow/core/lib/math/math_util_test.cc", "CeilOfRatioDenomMinusOne");

  const IntegralType kOne(1);
  return (numerator + denominator - kOne) / denominator;
}

// An implementation of FloorOfRatio that is correct when the denominator is
// positive and the numerator non-negative
template <typename IntegralType>
static IntegralType FloorOfRatioByDivision(IntegralType numerator,
                                           IntegralType denominator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_3(mht_3_v, 333, "", "./tensorflow/core/lib/math/math_util_test.cc", "FloorOfRatioByDivision");

  return numerator / denominator;
}

template <typename Integer, bool ComputeCeil>
static Integer CeilOrFloorOfRatioArithmetic(Integer numerator,
                                            Integer denominator) {
  if (ComputeCeil) {
    return CeilOfRatioDenomMinusOne(numerator, denominator);
  } else {
    return FloorOfRatioByDivision(numerator, denominator);
  }
}

void TestThatCeilOfRatioDenomMinusOneIsIncorrect(int64_t numerator,
                                                 int64_t denominator,
                                                 int64_t expected_error) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestThatCeilOfRatioDenomMinusOneIsIncorrect");

  const int64_t correct_result = MathUtil::CeilOfRatio(numerator, denominator);
  const int64_t result_by_denom_minus_one =
      CeilOfRatioDenomMinusOne(numerator, denominator);
  EXPECT_EQ(result_by_denom_minus_one + expected_error, correct_result)
      << "numerator = " << numerator << " denominator = " << denominator
      << " expected error = " << expected_error
      << " Actual difference: " << (correct_result - result_by_denom_minus_one);
}

// Here we demonstrate why not to use CeilOfRatioDenomMinusOne
void TestThatCeilOfRatioDenomMinusOneIsIncorrect() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_5(mht_5_v, 366, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestThatCeilOfRatioDenomMinusOneIsIncorrect");

  // It does not work with negative values
  TestThatCeilOfRatioDenomMinusOneIsIncorrect(-1LL, -2LL, -1LL);

  // This would also fail if given kint64max because of signed integer overflow.
}

TEST(MathUtil, CeilOfRatio) {
  TestCeilOfRatioUnsigned<uint8>(kuint8max);
  TestCeilOfRatioUnsigned<uint16>(kuint16max);
  TestCeilOfRatioUnsigned<uint32>(kuint32max);
  TestCeilOfRatioUnsigned<uint64>(kuint64max);
  TestCeilOfRatioSigned<int8>(kint8min, kint8max);
  TestCeilOfRatioSigned<int16>(kint16min, kint16max);
  TestCeilOfRatioSigned<int32>(kint32min, kint32max);
  TestCeilOfRatioSigned<int64_t>(kint64min, kint64max);
#if 0
  TestThatCeilOfRatioDenomMinusOneIsIncorrect();
#endif
}

struct GCDTestCase {
  unsigned int x;
  unsigned int y;
  unsigned int gcd;
};

TEST(MathUtil, GCD) {
  std::vector<GCDTestCase> testcases({
      {10, 20, 10},  //
      {27, 8, 1},    //
      {4, 3, 1},     //
      {6, 8, 2},     //
      {5, 0, 5},     //
      {5, 5, 5},     //
      {0, 0, 0}      //
  });

  for (const auto& tc : testcases) {
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint32>(tc.x, tc.y));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint32>(tc.y, tc.x));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint64>(tc.x, tc.y));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint64>(tc.y, tc.x));
  }

  const uint64 biggish_prime = 1666666667;
  EXPECT_EQ(biggish_prime,
            MathUtil::GCD<uint64>(biggish_prime * 3, biggish_prime * 4));
}

template <typename T>
void TestOneIPowN() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_6(mht_6_v, 420, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestOneIPowN");

  const T one{1};
  for (int i = 0; i < 1024; ++i) {
    // Computations are exact.
    EXPECT_EQ(MathUtil::IPow(one, i), one);
  }
}

template <typename T>
void TestTwoIPowN() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_7(mht_7_v, 432, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestTwoIPowN");

  int limit = std::is_integral<T>::value ? std::numeric_limits<T>::digits : 63;
  for (int i = 0; i < limit; ++i) {
    // Computations are exact.
    EXPECT_EQ(MathUtil::IPow(T{2}, i), static_cast<T>(1ull << i));
  }
}

template <typename T>
void TestFloatIPow(const int max_exponent, const T start, const T end,
                   const T step) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSmathPSmath_util_testDTcc mht_8(mht_8_v, 445, "", "./tensorflow/core/lib/math/math_util_test.cc", "TestFloatIPow");

  for (T f = start; f < end; f += step) {
    for (int i = 0; i < max_exponent; ++i) {
      EXPECT_FLOAT_EQ(MathUtil::IPow(f, i), std::pow(f, i));
    }
  }
}

TEST(MathUtil, IPow) {
  TestOneIPowN<double>();
  TestOneIPowN<float>();
  TestOneIPowN<int>();
  TestOneIPowN<int64_t>();
  TestTwoIPowN<double>();
  TestTwoIPowN<float>();
  TestTwoIPowN<int>();
  TestTwoIPowN<int64_t>();

  EXPECT_EQ(MathUtil::IPow(3, 0), 1);
  EXPECT_EQ(MathUtil::IPow(3, 1), 3);
  EXPECT_EQ(MathUtil::IPow(3, 2), 9);
  EXPECT_EQ(MathUtil::IPow(3, 3), 27);
  EXPECT_EQ(MathUtil::IPow(3, 4), 81);
  EXPECT_EQ(MathUtil::IPow(3, 5), 243);

  TestFloatIPow<float>(13, -16.0f, 16.0f, 1.0f / 8);
  TestFloatIPow<double>(13, -16.0, 16.0, 1.0 / 8);

  TestFloatIPow<float>(13, -1.0f / (1 << 12), -1.0f / (1 << 12),
                       1.0f / (1 << 16));
  TestFloatIPow<double>(13, -1.0 / (1 << 12), -1.0 / (1 << 12),
                        1.0 / (1 << 16));
}

TEST(MathUtil, IPowEdgeCases) {
  constexpr const double kInf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(MathUtil::IPow(-12345.0, 79), -kInf);
  EXPECT_EQ(MathUtil::IPow(-12345.0, 80), +kInf);

  // The semantics of the edge cases that follow  are defined in the standard:
  // http://en.cppreference.com/w/cpp/numeric/math/pow for a summary.

  // 1 - These edge cases apply.
  // pow(+0, exp), where exp is a positive odd integer, returns +0
  EXPECT_EQ(MathUtil::IPow(+0.0, 3), +0.0);
  // pow(-0, exp), where exp is a positive odd integer, returns -0
  EXPECT_EQ(MathUtil::IPow(-0.0, 3), -0.0);
  // pow(±0, exp), where exp is positive non-integer or a positive even integer,
  // returns +0
  EXPECT_EQ(MathUtil::IPow(+0.0, 42), +0.0);
  EXPECT_EQ(MathUtil::IPow(-0.0, 42), +0.0);
  // pow(base, ±0) returns 1 for any base, even when base is NaN
  EXPECT_EQ(MathUtil::IPow(-kInf, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-2.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-1.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-0.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+0.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+1.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+2.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+kInf, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(std::numeric_limits<double>::quiet_NaN(), 0.0), 1.0);
  // pow(-∞, exp) returns -∞ if exp is a positive odd integer
  EXPECT_EQ(MathUtil::IPow(-kInf, 43), -kInf);
  // pow(-∞, exp) returns +∞ if exp is a positive non-integer or even integer
  EXPECT_EQ(MathUtil::IPow(-kInf, 42), +kInf);
  // pow(+∞, exp) returns +∞ for any positive exp
  EXPECT_EQ(MathUtil::IPow(+kInf, 42), +kInf);
  EXPECT_EQ(MathUtil::IPow(+kInf, 43), +kInf);

  // 2 - These do not apply due to the restricted exp range.
  // pow(+0, exp), where exp is a negative odd integer, returns +∞ and raises
  // FE_DIVBYZERO pow(-0, exp), where exp is a negative odd integer, returns -∞
  // and raises FE_DIVBYZERO pow(±0, exp), where exp is negative, finite, and is
  // an even integer or a non-integer, returns +∞ and raises FE_DIVBYZERO
  // pow(-1, ±∞) returns 1
  // pow(+1, exp) returns 1 for any exp, even when exp is NaN
  // pow(±0, -∞) returns +∞ and may raise FE_DIVBYZERO
  // pow(base, exp) returns NaN and raises FE_INVALID if base is finite and
  // negative and exp is finite and non-integer. pow(base, -∞) returns +∞ for
  // any |base|<1 pow(base, -∞) returns +0 for any |base|>1 pow(base, +∞)
  // returns +0 for any |base|<1 pow(base, +∞) returns +∞ for any |base|>1
  // pow(-∞, exp) returns -0 if exp is a negative odd integer
  // pow(-∞, exp) returns +0 if exp is a negative non-integer or even integer
  // pow(+∞, exp) returns +0 for any negative exp
}

}  // namespace
}  // namespace tensorflow
