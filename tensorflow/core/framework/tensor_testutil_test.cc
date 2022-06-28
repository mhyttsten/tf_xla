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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutil_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutil_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutil_testDTcc() {
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

#include "tensorflow/core/framework/tensor_testutil.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace test {
namespace {

using internal_test::IsClose;

template <typename T>
void TestEdgeCasesNear() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutil_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/framework/tensor_testutil_test.cc", "TestEdgeCasesNear");

  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::infinity(),
                      Eigen::NumTraits<T>::infinity(), 0.0, 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::lowest(),
                      Eigen::NumTraits<T>::highest(),
                      Eigen::NumTraits<double>::infinity(), 0.0));
  EXPECT_FALSE(
      IsClose(Eigen::NumTraits<T>::lowest(), Eigen::NumTraits<T>::highest(),
              static_cast<double>(Eigen::NumTraits<T>::highest()), 0.0));
  EXPECT_FALSE(IsClose(Eigen::NumTraits<T>::quiet_NaN(), T(0.0), 0.0, 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<T>::quiet_NaN(), 0.0, 0.0));
  EXPECT_FALSE(IsClose(Eigen::NumTraits<T>::quiet_NaN(), T(0.0),
                       Eigen::NumTraits<double>::infinity(), 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<double>::infinity(), 0.0));
}

// For debug printing. Example usage:
// dumpFloatingPointStorage<Eigen::half, uint16>(
//     static_cast<Eigen::half>(-2.71f));
// dumpFloatingPointStorage<float, uint32>(-2.718281f);
// dumpFloatingPointStorage <double, uint64>(-2.71828182846);
template <typename T, typename U>
void dumpFloatingPointStorage(T value) {
  U* integral = reinterpret_cast<U*>(&value);
  int shift_amount = (sizeof(U) << 3) - 1;
  int exponent_bits = 2 + (log2(sizeof(U)) * 3);
  U mask = static_cast<U>(1) << shift_amount;
  for (int bits = 0; bits <= shift_amount; ++bits) {
    std::cout << ((*integral & mask) > 0);
    if (bits == 0 || bits == exponent_bits) std::cout << " ";
    mask >>= 1;
  }
  std::cout << std::endl;
  printf("%.20lf\n", static_cast<double>(value));
}

TEST(TensorTestUtilTest, ExpectTensorNearHalf) {
  // Eigen::half has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  // The exponent is offset at 15.
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
  typedef Eigen::half T;

  // Trivial cases: equalities.
  EXPECT_TRUE(IsClose(static_cast<T>(1.0f), static_cast<T>(1.0f), 0.0, 0.0));
  EXPECT_TRUE(IsClose(static_cast<T>(0.0f), static_cast<T>(-0.0f), 0.0, 0.0));
  EXPECT_TRUE(
      IsClose(static_cast<T>(3.141592f), static_cast<T>(3.141592f), 0.0, 0.0));

  // 0 10010 0001111110 -> 1150/128 = 8.984375 vs
  // 0 10010 0001111111 -> 1151/128 = 8.9921875 (diff = 0.0078125)
  EXPECT_TRUE(
      IsClose(static_cast<T>(8.9875f), static_cast<T>(8.99f), 0.0078125, 0.0));
  EXPECT_FALSE(
      IsClose(static_cast<T>(8.9875f), static_cast<T>(8.99f), 0.007, 0.0));

  // 0 11000 0110100000 -> 1440/2 = 720 vs
  // 0 11000 0110100001 -> 1441/2 = 720.5 (diff = 0.5)
  EXPECT_TRUE(
      IsClose(static_cast<T>(720.2f), static_cast<T>(720.3f), 0.5, 0.0));
  EXPECT_FALSE(
      IsClose(static_cast<T>(720.2f), static_cast<T>(720.3f), 0.4, 0.0));

  // 0 11001 0011010010 -> 1234 vs
  // 0 11001 0011010011 -> 1235 (diff = 1)
  // Rounds to even (1234.5 -> 1234).
  EXPECT_TRUE(
      IsClose(static_cast<T>(1234.f), static_cast<T>(1235.f), 1.0, 0.0));
  EXPECT_FALSE(
      IsClose(static_cast<T>(1234.5f), static_cast<T>(1235.f), 0.5, 0.0));
  EXPECT_TRUE(
      IsClose(static_cast<T>(1234.5f), static_cast<T>(1235.f), 1.0, 0.0));

  // 1 10000 0101101100 -> -1388/512 = -2.7109375 vs
  // 1 10000 0101110001 -> -1393/512 = -2.720703125 (diff = 0.009765625)
  EXPECT_TRUE(
      IsClose(static_cast<T>(-2.71f), static_cast<T>(-2.72f), 0.01, 0.0));

  TestEdgeCasesNear<T>();
}

TEST(TensorTestUtilTest, ExpectTensorNearFloat) {
  // float has 1 sign bit, 8 exponent bits, and 23 mantissa bits.
  // The exponent offset is 127.
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  typedef float T;
  // Trivial cases: equalities.
  EXPECT_TRUE(IsClose(1.0f, 1.0f, 0.0f, 0.0f));
  EXPECT_TRUE(IsClose(0.0f, -0.0f, 0.0f, 0.0f));
  EXPECT_TRUE(IsClose(3.14159265359f, 3.14159265359f, 0.0f, 0.0f));

  // 0 10000010 00011111100110011001101 -> 9,424,077/2^20 vs
  // 0 10000010 00011111100110100110110 -> 9,424,182/2^20
  // diff = 105/2^20 = 0.000100135803223
  EXPECT_TRUE(IsClose(8.9875f, 8.9876f, 0.0001002f, 0.0f));
  EXPECT_FALSE(IsClose(8.9875f, 8.9876f, 0.0001f, 0.0f));

  // 0 10001000 01101000000110011101001 -> 11,799,785/2^14 vs
  // 0 10001000 01101000000110011101010 -> 11,799,786/2^14
  // diff = 1/2^14 = 0.00006103515625
  EXPECT_TRUE(IsClose(720.2017f, 720.2018f, 0.0001f, 0.0f));
  EXPECT_FALSE(IsClose(720.20175f, 720.20185f, 0.0001f, 0.0f));
  EXPECT_TRUE(IsClose(720.20175f, 720.20185f, 0.00013f, 0.0f));

  // 0 10011001 11010110111100110100010 -> 15,432,098*2^3 vs
  // 0 10011001 11010110111100110100011 -> 15,432,099*2^3 (diff = 2^3 = 8)
  EXPECT_FALSE(IsClose(123456788.f, 123456789.f, 4.0f, 0.0f));
  EXPECT_TRUE(IsClose(123456788.f, 123456789.f, 8.0f, 0.0f));

  // 1 10000000 01011011111100001010001 -> 11,401,297/2^22 vs
  // 1 10000000 01011011111100001010101 -> 11,401,301/2^22
  // diff = 4/2^22 = 0.000000953674316
  EXPECT_TRUE(IsClose(-2.718281f, -2.718282f, 0.1f, 0.0f));

  TestEdgeCasesNear<T>();
}

TEST(TensorTestUtilTest, ExpectTensorNearDouble) {
  // double has 1 sign bit, 11 exponent bits, and 52 mantissa bits.
  // The exponent offset is 1,023.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
  typedef double T;
  // Trivial cases: equalities.
  EXPECT_TRUE(IsClose(1.0, 1.0, 0.0, 0.0));
  EXPECT_TRUE(IsClose(0.0, -0.0, 0.0, 0.0));
  EXPECT_TRUE(IsClose(3.14159265359, 3.14159265359, 0.0, 0.0));

  // 0 10000000010 0001111110011001100110011001100110011001100110011010
  //   -> 5,059,512,706,374,042/2^49 vs
  // 0 10000000010 0001111110011010011010110101000010110000111100101000
  //   -> 5,059,569,001,369,384/2^49
  // diff = 56,294,995,342/2^49 = 9.999999999976694198267E-5
  EXPECT_TRUE(IsClose(8.9875, 8.9876, 0.0001, 0.0));

  // 0 10000001111 1000100101110000001100111010100100101010001100000101
  //   -> 6,921,439,564,440,325/2^36
  // 0 10000001111 1000100101110000001100111010111110110111111010010001
  //   -> 6,921,439,571,312,273/2^36
  // diff = 6,871,948/2^36 = 1.000000047497451305389E-4
  EXPECT_FALSE(IsClose(100720.2018, 100720.2019, 0.0001, 0.0));
  EXPECT_TRUE(IsClose(100720.2018, 100720.2019, 1.00000005e-4, 0.0));

  // 0 10000110100 0101111011100010101000101110101101011010010111000100
  //   -> 6,172,839,450,617,284 * 2
  // 0 10000110100 0101111011100010101000101110101101011010010111000011
  //   -> 6,172,839,450,617,283 * 2
  // diff = 1 * 2 = 2
  EXPECT_FALSE(IsClose(12345678901234567., 12345678901234566., 1.0, 0.0));
  EXPECT_TRUE(IsClose(12345678901234567., 12345678901234566., 2.0, 0.0));

  // 1 10000000000 0101101111110000101010001011000101000101111111001111
  //   -> -6,121,026,514,870,223/2^51
  // 1 10000000000 0101101111110000101010001011000101001011011111000101
  //   -> -6,121,026,514,892,741/2^51
  // diff = 22,518/2^51 = 1.00000008274037099909E-11
  EXPECT_FALSE(IsClose(-2.71828182846, -2.71828182847, 1.0e-11, 0.0));
  EXPECT_TRUE(IsClose(-2.71828182846, -2.71828182847, 1.00000009e-11, 0.0));

  TestEdgeCasesNear<T>();
}

// Tensor::Slice() and Tensor::SubSlice() may return unaligned Tensor.
TEST(TensorTestUtilTest, ExpectTensorNearSlice) {
  Tensor x(DT_FLOAT, TensorShape({7, 3}));
  test::FillFn<float>(&x, [](int i) { return 1.0f; });

  test::ExpectTensorNear<float>(
      x.SubSlice(3), test::AsTensor<float>({1.0, 1.0, 1.0}, TensorShape({3})),
      1e-10);
}

template <typename T>
void TestEdgeCasesClose() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_testutil_testDTcc mht_1(mht_1_v, 374, "", "./tensorflow/core/framework/tensor_testutil_test.cc", "TestEdgeCasesClose");

  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::infinity(),
                      Eigen::NumTraits<T>::infinity(), 0.0, 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::lowest(),
                      Eigen::NumTraits<T>::highest(),
                      Eigen::NumTraits<double>::infinity(),
                      Eigen::NumTraits<double>::infinity()));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::lowest(),
                      Eigen::NumTraits<T>::highest(),
                      static_cast<double>(Eigen::NumTraits<T>::highest()),
                      static_cast<double>(Eigen::NumTraits<T>::highest())));
  EXPECT_FALSE(IsClose(Eigen::NumTraits<T>::quiet_NaN(), T(0.0), 0.0, 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<T>::quiet_NaN(), 0.0, 0.0));
  EXPECT_FALSE(IsClose(Eigen::NumTraits<T>::quiet_NaN(), T(0.0),
                       Eigen::NumTraits<double>::infinity(), 0.0));
  EXPECT_TRUE(IsClose(Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<T>::quiet_NaN(),
                      Eigen::NumTraits<double>::infinity(), 0.0));
}

TEST(TensorTestUtilTest, ExpectTensorCloseHalf) {
  typedef Eigen::half T;

  EXPECT_TRUE(IsClose(static_cast<T>(1.0f), static_cast<T>(1.1f), 0.1, 0.1));
  EXPECT_TRUE(IsClose(static_cast<T>(1.0f), static_cast<T>(1.0f), 0.0, 0.0));
  EXPECT_FALSE(IsClose(static_cast<T>(1.0f), static_cast<T>(1.1f), 0.0, 0.0));

  // Epsilon:            0 00010 0000000000 -> 2^-13  = 0.0001220703125
  // Default Tolerance:  0 00100 0100000000 -> 5/2^13 = 0.0006103515625

  // 1.234 -> 0 01111 0011110000 -> 1264/2^10 = 1.234375
  // 1.233 -> 0 01111 0011101111 -> 1263/2^10 = 1.2333984375
  // 1.235 -> 0 01111 0011110001 -> 1265/2^10 = 1.2353515625
  // 1.232 -> 0 01111 0011101110 -> 1262/2^10 = 1.232421875
  // 1.236 -> 0 01111 0011110010 -> 1266/2^10 = 1.236328125
  // 1/2^10 = 0.0009765625E
  // Threshold = 0.0013637542724609375
  EXPECT_TRUE(IsClose(static_cast<T>(1.234f), static_cast<T>(1.234f)));
  EXPECT_TRUE(IsClose(static_cast<T>(1.234f), static_cast<T>(1.233f)));
  EXPECT_TRUE(IsClose(static_cast<T>(1.234f), static_cast<T>(1.235f)));

  // Diff = 0.001953125
  EXPECT_FALSE(IsClose(static_cast<T>(1.234f), static_cast<T>(1.232f)));
  EXPECT_FALSE(IsClose(static_cast<T>(1.234f), static_cast<T>(1.236f)));
  EXPECT_TRUE(
      IsClose(static_cast<T>(1.234f), static_cast<T>(1.232f), 8e-4f, 1e-3f));
  EXPECT_TRUE(
      IsClose(static_cast<T>(1.234f), static_cast<T>(1.236f), 1.4e-3f, 5e-4f));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(
      IsClose(static_cast<T>(3.141592f), static_cast<T>(3.141593f), 0.0, 0.0));

  // Trivial case.
  EXPECT_FALSE(IsClose(static_cast<T>(1e4f), static_cast<T>(1e-4f)));

  TestEdgeCasesClose<T>();
}

TEST(TensorTestUtilTest, ExpectTensorCloseFloat) {
  typedef float T;

  EXPECT_TRUE(IsClose(1.0f, 1.1f, 0.1f, 0.1f));
  EXPECT_TRUE(IsClose(1.0f, 1.0f, 0.0f, 0.0f));
  EXPECT_FALSE(IsClose(1.0f, 1.1f, 0.0f, 0.0f));

  // Epsilon:            2^-23  ~ 0.00000011920928955078
  // Default Tolerance:  5/2^23 ~ 0.00000059604644775391

  // 1.234567f -> 10,356,299/2^23 ~ 1.234567046165466308594
  // 1.234568f -> 10,356,307/2^23 ~ 1.234567999839782714844
  // 1.234566f -> 10,356,290/2^23 ~ 1.234565973281860351563
  // 1.234569f -> 10,356,315/2^23 ~ 1.234568953514099121094
  // 1.234565f -> 10,356,282/2^23 ~ 1.234565019607543945313
  // Threshold ~ 0.00000133190576434572
  EXPECT_TRUE(IsClose(1.234567f, 1.234567f));
  EXPECT_TRUE(IsClose(1.234567f, 1.234568f));
  EXPECT_TRUE(IsClose(1.234567f, 1.234566f));
  EXPECT_FALSE(IsClose(1.234567f, 1.234569f));
  EXPECT_FALSE(IsClose(1.234567f, 1.234565f));
  EXPECT_TRUE(IsClose(1.234567f, 1.234569f, 8e-7f, 1e-6f));
  EXPECT_TRUE(IsClose(1.234567f, 1.234565f, 3e-7f, 1.5e-6f));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(IsClose(3.14159265f, 3.14159266f, 0.0f, 0.0f));

  // Trivial cases
  EXPECT_FALSE(IsClose(1e8f, 1e-8f));
  EXPECT_FALSE(IsClose(1e15f, 1e-15f));

  TestEdgeCasesClose<T>();
}

TEST(TensorTestUtilTest, ExpectTensorCloseDouble) {
  typedef double T;

  EXPECT_TRUE(IsClose(1.0, 1.1, 0.1, 0.1));
  EXPECT_TRUE(IsClose(1.0, 1.0, 0.0, 0.0));
  EXPECT_FALSE(IsClose(1.0, 1.1, 0.0, 0.0));

  // Epsilon:            2^-52  ~ 2.220446049250313080847E-16
  // Default Tolerance:  5/2^52 ~ 1.110223024625156540424E-15

  // 1.234567890123456 -> 5,559,999,489,923,576/2^52 ~ 1.234567890123456024298
  // 1.234567890123457 -> 5,559,999,489,923,580/2^52 ~ 1.234567890123456912477
  // 1.234567890123455 -> 5,559,999,489,923,571/2^52 ~ 1.234567890123454914075
  // 1.234567890123458 -> 5,559,999,489,923,585/2^52 ~ 1.2345678901234580227
  // 1.234567890123454 -> 5,559,999,489,923,567/2^52 ~ 1.234567890123454025897
  // 1.234567890123459 -> 5,559,999,489,923,589/2^52 ~ 1.234567890123458910878
  // 1.234567890123453 -> 5,559,999,489,923,562/2^52 ~ 1.234567890123452915674
  // Threshold ~ 2.480868721703117812159E-15
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123456));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123457));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123455));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123458));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123454));
  EXPECT_FALSE(IsClose(1.234567890123456, 1.234567890123459));
  EXPECT_FALSE(IsClose(1.234567890123456, 1.234567890123453));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123459, 9.5e-16, 1.6e-15));
  EXPECT_TRUE(IsClose(1.234567890123456, 1.234567890123453, 7e-16, 2e-15));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(IsClose(3.141592653589793238, 3.141592653589793239, 0.0, 0.0));

  // Trivial cases
  EXPECT_FALSE(IsClose(1e15, 1e-15));
  EXPECT_FALSE(IsClose(1e30, 1e-30));

  TestEdgeCasesClose<T>();
}

}  // namespace
}  // namespace test
}  // namespace tensorflow
