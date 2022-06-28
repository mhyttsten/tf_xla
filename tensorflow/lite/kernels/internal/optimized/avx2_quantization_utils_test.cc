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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utils_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utils_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"

#include <gmock/gmock.h>
#include "tensorflow/lite/kernels/internal/common.h"

#ifdef __AVX2__
namespace tflite {
namespace avx2_utils {
namespace {

using ::testing::ElementsAreArray;

__m256i FillVectorWithInt32(const std::vector<int32_t>& src) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utils_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils_test.cc", "FillVectorWithInt32");

  return _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2],
                          src[1], src[0]);
}

void CompareWithReferenceValue(std::vector<int32_t>& reference_values,
                               const __m256i& result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utils_testDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils_test.cc", "CompareWithReferenceValue");

  // As _mm256_extract_epi32 only supports const int, which should be known
  // at the comile time, it puts down 8 comparison instead of for-loop.
  EXPECT_NEAR(reference_values[0], _mm256_extract_epi32(result, 0), 1);
  EXPECT_NEAR(reference_values[1], _mm256_extract_epi32(result, 1), 1);
  EXPECT_NEAR(reference_values[2], _mm256_extract_epi32(result, 2), 1);
  EXPECT_NEAR(reference_values[3], _mm256_extract_epi32(result, 3), 1);
  EXPECT_NEAR(reference_values[4], _mm256_extract_epi32(result, 4), 1);
  EXPECT_NEAR(reference_values[5], _mm256_extract_epi32(result, 5), 1);
  EXPECT_NEAR(reference_values[6], _mm256_extract_epi32(result, 6), 1);
  EXPECT_NEAR(reference_values[7], _mm256_extract_epi32(result, 7), 1);
}

TEST(CastInt32ToInt16AndStoreTest, CastInt32ToInt16AndStoreTest) {
  const std::vector<int16_t> src = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t dst[8];
  const __m256i src_vector = _mm256_set_epi32(src[7], src[6], src[5], src[4],
                                              src[3], src[2], src[1], src[0]);
  CastInt32ToInt16AndStore(dst, src_vector);
  EXPECT_THAT(src, ElementsAreArray(dst));
}

TEST(MultiplyByQuantizedMultiplierTest, PositiveLeftShiftTest) {
  std::vector<int32_t> values = {100, 200, 300, 400, 500, 600, 700, 800};
  const __m256i src_vector = FillVectorWithInt32(values);
  const int32_t left_shift = 20;
  const int32_t multiplier = 12345;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, NegativeLeftShiftTest) {
  std::vector<int32_t> values = {1000, 2000, 3000, 4000,
                                 5000, 6000, 7000, 8000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const int32_t left_shift = -3;
  const int32_t multiplier = 1234567890;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, VectorPositiveLeftShiftTest) {
  std::vector<int32_t> values = {100, 200, 300, 400, 500, 600, 700, 800};
  const std::vector<int32_t> left_shifts = {20, 19, 18, 17, 16, 15, 14, 13};
  const std::vector<int32_t> multipliers = {10000, 20000, 30000, 40000,
                                            50000, 60000, 70000, 80000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const __m256i left_shifts_vector = FillVectorWithInt32(left_shifts);
  const __m256i multipliers_vector = FillVectorWithInt32(multipliers);

  const __m256i result = MultiplyByQuantizedMultiplier(
      src_vector, multipliers_vector, left_shifts_vector);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multipliers[i],
                                                      left_shifts[i]);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, VectorNegativeLeftShiftTest) {
  std::vector<int32_t> values = {1000, 2000, 3000, 4000,
                                 5000, 6000, 7000, 8000};
  const std::vector<int32_t> left_shifts = {-3, -4, -5, -6, -7, -8, -9, -10};
  const std::vector<int32_t> multipliers = {1000000000, 1100000000, 1200000000,
                                            1300000000, 1400000000, 1500000000,
                                            1600000000, 1700000000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const __m256i left_shifts_vector = FillVectorWithInt32(left_shifts);
  const __m256i multipliers_vector = FillVectorWithInt32(multipliers);

  const __m256i result = MultiplyByQuantizedMultiplier(
      src_vector, multipliers_vector, left_shifts_vector);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multipliers[i],
                                                      left_shifts[i]);
  }

  CompareWithReferenceValue(values, result);
}

}  // namespace
}  // namespace avx2_utils
}  // namespace tflite

#endif  //  __AVX2__
