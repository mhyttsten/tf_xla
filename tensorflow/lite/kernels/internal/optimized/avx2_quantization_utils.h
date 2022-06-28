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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh() {
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

#ifdef __AVX2__

#include <immintrin.h>

#include <limits>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace avx2_utils {

static inline void mm_storeu_si64(void *dst, __m128i v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "mm_storeu_si64");

#if (defined __clang__) || (defined _MSC_VER)
  _mm_storeu_si64(dst, v);
#else
  // GCC 9 lacks support for _mm_storeu_si64.
  *static_cast<std::int64_t *>(dst) = _mm_extract_epi64(v, 0);
#endif
}

static inline __m256i mm256_blendv_epi32(const __m256i &a, const __m256i &b,
                                         const __m256i &mask) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_1(mht_1_v, 211, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "mm256_blendv_epi32");

  __m256 result =
      _mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b),
                       _mm256_castsi256_ps(mask));
  return _mm256_castps_si256(result);
}

static inline __m256i rounding_right_shift(const __m256i &value,
                                           int32_t right_shift) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_2(mht_2_v, 222, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "rounding_right_shift");

  TFLITE_DCHECK_GT(right_shift, 0);
  const int32_t one_shift_exp_minus1 = 1 << (right_shift - 1);
  __m256i nudge = _mm256_set1_epi32(one_shift_exp_minus1);
  const __m256i r_plus_nudge = _mm256_add_epi32(value, nudge);
  const __m256i shifted_sum =
      _mm256_srav_epi32(r_plus_nudge, _mm256_set1_epi32(right_shift));

  // Identify overflow in each lane and create mask.
  const __m256i mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(
      value, _mm256_set1_epi32(0x7fffffff - one_shift_exp_minus1));
  // Fill results with either (value + nudge) >> exponent or
  // std::numeric_limits<std::int32_t>::max() in the case of overflow.
  return mm256_blendv_epi32(
      shifted_sum, _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max()),
      mask_num_plus_nudge_overflow);
}

static inline __m256i rounding_right_shift(const __m256i &value,
                                           const __m256i right_shift) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_3(mht_3_v, 244, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "rounding_right_shift");

  const __m256i zeros = _mm256_setzero_si256();
  const __m256i mask_rightshift_gtz = _mm256_cmpgt_epi32(right_shift, zeros);
  const __m256i one_shift_exp_minus1 =
      _mm256_sllv_epi32(_mm256_set1_epi32(1),
                        _mm256_sub_epi32(right_shift, _mm256_set1_epi32(1)));
  __m256i nudge =
      mm256_blendv_epi32(zeros, one_shift_exp_minus1, mask_rightshift_gtz);
  const __m256i r_plus_nudge = _mm256_add_epi32(value, nudge);
  const __m256i shifted_sum = _mm256_srav_epi32(r_plus_nudge, right_shift);

  // Identify overflow in each lane and create mask.
  const __m256i mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(
      value, _mm256_sub_epi32(_mm256_set1_epi32(0x7fffffff), nudge));
  // Fill results with either (value + nudge) >> exponent or
  // std::numeric_limits<std::int32_t>::max() in the case of overflow.
  return mm256_blendv_epi32(
      shifted_sum, _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max()),
      mask_num_plus_nudge_overflow);
}

inline void CastInt32ToInt16AndStore(int16 *dst, const __m256i v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_4(mht_4_v, 268, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "CastInt32ToInt16AndStore");

  // As _mm256_cvtepi32_epi16 is not supported in AVX2, use the below repack.
  // Select bytes 0, 1, 4, 5, 8, 9, 12, 13 within each lane, effectively
  // truncating each 16-bit integer.
  const __m256i repack_perm = _mm256_set1_epi64x(0x0d0c090805040100);
  const __m256i shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  mm_storeu_si64(dst, _mm256_extracti128_si256(shuffled_v, 0));
  mm_storeu_si64(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
}

inline __m256i MultiplyByQuantizedMultiplier(const __m256i &value,
                                             const int32_t multiplier,
                                             const int32_t left_shift) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_5(mht_5_v, 283, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "MultiplyByQuantizedMultiplier");

  const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i shifted_value =
      left_shift > 0 ? _mm256_sllv_epi32(value, _mm256_set1_epi32(left_shift))
                     : value;

  __m256i scaled_v_low = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 0)),
      _mm256_set1_epi64x(multiplier));
  __m256i scaled_v_high = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 1)),
      _mm256_set1_epi64x(multiplier));

  scaled_v_low = _mm256_srlv_epi64(scaled_v_low, _mm256_set1_epi64x(31));
  scaled_v_high = _mm256_srlv_epi64(scaled_v_high, _mm256_set1_epi64x(31));
  // As _mm256_cvtepi64_epi32 is not supported in AVX2, use the below permute.
  scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
  __m256i result = _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
  result = _mm256_permutevar8x32_epi32(result, repack_perm);
  if (left_shift >= 0) {
    return result;
  }
  return rounding_right_shift(result, -left_shift);
}

inline __m256i MultiplyByQuantizedMultiplier(const __m256i &value,
                                             const __m256i multiplier,
                                             const __m256i left_shift) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSavx2_quantization_utilsDTh mht_6(mht_6_v, 313, "", "./tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h", "MultiplyByQuantizedMultiplier");

  const __m256i zero_vector = _mm256_setzero_si256();
  const __m256i positive_left_shift = _mm256_max_epi32(left_shift, zero_vector);
  const __m256i positive_right_shift =
      _mm256_max_epi32(_mm256_sub_epi32(zero_vector, left_shift), zero_vector);

  const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i shifted_value = _mm256_sllv_epi32(value, positive_left_shift);

  const __m256i multiplier_low =
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(multiplier, 0));
  const __m256i multiplier_high =
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(multiplier, 1));

  __m256i scaled_v_low = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 0)),
      multiplier_low);
  __m256i scaled_v_high = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 1)),
      multiplier_high);

  scaled_v_low = _mm256_srlv_epi64(scaled_v_low, _mm256_set1_epi64x(31));
  scaled_v_high = _mm256_srlv_epi64(scaled_v_high, _mm256_set1_epi64x(31));
  // As _mm256_cvtepi64_epi32 is not supported in AVX2, use the below permute.
  scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
  __m256i result = _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
  result = _mm256_permutevar8x32_epi32(result, repack_perm);

  return rounding_right_shift(result, positive_right_shift);
}
}  // namespace avx2_utils
}  // namespace tflite

#endif  // __AVX2__
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
