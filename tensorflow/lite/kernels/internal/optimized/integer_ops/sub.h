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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh() {
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


#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void SubElementwiseInt16(int size, const ArithmeticParams& params,
                                const int16* input1_data,
                                const int16* input2_data, int16* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h", "SubElementwiseInt16");

  ruy::profiler::ScopeLabel label("SubElementwiseInt16/16bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

#ifdef __AVX2__
  const int32_t input1_left_shift = params.left_shift + params.input1_shift;
  const int32_t input2_left_shift = params.left_shift + params.input2_shift;
  const __m256i input1_offset = _mm256_set1_epi32(params.input1_offset);
  const __m256i input2_offset = _mm256_set1_epi32(params.input2_offset);
  const __m256i output_offset = _mm256_set1_epi32(params.output_offset);
  const __m256i clamp_max_v =
      _mm256_set1_epi32(params.quantized_activation_max);
  const __m256i clamp_min_v =
      _mm256_set1_epi32(params.quantized_activation_min);

  for (; i <= size - 16; i += 16) {
    const __m256i input1_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input1_data + i));
    const __m256i input2_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input2_data + i));

    __m256i s11 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input1_val_original));
    __m256i s12 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input1_val_original, 1));
    __m256i s21 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input2_val_original));
    __m256i s22 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input2_val_original, 1));

    s11 = _mm256_add_epi32(s11, input1_offset);
    s12 = _mm256_add_epi32(s12, input1_offset);
    s21 = _mm256_add_epi32(s21, input2_offset);
    s22 = _mm256_add_epi32(s22, input2_offset);

    s11 = avx2_utils::MultiplyByQuantizedMultiplier(
        s11, params.input1_multiplier, input1_left_shift);
    s12 = avx2_utils::MultiplyByQuantizedMultiplier(
        s12, params.input1_multiplier, input1_left_shift);
    s21 = avx2_utils::MultiplyByQuantizedMultiplier(
        s21, params.input2_multiplier, input2_left_shift);
    s22 = avx2_utils::MultiplyByQuantizedMultiplier(
        s22, params.input2_multiplier, input2_left_shift);

    __m256i s1 = _mm256_sub_epi32(s11, s21);
    __m256i s2 = _mm256_sub_epi32(s12, s22);

    s1 = avx2_utils::MultiplyByQuantizedMultiplier(s1, params.output_multiplier,
                                                   params.output_shift);
    s2 = avx2_utils::MultiplyByQuantizedMultiplier(s2, params.output_multiplier,
                                                   params.output_shift);

    s1 = _mm256_add_epi32(s1, output_offset);
    s2 = _mm256_add_epi32(s2, output_offset);

    s1 = _mm256_min_epi32(s1, clamp_max_v);
    s1 = _mm256_max_epi32(s1, clamp_min_v);
    s2 = _mm256_min_epi32(s2, clamp_max_v);
    s2 = _mm256_max_epi32(s2, clamp_min_v);

    avx2_utils::CastInt32ToInt16AndStore(output_data + i, s1);
    avx2_utils::CastInt32ToInt16AndStore(output_data + i + 8, s2);
  }
#endif  // __AVX2__

  for (; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int16>(clamped_output);
  }
}

inline void BroadcastSubFiveFold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& input1_shape,
                                 const int16* unswitched_input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 int16* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh mht_1(mht_1_v, 302, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h", "BroadcastSubFiveFold");

  ruy::profiler::ScopeLabel label("BroadcastSubFiveFold/16bit");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const int16_t* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const int16_t* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  int16_t* output_data_ptr = output_data;
  const int16_t* input1_data_ptr = input1_data;
  const int16_t* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // The flatsize for each inputs are as below.
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  const int y0 = params.broadcast_shape[0];
  const int y1 = params.broadcast_shape[1];
  const int y2 = params.broadcast_shape[2];
  const int y3 = params.broadcast_shape[3];
  const int y4 = params.broadcast_shape[4];
  for (int i0 = 0; i0 < y0; ++i0) {
    const int16_t* input2_data_ptr = nullptr;
    for (int i1 = 0; i1 < y1; ++i1) {
      input2_data_ptr = input2_data_reset;
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i3 = 0; i3 < y3; ++i3) {
          if (use_unswitched) {
            SubElementwiseInt16(y4, params, input1_data_ptr, input2_data_ptr,
                                output_data_ptr);
          } else {
            // When input1 and input2 are switched, calculate (input2 - input1)
            // and use unswitched_params as we switch the switched input here.
            SubElementwiseInt16(y4, unswitched_params, input2_data_ptr,
                                input1_data_ptr, output_data_ptr);
          }
          input2_data_ptr += y4;
          output_data_ptr += y4;
        }
        // We have broadcast y4 of input1 data y3 times, and now move on.
        input1_data_ptr += y4;
      }
    }
    // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
    input2_data_reset = input2_data_ptr;
  }
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh mht_2(mht_2_v, 371, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h", "Sub");

  ruy::profiler::ScopeLabel label("SubInt16/16bit");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  SubElementwiseInt16(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastSubDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int16* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16* input2_data,
                                 const RuntimeShape& output_shape,
                                 int16* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSsubDTh mht_3(mht_3_v, 394, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h", "BroadcastSubDispatch");

  ruy::profiler::ScopeLabel label("BroadcastSubDispatchInt16/16bit");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_ops::BroadcastQuantSubSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  BroadcastSubFiveFold(params, input1_shape, input1_data, input2_shape,
                       input2_data, output_shape, output_data);
}
}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_
