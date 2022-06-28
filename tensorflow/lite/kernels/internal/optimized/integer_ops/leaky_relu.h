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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSleaky_reluDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSleaky_reluDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSleaky_reluDTh() {
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

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const int16* input_data,
                              const RuntimeShape& output_shape,
                              int16* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSleaky_reluDTh mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/leaky_relu.h", "QuantizeLeakyRelu");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const int32_t quantized_min = std::numeric_limits<int16>::min();
  const int32_t quantized_max = std::numeric_limits<int16>::max();
  int i = 0;

#ifdef __AVX2__
  const __m256i input_offset = _mm256_set1_epi32(params.input_offset);
  const __m256i output_offset = _mm256_set1_epi32(params.output_offset);
  const __m256i output_muliplier_identity =
      _mm256_set1_epi32(params.output_multiplier_identity);
  const __m256i output_shift_identity =
      _mm256_set1_epi32(params.output_shift_identity);
  const __m256i output_multiplier_alpha =
      _mm256_set1_epi32(params.output_multiplier_alpha);
  const __m256i output_shift_alpha =
      _mm256_set1_epi32(params.output_shift_alpha);
  const __m256i clamp_max_v = _mm256_set1_epi32(quantized_max);
  const __m256i clamp_min_v = _mm256_set1_epi32(quantized_min);

  for (; i <= flat_size - 16; i += 16) {
    const __m256i input =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input_data + i));
    __m256i input_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input));
    __m256i input_high =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input, 1));
    input_low = _mm256_sub_epi32(input_low, input_offset);
    input_high = _mm256_sub_epi32(input_high, input_offset);

    const __m256i zeros = _mm256_setzero_si256();
    const __m256i input_low_mask = _mm256_cmpgt_epi32(input_low, zeros);
    const __m256i input_high_mask = _mm256_cmpgt_epi32(input_high, zeros);
    const __m256i input_low_output_multiplier = avx2_utils::mm256_blendv_epi32(
        output_multiplier_alpha, output_muliplier_identity, input_low_mask);
    const __m256i input_low_output_shift = avx2_utils::mm256_blendv_epi32(
        output_shift_alpha, output_shift_identity, input_low_mask);
    const __m256i input_high_output_multiplier = avx2_utils::mm256_blendv_epi32(
        output_multiplier_alpha, output_muliplier_identity, input_high_mask);
    const __m256i input_high_output_shift = avx2_utils::mm256_blendv_epi32(
        output_shift_alpha, output_shift_identity, input_high_mask);

    input_low = avx2_utils::MultiplyByQuantizedMultiplier(
        input_low, input_low_output_multiplier, input_low_output_shift);
    input_high = avx2_utils::MultiplyByQuantizedMultiplier(
        input_high, input_high_output_multiplier, input_high_output_shift);

    input_low = _mm256_add_epi32(input_low, output_offset);
    input_high = _mm256_add_epi32(input_high, output_offset);

    input_low = _mm256_min_epi32(input_low, clamp_max_v);
    input_low = _mm256_max_epi32(input_low, clamp_min_v);
    input_high = _mm256_min_epi32(input_high, clamp_max_v);
    input_high = _mm256_max_epi32(input_high, clamp_min_v);

    avx2_utils::CastInt32ToInt16AndStore(output_data + i, input_low);
    avx2_utils::CastInt32ToInt16AndStore(output_data + i + 8, input_high);
  }
#endif  // __AVX2__

  for (; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;
    int32_t unclamped_output;
    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_alpha,
                             params.output_shift_alpha);
    }
    const int16 clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<int16>(clamped_output);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_
