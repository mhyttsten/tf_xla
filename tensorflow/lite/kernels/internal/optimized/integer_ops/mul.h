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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh() {
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

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const int8* input1_data, const int8* input2_data,
                           int8* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mul.h", "MulElementwise");

  ruy::profiler::ScopeLabel label("MulElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const int16x8_t input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const int16x8_t input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const int16x8_t output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 16; i += 16) {
    // We load / store 16 at a time, multiplying as four sets of 4 int32s.
    const int8x16_t input1_val_original = vld1q_s8(input1_data + i);
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);

    const int16x8_t input1_val_s16_high =
        vmovl_s8(vget_high_s8(input1_val_original));
    const int16x8_t input1_val_s16_low =
        vmovl_s8(vget_low_s8(input1_val_original));

    const int16x8_t input2_val_s16_high =
        vmovl_s8(vget_high_s8(input2_val_original));
    const int16x8_t input2_val_s16_low =
        vmovl_s8(vget_low_s8(input2_val_original));
    const int16x8_t input1_val_high =
        vaddq_s16(input1_val_s16_high, input1_offset_vector);
    const int16x8_t input2_val_high =
        vaddq_s16(input2_val_s16_high, input2_offset_vector);
    const int16x8_t input1_val_low =
        vaddq_s16(input1_val_s16_low, input1_offset_vector);
    const int16x8_t input2_val_low =
        vaddq_s16(input2_val_s16_low, input2_offset_vector);
    const int16x4_t input1_val_high_high = vget_high_s16(input1_val_high);
    const int16x4_t input1_val_high_low = vget_low_s16(input1_val_high);
    const int16x4_t input1_val_low_high = vget_high_s16(input1_val_low);
    const int16x4_t input1_val_low_low = vget_low_s16(input1_val_low);
    const int16x4_t input2_val_high_high = vget_high_s16(input2_val_high);
    const int16x4_t input2_val_high_low = vget_low_s16(input2_val_high);
    const int16x4_t input2_val_low_high = vget_high_s16(input2_val_low);
    const int16x4_t input2_val_low_low = vget_low_s16(input2_val_low);

    auto p1 = vmull_s16(input2_val_high_high, input1_val_high_high);
    auto p2 = vmull_s16(input2_val_high_low, input1_val_high_low);
    auto p3 = vmull_s16(input2_val_low_high, input1_val_low_high);
    auto p4 = vmull_s16(input2_val_low_low, input1_val_low_low);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p3 = vshlq_s32(p3, left_shift_vec);
    p4 = vshlq_s32(p4, left_shift_vec);

    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    p3 = vqrdmulhq_n_s32(p3, params.output_multiplier);
    p4 = vqrdmulhq_n_s32(p4, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);
    p3 = RoundingDivideByPOT(p3, right_shift);
    p4 = RoundingDivideByPOT(p4, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p3_narrowed = vqmovn_s32(p3);
    const auto p4_narrowed = vqmovn_s32(p4);

    const int16x8_t p_part1 =
        vaddq_s16(vcombine_s16(p2_narrowed, p1_narrowed), output_offset_vector);
    const int16x8_t p_part2 =
        vaddq_s16(vcombine_s16(p4_narrowed, p3_narrowed), output_offset_vector);
    const int8x16_t p = vcombine_s8(vqmovn_s16(p_part2), vqmovn_s16(p_part1));

    const auto clamped = vmaxq_s8(output_activation_min_vector,
                                  vminq_s8(output_activation_max_vector, p));
    vst1q_s8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const int8 broadcast_value,
                               const int8* input2_data, int8* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh mht_1(mht_1_v, 315, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mul.h", "MulSimpleBroadcast");

  ruy::profiler::ScopeLabel label("BroadMulSimpleBroadcastInt8/8bit");
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 16; i += 16) {
    // We load / store 16 at a time, multiplying as four sets of 4 int32s.
    const auto input2_val_original = vld1q_s8(input2_data + i);
    const auto input2_val_s16_high =
        vmovl_s8(vget_high_s8(input2_val_original));
    const auto input2_val_s16_low = vmovl_s8(vget_low_s8(input2_val_original));

    const auto input2_val_high =
        vaddq_s16(input2_val_s16_high, input2_offset_vector);
    const auto input2_val_low =
        vaddq_s16(input2_val_s16_low, input2_offset_vector);

    const auto input2_val_low_low = vget_low_s16(input2_val_low);
    const auto input2_val_low_high = vget_high_s16(input2_val_low);
    const auto input2_val_high_low = vget_low_s16(input2_val_high);
    const auto input2_val_high_high = vget_high_s16(input2_val_high);

    auto p1 = vmull_n_s16(input2_val_high_high, input1_val);
    auto p2 = vmull_n_s16(input2_val_high_low, input1_val);
    auto p3 = vmull_n_s16(input2_val_low_high, input1_val);
    auto p4 = vmull_n_s16(input2_val_low_low, input1_val);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p3 = vshlq_s32(p3, left_shift_vec);
    p4 = vshlq_s32(p4, left_shift_vec);

    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    p3 = vqrdmulhq_n_s32(p3, params.output_multiplier);
    p4 = vqrdmulhq_n_s32(p4, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);
    p3 = RoundingDivideByPOT(p3, right_shift);
    p4 = RoundingDivideByPOT(p4, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p3_narrowed = vqmovn_s32(p3);
    const auto p4_narrowed = vqmovn_s32(p4);

    const int16x8_t p_part1 =
        vaddq_s16(vcombine_s16(p2_narrowed, p1_narrowed), output_offset_vector);
    const int16x8_t p_part2 =
        vaddq_s16(vcombine_s16(p4_narrowed, p3_narrowed), output_offset_vector);
    const int8x16_t p = vcombine_s8(vqmovn_s16(p_part2), vqmovn_s16(p_part1));

    const auto clamped = vmaxq_s8(output_activation_min_vector,
                                  vminq_s8(output_activation_max_vector, p));
    vst1q_s8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8* input1_data,
                const RuntimeShape& input2_shape, const int8* input2_data,
                const RuntimeShape& output_shape, int8* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh mht_2(mht_2_v, 410, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mul.h", "Mul");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("MulInt8/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastMulDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int8* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int8* input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmulDTh mht_3(mht_3_v, 429, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mul.h", "BroadcastMulDispatch");

  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_integer_ops::BroadcastMul4DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  optimized_ops::BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data, MulElementwise, MulSimpleBroadcast);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
