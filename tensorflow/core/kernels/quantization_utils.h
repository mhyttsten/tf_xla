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

#ifndef TENSORFLOW_CORE_KERNELS_QUANTIZATION_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_QUANTIZATION_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh() {
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


#include <cmath>
#define EIGEN_USE_THREADS

// This is a set of functions that standardizes how quantized values are
// interpreted as float numbers.
// All of the current implementations are for reference and have not been
// optimized. They should be implementable using fixed point representations
// to avoid a dependency on floating-point hardware.

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define QUANTIZATION_UTILS_USE_NEON
#include <arm_neon.h>
#endif

#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

// We have to be able to detect and handle overflows in int32, so this function
// uses doubles and int64's to make sure we have enough room.
template <class T>
inline int64_t FloatToQuantizedUnclamped(float input, float range_min,
                                         float range_max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatToQuantizedUnclamped");

  const int64_t lowest_quantized =
      static_cast<double>(Eigen::NumTraits<T>::lowest());
  if (range_min == range_max) {
    return lowest_quantized;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64_t quantized =
      (round(input * range_scale) - round(range_min * range_scale));
  quantized += lowest_quantized;
  return quantized;
}

template <>
inline int64_t FloatToQuantizedUnclamped<float>(float input, float range_min,
                                                float range_max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatToQuantizedUnclamped<float>");

  return -1;
}

// This converts the float into the final quantized type, clamping/saturating
// any over or underflows.
template <class T>
T FloatToQuantized(float input, float range_min, float range_max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatToQuantized");

  if (std::is_same<T, float>::value) {
    // Specialization for float. This is used in reference implementation
    // for float which is useful to compare performance between float
    // and quantized type.
    return input;
  }
  int64_t quantized = FloatToQuantizedUnclamped<T>(input, range_min, range_max);
  const int64_t lowest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<T>::lowest());
  const int64_t highest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<T>::highest());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32>(quantized));
}

template <class T>
float QuantizedToFloat(T input, float range_min, float range_max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_3(mht_3_v, 269, "", "./tensorflow/core/kernels/quantization_utils.h", "QuantizedToFloat");

  if (std::is_same<T, float>::value) {
    // Specialization for float. This is used in reference implementation
    // for float which is useful to compare performance between float
    // and quantized type.
    return input;
  }
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64_t lowest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<T>::lowest());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  // For compatibility with DEQUANTIZE_WITH_EIGEN, we should convert
  // range_scale to a float, otherwise range_min_rounded might be slightly
  // different.
  const double range_min_rounded =
      std::round(range_min / static_cast<float>(range_scale)) *
      static_cast<float>(range_scale);
  const double result = range_min_rounded + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T>
float FloatForOneQuantizedLevel(float range_min, float range_max) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_4(mht_4_v, 301, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatForOneQuantizedLevel");

  const int64_t highest = static_cast<int64_t>(Eigen::NumTraits<T>::highest());
  const int64_t lowest = static_cast<int64_t>(Eigen::NumTraits<T>::lowest());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64_t c_highest =
      static_cast<int64_t>(Eigen::NumTraits<T3>::highest());
  const int64_t c_lowest = static_cast<int64_t>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

// input_array is an eigen Tensor.  q2f is a QuantizedToFloatStruct.
// This evaluates to an eigen tensor expression, to be used like:
// auto tensor = DEQUANTIZE_WITH_EIGEN(input_tensor, q2f);
#define DEQUANTIZE_WITH_EIGEN(input_array, q2f)                         \
  ((q2f.range_min_rounded - q2f.lowest_quantized() * q2f.range_scale) + \
   input_array.template cast<float>() * q2f.range_scale)

// input_array is an eigen Tensor.  f2q is a FloatToQuantizedStruct.
// OutputType is the type of output (e.g. quint8).
// This evaluates to an eigen tensor expression, to be used like:
// auto tensor = QUANTIZE_WITH_EIGEN(input_tensor, f2q, T);
#define QUANTIZE_WITH_EIGEN(input_array, f2q, OutputType) \
  ((input_array * f2q.range_scale).round() -              \
   (f2q.range_min_scaled - f2q.lowest_quantized()))       \
      .cwiseMax(f2q.lower_bound_float())                  \
      .cwiseMin(f2q.upper_bound_float())                  \
      .template cast<int32>()                             \
      .template cast<OutputType>()

// For use with DEQUANTIZE_WITH_EIGEN.
template <typename T>
struct QuantizedToFloatStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64_t number_of_steps = static_cast<int64_t>(1)
                                             << number_of_bits;

  static float lowest_quantized() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_5(mht_5_v, 357, "", "./tensorflow/core/kernels/quantization_utils.h", "lowest_quantized");

    return static_cast<float>(Eigen::NumTraits<T>::lowest());
  }

  QuantizedToFloatStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale((range_max - range_min) / (number_of_steps - 1.0)),
        range_min_rounded(range_max == range_min
                              ? range_min
                              : std::round(range_min / range_scale) *
                                    range_scale) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_6(mht_6_v, 370, "", "./tensorflow/core/kernels/quantization_utils.h", "QuantizedToFloatStruct");
}

  const float range_min;
  const float range_scale;
  const float range_min_rounded;
};

// For use with QUANTIZE_WITH_EIGEN.
template <typename T>
struct FloatToQuantizedStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64_t number_of_steps = static_cast<int64_t>(1)
                                             << number_of_bits;
  static constexpr double range_adjust =
      (number_of_steps / (number_of_steps - 1.0));

  // Casting QInt32's lowest or highest to a float gives a float that can't be
  // cast back to int32 or QInt32.  Instead, use bounds that can be converted
  // back to int32 without going outside the range of an int32.
  static float lower_bound_float() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_7(mht_7_v, 392, "", "./tensorflow/core/kernels/quantization_utils.h", "lower_bound_float");

    return Eigen::numext::maxi(
        static_cast<float>(Eigen::NumTraits<T>::lowest()), -2.147483648e+09f);
  }
  static float upper_bound_float() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_8(mht_8_v, 399, "", "./tensorflow/core/kernels/quantization_utils.h", "upper_bound_float");

    return Eigen::numext::mini(
        static_cast<float>(Eigen::NumTraits<T>::highest()), +2.147483520e+09f);
  }

  static float lowest_quantized() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_9(mht_9_v, 407, "", "./tensorflow/core/kernels/quantization_utils.h", "lowest_quantized");

    return static_cast<float>(Eigen::NumTraits<T>::lowest());
  }

  FloatToQuantizedStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale(range_max == range_min
                        ? 0.0
                        : (number_of_steps - 1.0) / (range_max - range_min)),
        range_min_scaled(std::round(range_min * range_scale)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_10(mht_10_v, 419, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatToQuantizedStruct");
}

  const float range_min;
  const float range_scale;
  const float range_min_scaled;
};

template <class T1, class T2>
inline T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                               float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
inline void RequantizeManyInNewRange(const T1* input, int64_t count,
                                     float min_input, float max_input,
                                     float min_output, float max_output,
                                     T2* output) {
  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(input[index], min_input, max_input);
    output[index] = FloatToQuantized<T2>(input_float, min_output, max_output);
  }
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
inline void RequantizeManyInNewRangeReference(const qint32* input,
                                              int64_t count, float min_input,
                                              float max_input, float min_output,
                                              float max_output,
                                              quint8* output) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_11(mht_11_v, 455, "", "./tensorflow/core/kernels/quantization_utils.h", "RequantizeManyInNewRangeReference");

  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const float input_rezero = (min_input + max_input) / 2.0;
  const int64_t range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int64_t>(255.0 * (1 << fp_shift) *
                                                 input_range / output_range);
  const int64_t input_offset_fp =
      static_cast<int64_t>(input_rezero * recip_output_range * (1 << fp_shift));
  const int64_t output_offset_fp =
      output_range == 0.0
          ? 0
          : std::lround((1 << fp_shift) * (min_output * 255.0) / output_range);
  const int64_t rounding_delta = 1 << (fp_shift - 1);

  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (int64_t index = 0; index < count; ++index) {
    const int64_t input_value = static_cast<int64_t>(input[index]);
    const int64_t fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const int64_t offset_intermediate = fp_value - output_offset_fp;
    const int64_t round_intermediate = offset_intermediate + rounding_delta;
    int64_t quantized_int64 = round_intermediate >> fp_shift;
    quantized_int64 = std::max(quantized_int64, int64_t{0});
    quantized_int64 = std::min(quantized_int64, int64_t{255});
    output[index] = static_cast<quint8>(static_cast<int32>(quantized_int64));
  }
}

// Another common case is converting eight bit inputs up to thirty two bits, so
// we have specialized fixed-point code to accelerate that. There is also a NEON
// version for ARM devices below.
inline void RequantizeManyInNewRange8To32BitReference(
    const quint8* input, int64_t count, float min_input, float max_input,
    float min_output, float max_output, qint32* output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_12(mht_12_v, 501, "", "./tensorflow/core/kernels/quantization_utils.h", "RequantizeManyInNewRange8To32BitReference");

  const float code_0_float = QuantizedToFloat<quint8>(0, min_input, max_input);
  const float code_1_float = QuantizedToFloat<quint8>(1, min_input, max_input);
  const int64_t code_0_int64 =
      FloatToQuantizedUnclamped<qint32>(code_0_float, min_output, max_output);
  const int64_t code_1_int64 =
      FloatToQuantizedUnclamped<qint32>(code_1_float, min_output, max_output);
  const int32_t mult_int32 = code_1_int64 - code_0_int64;
  const int64_t lowest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<qint32>::lowest());
  const int64_t highest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<qint32>::highest());
  for (int64_t i = 0; i < count; ++i) {
    const int64_t input_value = static_cast<int64_t>(input[i]);
    int64_t output_value = code_0_int64 + (input_value * mult_int32);
    output_value = std::max(output_value, lowest_quantized);
    output_value = std::min(output_value, highest_quantized);
    output[i] = static_cast<int32>(output_value);
  }
}

#ifdef QUANTIZATION_UTILS_USE_NEON
// Speeds up the 32->8bit conversion using fixed-point arithmetic and NEON SIMD
// intrinsics for ARM platforms.
inline void RequantizeManyInNewRangeNeon(const qint32* input, int64 count,
                                         float min_input, float max_input,
                                         float min_output, float max_output,
                                         quint8* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
  const int fp_shift = 16;

  // Calculate range variables in advance.
  // Input range.
  const float input_range = max_input - min_input;
  // Output range.
  const float output_range = max_output - min_output;
  // Ratio of output range.
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  // Average of input range as zero position of input.
  const float input_rezero = (min_input + max_input) / 2.0;
  // In-out range scale.
  const int32 range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int32>(255.0 * (1 << (fp_shift - 16)) *
                                               input_range / output_range);
  // Input zero position offset to output.
  const int32 input_offset_fp =
      static_cast<int32>(input_rezero * recip_output_range * (1 << fp_shift));
  // Output min offset.
  const int32 output_offset_fp =
      output_range == 0.0
          ? 0
          : static_cast<int32>((1 << fp_shift) * (min_output * 255.0) /
                               output_range);
  const int32 rounding_delta = 1 << (fp_shift - 1);

  // broadcast range to each lane
  const int32x4_t range_scale_fp_32x4 = vmovq_n_s32(range_scale_fp);
  const int32x4_t input_offset_fp_32x4 = vmovq_n_s32(input_offset_fp);
  const int32x4_t output_offset_fp_32x4 = vmovq_n_s32(output_offset_fp);
  const int32x4_t rounding_delta_32x4 = vmovq_n_s32(rounding_delta);

  int64 index = 0;
  // Use SIMD to requantize.
  for (; index < (count - 7); index += 8) {
    const int32* input_ptr = &(input->value) + index;
    const int32x4_t input_value_low_32x4 = vld1q_s32(input_ptr);
    const int32x4_t input_value_high_32x4 = vld1q_s32(input_ptr + 4);
    const int32x4_t fp_value_low_32x4 = vaddq_s32(
        input_offset_fp_32x4,
        vmulq_s32(vshrq_n_s32(input_value_low_32x4, 16), range_scale_fp_32x4));
    const int32x4_t fp_value_high_32x4 = vaddq_s32(
        input_offset_fp_32x4,
        vmulq_s32(vshrq_n_s32(input_value_high_32x4, 16), range_scale_fp_32x4));
    const int32x4_t offset_intermediate_low_32x4 =
        vsubq_s32(fp_value_low_32x4, output_offset_fp_32x4);
    const int32x4_t offset_intermediate_high_32x4 =
        vsubq_s32(fp_value_high_32x4, output_offset_fp_32x4);
    const int32x4_t round_intermediate_low_32x4 =
        vaddq_s32(offset_intermediate_low_32x4, rounding_delta_32x4);
    const int32x4_t round_intermediate_high_32x4 =
        vaddq_s32(offset_intermediate_high_32x4, rounding_delta_32x4);
    const int16x4_t quantized_low_16x4 =
        vqmovn_s32(vshrq_n_s32(round_intermediate_low_32x4, fp_shift));
    const int16x4_t quantized_high_16x4 =
        vqmovn_s32(vshrq_n_s32(round_intermediate_high_32x4, fp_shift));
    const uint8x8_t quantized_8x8 =
        vqmovun_s16(vcombine_s16(quantized_low_16x4, quantized_high_16x4));
    uint8* output_ptr = &(output->value) + index;
    vst1_u8(output_ptr, quantized_8x8);
  }

  // Requantize remaining elements in array without SIMD.
  for (; index < count; ++index) {
    const int32 input_value = static_cast<int32>(input[index]);
    const int32 fp_value =
        static_cast<int32>(
            (static_cast<int32>(input_value >> 16) * (range_scale_fp))) +
        input_offset_fp;
    const int32 offset_intermediate = fp_value - output_offset_fp;
    const int32 round_intermediate = offset_intermediate + rounding_delta;
    int32 quantized_int32 = round_intermediate >> fp_shift;
    quantized_int32 = std::max(quantized_int32, 0);
    quantized_int32 = std::min(quantized_int32, 255);
    output[index] = static_cast<quint8>(static_cast<int32>(quantized_int32));
  }
}

template <>
inline void RequantizeManyInNewRange<qint32, quint8>(
    const qint32* input, int64 count, float min_input, float max_input,
    float min_output, float max_output, quint8* output) {
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  if ((input_range / output_range) > 16384.0f) {
    // Our NEON implementation uses 32-bit math and can't handle very
    // large ranges, so fall back to the reference implementation. We don't
    // expect these to be common in models, so this shouldn't be a performance
    // problem in practice.
    RequantizeManyInNewRangeReference(input, count, min_input, max_input,
                                      min_output, max_output, output);
  } else {
    RequantizeManyInNewRangeNeon(input, count, min_input, max_input, min_output,
                                 max_output, output);
  }
}

// NEON accelerated 16bit rounded division by 2^n.
template <int POW>
inline int16x8_t Divide16x8PowRound(const int16x8_t val) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_13(mht_13_v, 635, "", "./tensorflow/core/kernels/quantization_utils.h", "Divide16x8PowRound");

  const int16x8_t val_sign = vshrq_n_s16(val, 15);
  const int16x8_t val_xor = veorq_s16(val, val_sign);
  const int16x8_t val_pos = vsubq_s16(val_xor, val_sign);
  const int16x8_t shifted_val_pos = vrshrq_n_s16(val_pos, POW);
  const int16x8_t shifted_val_pos_xor = veorq_s16(shifted_val_pos, val_sign);
  const int16x8_t shifted_val = vsubq_s16(shifted_val_pos_xor, val_sign);
  return shifted_val;
}

// NEON accelerated 64bit rounded division by 2^n.
template <int POW>
inline int64x2_t Divide64x2PowRound(const int64x2_t val) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_14(mht_14_v, 650, "", "./tensorflow/core/kernels/quantization_utils.h", "Divide64x2PowRound");

  const int64x2_t val_sign = vshrq_n_s64(val, 63);
  const int64x2_t val_xor = veorq_s64(val, val_sign);
  const int64x2_t val_pos = vsubq_s64(val_xor, val_sign);
  const int64x2_t shifted_val_pos = vrshrq_n_s64(val_pos, POW);
  const int64x2_t shifted_val_pos_xor = veorq_s64(shifted_val_pos, val_sign);
  const int64x2_t shifted_val = vsubq_s64(shifted_val_pos_xor, val_sign);
  return shifted_val;
}

// NEON accelerated 16bit division by 2^n.
// CAVEAT: The input must be greater than min-int16 to avoid underflow.
template <int POW>
inline int16x8_t Divide16x8Pow(const int16x8_t val) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_15(mht_15_v, 666, "", "./tensorflow/core/kernels/quantization_utils.h", "Divide16x8Pow");

  static constexpr int16 FIRST_BIT_VAL = 0x0000000000000001;
  static const int16x8_t FIRST_BIT = vmovq_n_s16(FIRST_BIT_VAL);
  const int16x8_t val_sign = vshrq_n_s16(val, 15);
  const int16x8_t neg_offset = vandq_s16(val_sign, FIRST_BIT);
  const int16x8_t val_with_offset = vsubq_s16(val, neg_offset);
  const int16x8_t shifted_wo_offset =
      vsraq_n_s16(neg_offset, val_with_offset, POW);
  return shifted_wo_offset;
}

// NEON accelerated 64bit division by 2^n.
// CAVEAT: The input must be greater than min-int64 to avoid underflow.
template <int POW>
inline int64x2_t Divide64x2Pow(const int64x2_t val) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_16(mht_16_v, 683, "", "./tensorflow/core/kernels/quantization_utils.h", "Divide64x2Pow");

  static constexpr int64 FIRST_BIT_VAL = 0x0000000000000001;
  static const int64x2_t FIRST_BIT = vmovq_n_s64(FIRST_BIT_VAL);
  const int64x2_t val_sign = vshrq_n_s64(val, 63);
  const int64x2_t neg_offset = vandq_s64(val_sign, FIRST_BIT);
  const int64x2_t val_with_offset = vsubq_s64(val, neg_offset);
  const int64x2_t shifted_wo_offset =
      vsraq_n_s64(neg_offset, val_with_offset, POW);
  return shifted_wo_offset;
}

// 32bit x 2 NEON accelerated lerp computation.
template <int RESOLUTION>
inline int32x2_t ComputeLerp32x2(const int32x2_t top_left,
                                 const int32x2_t top_right,
                                 const int32x2_t bottom_left,
                                 const int32x2_t bottom_right,
                                 const int32x2_t x_lerp,
                                 const int32x2_t y_lerp) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_17(mht_17_v, 704, "", "./tensorflow/core/kernels/quantization_utils.h", "ComputeLerp32x2");

  static_assert(RESOLUTION < 31, "RESOLUTION must be less than 31");
  constexpr int32 RESOLUTION_MULT32 = (1 << RESOLUTION);
  static const int32x2_t RESOLUTION_MULT32x2 = vmov_n_s32(RESOLUTION_MULT32);

  const int64x2_t top_left_x_res = vmull_s32(top_left, RESOLUTION_MULT32x2);
  const int64x2_t bottom_left_x_res =
      vmull_s32(bottom_left, RESOLUTION_MULT32x2);

  const int32x2_t top_right_sub_top_left = vsub_s32(top_right, top_left);
  const int64x2_t top_x_res =
      vmlal_s32(top_left_x_res, top_right_sub_top_left, x_lerp);
  const int32x2_t bottom_right_sub_bottom_left =
      vsub_s32(bottom_right, bottom_left);
  const int64x2_t bottom_x_res =
      vmlal_s32(bottom_left_x_res, bottom_right_sub_bottom_left, x_lerp);

  const int64x2_t bottom_sub_top_x_res = vsubq_s64(bottom_x_res, top_x_res);
  const int64x2_t bottom_sub_top =
      Divide64x2Pow<RESOLUTION>(bottom_sub_top_x_res);
  const int32x2_t bottom_sub_top_32 = vqmovn_s64(bottom_sub_top);
  const int64x2_t top_add_bottom_sub_top_mul_ylerp_x_res =
      vmlal_s32(top_x_res, bottom_sub_top_32, y_lerp);
  const int64x2_t retval =
      Divide64x2PowRound<RESOLUTION>(top_add_bottom_sub_top_mul_ylerp_x_res);
  const int32x2_t retval32 = vqmovn_s64(retval);
  return retval32;
}

// 8bit x 8 NEON accelerated lerp computation.
template <int RESOLUTION>
inline uint8x8_t ComputeLerp8x8(const uint8x8_t top_left8x8,
                                const uint8x8_t top_right8x8,
                                const uint8x8_t bottom_left8x8,
                                const uint8x8_t bottom_right8x8,
                                const int16x8_t x_lerp,
                                const int16x8_t y_lerp) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_18(mht_18_v, 743, "", "./tensorflow/core/kernels/quantization_utils.h", "ComputeLerp8x8");

  static_assert(RESOLUTION < 8, "RESOLUTION must be less than 8");
  constexpr uint8 RESOLUTION_MULT_VAL = (1 << RESOLUTION);
  static const uint8x8_t RESOLUTION_MULT = vdup_n_u8(RESOLUTION_MULT_VAL);

  const int16x8_t top_left_x_res =
      vreinterpretq_s16_u16(vmull_u8(top_left8x8, RESOLUTION_MULT));
  const int16x8_t bottom_left_x_res =
      vreinterpretq_s16_u16(vmull_u8(bottom_left8x8, RESOLUTION_MULT));

  const int16x8_t top_right_sub_top_left =
      vreinterpretq_s16_u16(vsubl_u8(top_right8x8, top_left8x8));
  const int16x8_t top_x_res =
      vmlaq_s16(top_left_x_res, top_right_sub_top_left, x_lerp);

  const int16x8_t bottom_right_sub_bottom_left =
      vreinterpretq_s16_u16(vsubl_u8(bottom_right8x8, bottom_left8x8));
  const int16x8_t bottom_x_res =
      vmlaq_s16(bottom_left_x_res, bottom_right_sub_bottom_left, x_lerp);

  const int16x8_t bottom_sub_top_x_res = vsubq_s16(bottom_x_res, top_x_res);
  const int16x8_t bottom_sub_top =
      Divide16x8Pow<RESOLUTION>(bottom_sub_top_x_res);
  const int16x8_t top_add_bottom_sub_top_mul_ylerp_x_res =
      vmlaq_s16(top_x_res, bottom_sub_top, y_lerp);
  const int16x8_t retval16 =
      Divide16x8PowRound<RESOLUTION>(top_add_bottom_sub_top_mul_ylerp_x_res);
  const uint8x8_t retval = vmovn_u16(vreinterpretq_u16_s16(retval16));
  return retval;
}

// Requantize 8 x 8 quints to 8 x 32 qints in parallel by neon
// Return std::array instead of pointer to leverage return value optimization
inline std::array<int32x4_t, 2> Requantize8x8To32Neon(
    const uint8* input_ptr, const int64x2_t input_0_64x2,
    const int32x2_t input_mult_32x2) {
  const uint8x8_t input_value_8x8 = vld1_u8(input_ptr);
  const int16x8_t input_value_16x8 =
      vreinterpretq_s16_u16(vmovl_u8(input_value_8x8));
  const int16x4_t input_value_low_16x4 = vget_low_s16(input_value_16x8);
  const int16x4_t input_value_high_16x4 = vget_high_s16(input_value_16x8);
  const int32x4_t input_value_low_32x4 = vmovl_s16(input_value_low_16x4);
  const int32x4_t input_value_high_32x4 = vmovl_s16(input_value_high_16x4);
  const int32x2_t input_value_low_low_32x2 = vget_low_s32(input_value_low_32x4);
  const int32x2_t input_value_low_high_32x2 =
      vget_high_s32(input_value_low_32x4);
  const int32x2_t input_value_high_low_32x2 =
      vget_low_s32(input_value_high_32x4);
  const int32x2_t input_value_high_high_32x2 =
      vget_high_s32(input_value_high_32x4);
  const int64x2_t mult_result_low_low_64x2 =
      vmlal_s32(input_0_64x2, input_value_low_low_32x2, input_mult_32x2);
  const int64x2_t mult_result_low_high_64x2 =
      vmlal_s32(input_0_64x2, input_value_low_high_32x2, input_mult_32x2);
  const int64x2_t mult_result_high_low_64x2 =
      vmlal_s32(input_0_64x2, input_value_high_low_32x2, input_mult_32x2);
  const int64x2_t mult_result_high_high_64x2 =
      vmlal_s32(input_0_64x2, input_value_high_high_32x2, input_mult_32x2);
  const int32x2_t output_value_low_low_32x2 =
      vqmovn_s64(mult_result_low_low_64x2);
  const int32x2_t output_value_low_high_32x2 =
      vqmovn_s64(mult_result_low_high_64x2);
  const int32x2_t output_value_high_low_32x2 =
      vqmovn_s64(mult_result_high_low_64x2);
  const int32x2_t output_value_high_high_32x2 =
      vqmovn_s64(mult_result_high_high_64x2);
  const int32x4_t output_value_low_32x4 =
      vcombine_s32(output_value_low_low_32x2, output_value_low_high_32x2);
  const int32x4_t output_value_high_32x4 =
      vcombine_s32(output_value_high_low_32x2, output_value_high_high_32x2);
  return std::array<int32x4_t, 2>{
      {output_value_low_32x4, output_value_high_32x4}};
}

// Speeds up the 8->32bit conversion using fixed-point arithmetic and NEON SIMD
// intrinsics for ARM platforms.
template <>
inline void RequantizeManyInNewRange<quint8, qint32>(
    const quint8* input, int64 count, float min_input, float max_input,
    float min_output, float max_output, qint32* output) {
  // Pre-calculate zero position and multiplier.
  // Calculate 0 and 1 value in float.
  const float code_0_float = QuantizedToFloat<quint8>(0, min_input, max_input);
  const float code_1_float = QuantizedToFloat<quint8>(1, min_input, max_input);

  // Cast 0 and 1 value in int64.
  const int64 code_0_int64 =
      FloatToQuantizedUnclamped<qint32>(code_0_float, min_output, max_output);
  const int64 code_1_int64 =
      FloatToQuantizedUnclamped<qint32>(code_1_float, min_output, max_output);

  // Calculate multiplier.
  const int32 mult_int32 = static_cast<int32>(code_1_int64 - code_0_int64);

  // Broadcast 0 position and multiplier to lanes
  const int64x2_t code_0_64x2 = vmovq_n_s64(code_0_int64);
  const int32x2_t mult_32x2 = vmov_n_s32(mult_int32);

  int64 i = 0;

  // Use SIMD to requantize array.
  for (; i < (count - 7); i += 8) {
    const uint8* input_ptr = &(input->value) + i;
    int32* output_ptr = &(output->value) + i;
    const std::array<int32x4_t, 2> output_value =
        Requantize8x8To32Neon(input_ptr, code_0_64x2, mult_32x2);
    vst1q_s32(output_ptr + 0, output_value[0]);
    vst1q_s32(output_ptr + 4, output_value[1]);
  }

  // Requantize remaining elements in array without SIMD.
  const int64 lowest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64_t>(Eigen::NumTraits<qint32>::highest());

  for (; i < count; ++i) {
    const int64 input_value = static_cast<int64_t>(input[i]);
    int64 output_value = code_0_int64 + (input_value * mult_int32);
    output_value = std::max(output_value, lowest_quantized);
    output_value = std::min(output_value, highest_quantized);
    output[i] = static_cast<int32>(output_value);
  }
}

#else

// If SIMD implementations aren't available, then use these default reference
// versions.
template <>
inline void RequantizeManyInNewRange<qint32, quint8>(
    const qint32* input, int64_t count, float min_input, float max_input,
    float min_output, float max_output, quint8* output) {
  RequantizeManyInNewRangeReference(input, count, min_input, max_input,
                                    min_output, max_output, output);
}

template <>
inline void RequantizeManyInNewRange<quint8, qint32>(
    const quint8* input, int64_t count, float min_input, float max_input,
    float min_output, float max_output, qint32* output) {
  RequantizeManyInNewRange8To32BitReference(input, count, min_input, max_input,
                                            min_output, max_output, output);
}

#endif

template <int shift>
struct int64_right_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(int64_right_shift_op)
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const int64_t operator()(const int64_t a) const {
    return a >> shift;
  }
};

// See RequantizeManyInNewRange() for a non-eigen reference implementation.
template <class T1, class T2>
inline void RequantizeManyInNewRangeUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min_input,
    float max_input, float min_output, float max_output, Tensor* output) {
  auto input_array = input.flat<T1>();
  QuantizedToFloatStruct<T1> q2f(min_input, max_input);
  auto input_float = DEQUANTIZE_WITH_EIGEN(input_array, q2f);
  FloatToQuantizedStruct<T2> f2q(min_output, max_output);
  auto input_requantized = QUANTIZE_WITH_EIGEN(input_float, f2q, T2);

  output->flat<T2>().device(device) = input_requantized;
}

// See RequantizeManyInNewRange() for a non-eigen reference implementation.
//
// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRangeUsingEigen<qint32, quint8>(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min_input,
    float max_input, float min_output, float max_output, Tensor* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the non-Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const float input_rezero = (min_input + max_input) / 2.0;
  const int64_t range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int64_t>(255.0 * (1 << fp_shift) *
                                                 input_range / output_range);
  const int64_t input_offset_fp =
      static_cast<int64_t>(input_rezero * recip_output_range * (1 << fp_shift));
  const int64_t output_offset_fp =
      output_range == 0.0
          ? 0
          : std::lround((1 << fp_shift) * (min_output * 255.0) / output_range);
  const int64_t rounding_delta = 1 << (fp_shift - 1);

  // Inside this eigen expression we just do minimal adds, multiplies, and
  // shifts. It should be possible to perform all the calculations in 32-bit
  // rather than 64, but that's not been implemented yet.
  auto input_array = input.flat<qint32>();
  auto fp_value = ((input_array.template cast<int64_t>() * range_scale_fp)
                       .unaryExpr(int64_right_shift_op<32>())) +
                  (input_offset_fp - output_offset_fp + rounding_delta);
  auto intermediate = fp_value.unaryExpr(int64_right_shift_op<fp_shift>());
  auto input_requantized = intermediate.cwiseMax(int64_t{0})
                               .cwiseMin(int64_t{255})
                               .template cast<int32>()
                               .template cast<quint8>();
  output->flat<quint8>().device(device) = input_requantized;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void FloatTensorToQuantizedInPlaceUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min,
    float max, Tensor* result) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_19(mht_19_v, 964, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatTensorToQuantizedInPlaceUsingEigen");

  DCHECK_EQ(DataTypeToEnum<T>::v(), result->dtype());
  auto flat_input = input.flat<float>();
  auto flat_result = result->flat<T>();
  DCHECK_EQ(flat_input.size(), flat_result.size());

  FloatToQuantizedStruct<T> f2q(min, max);
  flat_result.device(device) = QUANTIZE_WITH_EIGEN(flat_input, f2q, T);
}

template <class T>
void FloatTensorToQuantizedInPlace(const Tensor& input, float min, float max,
                                   Tensor* result) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_20(mht_20_v, 979, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatTensorToQuantizedInPlace");

  DCHECK_EQ(DataTypeToEnum<T>::v(), result->dtype());
  auto flat_input = input.flat<float>();
  auto flat_result = result->flat<T>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());
  for (int i = 0; i < data_size; ++i) {
    flat_result(i) = FloatToQuantized<T>(flat_input(i), min, max);
  }
}

template <class T>
Tensor FloatTensorToQuantized(const Tensor& input, float min, float max) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_21(mht_21_v, 994, "", "./tensorflow/core/kernels/quantization_utils.h", "FloatTensorToQuantized");

  Tensor result(DataTypeToEnum<T>::v(), input.shape());
  FloatTensorToQuantizedInPlace<T>(input, min, max, &result);
  return result;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void QuantizedTensorToFloatInPlaceUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min,
    float max, Tensor* result) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_22(mht_22_v, 1007, "", "./tensorflow/core/kernels/quantization_utils.h", "QuantizedTensorToFloatInPlaceUsingEigen");

  DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  auto flat_input = input.flat<T>();
  auto flat_result = result->flat<float>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());

  QuantizedToFloatStruct<T> q2f(min, max);
  flat_result.device(device) = DEQUANTIZE_WITH_EIGEN(flat_input, q2f);
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void QuantizedTensorToFloatInPlace(const Tensor& input, float min, float max,
                                   Tensor* result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_23(mht_23_v, 1024, "", "./tensorflow/core/kernels/quantization_utils.h", "QuantizedTensorToFloatInPlace");

  DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  auto flat_input = input.flat<T>();
  auto flat_result = result->flat<float>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());
  for (int i = 0; i < data_size; ++i) {
    flat_result(i) = QuantizedToFloat<T>(flat_input(i), min, max);
  }
}

template <class T>
Tensor QuantizedTensorToFloat(const Tensor& input, float min, float max) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_24(mht_24_v, 1039, "", "./tensorflow/core/kernels/quantization_utils.h", "QuantizedTensorToFloat");

  Tensor result(DT_FLOAT, input.shape());
  QuantizedTensorToFloatInPlace<T>(input, min, max, &result);
  return result;
}

void GetOutputMinAndMaxForQuantizedAdd(float input_min, float input_max,
                                       float smaller_input_min,
                                       float smaller_input_max,
                                       float* output_min, float* output_max);

// Add <input> and <smaller_input>.  If <smaller_input> has fewer elements than
// <input>, then it is broadcast onto <input>.
template <typename T1, typename T2, typename T3>
void QuantizedAddUsingEigen(const Eigen::ThreadPoolDevice& device,
                            const Tensor& input, float input_min,
                            float input_max, const Tensor& smaller_input,
                            float smaller_input_min, float smaller_input_max,
                            Tensor* output, float* output_min,
                            float* output_max) {
  const auto& input_flat = input.flat<T1>();
  const auto& smaller_input_flat = smaller_input.flat<T2>();
  auto output_flat = output->flat<T3>();

  GetOutputMinAndMaxForQuantizedAdd(input_min, input_max, smaller_input_min,
                                    smaller_input_max, output_min, output_max);
  // To do addition properly, we need to compensate for a possibly unbalanced
  // zero point in the total representation. The quantized value that
  // represents the real number zero needs to be subtracted before addition to
  // make sure that the identity of zero + zero = zero holds.
  const T3 zero_in_total_space =
      FloatToQuantized<T3>(0.0f, *output_min, *output_max);

  const int64_t input_element_count = input.NumElements();
  const int64_t smaller_input_element_count = smaller_input.NumElements();

  QuantizedToFloatStruct<T1> input_q2f(input_min, input_max);
  QuantizedToFloatStruct<T2> smaller_input_q2f(smaller_input_min,
                                               smaller_input_max);
  FloatToQuantizedStruct<T3> f2q(*output_min, *output_max);

  auto smaller_input_float =
      DEQUANTIZE_WITH_EIGEN(smaller_input_flat, smaller_input_q2f);
  auto smaller_input_in_total_space =
      QUANTIZE_WITH_EIGEN(smaller_input_float, f2q, T3);

  auto input_float = DEQUANTIZE_WITH_EIGEN(input_flat, input_q2f);
  auto input_in_total_space = QUANTIZE_WITH_EIGEN(input_float, f2q, T3);

  Eigen::array<Eigen::DenseIndex, 1> bcast;
  bcast[0] = input_element_count / smaller_input_element_count;
  output_flat.device(device) =
      input_in_total_space +
      (smaller_input_in_total_space.broadcast(bcast) + zero_in_total_space);
}

// This is a reference implementation of the bias addition for quantized
// buffers, designed to provide a clear specification for the result we
// want. We'll want to specialize this for particular hardware, and
// probably even fuse it with matrix multiplications in a lot of cases. It's
// important to show the clamping behavior we want in particular.
template <typename T1, typename T2, typename T3>
void QuantizedAdd(const Eigen::ThreadPoolDevice& device, const Tensor& input,
                  float input_min, float input_max, const Tensor& smaller_input,
                  float smaller_input_min, float smaller_input_max,
                  Tensor* output, float* output_min, float* output_max) {
  const auto& input_flat = input.flat<T1>();
  const auto& smaller_input_flat = smaller_input.flat<T2>();
  auto output_flat = output->flat<T3>();

  GetOutputMinAndMaxForQuantizedAdd(input_min, input_max, smaller_input_min,
                                    smaller_input_max, output_min, output_max);
  // To do addition properly, we need to compensate for a possibly unbalanced
  // zero point in the total representation. The quantized value that
  // represents the real number zero needs to be subtracted before addition to
  // make sure that the identity of zero + zero = zero holds.
  const T3 zero_in_total_space =
      FloatToQuantized<T3>(0.0f, *output_min, *output_max);

  const int64_t input_element_count = input.NumElements();
  const int64_t smaller_input_element_count = smaller_input.NumElements();

  float total_min = *output_min;
  float total_max = *output_max;
  const size_t how_many_iterations =
      (input_element_count / smaller_input_element_count);
  for (size_t iteration = 0; iteration < how_many_iterations; ++iteration) {
    const size_t offset = iteration * smaller_input_element_count;
    for (int c = 0; c < smaller_input_element_count; ++c) {
      const int index = (offset + c);
      // The two numbers we're going to add can each be in very different
      // ranges (e.g. the quantized value '127' may represent very different
      // real numbers in both) so we need to convert them to a common range
      // before we sum them.
      const T1 input_value = input_flat(index);
      const T3 input_in_total_space = RequantizeInNewRange<T1, T3>(
          input_value, input_min, input_max, total_min, total_max);
      const T2 smaller_input_value = smaller_input_flat(c);
      const T3 smaller_input_in_total_space =
          RequantizeInNewRange<T2, T3>(smaller_input_value, smaller_input_min,
                                       smaller_input_max, total_min, total_max);
      const T3 total_pre = input_in_total_space + smaller_input_in_total_space;
      // As noted above, we need to compensate for the offset of the actual
      // zero point in the space we're operating in.
      const T3 total = total_pre + zero_in_total_space;
      output_flat(index) = total;
    }
  }
}

// See gemmlowp/internal/multi_thread_gemm.h for the semantics of Execute.
class TensorflowGemmlowpWorkersPool {
 public:
  TensorflowGemmlowpWorkersPool(thread::ThreadPool* workers)
      : workers_(workers) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_25(mht_25_v, 1156, "", "./tensorflow/core/kernels/quantization_utils.h", "TensorflowGemmlowpWorkersPool");
}

  ~TensorflowGemmlowpWorkersPool() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_26(mht_26_v, 1161, "", "./tensorflow/core/kernels/quantization_utils.h", "~TensorflowGemmlowpWorkersPool");

    // This workaround ensures that all worker tasks have exited methods in the
    // BlockingCounter. Without this, there is a race where the context is torn
    // down while the counter is in use.
    counter_to_decrement_when_ready_.Reset(0);
  }

  void Execute(const std::vector<gemmlowp::Task*>& tasks) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_27(mht_27_v, 1171, "", "./tensorflow/core/kernels/quantization_utils.h", "Execute");

    assert(!tasks.empty());
    assert(workers_ != nullptr);
    counter_to_decrement_when_ready_.Reset(tasks.size());
    for (gemmlowp::Task* task : tasks) {
      workers_->Schedule([this, task]() {
        // TODO(cwhipkey): get a local_allocator from a thread local storage.
        gemmlowp::Allocator local_allocator;
        CHECK(task != nullptr);
        task->local_allocator = &local_allocator;
        task->Run();
        counter_to_decrement_when_ready_.DecrementCount();
      });
    }
    counter_to_decrement_when_ready_.Wait();
    for (gemmlowp::Task* task : tasks) {
      delete task;
    }
  }

 private:
  thread::ThreadPool* const workers_;

  // The BlockingCounter used to wait for the workers.
  gemmlowp::BlockingCounter counter_to_decrement_when_ready_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorflowGemmlowpWorkersPool);
};

class TensorflowGemmContext : public gemmlowp::MultiThreadGemmContextBase {
 public:
  TensorflowGemmContext(int num_threads, thread::ThreadPool* workers)
      : workers_pool_(workers) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_28(mht_28_v, 1206, "", "./tensorflow/core/kernels/quantization_utils.h", "TensorflowGemmContext");

    set_max_num_threads(num_threads);
  }

  TensorflowGemmlowpWorkersPool* workers_pool() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantization_utilsDTh mht_29(mht_29_v, 1213, "", "./tensorflow/core/kernels/quantization_utils.h", "workers_pool");
 return &workers_pool_; }

 private:
  TensorflowGemmlowpWorkersPool workers_pool_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorflowGemmContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUANTIZATION_UTILS_H_
