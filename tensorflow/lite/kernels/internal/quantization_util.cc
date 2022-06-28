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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/quantization_util.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace tflite {

namespace {
// These constants are used to manipulate the binary representation of doubles.
// Double-precision binary64 floating point format is:
// Bit |  63  |  62-52   |   51-0   |
//     | Sign | Exponent | Fraction |
// To avoid 64-bit integers as much as possible, I break this into high and
// low 32-bit chunks. High is:
// Bit |  31  |  30-20   |      19-0     |
//     | Sign | Exponent | High Fraction |
// Low is:
// Bit |     31-0     |
//     | Low Fraction |
// We then access the components through logical bit-wise operations to
// extract the parts needed, with the positions and masks derived from the
// layout shown above.
constexpr uint64_t kSignMask = 0x8000000000000000LL;
constexpr uint64_t kExponentMask = 0x7ff0000000000000LL;
constexpr int32_t kExponentShift = 52;
constexpr int32_t kExponentBias = 1023;
constexpr uint32_t kExponentIsBadNum = 0x7ff;
constexpr uint64_t kFractionMask = 0x000fffffffc00000LL;
constexpr uint32_t kFractionShift = 22;
constexpr uint32_t kFractionRoundingMask = 0x003fffff;
constexpr uint32_t kFractionRoundingThreshold = 0x00200000;
}  // namespace

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "QuantizeMultiplier");

#if TFLITE_SINGLE_ROUNDING
  // Single-rounding MultiplyByQuantizedMultiplier only supports positive
  // multipliers.
  // TFLITE_DCHECK(double_multiplier >= 0);
#endif
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
#ifdef TFLITE_EMULATE_FLOAT
  // If we're trying to avoid the use of floating-point instructions (for
  // example on microcontrollers) then use an alternative implementation
  // that only requires integer and bitwise operations. To enable this, you
  // need to set the define during the build process for your platform.
  int64_t q_fixed = IntegerFrExp(double_multiplier, shift);
#else   // TFLITE_EMULATE_FLOAT
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1LL << 31)));
#endif  // TFLITE_EMULATE_FLOAT
  TFLITE_CHECK(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
#if TFLITE_SINGLE_ROUNDING
  // Single-rounding MultiplyByQuantizedMultiplier doesn't support a shift > 30,
  // saturate it.
  if (*shift > 30) {
    *shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
#endif
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_1(mht_1_v, 280, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "QuantizeMultiplierGreaterThanOne");

  TFLITE_CHECK_GT(double_multiplier, 1.);
  QuantizeMultiplier(double_multiplier, quantized_multiplier, left_shift);
  TFLITE_CHECK_GE(*left_shift, 0);
}

void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_2(mht_2_v, 291, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "QuantizeMultiplierSmallerThanOneExp");

  TFLITE_CHECK_LT(double_multiplier, 1.);
  TFLITE_CHECK_GT(double_multiplier, 0.);
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  TFLITE_CHECK_LE(shift, 0);
  *left_shift = shift;
}

int64_t IntegerFrExp(double input, int* shift) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_3(mht_3_v, 303, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "IntegerFrExp");

  // Make sure our assumptions about the double layout hold.
  TFLITE_CHECK_EQ(8, sizeof(double));

  // We want to access the bits of the input double value directly, which is
  // tricky to do safely, so use a union to handle the casting.
  union {
    double double_value;
    uint64_t double_as_uint;
  } cast_union;
  cast_union.double_value = input;
  const uint64_t u = cast_union.double_as_uint;

  // If the bitfield is all zeros apart from the sign bit, this is a normalized
  // zero value, so return standard values for this special case.
  if ((u & ~kSignMask) == 0) {
    *shift = 0;
    return 0;
  }

  // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
  // the exponent, and distinguished by whether the fractions are zero or
  // non-zero.
  const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
  if (exponent_part == kExponentIsBadNum) {
    *shift = std::numeric_limits<int>::max();
    if (u & kFractionMask) {
      // NaN, so just return zero (with the exponent set to INT_MAX).
      return 0;
    } else {
      // Infinity, so return +/- INT_MAX.
      if (u & kSignMask) {
        return std::numeric_limits<int64_t>::min();
      } else {
        return std::numeric_limits<int64_t>::max();
      }
    }
  }

  // The shift is fairly easy to extract from the high bits of the double value,
  // just by masking it out and applying a bias. The std::frexp() implementation
  // always returns values between 0.5 and 1.0 though, whereas the exponent
  // assumes 1.0 to 2.0 is the standard range, so I add on one to match that
  // interface.
  *shift = (exponent_part - kExponentBias) + 1;

  // There's an implicit high bit in the double format definition, so make sure
  // we include that at the top, and then reconstruct the rest of the fractional
  // value from the remaining fragments.
  int64_t fraction = 0x40000000 + ((u & kFractionMask) >> kFractionShift);

  // We're cutting off some bits at the bottom, so to exactly match the standard
  // frexp implementation here we'll apply rounding by adding one to the least
  // significant bit of the result if the discarded portion is over half of the
  // maximum.
  if ((u & kFractionRoundingMask) > kFractionRoundingThreshold) {
    fraction += 1;
  }
  // Negate the fraction if the sign bit was set.
  if (u & kSignMask) {
    fraction *= -1;
  }

  return fraction;
}

double DoubleFromFractionAndShift(int64_t fraction, int shift) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_4(mht_4_v, 372, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "DoubleFromFractionAndShift");

  union {
    double double_value;
    uint64_t double_as_uint;
  } result;

  // Detect NaNs and infinities.
  if (shift == std::numeric_limits<int>::max()) {
    if (fraction == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    } else if (fraction > 0) {
      return std::numeric_limits<double>::infinity();
    } else {
      return -std::numeric_limits<double>::infinity();
    }
  }

  // Return a normalized zero for a zero fraction.
  if (fraction == 0) {
    result.double_as_uint = 0;
    return result.double_value;
  }

  bool is_negative = (fraction < 0);
  int64_t encoded_fraction = is_negative ? -fraction : fraction;
  int64_t encoded_shift = (shift - 1);
  while (encoded_fraction < 0x40000000) {
    encoded_fraction *= 2;
    encoded_shift -= 1;
  }
  while (encoded_fraction > 0x80000000) {
    encoded_fraction /= 2;
    encoded_shift += 1;
  }
  encoded_fraction -= 0x40000000;
  if (encoded_shift < -1022) {
    encoded_shift = -1023;
  } else if (encoded_shift > 1022) {
    encoded_shift = 1023;
  }
  encoded_shift += kExponentBias;
  uint64_t encoded_sign = is_negative ? kSignMask : 0;
  result.double_as_uint = encoded_sign | (encoded_shift << kExponentShift) |
                          (encoded_fraction << kFractionShift);
  return result.double_value;
}

double IntegerDoubleMultiply(double a, double b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_5(mht_5_v, 422, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "IntegerDoubleMultiply");

  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);
  // Detect NaNs and infinities.
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int result_shift = a_shift + b_shift + 1;
  const int64_t result_fraction = (a_fraction * b_fraction) >> 32;
  return DoubleFromFractionAndShift(result_fraction, result_shift);
}

int IntegerDoubleCompare(double a, double b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_6(mht_6_v, 440, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "IntegerDoubleCompare");

  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);

  // Detect NaNs and infinities.
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return 1;
  }

  if ((a_fraction == 0) && (b_fraction < 0)) {
    return 1;
  } else if ((a_fraction < 0) && (b_fraction == 0)) {
    return -1;
  } else if (a_shift < b_shift) {
    return -1;
  } else if (a_shift > b_shift) {
    return 1;
  } else if (a_fraction < b_fraction) {
    return -1;
  } else if (a_fraction > b_fraction) {
    return 1;
  } else {
    return 0;
  }
}

void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_7(mht_7_v, 474, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "PreprocessSoftmaxScaling");

  // If the overall multiplier (input and beta) is large, then exp() of an
  // input difference of 1 scaled by this will be large.  In other words, we
  // can cap the multiplier and know that, when it is used, the output will be
  // (round to) zero wherever the input is not at the maximum value.

  // If the overall scale is less than one, and input_integer_bits=0, then the
  // result is double equivalent of Q0.31 (actually with more precision). Thus
  // this generates a Q(input_integer_bits).(31-input_integer_bits)
  // representation.
#if TFLITE_SINGLE_ROUNDING
  const double max_real_multiplier = (1LL << 30) - 1.0;
#else
  const double max_real_multiplier = (1LL << 31) - 1.0;
#endif

#ifdef TFLITE_EMULATE_FLOAT
  const double input_beta = IntegerDoubleMultiply(beta, input_scale);
  int shift;
  int64_t fraction = IntegerFrExp(input_beta, &shift);
  shift += (31 - input_integer_bits);
  double input_beta_real_multiplier =
      DoubleFromFractionAndShift(fraction, shift);
  if (IntegerDoubleCompare(input_beta_real_multiplier, max_real_multiplier) >
      0) {
    input_beta_real_multiplier = max_real_multiplier;
  }
#else   // TFLITE_EMULATE_FLOAT
  const double input_beta_real_multiplier =
      std::min<double>(beta * input_scale * (1 << (31 - input_integer_bits)),
                       max_real_multiplier);
#endif  // TFLITE_EMULATE_FLOAT

  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                   quantized_multiplier, left_shift);
}

void PreprocessLogSoftmaxScalingExp(double beta, double input_scale,
                                    int input_integer_bits,
                                    int32_t* quantized_multiplier,
                                    int* left_shift,
                                    int32_t* reverse_scaling_divisor,
                                    int* reverse_scaling_left_shift) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_8(mht_8_v, 519, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "PreprocessLogSoftmaxScalingExp");

  PreprocessSoftmaxScaling(beta, input_scale, input_integer_bits,
                           quantized_multiplier, left_shift);

  // Also calculate what amounts to the inverse scaling factor for the input.
  const double real_reverse_scaling_divisor =
      (1 << (31 - *left_shift)) / static_cast<double>(*quantized_multiplier);
  tflite::QuantizeMultiplierSmallerThanOneExp(real_reverse_scaling_divisor,
                                              reverse_scaling_divisor,
                                              reverse_scaling_left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_9(mht_9_v, 535, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "CalculateInputRadius");

#ifdef TFLITE_EMULATE_FLOAT
  int64_t result = (1 << input_integer_bits) - 1;
  result <<= (total_signed_bits - input_integer_bits);
  result >>= input_left_shift;
  return result;
#else   // TFLITE_EMULATE_FLOAT
  const double max_input_rescaled =
      1.0 * ((1 << input_integer_bits) - 1) *
      (1LL << (total_signed_bits - input_integer_bits)) /
      (1LL << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int>(std::floor(max_input_rescaled));
#endif  // TFLITE_EMULATE_FLOAT
}

void NudgeQuantizationRange(const float min, const float max,
                            const int quant_min, const int quant_max,
                            float* nudged_min, float* nudged_max,
                            float* nudged_scale) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_10(mht_10_v, 559, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "NudgeQuantizationRange");

  // This code originates from tensorflow/core/kernels/fake_quant_ops_functor.h.
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *nudged_scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / *nudged_scale;
  uint16_t nudged_zero_point;
  if (zero_point_from_min < quant_min_float) {
    nudged_zero_point = static_cast<uint16_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    nudged_zero_point = static_cast<uint16_t>(quant_max);
  } else {
    nudged_zero_point = static_cast<uint16_t>(TfLiteRound(zero_point_from_min));
  }
  *nudged_min = (quant_min_float - nudged_zero_point) * (*nudged_scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*nudged_scale);
}

void FakeQuantizeArray(const float nudged_scale, const float nudged_min,
                       const float nudged_max, const float* input_data,
                       float* output_data, const float size) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_11(mht_11_v, 582, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "FakeQuantizeArray");

  // This code originates from tensorflow/core/kernels/fake_quant_ops_functor.h.
  const float inv_nudged_scale = 1.0f / nudged_scale;

  for (int i = 0; i < size; i++) {
    const float src_val = input_data[i];
    const float clamped = std::min(nudged_max, std::max(nudged_min, src_val));
    const float clamped_shifted = clamped - nudged_min;
    const float dst_val =
        TfLiteRound(clamped_shifted * inv_nudged_scale) * nudged_scale +
        nudged_min;
    output_data[i] = dst_val;
  }
}

bool CheckedLog2(const float x, int* log2_result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_12(mht_12_v, 600, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "CheckedLog2");

  // Using TfLiteRound instead of std::round and std::log instead of
  // std::log2 to work around these functions being missing in a toolchain
  // used in some TensorFlow tests as of May 2018.
  const float x_log2 = std::log(x) * (1.0f / std::log(2.0f));
  const float x_log2_rounded = TfLiteRound(x_log2);
  const float x_log2_fracpart = x_log2 - x_log2_rounded;

  *log2_result = static_cast<int>(x_log2_rounded);
  return std::abs(x_log2_fracpart) < 1e-3f;
}

void QuantizeMultiplierArray(const double* effective_scales, size_t size,
                             int32_t* effective_scale_significand,
                             int* effective_shift) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSquantization_utilDTcc mht_13(mht_13_v, 617, "", "./tensorflow/lite/kernels/internal/quantization_util.cc", "QuantizeMultiplierArray");

  for (size_t i = 0; i < size; ++i) {
    QuantizeMultiplier(effective_scales[i], &effective_scale_significand[i],
                       &effective_shift[i]);
  }
}

}  // namespace tflite
