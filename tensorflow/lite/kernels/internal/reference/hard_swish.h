/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ACTIVATIONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ACTIVATIONS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh() {
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


#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline int16_t SaturatingLeftShift(int16_t value, int amount) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh mht_0(mht_0_v, 194, "", "./tensorflow/lite/kernels/internal/reference/hard_swish.h", "SaturatingLeftShift");

  int32_t result = static_cast<int32_t>(value) * (1 << amount);
  result = std::min<int32_t>(result, std::numeric_limits<int16_t>::max());
  result = std::max<int32_t>(result, std::numeric_limits<int16_t>::min());
  return result;
}

// Similar to ARM instruction SQDMULH.
// Similar to gemmlowp::SaturatingRoundingDoublingHighMul except
// rounding to zero instead of to nearest (SQRDMULH).
inline std::int16_t SaturatingDoublingHighMul(std::int16_t a, std::int16_t b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh mht_1(mht_1_v, 207, "", "./tensorflow/lite/kernels/internal/reference/hard_swish.h", "SaturatingDoublingHighMul");

  bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
  std::int32_t a_32(a);
  std::int32_t b_32(b);
  std::int32_t ab_32 = a_32 * b_32;
  std::int16_t ab_x2_high16 = static_cast<std::int16_t>((ab_32) / (1 << 15));
  return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}

template <typename T>
inline void HardSwish(const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh mht_2(mht_2_v, 221, "", "./tensorflow/lite/kernels/internal/reference/hard_swish.h", "HardSwish");

  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Float");
  auto matching_size = MatchingFlatSize(input_shape, output_shape);
  const T* in_end = input_data + matching_size;
  for (; input_data < in_end; input_data++, output_data++) {
    const float in = *input_data;
    *output_data =
        in * std::min(static_cast<T>(6), std::max(static_cast<T>(0), in + 3)) /
        6;
  }
}

template <typename T>
inline void HardSwish(const HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePShard_swishDTh mht_3(mht_3_v, 239, "", "./tensorflow/lite/kernels/internal/reference/hard_swish.h", "HardSwish");

  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Quantized");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int16_t input_value = input_data[i] - params.input_zero_point;
    // Left-shift as much as we can without overflow/saturation to put
    // significant bits in the high bits of our 16-bit fixedpoint values, so
    // that fixed-point approximate computations below are as accurate as
    // possible.
    const int16_t input_value_on_hires_input_scale = input_value * (1 << 7);
    // Compute the input value on essentially the output scale, just not
    // right-shifted yet. This is the value that we'll use in the (x >= +3)
    // case, and that in the general case we'll multiply against the "relu-ish"
    // fixed-point multiplier in [0, 1].
    const int16_t input_value_on_preshift_output_scale =
        gemmlowp::SaturatingRoundingDoublingHighMul(
            input_value_on_hires_input_scale,
            params.output_multiplier_fixedpoint_int16);
    // Now compute the "relu-ish multiplier". In the (-3 <= x <= +3) case, that
    // is just an affine rescaling of x from [-3, 3] to [0, 1]. In the general
    // case, it is just that plus saturation at the boundaries of [-3, 3].
    // First, we rescale from [-3, 3] to [-1, 1], saturating.
    // That is done by rescaling the input value with a fixed-point multiplier
    // (reluish_multiplier_fixedpoint) and bit-shift such that we represent
    // that input value on the scale where the real value 3.0f is represented
    // by the quantized value 32768.  (+32768 is actually not representable as
    // int16_t, so this saturates at +32767, and that is seen empirically to be
    // a negligible contribution to numerical error/bias).
    //
    // This code is careful to correctly implement any magnitude of multiplier,
    // involving either a right shift or a left shift, with correct saturation
    // behavior in the left-shift case. This forces this code to be more
    // complicated, but is necessary for real applications: a partially
    // trained quantized MobileNet v3-small model that motivated this code
    // exhibits some large [min, max] range boundaries, of the order of
    // magnitude of 10 or 100 depending on layers.
    //
    // The next few lines are basically just an ordinary
    // MultiplyByQuantizedMultiplier, except that we are more careful here
    // about the fine details of saturation when left-shifting, because here
    // overflow in left-shift is a common case, not an anomaly as
    // MultiplyByQuantizedMultiplier assumes.
    int16_t reluish_value = input_value_on_hires_input_scale;
    // Shift left, saturating, as much as we can while ensuring that this
    // saturation will not contribute to the result. That is, left shift amount
    // reduced by 1.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(
          reluish_value, params.reluish_multiplier_exponent - 1);
    }
    // Apply the fixed-point multiplier, dividing the value by a divisor
    // ranging in [1, 2].
    reluish_value = gemmlowp::SaturatingRoundingDoublingHighMul(
        reluish_value, params.reluish_multiplier_fixedpoint_int16);
    // Apply the last bit of left-shift. Thus, in the left-shifting case, if
    // any saturation affects the result, it is happening here --- any
    // saturation having occurred above is overwritten here, not affecting the
    // result.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(reluish_value, 1);
    }
    // Shift right, in the right-shifting case.
    if (params.reluish_multiplier_exponent < 0) {
      reluish_value = gemmlowp::RoundingDivideByPOT(
          reluish_value, -params.reluish_multiplier_exponent);
    }
    // At this point we have rescaled the value into a 16bit fixedpoint
    // reluish_value in [-1, 1].
    // We now convert that to a 16bit fixedpoint value in [0, 1].
    reluish_value = (reluish_value + (1 << 15)) >> 1;
    // Use of SaturatingDoublingHighMul here is important to cancel the biases
    // from the above SaturatingRoundingDoublingHighMul.
    //
    // On a partially trained MobileNet-v3-small,
    //
    //                                       | bias on    |  ImageNet
    //                                       | quantized  |  Top-1
    // Operation used here                   | values     |  accuracy (50k)
    // --------------------------------------+------------+-----------
    // SaturatingDoublingHighMul             | -0.0024    |  58.920
    // SaturatingRoundingDoublingHighMul     | -0.0067    |  58.064
    //
    // In activations_test, this is covered by this testcase:
    //     QuantizedActivationsOpTest.HardSwishBias
    //
    const int16_t preshift_output_value = SaturatingDoublingHighMul(
        reluish_value, input_value_on_preshift_output_scale);
    // We were so far operating on the pre-shift output scale. Now we finally
    // apply that output shift, arriving at the final output scale.
    int16_t output_value = gemmlowp::RoundingDivideByPOT(
        preshift_output_value, -params.output_multiplier_exponent);
    output_value += params.output_zero_point;
    output_value =
        std::min<int16_t>(output_value, std::numeric_limits<T>::max());
    output_value =
        std::max<int16_t>(output_value, std::numeric_limits<T>::min());
    output_data[i] = output_value;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
