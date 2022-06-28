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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh() {
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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Used in tests and template parameters to control which version of depthwise
// convolution is called. Primarily for reference code, and specializations
// forced in tests.
enum class DepthwiseConvImplementation {
  // Run all tests against kUseStandardEntry even if also testing another
  // kernel, since we need to be sure that the main DepthwiseConv() function in
  // optimized_ops.h dispatches to a correctly-executing kernel.
  kNone = 0,                 // The "default" option: use the normal
                             // DepthwiseConv kernel (entry) function.
  kUseGenericKernel,         // Forced use of generic kernel.
  kUseNeon3x3,               // 3x3 kernel that uses NEON when available.
  kUseNeon3x3DotProduct,     // 3x3 kernel that uses dot-product enabled NEON
                             // when available.
  kUseCModel3x3DotProduct,   // 3x3 kernel, reference C model that is intended
                             // to match overall design NEON code.
  kUseUnwound3x3DotProduct,  // 3x3 kernel, reference C model with unwound loops
                             // and some arrays.
  kUseIntrinsics3x3DotProduct,  // 3x3 kernel using NEON intrinsics.
};

// Category of depthwise convolution output rounding.
enum class DepthwiseConvOutputRounding {
  kNone = 0,      // Invalid: specific method must be specified.
  kAwayFromZero,  // Original method: exact halves rounded away from zero.
  kUpward,        // Halves towards +infinity: adds 0.5 before truncate.
  // This is where a future kNearestEven would be placed.
};

// Category of depthwise convolution depth multiplication.
enum class DepthwiseConvDepthMultiplication {
  kNoMultiplication = 0,  // Depth multiplier = 1.
  kUnitInputDepth,        // Input depth = 1, output depth = depth multiplier.
};

namespace reference_ops {
namespace depthwise_conv {

template <DepthwiseConvOutputRounding output_rounding>
inline int32_t DepthwiseConvRound(int32_t x, int32_t quantized_multiplier,
                                  int shift) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_0(mht_0_v, 235, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "DepthwiseConvRound");

  TFLITE_DCHECK_NE(output_rounding, DepthwiseConvOutputRounding::kNone);
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

// Single-rounding MultiplyByQuantizedMultiplier
#if TFLITE_SINGLE_ROUNDING
template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kAwayFromZero>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                 x * (1 << left_shift), quantized_multiplier),
                             right_shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>(
    int32_t x, int32_t quantized_multiplier, int shift) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_1(mht_1_v, 259, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>");

  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}
// Double-rounding MultiplyByQuantizedMultiplier
#else
template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kAwayFromZero>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>(
    int32_t x, int32_t quantized_multiplier, int shift) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_2(mht_2_v, 275, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>");

  using gemmlowp::SaturatingRoundingDoublingHighMul;
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int rounding_offset = right_shift > 0 ? 1 << (right_shift - 1) : 0;
  return (SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                            quantized_multiplier) +
          rounding_offset) >>
         right_shift;
}
#endif  // TFLITE_SINGLE_ROUNDING

template <DepthwiseConvOutputRounding output_rounding>
struct DepthwiseConvBasicKernel {
  static inline void Run(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const uint8_t* input_data, const RuntimeShape& filter_shape,
      const uint8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      uint8_t* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_3(mht_3_v, 297, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "Run");

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int ic = 0; ic < input_depth; ++ic) {
            for (int m = 0; m < depth_multiplier; m++) {
              const int oc = m + ic * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + dilation_width_factor * filter_x;
                  const int in_y =
                      in_y_origin + dilation_height_factor * filter_y;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    int32_t input_val =
                        input_data[Offset(input_shape, b, in_y, in_x, ic)];
                    int32_t filter_val = filter_data[Offset(
                        filter_shape, 0, filter_y, filter_x, oc)];
                    acc += (filter_val + filter_offset) *
                           (input_val + input_offset);
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[oc];
              }
              acc = DepthwiseConvRound<output_rounding>(acc, output_multiplier,
                                                        output_shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                  static_cast<uint8_t>(acc);
            }
          }
        }
      }
    }
  }

  // TODO(b/148596273): Reconcile reference versions, perhaps with common
  // MultiplyByQuantizedMultiplier or DepthwiseConvRound function.
  static inline void RunPerChannel(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const int8_t* input_data, const RuntimeShape& filter_shape,
      const int8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      int8_t* output_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_4(mht_4_v, 384, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "RunPerChannel");

    // Get parameters.
    // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t* output_multiplier = params.output_multiplier_per_channel;
    const int32_t* output_shift = params.output_shift_per_channel;

    // Check dimensions of the tensors.
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
            for (int m = 0; m < depth_multiplier; ++m) {
              const int output_channel = m + in_channel * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + dilation_width_factor * filter_x;
                  const int in_y =
                      in_y_origin + dilation_height_factor * filter_y;
                  // Zero padding by omitting the areas outside the image.
                  const bool is_point_inside_image =
                      (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height);
                  if (is_point_inside_image) {
                    int32_t input_val = input_data[Offset(
                        input_shape, batch, in_y, in_x, in_channel)];
                    int32_t filter_val = filter_data[Offset(
                        filter_shape, 0, filter_y, filter_x, output_channel)];
                    // Accumulate with 32 bits accumulator.
                    // In the nudging process during model quantization, we
                    // force real value of 0.0 be represented by a quantized
                    // value. This guarantees that the input_offset is a int8_t,
                    // even though it is represented using int32_t. int32_t +=
                    // int8_t
                    // * (int8_t - int8_t) so the highest value we can get from
                    // each accumulation is [-127, 127] * ([-128, 127] -
                    // [-128, 127]), which is [-32512, 32512]. log2(32512)
                    // = 14.98, which means we can accumulate at least 2^16
                    // multiplications without overflow. The accumulator is
                    // applied to a filter so the accumulation logic will hold
                    // as long as the filter size (filter_y * filter_x *
                    // in_channel) does not exceed 2^16, which is the case in
                    // all the models we have seen so far.
                    acc += filter_val * (input_val + input_offset);
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[output_channel];
              }
              acc = DepthwiseConvRound<output_rounding>(
                  acc, output_multiplier[output_channel],
                  output_shift[output_channel]);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, batch, out_y, out_x,
                                 output_channel)] = static_cast<int8_t>(acc);
            }
          }
        }
      }
    }
  }
};

}  // namespace depthwise_conv

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8_t* input_data, const RuntimeShape& filter_shape,
    const uint8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    uint8_t* output_data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdepthwiseconv_uint8DTh mht_5(mht_5_v, 491, "", "./tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h", "DepthwiseConv");

  return depthwise_conv::DepthwiseConvBasicKernel<
      DepthwiseConvOutputRounding::kAwayFromZero>::Run(params, input_shape,
                                                       input_data, filter_shape,
                                                       filter_data, bias_shape,
                                                       bias_data, output_shape,
                                                       output_data);
}

}  // namespace reference_ops
}  // end namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_
