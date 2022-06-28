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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc() {
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
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace {

void PickOutputMultiplier(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    float* output_multiplier) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/internal/depthwiseconv_per_channel_quantized_test.cc", "PickOutputMultiplier");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 input_offset = params.input_offset;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int output_accu_min = std::numeric_limits<std::int32_t>::max();
  int output_accu_max = std::numeric_limits<std::int32_t>::min();

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32 acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32 input_val = input_data[Offset(input_shape, batch, in_y,
                                                      in_x, in_channel)];
                  int32 filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            output_accu_max = std::max(acc, output_accu_max);
            output_accu_min = std::min(acc, output_accu_min);
          }
        }
      }
    }
  }

  // Since int8 ranges from -128 to 127, we need to squeeze the accumulator
  // min/max fit in those ranges correspondingly as much as possible.
  if (std::abs(output_accu_max) > std::abs(output_accu_min)) {
    *output_multiplier = 127.0f / std::abs(output_accu_max);
  } else {
    *output_multiplier = 128.0f / std::abs(output_accu_min);
  }
}

void PickReasonableMultiplier(
    const DepthwiseParams& params, int output_activation_min,
    int output_activation_max, int output_depth,
    const RuntimeShape& input_shape_inference, const std::int8_t* input_data,
    const RuntimeShape& filter_shape_inference, const std::int8_t* filter_data,
    const RuntimeShape& bias_shape_inference, const std::int32_t* bias_data,
    const RuntimeShape& output_shape_inference,
    std::int32_t* output_multiplier_ptr, std::int32_t* output_shift_ptr,
    std::int8_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc mht_1(mht_1_v, 294, "", "./tensorflow/lite/kernels/internal/depthwiseconv_per_channel_quantized_test.cc", "PickReasonableMultiplier");

  float output_multiplier;
  PickOutputMultiplier(params, input_shape_inference, input_data,
                       filter_shape_inference, filter_data,
                       bias_shape_inference, bias_data, output_shape_inference,
                       &output_multiplier);

  int base_multiplier;
  int base_shift;
  QuantizeMultiplier(output_multiplier, &base_multiplier, &base_shift);
  for (int i = 0; i < output_depth; ++i) {
    // multipliers typically range in [2^30 ; 2^31 - 1].
    // Values in [0, 2^30 - 1] are normally unused, but harmless.
    // Thus a good way to randomize multipliers is to subtract from them
    // a random value smaller than 2^30 but still significant compared to it.
    output_multiplier_ptr[i] = base_multiplier - (std::rand() % (1 << 26));
    output_shift_ptr[i] = base_shift - 1 + (std::rand() % 4);
  }
}

// The reference implementation & the fast kernel have different rounding
// mechanism, so we loosely compare the difference.
void CompareRoundingResults(int flat_size, const int depth_multiplier,
                            const std::int8_t* reference_result,
                            const std::int8_t* fast_kernel_result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc mht_2(mht_2_v, 321, "", "./tensorflow/lite/kernels/internal/depthwiseconv_per_channel_quantized_test.cc", "CompareRoundingResults");

  std::vector<int> diff(flat_size);
  std::int64_t sum_diff = 0;
  std::int64_t sum_abs_diff = 0;
  for (int i = 0; i < flat_size; i++) {
    diff[i] = static_cast<int>(fast_kernel_result[i]) -
              static_cast<int>(reference_result[i]);
    sum_diff += diff[i];
    sum_abs_diff += std::abs(diff[i]);
  }
  // These stats help understand test failures.
  std::sort(std::begin(diff), std::end(diff));
  const int min_diff = diff.front();
  const int max_diff = diff.back();
  const int median_diff = diff[diff.size() / 2];
  const float mean_diff = static_cast<float>(sum_diff) / flat_size;
  const float mean_abs_diff = static_cast<float>(sum_abs_diff) / flat_size;

  // The tolerance that we apply to means is tight, but we allow for a rounding
  // difference in one pixel, and loosen by another 1% for float comparison.
  const float mean_tolerance =
      std::max(1e-2f, 1.01f / flat_size * std::sqrt(1.f * depth_multiplier));
  const int diff_mean_tolerance = 256;
  const int diff_median_tolerance = 225;

  // Normally we should require bit-for-bit exact results. Unfortunately a bug
  // in the Intel arm_neon_sse.h translation header that we use for x86 tests
  // causes 1-bit inaccuracy in the vqrdmulh_n_s32 intrinsic, which causes
  // off-by-1 errors in quantized DepthwiseConv ops. So we have to live with a
  // few off-by-one errors for now, yet still ensure that no more than a small
  // minority of values are wrong.
  EXPECT_LT(std::abs(mean_diff), mean_tolerance);
  EXPECT_LT(mean_abs_diff, mean_tolerance);
  EXPECT_LE(std::abs(median_diff), diff_median_tolerance);
  EXPECT_LE(std::abs(min_diff), diff_mean_tolerance);
  EXPECT_LE(std::abs(max_diff), diff_mean_tolerance);
  EXPECT_TRUE(std::abs(mean_diff) < mean_tolerance &&
              mean_abs_diff < mean_tolerance &&
              std::abs(median_diff) <= diff_median_tolerance &&
              std::abs(min_diff) <= diff_mean_tolerance &&
              std::abs(max_diff) <= diff_mean_tolerance);
}

bool GenerateValidShapeConfigurations(
    int filter_width, int filter_height, int depth_multiplier,
    int dilation_width_factor, int dilation_height_factor,
    RuntimeShape* input_shape_inference, RuntimeShape* filter_shape_inference,
    RuntimeShape* output_shape_inference, int* pad_width, int* pad_height,
    int* stride) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc mht_3(mht_3_v, 372, "", "./tensorflow/lite/kernels/internal/depthwiseconv_per_channel_quantized_test.cc", "GenerateValidShapeConfigurations");

  const int batch = UniformRandomInt(1, 3);
  const int input_depth = 8 * ExponentialRandomPositiveInt(0.9f, 10, 50);
  const int input_width = UniformRandomInt(5, 50);
  const int input_height = UniformRandomInt(5, 50);
  *stride = UniformRandomInt(1, 2);
  const bool test_pad = UniformRandomInt(0, 1);
  const auto padding_type = test_pad ? PaddingType::kValid : PaddingType::kSame;

  const int output_depth = input_depth * depth_multiplier;

  input_shape_inference->BuildFrom(
      {batch, input_height, input_width, input_depth});

  filter_shape_inference->BuildFrom(
      {1, filter_height, filter_width, output_depth});

  EXPECT_TRUE(ComputeConvSizes(
      *input_shape_inference, output_depth, filter_width, filter_height,
      *stride, dilation_width_factor, dilation_height_factor, padding_type,
      output_shape_inference, pad_width, pad_height));

  // We just care about whether the shape is suitable so we use non-per-channel
  // case.
  return optimized_ops::depthwise_conv::Fast3x3FilterKernelSupported<
      optimized_ops::depthwise_conv::QuantizationType::kNonPerChannelUint8>(
      *input_shape_inference, *filter_shape_inference, *stride, *stride,
      dilation_width_factor, dilation_height_factor, *pad_width, *pad_height,
      depth_multiplier, *output_shape_inference, 0);
}

void TryTestOneDepthwiseConv3x3Filter() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSdepthwiseconv_per_channel_quantized_testDTcc mht_4(mht_4_v, 406, "", "./tensorflow/lite/kernels/internal/depthwiseconv_per_channel_quantized_test.cc", "TryTestOneDepthwiseConv3x3Filter");

  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier = 1;
  // We don't support dilations in the 3x3 filter.
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int output_activation_min = -128;
  const int output_activation_max = 127;

  const std::int32_t input_offset = UniformRandomInt(-25, 25);
  const std::int32_t output_offset = UniformRandomInt(-25, 25);

  RuntimeShape input_shape_inference;
  RuntimeShape filter_shape_inference;
  RuntimeShape output_shape_inference;
  int pad_width, pad_height;
  int stride;

  // Keeps trying until we get valid shape/configurations for 3x3 filter case.
  bool generated_valid_configurations_for_3x3_kernel = false;
  while (!generated_valid_configurations_for_3x3_kernel) {
    generated_valid_configurations_for_3x3_kernel =
        GenerateValidShapeConfigurations(
            filter_width, filter_height, depth_multiplier,
            dilation_width_factor, dilation_height_factor,
            &input_shape_inference, &filter_shape_inference,
            &output_shape_inference, &pad_width, &pad_height, &stride);
  }

  const int output_depth = output_shape_inference.Dims(3);

  RuntimeShape bias_shape_inference({1, 1, 1, output_depth});
  const int input_buffer_size = input_shape_inference.FlatSize();
  const int filter_buffer_size = filter_shape_inference.FlatSize();
  const int output_buffer_size = output_shape_inference.FlatSize();
  std::vector<std::int8_t> input_data(input_buffer_size);
  std::vector<std::int8_t> filter_data(filter_buffer_size);
  std::vector<std::int32_t> bias_data(output_depth);

  FillRandom(&input_data);
  FillRandom(&filter_data);
  FillRandom(&bias_data, -1000, 1000);

  DepthwiseParams params;
  params.stride_width = stride;
  params.stride_height = stride;
  params.dilation_height_factor = dilation_height_factor;
  params.dilation_width_factor = dilation_width_factor;
  params.padding_values.width = pad_width;
  params.padding_values.height = pad_height;
  params.depth_multiplier = depth_multiplier;
  params.input_offset = input_offset;
  params.output_offset = output_offset;
  params.weights_offset = 0;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;

  std::vector<std::int8_t> reference_output_data(output_buffer_size);
  std::vector<std::int8_t> neon_output_data(output_buffer_size);

  std::vector<std::int32_t> output_multiplier(output_depth);
  std::vector<std::int32_t> output_shift(output_depth);

  // It's hard to come up with a right multiplier, random guess basically makes
  // all the results saturated and becomes meaningfulless, so we first use
  // reference impl to poke the min/max value of the accumulation, then use that
  // value as a guided suggestion for us to populate meaningful multiplier &
  // shift.
  PickReasonableMultiplier(
      params, output_activation_min, output_activation_max, output_depth,
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, output_multiplier.data(), output_shift.data(),
      reference_output_data.data());

  EXPECT_TRUE(optimized_ops::depthwise_conv::Fast3x3FilterKernelSupported<
              optimized_ops::depthwise_conv::QuantizationType::kPerChannelInt8>(
      input_shape_inference, filter_shape_inference, stride, stride,
      dilation_width_factor, dilation_height_factor, pad_width, pad_height,
      depth_multiplier, output_shape_inference, 0, output_shift.data()));

  // The following tests compare reference impl and Neon general impl agrees,
  // and reference impl loosely agrees with fast kernel since they use different
  // rounding strategy.
  reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier.data(), output_shift.data(),
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, reference_output_data.data());

  optimized_integer_ops::depthwise_conv::DepthwiseConvGeneral(
      params, output_multiplier.data(), output_shift.data(),
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, neon_output_data.data(),
      /*thread_start=*/0,
      /*thread_end=*/output_shape_inference.Dims(1), /*thread_dim=*/1);

  // We have changed our rounding strategy to the ARM rounding-right-shift
  // instruction: breaking tie upward as it's much simpler.
  // So we allow some difference for the neon output VS. the reference output.
  CompareRoundingResults(output_buffer_size, depth_multiplier,
                         reference_output_data.data(), neon_output_data.data());

#if defined(__aarch64__) && !defined(GOOGLE_L4T)
  std::vector<std::int8_t> fast_kernel_output_data(output_buffer_size);
  optimized_ops::depthwise_conv::DepthwiseConv3x3FilterPerChannel<
      DepthwiseConvOutputRounding::kUpward>(
      params, output_multiplier.data(), output_shift.data(),
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, fast_kernel_output_data.data(),
      /*thread_start=*/0,
      /*thread_end=*/output_shape_inference.Dims(1), /*thread_dim=*/1);

  CompareRoundingResults(output_buffer_size, depth_multiplier,
                         reference_output_data.data(),
                         fast_kernel_output_data.data());
#endif
}

TEST(QuantizedDepthwiseConvPerChannelTest, FastKernelTest) {
  for (int i = 0; i < 60; ++i) {
    TryTestOneDepthwiseConv3x3Filter();
  }
}

}  // namespace
}  // namespace tflite
