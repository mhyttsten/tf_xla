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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"

namespace tflite {
namespace optimize {
namespace utils {

namespace {
const int8_t kMinQuantizedValue = -127;
const int8_t kMaxQuantizedValue = 127;

// The maximum number of dimensions supported in per-channel quantization.
constexpr int kPerChannelMaxDim = 4;
}  // namespace

TfLiteStatus NumElements(const TensorT& tensor, uint64_t* num_elements) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_0(mht_0_v, 217, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "NumElements");

  *num_elements = 1;
  for (const int64_t dim : tensor.shape) {
    if (dim <= 0 || *num_elements > UINT64_MAX / static_cast<uint64_t>(dim)) {
      return kTfLiteError;
    }
    *num_elements *= dim;
  }
  return kTfLiteOk;
}

// Nudge min and max so that floating point 0 falls exactly on a quantized
// value, returning the nudges scale and zero_point.
//
// Although this code originates from FakeQuantization in quantized training,
// we may deviate from that implementation as we please since we do not fine
// tune the weights with quantized training.
void GetAsymmetricQuantizationParams(
    float min, float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetAsymmetricQuantizationParams");

  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  // Adjust the boundaries to guarantee 0 is included.
  min = std::min(static_cast<float>(min), 0.0f);
  max = std::max(static_cast<float>(max), 0.0f);
  const float scale = (max - min) / (quant_max_float - quant_min_float);
  // Scale can be zero if min and max are exactly 0.0f.
  float zero_point_from_min = quant_min_float;
  if (scale != 0) {
    zero_point_from_min = quant_min_float - min / scale;
  }
  int64_t zero_point;
  if (zero_point_from_min < quant_min_float) {
    zero_point = static_cast<int64_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    zero_point = static_cast<int64_t>(quant_max);
  } else {
    zero_point = static_cast<int64_t>(std::round(zero_point_from_min));
  }
  quantization_params->min = std::vector<float>(1, min);
  quantization_params->max = std::vector<float>(1, max);
  quantization_params->scale = std::vector<float>(1, scale);
  quantization_params->zero_point = std::vector<int64_t>(1, zero_point);
}

void GetSymmetricQuantizationParams(
    float min, float max, const int half_quant_range,
    QuantizationParametersT* quantization_params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_2(mht_2_v, 270, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetSymmetricQuantizationParams");

  // Adjust the boundaries to guarantee 0 is included.
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);
  const float scale = std::max(std::abs(max), std::abs(min)) / half_quant_range;
  quantization_params->min = std::vector<float>(1, min);
  quantization_params->max = std::vector<float>(1, max);
  quantization_params->scale = std::vector<float>(1, scale);
  quantization_params->zero_point = std::vector<int64_t>(1, 0);
}

TfLiteStatus GetQuantizationParams(TensorT* tensor, TensorType activations_type,
                                   QuantizationParametersT* quantization_params,
                                   ErrorReporter* error_reporter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_3(mht_3_v, 286, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetQuantizationParams");

  if (activations_type == TensorType_INT8) {
    GetAsymmetricQuantizationParams(
        tensor->quantization->min[0], tensor->quantization->max[0],
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
        quantization_params);
  } else if (activations_type == TensorType_INT16) {
    const int half_quantized_range = 32767;
    GetSymmetricQuantizationParams(tensor->quantization->min[0],
                                   tensor->quantization->max[0],
                                   half_quantized_range, quantization_params);
  } else {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Unsupported activation type for quantize-activation: %d",
        activations_type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

// Set the max and min quantization parameter for a single tensor given its
// values.
void FillSingleMinMax(const float* const input, const uint64_t input_size,
                      QuantizationParametersT* quantization_params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_4(mht_4_v, 313, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "FillSingleMinMax");

  const auto minmax = std::minmax_element(input, input + input_size);
  quantization_params->min.assign(1, *minmax.first);
  quantization_params->max.assign(1, *minmax.second);
}

TfLiteStatus FillPerChannelMinMax(const float* const input,
                                  const std::vector<int32_t>& dimension,
                                  int32_t channel_dim_index,
                                  QuantizationParametersT* quantization_params,
                                  ErrorReporter* error_reporter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_5(mht_5_v, 326, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "FillPerChannelMinMax");

  if (!quantization_params->min.empty() || !quantization_params->max.empty()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Min or max already present in tensor quantization params.");
    return kTfLiteError;
  }

  if (dimension.size() > kPerChannelMaxDim) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Expected tensor with less than %d dimensions, but got %d.",
        kPerChannelMaxDim + 1, dimension.size());
    return kTfLiteError;
  }
  if (channel_dim_index >= dimension.size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Expected channel_dim_index to be less than %d, but got %d.",
        dimension.size(), channel_dim_index);
    return kTfLiteError;
  }

  const int32_t channel_dim_size = dimension[channel_dim_index];
  quantization_params->quantized_dimension = channel_dim_index;
  quantization_params->min = std::vector<float>(channel_dim_size);
  quantization_params->max = std::vector<float>(channel_dim_size);
  std::vector<bool> has_min_max_value(channel_dim_size, false);
  int indices[kPerChannelMaxDim];
  RuntimeShape unextended_tensor_dims(dimension.size(), dimension.data());
  RuntimeShape tensor_dims =
      RuntimeShape::ExtendedShape(kPerChannelMaxDim, unextended_tensor_dims);
  channel_dim_index +=
      kPerChannelMaxDim - unextended_tensor_dims.DimensionsCount();

  // Compute min max ranges per channel
  for (indices[0] = 0; indices[0] < tensor_dims.Dims(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor_dims.Dims(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor_dims.Dims(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor_dims.Dims(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          const float val = input[Offset(tensor_dims, indices)];
          if (has_min_max_value[channel_idx]) {
            if (quantization_params->min[channel_idx] > val) {
              quantization_params->min[channel_idx] = val;
            } else if (quantization_params->max[channel_idx] < val) {
              quantization_params->max[channel_idx] = val;
            }
          } else {
            quantization_params->min[channel_idx] = val;
            quantization_params->max[channel_idx] = val;
            has_min_max_value[channel_idx] = true;
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

// Populates the scales vector based on max and min values of quant_params
TfLiteStatus GetSymmetricScalesFromMaxMin(QuantizationParametersT* quant_params,
                                          std::vector<float>* scales,
                                          ErrorReporter* error_reporter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_6(mht_6_v, 392, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetSymmetricScalesFromMaxMin");

  // Check that max and min values are present and their sizes match.
  if (quant_params->min.empty() || quant_params->max.empty()) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Max and min values are not populated.");
    return kTfLiteError;
  }
  if (quant_params->min.size() != quant_params->max.size()) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Dimensions of max and min values do not match.");
    return kTfLiteError;
  }
  if (scales->size() != quant_params->min.size()) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Provided scale vector has incorrect size.");
    return kTfLiteError;
  }

  // num_channels is calculated from min.size() to infer whether quantization
  // is per axis.
  int num_channels = quant_params->min.size();
  // Calculate scales per channel.
  for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
    const float half_range = std::max(std::abs(quant_params->min[channel_idx]),
                                      std::abs(quant_params->max[channel_idx]));
    scales->at(channel_idx) = half_range / kMaxQuantizedValue;
  }
  return kTfLiteOk;
}

// Checks that the bias is quantized to within the middle half of the
// allowable bit range determined by the scales of the input and weight tensors.
// If this condition is not satisfied, the scale of the weights is increased in
// order to prevent overflow. The scale of the bias is not set here, only the
// min/max.
// The quant_params are the quantization parameters that correspond to the
// weight tensor.
TfLiteStatus AdjustWeightsForBiasScale(QuantizationParametersT* quant_params,
                                       const float* bias_data,
                                       const size_t bias_size,
                                       const float input_scale,
                                       ErrorReporter* error_reporter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_7(mht_7_v, 436, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "AdjustWeightsForBiasScale");

  // TODO(dmolitor) Allow adjusting activation scale.
  // TODO(dmolitor) Tighten scale adjustment.
  // TODO(dmolitor) Test using a separate strategy for scales of 0.
  const int32_t kScale = std::numeric_limits<int32_t>::max();
  if (quant_params == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Missing max and min values for weight tensor.");
    return kTfLiteError;
  }
  // channel_dim_size is calculated from min.size() to infer whether
  // quantization is per axis
  int channel_dim_size = quant_params->min.size();
  if (channel_dim_size == 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Missing weight scales. Unable to check compatibility with bias "
        "scale.");
    return kTfLiteError;
  }

  std::vector<float> weight_scales(channel_dim_size);
  TF_LITE_ENSURE_STATUS(GetSymmetricScalesFromMaxMin(
      quant_params, &weight_scales, error_reporter));

  // Per channel quantization
  if (channel_dim_size > 1) {
    for (int i = 0; i < channel_dim_size; ++i) {
      // Current scale is not compatible with bias. Adjust max/min values.
      if (std::abs(bias_data[i]) >=
          0.5 * input_scale * weight_scales[i] * kScale) {
        quant_params->max[i] = 2.0 * std::abs(bias_data[i]) / kScale *
                               (kMaxQuantizedValue / input_scale);
        quant_params->min[i] = -quant_params->max[i];
      }
    }
    // Per layer quantization
  } else if (channel_dim_size == 1) {
    const auto minmax = std::minmax_element(bias_data, bias_data + bias_size);
    const float bias_half_range =
        std::max(std::abs(*minmax.first), std::abs(*minmax.second));

    // Need to adjust weight min/max; not compatible with bias.
    if (bias_half_range / kScale >= 0.5 * input_scale * weight_scales[0]) {
      quant_params->min[0] =
          2.0 * bias_half_range / kScale * (kMinQuantizedValue / input_scale);
      quant_params->max[0] =
          2.0 * bias_half_range / kScale * (kMaxQuantizedValue / input_scale);
    }
  }
  return kTfLiteOk;
}

// Per-channel quantize a tensor at the given index and fills both scales and
// quantized values.
TfLiteStatus SymmetricPerChannelQuantization(TensorT* tensor,
                                             const float* const input,
                                             int32_t channel_dim_index,
                                             std::vector<float>* output_scales,
                                             std::vector<int8_t>* output_value,
                                             ErrorReporter* error_reporter) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_8(mht_8_v, 499, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricPerChannelQuantization");

  if (tensor == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter, "Cannot quantize. Tensor is null.");
    return kTfLiteError;
  }
  const int32_t channel_dim_size = tensor->shape[channel_dim_index];
  // Fill per channel max and min values if needed
  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  if (!HasMinMax(tensor)) {
    TF_LITE_ENSURE_STATUS(
        FillPerChannelMinMax(input, tensor->shape, channel_dim_index,
                             tensor->quantization.get(), error_reporter));
  }

  // Calculate scales per channel using max and min values from tensor.
  std::vector<float> scale_invs(channel_dim_size);
  const float half_scale = kMaxQuantizedValue;
  for (int channel_idx = 0; channel_idx < channel_dim_size; channel_idx++) {
    const float half_range =
        std::max(std::abs(tensor->quantization->min[channel_idx]),
                 std::abs(tensor->quantization->max[channel_idx]));
    output_scales->at(channel_idx) = half_range / half_scale;
    if (half_range == 0) {
      scale_invs[channel_idx] = 0;
    } else {
      scale_invs[channel_idx] = half_scale / half_range;
    }
  }

  // Quantize the input values.
  SymmetricPerChannelQuantizeValues(input, scale_invs, tensor->shape,
                                    channel_dim_index, output_value);
  return kTfLiteOk;
}

std::vector<int16_t> SymmetricQuantizeFloatsToInt16(const float* data,
                                                    uint64_t num_elements,
                                                    float scaling_factor) {
  // Compute the inverse of scale.
  const float scaling_factor_inv =
      (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
  std::vector<int16_t> buffer(num_elements);
  const int32_t kScale = std::numeric_limits<int16_t>::max();

  for (size_t i = 0; i < num_elements; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(data[i] * scaling_factor_inv));
    buffer[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

TfLiteStatus SymmetricQuantizeFloatsToInt16(ModelT* model, TensorT* tensor,
                                            float scaling_factor,
                                            ErrorReporter* error_reporter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_9(mht_9_v, 558, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricQuantizeFloatsToInt16");

  const BufferT* buffer = model->buffers[tensor->buffer].get();
  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  auto final_buffer =
      SymmetricQuantizeFloatsToInt16(float_data, num_elements, scaling_factor);
  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  size_t buffer_size = num_elements * sizeof(int16_t);
  std::vector<float> scales(1, scaling_factor);
  std::vector<int64_t> zero_points(1, 0);
  return AddQuantizationParams(scales, zero_points, 0, uint8_buffer,
                               buffer_size, TensorType_INT16, model, tensor,
                               error_reporter);
}

void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_10(mht_10_v, 583, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricPerChannelQuantizeValues");

  // Quantize the values.
  int indices[kPerChannelMaxDim];
  RuntimeShape unextended_tensor_dims(dimension.size(), dimension.data());
  RuntimeShape tensor_dims =
      RuntimeShape::ExtendedShape(kPerChannelMaxDim, unextended_tensor_dims);
  channel_dim_index +=
      kPerChannelMaxDim - unextended_tensor_dims.DimensionsCount();
  for (indices[0] = 0; indices[0] < tensor_dims.Dims(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor_dims.Dims(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor_dims.Dims(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor_dims.Dims(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = Offset(tensor_dims, indices);
          const float val = input[index];
          const int32_t quantized_value =
              static_cast<int32_t>(TfLiteRound(val * scales_inv[channel_idx]));
          output_value->at(index) = std::min<int8_t>(
              kMaxQuantizedValue,
              std::max<int8_t>(kMinQuantizedValue, quantized_value));
        }
      }
    }
  }
}

// Quantize the tensor using the max and min values recorded in its quantization
// parameters. Applies per-layer quantization.
TfLiteStatus SymmetricQuantizeTensorFromMinMax(ModelT* model, TensorT* tensor,
                                               ErrorReporter* error_reporter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_11(mht_11_v, 615, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricQuantizeTensorFromMinMax");

  if (model == nullptr || tensor == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter, "No tensor to quantize.");
    return kTfLiteError;
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  if (buffer == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter, "Missing buffer.");
    return kTfLiteError;
  }

  if (!HasMinMax(tensor)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Missing min or max values for quantization.");
    return kTfLiteError;
  }
  if (tensor->quantization->min.size() != 1 ||
      tensor->quantization->max.size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Expected single entry in max and min.");
    return kTfLiteError;
  }

  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  std::vector<int8_t> quantized_buffer;
  quantized_buffer.resize(num_elements);

  // Quantize tensor using recorded min and max values
  float scaling_factor;
  tensor_utils::SymmetricQuantizeFloats(
      float_data, num_elements, quantized_buffer.data(),
      tensor->quantization->min[0], tensor->quantization->max[0],
      &scaling_factor);
  tensor->quantization->scale = std::vector<float>(1, scaling_factor);
  tensor->quantization->zero_point = std::vector<int64_t>(1, 0);

  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(uint8_buffer,
                                              uint8_buffer + num_elements);
  // Update the tensor type.
  tensor->type = TensorType_INT8;

  return kTfLiteOk;
}

TfLiteStatus SymmetricQuantizeTensor(ModelT* model, TensorT* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_12(mht_12_v, 667, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricQuantizeTensor");

  if (model == nullptr || tensor == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR, "No tensor to quantize.");
    return kTfLiteError;
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  if (buffer == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR, "Missing buffer.");
    return kTfLiteError;
  }
  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  std::vector<int8_t> quantized_buffer;
  quantized_buffer.resize(num_elements);

  float min_value, max_value, scaling_factor;
  tensor_utils::SymmetricQuantizeFloats(float_data, num_elements,
                                        quantized_buffer.data(), &min_value,
                                        &max_value, &scaling_factor);

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  tensor->quantization->scale = std::vector<float>(1, scaling_factor);
  tensor->quantization->zero_point = std::vector<int64_t>(1, 0);

  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(uint8_buffer,
                                              uint8_buffer + num_elements);

  // Update the tensor type.
  tensor->type = TensorType_INT8;

  return kTfLiteOk;
}

TfLiteStatus QuantizeTensorFloat16(ModelT* model, TensorT* tensor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_13(mht_13_v, 709, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "QuantizeTensorFloat16");

  if (model == nullptr || tensor == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR, "No tensor to quantize.");
    return kTfLiteError;
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  if (buffer == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR, "Missing buffer.");
    return kTfLiteError;
  }

  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  // Copy single byte buffer data to float vector to guard against misalignment.
  std::vector<float> float_vector(num_elements);
  uint8_t* first = buffer->data.data();
  std::copy(first, first + buffer->data.size(),
            reinterpret_cast<uint8_t*>(float_vector.data()));

  // Transform float data to float16.
  std::vector<Eigen::half> quantized_buffer;
  quantized_buffer.resize(num_elements);
  constexpr float kMaxFloat16Value = 65504.f;
  constexpr float kMinFloat16Value = -65504.f;
  std::transform(float_vector.begin(), float_vector.end(),
                 quantized_buffer.begin(), [=](float a) {
                   float clamped = std::min(std::max(a, kMinFloat16Value),
                                            kMaxFloat16Value);
                   return static_cast<Eigen::half>(clamped);
                 });

  char* half_buffer = reinterpret_cast<char*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(
      half_buffer, half_buffer + sizeof(Eigen::half) * num_elements);

  // Update the tensor type.
  tensor->type = TensorType_FLOAT16;

  return kTfLiteOk;
}

TfLiteStatus AddQuantizationParams(const std::vector<float>& scales,
                                   const std::vector<int64_t>& zero_point,
                                   int quantized_dimension,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor,
                                   ErrorReporter* error_reporter) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_14(mht_14_v, 761, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "AddQuantizationParams");

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  tensor->quantization->scale.assign(scales.begin(), scales.end());
  if (zero_point.size() != scales.size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Received zero_point of size %d and scales of size %d. "
        "These sizes should match.",
        zero_point.size(), scales.size());
    return kTfLiteError;
  }
  tensor->quantization->zero_point.assign(zero_point.begin(), zero_point.end());
  tensor->quantization->quantized_dimension = quantized_dimension;
  model->buffers[tensor->buffer]->data.assign(buffer_data,
                                              buffer_data + buffer_size);
  // Update the tensor type.
  tensor->type = output_type;
  return kTfLiteOk;
}

TfLiteStatus SymmetricQuantizeTensorPerChannel(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index,
                                               ErrorReporter* error_reporter) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_15(mht_15_v, 788, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricQuantizeTensorPerChannel");

  if (tensor->shape.size() > kPerChannelMaxDim) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "SymmetricQuantizeTensorPerChannel requires tensor with less than %d "
        "dimensions, but got %d dimension(s).",
        kPerChannelMaxDim + 1, tensor->shape.size());
    return kTfLiteError;
  }

  // Get dimensions.
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));
  const int32_t channel_dim_size = tensor->shape[channel_dim_index];

  // Get input float data.
  const BufferT* buffer = model->buffers[tensor->buffer].get();
  const float* float_input_data =
      reinterpret_cast<const float*>(buffer->data.data());

  // Create container for output scale and output data.
  std::vector<float> scales(channel_dim_size);
  std::vector<int8_t> final_buffer(num_elements);

  // Quantize the input data with respect to channel_dim_index.
  TF_LITE_ENSURE_STATUS(SymmetricPerChannelQuantization(
      tensor, float_input_data, channel_dim_index, &scales, &final_buffer,
      error_reporter));

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  const size_t buffer_size = num_elements * sizeof(int8_t);
  std::vector<int64_t> zero_point(scales.size(), 0);
  return AddQuantizationParams(scales, zero_point, channel_dim_index,
                               uint8_buffer, buffer_size, TensorType_INT8,
                               model, tensor, error_reporter);
}

template <class BiasType>
std::vector<BiasType> SymmetricBiasQuantize(const float* data,
                                            uint64_t num_elements,
                                            const std::vector<float>& scales) {
  std::vector<BiasType> buffer(num_elements);
  const BiasType kScale = std::numeric_limits<BiasType>::max();
  float scaling_factor_inv_per_layer = (scales[0] == 0) ? 0 : 1.0 / scales[0];

  for (int32_t idx = 0; idx < num_elements; idx++) {
    float scaling_factor_inv =
        scales.size() == 1 ? scaling_factor_inv_per_layer
                           : ((scales[idx] == 0) ? 0 : 1.0 / scales[idx]);
    const BiasType quantized_value =
        tflite::SafeCast<BiasType>(TfLiteRound(data[idx] * scaling_factor_inv));
    buffer[idx] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

template std::vector<std::int32_t> SymmetricBiasQuantize<std::int32_t>(
    const float* data, uint64_t num_elements, const std::vector<float>& scales);

template <class BiasType>
TfLiteStatus SymmetricPerLayerBiasQuantize(ModelT* model, TensorT* tensor,
                                           float scaling_factor,
                                           ErrorReporter* error_reporter) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_16(mht_16_v, 854, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricPerLayerBiasQuantize");

  const BufferT* buffer = model->buffers[tensor->buffer].get();
  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  auto final_buffer = SymmetricBiasQuantize<BiasType>(float_data, num_elements,
                                                      {scaling_factor});

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  size_t buffer_size = num_elements * sizeof(BiasType);
  std::vector<float> scales(1, scaling_factor);
  std::vector<int64_t> zero_points(1, 0);

  auto output_type = std::is_same<BiasType, std::int32_t>::value
                         ? TensorType_INT32
                         : TensorType_INT64;
  return AddQuantizationParams(scales, zero_points, 0, uint8_buffer,
                               buffer_size, output_type, model, tensor,
                               error_reporter);
}

template TfLiteStatus SymmetricPerLayerBiasQuantize<std::int32_t>(
    ModelT* model, TensorT* tensor, float scaling_factor,
    ErrorReporter* error_reporter);

template TfLiteStatus SymmetricPerLayerBiasQuantize<std::int64_t>(
    ModelT* model, TensorT* tensor, float scaling_factor,
    ErrorReporter* error_reporter);

template <class BiasType>
TfLiteStatus SymmetricPerChannelBiasQuantize(ModelT* model, TensorT* tensor,
                                             float input_scale,
                                             const float* weight_scales,
                                             int number_of_dimension,
                                             ErrorReporter* error_reporter) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_17(mht_17_v, 893, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "SymmetricPerChannelBiasQuantize");

  // Compute scales.
  std::vector<float> scales(number_of_dimension);
  for (int i = 0; i < number_of_dimension; i++) {
    scales[i] = input_scale * weight_scales[i];
  }

  const BufferT* buffer = model->buffers[tensor->buffer].get();
  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(NumElements(*tensor, &num_elements));

  auto final_buffer =
      SymmetricBiasQuantize<BiasType>(float_data, num_elements, scales);

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  size_t buffer_size = num_elements * sizeof(BiasType);
  std::vector<int64_t> zero_point(scales.size(), 0);

  auto output_type = std::is_same<BiasType, std::int32_t>::value
                         ? TensorType_INT32
                         : TensorType_INT64;
  return AddQuantizationParams(scales, zero_point, 0, uint8_buffer, buffer_size,
                               output_type, model, tensor, error_reporter);
}

template TfLiteStatus SymmetricPerChannelBiasQuantize<std::int64_t>(
    ModelT* model, TensorT* tensor, float input_scale,
    const float* weight_scales, int number_of_dimension,
    ErrorReporter* error_reporter);

template TfLiteStatus SymmetricPerChannelBiasQuantize<std::int32_t>(
    ModelT* model, TensorT* tensor, float input_scale,
    const float* weight_scales, int number_of_dimension,
    ErrorReporter* error_reporter);

TfLiteStatus QuantizeWeight(ModelT* model, TensorT* tensor, bool per_channel,
                            int per_axis_index, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_18(mht_18_v, 934, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "QuantizeWeight");

  // TODO(suharshs): Currently we conflate quantizing weights and constants. Its
  // possible that the right thing to do is asymmetric quantize the weight. Add
  // support for this.
  if (per_channel) {
    return SymmetricQuantizeTensorPerChannel(model, tensor, per_axis_index,
                                             error_reporter);
  } else if (HasMinMax(tensor) && (tensor->quantization->min.size() == 1) &&
             (tensor->quantization->max.size() == 1)) {
    // Quantize using recorded min/max values if per-tensor.
    return SymmetricQuantizeTensorFromMinMax(model, tensor, error_reporter);
  } else {
    // Quantize using min/max from buffer.
    return SymmetricQuantizeTensor(model, tensor);
  }
}

float GetEffectiveScale(ModelT* model, SubGraphT* subgraph, int op_idx,
                        std::vector<int> input_index,
                        std::vector<int> intermediate_index,
                        std::vector<float> factors) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_19(mht_19_v, 957, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetEffectiveScale");

  float scale = 1.0f;
  OperatorT* op = subgraph->operators[op_idx].get();
  for (int i = 0, end = input_index.size(); i < end; ++i) {
    const int index_local = input_index[i];
    const int index_global = op->inputs[index_local];
    const TensorT* tensor = subgraph->tensors[index_global].get();
    scale *= tensor->quantization->scale[0];
  }
  for (int i = 0, end = intermediate_index.size(); i < end; ++i) {
    const int index_local = intermediate_index[i];
    const int index_global = op->intermediates[index_local];
    const TensorT* tensor = subgraph->tensors[index_global].get();
    scale *= tensor->quantization->scale[0];
  }
  for (int i = 0, end = factors.size(); i < end; ++i) {
    scale *= factors[i];
  }
  return scale;
}

TfLiteStatus QuantizeActivation(TensorT* tensor, TensorType activations_type,
                                ErrorReporter* error_reporter) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_20(mht_20_v, 982, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "QuantizeActivation");

  TF_LITE_ENSURE_STATUS(GetQuantizationParams(
      tensor, activations_type, tensor->quantization.get(), error_reporter));
  tensor->type = activations_type;
  return kTfLiteOk;
}

TfLiteStatus QuantizeActivationToInt16(TensorT* tensor, float scale) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_21(mht_21_v, 992, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "QuantizeActivationToInt16");

  const int32_t zero_point = 0;
  tensor->quantization = absl::make_unique<QuantizationParametersT>();
  tensor->quantization->scale.push_back(scale);
  tensor->quantization->zero_point.push_back(zero_point);
  tensor->type = TensorType_INT16;
  return kTfLiteOk;
}

int GetPowerOfTwoScale(float min, float max) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_utilsDTcc mht_22(mht_22_v, 1004, "", "./tensorflow/lite/tools/optimize/quantization_utils.cc", "GetPowerOfTwoScale");

  const float range = std::max(std::abs(min), std::abs(max));
  int pot = 0;
  for (int i = 0; i < 10; i++) {
    // NOTE: use std::pow() for bitwise accuracy.
    if (std::pow(2, pot) < range) {  // NOLINT
      pot++;
    }
  }
  return pot;
}

}  // namespace utils
}  // namespace optimize
}  // namespace tflite
