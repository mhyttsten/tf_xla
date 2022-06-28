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
class MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace pooling_3d {
namespace {

// TODO(b/175003241): If promoting this op to a builtin op, move this struct to
// lite/c/builtin_opdata.h.
struct Pool3DParams {
  TfLiteFusedActivation activation;
  TfLitePadding padding_type;
  Padding3DValues padding_values;
  int stride_depth;
  int stride_height;
  int stride_width;
  int filter_depth;
  int filter_height;
  int filter_width;
  // int8_t and int16_t activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

template <typename T, typename ActivationT>
inline T RoundAndAverage(ActivationT sum, int count) {
  // Round to the closest integer value.
  return sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
}

template <>
inline float RoundAndAverage(float sum, int count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_0(mht_0_v, 233, "", "./tensorflow/lite/kernels/pooling3d.cc", "RoundAndAverage");

  // No rounding for float type.
  return sum / count;
}

// TODO(b/175003241): If promoting this op to a builtin op, move AveragePool3D
// and MaxPool3D to a dedicated header.
template <typename T, typename ActivationT>
inline void AveragePool3D(const Pool3DParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  ActivationT activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int channels = MatchingDim(input_shape, 4, output_shape, 4);

  const int in_spatial_dim_1 = input_shape.Dims(1);
  const int in_spatial_dim_2 = input_shape.Dims(2);
  const int in_spatial_dim_3 = input_shape.Dims(3);
  const int out_spatial_dim_1 = output_shape.Dims(1);
  const int out_spatial_dim_2 = output_shape.Dims(2);
  const int out_spatial_dim_3 = output_shape.Dims(3);

  const int stride_spatial_dim_1 = params.stride_depth;
  const int stride_spatial_dim_2 = params.stride_height;
  const int stride_spatial_dim_3 = params.stride_width;
  const int filter_spatial_dim_1 = params.filter_depth;
  const int filter_spatial_dim_2 = params.filter_height;
  const int filter_spatial_dim_3 = params.filter_width;
  const int padding_spatial_dim_1 = params.padding_values.depth;
  const int padding_spatial_dim_2 = params.padding_values.height;
  const int padding_spatial_dim_3 = params.padding_values.width;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_d1 = 0; out_d1 < out_spatial_dim_1; ++out_d1) {
      const int in_d1_origin =
          (out_d1 * stride_spatial_dim_1) - padding_spatial_dim_1;
      const int filter_d1_start = std::max(0, -in_d1_origin);
      const int filter_d1_end =
          std::min(filter_spatial_dim_1, in_spatial_dim_1 - in_d1_origin);
      for (int out_d2 = 0; out_d2 < out_spatial_dim_2; ++out_d2) {
        const int in_d2_origin =
            (out_d2 * stride_spatial_dim_2) - padding_spatial_dim_2;
        const int filter_d2_start = std::max(0, -in_d2_origin);
        const int filter_d2_end =
            std::min(filter_spatial_dim_2, in_spatial_dim_2 - in_d2_origin);
        for (int out_d3 = 0; out_d3 < out_spatial_dim_3; ++out_d3) {
          const int in_d3_origin =
              (out_d3 * stride_spatial_dim_3) - padding_spatial_dim_3;
          const int filter_d3_start = std::max(0, -in_d3_origin);
          const int filter_d3_end =
              std::min(filter_spatial_dim_3, in_spatial_dim_3 - in_d3_origin);
          for (int channel = 0; channel < channels; ++channel) {
            ActivationT total = 0;
            for (int filter_d1 = filter_d1_start; filter_d1 < filter_d1_end;
                 ++filter_d1) {
              const int in_d1 = in_d1_origin + filter_d1;
              for (int filter_d2 = filter_d2_start; filter_d2 < filter_d2_end;
                   ++filter_d2) {
                const int in_d2 = in_d2_origin + filter_d2;
                for (int filter_d3 = filter_d3_start; filter_d3 < filter_d3_end;
                     ++filter_d3) {
                  const int in_d3 = in_d3_origin + filter_d3;
                  total += input_data[Offset(input_shape, batch, in_d1, in_d2,
                                             in_d3, channel)];
                }
              }
            }
            const int filter_count = (filter_d1_end - filter_d1_start) *
                                     (filter_d2_end - filter_d2_start) *
                                     (filter_d3_end - filter_d3_start);
            T average = pooling_3d::RoundAndAverage<T, ActivationT>(
                total, filter_count);
            average = std::max<T>(average, activation_min);
            average = std::min<T>(average, activation_max);
            output_data[Offset(output_shape, batch, out_d1, out_d2, out_d3,
                               channel)] = average;
          }
        }
      }
    }
  }
}

template <typename T, typename ActivationT>
inline void MaxPool3D(const Pool3DParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  ActivationT activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int channels = MatchingDim(input_shape, 4, output_shape, 4);

  const int in_spatial_dim_1 = input_shape.Dims(1);
  const int in_spatial_dim_2 = input_shape.Dims(2);
  const int in_spatial_dim_3 = input_shape.Dims(3);
  const int out_spatial_dim_1 = output_shape.Dims(1);
  const int out_spatial_dim_2 = output_shape.Dims(2);
  const int out_spatial_dim_3 = output_shape.Dims(3);

  const int stride_spatial_dim_1 = params.stride_depth;
  const int stride_spatial_dim_2 = params.stride_height;
  const int stride_spatial_dim_3 = params.stride_width;
  const int filter_spatial_dim_1 = params.filter_depth;
  const int filter_spatial_dim_2 = params.filter_height;
  const int filter_spatial_dim_3 = params.filter_width;
  const int padding_spatial_dim_1 = params.padding_values.depth;
  const int padding_spatial_dim_2 = params.padding_values.height;
  const int padding_spatial_dim_3 = params.padding_values.width;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_d1 = 0; out_d1 < out_spatial_dim_1; ++out_d1) {
      const int in_d1_origin =
          (out_d1 * stride_spatial_dim_1) - padding_spatial_dim_1;
      const int filter_d1_start = std::max(0, -in_d1_origin);
      const int filter_d1_end =
          std::min(filter_spatial_dim_1, in_spatial_dim_1 - in_d1_origin);
      for (int out_d2 = 0; out_d2 < out_spatial_dim_2; ++out_d2) {
        const int in_d2_origin =
            (out_d2 * stride_spatial_dim_2) - padding_spatial_dim_2;
        const int filter_d2_start = std::max(0, -in_d2_origin);
        const int filter_d2_end =
            std::min(filter_spatial_dim_2, in_spatial_dim_2 - in_d2_origin);
        for (int out_d3 = 0; out_d3 < out_spatial_dim_3; ++out_d3) {
          const int in_d3_origin =
              (out_d3 * stride_spatial_dim_3) - padding_spatial_dim_3;
          const int filter_d3_start = std::max(0, -in_d3_origin);
          const int filter_d3_end =
              std::min(filter_spatial_dim_3, in_spatial_dim_3 - in_d3_origin);
          for (int channel = 0; channel < channels; ++channel) {
            T max = std::numeric_limits<T>::lowest();
            for (int filter_d1 = filter_d1_start; filter_d1 < filter_d1_end;
                 ++filter_d1) {
              const int in_d1 = in_d1_origin + filter_d1;
              for (int filter_d2 = filter_d2_start; filter_d2 < filter_d2_end;
                   ++filter_d2) {
                const int in_d2 = in_d2_origin + filter_d2;
                for (int filter_d3 = filter_d3_start; filter_d3 < filter_d3_end;
                     ++filter_d3) {
                  const int in_d3 = in_d3_origin + filter_d3;
                  max =
                      std::max(max, input_data[Offset(input_shape, batch, in_d1,
                                                      in_d2, in_d3, channel)]);
                }
              }
            }
            max = std::max<T>(max, activation_min);
            max = std::min<T>(max, activation_max);
            output_data[Offset(output_shape, batch, out_d1, out_d2, out_d3,
                               channel)] = max;
          }
        }
      }
    }
  }
}
}  // namespace

enum PoolType {
  kAverage,
  kMax,
};

constexpr const char kPoolSizeStr[] = "ksize";
constexpr const char kStridesStr[] = "strides";
constexpr const char kPaddingStr[] = "padding";
constexpr const char kDataFormatStr[] = "data_format";
constexpr const char kPaddingSameStr[] = "SAME";
constexpr const char kPaddingValidStr[] = "VALID";

struct OpData {
  Pool3DParams params;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_1(mht_1_v, 417, "", "./tensorflow/lite/kernels/pooling3d.cc", "Init");

  OpData* opdata = new OpData;
  opdata->params.activation = kTfLiteActNone;

  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  const std::string data_format = m[kDataFormatStr].AsString().str();
  TFLITE_CHECK_EQ(data_format, "NDHWC");

  const std::string padding = m[kPaddingStr].AsString().str();
  if (padding == kPaddingValidStr) {
    opdata->params.padding_type = kTfLitePaddingValid;
  } else if (padding == kPaddingSameStr) {
    opdata->params.padding_type = kTfLitePaddingSame;
  } else {
    opdata->params.padding_type = kTfLitePaddingUnknown;
  }

  // The first and last element of pool_size are always 1.
  const auto pool_size = m[kPoolSizeStr].AsTypedVector();
  TFLITE_CHECK_EQ(pool_size.size(), 5);
  TFLITE_CHECK_EQ(pool_size[0].AsInt32(), 1);
  TFLITE_CHECK_EQ(pool_size[4].AsInt32(), 1);
  opdata->params.filter_depth = pool_size[1].AsInt32();
  opdata->params.filter_height = pool_size[2].AsInt32();
  opdata->params.filter_width = pool_size[3].AsInt32();

  // The first and last element of strides are always 1.
  const auto strides = m[kStridesStr].AsTypedVector();
  TFLITE_CHECK_EQ(strides.size(), 5);
  TFLITE_CHECK_EQ(strides[0].AsInt32(), 1);
  TFLITE_CHECK_EQ(strides[4].AsInt32(), 1);
  opdata->params.stride_depth = strides[1].AsInt32();
  opdata->params.stride_height = strides[2].AsInt32();
  opdata->params.stride_width = strides[3].AsInt32();
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_2(mht_2_v, 459, "", "./tensorflow/lite/kernels/pooling3d.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_3(mht_3_v, 466, "", "./tensorflow/lite/kernels/pooling3d.cc", "GenericPrepare");

  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  Pool3DParams& params = opdata->params;

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 5);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context,
                    input->type == kTfLiteFloat32 ||
                        input->type == kTfLiteInt16 ||
                        input->type == kTfLiteInt8,
                    true);

  int batches = input->dims->data[0];
  int depth = input->dims->data[1];
  int height = input->dims->data[2];
  int width = input->dims->data[3];
  int channels = input->dims->data[4];

  // Prevent division by 0 in optimized pooling implementations
  TF_LITE_ENSURE(context, params.stride_depth > 0);
  TF_LITE_ENSURE(context, params.stride_height > 0);
  TF_LITE_ENSURE(context, params.stride_width > 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height, out_depth;
  params.padding_values = ComputePadding3DValues(
      params.stride_height, params.stride_width, params.stride_depth, 1, 1, 1,
      height, width, depth, params.filter_height, params.filter_width,
      params.filter_depth, params.padding_type, &out_height, &out_width,
      &out_depth);

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_NEAR(context, input->params.scale, output->params.scale,
                        1.0e-6);
    TFLITE_DCHECK_EQ(input->params.zero_point, output->params.zero_point);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(5);
  output_size->data[0] = batches;
  output_size->data[1] = out_depth;
  output_size->data[2] = out_height;
  output_size->data[3] = out_width;
  output_size->data[4] = channels;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_4(mht_4_v, 521, "", "./tensorflow/lite/kernels/pooling3d.cc", "AverageEval");

  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  Pool3DParams& params = opdata->params;

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

#define TF_LITE_AVERAGE_POOL_3D(type, activation_type)           \
  SetActivationParams(activation_min, activation_max, &params);  \
  AveragePool3D<type, activation_type>(                          \
      params, GetTensorShape(input), GetTensorData<type>(input), \
      GetTensorShape(output), GetTensorData<type>(output))

  switch (input->type) {
    case kTfLiteFloat32: {
      float activation_min, activation_max;
      CalculateActivationRange(params.activation, &activation_min,
                               &activation_max);
      TF_LITE_AVERAGE_POOL_3D(float, float);
    } break;
    case kTfLiteInt8: {
      int32_t activation_min;
      int32_t activation_max;
      CalculateActivationRangeQuantized(context, params.activation, output,
                                        &activation_min, &activation_max);
      TF_LITE_AVERAGE_POOL_3D(int8_t, int32_t);
    } break;
    case kTfLiteInt16: {
      int32_t activation_min;
      int32_t activation_max;
      CalculateActivationRangeQuantized(context, params.activation, output,
                                        &activation_min, &activation_max);
      TF_LITE_AVERAGE_POOL_3D(int16_t, int32_t);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
#undef TF_LITE_AVERAGE_POOL_3D
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_5(mht_5_v, 569, "", "./tensorflow/lite/kernels/pooling3d.cc", "MaxEval");

  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  Pool3DParams& params = opdata->params;

#define TF_LITE_MAX_POOL_3D(type, activation_type)               \
  SetActivationParams(activation_min, activation_max, &params);  \
  MaxPool3D<type, activation_type>(                              \
      params, GetTensorShape(input), GetTensorData<type>(input), \
      GetTensorShape(output), GetTensorData<type>(output))

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

  switch (input->type) {
    case kTfLiteFloat32: {
      float activation_min, activation_max;
      CalculateActivationRange(params.activation, &activation_min,
                               &activation_max);
      TF_LITE_MAX_POOL_3D(float, float);
    } break;
    case kTfLiteInt8: {
      int32_t activation_min;
      int32_t activation_max;
      CalculateActivationRangeQuantized(context, params.activation, output,
                                        &activation_min, &activation_max);
      TF_LITE_MAX_POOL_3D(int8_t, int32_t);
    } break;
    case kTfLiteInt16: {
      int32_t activation_min;
      int32_t activation_max;
      CalculateActivationRangeQuantized(context, params.activation, output,
                                        &activation_min, &activation_max);
      TF_LITE_MAX_POOL_3D(int16_t, int32_t);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
#undef TF_LITE_MAX_POOL_3D
  return kTfLiteOk;
}

}  // namespace pooling_3d

TfLiteRegistration* Register_AVG_POOL_3D() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_6(mht_6_v, 619, "", "./tensorflow/lite/kernels/pooling3d.cc", "Register_AVG_POOL_3D");

  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::GenericPrepare,
                                 pooling_3d::AverageEval};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_3D() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3dDTcc mht_7(mht_7_v, 629, "", "./tensorflow/lite/kernels/pooling3d.cc", "Register_MAX_POOL_3D");

  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::GenericPrepare,
                                 pooling_3d::MaxEval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
