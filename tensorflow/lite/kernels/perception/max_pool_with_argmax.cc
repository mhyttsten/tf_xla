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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc() {
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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace max_pool_with_argmax {
namespace {
// TODO(b/175003241): Move this logic to lite/kernels/internal when promoting
// this op to a builtin op.
template <typename T>
inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const RuntimeShape& output_shape, const T* input_data,
                    T* output_data, int32_t* indices_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "MaxPool");

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t output_height = output_shape.Dims(1);
  const int32_t output_width = output_shape.Dims(2);
  const int32_t stride_height = params.stride_height;
  const int32_t stride_width = params.stride_width;
  for (int32_t batch = 0; batch < batches; ++batch) {
    for (int32_t out_y = 0; out_y < output_height; ++out_y) {
      for (int32_t out_x = 0; out_x < output_width; ++out_x) {
        for (int32_t channel = 0; channel < depth; ++channel) {
          const int32_t in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int32_t in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int32_t filter_x_start = std::max(0, -in_x_origin);
          const int32_t filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int32_t filter_y_start = std::max(0, -in_y_origin);
          const int32_t filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          int32_t max_x = 0;
          int32_t max_y = 0;

          for (int32_t filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int32_t filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int32_t in_x = in_x_origin + filter_x;
              const int32_t in_y = in_y_origin + filter_y;
              float cur =
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              if (cur > max) {
                max = cur;
                max_x = in_x;
                max_y = in_y;
              }
            }
          }
          int32_t output_idx =
              Offset(output_shape, batch, out_y, out_x, channel);
          output_data[output_idx] = ActivationFunctionWithMinMax(
              max, params.float_activation_min, params.float_activation_max);
          indices_data[output_idx] =
              (max_y * input_width + max_x) * depth + channel;
        }
      }
    }
  }
}

}  // namespace

constexpr int kDataInputTensor = 0;
constexpr int kDataOutputTensor = 0;
constexpr int kIndicesOutputTensor = 1;

constexpr const char kIncludeBatchStr[] = "include_batch_in_index";
constexpr const char kPoolSizeStr[] = "ksize";
constexpr const char kStridesStr[] = "strides";
constexpr const char kPaddingStr[] = "padding";
constexpr const char kPaddingSameStr[] = "SAME";
constexpr const char kPaddingValidStr[] = "VALID";

struct OpData {
  TfLitePoolParams params;
  bool include_batch_in_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_1(mht_1_v, 285, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "Init");

  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();

  OpData* op_data = new OpData;
  op_data->params.computed.padding = TfLitePaddingValues{0, 0, 0, 0};
  op_data->include_batch_in_index = m[kIncludeBatchStr].AsBool();
  op_data->params.activation = kTfLiteActNone;

  const std::string padding = m[kPaddingStr].AsString().str();
  if (padding == kPaddingValidStr) {
    op_data->params.padding = kTfLitePaddingValid;
  } else if (padding == kPaddingSameStr) {
    op_data->params.padding = kTfLitePaddingSame;
  } else {
    op_data->params.padding = kTfLitePaddingUnknown;
  }

  // The first and last element of pool_size are always 1.
  const auto pool_size = m[kPoolSizeStr].AsTypedVector();
  TFLITE_CHECK_EQ(pool_size.size(), 4);
  TFLITE_CHECK_EQ(pool_size[0].AsInt32(), 1);
  TFLITE_CHECK_EQ(pool_size[3].AsInt32(), 1);
  op_data->params.filter_height = pool_size[1].AsInt32();
  op_data->params.filter_width = pool_size[2].AsInt32();

  // The first and last element of strides are always 1.
  const auto strides = m[kStridesStr].AsTypedVector();
  TFLITE_CHECK_EQ(strides.size(), 4);
  TFLITE_CHECK_EQ(strides[0].AsInt32(), 1);
  TFLITE_CHECK_EQ(strides[3].AsInt32(), 1);
  op_data->params.stride_height = strides[1].AsInt32();
  op_data->params.stride_width = strides[2].AsInt32();

  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_2(mht_2_v, 326, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_3(mht_3_v, 333, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "Prepare");

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  TfLiteTensor *output, *indices;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kDataOutputTensor, &output));
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kIndicesOutputTensor, &indices));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kDataInputTensor, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, indices->type == kTfLiteInt32);
  TF_LITE_ENSURE(context, op_data->params.padding != kTfLitePaddingUnknown);
  TF_LITE_ENSURE_MSG(
      context, !op_data->include_batch_in_index,
      "Include batch dimension in flattened index is not yet supported.");

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height;
  op_data->params.computed.padding = ComputePaddingHeightWidth(
      op_data->params.stride_height, op_data->params.stride_width, 1, 1, height,
      width, op_data->params.filter_height, op_data->params.filter_width,
      op_data->params.padding, &out_height, &out_width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  TfLiteIntArray* indices_size = TfLiteIntArrayCopy(output_size);

  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, indices, indices_size));
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_4(mht_4_v, 381, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "Eval");

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  float activation_min, activation_max;
  CalculateActivationRange(op_data->params.activation, &activation_min,
                           &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = op_data->params.stride_height;
  op_params.stride_width = op_data->params.stride_width;
  op_params.filter_height = op_data->params.filter_height;
  op_params.filter_width = op_data->params.filter_width;
  op_params.padding_values.height = op_data->params.computed.padding.height;
  op_params.padding_values.width = op_data->params.computed.padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;

  TfLiteTensor *output, *indices;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kDataOutputTensor, &output));
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kIndicesOutputTensor, &indices));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kDataInputTensor, &input));

  switch (input->type) {
    case kTfLiteFloat32:
      MaxPool<float>(op_params, GetTensorShape(input), GetTensorShape(output),
                     GetTensorData<float>(input), GetTensorData<float>(output),
                     GetTensorData<int32_t>(indices));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace max_pool_with_argmax

TfLiteRegistration* RegisterMaxPoolWithArgmax() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_5(mht_5_v, 425, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "RegisterMaxPoolWithArgmax");

  static TfLiteRegistration r = {
      max_pool_with_argmax::Init, max_pool_with_argmax::Free,
      max_pool_with_argmax::Prepare, max_pool_with_argmax::Eval};
  return &r;
}

// Alias for selective build.
TfLiteRegistration* Register_MAX_POOL_WITH_ARGMAX() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmaxDTcc mht_6(mht_6_v, 436, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax.cc", "Register_MAX_POOL_WITH_ARGMAX");

  return RegisterMaxPoolWithArgmax();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
