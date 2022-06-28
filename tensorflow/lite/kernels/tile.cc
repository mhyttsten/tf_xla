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
class MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc() {
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
#include <stdint.h>

#include <algorithm>
#include <tuple>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace tile {

constexpr int kInputTensor = 0;
constexpr int kInputMultipliers = 1;
constexpr int kOutputTensor = 0;

namespace {
template <typename T>
TfLiteIntArray* MultiplyShapeDims(const TfLiteIntArray& shape,
                                  const TfLiteTensor* multipliers,
                                  int num_dimensions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/tile.cc", "MultiplyShapeDims");

  const T* multipliers_v = GetTensorData<T>(multipliers);

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->data[i] = shape.data[i] * multipliers_v[i];
  }
  return output_shape;
}

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/kernels/tile.cc", "ResizeOutput");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* multipliers;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputMultipliers, &multipliers));

  const int num_dimensions = NumDimensions(input);
  const int num_multipliers = NumElements(multipliers);
  TF_LITE_ENSURE_EQ(context, num_dimensions, num_multipliers);
  switch (multipliers->type) {
    case kTfLiteInt32:
      return context->ResizeTensor(
          context, output,
          MultiplyShapeDims<int32_t>(*input->dims, multipliers,
                                     num_dimensions));
    case kTfLiteInt64:
      return context->ResizeTensor(
          context, output,
          MultiplyShapeDims<int64_t>(*input->dims, multipliers,
                                     num_dimensions));
    default:
      context->ReportError(
          context, "Multipliers of type '%s' are not supported by tile.",
          TfLiteTypeGetName(multipliers->type));
      return kTfLiteError;
  }
}

template <typename T, typename M>
void CopyMultipleTimes(const T* in_data, int32_t in_size, M multiplier,
                       T* out_data) {
  for (M i = 0; i < multiplier; ++i) {
    const T* in_end = in_data + in_size;
    T* new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename M>
void CopyStringMultipleTimes(const TfLiteTensor* in_data, int in_data_index,
                             const int dimension_size, M multiplier,
                             DynamicBuffer* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_2(mht_2_v, 272, "", "./tensorflow/lite/kernels/tile.cc", "CopyStringMultipleTimes");

  for (M i = 0; i < multiplier; ++i) {
    for (int j = 0; j < dimension_size; ++j) {
      const auto string_ref = GetString(in_data, in_data_index + j);
      buffer->AddString(string_ref.str, string_ref.len);
    }
  }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const TfLiteIntArray& in_dimensions,
                                     const T* in_data, const M* multipliers,
                                     T* out_data, int dimension) {
  if (in_dimensions.size == 0) {
    // If input tensor is a scalar, then just copy it to output (no need to
    // multiply).
    *out_data = *in_data;
    return std::make_pair(0, 0);
  }

  const int dimension_size = in_dimensions.data[dimension];
  if (dimension == in_dimensions.size - 1) {
    CopyMultipleTimes(in_data, dimension_size, multipliers[dimension],
                      out_data);
    return std::make_pair(
        dimension_size,
        dimension_size * static_cast<int>(multipliers[dimension]));
  }
  int total_stride_size = 0, total_tiled_stride_size = 0;
  const T* copy_from_data = in_data;
  T* copy_to_data = out_data;
  for (int i = 0; i < dimension_size; ++i) {
    int stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
        TileOneDimension(in_dimensions, copy_from_data, multipliers,
                         copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size,
                    multipliers[dimension] - 1,
                    out_data + total_tiled_stride_size);
  return std::make_pair(
      total_stride_size,
      static_cast<int>(total_tiled_stride_size * multipliers[dimension]));
}

template <typename M>
std::pair<int, int> TileStringOneDimension(
    const TfLiteIntArray& in_dimensions, const TfLiteTensor* in_data,
    int in_data_index, const M* multipliers, DynamicBuffer* buffer,
    int buffer_index, int dimension, TfLiteTensor* out_data) {
  const int dimension_size = in_dimensions.data[dimension];
  if (dimension == in_dimensions.size - 1) {
    CopyStringMultipleTimes(in_data, in_data_index, dimension_size,
                            multipliers[dimension], buffer);
    return {dimension_size,
            dimension_size * static_cast<int>(multipliers[dimension])};
  }

  int total_stride_size = 0, total_tiled_stride_size = 0;
  for (int i = 0; i < dimension_size; ++i) {
    int stride_size, tiled_stride_size;
    std::tie(stride_size, tiled_stride_size) = TileStringOneDimension(
        in_dimensions, in_data, in_data_index + total_stride_size, multipliers,
        buffer, buffer_index + total_tiled_stride_size, dimension + 1,
        out_data);
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }

  buffer->WriteToTensor(out_data, /*new_shape=*/nullptr);
  CopyStringMultipleTimes(out_data, buffer_index, total_tiled_stride_size,
                          multipliers[dimension] - 1, buffer);

  return {total_stride_size,
          total_tiled_stride_size * static_cast<int>(multipliers[dimension])};
}

template <typename T>
void Tile(const TfLiteIntArray& in_dimensions, const TfLiteTensor* in_data,
          const TfLiteTensor* multipliers, TfLiteTensor* out_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_3(mht_3_v, 358, "", "./tensorflow/lite/kernels/tile.cc", "Tile");

  // Doing recursively tiling from top to down dimension.
  switch (multipliers->type) {
    case kTfLiteInt32:
      TileOneDimension(in_dimensions, GetTensorData<T>(in_data),
                       GetTensorData<int32_t>(multipliers),
                       GetTensorData<T>(out_data), 0);
      break;
    case kTfLiteInt64:
      TileOneDimension(in_dimensions, GetTensorData<T>(in_data),
                       GetTensorData<int64_t>(multipliers),
                       GetTensorData<T>(out_data), 0);
      break;
    default:
      break;
  }
}

void TileString(const TfLiteIntArray& in_dimensions,
                const TfLiteTensor* in_data, const TfLiteTensor* multipliers,
                DynamicBuffer* buffer, TfLiteTensor* out_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_4(mht_4_v, 381, "", "./tensorflow/lite/kernels/tile.cc", "TileString");

  // Doing recursively tiling from top to down dimension.
  switch (multipliers->type) {
    case kTfLiteInt32:
      TileStringOneDimension(in_dimensions, in_data, 0,
                             GetTensorData<int32_t>(multipliers), buffer, 0, 0,
                             out_data);
      break;
    case kTfLiteInt64:
      TileStringOneDimension(in_dimensions, in_data, 0,
                             GetTensorData<int64_t>(multipliers), buffer, 0, 0,
                             out_data);
      break;
    default:
      break;
  }
}
}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_5(mht_5_v, 403, "", "./tensorflow/lite/kernels/tile.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  const TfLiteTensor* multipliers;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputMultipliers, &multipliers));
  // Only int32 and int64 multipliers type is supported.
  if (multipliers->type != kTfLiteInt32 && multipliers->type != kTfLiteInt64) {
    context->ReportError(context,
                         "Multipliers of type '%s' are not supported by tile.",
                         TfLiteTypeGetName(multipliers->type));
    return kTfLiteError;
  }

  if (IsConstantTensor(multipliers)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  } else {
    SetTensorToDynamic(output);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_6(mht_6_v, 437, "", "./tensorflow/lite/kernels/tile.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* multipliers;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputMultipliers, &multipliers));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }
  if (GetTensorShape(output).FlatSize() == 0) {
    return kTfLiteOk;
  }

  switch (output->type) {
    case kTfLiteFloat32:
      Tile<float>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteInt8:
      Tile<int8_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteUInt8:
      Tile<uint8_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteInt32:
      Tile<int32_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteInt64:
      Tile<int64_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteString: {
      DynamicBuffer buffer;
      TileString(*(input->dims), input, multipliers, &buffer, output);
      buffer.WriteToTensor(output, /*new_shape=*/nullptr);
      break;
    }
    case kTfLiteBool:
      Tile<bool>(*(input->dims), input, multipliers, output);
      break;
    default:
      context->ReportError(context, "Type '%s' is not supported by tile.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace tile
TfLiteRegistration* Register_TILE() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStileDTcc mht_7(mht_7_v, 491, "", "./tensorflow/lite/kernels/tile.cc", "Register_TILE");

  static TfLiteRegistration r = {nullptr, nullptr, tile::Prepare, tile::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
