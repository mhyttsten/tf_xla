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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dynamic_update_slice {

constexpr int kOperandTensor = 0;
constexpr int kUpdateTensor = 1;
constexpr int kStartIndicesTensor = 2;
constexpr int kOutputTensor = 0;

// TFLite DynamicUpdateSlice op follows the semantics of XLA DynamicUpdateSlice
// op. See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
// for details.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/dynamic_update_slice.cc", "Prepare");

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* update;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kUpdateTensor, &update));
  const TfLiteTensor* start_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartIndicesTensor,
                                          &start_indices));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // The shape of start_indices must be rank == 1, with dimension size equal to
  // the rank of operand.
  TF_LITE_ENSURE(context, NumDimensions(start_indices) == 1);
  TF_LITE_ENSURE(context,
                 SizeOfDimension(start_indices, 0) == NumDimensions(operand));

  // Update must be less than or equal to the operand size for each dimension to
  // avoid generating out-of-bounds update indices.
  TF_LITE_ENSURE(context, NumDimensions(update) == NumDimensions(operand));
  for (int i = 0; i < NumDimensions(operand); i++) {
    TF_LITE_ENSURE(context,
                   SizeOfDimension(update, i) <= SizeOfDimension(operand, i));
  }

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, update->type);
  TF_LITE_ENSURE_TYPES_EQ(context, start_indices->type, kTfLiteInt32);

  output->type = operand->type;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(operand->dims);
  return context->ResizeTensor(context, output, output_size);
}

// A helper function that converts a tensor index into a flat array index.
// Takes `start_indices` as an offset if not null.
int TensorIndexToFlat(const int* index, const int dims,
                      const RuntimeShape& shape,
                      const int* start_indices = nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc mht_1(mht_1_v, 256, "", "./tensorflow/lite/kernels/dynamic_update_slice.cc", "TensorIndexToFlat");

  int flat_index = index[0] + (start_indices ? start_indices[0] : 0);
  for (int i = 1; i < dims; i++) {
    flat_index = flat_index * shape.Dims(i) + index[i] +
                 (start_indices ? start_indices[i] : 0);
  }
  return flat_index;
}

// A helper function to compute the clamped start indices to ensure they are
// not out of bounds.
std::vector<int> ClampStartIndices(int input_dims, const int32_t* indices_data,
                                   const RuntimeShape& input_shape,
                                   const RuntimeShape& update_shape) {
  std::vector<int> clamped_start_indices(input_dims, 0);
  for (int i = 0; i < input_dims; i++) {
    clamped_start_indices[i] =
        std::min(std::max(0, indices_data[i]),
                 input_shape.Dims(i) - update_shape.Dims(i));
  }
  return clamped_start_indices;
}

template <typename T>
void DynamicUpdateSlice(const TfLiteTensor* input, const TfLiteTensor* update,
                        const TfLiteTensor* indice, TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc mht_2(mht_2_v, 284, "", "./tensorflow/lite/kernels/dynamic_update_slice.cc", "DynamicUpdateSlice");

  const auto& input_shape = GetTensorShape(input);
  const auto& update_shape = GetTensorShape(update);
  const T* update_data = GetTensorData<T>(update);
  const int32_t* indices_data = GetTensorData<int32_t>(indice);
  T* output_data = GetTensorData<T>(output);

  const int input_dims = input_shape.DimensionsCount();
  // Computes the effective slice indices.
  // The clamped indices are gauranteed to >= 0 since update is less than or
  // equal to the operand size for each dimension.
  std::vector<int> clamped_start_indices =
      ClampStartIndices(input_dims, indices_data, input_shape, update_shape);

  // Copies input to output first.
  memcpy(output->data.raw, input->data.raw, input->bytes);

  std::vector<int> current_dim(input_dims, 0);
  // Overwrites update to output.
  do {
    int flat_update_index =
        TensorIndexToFlat(current_dim.data(), input_dims, update_shape);
    int flat_input_index =
        TensorIndexToFlat(current_dim.data(), input_dims, input_shape,
                          clamped_start_indices.data());
    output_data[flat_input_index] = update_data[flat_update_index];
  } while (NextIndex(input_dims,
                     reinterpret_cast<const int*>(update_shape.DimsData()),
                     current_dim.data()));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc mht_3(mht_3_v, 318, "", "./tensorflow/lite/kernels/dynamic_update_slice.cc", "Eval");

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* update;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kUpdateTensor, &update));
  const TfLiteTensor* indice;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kStartIndicesTensor, &indice));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (operand->type) {
    case kTfLiteFloat32:
      DynamicUpdateSlice<float>(operand, update, indice, output);
      break;
    case kTfLiteBool:
      DynamicUpdateSlice<bool>(operand, update, indice, output);
      break;
    case kTfLiteInt8:
      DynamicUpdateSlice<int8_t>(operand, update, indice, output);
      break;
    case kTfLiteInt32:
      DynamicUpdateSlice<int32_t>(operand, update, indice, output);
      break;
    case kTfLiteInt64:
      DynamicUpdateSlice<int64_t>(operand, update, indice, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "DynamicUpdateSlice only currently supports "
                         "1-bit/8-bit/32-bit/64-bit integer or "
                         "float type, got %d.",
                         operand->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace dynamic_update_slice

TfLiteRegistration* Register_DYNAMIC_UPDATE_SLICE() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_sliceDTcc mht_4(mht_4_v, 364, "", "./tensorflow/lite/kernels/dynamic_update_slice.cc", "Register_DYNAMIC_UPDATE_SLICE");

  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 dynamic_update_slice::Prepare,
                                 dynamic_update_slice::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
