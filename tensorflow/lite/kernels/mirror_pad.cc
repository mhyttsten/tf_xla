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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc() {
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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mirror_pad {
namespace {

// Nil value for paddingMode/offset.
const int kUnsetOffset = -1;

// Wrapper for params passed to the Eval<T> function.
template <typename T>
struct EvalData {
  const TfLiteTensor* padding_matrix = nullptr;
  const TfLiteIntArray* input_dims = nullptr;
  // Holds number of elements at the nth dimension.
  // value at last dimension = 1, at second to last = sizeof last dimension.
  const std::vector<int>* output_dims_num_elements = nullptr;
  const std::vector<int>* input_dims_num_elements = nullptr;
  const T* input_data = nullptr;

  int offset = kUnsetOffset;
  T* output_data = nullptr;
  int num_dims = 0;
};

// Helper method that fills the left and right pads.
template <typename T>
inline void GetPadding(const T* data, int offset, int64_t* left_pad,
                       int64_t* right_pad) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/mirror_pad.cc", "GetPadding");

  *left_pad = static_cast<int64_t>(*(data + offset * 2));
  *right_pad = static_cast<int64_t>(*(data + offset * 2 + 1));
}

inline void GetPadding(const TfLiteTensor* padding_matrix, int dimension,
                       int64_t* left_pad, int64_t* right_pad) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/kernels/mirror_pad.cc", "GetPadding");

  switch (padding_matrix->type) {
    case kTfLiteInt32:
      GetPadding(padding_matrix->data.i32, dimension, left_pad, right_pad);
      break;
    case kTfLiteInt64:
      GetPadding(padding_matrix->data.i64, dimension, left_pad, right_pad);
      break;
    default:
      return;
  }
}

// Returns the shape of the final output after padding.
std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> GetPaddedOutputShape(
    const TfLiteTensor* input, const TfLiteTensor* padding_matrix) {
  const int input_dims = NumDimensions(input);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(input_dims), TfLiteIntArrayFree);

  int64_t left_pad = 0, right_pad = 0;
  for (int i = 0; i < input_dims; ++i) {
    GetPadding(padding_matrix, i, &left_pad, &right_pad);
    shape->data[i] = SizeOfDimension(input, i) + left_pad + right_pad;
  }
  return shape;
}

// Given dimension index and the left/right padding.
// Returns the corresponding dimension in the input array.
inline int GetInputDimension(int padded_dimension, int left_pad, int right_pad,
                             int input_dim_size, int offset) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_2(mht_2_v, 275, "", "./tensorflow/lite/kernels/mirror_pad.cc", "GetInputDimension");

  if (padded_dimension < left_pad) {
    const int original_ind = left_pad + offset - 1;
    return original_ind - (std::min(padded_dimension, original_ind - offset));
  }
  padded_dimension -= left_pad;
  if (padded_dimension >= input_dim_size) {
    padded_dimension -= input_dim_size;
    const int original_ind = input_dim_size - (1 + offset);
    return original_ind - std::min(padded_dimension, original_ind);
  }
  return padded_dimension;
}

// Given and index in output array, returns the index of the value
// in input array.
template <typename T>
int GetFlatIndex(int index, EvalData<T>* eval_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_3(mht_3_v, 295, "", "./tensorflow/lite/kernels/mirror_pad.cc", "GetFlatIndex");

  int flat_index = 0;
  int64_t left_pad = 0, right_pad = 0, dimension_index, index_in_input;
  for (int i = 0; i < eval_data->num_dims; ++i) {
    switch (eval_data->padding_matrix->type) {
      case kTfLiteInt32:
        GetPadding(eval_data->padding_matrix->data.i32, i, &left_pad,
                   &right_pad);
        break;
      case kTfLiteInt64:
        GetPadding(eval_data->padding_matrix->data.i64, i, &left_pad,
                   &right_pad);
        break;
      default:
        break;
    }
    dimension_index = index / (*eval_data->output_dims_num_elements)[i];
    index_in_input =
        GetInputDimension(dimension_index, left_pad, right_pad,
                          eval_data->input_dims->data[i], eval_data->offset);
    flat_index += index_in_input * (*eval_data->input_dims_num_elements)[i];
    index %= (*eval_data->output_dims_num_elements)[i];
  }
  return flat_index;
}

template <typename T>
struct MirrorPadWorkerTask : cpu_backend_threadpool::Task {
  MirrorPadWorkerTask(EvalData<T>* eval_data, int start, int end)
      : eval_data(eval_data), start(start), end(end) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_4(mht_4_v, 327, "", "./tensorflow/lite/kernels/mirror_pad.cc", "MirrorPadWorkerTask");
}
  void Run() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_5(mht_5_v, 331, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Run");

    auto* input_data = eval_data->input_data;
    auto* output_data = eval_data->output_data;
    for (int i = start; i < end; ++i) {
      output_data[i] = input_data[GetFlatIndex(i, eval_data)];
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_6(mht_6_v, 350, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Eval");

  ruy::profiler::ScopeLabel label("MirrorPad");
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  const TfLiteTensor* padding_matrix;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &padding_matrix));
  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);

  if (params == nullptr) {
    return kTfLiteError;
  }
  const int input_dims = NumDimensions(input_tensor);

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  if (IsDynamicTensor(output_tensor)) {
    auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
    if (output_size == nullptr) {
      return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(
        context->ResizeTensor(context, output_tensor, output_size.release()));
  }

  std::vector<int> output_dims_num_elements(input_dims, 1);
  std::vector<int> input_dims_num_elements(input_dims, 1);
  for (int i = input_dims - 2; i >= 0; i--) {
    output_dims_num_elements[i] =
        output_dims_num_elements[i + 1] * output_tensor->dims->data[i + 1];
    input_dims_num_elements[i] =
        input_dims_num_elements[i + 1] * input_tensor->dims->data[i + 1];
  }

  const int offset =
      params->mode != TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect ? 0
                                                                           : 1;

  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const int thread_count = cpu_backend_context->max_num_threads();
  TfLiteStatus status = kTfLiteOk;
  const int output_size = NumElements(output_tensor);
#define TF_LITE_MIRROR_PAD(type)                                           \
  EvalData<type> eval_data;                                                \
  eval_data.input_data = GetTensorData<type>(input_tensor);                \
  eval_data.input_dims = input_tensor->dims;                               \
  eval_data.input_dims = input_tensor->dims;                               \
  eval_data.output_dims_num_elements = &output_dims_num_elements;          \
  eval_data.input_dims_num_elements = &input_dims_num_elements;            \
  eval_data.num_dims = input_dims;                                         \
  eval_data.offset = offset;                                               \
  eval_data.output_data = GetTensorData<type>(output_tensor);              \
  eval_data.padding_matrix = padding_matrix;                               \
  std::vector<MirrorPadWorkerTask<type>> tasks;                            \
  tasks.reserve(thread_count);                                             \
  int start = 0;                                                           \
  for (int i = 0; i < thread_count; ++i) {                                 \
    int end = start + (output_size - start) / (thread_count - i);          \
    tasks.emplace_back(MirrorPadWorkerTask<type>(&eval_data, start, end)); \
    start = end;                                                           \
  }                                                                        \
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),              \
                                  cpu_backend_context);

  switch (output_tensor->type) {
    case kTfLiteFloat32: {
      TF_LITE_MIRROR_PAD(float);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_MIRROR_PAD(int32_t);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_MIRROR_PAD(uint8_t);
      break;
    }
    case kTfLiteInt8: {
      TF_LITE_MIRROR_PAD(int8_t);
      break;
    }
    case kTfLiteInt64: {
      TF_LITE_MIRROR_PAD(int64_t);
      break;
    }
    default:
      status = kTfLiteError;
      break;
  }
#undef TF_LITE_MIRROR_PAD
  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_7(mht_7_v, 448, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Init");

  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_8(mht_8_v, 455, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Free");
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_9(mht_9_v, 460, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Prepare");

  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  const TfLiteTensor* padding_matrix;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &padding_matrix));
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));

  TF_LITE_ENSURE_EQ(context, NumDimensions(padding_matrix), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 0),
                    NumDimensions(input_tensor));

  if (!IsConstantTensor(padding_matrix)) {
    SetTensorToDynamic(output_tensor);
    return kTfLiteOk;
  }
  // We have constant padding, so we can infer output size.
  auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
  if (output_size == nullptr) {
    return kTfLiteError;
  }
  return context->ResizeTensor(context, output_tensor, output_size.release());
}

}  // namespace mirror_pad
TfLiteRegistration* Register_MIRROR_PAD() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_padDTcc mht_10(mht_10_v, 488, "", "./tensorflow/lite/kernels/mirror_pad.cc", "Register_MIRROR_PAD");

  static TfLiteRegistration r = {mirror_pad::Init, mirror_pad::Free,
                                 mirror_pad::Prepare, mirror_pad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
