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
class MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc() {
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
#include <iterator>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace topk_v2 {
constexpr int kInputTensor = 0;
constexpr int kInputTopK = 1;
constexpr int kOutputValues = 0;
constexpr int kOutputIndexes = 1;

namespace {
TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/topk_v2.cc", "ResizeOutput");

  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  // INT32 number of top results is supported.
  TF_LITE_ENSURE_TYPES_EQ(context, top_k->type, kTfLiteInt32);
  // Check that the tensor contains only one value.
  TF_LITE_ENSURE_EQ(context, NumElements(top_k), 1);
  const int32 k = *GetTensorData<int32_t>(top_k);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const int num_dimensions = NumDimensions(input);
  // Check that input has one or more dimensions.
  TF_LITE_ENSURE_MSG(context, input->dims->size >= 1,
                     "TopK k input must have 1 or more dimensions.");
  // Check that k is less or equal the internal dimension.
  TF_LITE_ENSURE_MSG(context, k <= input->dims->data[num_dimensions - 1],
                     "TopK k is higher than the internal dimension.");

  TfLiteIntArray* output_indexes_shape = TfLiteIntArrayCreate(num_dimensions);
  TfLiteIntArray* output_values_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions - 1; ++i) {
    output_indexes_shape->data[i] = input->dims->data[i];
    output_values_shape->data[i] = input->dims->data[i];
  }
  output_indexes_shape->data[num_dimensions - 1] = k;
  output_values_shape->data[num_dimensions - 1] = k;
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  // Force output types.
  output_indexes->type = kTfLiteInt32;
  output_values->type = input->type;
  auto resize_tensor = [context](TfLiteTensor* tensor, TfLiteIntArray* new_size,
                                 TfLiteIntArray* delete_on_error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_1(mht_1_v, 246, "", "./tensorflow/lite/kernels/topk_v2.cc", "lambda");

    TfLiteStatus status = context->ResizeTensor(context, tensor, new_size);
    if (status != kTfLiteOk) {
      if (delete_on_error != nullptr) {
        TfLiteIntArrayFree(delete_on_error);
      }
    }
    return status;
  };
  TF_LITE_ENSURE_OK(context, resize_tensor(output_indexes, output_indexes_shape,
                                           output_values_shape));
  TF_LITE_ENSURE_OK(context,
                    resize_tensor(output_values, output_values_shape, nullptr));
  return kTfLiteOk;
}

// Class that collects indices of top k values.  Based on template
// tensorflow::gtl::TopN<> but, for optimization, it re-uses the same container.
template <typename T>
class TopContainer {
 public:
  TopContainer() = delete;
  TopContainer(int32 k, int32 row_size) : k_(k) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_2(mht_2_v, 271, "", "./tensorflow/lite/kernels/topk_v2.cc", "TopContainer");

    container_.reserve(std::min(k, row_size) + 1);
  }

  void start_collecting(const T* values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_3(mht_3_v, 278, "", "./tensorflow/lite/kernels/topk_v2.cc", "start_collecting");

    values_ = values;
    container_.clear();
    is_heap_ = false;
  }

  void push(int32 a) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/kernels/topk_v2.cc", "push");

    auto comparator = [this](int32 a, int32 b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_5(mht_5_v, 291, "", "./tensorflow/lite/kernels/topk_v2.cc", "lambda");
 return compare_fun(a, b); };
    if (!is_heap_) {
      container_.push_back(a);
      if (container_.size() == k_ + 1) {
        std::make_heap(container_.begin(), container_.end(), comparator);
        std::pop_heap(container_.begin(), container_.end(), comparator);
        container_.pop_back();
        is_heap_ = true;
      }
    } else if (comparator(a, container_.front())) {
      // Due to how we defined comparator / compare_fun, container_.front()
      // contains the index of the smallest of the top-k elements seen so far.
      //
      // If control reaches this point, we know that the current index a
      // corresponds to an element which is bigger than the smallest of the
      // top-k elements seen so far.  Hence, we have to update the indices of
      // the top-k elements, by removing the index of the smallest top-k
      // element, adding a, and making sure container_[0:k] is still a heap.
      std::pop_heap(container_.begin(), container_.end(), comparator);
      container_.back() = a;
      std::push_heap(container_.begin(), container_.end(), comparator);
    }
  }

  const std::vector<int32>& sorted_result() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_6(mht_6_v, 318, "", "./tensorflow/lite/kernels/topk_v2.cc", "sorted_result");

    auto comparator = [this](int32 a, int32 b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_7(mht_7_v, 322, "", "./tensorflow/lite/kernels/topk_v2.cc", "lambda");
 return compare_fun(a, b); };
    if (!is_heap_) {
      // Note: due to the way we defined compare_fun (see comments for that
      // function) std::sort puts the indices from container_ in decreasing
      // order of the corresponding elements.
      std::sort(container_.begin(), container_.end(), comparator);
    } else {
      std::sort_heap(container_.begin(), container_.end(), comparator);
    }
    return container_;
  }

 private:
  const int32 k_;

  // container_[0,k) holds the indices of the largest k elements from values_
  // seen so far.  If more than k elements are pushed, then elements are
  // maintained in a min-heap order: container_.front() is
  // the index of the smallest of the top-k elements see so far.
  std::vector<int32> container_;

  // Once more than k elements are pushed, the container becomes a min heap,
  // and is_heap_ becomes true.
  bool is_heap_ = false;

  const T* values_ = nullptr;

  // Compares indices a and b based on the corresponding elements from values_.
  //
  // Intuitively, compare_fun(a, b) returns true iff values_[b] < values_[a]
  // (notice the inversion of direction, not a typo); ties (==) are broken in
  // favor of earlier elements (i.e., a < b).
  bool compare_fun(int32 a, int32 b) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_8(mht_8_v, 357, "", "./tensorflow/lite/kernels/topk_v2.cc", "compare_fun");

    if (values_[b] < values_[a]) {
      return true;
    } else if (values_[b] > values_[a]) {
      return false;
    } else {
      return a < b;
    }
  }
};

// Mostly modeled on tensorflow/core/kernels/topk_op.cc for CPU.
template <typename T>
void TopK(int32 row_size, int32 num_rows, const T* data, int32 k,
          int32* output_indexes, T* output_values) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_9(mht_9_v, 374, "", "./tensorflow/lite/kernels/topk_v2.cc", "TopK");

  TopContainer<T> topc(k, row_size);
  for (int row = 0; row < num_rows; ++row) {
    const T* values_row = data + row * row_size;
    topc.start_collecting(values_row);
    for (int32 c = 0; c < row_size; ++c) {
      topc.push(c);
    }

    // Prepare output buffers.
    int32* indexes_row = output_indexes + row * k;
    T* output_row = output_values + row * k;
    // We always assume that the output is sorted.
    const auto& top_k = topc.sorted_result();
    std::copy(top_k.begin(), top_k.end(), indexes_row);
    std::transform(top_k.begin(), top_k.end(), output_row,
                   [values_row](const int32 loc) { return values_row[loc]; });
  }
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_10(mht_10_v, 399, "", "./tensorflow/lite/kernels/topk_v2.cc", "Prepare");

  // Check that the inputs and outputs have the right sizes and types.
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output_values->type);

  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  TF_LITE_ENSURE_TYPES_EQ(context, top_k->type, kTfLiteInt32);

  // Set output dynamic if the `top_k` tensor is not constant, or the input has
  // dynamic dimensions (indicated by dims signature).
  if (IsConstantTensor(top_k) && !HasUnspecifiedDimension(input)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  } else {
    TfLiteTensor* output_indexes;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
    TfLiteTensor* output_values;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kOutputValues, &output_values));
    SetTensorToDynamic(output_indexes);
    SetTensorToDynamic(output_values);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_11(mht_11_v, 435, "", "./tensorflow/lite/kernels/topk_v2.cc", "Eval");

  TfLiteTensor* output_values;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputValues, &output_values));
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputIndexes, &output_indexes));
  if (IsDynamicTensor(output_values)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }
  const TfLiteTensor* top_k;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTopK, &top_k));
  const int32 k = top_k->data.i32[0];
  // The tensor can have more than 2 dimensions or even be a vector, the code
  // anyway calls the internal dimension as row;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const int32 row_size = input->dims->data[input->dims->size - 1];
  int32 num_rows = 1;
  for (int i = 0; i < input->dims->size - 1; ++i) {
    num_rows *= input->dims->data[i];
  }
  switch (output_values->type) {
    case kTfLiteFloat32:
      TopK(row_size, num_rows, GetTensorData<float>(input), k,
           output_indexes->data.i32, GetTensorData<float>(output_values));
      break;
    case kTfLiteUInt8:
      TopK(row_size, num_rows, input->data.uint8, k, output_indexes->data.i32,
           output_values->data.uint8);
      break;
    case kTfLiteInt8:
      TopK(row_size, num_rows, input->data.int8, k, output_indexes->data.i32,
           output_values->data.int8);
      break;
    case kTfLiteInt32:
      TopK(row_size, num_rows, input->data.i32, k, output_indexes->data.i32,
           output_values->data.i32);
      break;
    case kTfLiteInt64:
      TopK(row_size, num_rows, input->data.i64, k, output_indexes->data.i32,
           output_values->data.i64);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is currently not supported by TopK.",
                         TfLiteTypeGetName(output_values->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace topk_v2
TfLiteRegistration* Register_TOPK_V2() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStopk_v2DTcc mht_12(mht_12_v, 490, "", "./tensorflow/lite/kernels/topk_v2.cc", "Register_TOPK_V2");

  static TfLiteRegistration r = {nullptr, nullptr, topk_v2::Prepare,
                                 topk_v2::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
