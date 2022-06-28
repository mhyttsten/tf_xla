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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh() {
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


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace concat_split_util {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Concatenates 'inputs' into a single tensor along the zeroth dimension.
// Requires that all elements of 'inputs' have element type T. Writes to
// 'output' using 'context' for the allocation to ensure proper device
// placement.
template <typename T>
Status Concat(OpKernelContext* context, const gtl::ArraySlice<Tensor> inputs,
              Tensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/batching_util/concat_split_util.h", "Concat");

  const int input_dims = inputs[0].dims();
  const TensorShape& input_shape = inputs[0].shape();

  // Note that we reduce the concat of k-dimensional tensors into a two
  // dimensional concat. Assuming the dimensions of any input tensor are
  // {y0, y1,...,ym-1}, we flatten it to {1, y}, where y = Prod_i(yi).
  std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>> inputs_flat;
  inputs_flat.reserve(inputs.size());
  int64_t output_dim0 = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& input = inputs[i];
    if (input.dims() != input_dims) {
      return errors::InvalidArgument(
          "Ranks of all input tensors should match: shape[0] = ",
          input_shape.DebugString(), " vs. shape[", i,
          "] = ", input.shape().DebugString());
    }
    for (int j = 1; j < input_dims; ++j) {
      if (input.dim_size(j) != input_shape.dim_size(j)) {
        return errors::InvalidArgument(
            "Dimensions of inputs should match: shape[0] = ",
            input_shape.DebugString(), " vs. shape[", i,
            "] = ", input.shape().DebugString());
      }
    }
    if (input.NumElements() > 0) {
      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          input.shaped<T, 2>({1, input.NumElements()})));
    }
    output_dim0 += input.dim_size(0);
  }

  TensorShape output_shape(input_shape);
  output_shape.set_dim(0, output_dim0);
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<T>::value,
                                            output_shape, output, attr));
  if (output->NumElements() > 0) {
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
    if (std::is_same<Device, GPUDevice>::value) {
      ConcatGPU<T>(context, inputs_flat, output, &output_flat);
      return Status::OK();
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    ConcatCPU<T>(context->device(), inputs_flat, &output_flat);
  }

  return Status::OK();
}

// Same as 'Concat' above, but handles Tensor dtype deduction automatically.
inline Status Concat(OpKernelContext* context,
                     const gtl::ArraySlice<Tensor> inputs, Tensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh mht_1(mht_1_v, 267, "", "./tensorflow/core/kernels/batching_util/concat_split_util.h", "Concat");

  const DataType type = inputs[0].dtype();
  Status concat_status;
  switch (type) {
#define CASE(type)                                         \
  case DataTypeToEnum<type>::value:                        \
    concat_status = Concat<type>(context, inputs, output); \
    break;
    TF_CALL_ALL_TYPES(CASE);
#undef CASE
    default:
      concat_status = errors::InvalidArgument("Unsupported data type: ", type);
      break;
  }
  return concat_status;
}

// The Split*() functions split 'input' with element type T into 'sizes.size()'
// tensors along the zeroth dimension, with the ith split having zeroth-
// dimension size 'sizes[i]'. They allocate the output tensors using 'context',
// for proper device placement.

// Handles special cases that are cheap. Sets 'done==true' iff it found an
// applicable special case and wrote to the outputs. Otherwise acts as a no-op.
template <typename T>
Status SplitEasyCases(OpKernelContext* context, const Tensor& input,
                      const gtl::ArraySlice<int64_t> sizes,
                      std::vector<Tensor>* outputs, bool* done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh mht_2(mht_2_v, 297, "", "./tensorflow/core/kernels/batching_util/concat_split_util.h", "SplitEasyCases");

  *done = false;

  int64_t total_size = 0;
  for (const int64_t size : sizes) {
    total_size += size;
  }
  if (total_size > input.shape().dim_size(0)) {
    return errors::InvalidArgument(
        "Sum of split sizes must not exceed dim0-size of input tensor");
  }

  // Special case 0: trivial 1-way split.
  if (sizes.size() == 1 && sizes.at(0) == input.shape().dim_size(0)) {
    outputs->push_back(input);
    *done = true;
    return Status::OK();
  }

  // Special case 1: input is aligned.
  if (IsInnerDimsSizeAligned<T>(input.shape())) {
    int64_t position = 0;
    for (const int64_t size : sizes) {
      outputs->emplace_back(input.Slice(position, position + size));
      position += size;
    }
    *done = true;
    return Status::OK();
  }

  return Status::OK();
}

// Handles the general case, on CPU.
template <typename T>
Status SplitCPU(OpKernelContext* context, const Tensor& input,
                const gtl::ArraySlice<int64_t> sizes,
                std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh mht_3(mht_3_v, 337, "", "./tensorflow/core/kernels/batching_util/concat_split_util.h", "SplitCPU");

  int64_t suffix_dim_size = 1;
  for (int i = 1; i < input.shape().dims(); ++i) {
    suffix_dim_size *= input.shape().dim_size(i);
  }
  auto input_reshaped =
      input.shaped<T, 2>({input.shape().dim_size(0), suffix_dim_size});

  int64_t position = 0;
  for (const int64_t size : sizes) {
    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, size);
    Tensor output;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(
        context->allocate_temp(input.dtype(), output_shape, &output, attr));
    auto output_shaped = output.shaped<T, 2>({size, suffix_dim_size});

    Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{
        static_cast<Eigen::DenseIndex>(position), 0};
    Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{
        static_cast<Eigen::DenseIndex>(size),
        static_cast<Eigen::DenseIndex>(suffix_dim_size)};
    functor::Split<CPUDevice, T, 2>()(context->eigen_device<CPUDevice>(),
                                      output_shaped, input_reshaped,
                                      slice_indices, slice_sizes);

    outputs->emplace_back(output);

    position += size;
  }

  return Status::OK();
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Handles the general case, on GPU.
template <typename T>
Status SplitGPU(OpKernelContext* context, const Tensor& input,
                const gtl::ArraySlice<int64_t>& sizes,
                std::vector<Tensor>* outputs) {
  // TODO(olston, apassos): Implement this.
  LOG(FATAL) << "Not yet implemented";  // Crash ok
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// The outer function that dispatches to the various Split*() functions above.
template <typename T>
Status Split(OpKernelContext* context, const Tensor& input,
             const gtl::ArraySlice<int64_t> sizes,
             std::vector<Tensor>* outputs) {
  bool easy_cases_done;
  TF_RETURN_IF_ERROR(
      SplitEasyCases<T>(context, input, sizes, outputs, &easy_cases_done));
  if (easy_cases_done) {
    return Status::OK();
  }

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// TODO(olston, apassos): Handle non-CPU cases.
// return SplitGPU<T>(context, input, sizes, outputs);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return SplitCPU<T>(context, input, sizes, outputs);
}

// Same as 'Split' above, but handles Tensor dtype automatically.
inline Status Split(OpKernelContext* context, const Tensor& input,
                    const gtl::ArraySlice<int64_t> sizes,
                    std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSconcat_split_utilDTh mht_4(mht_4_v, 413, "", "./tensorflow/core/kernels/batching_util/concat_split_util.h", "Split");

  const DataType type = input.dtype();
  Status split_status;
  switch (type) {
#define CASE(type)                                              \
  case DataTypeToEnum<type>::value:                             \
    split_status = Split<type>(context, input, sizes, outputs); \
    break;
    TF_CALL_ALL_TYPES(CASE);
#undef CASE
    default:
      split_status = errors::InvalidArgument("Unsupported data type: ", type);
      break;
  }
  return split_status;
}

}  // namespace concat_split_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_CONCAT_SPLIT_UTIL_H_
