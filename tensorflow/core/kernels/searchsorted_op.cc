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
class MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/searchsorted_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T, typename OutType>
struct UpperBoundFunctor<CPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/searchsorted_op.cc", "Compute");

    auto work_fn = [&](int64_t first, int64_t last) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/searchsorted_op.cc", "lambda");

      for (int b = 0; b < batch_size; ++b) {
        const T* sorted_inputs_ptr = sorted_inputs.data() + b * num_inputs;
        OutType* output_ptr = output->data() + b * num_values;
        for (int i = first; i < last; ++i) {
          output_ptr[i] = std::upper_bound(sorted_inputs_ptr,
                                           sorted_inputs_ptr + num_inputs,
                                           values(i + b * num_values)) -
                          sorted_inputs_ptr;
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;
    const float kCostMultiplier = 1.f;  // Can be tuned to minimize overhead
    int64_t cost_per_unit =
        kCostMultiplier * batch_size * Log2Ceiling(num_inputs);
    thread_pool->ParallelFor(num_values, cost_per_unit, work_fn);
    return Status::OK();
  }
};

template <typename T, typename OutType>
struct LowerBoundFunctor<CPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/searchsorted_op.cc", "Compute");

    auto work_fn = [&](int64_t first, int64_t last) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/kernels/searchsorted_op.cc", "lambda");

      for (int b = 0; b < batch_size; ++b) {
        const T* sorted_inputs_ptr = sorted_inputs.data() + b * num_inputs;
        OutType* output_ptr = output->data() + b * num_values;
        for (int i = first; i < last; ++i) {
          output_ptr[i] = std::lower_bound(sorted_inputs_ptr,
                                           sorted_inputs_ptr + num_inputs,
                                           values(i + b * num_values)) -
                          sorted_inputs_ptr;
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;
    const float kCostMultiplier = 1.f;  // Can be tuned to minimize overhead
    int64_t cost_per_unit =
        kCostMultiplier * batch_size * Log2Ceiling(num_inputs);
    thread_pool->ParallelFor(num_values, cost_per_unit, work_fn);
    return Status::OK();
  }
};
}  // namespace functor

template <typename Device, typename T, typename OutType>
class UpperBoundOp : public OpKernel {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_4(mht_4_v, 278, "", "./tensorflow/core/kernels/searchsorted_op.cc", "UpperBoundOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/kernels/searchsorted_op.cc", "Compute");

    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // inputs must be at least a matrix
    OP_REQUIRES(
        ctx, sorted_inputs_t.shape().dims() >= 2,
        errors::InvalidArgument("sorted input argument must be a matrix"));
    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                Status(error::INVALID_ARGUMENT,
                       "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                Status(error::INVALID_ARGUMENT,
                       "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();
    OP_REQUIRES_OK(
        ctx, functor::UpperBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

template <typename Device, typename T, typename OutType>
class LowerBoundOp : public OpKernel {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/kernels/searchsorted_op.cc", "LowerBoundOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsearchsorted_opDTcc mht_7(mht_7_v, 334, "", "./tensorflow/core/kernels/searchsorted_op.cc", "Compute");

    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // inputs must be at least a matrix
    OP_REQUIRES(
        ctx, sorted_inputs_t.shape().dims() >= 2,
        errors::InvalidArgument("sorted input argument must be a matrix"));
    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                Status(error::INVALID_ARGUMENT,
                       "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                Status(error::INVALID_ARGUMENT,
                       "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();
    OP_REQUIRES_OK(
        ctx, functor::LowerBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow
