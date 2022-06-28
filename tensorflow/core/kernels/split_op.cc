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
class MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/work_sharder.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/split_lib_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SplitOpBase : public OpKernel {
 public:
  explicit SplitOpBase(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/split_op.cc", "SplitOpBase");
}

  void ComputeEasyCases(OpKernelContext* context, bool* done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/split_op.cc", "ComputeEasyCases");

    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_dim_tensor = context->input(0);
    OP_REQUIRES(
        context, split_dim_tensor.shape().dims() == 0,
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_tensor.shape().dims()));
    const int32_t split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32_t num_split = num_outputs();

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input_shape.dims(),
        errors::InvalidArgument("-input rank(-", input.dims(),
                                ") <= split_dim < input rank (", input.dims(),
                                "), but got ", split_dim_orig));

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(context, input_shape.dim_size(split_dim) % num_split == 0,
                errors::InvalidArgument(
                    "Number of ways to split should evenly divide the split "
                    "dimension, but got split_dim ",
                    split_dim, " (size = ", input_shape.dim_size(split_dim),
                    ") ", "and num_split ", num_split));
    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      VLOG(1) << "Split identity";
      context->set_output(0, context->input(1));
      *done = true;
      return;
    }

    // Special case 2: split along the 1st dimension. We can share the
    // underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if ((split_dim == 0) && IsInnerDimsSizeAligned<T>(input_shape)) {
      VLOG(1) << "Slice dim 0: " << input_shape.DebugString();
      const int64_t delta = input_shape.dim_size(0) / num_split;
      for (int i = 0; i < num_split; ++i) {
        context->set_output(i, input.Slice(i * delta, (i + 1) * delta));
      }
      *done = true;
      return;
    }
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, int32_t split_dim) const {
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer type");
    int32_t prefix_dim_size = 1;
    for (int i = 0; i < split_dim; ++i) {
      prefix_dim_size *= input_shape.dim_size(i);
    }

    // Caller must ensure that dim_size and suffix_dim_size are <
    // std::numeric_limits<IndexType>::max()
    IndexType split_dim_size =
        static_cast<IndexType>(input_shape.dim_size(split_dim));

    IndexType suffix_dim_size = 1;
    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      suffix_dim_size *= static_cast<IndexType>(input_shape.dim_size(i));
    }
    return std::make_tuple(prefix_dim_size, split_dim_size, suffix_dim_size);
  }
};

template <typename T, typename InputReshapedType, int NDims>
class SplitOpCPUImpl {
 public:
  template <typename MakeSizesType, typename ReshapeResultType>
  void operator()(OpKernelContext* context,
                  const InputReshapedType& input_reshaped,
                  const TensorShape& input_shape, int32_t split_dim,
                  Eigen::DenseIndex prefix_dim_size,
                  Eigen::DenseIndex split_dim_size,
                  Eigen::DenseIndex suffix_dim_size,
                  const MakeSizesType& make_sizes,
                  const ReshapeResultType& reshape_result, int32_t num_split,
                  int64_t split_dim_output_size) const {
    const auto num_threads =
        context->device()->tensorflow_cpu_worker_threads()->num_threads;
    // TODO(jewillco): Tune heuristic further.
    const auto input_element_count = input_shape.num_elements();
    const bool use_parallelism_between_outputs =
        (num_split >= 4 &&
         input_element_count >= std::max(num_threads, num_split) * 4096 &&
         input_element_count < num_split * 180 * 1024);
    Eigen::DSizes<Eigen::DenseIndex, NDims> indices;
    for (int i = 0; i < NDims; ++i) {
      indices[i] = 0;
    }
    auto sizes = make_sizes(split_dim_output_size);
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    auto range_output_func = [&indices, context, &output_shape, prefix_dim_size,
                              split_dim_output_size, suffix_dim_size, &sizes,
                              use_parallelism_between_outputs, &input_reshaped,
                              &reshape_result](int64_t start, int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_2(mht_2_v, 334, "", "./tensorflow/core/kernels/split_op.cc", "lambda");

      for (int64_t i = start; i < limit; ++i) {
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i, output_shape, &result));
        if (prefix_dim_size * split_dim_output_size * suffix_dim_size > 0) {
          Eigen::DSizes<Eigen::DenseIndex, NDims> slice_indices;
          Eigen::DSizes<Eigen::DenseIndex, NDims> slice_sizes;
          for (int j = 0; j < NDims; ++j) {
            slice_indices[j] =
                (j == NDims - 2 ? i * split_dim_output_size : indices[j]);
            slice_sizes[j] = sizes[j];
          }

          auto result_shaped = reshape_result(result, split_dim_output_size);

          if (use_parallelism_between_outputs) {
            // Use sequential implementation for single output.
            result_shaped = input_reshaped.slice(slice_indices, slice_sizes);
          } else {
            // This implementation may be parallel internally.
            functor::Split<CPUDevice, T, NDims>()(
                context->eigen_device<CPUDevice>(), result_shaped,
                input_reshaped, slice_indices, slice_sizes);
          }
        }
      }
    };
    if (use_parallelism_between_outputs) {
      // Run in parallel, disabling parallelism in functor.
      context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
          num_split, input_element_count / num_split, range_output_func);
    } else {
      // Run sequentially, but allow internal parallelism in functor.
      range_output_func(0, num_split);
    }
  }
};

template <typename T>
class SplitOpCPU : public SplitOpBase<CPUDevice, T> {
 public:
  typedef SplitOpBase<CPUDevice, T> Base;
  explicit SplitOpCPU(OpKernelConstruction* c) : Base(c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_3(mht_3_v, 380, "", "./tensorflow/core/kernels/split_op.cc", "SplitOpCPU");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_4(mht_4_v, 385, "", "./tensorflow/core/kernels/split_op.cc", "Compute");

    bool done = false;
    Base::ComputeEasyCases(context, &done);
    if (!context->status().ok() || done) {
      return;
    }
    const int32_t num_split = Base::num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32_t split_dim_orig = context->input(0).flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    // Android also uses int32 indexing, so check here also.
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(),
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("Split requires input size < ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));

    Eigen::DenseIndex prefix_dim_size;
    Eigen::DenseIndex split_dim_size;
    Eigen::DenseIndex suffix_dim_size;

    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<Eigen::DenseIndex>(input_shape, split_dim);

    const int64_t split_dim_output_size = split_dim_size / num_split;

    if (prefix_dim_size == 1) {
      auto input_reshaped =
          input.shaped<T, 2>({split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_5(mht_5_v, 421, "", "./tensorflow/core/kernels/split_op.cc", "lambda");

        return Eigen::DSizes<Eigen::DenseIndex, 2>{split_size, suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Eigen::DenseIndex split_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_6(mht_6_v, 427, "", "./tensorflow/core/kernels/split_op.cc", "lambda");

        return result->shaped<T, 2>({split_size, suffix_dim_size});
      };
      SplitOpCPUImpl<T, decltype(input_reshaped), 2>{}(
          context, input_reshaped, input_shape, split_dim, prefix_dim_size,
          split_dim_size, suffix_dim_size, make_sizes, reshape_result,
          num_split, split_dim_output_size);
    } else {
      auto input_reshaped = input.shaped<T, 3>(
          {prefix_dim_size, split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_7(mht_7_v, 440, "", "./tensorflow/core/kernels/split_op.cc", "lambda");

        return Eigen::DSizes<Eigen::DenseIndex, 3>{prefix_dim_size, split_size,
                                                   suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Eigen::DenseIndex split_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_8(mht_8_v, 447, "", "./tensorflow/core/kernels/split_op.cc", "lambda");

        return result->shaped<T, 3>(
            {prefix_dim_size, split_size, suffix_dim_size});
      };
      SplitOpCPUImpl<T, decltype(input_reshaped), 3>{}(
          context, input_reshaped, input_shape, split_dim, prefix_dim_size,
          split_dim_size, suffix_dim_size, make_sizes, reshape_result,
          num_split, split_dim_output_size);
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Partial specialization for GPU
template <typename T>
class SplitOpGPU : public SplitOpBase<GPUDevice, T> {
 public:
  typedef SplitOpBase<GPUDevice, T> Base;
  explicit SplitOpGPU(OpKernelConstruction* c) : Base(c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_9(mht_9_v, 469, "", "./tensorflow/core/kernels/split_op.cc", "SplitOpGPU");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_opDTcc mht_10(mht_10_v, 474, "", "./tensorflow/core/kernels/split_op.cc", "Compute");

    bool done = false;
    Base::ComputeEasyCases(context, &done);
    if (!context->status().ok() || done) {
      return;
    }
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32_t split_dim_orig = context->input(0).flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32_t num_split = Base::num_outputs();
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Split on GPU requires input size "
                                "< max int32"));
    int32_t prefix_dim_size;
    int32_t split_dim_size;
    int32_t suffix_dim_size;
    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<int32>(input_shape, split_dim);

    const int32_t split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    GpuDeviceArrayOnHost<T*> ptrs(context, num_split);
    OP_REQUIRES_OK(context, ptrs.Init());

    for (int i = 0; i < num_split; ++i) {
      Tensor* result = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &result));
      ptrs.Set(i, result->flat<T>().data());
    }
    if (prefix_dim_size * split_dim_output_size * suffix_dim_size == 0) {
      return;
    }
    OP_REQUIRES_OK(context, ptrs.Finalize());

    SplitOpGPULaunch<T>().Run(context->eigen_device<GPUDevice>(),
                              input.flat<T>().data(), prefix_dim_size,
                              split_dim_size, suffix_dim_size, ptrs.data());
    OP_REQUIRES(context, context->op_device_context()->stream()->ok(),
                errors::Internal("Launch of gpu kernel for SplitOp failed"));
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#define REGISTER_SPLIT(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpCPU<type>)

TF_CALL_ALL_TYPES(REGISTER_SPLIT);
REGISTER_SPLIT(quint8);

#undef REGISTER_SPLIT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpGPU<type>)

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // end namespace tensorflow
