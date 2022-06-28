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
class MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/where_op.h"

#include <memory>
#include <numeric>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/util/gpu_solvers.h"
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // TENSORFLOW_USE_ROCM
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {
template <typename T>
int64_t CountAccumulator(const T* begin, const T* end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/kernels/where_op.cc", "CountAccumulator");

  return std::accumulate(begin, end, 0LL, [](int64_t accum, const T& val) {
    return accum + (val != T(0));
  });
}

template <>
int64_t CountAccumulator<bool>(const bool* begin, const bool* end) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/kernels/where_op.cc", "CountAccumulator<bool>");

  return std::accumulate(begin, end, 0LL);
}

}  // namespace

template <typename T>
struct NumTrue<CPUDevice, T, int64_t> {
  static Status Compute(OpKernelContext* ctx, const CPUDevice& d,
                        typename TTypes<T>::ConstFlat input,
                        TTypes<int64_t>::UnalignedScalar num_true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/kernels/where_op.cc", "Compute");

    num_true() = CountAccumulator<T>(input.data(), input.data() + input.size());
    return Status::OK();
  }
};

template <int DIMS, typename T, typename TIndex>
struct Where<CPUDevice, DIMS, T, TIndex> {
  EIGEN_ALWAYS_INLINE static void WriteIndexRowMajor(
      typename TTypes<int64_t>::Matrix output,
      const typename Eigen::DSizes<TIndex, DIMS>& strides, TIndex true_n,
      TIndex index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/kernels/where_op.cc", "WriteIndexRowMajor");

    for (int i = 0; i < DIMS; ++i) {
      output(true_n, i) = index / strides[i];
      index -= output(true_n, i) * strides[i];
    }
  }

  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const CPUDevice& d,
      typename TTypes<T, DIMS>::ConstTensor input,
      typename TTypes<int64_t>::Matrix output, TIndex* found_true) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/kernels/where_op.cc", "Compute");

    Eigen::DSizes<Eigen::DenseIndex, DIMS> dims = input.dimensions();
    Eigen::DSizes<TIndex, DIMS> strides;

    EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                         static_cast<int>(Eigen::RowMajor)),
                        INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);

    strides[DIMS - 1] = 1;
    for (int i = DIMS - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }

    Eigen::DenseIndex output_size = output.dimension(0);
    for (Eigen::DenseIndex n = 0; n < input.size(); ++n) {
      if (input.data()[n] != T(0)) {
        if (FastBoundsCheck(*found_true, output_size)) {
          WriteIndexRowMajor(output, strides, *found_true, n);
        }
        ++*found_true;
      }
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename T>
class WhereCPUOp : public OpKernel {
 public:
  explicit WhereCPUOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_5(mht_5_v, 314, "", "./tensorflow/core/kernels/where_op.cc", "WhereCPUOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_6(mht_6_v, 319, "", "./tensorflow/core/kernels/where_op.cc", "Compute");

    const Tensor& input = context->input(0);

    OP_REQUIRES(
        context, input.dtype() != DT_HALF,
        errors::Unimplemented("No WhereOp available for float16/half type on "
                              "CPU; dying in CPU WhereOp to avoid silently "
                              "creating costly copies from device."));

    const int input_dims = input.dims();

    int64_t num_true;
    TTypes<int64_t>::UnalignedScalar num_true_t(&num_true);

    Status s = functor::NumTrue<CPUDevice, T, int64_t>::Compute(
        context, context->eigen_device<CPUDevice>(), input.flat<T>(),
        num_true_t);
    OP_REQUIRES_OK(context, s);
    TensorShape output_shape({num_true, input_dims});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // TODO(ebrevdo): Replace single-threaded copy with a multithreaded block
    // copy by getting block counts above instead of a global NumTrue, then
    // having each block filled in separate threads below.
    int64_t found_true = 0;

#define HANDLE_DIM(NDIM)                                                      \
  case NDIM: {                                                                \
    Status s = functor::Where<CPUDevice, NDIM, T, int64_t>::Compute(          \
        context, context->eigen_device<CPUDevice>(), input.tensor<T, NDIM>(), \
        output->matrix<int64_t>(), &found_true);                              \
    OP_REQUIRES_OK(context, s);                                               \
  } break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp : Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM

    OP_REQUIRES(
        context, found_true == num_true_t(),
        errors::InvalidArgument(
            "WhereOp: Race condition between counting the number of true "
            "elements and writing them.  When counting, saw ",
            num_true_t(), " elements; but when writing their indices, saw ",
            found_true, " elements."));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhereCPUOp);
};

#define REGISTER_WHERE_OP(T) \
  REGISTER_KERNEL_BUILDER(   \
      Name("Where").Device(DEVICE_CPU).TypeConstraint<T>("T"), WhereCPUOp<T>);

TF_CALL_NUMBER_TYPES(REGISTER_WHERE_OP);
TF_CALL_bool(REGISTER_WHERE_OP);

#undef REGISTER_WHERE_OP

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

#define DECLARE_GPU_NUMTRUE(T, Tindex)                                      \
  template <>                                                               \
  Status NumTrue<GPUDevice, T, Tindex>::Compute(                            \
      OpKernelContext* ctx, const GPUDevice& d, TTypes<T>::ConstFlat input, \
      TTypes<Tindex>::UnalignedScalar num_true);                            \
  extern template struct NumTrue<GPUDevice, T, Tindex>

#define DECLARE_GPU_NUMTRUE_TYPE(T) \
  DECLARE_GPU_NUMTRUE(T, int32);    \
  DECLARE_GPU_NUMTRUE(T, int64_t);

TF_CALL_NUMBER_TYPES(DECLARE_GPU_NUMTRUE_TYPE);
TF_CALL_bool(DECLARE_GPU_NUMTRUE_TYPE);

#undef DECLARE_GPU_NUMTRUE_TYPE
#undef DECLARE_GPU_NUMTRUE

#define DECLARE_GPU_WHERE_INDEX(Dims, T, Tindex)                    \
  template <>                                                       \
  Status Where<GPUDevice, Dims, T, Tindex>::Compute(                \
      OpKernelContext* ctx, const GPUDevice& d,                     \
      typename TTypes<T, Dims>::ConstTensor input,                  \
      typename TTypes<int64_t>::Matrix output, Tindex* found_true); \
  extern template struct Where<GPUDevice, Dims, T, Tindex>;
#define DECLARE_GPU_WHERE(Dims, T)         \
  DECLARE_GPU_WHERE_INDEX(Dims, T, int32); \
  DECLARE_GPU_WHERE_INDEX(Dims, T, int64_t);

#define DECLARE_GPU_WHERE_TYPES(T) \
  DECLARE_GPU_WHERE(1, T);         \
  DECLARE_GPU_WHERE(2, T);         \
  DECLARE_GPU_WHERE(3, T);         \
  DECLARE_GPU_WHERE(4, T);         \
  DECLARE_GPU_WHERE(5, T);         \
  DECLARE_GPU_WHERE(6, T);         \
  DECLARE_GPU_WHERE(7, T);         \
  DECLARE_GPU_WHERE(8, T);

TF_CALL_WHERE_GPU_TYPES(DECLARE_GPU_WHERE_TYPES);

#undef DECLARE_GPU_WHERE_TYPES
#undef DECLARE_GPU_WHERE
#undef DECLARE_GPU_WHERE_INDEX

}  // namespace functor

template <typename T>
class WhereGPUOp : public AsyncOpKernel {
 public:
  explicit WhereGPUOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_7(mht_7_v, 449, "", "./tensorflow/core/kernels/where_op.cc", "WhereGPUOp");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_8(mht_8_v, 454, "", "./tensorflow/core/kernels/where_op.cc", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    if (input.NumElements() < std::numeric_limits<int32>::max()) {
      ComputeAsyncType<int32>(input, input_dims, context, done);
    } else {
      ComputeAsyncType<int64_t>(input, input_dims, context, done);
    }
  }

  template <typename Tindex>
  void ComputeAsyncType(const Tensor& input, const int input_dims,
                        OpKernelContext* context, DoneCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_9(mht_9_v, 470, "", "./tensorflow/core/kernels/where_op.cc", "ComputeAsyncType");

    // Step 0: alloc nnz
    // Step 1: call nnz kernel
    // Step 2: call create_output
    // Step 3: call where kernel

    // Allocate pinned memory for `num_true`.  This memory is accessible on host
    // and device.
    ScratchSpace<Tindex> num_true(context, 1, /*on_host=*/true);
    typename TTypes<Tindex>::UnalignedScalar num_true_t(
        num_true.mutable_data());

    // Push kernel to stream to get number of true elements.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Status s = functor::NumTrue<GPUDevice, T, Tindex>::Compute(
        context, d, input.flat<T>(), num_true_t);
    OP_REQUIRES_OK_ASYNC(context, s, done);

    auto create_and_check_output = [context, &d, &input, input_dims,
                                    num_true = std::move(num_true), done]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_opDTcc mht_10(mht_10_v, 492, "", "./tensorflow/core/kernels/where_op.cc", "lambda");

      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      // TODO(ebrevdo): Properly copy back found_true value to CPU for
      // validation checking.  Currently Where<GPUDevice>::Compute()
      // does not perform this copy back to CPU.
      Tindex found_true = -1;

      // Step 1: Allocate the output and perform the selection/copy.
      Tensor* output;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_output(
              0, TensorShape({*num_true.data(), input_dims}), &output),
          done);

#define HANDLE_DIM(NDIM)                                                \
  case NDIM: {                                                          \
    Status s = functor::Where<GPUDevice, NDIM, T, Tindex>::Compute(     \
        context, d, input.tensor<T, NDIM>(), output->matrix<int64_t>(), \
        &found_true);                                                   \
    OP_REQUIRES_OK_ASYNC(context, s, done);                             \
  } break;

      switch (input_dims) {
        HANDLE_DIM(1);
        HANDLE_DIM(2);
        HANDLE_DIM(3);
        HANDLE_DIM(4);
        HANDLE_DIM(5);
        HANDLE_DIM(6);
        HANDLE_DIM(7);
        HANDLE_DIM(8);

        default:
          OP_REQUIRES_ASYNC(
              context, false,
              errors::InvalidArgument("WhereOp: Unhandled input dimensions: ",
                                      input_dims),
              done);
      }
#undef HANDLE_DIM

      // TODO(ebrevdo): Fix the copy back to host.

      // OP_REQUIRES_ASYNC(
      //     context, found_true == num_true,
      //     errors::InvalidArgument(
      //         "WhereOp: Race condition between counting the number of true "
      //         "elements and writing them.  When counting, saw ",
      //         num_true, " elements; but when writing their indices, saw ",
      //         found_true, " elements."),
      //     done);

      done();
    };

    auto stream = context->op_device_context()->stream();
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, create_and_check_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhereGPUOp);
};

#define REGISTER_GPU_WHERE_OP(T) \
  REGISTER_KERNEL_BUILDER(       \
      Name("Where").Device(DEVICE_GPU).TypeConstraint<T>("T"), WhereGPUOp<T>);

TF_CALL_WHERE_GPU_TYPES(REGISTER_GPU_WHERE_OP);
#undef REGISTER_GPU_WHERE_OP

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_KERNEL_BUILDER(Name("Where")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("index"),
                        WhereCPUOp<int32>);

}  // namespace tensorflow
