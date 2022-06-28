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
class MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc() {
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

// See docs in ../ops/math_ops.cc.

#include "tensorflow/core/platform/errors.h"
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bincount_op.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {

using thread::ThreadPool;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Tidx, typename T>
struct BincountFunctor<CPUDevice, Tidx, T, true> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(context->eigen_cpu_device()) =
        (arr >= Tidx(0)).all();
    if (!all_nonneg_t.scalar<bool>()()) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64_t num_threads = thread_pool->NumThreads() + 1;
    Tensor partial_bins_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({num_threads, num_bins}), &partial_bins_t));
    auto partial_bins = partial_bins_t.matrix<bool>();
    partial_bins.setZero();
    thread_pool->ParallelForWithWorkerId(
        arr.size(), 8 /* cost */,
        [&](int64_t start_ind, int64_t limit_ind, int64_t worker_id) {
          for (int64_t i = start_ind; i < limit_ind; i++) {
            Tidx value = arr(i);
            if (value < num_bins) {
              partial_bins(worker_id, value) = true;
            }
          }
        });

    // Sum the partial bins along the 0th axis.
    Eigen::array<int, 1> reduce_dim({0});
    output.device(context->eigen_cpu_device()) =
        partial_bins.any(reduce_dim).cast<T>();
    return Status::OK();
  }
};

template <typename Tidx, typename T>
struct BincountFunctor<CPUDevice, Tidx, T, false> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(context->eigen_cpu_device()) =
        (arr >= Tidx(0)).all();
    if (!all_nonneg_t.scalar<bool>()()) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64_t num_threads = thread_pool->NumThreads() + 1;
    const Tidx* arr_data = arr.data();
    const std::ptrdiff_t arr_size = arr.size();
    const T* weight_data = weights.data();
    if (weights.size() && weights.size() != arr_size) {
      return errors::InvalidArgument(
          "Input indices and weights must have the same size.");
    }
    if (num_threads == 1) {
      output.setZero();
      T* output_data = output.data();
      if (weights.size()) {
        for (int64_t i = 0; i < arr_size; i++) {
          const Tidx value = arr_data[i];
          if (value < num_bins) {
            output_data[value] += weight_data[i];
          }
        }
      } else {
        for (int64_t i = 0; i < arr_size; i++) {
          const Tidx value = arr_data[i];
          if (value < num_bins) {
            // Complex numbers don't support "++".
            output_data[value] += T(1);
          }
        }
      }
    } else {
      Tensor partial_bins_t;
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<T>::value, TensorShape({num_threads, num_bins}),
          &partial_bins_t));
      auto partial_bins = partial_bins_t.matrix<T>();
      partial_bins.setZero();
      thread_pool->ParallelForWithWorkerId(
          arr_size, 8 /* cost */,
          [&](int64_t start_ind, int64_t limit_ind, int64_t worker_id) {
            if (weights.size()) {
              for (int64_t i = start_ind; i < limit_ind; i++) {
                Tidx value = arr_data[i];
                if (value < num_bins) {
                  partial_bins(worker_id, value) += weight_data[i];
                }
              }
            } else {
              for (int64_t i = start_ind; i < limit_ind; i++) {
                Tidx value = arr_data[i];
                if (value < num_bins) {
                  // Complex numbers don't support "++".
                  partial_bins(worker_id, value) += T(1);
                }
              }
            }
          });

      // Sum the partial bins along the 0th axis.
      Eigen::array<int, 1> reduce_dim({0});
      output.device(context->eigen_cpu_device()) = partial_bins.sum(reduce_dim);
    }
    return Status::OK();
  }
};

template <typename Tidx, typename T, bool binary_output>
struct BincountReduceFunctor<CPUDevice, Tidx, T, binary_output> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 2>::ConstTensor& in,
                        const typename TTypes<T, 2>::ConstTensor& weights,
                        typename TTypes<T, 2>::Tensor& out,
                        const Tidx num_bins) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_2(mht_2_v, 348, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    const int num_rows = out.dimension(0);
    const int num_cols = in.dimension(1);
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelForWithWorkerId(
        num_rows, 8 /* cost */,
        [&](int64_t start_row, int64_t end_row, int64_t worker_id) {
          for (int64_t i = start_row; i < end_row; ++i) {
            for (int64_t j = 0; j < num_cols; ++j) {
              Tidx value = in(i, j);
              if (value < num_bins) {
                if (binary_output) {
                  out(i, value) = T(1);
                } else {
                  if (weights.size()) {
                    out(i, value) += weights(i, j);
                  } else {
                    out(i, value) += T(1);
                  }
                }
              }
            }
          }
        });
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T>
class BincountOp : public OpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_3(mht_3_v, 385, "", "./tensorflow/core/kernels/bincount_op.cc", "BincountOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_4(mht_4_v, 390, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    const Tensor& arr_t = ctx->input(0);
    const Tensor& size_tensor = ctx->input(1);
    OP_REQUIRES(ctx, size_tensor.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_tensor.dims()));
    int32_t size = size_tensor.scalar<int32_t>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    const Tensor& weights_t = ctx->input(2);
    const auto arr = arr_t.flat<int32_t>();
    const auto weights = weights_t.flat<T>();
    Tensor* output_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({size}), &output_t));
    auto output = output_t->flat<T>();
    OP_REQUIRES_OK(ctx,
                   functor::BincountFunctor<Device, int32_t, T, false>::Compute(
                       ctx, arr, weights, output, size));
  }
};

#define REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Bincount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BincountOp<CPUDevice, type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Bincount")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          BincountOp<GPUDevice, type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename Tidx, typename T>
class DenseBincountOp : public OpKernel {
 public:
  explicit DenseBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_5(mht_5_v, 443, "", "./tensorflow/core/kernels/bincount_op.cc", "DenseBincountOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(
          ctx, !OpDeterminismRequired(),
          errors::Unimplemented(
              "Determinism is not yet supported in GPU implementation of "
              "DenseBincount."));
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_6(mht_6_v, 457, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    const Tensor& data = ctx->input(0);
    OP_REQUIRES(ctx, data.dims() <= 2,
                errors::InvalidArgument(
                    "Shape must be at most rank 2 but is rank ", data.dims()));

    const Tensor& size_t = ctx->input(1);
    const Tensor& weights = ctx->input(2);

    OP_REQUIRES(ctx, size_t.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_t.dims()));
    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    Tensor* out_t;
    functor::SetZeroFunctor<Device, T> fill;
    if (data.dims() == 1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({size}), &out_t));
      auto out = out_t->flat<T>();
      fill(ctx->eigen_device<Device>(), out);
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      } else {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, false>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      }
    } else if (data.dims() == 2) {
      const int64_t num_rows = data.dim_size(0);
      auto weight_matrix =
          (weights.NumElements() == 0)
              ? weights.shaped<T, 2>(gtl::InlinedVector<int64_t, 2>(2, 0))
              : weights.matrix<T>();
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
      auto out = out_t->matrix<T>();
      fill(ctx->eigen_device<Device>(), out_t->flat<T>());
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountReduceFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      } else {
        OP_REQUIRES_OK(
            ctx,
            functor::BincountReduceFunctor<Device, Tidx, T, false>::Compute(
                ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("DenseBincount")              \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          DenseBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64_t, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("DenseBincount")              \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("size")            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          DenseBincountOp<GPUDevice, Tidx, T>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64_t, T);

TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_float(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename Tidx, typename T>
class SparseBincountOp : public OpKernel {
 public:
  explicit SparseBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_7(mht_7_v, 556, "", "./tensorflow/core/kernels/bincount_op.cc", "SparseBincountOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_8(mht_8_v, 563, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    const Tensor& indices = ctx->input(0);
    const auto values = ctx->input(1).flat<Tidx>();
    const Tensor& dense_shape = ctx->input(2);
    const Tensor& size_t = ctx->input(3);
    const auto weights = ctx->input(4).flat<T>();
    const int64_t weights_size = weights.size();

    OP_REQUIRES(ctx, size_t.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_t.dims()));
    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    bool is_1d = dense_shape.NumElements() == 1;

    Tensor* out_t;
    functor::SetZeroFunctor<Device, T> fill;
    if (is_1d) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({size}), &out_t));
      auto out = out_t->flat<T>();
      fill(ctx->eigen_device<Device>(), out);
      if (binary_output_) {
        OP_REQUIRES_OK(ctx,
                       functor::BincountFunctor<Device, Tidx, T, true>::Compute(
                           ctx, values, weights, out, size));
      } else {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, false>::Compute(
                     ctx, values, weights, out, size));
      }
    } else {
      const auto shape = dense_shape.flat<int64_t>();
      const int64_t num_rows = shape(0);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
      const auto out = out_t->matrix<T>();
      fill(ctx->eigen_device<Device>(), out_t->flat<T>());
      const auto indices_mat = indices.matrix<int64_t>();
      for (int64_t i = 0; i < indices_mat.dimension(0); ++i) {
        const int64_t batch = indices_mat(i, 0);
        const Tidx bin = values(i);
        OP_REQUIRES(
            ctx, batch < out.dimension(0),
            errors::InvalidArgument("Index out of bound. `batch` (", batch,
                                    ") must be less than the dimension size (",
                                    out.dimension(0), ")."));
        OP_REQUIRES(
            ctx, bin < out.dimension(1),
            errors::InvalidArgument("Index out ouf bound. `bin` (", bin,
                                    ") must be less then the dimension size (",
                                    out.dimension(1), ")."));
        if (bin < size) {
          if (binary_output_) {
            out(batch, bin) = T(1);
          } else {
            if (weights_size) {
              out(batch, bin) += weights(i);
            } else {
              out(batch, bin) += T(1);
            }
          }
        }
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseBincount")             \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          SparseBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64_t, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename Tidx, typename T>
class RaggedBincountOp : public OpKernel {
 public:
  explicit RaggedBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_9(mht_9_v, 656, "", "./tensorflow/core/kernels/bincount_op.cc", "RaggedBincountOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_opDTcc mht_10(mht_10_v, 663, "", "./tensorflow/core/kernels/bincount_op.cc", "Compute");

    const auto splits = ctx->input(0).flat<int64_t>();
    const auto values = ctx->input(1).flat<Tidx>();
    const Tensor& size_t = ctx->input(2);
    const auto weights = ctx->input(3).flat<T>();
    const int64_t weights_size = weights.size();

    OP_REQUIRES(ctx, size_t.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_t.dims()));
    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    int num_rows = splits.size() - 1;
    int num_values = values.size();
    int batch_idx = 0;

    OP_REQUIRES(ctx, splits(0) == 0,
                errors::InvalidArgument("Splits must start with 0, not with ",
                                        splits(0)));

    OP_REQUIRES(ctx, splits(num_rows) == num_values,
                errors::InvalidArgument(
                    "Splits must end with the number of values, got ",
                    splits(num_rows), " instead of ", num_values));

    Tensor* out_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
    functor::SetZeroFunctor<Device, T> fill;
    fill(ctx->eigen_device<Device>(), out_t->flat<T>());
    const auto out = out_t->matrix<T>();

    for (int idx = 0; idx < num_values; ++idx) {
      while (idx >= splits(batch_idx)) {
        batch_idx++;
      }
      Tidx bin = values(idx);
      OP_REQUIRES(ctx, bin >= 0,
                  errors::InvalidArgument("Input must be non-negative"));
      if (bin < size) {
        if (binary_output_) {
          out(batch_idx - 1, bin) = T(1);
        } else {
          T value = (weights_size > 0) ? weights(idx) : T(1);
          out(batch_idx - 1, bin) += value;
        }
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("RaggedBincount")             \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          RaggedBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64_t, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // end namespace tensorflow
