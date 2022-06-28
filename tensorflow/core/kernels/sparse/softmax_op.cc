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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Implements the kernel for the CSRSoftmax op, which performs softmax
// along the innermost (col) dimension of a CSRSparseMatrix object
// stored in a DT_VARIANT.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRSoftmaxOp : public OpKernel {
 public:
  explicit CSRSoftmaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/sparse/softmax_op.cc", "CSRSoftmaxOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/sparse/softmax_op.cc", "Compute");

    const CSRSparseMatrix* logits_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &logits_matrix));
    OP_REQUIRES(
        ctx, logits_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of logits is not equal to 'type': ",
                                DataTypeString(logits_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    // Allocate output shapes
    const int total_nnz = logits_matrix->total_nnz();
    Tensor output_values_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({total_nnz}), &output_values_t));

    CSRSparseMatrix output_matrix;

    Tensor dense_shape_t = logits_matrix->dense_shape();

    OP_REQUIRES_OK(
        ctx,
        CSRSparseMatrix::CreateCSRSparseMatrix(
            DataTypeToEnum<T>::value, dense_shape_t,
            logits_matrix->batch_pointers(), logits_matrix->row_pointers(),
            logits_matrix->col_indices(), output_values_t, &output_matrix));

    if (total_nnz > 0) {
      functor::CSRSparseMatrixSoftmax<Device, T> softmax;
      OP_REQUIRES_OK(
          ctx, softmax(ctx, *logits_matrix, output_matrix.values().vec<T>()));
    }

    Tensor output_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    output_t.scalar<Variant>()() = std::move(output_matrix);
    ctx->set_output(0, output_t);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(DEV, T)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSoftmax")     \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRSoftmaxOp<DEV##Device, T>);

REGISTER(GPU, float)
REGISTER(GPU, double)

#undef REGISTER

namespace functor {
#define DECLARE_GPU_SPEC(T)                                \
  template <>                                              \
  Status CSRSparseMatrixSoftmax<GPUDevice, T>::operator()( \
      OpKernelContext* ctx, const CSRSparseMatrix& logits, \
      typename TTypes<T>::Vec softmax_values);             \
  extern template struct CSRSparseMatrixSoftmax<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class CSRSoftmaxGradOp : public OpKernel {
 public:
  explicit CSRSoftmaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc mht_2(mht_2_v, 293, "", "./tensorflow/core/kernels/sparse/softmax_op.cc", "CSRSoftmaxGradOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsoftmax_opDTcc mht_3(mht_3_v, 298, "", "./tensorflow/core/kernels/sparse/softmax_op.cc", "Compute");

    const CSRSparseMatrix* softmax_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &softmax_matrix));
    OP_REQUIRES(ctx, softmax_matrix->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of softmax is not equal to 'type': ",
                    DataTypeString(softmax_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));

    const CSRSparseMatrix* grad_softmax_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 1, &grad_softmax_matrix));
    OP_REQUIRES(ctx, grad_softmax_matrix->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of grad_softmax is not equal to 'type': ",
                    DataTypeString(grad_softmax_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));

    OP_REQUIRES(
        ctx, softmax_matrix->dims() == grad_softmax_matrix->dims(),
        errors::InvalidArgument(
            "Ranks of softmax and grad_softmax matrices differ: ",
            softmax_matrix->dims(), " vs. ", grad_softmax_matrix->dims()));

    OP_REQUIRES(
        ctx, softmax_matrix->dims() == grad_softmax_matrix->dims(),
        errors::InvalidArgument(
            "Ranks of softmax and grad_softmax matrices differ: ",
            softmax_matrix->dims(), " vs. ", grad_softmax_matrix->dims()));

    Tensor dense_shape_t = softmax_matrix->dense_shape();
    auto host_dense_shape =
        static_cast<const Tensor>(dense_shape_t).vec<int64_t>();

    auto host_grad_dense_shape =
        grad_softmax_matrix->dense_shape().vec<int64_t>();

    for (int i = 0; i < host_dense_shape.size(); ++i) {
      OP_REQUIRES(ctx, host_dense_shape(i) == host_grad_dense_shape(i),
                  errors::InvalidArgument(
                      "Shapes of softmax and grad_softmax matrices differ: ",
                      dense_shape_t.SummarizeValue(3), " vs. ",
                      grad_softmax_matrix->dense_shape().SummarizeValue(3)));
    }

    // Allocate output shapes.  Note that since the Softmax Gradient
    // tensor is the elementwise product of some function with the
    // softmax value, it will keep the sparsity structure of the softmax.
    const int total_nnz = softmax_matrix->total_nnz();
    Tensor gradient_values;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({total_nnz}), &gradient_values));

    CSRSparseMatrix gradient_matrix;

    OP_REQUIRES_OK(
        ctx,
        CSRSparseMatrix::CreateCSRSparseMatrix(
            DataTypeToEnum<T>::value, dense_shape_t,
            softmax_matrix->batch_pointers(), softmax_matrix->row_pointers(),
            softmax_matrix->col_indices(), gradient_values, &gradient_matrix));

    if (total_nnz > 0) {
      functor::CSRSparseMatrixSoftmaxGrad<Device, T> softmax_grad;
      OP_REQUIRES_OK(ctx,
                     softmax_grad(ctx, *softmax_matrix, *grad_softmax_matrix,
                                  gradient_matrix.values().vec<T>()));
    }

    Tensor gradient_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    gradient_t.scalar<Variant>()() = std::move(gradient_matrix);
    ctx->set_output(0, gradient_t);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(DEV, T)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSoftmaxGrad") \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRSoftmaxGradOp<DEV##Device, T>);

REGISTER(GPU, float)
REGISTER(GPU, double)

#undef REGISTER

namespace functor {
#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  Status CSRSparseMatrixSoftmaxGrad<GPUDevice, T>::operator()( \
      OpKernelContext* ctx, const CSRSparseMatrix& softmax,    \
      const CSRSparseMatrix& grad_softmax,                     \
      typename TTypes<T>::Vec gradient_values);                \
  extern template struct CSRSparseMatrixSoftmaxGrad<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
