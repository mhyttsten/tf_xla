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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc() {
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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRMulOp : public OpKernel {
 public:
  explicit CSRMulOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/sparse/mul_op.cc", "CSRMulOp");
}

  void Compute(OpKernelContext* ctx) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/sparse/mul_op.cc", "Compute");

    const CSRSparseMatrix* a_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    const Tensor& b_t = ctx->input(1);

    OP_REQUIRES(ctx, a_matrix->dtype() == b_t.dtype(),
                errors::InvalidArgument(
                    "Input types don't match.  a.dtype == ",
                    DataTypeString(a_matrix->dtype()),
                    " vs. b.dtype == ", DataTypeString(b_t.dtype())));

    const int b_rank = b_t.dims();

    const Tensor& a_dense_shape_t = a_matrix->dense_shape();
    auto a_dense_shape = a_dense_shape_t.vec<int64_t>();
    const int batch_size = a_dense_shape(0);
    if (b_rank == 3) {
      OP_REQUIRES(
          ctx,
          ((a_matrix->dims() == 3) && (b_t.dim_size(0) == batch_size) &&
           (b_t.NumElements() == batch_size)),
          errors::InvalidArgument(
              "If b is a rank-3 tensor, then a must be a rank 3 and the size "
              "of b be "
              "[batch_size, 1, 1].  But the shape of b is: ",
              b_t.shape().DebugString(),
              " and the shape of a is: ", a_dense_shape_t.DebugString()));
    } else {
      OP_REQUIRES(ctx, b_rank == 0,
                  errors::Unimplemented(
                      "Multiplying by a 2D+ dense tensor is not currently "
                      "supported, but shape of b is: ",
                      b_t.shape().DebugString()));
    }

    Tensor c_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    CSRSparseMatrix c_matrix;
    if (b_rank == 0) {
      auto b = b_t.scalar<T>();
      // TODO(ebrevdo): call other functor if b is nonscalar.
      functor::CSRSparseMatrixMulScalar<Device, T> csrmul_scalar;
      OP_REQUIRES_OK(ctx, csrmul_scalar.Compute(ctx, *a_matrix, b, &c_matrix));
    } else {
      // b_rank == 1 and a_matrix is rank-3.
      auto b = b_t.flat<T>();
      functor::CSRSparseMatrixBatchMulVec<Device, T> csrmul_batch_vec;
      OP_REQUIRES_OK(ctx,
                     csrmul_batch_vec.Compute(ctx, *a_matrix, b, &c_matrix));
    }
    c_t.scalar<Variant>()() = std::move(c_matrix);
    ctx->set_output(0, c_t);
  }
};

#define REGISTER(DEV, T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseMatrixMul").Device(DEVICE_##DEV).TypeConstraint<T>("T"), \
      CSRMulOp<DEV##Device, T>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(T) REGISTER(GPU, T)

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T>
class CSRSparseMatrixMulScalar<GPUDevice, T> {
 public:
  explicit CSRSparseMatrixMulScalar() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc mht_2(mht_2_v, 301, "", "./tensorflow/core/kernels/sparse/mul_op.cc", "CSRSparseMatrixMulScalar");
}

  Status Compute(OpKernelContext* ctx, const CSRSparseMatrix& a,
                 typename TTypes<T>::ConstScalar b, CSRSparseMatrix* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSmul_opDTcc mht_3(mht_3_v, 307, "", "./tensorflow/core/kernels/sparse/mul_op.cc", "Compute");

    const int total_nnz = a.total_nnz();
    Tensor c_values_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({total_nnz}), &c_values_t));
    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(),
        a.row_pointers(), a.col_indices(), c_values_t, c));

    auto a_values = a.values().flat<T>();
    auto c_values = c_values_t.flat<T>();

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    bool error;
    bool* const error_ptr = functor::mul<T>::has_errors ? &error : nullptr;

    // tensor * scalar
    functor::BinaryFunctor<GPUDevice, functor::mul<T>, 1>().Right(
        d, c_values, a_values, b, error_ptr);

    return Status::OK();
  }
};

#define DECLARE_GPU_SPEC(T)                                 \
  template <>                                               \
  Status CSRSparseMatrixBatchMulVec<GPUDevice, T>::Compute( \
      OpKernelContext* ctx, const CSRSparseMatrix& a,       \
      typename TTypes<T>::ConstFlat b, CSRSparseMatrix* c); \
  extern template struct CSRSparseMatrixBatchMulVec<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(std::complex<float>);
DECLARE_GPU_SPEC(std::complex<double>);

#undef DECLARE_GPU_SPEC

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
