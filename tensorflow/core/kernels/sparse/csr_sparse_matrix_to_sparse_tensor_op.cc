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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc() {
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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {
namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Validate that CSR SparseMatrix has the expected dtype and rank 2 or 3.
Status ValidateCSRSparseMatrix(const CSRSparseMatrix& csr_sparse_matrix,
                               DataType expected_dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "ValidateCSRSparseMatrix");

  if (csr_sparse_matrix.dtype() != expected_dtype) {
    return errors::InvalidArgument(
        "Expected a CSRSparseMatrix of type ", DataTypeString(expected_dtype),
        " but saw type: ", DataTypeString(csr_sparse_matrix.dtype()));
  }
  const int rank = csr_sparse_matrix.dense_shape().dim_size(0);
  if (rank != 2 && rank != 3) {
    return errors::InvalidArgument("CSR SparseMatrix must have rank 2 or 3; ",
                                   "but dense_shape has size ", rank);
  }
  return Status::OK();
}
}  // namespace

// Op to convert a (batched) CSR SparseMatrix to SparseTensors on the CPU.
// The resulting SparseTensor will have the same dense shape and non-zero values
// as the CSR SparseMatrix. rank 2 or (if batched) 3. Moreover, the resulting
// SparseTensor's indices will be present in the canonical, row-major ordering.
template <typename T>
class CSRSparseMatrixToSparseTensorCPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToSparseTensorCPUOp(OpKernelConstruction* c)
      : OpKernel(c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "CSRSparseMatrixToSparseTensorCPUOp");
}

  void Compute(OpKernelContext* c) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "Compute");

    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));
    OP_REQUIRES_OK(c, ValidateCSRSparseMatrix(*csr_sparse_matrix,
                                              DataTypeToEnum<T>::value));

    // Copy the SparseTensor's dense_shape and values from the CSRSparseMatrix.
    c->set_output(1, csr_sparse_matrix->values());
    const Tensor& dense_shape = csr_sparse_matrix->dense_shape();
    c->set_output(2, dense_shape);

    const int batch_size = csr_sparse_matrix->batch_size();
    const int64_t total_nnz = csr_sparse_matrix->total_nnz();
    const int rank = csr_sparse_matrix->dense_shape().dim_size(0);
    auto dense_shape_vec = dense_shape.vec<int64_t>();
    const int64_t num_rows = dense_shape_vec((rank == 2) ? 0 : 1);

    Tensor* indices;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, TensorShape({total_nnz, rank}), &indices));
    auto indices_flat = indices->template flat<int64_t>();

    auto csr_row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto csr_col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();

    // Process the individual batches in parallel using a threadpool.
    auto shard = [&](int64_t batch_begin, int64_t batch_end) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "lambda");

      for (int64_t batch_idx = batch_begin; batch_idx < batch_end;
           ++batch_idx) {
        const int64_t csr_batch_offset = batch_ptrs(batch_idx);

        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
          const int64_t row_offset = batch_idx * (num_rows + 1) + row_idx;

          // The column indices of the current row lie in the range:
          //  [csr_row_ptr[row_offset], csr_row_ptr[row_offset + 1])
          const int64_t col_begin = csr_row_ptr(row_offset);
          const int64_t col_end = csr_row_ptr(row_offset + 1);
          for (int64_t i = col_begin; i < col_end; ++i) {
            const int64_t col_idx = csr_col_ind(csr_batch_offset + i);
            const int64_t indices_offset = rank * (csr_batch_offset + i);

            if (rank == 2) {
              indices_flat(indices_offset) = row_idx;
              indices_flat(indices_offset + 1) = col_idx;
            } else {  // rank == 3
              indices_flat(indices_offset) = batch_idx;
              indices_flat(indices_offset + 1) = row_idx;
              indices_flat(indices_offset + 2) = col_idx;
            }
          }
        }
      }
    };
    auto worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
    // TODO(anudhyan): Estimate the cost per unit based on Eigen::TensorOpCost
    // units and scale based on benchmarks.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          csr_sparse_matrix->total_nnz() / batch_size /* cost per unit */,
          shard);
  }
};

template <typename Device, typename T>
class CSRSparseMatrixToSparseTensorGPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToSparseTensorGPUOp(OpKernelConstruction* c)
      : OpKernel(c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_4(mht_4_v, 321, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "CSRSparseMatrixToSparseTensorGPUOp");
}

  void Compute(OpKernelContext* c) final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePScsr_sparse_matrix_to_sparse_tensor_opDTcc mht_5(mht_5_v, 326, "", "./tensorflow/core/kernels/sparse/csr_sparse_matrix_to_sparse_tensor_op.cc", "Compute");

    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));
    OP_REQUIRES_OK(c, ValidateCSRSparseMatrix(*csr_sparse_matrix,
                                              DataTypeToEnum<T>::value));

    const Tensor& dense_shape_t = csr_sparse_matrix->dense_shape();
    c->set_output(2, dense_shape_t);
    const int rank = dense_shape_t.dim_size(0);
    const int batch_size = csr_sparse_matrix->batch_size();
    const int64_t total_nnz = csr_sparse_matrix->total_nnz();

    auto dense_shape = dense_shape_t.vec<int64_t>();
    const int64_t rows = dense_shape((rank == 2) ? 0 : 1);

    Tensor* indices_t;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, TensorShape({total_nnz, rank}), &indices_t));

    Tensor* values_t;
    OP_REQUIRES_OK(c,
                   c->allocate_output(1, TensorShape({total_nnz}), &values_t));

    functor::CSRSparseMatrixToCOOSparseMatrix<Device> csr_to_coo;
    auto indices = indices_t->matrix<int64_t>();

    auto csr_row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto coo_col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();

    Tensor coo_row_ind_t;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                       &coo_row_ind_t));
    auto coo_row_ind = coo_row_ind_t.vec<int32>();

    // TODO(ebrevdo): Convert to one or two single kernel calls,
    // where the kernels are batch-friendly.
    for (int i = 0; i < batch_size; ++i) {
      const int nnz_i = csr_sparse_matrix->nnz(i);
      if (nnz_i == 0) {
        // No copying required.  Avoid failure case below.
        continue;
      }
      const TTypes<int32>::UnalignedConstVec csr_row_ptr_i(
          &csr_row_ptr((rows + 1) * i), rows + 1);
      const TTypes<int32>::UnalignedVec coo_row_ind_i(
          &coo_row_ind(csr_sparse_matrix->batch_offset(i)), nnz_i);
      OP_REQUIRES_OK(c, csr_to_coo(c, csr_row_ptr_i, coo_row_ind_i));
    }

    if (total_nnz > 0) {
      functor::COOSparseMatrixToSparseTensor<Device> coo_to_st;
      OP_REQUIRES_OK(c, coo_to_st(c, dense_shape, batch_ptrs, coo_row_ind,
                                  coo_col_ind, indices));
    }

    *values_t = csr_sparse_matrix->values();
  }
};

#define REGISTER_GPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToSparseTensor") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("type")        \
                              .HostMemory("dense_shape"),       \
                          CSRSparseMatrixToSparseTensorGPUOp<GPUDevice, T>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
template <>
struct COOSparseMatrixToSparseTensor<GPUDevice> {
  Status operator()(OpKernelContext* ctx,
                    TTypes<int64_t>::ConstVec host_dense_shape,
                    TTypes<int>::ConstVec host_batch_ptrs,
                    TTypes<int>::Vec coo_row_ind,
                    TTypes<int>::ConstVec coo_col_ind,
                    TTypes<int64_t>::Matrix indices);
};
extern template struct COOSparseMatrixToSparseTensor<GPUDevice>;

template <>
struct CSRSparseMatrixToCOOSparseMatrix<GPUDevice> {
  Status operator()(OpKernelContext* c,
                    TTypes<const int>::UnalignedVec csr_row_ptr,
                    TTypes<int>::UnalignedVec coo_row_ind);
};
extern template struct CSRSparseMatrixToCOOSparseMatrix<GPUDevice>;

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToSparseTensor") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<T>("type"),       \
                          CSRSparseMatrixToSparseTensorCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

}  // namespace tensorflow
