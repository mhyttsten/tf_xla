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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc() {
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

// Implements the kernel for the CSRTranspose op, which transposes the
// two innermost dimensions of a CSRSparseMatrix object stored in a
// DT_VARIANT.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#define EIGEN_USE_GPU
#endif

#include <numeric>

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/kernels/sparse/transpose_op.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
Status ValidateTransposeInputs(const ConstCSRComponent<T>& input,
                               const CSRComponent<T>& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/kernels/sparse/transpose_op.cc", "ValidateTransposeInputs");

  const int rank = input.dense_shape_host.size();
  const int64_t nnz = input.col_ind.size();
  const int num_rows = input.row_ptr.size() - 1;
  const int num_cols = input.dense_shape_host(rank - 1);

  if (nnz != input.values.size()) {
    return errors::InvalidArgument(
        "Input nnz should equal the input values size. Got ", nnz, " vs. ",
        input.values.size());
  }
  if (num_cols + 1 != output.row_ptr.size()) {
    return errors::InvalidArgument(
        "Input num_cols should be equal to output num_rows. Got ", num_cols,
        " vs. ", output.row_ptr.size());
  }
  if (rank != output.dense_shape_host.size()) {
    return errors::InvalidArgument(
        "Input rank should be equal to the output rank. Got ", rank, " vs. ",
        output.dense_shape_host.size());
  }
  if (num_rows != output.dense_shape_host(rank - 1)) {
    return errors::InvalidArgument(
        "Input num_rows should be equal to the output num_cols. Got ", num_rows,
        " vs. ", output.dense_shape_host(rank - 1));
  }
  if (nnz != output.col_ind.size()) {
    return errors::InvalidArgument(
        "Input nnz should equal the output col_ind size. Got ", nnz, " vs. ",
        output.col_ind.size());
  }
  if (nnz != output.values.size()) {
    return errors::InvalidArgument(
        "Input nnz should equal the output values size. Got ", nnz, " vs. ",
        output.values.size());
  }
  return Status::OK();
}
}  // namespace

template <typename Device, typename T>
class CSRTransposeOp : public OpKernel {
 public:
  explicit CSRTransposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc mht_1(mht_1_v, 269, "", "./tensorflow/core/kernels/sparse/transpose_op.cc", "CSRTransposeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("conjugate", &conjugate_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePStranspose_opDTcc mht_2(mht_2_v, 276, "", "./tensorflow/core/kernels/sparse/transpose_op.cc", "Compute");

    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));
    OP_REQUIRES(
        ctx, input_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of input is not equal to 'type': ",
                                DataTypeString(input_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    // Allocate output shapes
    functor::CSRSparseMatrixTranspose<Device, T> transpose;
    CSRSparseMatrix output_matrix;
    OP_REQUIRES_OK(ctx,
                   transpose(ctx, conjugate_, *input_matrix, &output_matrix));
    Tensor output_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    output_t.scalar<Variant>()() = std::move(output_matrix);
    ctx->set_output(0, output_t);
  }

 private:
  bool conjugate_;
};

#define REGISTER_TRANSPOSE(DEV, T)                        \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixTranspose")   \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRTransposeOp<DEV##Device, T>);

REGISTER_TRANSPOSE(CPU, float)
REGISTER_TRANSPOSE(CPU, double)
REGISTER_TRANSPOSE(CPU, complex64)
REGISTER_TRANSPOSE(CPU, complex128)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_TRANSPOSE(GPU, float)
REGISTER_TRANSPOSE(GPU, double)
REGISTER_TRANSPOSE(GPU, complex64)
REGISTER_TRANSPOSE(GPU, complex128)
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_TRANSPOSE

namespace functor {

template <typename Device, typename T>
Status CSRSparseMatrixTranspose<Device, T>::operator()(
    OpKernelContext* ctx, bool conjugate, const CSRSparseMatrix& input_matrix,
    CSRSparseMatrix* output_matrix) {
  const int rank = input_matrix.dims();
  Tensor output_dense_shape_t(cpu_allocator(), DT_INT64, TensorShape({rank}));
  const Tensor& input_dense_shape_t = input_matrix.dense_shape();
  auto input_dense_shape = input_dense_shape_t.vec<int64_t>();
  auto output_dense_shape = output_dense_shape_t.vec<int64_t>();
  const int64_t batch_size = input_matrix.batch_size();
  if (rank == 3) {
    output_dense_shape(0) = batch_size;
  }
  output_dense_shape(rank - 2) = input_dense_shape(rank - 1);
  output_dense_shape(rank - 1) = input_dense_shape(rank - 2);
  const int64_t output_rows = output_dense_shape(rank - 2);

  // nnzs per batch do not change with matrix transposition.
  Tensor batch_ptr_t = input_matrix.batch_pointers();
  const int total_nnz = input_matrix.total_nnz();

  Tensor output_row_ptr_t;
  Tensor output_col_ind_t;
  Tensor output_values_t;

  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT32, TensorShape({batch_size * (output_rows + 1)}),
      &output_row_ptr_t));
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                        &output_col_ind_t));
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DataTypeToEnum<T>::value, TensorShape({total_nnz}), &output_values_t));

  TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
      DataTypeToEnum<T>::value, output_dense_shape_t, batch_ptr_t,
      output_row_ptr_t, output_col_ind_t, output_values_t, output_matrix));

  // Set the output row pointers to zero, in case we hit any empty
  // input batches.
  functor::SetZeroFunctor<Device, int32> set_zero;
  const Device& d = ctx->eigen_device<Device>();
  set_zero(d, output_row_ptr_t.flat<int32>());

  functor::CSRSparseMatrixTransposeComponent<Device, T> transpose_component;
  for (int i = 0; i < batch_size; ++i) {
    if (output_matrix->nnz(i) == 0) {
      continue;
    }
    ConstCSRComponent<T> input_comp{
        input_matrix.row_pointers_vec(i), input_matrix.col_indices_vec(i),
        input_matrix.values_vec<T>(i), input_dense_shape};
    CSRComponent<T> output_comp{
        output_matrix->row_pointers_vec(i), output_matrix->col_indices_vec(i),
        output_matrix->values_vec<T>(i), output_dense_shape};

    TF_RETURN_IF_ERROR(transpose_component(ctx, input_comp, &output_comp));
  }
  if (conjugate) {
    // conjugate all values with a single kernel launch.
    maybe_conj_inplace<Device, T>::run(d, &output_values_t);
  }

  return Status::OK();
}

// CPU kernel for transposing a single component of a CSR SparseMatrix.
template <typename T>
struct CSRSparseMatrixTransposeComponent<CPUDevice, T> {
  using SparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;

  Status operator()(OpKernelContext* ctx, const ConstCSRComponent<T>& input,
                    CSRComponent<T>* output) {
    TF_RETURN_IF_ERROR(ValidateTransposeInputs(input, *output));

    const int rank = input.dense_shape_host.size();
    const int num_rows = input.row_ptr.size() - 1;
    const int num_cols = input.dense_shape_host(rank - 1);
    const int64_t nnz = input.col_ind.size();

    // Compute the column counts; whose prefix sums make up the output row
    // pointers.
    for (int64_t i = 0; i < nnz; ++i) {
      output->row_ptr(input.col_ind(i) + 1) += 1;
    }
    std::partial_sum(output->row_ptr.data(),
                     output->row_ptr.data() + num_cols + 1,
                     output->row_ptr.data());

    // Iterate through each row of the input, and place each non-zero element
    // into the target output row (based on the current column count).
    std::vector<int> current_col_count(num_cols);
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const int64_t row_begin = input.row_ptr(row_idx);
      const int64_t row_end = input.row_ptr(row_idx + 1);
      for (int64_t i = row_begin; i < row_end; ++i) {
        const int col_idx = input.col_ind(i);
        const int64_t offset =
            output->row_ptr(col_idx) + current_col_count[col_idx];
        output->col_ind(offset) = row_idx;
        output->values(offset) = input.values(i);
        current_col_count[col_idx] += 1;
      }
    }
    return Status::OK();
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
struct CSRSparseMatrixTransposeComponent<GPUDevice, T> {
  Status operator()(OpKernelContext* ctx, const ConstCSRComponent<T>& x,
                    CSRComponent<T>* y) {
    TF_RETURN_IF_ERROR(ValidateTransposeInputs(x, *y));
    GpuSparse cuda_sparse(ctx);
    TF_RETURN_IF_ERROR(cuda_sparse.Initialize());
    const gpusparseAction_t copyValues = GPUSPARSE(ACTION_NUMERIC);
    const int rank = x.dense_shape_host.size();
    const int m = x.row_ptr.size() - 1;
    const int n = x.dense_shape_host(rank - 1);
    const int nnz = x.col_ind.size();
    DCHECK_EQ(nnz, x.values.size());
    DCHECK_EQ(n, y->row_ptr.size() - 1);
    DCHECK_EQ(rank, y->dense_shape_host.size());
    DCHECK_EQ(m, y->dense_shape_host(rank - 1));
    DCHECK_EQ(nnz, y->col_ind.size());
    DCHECK_EQ(nnz, y->values.size());

    return cuda_sparse.Csr2csc(
        m, n, nnz, x.values.data() /*csrVal*/, x.row_ptr.data() /*csrRowPtr*/,
        x.col_ind.data() /*csrColInd*/, y->values.data() /*cscVal*/,
        y->col_ind.data() /*cscRowInd*/, y->row_ptr.data() /*cscColPtr*/,
        copyValues);
    return Status::OK();
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace functor

}  // namespace tensorflow
