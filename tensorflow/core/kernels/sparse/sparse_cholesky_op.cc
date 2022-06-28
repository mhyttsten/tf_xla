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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc() {
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

#include <atomic>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/op_requires.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/OrderingMethods"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Op to compute the sparse Cholesky factorization of a sparse matrix.
//
// Implements a CPU kernel which returns the lower triangular sparse Cholesky
// factor of a CSRSparseMatrix, using the fill-in reducing permutation.
//
// The CSRSparseMatrix may represent a single sparse matrix (rank 2) or a batch
// of sparse matrices (rank 3). Each component must represent a symmetric
// positive definite (SPD) matrix. In particular, this means the component
// matrices must be square. We don't actually check if the input is symmetric,
// only the lower triangular part of each component is read.
//
// The associated permutation must be a Tensor of rank (R - 1), where the
// CSRSparseMatrix has rank R. Additionally, the batch dimension of the
// CSRSparseMatrix and the permutation must be the same. Each batch of
// the permutation should the contain each of the integers [0,..,N - 1] exactly
// once, where N is the number of rows of each CSR SparseMatrix component.
// TODO(anudhyan): Add checks to throw an InvalidArgument error if the
// permutation is not valid.
//
// Returns a CSRSparseMatrix representing the lower triangular (batched)
// Cholesky factors. It has the same shape as the input CSRSparseMatrix. For
// each component sparse matrix A, the corresponding output sparse matrix L
// satisfies the identity:
//   A = L * Lt
// where Lt denotes the adjoint of L.
//
// TODO(b/126472741): Due to the multiple batches of a 3D CSRSparseMatrix being
// laid out in contiguous memory, this implementation allocates memory to store
// a temporary copy of the Cholesky factor. Consequently, it uses roughly twice
// the amount of memory that it needs to. This may cause a memory blowup for
// sparse matrices with a high number of non-zero elements.
template <typename T>
class CSRSparseCholeskyCPUOp : public OpKernel {
  // Note: We operate in column major (CSC) format in this Op since the
  // SimplicialLLT returns the factor in column major.
  using SparseMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor>;

 public:
  explicit CSRSparseCholeskyCPUOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc mht_0(mht_0_v, 246, "", "./tensorflow/core/kernels/sparse/sparse_cholesky_op.cc", "CSRSparseCholeskyCPUOp");
}

  void Compute(OpKernelContext* ctx) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/kernels/sparse/sparse_cholesky_op.cc", "Compute");

    // Extract inputs and validate shapes and types.
    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));
    const Tensor& input_permutation_indices = ctx->input(1);

    int64_t num_rows;
    int batch_size;
    OP_REQUIRES_OK(ctx, ValidateInputs(*input_matrix, input_permutation_indices,
                                       &batch_size, &num_rows));

    // Allocate batch pointers.
    Tensor batch_ptr(cpu_allocator(), DT_INT32, TensorShape({batch_size + 1}));
    auto batch_ptr_vec = batch_ptr.vec<int32>();
    batch_ptr_vec(0) = 0;

    // Temporary vector of Eigen SparseMatrices to store the Sparse Cholesky
    // factors.
    // Note: we use column-compressed (CSC) SparseMatrix because SimplicialLLT
    // returns the factors in column major format. Since our input should be
    // symmetric, column major and row major is identical in storage. We just
    // have to switch to reading the upper triangular part of the input, which
    // corresponds to the lower triangular part in row major format.
    std::vector<SparseMatrix> sparse_cholesky_factors(batch_size);

    // TODO(anudhyan): Tune the cost per unit based on benchmarks.
    const double nnz_per_row =
        (input_matrix->total_nnz() / batch_size) / num_rows;
    const int64_t sparse_cholesky_cost_per_batch =
        nnz_per_row * nnz_per_row * num_rows;
    // Perform sparse Cholesky factorization of each batch in parallel.
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    std::atomic<int64_t> invalid_input_index(-1);
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          sparse_cholesky_cost_per_batch,
          [&](int64_t batch_begin, int64_t batch_end) {
            for (int64_t batch_index = batch_begin; batch_index < batch_end;
                 ++batch_index) {
              // Define an Eigen SparseMatrix Map to operate on the
              // CSRSparseMatrix component without copying the data.
              Eigen::Map<const SparseMatrix> sparse_matrix(
                  num_rows, num_rows, input_matrix->nnz(batch_index),
                  input_matrix->row_pointers_vec(batch_index).data(),
                  input_matrix->col_indices_vec(batch_index).data(),
                  input_matrix->values_vec<T>(batch_index).data());

              Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper,
                                   Eigen::NaturalOrdering<int>>
                  solver;
              auto permutation_indices_flat =
                  input_permutation_indices.flat<int32>().data();

              // Invert the fill-in reducing ordering and apply it to the input
              // sparse matrix.
              Eigen::Map<
                  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>>
                  permutation(permutation_indices_flat + batch_index * num_rows,
                              num_rows);
              auto permutation_inverse = permutation.inverse();

              SparseMatrix permuted_sparse_matrix;
              permuted_sparse_matrix.template selfadjointView<Eigen::Upper>() =
                  sparse_matrix.template selfadjointView<Eigen::Upper>()
                      .twistedBy(permutation_inverse);

              // Compute the Cholesky decomposition.
              solver.compute(permuted_sparse_matrix);
              if (solver.info() != Eigen::Success) {
                invalid_input_index = batch_index;
                return;
              }

              // Get the upper triangular factor, which would end up in the
              // lower triangular part of the output CSRSparseMatrix when
              // interpreted in row major format.
              sparse_cholesky_factors[batch_index] =
                  std::move(solver.matrixU());
              // For now, batch_ptr contains the number of nonzeros in each
              // batch.
              batch_ptr_vec(batch_index + 1) =
                  sparse_cholesky_factors[batch_index].nonZeros();
            }
          });

    // Check for invalid input.
    OP_REQUIRES(
        ctx, invalid_input_index == -1,
        errors::InvalidArgument(
            "Sparse Cholesky factorization failed for batch index ",
            invalid_input_index.load(), ". The input might not be valid."));

    // Compute a cumulative sum to obtain the batch pointers.
    std::partial_sum(batch_ptr_vec.data(),
                     batch_ptr_vec.data() + batch_size + 1,
                     batch_ptr_vec.data());

    // Allocate output Tensors.
    const int64_t total_nnz = batch_ptr_vec(batch_size);
    Tensor output_row_ptr(cpu_allocator(), DT_INT32,
                          TensorShape({(num_rows + 1) * batch_size}));
    Tensor output_col_ind(cpu_allocator(), DT_INT32, TensorShape({total_nnz}));
    Tensor output_values(cpu_allocator(), DataTypeToEnum<T>::value,
                         TensorShape({total_nnz}));
    auto output_row_ptr_ptr = output_row_ptr.flat<int32>().data();
    auto output_col_ind_ptr = output_col_ind.flat<int32>().data();
    auto output_values_ptr = output_values.flat<T>().data();

    // Copy the output matrices from each batch into the CSRSparseMatrix
    // Tensors.
    // TODO(b/129906419): Factor out the copy from Eigen SparseMatrix to
    // CSRSparseMatrix into common utils. This is also used in
    // SparseMatrixSparseMatMul.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          (3 * total_nnz) / batch_size /* cost per unit */,
          [&](int64_t batch_begin, int64_t batch_end) {
            for (int64_t batch_index = batch_begin; batch_index < batch_end;
                 ++batch_index) {
              const SparseMatrix& cholesky_factor =
                  sparse_cholesky_factors[batch_index];
              const int64_t nnz = cholesky_factor.nonZeros();

              std::copy(cholesky_factor.outerIndexPtr(),
                        cholesky_factor.outerIndexPtr() + num_rows + 1,
                        output_row_ptr_ptr + batch_index * (num_rows + 1));
              std::copy(cholesky_factor.innerIndexPtr(),
                        cholesky_factor.innerIndexPtr() + nnz,
                        output_col_ind_ptr + batch_ptr_vec(batch_index));
              std::copy(cholesky_factor.valuePtr(),
                        cholesky_factor.valuePtr() + nnz,
                        output_values_ptr + batch_ptr_vec(batch_index));
            }
          });

    // Create the CSRSparseMatrix instance from its component Tensors and
    // prepare the Variant output Tensor.
    CSRSparseMatrix output_csr_matrix;
    OP_REQUIRES_OK(
        ctx,
        CSRSparseMatrix::CreateCSRSparseMatrix(
            DataTypeToEnum<T>::value, input_matrix->dense_shape(), batch_ptr,
            output_row_ptr, output_col_ind, output_values, &output_csr_matrix));
    Tensor* output_csr_matrix_tensor;
    AllocatorAttributes cpu_alloc;
    cpu_alloc.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_csr_matrix_tensor,
                                  cpu_alloc));
    output_csr_matrix_tensor->scalar<Variant>()() =
        std::move(output_csr_matrix);
  }

 private:
  Status ValidateInputs(const CSRSparseMatrix& sparse_matrix,
                        const Tensor& permutation_indices, int* batch_size,
                        int64_t* num_rows) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_cholesky_opDTcc mht_2(mht_2_v, 408, "", "./tensorflow/core/kernels/sparse/sparse_cholesky_op.cc", "ValidateInputs");

    if (sparse_matrix.dtype() != DataTypeToEnum<T>::value)
      return errors::InvalidArgument(
          "Asked for a CSRSparseMatrix of type ",
          DataTypeString(DataTypeToEnum<T>::value),
          " but saw dtype: ", DataTypeString(sparse_matrix.dtype()));

    const Tensor& dense_shape = sparse_matrix.dense_shape();
    const int rank = dense_shape.dim_size(0);
    if (rank < 2 || rank > 3)
      return errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                     "but dense_shape has size ", rank);
    const int row_dim = (rank == 2) ? 0 : 1;
    auto dense_shape_vec = dense_shape.vec<int64_t>();
    *num_rows = dense_shape_vec(row_dim);
    const int64_t num_cols = dense_shape_vec(row_dim + 1);
    if (*num_rows != num_cols)
      return errors::InvalidArgument(
          "sparse matrix must be square; got: ", *num_rows, " != ", num_cols);
    const TensorShape& perm_shape = permutation_indices.shape();
    if (perm_shape.dims() + 1 != rank)
      return errors::InvalidArgument(
          "sparse matrix must have the same rank as permutation; got: ", rank,
          " != ", perm_shape.dims(), " + 1.");
    if (perm_shape.dim_size(rank - 2) != *num_rows)
      return errors::InvalidArgument(
          "permutation must have the same number of elements in each batch "
          "as the number of rows in sparse matrix; got: ",
          perm_shape.dim_size(rank - 2), " != ", *num_rows);

    *batch_size = sparse_matrix.batch_size();
    if (*batch_size > 1) {
      if (perm_shape.dim_size(0) != *batch_size)
        return errors::InvalidArgument(
            "permutation must have the same batch size "
            "as sparse matrix; got: ",
            perm_shape.dim_size(0), " != ", *batch_size);
    }

    return Status::OK();
  }
};

#define REGISTER_CPU(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSparseCholesky") \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("type"),    \
                          CSRSparseCholeskyCPUOp<T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#undef REGISTER_CPU

}  // namespace tensorflow
