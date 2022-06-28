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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_ordering_amd_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_ordering_amd_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_ordering_amd_opDTcc() {
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

#include <vector>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/OrderingMethods"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Op to compute the Approximate Minimum Degree (AMD) ordering for a sparse
// matrix.
//
// Accepts a CSRSparseMatrix which may represent a single sparse matrix (rank 2)
// or a batch of sparse matrices (rank 3). Each component must be a square
// matrix. The input is assumed to be symmetric; only the lower triangular part
// of each component matrix is read. The numeric values of the sparse matrix
// does not affect the returned AMD ordering; only the sparsity pattern does.
//
// For each component sparse matrix A, the corresponding output Tensor
// represents the AMD ordering of A's rows and columns. The ordering is returned
// as a 1D Tensor (per batch) containing the list of indices, i.e. it contains
// each of the integers {0, .. N-1} exactly once; where N is the number of rows
// of the sparse matrix. The ith element represents the index of the row that
// the ith row should map to.

// If P represents the permutation matrix corresponding to the indices, then the
// matrix:
//   P^{-1} * A * P
// would have a sparse Cholesky decomposition with fewer structural non-zero
// elements than the sparse Cholesky decomposition of A itself.
class CSROrderingAMDCPUOp : public OpKernel {
  using SparseMatrix = Eigen::SparseMatrix<int, Eigen::RowMajor>;
  using Indices =
      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IndicesMap = Eigen::Map<Indices>;
  using ConstIndicesMap = Eigen::Map<const Indices>;

 public:
  explicit CSROrderingAMDCPUOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_ordering_amd_opDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/kernels/sparse/sparse_ordering_amd_op.cc", "CSROrderingAMDCPUOp");
}

  void Compute(OpKernelContext* ctx) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_ordering_amd_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/sparse/sparse_ordering_amd_op.cc", "Compute");

    // Extract the input CSRSparseMatrix.
    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));

    const Tensor& dense_shape = input_matrix->dense_shape();
    const int rank = dense_shape.dim_size(0);
    OP_REQUIRES(ctx, rank == 2 || rank == 3,
                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                        "but dense_shape has size ", rank));

    auto dense_shape_vec = dense_shape.vec<int64_t>();
    const int64_t num_rows = dense_shape_vec((rank == 2) ? 0 : 1);
    const int64_t num_cols = dense_shape_vec((rank == 2) ? 1 : 2);

    OP_REQUIRES(ctx, num_rows == num_cols,
                errors::InvalidArgument("sparse matrix must be square; got: ",
                                        num_rows, " != ", num_cols));

    // Allocate the output permutation indices.
    const int batch_size = input_matrix->batch_size();
    TensorShape permutation_indices_shape =
        (rank == 2) ? TensorShape{num_rows} : TensorShape{batch_size, num_rows};
    Tensor permutation_indices(cpu_allocator(), DT_INT32,
                               permutation_indices_shape);
    ctx->set_output(0, permutation_indices);

    // Parallelize AMD computation across batches using a threadpool.
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64_t amd_cost_per_batch =
        10 * num_rows * (input_matrix->total_nnz() / batch_size);
    Shard(
        worker_threads.num_threads, worker_threads.workers, batch_size,
        amd_cost_per_batch, [&](int64_t batch_begin, int64_t batch_end) {
          for (int64_t batch_index = batch_begin; batch_index < batch_end;
               ++batch_index) {
            // Define an Eigen SparseMatrix Map to operate on the
            // CSRSparseMatrix component without copying the data.
            // The values doesn't matter for computing the ordering, hence we
            // reuse the column pointers as dummy values.
            Eigen::Map<const SparseMatrix> sparse_matrix(
                num_rows, num_rows, input_matrix->nnz(batch_index),
                input_matrix->row_pointers_vec(batch_index).data(),
                input_matrix->col_indices_vec(batch_index).data(),
                input_matrix->col_indices_vec(batch_index).data());
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>
                permutation_matrix;
            // Compute the AMD ordering.
            Eigen::AMDOrdering<int> amd_ordering;
            amd_ordering(sparse_matrix.template selfadjointView<Eigen::Lower>(),
                         permutation_matrix);
            // Define an Eigen Map over the allocated output Tensor so that it
            // can be mutated in place.
            IndicesMap permutation_map(
                permutation_indices.flat<int>().data() + batch_index * num_rows,
                num_rows, 1);
            permutation_map = permutation_matrix.indices();
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseMatrixOrderingAMD").Device(DEVICE_CPU),
                        CSROrderingAMDCPUOp);

}  // namespace tensorflow
