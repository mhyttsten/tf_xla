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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc() {
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

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Scalar, typename Tidx>
class LuOp : public OpKernel {
 public:
  explicit LuOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/linalg/lu_op.cc", "LuOp");
}

 protected:
  using TensorShapes = gtl::InlinedVector<TensorShape, 4>;
  using TensorOutputs = gtl::InlinedVector<Tensor*, 4>;

  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  using Indices =
      Eigen::Matrix<Tidx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IndicesMap = Eigen::Map<Indices>;
  using ConstIndicesMap = Eigen::Map<const Indices>;

 public:
  // Returns the cost per matrix operation. This is used to determine the
  // number of threads to use for parallelizing factorization in batch mode.
  // Cost per unit is assumed to be roughly 1ns, based on comments
  // in core/util/work_sharder.cc.
  // LU decomposition for a square matrix takes roughly (2/3) * (num_rows)^3.
  // TODO(anudhyan): Refine this estimate after taking constant factors into
  // account.
  int64_t GetCostPerUnit(const TensorShape& input_matrix_shape) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/linalg/lu_op.cc", "GetCostPerUnit");

    double num_rows = static_cast<double>(input_matrix_shape.dim_size(0));
    double cost = (2 / 3.0) * MathUtil::IPow(num_rows, 3);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/kernels/linalg/lu_op.cc", "Compute");

    OP_REQUIRES(context, context->num_inputs() == 1,
                errors::InvalidArgument("Expecting exactly one input, got ",
                                        context->num_inputs()));

    const Tensor& input = context->input(0);
    int input_rank = input.dims();
    OP_REQUIRES(context, input_rank >= 2,
                errors::InvalidArgument(
                    "Input tensor must have rank >= 2, got ", input_rank));

    // If the tensor rank is greater than 2, we consider the inner-most
    // dimensions as matrices, and loop over all the other outer ("batch")
    // dimensions to compute the results.
    TensorShape input_matrix_shape;
    TensorShape batch_shape;
    for (int dim = 0; dim < input_rank - 2; ++dim) {
      batch_shape.AddDim(input.dim_size(dim));
    }
    const int64_t num_rows = input.dim_size(input_rank - 2);
    const int64_t num_cols = input.dim_size(input_rank - 1);

    input_matrix_shape.AppendShape({num_rows, num_cols});
    OP_REQUIRES(context, TensorShapeUtils::IsSquareMatrix(input_matrix_shape),
                errors::InvalidArgument("Input matrix must be square."));

    // packed_triangular_factors is a matrix with the same shape as the input;
    // permutation is a vector.
    TensorShape permutation_shape = batch_shape;
    permutation_shape.AddDim(num_rows);

    TensorShapes output_matrix_shapes({input.shape(), permutation_shape});

    TensorOutputs outputs;
    Tensor* output_packed_triangular_factors = nullptr;
    OP_REQUIRES_OK(
        context, context->forward_input_or_allocate_output(
                     {0}, 0, input.shape(), &output_packed_triangular_factors));
    outputs.emplace_back(output_packed_triangular_factors);

    Tensor* output_permutation = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, permutation_shape,
                                                     &output_permutation));
    outputs.emplace_back(output_permutation);

    if (num_rows == 0) {
      return;
    }

    // Process the individual matrix problems in parallel using a threadpool.
    auto shard = [this, &input, &num_rows, &num_cols, &outputs,
                  &output_matrix_shapes, context](int64_t begin, int64_t end) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/kernels/linalg/lu_op.cc", "lambda");

      for (int64_t i = begin; i < end; ++i) {
        ComputeTensorSlice(context, i, input, num_rows, num_cols, outputs,
                           output_matrix_shapes);
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          batch_shape.num_elements(), GetCostPerUnit(input_matrix_shape),
          shard);
  }

  void ComputeTensorSlice(OpKernelContext* context, int64_t matrix_index,
                          const Tensor& input, int64_t num_rows,
                          int64_t num_cols, const TensorOutputs& outputs,
                          const TensorShapes& output_matrix_shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_opDTcc mht_4(mht_4_v, 312, "", "./tensorflow/core/kernels/linalg/lu_op.cc", "ComputeTensorSlice");

    // TODO(kalakris): Handle alignment if possible. Eigen::Map is
    // unaligned by default.
    ConstMatrixMap input_matrix(
        input.flat<Scalar>().data() + matrix_index * num_rows * num_cols,
        num_rows, num_cols);

    // packed_triangular_factors has shape [num_rows, num_cols]
    MatrixMap packed_triangular_factors(
        outputs[0]->flat<Scalar>().data() + matrix_index * num_rows * num_cols,
        num_rows, num_rows);

    // permutation has shape [num_rows, 1]
    IndicesMap permutation_indices(
        outputs[1]->flat<Tidx>().data() + matrix_index * num_rows, num_rows, 1);

    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        lu_decomposition(input_matrix);

    // Output the packed triangular factors in a dense form.
    // The lower triangular factor L corresponds to the strictly lower
    // triangular part of packed_triangular_factors with an implicit unit
    // diagonal. The upper triangular factor U is the upper triangular part of
    // packed_triangular_factors. The triangular factors satisfy the equation
    //     P * input_matrix = L * U
    // where P is the permutation matrix corresponding to the indices in
    // permutation_indices.
    packed_triangular_factors = lu_decomposition.matrixLU();
    // Output the permutation matrix used for pivoting.
    Eigen::PermutationMatrix<-1, -1, Tidx> permutation =
        lu_decomposition.permutationP().transpose();
    permutation_indices = permutation.indices();

    // PartialPivLU cannot give strong guarantees on invertibility,
    // but we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes such providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const RealScalar min_abs_pivot =
        packed_triangular_factors.diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument("Input is not invertible."));
  }
};

#define REGISTER_LU(type, idx_type)                                         \
  REGISTER_KERNEL_BUILDER(Name("Lu")                                        \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<idx_type>("output_idx_type"), \
                          LuOp<type, idx_type>);

REGISTER_LU(float, int32);
REGISTER_LU(double, int32);
REGISTER_LU(complex64, int32);
REGISTER_LU(complex128, int32);

REGISTER_LU(float, int64_t);
REGISTER_LU(double, int64_t);
REGISTER_LU(complex64, int64_t);
REGISTER_LU(complex128, int64_t);

}  // namespace tensorflow
