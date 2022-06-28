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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_SOLVE_LS_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_SOLVE_LS_OP_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh() {
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


// See docs in ../ops/linalg_ops.cc.

#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class MatrixSolveLsOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit MatrixSolveLsOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "MatrixSolveLsOp");

    OP_REQUIRES_OK(context, context->GetAttr("fast", &fast_));
  }

  using TensorShapes = typename Base::TensorShapes;
  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  // Tell the base class to ignore the regularization parameter
  // in context->input(2).
  int NumMatrixInputs(const OpKernelContext* context) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "NumMatrixInputs");
 return 2; }

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_2(mht_2_v, 230, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "ValidateInputMatrixShapes");

    Base::ValidateSolver(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_3(mht_3_v, 238, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "GetOutputMatrixShapes");

    return TensorShapes({TensorShape({input_matrix_shapes[0].dim_size(1),
                                      input_matrix_shapes[1].dim_size(1)})});
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_4(mht_4_v, 246, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "GetCostPerUnit");

    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double num_rhss = static_cast<double>(input_matrix_shapes[1].dim_size(1));
    double cost = std::max(m, n) * std::min(m, n) * (std::min(m, n) + num_rhss);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  bool EnableInputForwarding() const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_5(mht_5_v, 258, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "EnableInputForwarding");
 return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_solve_ls_op_implDTh mht_6(mht_6_v, 264, "", "./tensorflow/core/kernels/linalg/matrix_solve_ls_op_impl.h", "ComputeMatrix");

    const ConstMatrixMap& matrix = inputs[0];
    const ConstMatrixMap& rhs = inputs[1];
    const auto& l2_regularizer_in = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(l2_regularizer_in.shape()),
        errors::InvalidArgument("l2_regularizer must be scalar, got shape ",
                                l2_regularizer_in.shape().DebugString()));
    const double l2_regularizer = l2_regularizer_in.scalar<double>()();
    OP_REQUIRES(context, l2_regularizer >= 0,
                errors::InvalidArgument("l2_regularizer must be >= 0."));

    const int64_t rows = matrix.rows();
    const int64_t cols = matrix.cols();
    if (rows == 0 || cols == 0 || rhs.rows() == 0 || rhs.cols() == 0) {
      // The result is the empty matrix.
      return;
    }
    if (fast_) {
      // The fast branch assumes that matrix is not rank deficient and
      // not too ill-conditioned. Specifically, the reciprocal condition number
      // should be greater than the square root of the machine precision, i.e.
      //   1 / cond(matrix) > sqrt(std::numeric_limits<Scalar>::epsilon()).
      // This branch solves over- or underdetermined least-squares problems
      // via the normal equations and Cholesky decomposition.
      if (rows >= cols) {
        // Overdetermined case (rows >= cols): Solves the ordinary (possibly
        // regularized) least-squares problem
        //   min || A * X - RHS ||_F^2 + l2_regularizer ||X||_F^2
        // by solving the normal equations
        //    (A^T * A + l2_regularizer * I) X = A^T RHS
        // using Cholesky decomposition.
        Matrix gramian(cols, cols);
        gramian.template triangularView<Eigen::Lower>() =
            matrix.adjoint() * matrix;
        if (l2_regularizer > 0) {
          gramian +=
              (Scalar(l2_regularizer) * Matrix::Ones(cols, 1)).asDiagonal();
        }
        const Eigen::LLT<Eigen::Ref<Matrix>, Eigen::Lower> llt(gramian);
        OP_REQUIRES(
            context, llt.info() == Eigen::Success,
            errors::InvalidArgument("Input matrix was rank deficient or "
                                    "ill-conditioned. Try setting fast=False "
                                    "or provide a larger l2_regularizer > 0."));
        outputs->at(0).noalias() = matrix.adjoint() * rhs;
        llt.solveInPlace(outputs->at(0));
      } else {
        // Underdetermined case (rows < cols): Solves the minimum-norm problem
        //   min ||X||_F^2 s.t. A*X = RHS
        // by solving the normal equations of the second kind
        //   (A * A^T + l2_regularizer * I) Z = RHS,  X = A^T * Z
        // using Cholesky decomposition.
        Matrix gramian(rows, rows);
        gramian.template triangularView<Eigen::Lower>() =
            matrix * matrix.adjoint();
        if (l2_regularizer > 0) {
          gramian +=
              (Scalar(l2_regularizer) * Matrix::Ones(rows, 1)).asDiagonal();
        }
        const Eigen::LLT<Eigen::Ref<Matrix>, Eigen::Lower> llt(gramian);
        OP_REQUIRES(
            context, llt.info() == Eigen::Success,
            errors::InvalidArgument("Input matrix was rank deficient or "
                                    "ill-conditioned. Try setting fast=False "
                                    "or provide an l2_regularizer > 0."));
        outputs->at(0).noalias() = matrix.adjoint() * llt.solve(rhs);
      }
    } else {
      // Use complete orthogonal decomposition which is backwards stable and
      // will compute the minimum-norm solution for rank-deficient matrices.
      // This is 6-7 times slower than the fast path.
      //
      // TODO(rmlarsen): The implementation of
      //   Eigen::CompleteOrthogonalDecomposition is not blocked, so for
      //   matrices that do not fit in cache, it is significantly slower than
      //   the equivalent blocked LAPACK routine xGELSY (e.g. Eigen is ~3x
      //   slower for 4k x 4k matrices).
      //   See http://www.netlib.org/lapack/lawnspdf/lawn114.pdf
      outputs->at(0) = matrix.completeOrthogonalDecomposition().solve(rhs);
    }
  }

 private:
  bool fast_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_SOLVE_LS_OP_IMPL_H_
