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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc() {
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

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"

namespace tensorflow {

template <typename Scalar>
class CholeskyGrad : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit CholeskyGrad(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/linalg/cholesky_grad.cc", "CholeskyGrad");
}

  using TensorShapes = typename Base::TensorShapes;
  using Matrix = typename Base::Matrix;
  using MatrixMap = typename Base::MatrixMap;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;
  using ConstRef = Eigen::Ref<const Matrix>;
  using Ref = Eigen::Ref<Matrix>;

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/linalg/cholesky_grad.cc", "ValidateInputMatrixShapes");

    OP_REQUIRES(context, input_matrix_shapes.size() == 2,
                errors::InvalidArgument("Expected two input matrices, got %d.",
                                        input_matrix_shapes.size()));
    OP_REQUIRES(context, input_matrix_shapes[0] == input_matrix_shapes[1],
                errors::InvalidArgument(
                    "Inputs (L and grad) must have the same shape."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
                errors::InvalidArgument("Inputs must be a square matrices."));
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/linalg/cholesky_grad.cc", "GetOutputMatrixShapes");

    return TensorShapes({input_matrix_shapes[0]});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/kernels/linalg/cholesky_grad.cc", "ComputeMatrix");

    const ConstMatrixMap& input_matrix_l_full = inputs[0];
    const ConstMatrixMap& input_matrix_grad = inputs[1];
    MatrixMap output_matrix = outputs->at(0);

    // Algorithm only depends on lower triangular half on input_matrix_l.
    const Matrix input_matrix_l =
        input_matrix_l_full.template triangularView<Eigen::Lower>();
    // Algorithm only depends on lower triangular half on input_matrix_grad.
    output_matrix = input_matrix_grad.template triangularView<Eigen::Lower>();

    const int64_t kMatrixSize = input_matrix_l.rows();
    const int64_t kMaxBlockSize = 32;

    for (int64_t block_end = kMatrixSize; block_end > 0;
         block_end -= kMaxBlockSize) {
      /* This shows the block structure.

      /      \
      |      |
      | R D  |
      \ B C  /

      Variables names representing the derivative matrix have a trailing '_bar'.
      */

      const int64_t block_begin =
          std::max(int64_t{0}, block_end - kMaxBlockSize);
      const int64_t block_size = block_end - block_begin;
      const int64_t trailing_size = kMatrixSize - block_end;

      auto B = input_matrix_l.block(block_end, 0, trailing_size, block_begin);
      auto B_bar =
          output_matrix.block(block_end, 0, trailing_size, block_begin);

      auto C = input_matrix_l.block(block_end, block_begin, trailing_size,
                                    block_size);
      auto C_bar = output_matrix.block(block_end, block_begin, trailing_size,
                                       block_size);

      auto D = input_matrix_l.block(block_begin, block_begin, block_size,
                                    block_size);
      auto D_bar =
          output_matrix.block(block_begin, block_begin, block_size, block_size);

      auto R = input_matrix_l.block(block_begin, 0, block_size, block_begin);
      auto R_bar = output_matrix.block(block_begin, 0, block_size, block_begin);

      C_bar = D.adjoint()
                  .template triangularView<Eigen::Upper>()
                  .solve(C_bar.adjoint())
                  .adjoint();
      D_bar -= (C_bar.adjoint() * C).template triangularView<Eigen::Lower>();
      B_bar -= C_bar * R;
      R_bar -= C_bar.adjoint() * B;
      CholeskyGradUnblocked(D, D_bar);
      R_bar -= (D_bar + D_bar.adjoint()) * R;
    }
    output_matrix = (0.5 * (output_matrix + output_matrix.transpose())).eval();
  }

 private:
  void CholeskyGradUnblocked(const ConstRef& l_block, Ref grad_block) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_gradDTcc mht_4(mht_4_v, 304, "", "./tensorflow/core/kernels/linalg/cholesky_grad.cc", "CholeskyGradUnblocked");

    const int64_t kMatrixSize = l_block.rows();
    for (int64_t k = kMatrixSize - 1; k >= 0; k--) {
      /* This shows the block structure.

      /      \
      |      |
      | r d  |
      \ B c  /

      Variables names representing the derivative matrix have a trailing '_bar'.
      */

      const int64_t number_rows_B = kMatrixSize - (k + 1);
      const int64_t number_rows_r_stack_B = number_rows_B + 1;

      auto r = l_block.block(k, 0, 1, k);
      auto r_bar = grad_block.block(k, 0, 1, k);
      auto d = l_block(k, k);  // This needs to be a scalar rather than a view.
      auto d_bar = grad_block.block(k, k, 1, 1);
      // B is not included explicitly because it is not used on its own.
      auto B_bar = grad_block.block(k + 1, 0, number_rows_B, k);
      auto c = l_block.block(k + 1, k, number_rows_B, 1);
      auto c_bar = grad_block.block(k + 1, k, number_rows_B, 1);
      // Result of vertical stacking d_bar and c_bar.
      auto d_stack_c_bar = grad_block.block(k, k, number_rows_r_stack_B, 1);
      // Result of vertical stacking of r and B.
      auto r_stack_B = l_block.block(k, 0, number_rows_r_stack_B, k);
      d_bar -= (c.adjoint() * c_bar) / d;
      d_stack_c_bar /= d;
      r_bar -= d_stack_c_bar.adjoint() * r_stack_B;
      B_bar -= c_bar * r;
      d_bar /= 2.;
    }
  }
};

REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<float>), float);
REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<double>), double);
REGISTER_LINALG_OP("BatchCholeskyGrad", (CholeskyGrad<float>), float);
REGISTER_LINALG_OP("BatchCholeskyGrad", (CholeskyGrad<double>), double);

}  // namespace tensorflow
