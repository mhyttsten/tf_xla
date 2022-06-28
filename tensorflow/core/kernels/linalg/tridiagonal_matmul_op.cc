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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc() {
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

// See docs in ../ops/linalg_ops.cc.

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TODO(b/131583008): add broadcast support (for batch dimensions).
template <class Scalar>
class TridiagonalMatMulOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit TridiagonalMatMulOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "TridiagonalMatMulOp");
}

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "ValidateInputMatrixShapes");

    auto num_inputs = input_matrix_shapes.size();
    OP_REQUIRES(
        context, num_inputs == 4,
        errors::InvalidArgument("Expected 4 inputs, got ", num_inputs, "."));

    auto n = input_matrix_shapes[3].dim_size(0);

    OP_REQUIRES(context,
                input_matrix_shapes[0].dim_size(0) == 1 &&
                    input_matrix_shapes[0].dim_size(1) == n,
                errors::InvalidArgument("Invalid superdiagonal shape."));

    OP_REQUIRES(context,
                input_matrix_shapes[1].dim_size(0) == 1 &&
                    input_matrix_shapes[1].dim_size(1) == n,
                errors::InvalidArgument("Invalid main diagonal shape."));

    OP_REQUIRES(context,
                input_matrix_shapes[2].dim_size(0) == 1 &&
                    input_matrix_shapes[2].dim_size(1) == n,
                errors::InvalidArgument("Invalid subdiagonal shape."));
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "GetOutputMatrixShapes");

    return TensorShapes({input_matrix_shapes[3]});
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "GetCostPerUnit");

    const int num_eqs = static_cast<int>(input_matrix_shapes[0].dim_size(1));
    const int num_rhss = static_cast<int>(input_matrix_shapes[3].dim_size(0));

    const double add_cost = Eigen::TensorOpCost::AddCost<Scalar>();
    const double mult_cost = Eigen::TensorOpCost::MulCost<Scalar>();

    const double cost = num_rhss * ((3 * num_eqs - 2) * mult_cost +
                                    (2 * num_eqs - 2) * add_cost);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  // Needed to prevent writing result to the same location where input is.
  bool EnableInputForwarding() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "EnableInputForwarding");
 return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_opDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op.cc", "ComputeMatrix");

    // Superdiagonal elements. Must have length m.
    // Last element is ignored.
    const auto& superdiag = inputs[0].row(0);

    // Diagonal elements. Must have length m.
    const auto& maindiag = inputs[1].row(0);

    // Subdiagonal elements. Must have length m.
    // First element is ignored.
    const auto& subdiag = inputs[2].row(0);

    // Right-hand matrix. Size m x n.
    const auto& rhs = inputs[3];

    MatrixMap& result = outputs->at(0);

    const int m = rhs.rows();
    const int n = rhs.cols();

    ConstVectorMap subdiag_map(subdiag.data() + 1, m - 1);
    ConstVectorMap superdiag_map(superdiag.data(), m - 1);
    ConstMatrixMap rhs_except_first_row(rhs.data() + n, m - 1, n);
    ConstMatrixMap rhs_except_last_row(rhs.data(), m - 1, n);

    MatrixMap result_except_first_row(result.data() + n, m - 1, n);
    MatrixMap result_except_last_row(result.data(), m - 1, n);
    result.array() = rhs.array().colwise() * maindiag.transpose().array();
    result_except_first_row.noalias() +=
        (rhs_except_last_row.array().colwise() *
         subdiag_map.transpose().array())
            .matrix();
    result_except_last_row.noalias() +=
        (rhs_except_first_row.array().colwise() *
         superdiag_map.transpose().array())
            .matrix();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalMatMulOp);
};

REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<float>),
                       float);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<double>),
                       double);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<complex64>),
                       complex64);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<complex128>),
                       complex128);
}  // namespace tensorflow
