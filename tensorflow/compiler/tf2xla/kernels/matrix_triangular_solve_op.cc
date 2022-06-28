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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {
namespace {

class MatrixTriangularSolveOp : public XlaOpKernel {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/tf2xla/kernels/matrix_triangular_solve_op.cc", "MatrixTriangularSolveOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint", &adjoint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/matrix_triangular_solve_op.cc", "Compile");

    const TensorShape lhs_shape = ctx->InputShape(0);
    const TensorShape rhs_shape = ctx->InputShape(1);

    // By TensorFlow conventions the inputs may not have the same
    // shapes, in which case they will be automatically broadcast if
    // possible before mapping. Use the standard TensorFlow helper to
    // compute valid broadcast shapes, but rely below on XLA to
    // automatically perform the broadcast assuming its valid shapes are
    // a superset of TensorFlow's valid shapes.
    MatMulBCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape));
    if (!bcast.IsValid()) {
      ctx->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ", lhs_shape.DebugString(), " vs. ",
          rhs_shape.DebugString()));
      return;
    }

    auto lhs_size = lhs_shape.dims();
    OP_REQUIRES(
        ctx,
        lhs_shape.dim_size(lhs_size - 1) == lhs_shape.dim_size(lhs_size - 2),
        errors::InvalidArgument("The coefficient matrix must be square in "
                                "the inner-most two dimensions: ",
                                lhs_shape.DebugString()));

    xla::XlaOp a = ctx->Input(0);
    xla::XlaOp b = ctx->Input(1);
    std::tie(a, b) = Broadcast(a, lhs_shape, b, rhs_shape, bcast);
    auto result = xla::TriangularSolve(
        a, b, /*left_side=*/true,
        /*lower=*/lower_, /*unit_diagonal=*/false,
        /*transpose_a=*/
        adjoint_ ? xla::TriangularSolveOptions::ADJOINT
                 : xla::TriangularSolveOptions::NO_TRANSPOSE);
    ctx->SetOutput(0, result);
  }

 private:
  static std::pair<xla::XlaOp, xla::XlaOp> Broadcast(
      xla::XlaOp lhs, const TensorShape& lhs_shape, xla::XlaOp rhs,
      const TensorShape& rhs_shape, const MatMulBCast& broadcast_helper);
  bool lower_;
  bool adjoint_;
};

/* static */ std::pair<xla::XlaOp, xla::XlaOp>
MatrixTriangularSolveOp::Broadcast(xla::XlaOp lhs, const TensorShape& lhs_shape,
                                   xla::XlaOp rhs, const TensorShape& rhs_shape,
                                   const MatMulBCast& broadcast_helper) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatrix_triangular_solve_opDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/tf2xla/kernels/matrix_triangular_solve_op.cc", "MatrixTriangularSolveOp::Broadcast");

  // Get the batch shape.
  int64_t m = lhs_shape.dim_size(lhs_shape.dims() - 1);
  int64_t n = rhs_shape.dim_size(rhs_shape.dims() - 1);

  TensorShape lhs_broadcast_shape(broadcast_helper.output_batch_shape());
  lhs_broadcast_shape.AddDim(m);
  lhs_broadcast_shape.AddDim(m);
  auto lhs_output = BroadcastTo(lhs, lhs_broadcast_shape.dim_sizes());
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }

  TensorShape rhs_broadcast_shape(broadcast_helper.output_batch_shape());
  rhs_broadcast_shape.AddDim(m);
  rhs_broadcast_shape.AddDim(n);
  auto rhs_output = BroadcastTo(rhs, rhs_broadcast_shape.dim_sizes());
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.ValueOrDie(), rhs_output.ValueOrDie()};
}

REGISTER_XLA_OP(Name("MatrixTriangularSolve"), MatrixTriangularSolveOp);

}  // namespace
}  // namespace tensorflow
