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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific MatMul Op.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 10> kMatmulTypes = {
    {DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
     DT_INT32, DT_INT64, DT_INT16, DT_INT8}};

class MatMulOp : public XlaOpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* ctx, bool is_sparse = false)
      : XlaOpKernel(ctx), is_sparse_(is_sparse) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/matmul_op.cc", "MatMulOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    if (is_sparse) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Ta", &a_type_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Tb", &b_type_));
      // SparseMatMul is actually dense matmul with a hint that one or
      // both of the inputs may contain a lot of zeroes. On CPU these
      // inputs are dynamically converted to sparse representation
      // before multiplication. For now in XLA we ignore the hints
      // and always do dense multiplication.
      bool dummy_is_sparse;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("a_is_sparse", &dummy_is_sparse));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("b_is_sparse", &dummy_is_sparse));
    }
  }

  ~MatMulOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/tf2xla/kernels/matmul_op.cc", "Compile");

    const TensorShape a_shape = ctx->InputShape(0);
    const TensorShape b_shape = ctx->InputShape(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, a_shape.dims() == b_shape.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        a_shape.DebugString(), " vs. ",
                                        b_shape.DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a_shape),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a_shape.DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b_shape),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b_shape.DebugString()));
    int first_index = transpose_a_ ? 0 : 1;
    int second_index = transpose_b_ ? 1 : 0;

    OP_REQUIRES(ctx,
                a_shape.dim_size(first_index) == b_shape.dim_size(second_index),
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0]: ", a_shape.DebugString(),
                    ", In[1]: ", b_shape.DebugString()));

    xla::XlaOp a = ctx->Input(0);
    xla::XlaOp b = ctx->Input(1);
    if (is_sparse_) {
      if (a_type_ == DT_BFLOAT16) {
        a = xla::ConvertElementType(a, xla::F32);
      }
      if (b_type_ == DT_BFLOAT16) {
        b = xla::ConvertElementType(b, xla::F32);
      }
    }
    ctx->SetOutput(0, xla::BatchDot(a, transpose_a_, b, transpose_b_));
  }

 private:
  bool is_sparse_;
  bool transpose_a_;
  bool transpose_b_;
  DataType a_type_;
  DataType b_type_;
};

REGISTER_XLA_OP(Name("MatMul").TypeConstraint("T", kMatmulTypes), MatMulOp);

class SparseMatMulOp : public MatMulOp {
 public:
  explicit SparseMatMulOp(OpKernelConstruction* ctx) : MatMulOp(ctx, true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmatmul_opDTcc mht_2(mht_2_v, 281, "", "./tensorflow/compiler/tf2xla/kernels/matmul_op.cc", "SparseMatMulOp");
}

  ~SparseMatMulOp() override = default;
};

REGISTER_XLA_OP(Name("SparseMatMul"), SparseMatMulOp);

}  // namespace
}  // namespace tensorflow
