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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

class MirrorPadOp : public XlaOpKernel {
 public:
  explicit MirrorPadOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/tf2xla/kernels/mirror_pad_op.cc", "MirrorPadOp");
}

  StatusOr<xla::XlaOp> DoMirrorPad(const xla::XlaOp t,
                                   const xla::Shape& original_shape,
                                   const xla::LiteralSlice& pad_literal,
                                   const MirrorPadMode mode,
                                   xla::XlaBuilder* b) {
    // The difference in the semantics of REFLECT and SYMMETRIC is that REFLECT
    // will not mirror the border values while symmetric does.
    // e.g. input is [1, 2, 3] and paddings is [0, 2], then the output is:
    // - [1, 2, 3, 2, 1] in reflect mode
    // - [1, 2, 3, 3, 2] in symmetric mode.
    int64_t excluded_edges = mode == MirrorPadMode::REFLECT ? 1 : 0;
    xla::XlaOp accum = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      auto t_rev = xla::Rev(accum, {dimno});
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);

      // Padding amounts on each side must be no more than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 &&
                   lhs_padding <= dim_size - excluded_edges);
      TF_RET_CHECK(rhs_padding >= 0 &&
                   rhs_padding <= dim_size - excluded_edges);

      auto lhs_pad =
          xla::SliceInDim(t_rev, dim_size - excluded_edges - lhs_padding,
                          dim_size - excluded_edges, 1, dimno);
      auto rhs_pad = xla::SliceInDim(t_rev, excluded_edges,
                                     excluded_edges + rhs_padding, 1, dimno);
      accum = xla::ConcatInDim(b, {lhs_pad, accum, rhs_pad}, dimno);
    }
    return accum;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/tf2xla/kernels/mirror_pad_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    MirrorPadMode mode;
    OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "mode", &mode));
    OP_REQUIRES(
        ctx, mode == MirrorPadMode::REFLECT || mode == MirrorPadMode::SYMMETRIC,
        xla::Unimplemented("Unsupported MirrorPad mode. Only SYMMETRIC and "
                           "REFLECT modes are currently supported"));

    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    OP_REQUIRES(
        ctx, dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    // Evaluate the 'padding' constant input, reshaping to a matrix.
    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal("paddings", &pad_literal));

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input("input");
    StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    StatusOr<xla::XlaOp> accum_status =
        DoMirrorPad(in0, in0_shape.ValueOrDie(), pad_literal, mode, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.ValueOrDie());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MirrorPadOp);
};

REGISTER_XLA_OP(Name("MirrorPad").CompileTimeConstantInput("paddings"),
                MirrorPadOp);

class MirrorPadGradOp : public XlaOpKernel {
 public:
  explicit MirrorPadGradOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc mht_2(mht_2_v, 291, "", "./tensorflow/compiler/tf2xla/kernels/mirror_pad_op.cc", "MirrorPadGradOp");
}

  StatusOr<xla::XlaOp> DoMirrorPadGrad(const xla::XlaOp t,
                                       const xla::Shape& original_shape,
                                       const xla::LiteralSlice& pad_literal,
                                       const MirrorPadMode mode,
                                       xla::XlaBuilder* b) {
    // The difference in the semantics of REFLECT and SYMMETRIC is that REFLECT
    // will not mirror the border values while symmetric does.
    // e.g. input is [1, 2, 3] and paddings is [0, 2], then the output is:
    // - [1, 2, 3, 2, 1] in reflect mode
    // - [1, 2, 3, 3, 2] in symmetric mode.
    int64_t excluded_edges = mode == MirrorPadMode::REFLECT ? 1 : 0;
    xla::XlaOp grad = t;
    for (int64_t dimno = original_shape.rank() - 1; dimno >= 0; --dimno) {
      int64_t lhs_padding = pad_literal.Get<int64_t>({dimno, 0});
      int64_t rhs_padding = pad_literal.Get<int64_t>({dimno, 1});
      int64_t dim_size = original_shape.dimensions(dimno);
      int64_t result_dim_size = dim_size - lhs_padding - rhs_padding;

      // Padding amounts on each side must be no more than the size of the
      // original shape.
      TF_RET_CHECK(lhs_padding >= 0 &&
                   lhs_padding <= dim_size - excluded_edges);
      TF_RET_CHECK(rhs_padding >= 0 &&
                   rhs_padding <= dim_size - excluded_edges);

      xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dimno);
      xla::XlaOp reverse_lhs_pad = xla::Rev(lhs_pad, {dimno});
      xla::XlaOp padded_lhs_pad = xla::PadInDim(
          reverse_lhs_pad, xla::ScalarLike(reverse_lhs_pad, 0), dimno,
          /*pad_lo=*/excluded_edges,
          /*pad_hi=*/result_dim_size - lhs_padding - excluded_edges);

      xla::XlaOp rhs_pad =
          xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dimno);
      xla::XlaOp reverse_rhs_pad = xla::Rev(rhs_pad, {dimno});
      xla::XlaOp padded_rhs_pad = xla::PadInDim(
          reverse_rhs_pad, xla::ScalarLike(reverse_rhs_pad, 0), dimno,
          /*pad_lo=*/result_dim_size - rhs_padding - excluded_edges,
          /*pad_hi=*/excluded_edges);

      xla::XlaOp grad_core =
          xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dimno);

      grad = padded_lhs_pad + grad_core + padded_rhs_pad;
    }
    return grad;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSmirror_pad_opDTcc mht_3(mht_3_v, 344, "", "./tensorflow/compiler/tf2xla/kernels/mirror_pad_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");

    MirrorPadMode mode;
    OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "mode", &mode));
    OP_REQUIRES(
        ctx, mode == MirrorPadMode::REFLECT || mode == MirrorPadMode::SYMMETRIC,
        xla::Unimplemented("Unsupported MirrorPadGrad mode. Only SYMMETRIC and "
                           "REFLECT modes are currently supported"));

    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    OP_REQUIRES(
        ctx, dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    // Evaluate the 'padding' constant input, reshaping to a matrix.
    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal("paddings", &pad_literal));

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input("input");
    StatusOr<xla::Shape> in0_shape = b->GetShape(in0);
    OP_REQUIRES(ctx, in0_shape.ok(), in0_shape.status());
    StatusOr<xla::XlaOp> accum_status =
        DoMirrorPadGrad(in0, in0_shape.ValueOrDie(), pad_literal, mode, b);

    OP_REQUIRES_OK(ctx, accum_status.status());

    ctx->SetOutput(0, accum_status.ValueOrDie());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MirrorPadGradOp);
};

REGISTER_XLA_OP(Name("MirrorPadGrad").CompileTimeConstantInput("paddings"),
                MirrorPadGradOp);

}  // namespace
}  // namespace tensorflow
