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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc() {
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

// XLA-specific Shape Ops.

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ShapeOp : public XlaOpKernel {
 public:
  explicit ShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "ShapeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);
    std::vector<xla::XlaOp> operands;
    const int rank = input_shape.dims();
    if (rank != 0) {
      for (int64_t i = 0; i < rank; ++i) {
        operands.push_back(xla::Broadcast(
            xla::ConvertElementType(xla::GetDimensionSize(ctx->Input(0), i),
                                    ctx->output_xla_type(0)),
            {1}));
      }

      ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), operands, 0));
    } else {
      // Rank 0 won't have dynamic size dimension, use constant output.
      Tensor shape_constant(out_dtype_, TensorShape({input_shape.dims()}));
      OP_REQUIRES_OK(ctx, TensorShapeToConstant(input_shape, &shape_constant));
      ctx->SetConstantOutput(0, shape_constant);
    }
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("Shape").CompilationOnly().IsMetadataOp(), ShapeOp);

class XlaSetBoundOp : public XlaOpKernel {
 public:
  explicit XlaSetBoundOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "XlaSetBoundOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape bound_shape = ctx->InputShape("bound");

    OP_REQUIRES(
        ctx,
        ctx->InputType("bound") == DT_INT32 &&
            ctx->InputType("input") == DT_INT32,
        errors::InvalidArgument(
            "XlaSetBound can only set bound for int32 scalar value: got",
            input_shape.DebugString()));

    OP_REQUIRES(
        ctx, input_shape.dims() == 0,
        errors::InvalidArgument("XlaSetBound should only be used to set a "
                                "bound to the an int32 scalar value: got",
                                input_shape.DebugString()));

    OP_REQUIRES(
        ctx, bound_shape.dims() == 0,
        errors::InvalidArgument("XlaSetBound should only be used to set a "
                                "bound to the an int32 scalar value: got",
                                bound_shape.DebugString()));
    int64_t bound;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("bound", &bound));
    xla::Literal bound_literal = xla::LiteralUtil::CreateR0<int32>(bound);
    xla::XlaOp result =
        xla::CustomCall(ctx->builder(), "SetBound", {ctx->Input("input")},
                        ctx->InputXlaShape("input").ValueOrDie(), "", false, {},
                        &bound_literal);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("XlaSetBound").CompileTimeConstantInput("bound"),
                XlaSetBoundOp);

class XlaSetDynamicDimensionSizeOp : public XlaOpKernel {
 public:
  explicit XlaSetDynamicDimensionSizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_4(mht_4_v, 297, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "XlaSetDynamicDimensionSizeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_5(mht_5_v, 302, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape dim_index_shape = ctx->InputShape("dim_index");
    const TensorShape size_shape = ctx->InputShape("size");

    OP_REQUIRES(ctx,
                ctx->InputType("dim_index") == DT_INT32 &&
                    ctx->InputType("size") == DT_INT32,
                errors::InvalidArgument("dim_index and size has to be int32 for"
                                        "XlaSetDynamicDimensionSizeOp"));

    OP_REQUIRES(
        ctx, dim_index_shape.dims() == 0 && size_shape.dims() == 0,
        errors::InvalidArgument("XlaSetDynamicDimensionSizeOp's dim_index and "
                                "size has to be int32 scalar value"));
    int64_t dim_index;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("dim_index", &dim_index));

    xla::XlaOp result =
        xla::SetDimensionSize(ctx->Input(0), ctx->Input("size"), dim_index);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(
    Name("XlaSetDynamicDimensionSize").CompileTimeConstantInput("dim_index"),
    XlaSetDynamicDimensionSizeOp);

class XlaRemoveDynamicDimensionSizeOp : public XlaOpKernel {
 public:
  explicit XlaRemoveDynamicDimensionSizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_6(mht_6_v, 335, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "XlaRemoveDynamicDimensionSizeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_7(mht_7_v, 340, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape dim_index_shape = ctx->InputShape("dim_index");

    OP_REQUIRES(ctx, ctx->InputType("dim_index") == DT_INT32,
                errors::InvalidArgument("dim_index has to be int32 for"
                                        "XlaRemoveDynamicDimensionSizeOp"));

    OP_REQUIRES(
        ctx, dim_index_shape.dims() == 0,
        errors::InvalidArgument("XlaRemoveDynamicDimensionSizeOp's dim_index "
                                "has to be int32 scalar value"));
    int64_t dim_index;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("dim_index", &dim_index));

    xla::XlaOp result = xla::RemoveDynamicDimension(ctx->Input(0), dim_index);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(
    Name("XlaRemoveDynamicDimensionSize").CompileTimeConstantInput("dim_index"),
    XlaRemoveDynamicDimensionSizeOp);

class ShapeNOp : public XlaOpKernel {
 public:
  explicit ShapeNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_8(mht_8_v, 368, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "ShapeNOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_9(mht_9_v, 375, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const TensorShape input_shape = ctx->InputShape(i);
      std::vector<xla::XlaOp> operands;

      const int rank = input_shape.dims();
      if (rank != 0) {
        // Each dimension can be dynamic, so use GetDimensionSize to get the
        // runtime dimension.
        for (int64_t dim = 0; dim < rank; ++dim) {
          operands.push_back(xla::Broadcast(
              xla::ConvertElementType(xla::GetDimensionSize(ctx->Input(i), dim),
                                      ctx->output_xla_type(i)),
              {1}));
        }

        ctx->SetOutput(i, xla::ConcatInDim(ctx->builder(), operands, 0));
      } else {
        // Rank 0 won't have dynamic size dimension, use constant output.
        Tensor shape_constant(out_dtype_, TensorShape({input_shape.dims()}));
        OP_REQUIRES_OK(ctx,
                       TensorShapeToConstant(input_shape, &shape_constant));
        ctx->SetConstantOutput(i, shape_constant);
      }
    }
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_10(mht_10_v, 405, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "IsExpensive");
 return false; }

 private:
  DataType out_dtype_;
};
REGISTER_XLA_OP(Name("ShapeN").CompilationOnly().IsMetadataOp(), ShapeNOp);

class RankOp : public XlaOpKernel {
 public:
  explicit RankOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_11(mht_11_v, 417, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "RankOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_12(mht_12_v, 422, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);
    const int rank = input_shape.dims();
    Tensor rank_constant(DT_INT32, TensorShape({}));
    rank_constant.scalar<int32>()() = rank;

    ctx->SetConstantOutput(0, rank_constant);
  }
};

REGISTER_XLA_OP(Name("Rank").CompilationOnly().IsMetadataOp(), RankOp);

class SizeOp : public XlaOpKernel {
 public:
  explicit SizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_13(mht_13_v, 439, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "SizeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_14(mht_14_v, 444, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx,
                FastBoundsCheck(input_shape.num_elements(),
                                std::numeric_limits<int32>::max()),
                errors::InvalidArgument("Size does not work for tensors > "
                                        "int32 max."));
    Tensor size_constant(DT_INT32, TensorShape({}));
    const int rank = input_shape.dims();
    xla::XlaBuilder* builder = ctx->builder();
    auto size = xla::One(builder, xla::U32);
    for (int64_t i = 0; i < rank; ++i) {
      size = xla::Mul(
          size, xla::ConvertElementType(xla::GetDimensionSize(ctx->Input(0), i),
                                        xla::U32));
    }
    size = xla::ConvertElementType(size, ctx->output_xla_type(0));
    ctx->SetOutput(0, size);
  }
};

REGISTER_XLA_OP(Name("Size").CompilationOnly().IsMetadataOp(), SizeOp);

class ExpandDimsOp : public XlaOpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_15(mht_15_v, 472, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "ExpandDimsOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_16(mht_16_v, 477, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape dim_shape = ctx->InputShape("dim");

    std::vector<int64_t> dims;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector("dim", &dims));
    OP_REQUIRES(ctx, dims.size() == 1,
                errors::InvalidArgument(absl::StrCat(
                    "dim input to ExpandDims must be a scalar; got ",
                    dim_shape.DebugString())));
    int dim = dims[0];

    OP_REQUIRES(ctx,
                (dim >= -1 - input_shape.dims() && dim <= input_shape.dims()),
                errors::InvalidArgument("Tried to expand dim index ", dim,
                                        " for tensor with ", input_shape.dims(),
                                        " dimensions."));

    auto existing_dims = input_shape.dim_sizes();
    // Safe - # elements in tensor dims bounded.
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64_t> new_shape(existing_dims_size);
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      dim += existing_dims.size() + 1;
    }

    // Clamp to the end if needed.
    dim = std::min<int32>(dim, existing_dims_size);
    new_shape.emplace(new_shape.begin() + dim, 1);

    ctx->SetOutput(0, xla::Reshape(ctx->Input("input"), new_shape));
  }
};
REGISTER_XLA_OP(Name("ExpandDims").CompileTimeConstantInput("dim"),
                ExpandDimsOp);

class SqueezeOp : public XlaOpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_17(mht_17_v, 524, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "SqueezeOp");

    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_18(mht_18_v, 533, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    StatusOr<xla::Shape> input_shape = ctx->builder()->GetShape(ctx->Input(0));
    OP_REQUIRES_OK(ctx, input_shape.status());
    xla::Shape shape = input_shape.ValueOrDie();
    int64_t rank = shape.rank();

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    std::vector<int64_t> new_shape;
    // Validate squeeze dims against the input.
    for (int32_t dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -rank && dim < rank),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", rank, " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = rank + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (int i = 0; i < rank; ++i) {
      auto existing_dim = shape.dimensions(i);

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument(
                          "Tried to explicitly squeeze dimension ", i,
                          " but dimension was not 1: ", existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        OP_REQUIRES(
            ctx, !shape.is_dynamic_dimension(i),
            errors::InvalidArgument("Squeeze op does not support bounded "
                                    "dynamic dimensions. Input shape: ",
                                    shape.DebugString()));
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    ctx->SetOutput(0, xla::Reshape(ctx->Input(0), new_shape));
  }

 private:
  std::unordered_set<int32> squeeze_dims_;
};

REGISTER_XLA_OP(Name("Squeeze"), SqueezeOp);

class ZerosLikeOp : public XlaOpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_19(mht_19_v, 597, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "ZerosLikeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_20(mht_20_v, 602, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    if (IsTensorListInput(ctx, 0)) {
      // Input is a TensorList.

      // Check the TensorList input is initialized.
      xla::XlaOp list = ctx->Input(0);
      bool is_initialized;
      OP_REQUIRES_OK(ctx, IsTensorListInitialized(list, &is_initialized));
      OP_REQUIRES(
          ctx, is_initialized,
          errors::InvalidArgument(
              "TensorList input for ZerosLike op is an uninitialized list"));

      auto list_shape_or = ctx->builder()->GetShape(list);
      OP_REQUIRES_OK(ctx, list_shape_or.status());
      const xla::Shape& list_shape = list_shape_or.ValueOrDie();
      std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
      list_dynamic_dims.reserve(list_shape.tuple_shapes_size() - 1);
      for (int i = 0; i < list_shape.tuple_shapes_size() - 1; ++i) {
        // Set dynamic dimension size to 0 for initialization value.
        std::vector<xla::XlaOp> dynamic_dims;
        const xla::Shape& shape = list_shape.tuple_shapes(i);
        auto sub_element = xla::GetTupleElement(list, i);
        for (int64_t dim = 0; dim < shape.dimensions_size(); ++dim) {
          dynamic_dims.push_back(xla::GetDimensionSize(sub_element, dim));
        }
        list_dynamic_dims.push_back(dynamic_dims);
      }
      xla::XlaOp new_list;
      OP_REQUIRES_OK(
          ctx, CreateZerosTensorListWithShape(ctx->builder(), list_shape,
                                              list_dynamic_dims, &new_list));

      xla::XlaOp push_index;
      OP_REQUIRES_OK(ctx, GetTensorListPushIndex(list, &push_index));

      xla::XlaOp result;
      OP_REQUIRES_OK(ctx,
                     SetTensorListPushIndex(new_list, push_index, &result));
      ctx->SetTensorListOutput(0, result);
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      xla::XlaOp input = ctx->Input(0);
      auto input_shape = ctx->InputXlaShape(0).ValueOrDie();
      auto result = xla::Broadcast(zero, input_shape.dimensions());

      // Setting up dynamic dimensions of the broadcast.
      for (int64_t i = 0; i < input_shape.dimensions_size(); ++i) {
        if (input_shape.is_dynamic_dimension(i)) {
          xla::XlaOp input_dynamic_dim = xla::GetDimensionSize(input, i);
          result = xla::SetDimensionSize(result, input_dynamic_dim, i);
        }
      }

      ctx->SetOutput(0, result);
    }
  }
};

REGISTER_XLA_OP(Name("ZerosLike").AllowVariantTypes(), ZerosLikeOp);

class OnesLikeOp : public XlaOpKernel {
 public:
  explicit OnesLikeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_21(mht_21_v, 668, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "OnesLikeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSshape_opDTcc mht_22(mht_22_v, 673, "", "./tensorflow/compiler/tf2xla/kernels/shape_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);

    auto one = XlaHelpers::One(ctx->builder(), input_type(0));
    ctx->SetOutput(0, xla::Broadcast(one, input_shape.dim_sizes()));
  }
};

REGISTER_XLA_OP(Name("OnesLike"), OnesLikeOp);

}  // namespace
}  // namespace tensorflow
