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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc() {
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

// XLA-specific Ops for 2D convolution.

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class ConvOp : public XlaOpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* ctx, int num_spatial_dims,
                  bool depthwise)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "ConvOp");

    StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Compile");

    StatusOr<xla::XlaOp> conv = MakeXlaForwardConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), ctx->Input(1), attrs_);
    OP_REQUIRES_OK(ctx, conv.status());
    ctx->SetOutput(0, conv.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvOp);
};

class Conv2DOp : public ConvOp {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv2DOp");
}
};
REGISTER_XLA_OP(Name("Conv2D").TypeConstraint("T", GetXlaConvTypes()),
                Conv2DOp);

class Conv3DOp : public ConvOp {
 public:
  explicit Conv3DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv3DOp");
}
};
REGISTER_XLA_OP(Name("Conv3D").TypeConstraint("T", GetXlaConvTypes()),
                Conv3DOp);

class DepthwiseConv2DOp : public ConvOp {
 public:
  explicit DepthwiseConv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "DepthwiseConv2DOp");
}
};
REGISTER_XLA_OP(
    Name("DepthwiseConv2dNative").TypeConstraint("T", GetXlaConvTypes()),
    DepthwiseConv2DOp);

// Backprop for input.
class ConvBackpropInputOp : public XlaOpKernel {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* ctx, int num_spatial_dims,
                               bool depthwise)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_5(mht_5_v, 282, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "ConvBackpropInputOp");

    StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_6(mht_6_v, 292, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Compile");

    TensorShape input_tensor_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(0, &input_tensor_shape,
                                       xla::ValueInferenceMode::kUpperBound));
    xla::Shape input_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(1), input_tensor_shape);
    OP_REQUIRES(ctx, input_shape.rank() == attrs_.num_spatial_dims + 2,
                errors::InvalidArgument(
                    "The rank of the specified input shape must be "
                    "num_spatial_dims + 2. Expected ",
                    attrs_.num_spatial_dims + 2, " got ", input_shape.rank()));
    xla::XlaOp input_sizes = ctx->Input(0);
    StatusOr<xla::XlaOp> in_backprop = MakeXlaBackpropInputConvOp(
        ctx->op_kernel().type_string(), input_shape, ctx->Input(1),
        ctx->Input(2), attrs_, nullptr, &input_sizes);
    OP_REQUIRES_OK(ctx, in_backprop.status());
    ctx->SetOutput(0, in_backprop.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropInputOp);
};

class Conv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_7(mht_7_v, 325, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv2DBackpropInputOp");
}
};
REGISTER_XLA_OP(Name("Conv2DBackpropInput")
                    .CompileTimeConstantInput("input_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                Conv2DBackpropInputOp);

class Conv3DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_8(mht_8_v, 338, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv3DBackpropInputOp");
}
};
REGISTER_XLA_OP(Name("Conv3DBackpropInputV2")
                    .CompileTimeConstantInput("input_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                Conv3DBackpropInputOp);

class DepthwiseConv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit DepthwiseConv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_9(mht_9_v, 351, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "DepthwiseConv2DBackpropInputOp");
}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropInput")
                    .CompileTimeConstantInput("input_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                DepthwiseConv2DBackpropInputOp);

class ConvBackpropFilterOp : public XlaOpKernel {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* ctx, int num_spatial_dims,
                                bool depthwise)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_10(mht_10_v, 365, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "ConvBackpropFilterOp");

    StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_11(mht_11_v, 375, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Compile");

    TensorShape filter_tensor_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(1, &filter_tensor_shape,
                                       xla::ValueInferenceMode::kUpperBound));
    xla::Shape filter_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(0), filter_tensor_shape);

    StatusOr<xla::XlaOp> filter_backprop = MakeXlaBackpropFilterConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), filter_shape,
        ctx->Input(2), attrs_);
    OP_REQUIRES_OK(ctx, filter_backprop.status());
    ctx->SetOutput(0, filter_backprop.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropFilterOp);
};

class Conv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_12(mht_12_v, 403, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv2DBackpropFilterOp");

  }
};
REGISTER_XLA_OP(Name("Conv2DBackpropFilter")
                    .CompileTimeConstantInput("filter_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                Conv2DBackpropFilterOp);

class Conv3DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_13(mht_13_v, 417, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "Conv3DBackpropFilterOp");

  }
};
REGISTER_XLA_OP(Name("Conv3DBackpropFilterV2")
                    .CompileTimeConstantInput("filter_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                Conv3DBackpropFilterOp);

class DepthwiseConv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit DepthwiseConv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconv_opsDTcc mht_14(mht_14_v, 431, "", "./tensorflow/compiler/tf2xla/kernels/conv_ops.cc", "DepthwiseConv2DBackpropFilterOp");
}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropFilter")
                    .CompileTimeConstantInput("filter_sizes")
                    .TypeConstraint("T", GetXlaConvTypes()),
                DepthwiseConv2DBackpropFilterOp);

}  // namespace
}  // namespace tensorflow
