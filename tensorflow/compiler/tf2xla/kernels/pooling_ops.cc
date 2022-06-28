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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc() {
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

// XLA specific pooling ops.

#include <string>

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

// Superclass of pooling ops.
class PoolingOp : public XlaOpKernel {
 public:
  PoolingOp(OpKernelConstruction* ctx, int num_spatial_dims,
            const DataType reduction_type)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        reduction_type_(reduction_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "PoolingOp");

    if (ctx->num_inputs() == 1) {
      std::vector<int32> ksize_int;
      std::vector<int32> stride_int;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_int));
      OP_REQUIRES(ctx, ksize_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_int));
      OP_REQUIRES(ctx, stride_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      for (int i = 0; i < num_dims(); ++i) {
        ksize_.push_back(ksize_int[i]);
        stride_.push_back(stride_int[i]);
      }
    }
    Padding padding;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    OP_REQUIRES(ctx, padding != EXPLICIT,
                errors::Unimplemented(
                    "XLA does not support pooling ops with explicit padding."));
    padding_ = (padding == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(reduction_type_, &xla_reduction_type_));
  }

  int num_dims() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_1(mht_1_v, 254, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "num_dims");
 return num_spatial_dims_ + 2; }

 protected:
  StatusOr<std::vector<int64_t>> GetKernelSize(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return ksize_;
    }
    const TensorShape ksize_shape = ctx->InputShape(1);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(ksize_shape)) {
      return errors::InvalidArgument("ksize must be a vector, not shape ",
                                     ksize_shape.DebugString());
    }
    if (ksize_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window ksize field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64_t> ksize;
    auto status = ctx->ConstantInputAsIntVector(1, &ksize);
    if (!status.ok()) {
      return status;
    }
    return ksize;
  }

  StatusOr<std::vector<int64_t>> GetStride(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return stride_;
    }
    const TensorShape stride_shape = ctx->InputShape(2);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(stride_shape)) {
      return errors::InvalidArgument("stride must be a vector, not shape ",
                                     stride_shape.DebugString());
    }
    if (stride_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window stride field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64_t> stride;
    auto status = ctx->ConstantInputAsIntVector(2, &stride);
    if (!status.ok()) {
      return status;
    }
    return stride;
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64_t> ksize_;
  std::vector<int64_t> stride_;
  xla::Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
  DataType reduction_type_;
  xla::PrimitiveType xla_reduction_type_;
};

// Converts the tensor data format to the one required by the XLA pooling
// library.
xla::TensorFormat XlaTensorFormat(tensorflow::TensorFormat data_format,
                                  int num_spatial_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_2(mht_2_v, 321, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "XlaTensorFormat");

  int num_dims = num_spatial_dims + 2;
  int batch_dimension = GetTensorBatchDimIndex(num_dims, data_format);
  int feature_dimension = GetTensorFeatureDimIndex(num_dims, data_format);
  absl::InlinedVector<int64_t, 4> spatial_dimensions(num_spatial_dims);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    spatial_dimensions[spatial_dim] =
        GetTensorSpatialDimIndex(num_dims, data_format, spatial_dim);
  }
  return xla::TensorFormat(/*batch_dimension=*/batch_dimension,
                           /*feature_dimension=*/feature_dimension,
                           /*spatial_dimensions=*/spatial_dimensions);
}

class MaxPoolOp : public PoolingOp {
 public:
  MaxPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, /*num_spatial_dims=*/num_spatial_dims,
                  /*reduction_type=*/ctx->input_type(0)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_3(mht_3_v, 342, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPoolOp");

    std::string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(ctx, data_format_ != FORMAT_NHWC_VECT_W,
                errors::Unimplemented(
                    "XLA does not support the VECT_NHWC_VECT_W data format. "
                    "Returning unimplemented from MaxPool to keep "
                    "Tensorflow's intended optimized MaxPool here."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_4(mht_4_v, 357, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "Compile");

    auto ksize_or_error = GetKernelSize(ctx);
    OP_REQUIRES_OK(ctx, ksize_or_error.status());
    std::vector<int64_t> ksize = ksize_or_error.ValueOrDie();

    auto stride_or_error = GetStride(ctx);
    OP_REQUIRES_OK(ctx, stride_or_error.status());
    std::vector<int64_t> stride = stride_or_error.ValueOrDie();

    xla::XlaOp input = ctx->Input(0);

    StatusOr<xla::Shape> input_shape = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape.status());

    // For VECT_C max-pool ops, transpose to plain NCHW, do the max-pool, and
    // transpose back.  This isn't necessarily the most efficient algorithm, but
    // it's ok for starters.
    absl::optional<int64_t> vect_width;
    if (data_format_ == FORMAT_NCHW_VECT_C) {
      vect_width = input_shape->dimensions().back();
      input = xla::Collapse(xla::Transpose(input, {0, 1, 4, 2, 3}), {1, 2});

      input_shape = ctx->builder()->GetShape(input);
      OP_REQUIRES_OK(ctx, input_shape.status());
    }

    OP_REQUIRES(ctx, input_shape->dimensions_size() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));
    auto pooling = xla::MaxPool(
        input, ksize, stride, padding_,
        XlaTensorFormat(
            data_format_ == FORMAT_NCHW_VECT_C ? FORMAT_NCHW : data_format_,
            input_shape->dimensions_size() - 2));

    if (data_format_ == FORMAT_NCHW_VECT_C) {
      StatusOr<xla::Shape> result_shape = ctx->builder()->GetShape(pooling);
      OP_REQUIRES_OK(ctx, result_shape.status());

      int64 num_channels = result_shape->dimensions(1);
      OP_REQUIRES(
          ctx, num_channels % *vect_width == 0,
          errors::FailedPrecondition("Result of NCHW_VECT_C op must have "
                                     "channels multiple of ",
                                     *vect_width, ", but was ", num_channels));

      absl::InlinedVector<int64, 5> new_dims(result_shape->dimensions().begin(),
                                             result_shape->dimensions().end());
      new_dims[1] /= *vect_width;
      new_dims.insert(new_dims.begin() + 2, *vect_width);
      pooling =
          xla::Transpose(xla::Reshape(pooling, new_dims), {0, 1, 3, 4, 2});
    }

    ctx->SetOutput(0, pooling);
  }
};

class MaxPool2DOp : public MaxPoolOp {
 public:
  explicit MaxPool2DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/2) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_5(mht_5_v, 422, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPool2DOp");
}
};
REGISTER_XLA_OP(Name("MaxPool"), MaxPool2DOp);
REGISTER_XLA_OP(Name("MaxPoolV2")
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DOp);

class MaxPool3DOp : public MaxPoolOp {
 public:
  explicit MaxPool3DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/3) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_6(mht_6_v, 436, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPool3DOp");
}
};
REGISTER_XLA_OP(Name("MaxPool3D"), MaxPool3DOp);

class AvgPoolOp : public PoolingOp {
 public:
  AvgPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, /*num_spatial_dims=*/num_spatial_dims,
                  /*reduction_type=*/
                  XlaHelpers::SumAccumulationType(ctx->input_type(0))) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_7(mht_7_v, 448, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "AvgPoolOp");

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_8(mht_8_v, 458, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "Compile");

    auto ksize_or_error = GetKernelSize(ctx);
    OP_REQUIRES_OK(ctx, ksize_or_error.status());
    std::vector<int64_t> ksize = ksize_or_error.ValueOrDie();

    auto stride_or_error = GetStride(ctx);
    OP_REQUIRES_OK(ctx, stride_or_error.status());
    std::vector<int64_t> stride = stride_or_error.ValueOrDie();

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, input_shape.dims() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));

    auto xla_data_format =
        XlaTensorFormat(data_format_, input_shape.dims() - 2);
    auto spatial_padding = MakeSpatialPadding(
        input_shape.dim_sizes(), ksize, stride, padding_, xla_data_format);

    // Convert the input to the reduction type.
    auto converted_input =
        ConvertElementType(ctx->Input(0), xla_reduction_type_);
    auto pooling =
        xla::AvgPool(converted_input, ksize, stride, spatial_padding,
                     xla_data_format, padding_ == xla::Padding::kValid);
    // Convert the pooling result back to the input type before returning it.
    ctx->SetOutput(0, ConvertElementType(pooling, ctx->input_xla_type(0)));
  }
};

class AvgPool2DOp : public AvgPoolOp {
 public:
  explicit AvgPool2DOp(OpKernelConstruction* ctx)
      : AvgPoolOp(ctx, /*num_spatial_dims=*/2) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_9(mht_9_v, 495, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "AvgPool2DOp");
}
};
REGISTER_XLA_OP(Name("AvgPool"), AvgPool2DOp);

REGISTER_XLA_OP(Name("AvgPool3D"), MlirXlaOpKernel);

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class MaxPoolGradOp : public XlaOpKernel {
 public:
  MaxPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_10(mht_10_v, 513, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPoolGradOp");

    if (ctx->num_inputs() == 3) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, padding_ != EXPLICIT,
                errors::Unimplemented(
                    "XLA does not support maxpoolgrad with explicit padding."));
    // When determinism is enabled, the use of SelectAndScatter causes a generic
    // error to be raised. We raise a more informative error here before
    // SelectAndScatter is used.
    OP_REQUIRES(
        ctx, !tensorflow::OpDeterminismRequired(),
        errors::Unimplemented("GPU MaxPool gradient ops do not yet have a "
                              "deterministic XLA implementation."));
  }

  int num_dims() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_11(mht_11_v, 534, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "num_dims");
 return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_12(mht_12_v, 539, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "Compile");

    if (ctx->num_inputs() != 3) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 5,
          errors::InvalidArgument("Must supply ksize and stride arguments."));
      const TensorShape ksize_shape = ctx->InputShape(3);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape ",
                                          ksize_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(3, &ksize_));

      const TensorShape stride_shape = ctx->InputShape(4);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape ",
                                          stride_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(4, &stride_));
    }

    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));

    const TensorShape tensor_in_shape = ctx->InputShape(0);
    const TensorShape tensor_out_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // For maxpooling, tensor_in should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_in_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_in must be ", num_dims(),
                                        "-dimensional"));
    OP_REQUIRES(ctx, tensor_out_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_out must be ", num_dims(),
                                        "-dimensional"));
    // For maxpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    // TODO(phawkins): The XLA version doesn't need tensor_out. Investigate
    // whether this is a good time/space tradeoff.
    auto input = ctx->Input(0);
    auto out_backprop = ctx->Input(2);
    // We ensured padding_ is not EXPLICIT in the constructor.
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    // Create a MaxPool operation to check the expected resulting shape, and
    // then throw away the operation because we don't actually need it here.
    TensorShape expected_out_shape;
    auto pooling =
        xla::MaxPool(ctx->Input(0), ksize_, stride_, xla_padding,
                     XlaTensorFormat(data_format_, tensor_in_shape.dims() - 2));
    auto status_or_shape = pooling.builder()->GetShape(pooling);
    OP_REQUIRES_OK(ctx, status_or_shape.status());
    OP_REQUIRES_OK(ctx, XLAShapeToTensorShape(status_or_shape.ValueOrDie(),
                                              &expected_out_shape));
    OP_REQUIRES(ctx, expected_out_shape == out_backprop_shape,
                errors::Unimplemented("The output dimensions do not match the "
                                      "other input values."));

    xla::PrimitiveType element_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(2), &element_type));
    xla::XlaOp init_value = XlaHelpers::Zero(ctx->builder(), input_type(2));
    auto select = CreateScalarGeComputation(element_type, ctx->builder());
    auto scatter = CreateScalarAddComputation(element_type, ctx->builder());
    xla::XlaOp gradients =
        xla::SelectAndScatter(input, select, ksize_, stride_, xla_padding,
                              out_backprop, init_value, scatter);

    ctx->SetOutput(0, gradients);
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64_t> ksize_;
  std::vector<int64_t> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class MaxPool2DGradOp : public MaxPoolGradOp {
 public:
  explicit MaxPool2DGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradOp(ctx, /*num_spatial_dims=*/2) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_13(mht_13_v, 632, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPool2DGradOp");

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPoolGrad"), MaxPool2DGradOp);
REGISTER_XLA_OP(Name("MaxPoolGradV2")
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DGradOp);

REGISTER_XLA_OP(Name("MaxPool3DGrad"), MlirXlaOpKernel);

// Average-pooling gradient
class AvgPoolGradOp : public XlaOpKernel {
 public:
  AvgPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_14(mht_14_v, 654, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "AvgPoolGradOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, padding_ != EXPLICIT,
                errors::Unimplemented(
                    "XLA does not support avgpoolgrad with explicit padding."));
    OP_REQUIRES(ctx, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  int num_dims() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_15(mht_15_v, 682, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "num_dims");
 return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_16(mht_16_v, 687, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "Compile");

    TensorShape gradients_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(0, &gradients_shape,
                                       xla::ValueInferenceMode::kUpperBound));

    const TensorShape out_backprop_shape = ctx->InputShape(1);

    // For avgpooling, tensor_in_shape should have num_dims() dimensions.
    OP_REQUIRES(ctx, gradients_shape.dims() == num_dims(),
                errors::InvalidArgument("orig_input_shape must be ", num_dims(),
                                        "-dimensional"));

    // For avgpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    auto out_backprop = ctx->Input(1);
    std::vector<int64_t> stride_int64s(stride_.begin(), stride_.end());
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
    xla::PrimitiveType xla_reduction_type;
    auto reduction_type = XlaHelpers::SumAccumulationType(ctx->input_type(1));
    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(reduction_type, &xla_reduction_type));
    auto converted_out_backprop =
        xla::ConvertElementType(out_backprop, xla_reduction_type);
    auto xla_data_format =
        XlaTensorFormat(data_format_, gradients_shape.dims() - 2);
    auto padding_values =
        MakeSpatialPadding(gradients_shape.dim_sizes(), ksize_, stride_int64s,
                           xla_padding, xla_data_format);
    auto in_backprop =
        xla::AvgPoolGrad(converted_out_backprop, gradients_shape.dim_sizes(),
                         ksize_, stride_int64s, padding_values, xla_data_format,
                         /*counts_include_padding=*/padding_ == VALID);
    // Convert the pooling result back to the input type before returning it.
    xla::PrimitiveType xla_out_backprop_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(1),
                                                &xla_out_backprop_type));
    ctx->SetOutput(0,
                   xla::ConvertElementType(in_backprop, xla_out_backprop_type));
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64_t> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class AvgPool2DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool2DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/2) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_17(mht_17_v, 746, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "AvgPool2DGradOp");
}
};
REGISTER_XLA_OP(
    Name("AvgPoolGrad").CompileTimeConstantInput("orig_input_shape"),
    AvgPool2DGradOp);

class AvgPool3DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool3DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/3) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_18(mht_18_v, 758, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "AvgPool3DGradOp");
}
};
REGISTER_XLA_OP(
    Name("AvgPool3DGrad").CompileTimeConstantInput("orig_input_shape"),
    AvgPool3DGradOp);

class MaxPoolGradGradOp : public XlaOpKernel {
 public:
  MaxPoolGradGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_19(mht_19_v, 770, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPoolGradGradOp");

    if (ctx->num_inputs() == 3) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(
        ctx, padding_ != EXPLICIT,
        errors::Unimplemented(
            "XLA does not support maxpoolgradgrad with explicit padding."));
  }

  int num_dims() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_20(mht_20_v, 785, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "num_dims");
 return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_21(mht_21_v, 790, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "Compile");

    if (ctx->num_inputs() != 3) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 5,
          errors::InvalidArgument("Must supply ksize and stride arguments."));
      const TensorShape ksize_shape = ctx->InputShape(3);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape ",
                                          ksize_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(3, &ksize_));

      const TensorShape stride_shape = ctx->InputShape(4);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape ",
                                          stride_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(4, &stride_));
    }

    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));

    const TensorShape tensor_in_shape = ctx->InputShape(0);
    const TensorShape tensor_out_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // For maxpooling, tensor_in should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_in_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_in must be ", num_dims(),
                                        "-dimensional"));
    OP_REQUIRES(ctx, tensor_out_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_out must be ", num_dims(),
                                        "-dimensional"));
    // For maxpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    // What we want to compute:
    // Given y = MaxPool(x), and xs_grad = MaxPoolGrad(x, y, ys_grad)
    // MaxPoolGradGrad computes {ys_grad}_grad given x, y, and {xs_grad}_grad.
    //
    // In the regular TF op, this amounts to selecting for each window the
    // incoming backprop value from xs_grad_grad that corresponds to the maximal
    // value in the corresponding window of x.
    //
    // TODO(b/73062247): What we really want is a ReduceWindow with different
    // arrays for index selection vs return value selection--a select-to-gather.
    //
    // Here, we implement a bitwise hack: we use the hi 16 bits of input for
    // separate max pooling alongside each of the hi and lo 16 bits of
    // out_backprop packed into 16 lo bits, which we then glue back together at
    // the end to get a full 32 bits of gradient.
    //
    // This could select the wrong backprop value for two x values that are
    // equally maximal up to the first 16 bits, in which case we are taking the
    // latter.
    //
    // Note that in principle we could use 32 separate maxpools to recover each
    // of 32 bits of the gradient while preserving 31 bits of input for the max
    // pooling criteria; here, we just truncate to the first 16 bits of input.

    auto input = ctx->Input(0);
    auto out_backprop = ctx->Input(2);

    auto b = ctx->builder();

    auto sixteen = xla::ConstantR0<uint32>(b, 16);
    // in (f32) -> round to 7 mantissa bits (bf16)-> 16-high-bit u32.
    //
    // NOTE: Use a ReducePrecision operation instead of a cast to BF16 and back
    // to F32 since the XLA compiler may ignore narrowing casts to floating
    // point types if the debug option xla_allow_excess_precision is set.
    auto in_hi = xla::BitcastConvertType(
        xla::ReducePrecision(input, /*exponent_bits=*/8, /*mantissa_bits=*/7),
        xla::U32);
    auto bp_int = xla::BitcastConvertType(out_backprop, xla::U32);
    auto bp_hi = xla::ShiftRightLogical(bp_int, sixteen);
    auto bp_lo =
        xla::ShiftRightLogical(xla::ShiftLeft(bp_int, sixteen), sixteen);
    auto in_hi_bp_hi = xla::Add(in_hi, bp_hi);  // Want an unsigned add.
    auto in_hi_bp_lo = xla::Add(in_hi, bp_lo);  // Want an unsigned add.

    auto init_value = xla::MinValue(b, xla::F32);
    // We will reduce by taking the maximal value up to 16 bits (ignoring the lo
    // 16 bits of packed-in hi/lo backprop value).
    auto rb = b->CreateSubBuilder("GreaterOrEqOf_ByFirst16Bits");
    {
      // F32 parameters to satisfy lowering type restriction for reduce opcode.
      const xla::Shape scalar = xla::ShapeUtil::MakeShape(xla::F32, {});
      auto lhs = xla::Parameter(rb.get(), 0, scalar, "lhs");
      auto rhs = xla::Parameter(rb.get(), 1, scalar, "rhs");
      auto sixteen = xla::ConstantR0<int32>(rb.get(), 16);
      auto lhs_criteria =
          xla::ShiftLeft(xla::ShiftRightLogical(
                             xla::BitcastConvertType(lhs, xla::S32), sixteen),
                         sixteen);
      auto rhs_criteria =
          xla::ShiftLeft(xla::ShiftRightLogical(
                             xla::BitcastConvertType(rhs, xla::S32), sixteen),
                         sixteen);
      // Must use a F32 comparison, because S32 would not work for negatives.
      xla::Select(xla::Ge(xla::BitcastConvertType(lhs_criteria, xla::F32),
                          xla::BitcastConvertType(rhs_criteria, xla::F32)),
                  lhs, rhs);
    }
    auto reduce = rb->BuildAndNoteError();
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
    auto pooled_hi =
        xla::ReduceWindow(xla::BitcastConvertType(in_hi_bp_hi, xla::F32),
                          init_value, reduce, ksize_, stride_, xla_padding);
    auto pooled_lo =
        xla::ReduceWindow(xla::BitcastConvertType(in_hi_bp_lo, xla::F32),
                          init_value, reduce, ksize_, stride_, xla_padding);
    auto grads_hi =
        xla::ShiftLeft(xla::BitcastConvertType(pooled_hi, xla::U32), sixteen);
    auto grads_lo = xla::ShiftRightLogical(
        xla::ShiftLeft(xla::BitcastConvertType(pooled_lo, xla::U32), sixteen),
        sixteen);
    auto grads = xla::Add(grads_hi, grads_lo);  // Want an unsigned add.

    xla::PrimitiveType element_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(2), &element_type));
    ctx->SetOutput(0, xla::BitcastConvertType(grads, element_type));
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64_t> ksize_;
  std::vector<int64_t> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class MaxPool2DGradGradOp : public MaxPoolGradGradOp {
 public:
  explicit MaxPool2DGradGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradGradOp(ctx, /*num_spatial_dims=*/2) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_22(mht_22_v, 938, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPool2DGradGradOp");

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPoolGradGrad").TypeConstraint("T", DT_FLOAT),
                MaxPool2DGradGradOp);
REGISTER_XLA_OP(Name("MaxPoolGradGradV2")
                    .TypeConstraint("T", DT_FLOAT)
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DGradGradOp);

class MaxPool3DGradGradOp : public MaxPoolGradGradOp {
 public:
  explicit MaxPool3DGradGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradGradOp(ctx, /*num_spatial_dims=*/3) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpooling_opsDTcc mht_23(mht_23_v, 959, "", "./tensorflow/compiler/tf2xla/kernels/pooling_ops.cc", "MaxPool3DGradGradOp");

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPool3DGradGrad").TypeConstraint("T", DT_FLOAT),
                MaxPool3DGradGradOp);

}  // anonymous namespace
}  // namespace tensorflow
