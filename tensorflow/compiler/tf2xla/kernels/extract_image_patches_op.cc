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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSextract_image_patches_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSextract_image_patches_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSextract_image_patches_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

class ExtractImagePatchesOp : public XlaOpKernel {
 public:
  explicit ExtractImagePatchesOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSextract_image_patches_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/extract_image_patches_op.cc", "ExtractImagePatchesOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksizes", &ksizes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rates", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSextract_image_patches_opDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2xla/kernels/extract_image_patches_op.cc", "Compile");

    const TensorFormat data_format = FORMAT_NHWC;
    const int num_dims = ksizes_.size();

    OP_REQUIRES(
        ctx, num_dims >= 3,
        errors::InvalidArgument("Kernel size must have at least 3 dimensions"));
    const int num_spatial_dims = num_dims - 2;

    OP_REQUIRES(ctx, strides_.size() == num_dims,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims, " dimensions"));
    OP_REQUIRES(ctx, dilations_.size() == num_dims,
                errors::InvalidArgument("Dilations field must "
                                        "specify ",
                                        num_dims, " dimensions"));

    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
    OP_REQUIRES(
        ctx, ksizes_[batch_dim] == 1 && ksizes_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "kernel sizes > 1 in the batch and depth "
                              "dimensions."));
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not support "
                              "dilations in the batch and depth dimensions."));

    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      OP_REQUIRES(
          ctx, ksizes_[input_dim] >= 0,
          errors::Unimplemented("Kernel size values must be non-negative; ", i,
                                "th spatial dimension had dilation ",
                                dilations_[input_dim]));
      OP_REQUIRES(ctx, strides_[input_dim] >= 1,
                  errors::Unimplemented("Stride values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
      OP_REQUIRES(ctx, dilations_[input_dim] >= 1,
                  errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
    }

    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &type));

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == num_dims,
        errors::InvalidArgument("input must be ", num_dims, "-dimensional",
                                input_shape.DebugString()));
    const int64_t depth = input_shape.dim_size(feature_dim);

    xla::XlaBuilder* builder = ctx->builder();

    // The following code is equivalent to:
    // eye = np.eye(kH * kW * D).reshape([kH, kW, D, kH * kW * kD])
    int64_t kernel_size = 1;
    std::vector<int64_t> kernel_shape(num_dims, 1);
    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      kernel_shape[i] = ksizes_[input_dim];
      kernel_size *= ksizes_[input_dim];
    }
    kernel_shape[num_spatial_dims] = 1;
    kernel_shape[num_spatial_dims + 1] = kernel_size * depth;
    xla::Shape iota_kernel_shape =
        xla::ShapeUtil::MakeShape(xla::S32, {kernel_size, depth, kernel_size});
    xla::XlaOp filter =
        xla::Reshape(xla::ConvertElementType(
                         xla::Eq(xla::Iota(builder, iota_kernel_shape, 0),
                                 xla::Iota(builder, iota_kernel_shape, 2)),
                         type),
                     kernel_shape);

    xla::ConvolutionDimensionNumbers dims;
    std::vector<int64_t> window_strides(num_spatial_dims);
    std::vector<int64_t> lhs_dilation(num_spatial_dims, 1);
    std::vector<int64_t> rhs_dilation(num_spatial_dims);
    std::vector<std::pair<int64_t, int64_t>> padding(num_spatial_dims);

    dims.set_input_batch_dimension(batch_dim);
    dims.set_output_batch_dimension(batch_dim);
    dims.set_input_feature_dimension(feature_dim);
    dims.set_output_feature_dimension(feature_dim);
    dims.set_kernel_input_feature_dimension(num_spatial_dims);
    dims.set_kernel_output_feature_dimension(num_spatial_dims + 1);

    for (int i = 0; i < num_spatial_dims; ++i) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      dims.add_input_spatial_dimensions(dim);
      dims.add_kernel_spatial_dimensions(i);
      dims.add_output_spatial_dimensions(dim);
      window_strides[i] = strides_.at(dim);
      rhs_dilation[i] = dilations_.at(dim);

      int64_t unused_output_size;
      OP_REQUIRES_OK(
          ctx, GetWindowedOutputSizeVerboseV2(
                   input_shape.dim_size(dim), ksizes_[dim], rhs_dilation[i],
                   window_strides[i], padding_, &unused_output_size,
                   &padding[i].first, &padding[i].second));
    }

    xla::XlaOp conv =
        xla::ConvGeneralDilated(ctx->Input(0), filter, window_strides, padding,
                                lhs_dilation, rhs_dilation, dims, depth);
    // Feature group convolution, will end up with the kernel_size change more
    // rapidly than the depth. Reshape, transpose and reshape to reorder them.
    std::vector<int64_t> conv_dims =
        xla::SpanToVector(builder->GetShape(conv).ValueOrDie().dimensions());
    conv_dims.back() = depth;
    conv_dims.push_back(kernel_size);
    conv = xla::TransposeInMinorDims(xla::Reshape(conv, conv_dims));
    conv_dims.pop_back();
    conv_dims.back() *= kernel_size;
    conv = xla::Reshape(conv, conv_dims);
    ctx->SetOutput(0, conv);
  }

 protected:
  std::vector<int32> ksizes_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ExtractImagePatchesOp);
};

// We don't support integers for the convolution used in the implementation of
// this op, so we limit the supported types.
REGISTER_XLA_OP(
    Name("ExtractImagePatches").TypeConstraint("T", GetXlaConvTypes()),
    ExtractImagePatchesOp);

}  // namespace
}  // namespace tensorflow
