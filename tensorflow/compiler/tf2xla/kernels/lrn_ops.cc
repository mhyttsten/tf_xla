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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc() {
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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

// Local response normalization
class LRNOp : public XlaOpKernel {
 public:
  explicit LRNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2xla/kernels/lrn_ops.cc", "LRNOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("depth_radius", &depth_radius_));

    // TODO(phawkins): handle non-float types for attributes.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/tf2xla/kernels/lrn_ops.cc", "Compile");

    const TensorShape in_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, in_shape.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);

    // sqr_sum[a, b, c, d] =
    //    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    // output = input / (bias + alpha * sqr_sum) ** beta

    // We use a window of depth_radius_ * 2 + 1, to account for the current
    // element and a depth_radius_ on either side.
    auto accumulation_type = XlaHelpers::SumAccumulationType(input_type(0));
    auto converted = XlaHelpers::ConvertElementType(input, accumulation_type);
    auto squared = xla::Mul(converted, converted);
    auto reduce = xla::ReduceWindow(
        squared, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto sqr_sum = XlaHelpers::ConvertElementType(reduce, input_type(0));

    auto scale = xla::Pow(
        xla::Add(xla::ConstantR0<float>(builder, bias_),
                 xla::Mul(xla::ConstantR0<float>(builder, alpha_), sqr_sum)),
        xla::ConstantR0<float>(builder, -beta_));

    ctx->SetOutput(0, xla::Mul(input, scale));
  }

 private:
  int64_t depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP(Name("LRN"), LRNOp);

class LRNGradOp : public XlaOpKernel {
 public:
  explicit LRNGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/tf2xla/kernels/lrn_ops.cc", "LRNGradOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("depth_radius", &depth_radius_));

    // TODO(phawkins): handle non-float types for attributes.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlrn_opsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/tf2xla/kernels/lrn_ops.cc", "Compile");

    const TensorShape in_grads_shape = ctx->InputShape(0);
    const TensorShape in_image_shape = ctx->InputShape(1);
    const TensorShape out_image_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx, in_grads_shape.dims() == 4 && in_image_shape.dims() == 4,
                errors::InvalidArgument("inputs must be 4-dimensional"));
    const int64_t batch = in_grads_shape.dim_size(0);
    const int64_t rows = in_grads_shape.dim_size(1);
    const int64_t cols = in_grads_shape.dim_size(2);
    const int64_t depth = in_grads_shape.dim_size(3);
    OP_REQUIRES(
        ctx, in_image_shape.dim_size(0) == batch &&
                 in_image_shape.dim_size(1) == rows &&
                 in_image_shape.dim_size(2) == cols &&
                 in_image_shape.dim_size(3) == depth &&
                 out_image_shape.dim_size(0) == batch &&
                 out_image_shape.dim_size(1) == rows &&
                 out_image_shape.dim_size(2) == cols &&
                 out_image_shape.dim_size(3) == depth,
        errors::InvalidArgument(
            "input_grads, input_image, and out_image should have the same "
            "shape"));

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp in_grads = ctx->Input(0);
    xla::XlaOp in_image = ctx->Input(1);
    xla::XlaOp out_image = ctx->Input(2);

    // This code is ported from tensorflow/core/kernels/lrn_op.cc. In Python
    // pseudo-code, the Eigen code does this for each spatial position:
    // grads = [0.0] * depth
    // for j in range(depth):
    //   depth_begin = max(0, j - depth_radius)
    //   depth_end = min(depth, j + depth_radius + 1)
    //
    //   norm = 0
    //   for k in range(depth_begin, depth_end):
    //     norm += in_image[k] * in_image[k]
    //   norm = alpha * norm + bias
    //
    //   for k in range(depth_begin, depth_end):
    //     dyi = -2.0 * alpha * beta * in_image[k] * out_image[j] / norm
    //     if k == j:
    //       dyi += norm ** (-beta)
    //     dyi *= out_grads[j]
    //     grads[k] += dyi

    auto accumulation_type = XlaHelpers::SumAccumulationType(input_type(0));
    auto converted =
        XlaHelpers::ConvertElementType(in_image, accumulation_type);
    auto squared = xla::Mul(converted, converted);
    auto reduce = xla::ReduceWindow(
        squared, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto sqr_sum = XlaHelpers::ConvertElementType(reduce, input_type(0));

    auto norm =
        xla::Add(xla::ConstantR0<float>(builder, bias_),
                 xla::Mul(xla::ConstantR0<float>(builder, alpha_), sqr_sum));

    auto dy = xla::Mul(
        xla::Mul(xla::ConstantR0<float>(builder, -2.0f * alpha_ * beta_),
                 xla::Div(out_image, norm)),
        in_grads);

    auto converted_dy = XlaHelpers::ConvertElementType(dy, accumulation_type);
    auto dy_reduce = xla::ReduceWindow(
        converted_dy, XlaHelpers::Zero(builder, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type),
        /* window_dimensions = */ {1, 1, 1, depth_radius_ * 2 + 1},
        /* window_strides = */ {1, 1, 1, 1}, xla::Padding::kSame);
    auto dy_reduced = XlaHelpers::ConvertElementType(dy_reduce, input_type(0));

    xla::XlaOp gradients = xla::Add(
        xla::Mul(in_image, dy_reduced),
        xla::Mul(in_grads,
                 xla::Pow(norm, xla::ConstantR0<float>(builder, -beta_))));

    ctx->SetOutput(0, gradients);
  }

 private:
  int64_t depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_XLA_OP(Name("LRNGrad"), LRNGradOp);

}  // anonymous namespace
}  // namespace tensorflow
