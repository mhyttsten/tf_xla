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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc() {
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

// XLA implementation of BatchNorm operations.
#include <algorithm>
#include <numeric>
#include <string>

#include "tensorflow/compiler/tf2xla/kernels/relu_op.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class FusedBatchNormOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx, false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "FusedBatchNormOp");
}

  FusedBatchNormOp(OpKernelConstruction* ctx, bool is_batch_norm_ex)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "FusedBatchNormOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("exponential_avg_factor", &exponential_avg_factor_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));

    if (is_batch_norm_ex) {
      int num_side_inputs;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(ctx, num_side_inputs >= 0 && num_side_inputs <= 1,
                  errors::InvalidArgument(
                      "FusedBatchNormEx supports at most 1 side input."));
      add_side_input_ = (num_side_inputs == 1);
      string activation_mode;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode));
      OP_REQUIRES(ctx,
                  activation_mode == "Identity" || activation_mode == "Relu",
                  errors::InvalidArgument(
                      "Unsupported FusedBatchNormEx activation mode: ",
                      activation_mode));
      apply_relu_ = (activation_mode == "Relu");
    } else {
      add_side_input_ = false;
      apply_relu_ = false;
    }
    is_on_gpu_ = ctx->device_type().type_string() == DEVICE_GPU_XLA_JIT;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "Compile");
 CompileImpl(ctx); }

 protected:
  virtual void CompileImpl(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_3(mht_3_v, 255, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "CompileImpl");

    xla::XlaBuilder* const b = ctx->builder();
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::PrimitiveType scale_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(1), &scale_type));

    xla::XlaOp input = ctx->Input(0);
    TensorShape input_shape = ctx->InputShape(0);

    int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. As a workaround, cast everything to the statistics type (which
    // may be more precise than the input type).
    input = xla::ConvertElementType(input, scale_type);

    if (is_training_) {
      xla::XlaOp output = xla::BatchNormTraining(
          input, ctx->Input(1), ctx->Input(2), epsilon_, feature_index);

      // In training mode, outputs the normalized value as well as the
      // calculated mean and variance. Optionally we add side input and apply
      // relu activation.
      xla::XlaOp converted =
          xla::ConvertElementType(xla::GetTupleElement(output, 0), input_type);
      if (add_side_input_ && apply_relu_) {
        ctx->SetOutput(0, xla::Relu(xla::Add(ctx->Input(5), converted)));
      } else if (apply_relu_) {
        ctx->SetOutput(0, xla::Relu(converted));
      } else {
        ctx->SetOutput(0, converted);
      }

      xla::XlaOp variance = xla::GetTupleElement(output, 2);
      // Apply Bessel's correction.
      int total_input_size = ctx->InputShape(0).num_elements();
      int total_scale_size = ctx->InputShape(1).num_elements();
      int sample_size =
          total_scale_size > 0 ? total_input_size / total_scale_size : 0;
      int sample_size_minus_one = std::max(1, sample_size - 1);
      double factor = static_cast<double>(sample_size) /
                      static_cast<double>(sample_size_minus_one);

      constexpr int kVarianceOutputIndex = 2;
      xla::XlaOp corrected =
          xla::Mul(variance, xla::ScalarLike(variance, factor));
      if (input_shape.num_elements() == 0) {
        auto status_or_output_shape = b->GetShape(corrected);
        OP_REQUIRES_OK(ctx, status_or_output_shape.status());
        ctx->SetOutput(1, xla::GetTupleElement(output, 1));
        ctx->SetOutput(
            kVarianceOutputIndex,
            xla::Broadcast(
                xla::NanValue(b, ctx->output_xla_type(kVarianceOutputIndex)),
                status_or_output_shape.ValueOrDie().dimensions()));

      } else {
        if (exponential_avg_factor_ == 1.0f) {
          ctx->SetOutput(1, xla::GetTupleElement(output, 1));
          ctx->SetOutput(2, corrected);
        } else {
          xla::XlaOp old_mean = ctx->Input(3);
          xla::XlaOp alpha =
              xla::ScalarLike(old_mean, 1.0f - exponential_avg_factor_);
          xla::XlaOp beta = xla::ScalarLike(old_mean, exponential_avg_factor_);
          // new_running_mean = alpha * old_mean + beta * batch_mean.
          xla::XlaOp new_running_mean =
              xla::Add(xla::Mul(old_mean, alpha),
                       xla::Mul(xla::GetTupleElement(output, 1), beta));
          ctx->SetOutput(1, new_running_mean);

          xla::XlaOp old_variance = ctx->Input(4);
          xla::XlaOp new_running_variance = xla::Add(
              xla::Mul(old_variance, alpha), xla::Mul(corrected, beta));
          // new_running_variance = alpha * old_variance + beta *
          // batch_variance.
          ctx->SetOutput(2, new_running_variance);
        }
      }

      // Output 3 and 4 for "FusedBatchNorm" are currently marked as "reserved
      // space 1 & 2". They are used to pass the per-batch mean and
      // variance to the gradient. Here we maintain the same behavior by setting
      // them to the mean and variance calculated by BatchNormTraining.
      ctx->SetOutput(3, xla::GetTupleElement(output, 1));
      if (is_on_gpu_) {
        // The last two outputs from the FusedBatchNorm training TensorFlow GPU
        // op are implementation defined.  For now we rely on the in-practice
        // behavior of the op:
        //   output 3 is the mean
        //   output 4 is rsqrt(variance + epsilon)
        ctx->SetOutput(4, xla::Rsqrt(xla::Add(
                              variance, xla::ScalarLike(variance, epsilon_))));
      } else {
        ctx->SetOutput(4, variance);
      }
    } else {
      xla::XlaOp output = xla::BatchNormInference(
          input, ctx->Input(1), ctx->Input(2), ctx->Input(3), ctx->Input(4),
          epsilon_, feature_index);

      xla::XlaOp converted = xla::ConvertElementType(output, input_type);
      if (add_side_input_ && apply_relu_) {
        ctx->SetOutput(0, xla::Relu(xla::Add(ctx->Input(5), converted)));
      } else if (apply_relu_) {
        ctx->SetOutput(0, xla::Relu(converted));
      } else {
        ctx->SetOutput(0, converted);
      }

      // Directly send input to output as mean and variance in inference mode.
      ctx->SetOutput(1, ctx->Input(3));
      ctx->SetOutput(2, ctx->Input(4));
      ctx->SetOutput(3, ctx->Input(3));
      ctx->SetOutput(4, ctx->Input(4));
    }
  }

 private:
  float epsilon_;
  TensorFormat data_format_;
  bool is_training_;
  float exponential_avg_factor_;
  bool add_side_input_;
  bool apply_relu_;
  bool is_on_gpu_;
};

class FusedBatchNormOpV3 : public FusedBatchNormOp {
 public:
  explicit FusedBatchNormOpV3(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_4(mht_4_v, 393, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "FusedBatchNormOpV3");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_5(mht_5_v, 398, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "Compile");

    FusedBatchNormOp::CompileImpl(ctx);
    if (!ctx->status().ok()) {
      return;
    }
    ctx->SetConstantOutput(5, Tensor());
  }
};

class FusedBatchNormOpEx : public FusedBatchNormOp {
 public:
  explicit FusedBatchNormOpEx(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx, /*is_batch_norm_ex=*/true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_6(mht_6_v, 413, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "FusedBatchNormOpEx");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_7(mht_7_v, 418, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "Compile");

    FusedBatchNormOp::CompileImpl(ctx);
    if (!ctx->status().ok()) {
      return;
    }
    ctx->SetConstantOutput(5, Tensor());
  }
};

REGISTER_XLA_OP(Name("FusedBatchNorm"), FusedBatchNormOp);
REGISTER_XLA_OP(Name("FusedBatchNormV2"), FusedBatchNormOp);
REGISTER_XLA_OP(Name("FusedBatchNormV3"), FusedBatchNormOpV3);
REGISTER_XLA_OP(Name("_FusedBatchNormEx"), FusedBatchNormOpEx);

class FusedBatchNormGradOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_8(mht_8_v, 437, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "FusedBatchNormGradOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));
    is_on_gpu_ = ctx->device_type().type_string() == DEVICE_GPU_XLA_JIT;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatch_norm_opDTcc mht_9(mht_9_v, 451, "", "./tensorflow/compiler/tf2xla/kernels/batch_norm_op.cc", "Compile");

    xla::XlaBuilder* const b = ctx->builder();
    DataType input_dtype = ctx->input_type(0);
    DataType scale_dtype = ctx->input_type(2);

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. For now, cast everything to the statistics type (which
    // may be more precise than the input type).
    auto grad_backprop =
        XlaHelpers::ConvertElementType(ctx->Input(0), scale_dtype);
    auto activations =
        XlaHelpers::ConvertElementType(ctx->Input(1), scale_dtype);
    auto scale = ctx->Input(2);
    auto mean = ctx->Input(3);
    auto var = ctx->Input(4);

    const int input_dims = ctx->InputShape(0).dims();
    const int feature_index =
        GetTensorFeatureDimIndex(input_dims, data_format_);

    xla::XlaOp x_backprop;
    xla::XlaOp scale_backprop;
    xla::XlaOp offset_backprop;
    if (is_training_) {
      if (is_on_gpu_) {
        // The last two inputs to the FusedBatchNormGrad training TensorFlow GPU
        // op are implementation defined.  For now we rely on the in-practice
        // behavior of the op: input 3 is the mean input 4 is rsqrt(variance +
        // epsilon)
        //
        // The XLA op expects:
        //   input 3 is the mean
        //   input 4 is the variance
        //
        // so we adjust input 4 here.
        xla::XlaOp one = xla::ScalarLike(var, 1.0f);
        xla::XlaOp epsilon = xla::ScalarLike(var, epsilon_);
        var = xla::Sub(one / (var * var), epsilon);
      }

      xla::XlaOp output =
          xla::BatchNormGrad(activations, scale, mean, var, grad_backprop,
                             epsilon_, feature_index);

      x_backprop = xla::GetTupleElement(output, 0);
      scale_backprop = xla::GetTupleElement(output, 1);
      offset_backprop = xla::GetTupleElement(output, 2);
    } else {
      // Reduce over all dimensions except the feature dim.
      std::vector<int64_t> reduction_dims(input_dims - 1);
      std::iota(reduction_dims.begin(), reduction_dims.begin() + feature_index,
                0);
      std::iota(reduction_dims.begin() + feature_index, reduction_dims.end(),
                feature_index + 1);
      // offset_backprop  = sum(y_backprop)
      // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var +
      // epsilon))
      // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))
      const DataType accumulation_type =
          XlaHelpers::SumAccumulationType(scale_dtype);
      auto converted =
          XlaHelpers::ConvertElementType(grad_backprop, accumulation_type);
      auto reduce =
          xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                      *ctx->GetOrCreateAdd(accumulation_type), reduction_dims);
      offset_backprop = XlaHelpers::ConvertElementType(reduce, scale_dtype);

      // scratch1 = rsqrt(pop_var + epsilon)
      auto epsilon = XlaHelpers::FloatLiteral(b, scale_dtype, epsilon_);
      auto scratch1 = xla::Rsqrt(xla::Add(var, epsilon));

      // scratch2 = sum(y_backprop * (x - mean))
      auto mul =
          xla::Mul(grad_backprop, xla::Sub(activations, mean, {feature_index}));
      converted = XlaHelpers::ConvertElementType(mul, accumulation_type);
      reduce =
          xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                      *ctx->GetOrCreateAdd(accumulation_type), reduction_dims);
      auto scratch2 = XlaHelpers::ConvertElementType(reduce, scale_dtype);

      x_backprop =
          xla::Mul(grad_backprop, xla::Mul(scratch1, scale), {feature_index});
      scale_backprop = xla::Mul(scratch1, scratch2);
    }

    ctx->SetOutput(0, XlaHelpers::ConvertElementType(x_backprop, input_dtype));
    ctx->SetOutput(1, scale_backprop);
    ctx->SetOutput(2, offset_backprop);
    ctx->SetConstantOutput(3, Tensor());
    ctx->SetConstantOutput(4, Tensor());
  }

 private:
  TensorFormat data_format_;
  float epsilon_;
  bool is_training_;
  bool is_on_gpu_;
};

REGISTER_XLA_OP(Name("FusedBatchNormGrad"), FusedBatchNormGradOp);
REGISTER_XLA_OP(Name("FusedBatchNormGradV2"), FusedBatchNormGradOp);
REGISTER_XLA_OP(Name("FusedBatchNormGradV3"), FusedBatchNormGradOp);

}  // namespace
}  // namespace tensorflow
