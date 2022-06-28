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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// Gymnastics with nudged zero point is to ensure that the real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
void CpuNudge(const float min, const float max, const float quant_min,
              const float quant_max, float* nudged_min, float* nudged_max,
              float* scale) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "CpuNudge");

  *scale = (max - min) / (quant_max - quant_min);

  const float zero_point_from_min = quant_min - min / *scale;
  float nudged_zero_point;
  if (zero_point_from_min <= quant_min) {
    nudged_zero_point = quant_min;
  } else if (zero_point_from_min >= quant_max) {
    nudged_zero_point = quant_max;
  } else {
    nudged_zero_point = std::round(zero_point_from_min);
  }

  *nudged_min = (quant_min - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max - nudged_zero_point) * (*scale);
}

// An XLA version of CpuNudge().
void XlaNudge(xla::XlaBuilder* b, const DataType data_type,
              const xla::XlaOp& min, const xla::XlaOp& max,
              const float quant_min_value, const float quant_max_value,
              xla::XlaOp* nudged_min, xla::XlaOp* nudged_max,
              xla::XlaOp* scale) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "XlaNudge");

  *scale = xla::Div(xla::Sub(max, min),
                    XlaHelpers::FloatLiteral(
                        b, data_type, quant_max_value - quant_min_value));
  xla::XlaOp quant_min =
      XlaHelpers::FloatLiteral(b, data_type, quant_min_value);
  xla::XlaOp zero_point_from_min = xla::Sub(quant_min, xla::Div(min, *scale));
  xla::XlaOp quant_max =
      XlaHelpers::FloatLiteral(b, data_type, quant_max_value);
  xla::XlaOp nudged_zero_point =
      xla::Select(xla::Le(zero_point_from_min, quant_min), quant_min,
                  xla::Select(xla::Ge(zero_point_from_min, quant_max),
                              quant_max, xla::Round(zero_point_from_min)));
  *nudged_min = xla::Mul(xla::Sub(quant_min, nudged_zero_point), *scale);
  *nudged_max = xla::Mul(xla::Sub(quant_max, nudged_zero_point), *scale);
}

xla::XlaOp Quantize(xla::XlaBuilder* b, const xla::XlaOp& input,
                    const DataType data_type,
                    const xla::XlaOp& nudged_input_min,
                    const xla::XlaOp& nudged_input_max,
                    const xla::XlaOp& input_scale) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Quantize");

  xla::XlaOp one = XlaHelpers::FloatLiteral(b, data_type, 1.0f);
  xla::XlaOp inv_scale = xla::Div(one, input_scale);
  xla::XlaOp half = XlaHelpers::FloatLiteral(b, data_type, 0.5f);

  xla::XlaOp clamped = xla::Clamp(nudged_input_min, input, nudged_input_max);
  xla::XlaOp clamped_shifted = xla::Sub(clamped, nudged_input_min);
  xla::XlaOp rounded =
      xla::Floor(xla::Add(xla::Mul(clamped_shifted, inv_scale), half));
  return xla::Add(xla::Mul(rounded, input_scale), nudged_input_min);
}

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxArgs"), MlirXlaOpKernel);

class FakeQuantWithMinMaxArgsGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxArgsGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_3(mht_3_v, 269, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "FakeQuantWithMinMaxArgsGradOp");

    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    const float quant_min = narrow_range ? 1 : 0;
    const float quant_max = (1 << num_bits) - 1;

    float input_min, input_max, scale;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min", &input_min));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max", &input_max));
    CpuNudge(input_min, input_max, quant_min, quant_max, &nudged_input_min_,
             &nudged_input_max_, &scale);
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Compile");

    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp nudged_input_min =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_min_);
    xla::XlaOp nudged_input_max =
        XlaHelpers::FloatLiteral(b, data_type, nudged_input_max_);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zeroes = xla::Broadcast(XlaHelpers::Zero(b, data_type),
                                       gradient_shape.dim_sizes());
    xla::XlaOp output = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output);
  }

 private:
  float nudged_input_min_;
  float nudged_input_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxArgsGradient"),
                FakeQuantWithMinMaxArgsGradOp);

class FakeQuantWithMinMaxVarsOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_5(mht_5_v, 325, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "FakeQuantWithMinMaxVarsOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ >= 2 && num_bits_ <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    quant_min_ = narrow_range_ ? 1 : 0;
    quant_max_ = (1 << num_bits_) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_6(mht_6_v, 339, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);
    xla::XlaOp input_min = ctx->Input(1);
    xla::XlaOp input_max = ctx->Input(2);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp output = Quantize(b, input, data_type, nudged_input_min,
                                 nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
  int num_bits_;
  bool narrow_range_;
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVars"), FakeQuantWithMinMaxVarsOp);

class FakeQuantWithMinMaxVarsGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_7(mht_7_v, 370, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "FakeQuantWithMinMaxVarsGradOp");

    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_8(mht_8_v, 386, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Compile");

    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);
    const DataType accumulation_type =
        XlaHelpers::SumAccumulationType(data_type);
    xla::XlaOp input_min = ctx->Input(2);
    xla::XlaOp input_max = ctx->Input(3);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    xla::XlaOp zeroes = xla::Broadcast(zero, gradient_shape.dim_sizes());
    xla::XlaOp output0 = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output0);

    xla::XlaOp below_min = xla::Lt(input, nudged_input_min);
    xla::XlaOp select1 = xla::Select(below_min, gradient, zeroes);
    xla::XlaOp reduce1 = xla::ReduceAll(
        XlaHelpers::ConvertElementType(select1, accumulation_type),
        XlaHelpers::Zero(b, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type));
    xla::XlaOp output1 = XlaHelpers::ConvertElementType(reduce1, data_type);
    ctx->SetOutput(1, output1);

    xla::XlaOp above_max = xla::Gt(input, nudged_input_max);
    xla::XlaOp select2 = xla::Select(above_max, gradient, zeroes);
    xla::XlaOp reduce2 = xla::ReduceAll(
        XlaHelpers::ConvertElementType(select2, accumulation_type),
        XlaHelpers::Zero(b, accumulation_type),
        *ctx->GetOrCreateAdd(accumulation_type));
    xla::XlaOp output2 = XlaHelpers::ConvertElementType(reduce2, data_type);
    ctx->SetOutput(2, output2);
  }

 private:
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsGradient"),
                FakeQuantWithMinMaxVarsGradOp);

class FakeQuantWithMinMaxVarsPerChannelOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_9(mht_9_v, 441, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "FakeQuantWithMinMaxVarsPerChannelOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ >= 2 && num_bits_ <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    quant_min_ = narrow_range_ ? 1 : 0;
    quant_max_ = (1 << num_bits_) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_10(mht_10_v, 455, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp input = ctx->Input(0);

    const DataType data_type = ctx->input_type(0);
    xla::XlaOp input_min = ctx->Input(1);
    xla::XlaOp input_max = ctx->Input(2);

    xla::Shape input_shape = b->GetShape(input).ValueOrDie();
    absl::Span<const int64_t> input_dimensions = input_shape.dimensions();
    auto convert_to_input_shape = [&](const xla::XlaOp op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_11(mht_11_v, 468, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "lambda");

      return xla::BroadcastInDim(op, input_dimensions,
                                 {input_shape.rank() - 1});
    };
    input_min = convert_to_input_shape(input_min);
    input_max = convert_to_input_shape(input_max);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp output = Quantize(b, input, data_type, nudged_input_min,
                                 nudged_input_max, input_scale);
    ctx->SetOutput(0, output);
  }

 private:
  int num_bits_;
  bool narrow_range_;
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsPerChannel"),
                FakeQuantWithMinMaxVarsPerChannelOp);

class FakeQuantWithMinMaxVarsPerChannelGradOp : public XlaOpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelGradOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_12(mht_12_v, 500, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "FakeQuantWithMinMaxVarsPerChannelGradOp");

    int num_bits;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(ctx, num_bits >= 2 && num_bits <= 16,
                errors::InvalidArgument("num_bits is out of range, expected "
                                        "between 2 and 16, was: ",
                                        num_bits));
    bool narrow_range;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_13(mht_13_v, 516, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "Compile");

    xla::XlaOp gradient = ctx->Input(0);
    const TensorShape gradient_shape = ctx->InputShape(0);
    xla::XlaOp input = ctx->Input(1);
    const DataType data_type = ctx->input_type(1);
    const DataType accumulation_type =
        XlaHelpers::SumAccumulationType(data_type);
    xla::XlaOp input_min = ctx->Input(2);
    xla::XlaOp input_max = ctx->Input(3);

    xla::XlaBuilder* b = ctx->builder();
    xla::Shape input_shape = b->GetShape(input).ValueOrDie();
    absl::Span<const int64_t> input_dimensions = input_shape.dimensions();

    std::vector<int64_t> reduce_axes;
    for (int64_t i = 0; i + 1 < input_shape.rank(); ++i) {
      reduce_axes.push_back(i);
    }

    auto convert_to_input_shape = [&](const xla::XlaOp op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfake_quantize_opsDTcc mht_14(mht_14_v, 538, "", "./tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc", "lambda");

      return xla::BroadcastInDim(op, input_dimensions,
                                 {input_shape.rank() - 1});
    };
    input_min = convert_to_input_shape(input_min);
    input_max = convert_to_input_shape(input_max);

    xla::XlaOp nudged_input_min, nudged_input_max, input_scale;
    XlaNudge(b, data_type, input_min, input_max, quant_min_, quant_max_,
             &nudged_input_min, &nudged_input_max, &input_scale);

    xla::XlaOp between_nudged_min_max = xla::And(
        xla::Le(nudged_input_min, input), xla::Le(input, nudged_input_max));
    xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    xla::XlaOp zeroes = xla::Broadcast(zero, gradient_shape.dim_sizes());
    xla::XlaOp output0 = xla::Select(between_nudged_min_max, gradient, zeroes);
    ctx->SetOutput(0, output0);

    xla::XlaOp below_min = xla::Lt(input, nudged_input_min);
    xla::XlaOp select1 = xla::Select(below_min, gradient, zeroes);
    xla::XlaOp reduce1 =
        xla::Reduce(XlaHelpers::ConvertElementType(select1, accumulation_type),
                    XlaHelpers::Zero(b, accumulation_type),
                    *ctx->GetOrCreateAdd(accumulation_type), reduce_axes);
    xla::XlaOp output1 = XlaHelpers::ConvertElementType(reduce1, data_type);
    ctx->SetOutput(1, output1);

    xla::XlaOp above_max = xla::Gt(input, nudged_input_max);
    xla::XlaOp select2 = xla::Select(above_max, gradient, zeroes);
    xla::XlaOp reduce2 =
        xla::Reduce(XlaHelpers::ConvertElementType(select2, accumulation_type),
                    XlaHelpers::Zero(b, accumulation_type),
                    *ctx->GetOrCreateAdd(accumulation_type), reduce_axes);
    xla::XlaOp output2 = XlaHelpers::ConvertElementType(reduce2, data_type);
    ctx->SetOutput(2, output2);
  }

 private:
  float quant_min_;
  float quant_max_;
};

REGISTER_XLA_OP(Name("FakeQuantWithMinMaxVarsPerChannelGradient"),
                FakeQuantWithMinMaxVarsPerChannelGradOp);

}  // namespace
}  // namespace tensorflow
