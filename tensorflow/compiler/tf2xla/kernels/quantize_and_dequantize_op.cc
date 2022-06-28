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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc() {
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

#include <cstddef>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

enum QuantizerRoundMode {
  // Round half up: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5
  // E.g., -5.5 gets rounded to -5, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_UP,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};

class QuantizeAndDequantizeOp : public XlaOpKernel {
 public:
  explicit QuantizeAndDequantizeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/tf2xla/kernels/quantize_and_dequantize_op.cc", "QuantizeAndDequantizeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    round_mode_ = ROUND_HALF_TO_EVEN;
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/tf2xla/kernels/quantize_and_dequantize_op.cc", "Compile");

    xla::XlaOp input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);

    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(data_type, &xla_type));

    xla::XlaBuilder* b = ctx->builder();

    // The implementation follows
    // tensorflow/core/kernels/quantize_and_dequantize_op.h closely.
    xla::XlaOp min_range, max_range;
    if (range_given_) {
      min_range = ctx->Input(1);
      max_range = ctx->Input(2);
    } else {
      const xla::XlaComputation* fmax = ctx->GetOrCreateMax(data_type);
      const xla::XlaComputation* fmin = ctx->GetOrCreateMin(data_type);
      if (axis_ == -1) {
        min_range = ReduceAll(input, xla::MaxValue(b, xla_type), *fmin);
        max_range = ReduceAll(input, xla::MinValue(b, xla_type), *fmax);
      } else {
        std::vector<int64_t> dimensions_to_reduce;
        TensorShape input_shape = ctx->InputShape(0);
        int64_t input_rank = input_shape.dims();
        OP_REQUIRES(ctx, input_rank >= 1,
                    errors::Unimplemented("QuantizeAndDequantizeOp with axis "
                                          "!= -1 requires minimum rank 1"));
        OP_REQUIRES(
            ctx, axis_ >= 0 && axis_ < input_rank,
            errors::Unimplemented("QuantizeAndDequantizeOp with invalid axis"));
        dimensions_to_reduce.reserve(input_rank - 1);
        for (int64_t i = 0; i < input_rank; ++i) {
          if (i != axis_) {
            dimensions_to_reduce.push_back(i);
          }
        }
        min_range = Reduce(input, xla::MaxValue(b, xla_type), *fmin,
                           dimensions_to_reduce);
        max_range = Reduce(input, xla::MinValue(b, xla_type), *fmax,
                           dimensions_to_reduce);
      }
    }

    xla::XlaOp num_bits;
    if (num_bits_ < 0) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 4,
          errors::Internal("Expected 4 inputs to QuantizeAndDequantize"));
      num_bits = ctx->Input(3);
    } else {
      num_bits = xla::ConstantR0<int32>(b, num_bits_);
    }

    const xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    const xla::XlaOp one = XlaHelpers::One(b, data_type);
    const xla::XlaOp two = XlaHelpers::FloatLiteral(b, data_type, 2.0);
    const xla::XlaOp half = XlaHelpers::FloatLiteral(b, data_type, 0.5);

    // Calculate the range for the simulated integer quantization:
    // e.g. [-128,127] for signed = true, num_bits = 8,
    // or [0, 255] for signed = false, num_bits = 8.
    // We do this in floating point for hardware that does not have 64-bit
    // integer support.
    xla::XlaOp min_quantized, max_quantized;
    if (signed_input_) {
      if (narrow_range_) {
        min_quantized =
            -Pow(two, ConvertElementType(
                          num_bits - xla::ConstantR0<int32>(b, 1), xla_type)) +
            one;
      } else {
        min_quantized =
            -Pow(two, ConvertElementType(
                          num_bits - xla::ConstantR0<int32>(b, 1), xla_type));
      }
      max_quantized =
          Pow(two, ConvertElementType(num_bits - xla::ConstantR0<int32>(b, 1),
                                      xla_type)) -
          one;
    } else {
      min_quantized = zero;
      max_quantized = Pow(two, ConvertElementType(num_bits, xla_type)) - one;
    }

    // Determine the maximum scaling factor that would scale
    // [min_range, max_range] to not exceed [min_quantized, max_quantized],
    // while keeping 0 unchanged.
    xla::XlaOp scale_from_min_side =
        Select(Gt(min_quantized * min_range, zero), min_quantized / min_range,
               xla::MaxFiniteValue(b, xla_type));
    xla::XlaOp scale_from_max_side =
        Select(Gt(max_quantized * max_range, zero), max_quantized / max_range,
               xla::MaxFiniteValue(b, xla_type));

    // Note: Avoids changing the side of the range that determines scale.
    xla::XlaOp cond = Lt(scale_from_min_side, scale_from_max_side);
    xla::XlaOp scale = Select(cond, scale_from_min_side, scale_from_max_side);
    xla::XlaOp inverse_scale =
        Select(cond, min_range / min_quantized, max_range / max_quantized);
    min_range = Select(cond, min_range, min_quantized * inverse_scale);
    max_range = Select(cond, max_quantized * inverse_scale, max_range);

    // The instruction min_range has the shape of the axis, which is also the
    // shape for max_range, scale and inverse_scale.
    xla::Shape axis_shape = b->GetShape(min_range).ValueOrDie();
    // The XLA client library can handle implicit broadcast from scalar. Add
    // explicit broadcast if the axis has a non-scalar shape.
    if (!xla::ShapeUtil::IsScalar(axis_shape)) {
      xla::Shape input_shape = b->GetShape(input).ValueOrDie();
      absl::Span<const int64_t> input_dimensions = input_shape.dimensions();
      auto convert_to_input_shape = [&](const xla::XlaOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc mht_2(mht_2_v, 343, "", "./tensorflow/compiler/tf2xla/kernels/quantize_and_dequantize_op.cc", "lambda");

        return xla::BroadcastInDim(op, input_dimensions, {axis_});
      };
      min_range = convert_to_input_shape(min_range);
      max_range = convert_to_input_shape(max_range);
      scale = convert_to_input_shape(scale);
      inverse_scale = convert_to_input_shape(inverse_scale);
    }

    if (range_given_) {
      // Note: The clamping here is to avoid overflow in the quantized type.
      // The semantics of the op does not guarantee to clamp to the specified
      // min_range and max_range - because we may have changed either min_range
      // or max_range.
      // No need to clamp to min_range and max_range if range_given_ == false as
      // in that case they were measured from the tensor.
      input = Clamp(min_range, input, max_range);
    }
    xla::XlaOp result;
    switch (round_mode_) {
      case ROUND_HALF_TO_EVEN: {
        result = xla::RoundToEven(input * scale) * inverse_scale;
        break;
      }
      case ROUND_HALF_UP: {
        result = Floor(input * scale + half) * inverse_scale;
        break;
      }
    }
    ctx->SetOutput(0, result);
  }

 protected:
  int64_t num_bits_ = -1;
  int axis_;
  bool signed_input_;
  bool range_given_;
  bool narrow_range_;
  QuantizerRoundMode round_mode_;
};

class QuantizeAndDequantizeV2Op : public QuantizeAndDequantizeOp {
 public:
  explicit QuantizeAndDequantizeV2Op(OpKernelConstruction* ctx)
      : QuantizeAndDequantizeOp(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSquantize_and_dequantize_opDTcc mht_3(mht_3_v, 390, "", "./tensorflow/compiler/tf2xla/kernels/quantize_and_dequantize_op.cc", "QuantizeAndDequantizeV2Op");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
    string round_mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(
        ctx,
        (round_mode_string == "HALF_UP" || round_mode_string == "HALF_TO_EVEN"),
        errors::InvalidArgument("Round mode string must be "
                                "'HALF_UP' or "
                                "'HALF_TO_EVEN', is '" +
                                round_mode_string + "'"));
    if (round_mode_string == "HALF_UP") {
      round_mode_ = ROUND_HALF_UP;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      round_mode_ = ROUND_HALF_TO_EVEN;
    }
  }
};

REGISTER_XLA_OP(Name("QuantizeAndDequantizeV2"), QuantizeAndDequantizeV2Op);
REGISTER_XLA_OP(Name("QuantizeAndDequantizeV3"), QuantizeAndDequantizeOp);
REGISTER_XLA_OP(Name("QuantizeAndDequantizeV4"), QuantizeAndDequantizeV2Op);

}  // namespace
}  // namespace tensorflow
