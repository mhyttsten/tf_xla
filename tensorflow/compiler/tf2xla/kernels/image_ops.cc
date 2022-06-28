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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc() {
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

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {

// Converts 'input' from RGB format to HSV format.
// 'shape' is the shape of the red/green/blue tensors.
std::array<xla::XlaOp, 3> RGBToHSV(XlaOpKernelContext* ctx, xla::XlaBuilder* b,
                                   const std::array<xla::XlaOp, 3>& rgb,
                                   DataType dtype, const TensorShape& shape) {
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::One(b, dtype);

  auto red = rgb[0];
  auto green = rgb[1];
  auto blue = rgb[2];
  auto value = xla::Max(xla::Max(red, green), blue);
  auto minimum = xla::Min(xla::Min(red, green), blue);
  auto range = xla::Sub(value, minimum);

  auto zeros = xla::Broadcast(zero, shape.dim_sizes());
  auto saturation =
      xla::Select(xla::Gt(value, zero), xla::Div(range, value), zeros);

  auto norm = xla::Div(XlaHelpers::FloatLiteral(b, dtype, 1.0 / 6.0), range);

  auto hue =
      xla::Select(xla::Eq(green, value),
                  xla::Add(xla::Mul(norm, xla::Sub(blue, red)),
                           XlaHelpers::FloatLiteral(b, dtype, 2.0 / 6.0)),
                  xla::Add(xla::Mul(norm, xla::Sub(red, green)),
                           XlaHelpers::FloatLiteral(b, dtype, 4.0 / 6.0)));
  hue = xla::Select(xla::Eq(red, value), xla::Mul(norm, xla::Sub(green, blue)),
                    hue);
  hue = xla::Select(xla::Gt(range, zero), hue, zeros);
  hue = xla::Select(xla::Lt(hue, zero), xla::Add(hue, one), hue);
  return {hue, saturation, value};
}

// Converts 'input' from HSV format to RGB format.
std::array<xla::XlaOp, 3> HSVToRGB(xla::XlaBuilder* b,
                                   const std::array<xla::XlaOp, 3>& hsv,
                                   DataType dtype) {
  xla::XlaOp hue = hsv[0];
  xla::XlaOp saturation = hsv[1];
  xla::XlaOp value = hsv[2];
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
  auto two = XlaHelpers::FloatLiteral(b, dtype, 2.0);
  auto three = XlaHelpers::FloatLiteral(b, dtype, 3.0);
  auto four = XlaHelpers::FloatLiteral(b, dtype, 4.0);
  auto six = XlaHelpers::FloatLiteral(b, dtype, 6.0);

  auto dh = xla::Mul(hue, six);
  auto dr = xla::Clamp(zero, xla::Sub(xla::Abs(xla::Sub(dh, three)), one), one);
  auto dg = xla::Clamp(zero, xla::Sub(two, xla::Abs(xla::Sub(dh, two))), one);
  auto db = xla::Clamp(zero, xla::Sub(two, xla::Abs(xla::Sub(dh, four))), one);
  auto one_minus_s = xla::Sub(one, saturation);

  auto red = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, dr)), value);
  auto green = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, dg)), value);
  auto blue = xla::Mul(xla::Add(one_minus_s, xla::Mul(saturation, db)), value);
  return {red, green, blue};
}

class RGBToHSVOp : public XlaOpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_0(mht_0_v, 274, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "RGBToHSVOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_1(mht_1_v, 279, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input_shape.DebugString()));
    int channel_dim = input_shape.dims() - 1;
    int64_t channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::FailedPrecondition("input must have 3 channels but input has ",
                                   channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);

    xla::XlaOp red = xla::SliceInDim(input, /*start_index=*/0,
                                     /*limit_index=*/1, /*stride=*/1,
                                     /*dimno=*/channel_dim);
    xla::XlaOp green = xla::SliceInDim(input, /*start_index=*/1,
                                       /*limit_index=*/2, /*stride=*/1,
                                       /*dimno=*/channel_dim);
    xla::XlaOp blue = xla::SliceInDim(input, /*start_index=*/2,
                                      /*limit_index=*/3, /*stride=*/1,
                                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv = RGBToHSV(context, b, {red, green, blue}, context->input_type(0),
                        channel_shape);

    context->SetOutput(0, xla::ConcatInDim(b, hsv, channel_dim));
  }
};
REGISTER_XLA_OP(Name("RGBToHSV"), RGBToHSVOp);

class HSVToRGBOp : public XlaOpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_2(mht_2_v, 318, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "HSVToRGBOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_3(mht_3_v, 323, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input_shape.DebugString()));
    int channel_dim = input_shape.dims() - 1;
    int64_t channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::FailedPrecondition("input must have 3 channels but input has ",
                                   channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input = context->Input(0);
    xla::XlaOp hue = xla::SliceInDim(input, /*start_index=*/0,
                                     /*limit_index=*/1, /*stride=*/1,
                                     /*dimno=*/channel_dim);
    xla::XlaOp saturation = xla::SliceInDim(input, /*start_index=*/1,
                                            /*limit_index=*/2, /*stride=*/1,
                                            /*dimno=*/channel_dim);
    xla::XlaOp value = xla::SliceInDim(input, /*start_index=*/2,
                                       /*limit_index=*/3, /*stride=*/1,
                                       /*dimno=*/channel_dim);

    auto rgb = HSVToRGB(context->builder(), {hue, saturation, value},
                        context->input_type(0));

    context->SetOutput(0, xla::ConcatInDim(b, rgb, channel_dim));
  }
};
REGISTER_XLA_OP(Name("HSVToRGB"), HSVToRGBOp);

class AdjustContrastOpV2 : public XlaOpKernel {
 public:
  explicit AdjustContrastOpV2(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_4(mht_4_v, 361, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "AdjustContrastOpV2");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_5(mht_5_v, 366, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& factor_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    int height_dim = input_shape.dims() - 3;
    int width_dim = input_shape.dims() - 2;
    int channel_dim = input_shape.dims() - 1;
    const int64_t height = input_shape.dim_size(height_dim);
    const int64_t width = input_shape.dim_size(width_dim);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor_shape),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor_shape.DebugString()));

    xla::XlaBuilder* b = context->builder();
    DataType type = context->input_type(0);

    xla::XlaOp input = context->Input(0);
    xla::XlaOp factor = XlaHelpers::ConvertElementType(context->Input(1), type);

    const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
    auto converted = XlaHelpers::ConvertElementType(input, accumulation_type);
    auto reduce = xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                              *context->GetOrCreateAdd(accumulation_type),
                              {height_dim, width_dim});

    auto output = xla::Div(
        reduce, XlaHelpers::FloatLiteral(b, accumulation_type, height * width));
    output = XlaHelpers::ConvertElementType(output, type);

    std::vector<int64_t> broadcast_dims(input_shape.dims() - 2);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    broadcast_dims.back() = channel_dim;
    output =
        xla::Add(xla::Mul(input, factor),
                 xla::Mul(output, xla::Sub(XlaHelpers::One(b, type), factor)),
                 broadcast_dims);
    context->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("AdjustContrastv2"), AdjustContrastOpV2);

class AdjustSaturationOp : public XlaOpKernel {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_6(mht_6_v, 416, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "AdjustSaturationOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_7(mht_7_v, 421, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& scale_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_shape),
                errors::InvalidArgument("scale must be scalar: ",
                                        scale_shape.DebugString()));
    const int channel_dim = input_shape.dims() - 1;
    const int64_t channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input =
        XlaHelpers::ConvertElementType(context->Input(0), DT_FLOAT);
    xla::XlaOp scale =
        XlaHelpers::ConvertElementType(context->Input(1), DT_FLOAT);

    DataType type = context->input_type(0);

    xla::XlaOp red = xla::SliceInDim(input, /*start_index=*/0,
                                     /*limit_index=*/1, /*stride=*/1,
                                     /*dimno=*/channel_dim);
    xla::XlaOp green = xla::SliceInDim(input, /*start_index=*/1,
                                       /*limit_index=*/2, /*stride=*/1,
                                       /*dimno=*/channel_dim);
    xla::XlaOp blue = xla::SliceInDim(input, /*start_index=*/2,
                                      /*limit_index=*/3, /*stride=*/1,
                                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv =
        RGBToHSV(context, b, {red, green, blue}, DT_FLOAT, channel_shape);

    hsv[1] = xla::Clamp(XlaHelpers::Zero(b, DT_FLOAT), xla::Mul(hsv[1], scale),
                        XlaHelpers::One(b, DT_FLOAT));

    auto rgb = HSVToRGB(context->builder(), hsv, DT_FLOAT);

    auto output = XlaHelpers::ConvertElementType(
        xla::ConcatInDim(b, rgb, channel_dim), type);
    context->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("AdjustSaturation"), AdjustSaturationOp);

class AdjustHueOp : public XlaOpKernel {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_8(mht_8_v, 476, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "AdjustHueOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_9(mht_9_v, 481, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    const TensorShape& input_shape = context->InputShape(0);
    const TensorShape& delta_shape = context->InputShape(1);
    OP_REQUIRES(context, input_shape.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta_shape),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta_shape.DebugString()));
    const int channel_dim = input_shape.dims() - 1;
    const int64_t channels = input_shape.dim_size(channel_dim);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    xla::XlaBuilder* b = context->builder();
    xla::XlaOp input =
        XlaHelpers::ConvertElementType(context->Input(0), DT_FLOAT);
    xla::XlaOp delta =
        XlaHelpers::ConvertElementType(context->Input(1), DT_FLOAT);

    DataType type = context->input_type(0);

    xla::XlaOp red = xla::SliceInDim(input, /*start_index=*/0,
                                     /*limit_index=*/1, /*stride=*/1,
                                     /*dimno=*/channel_dim);
    xla::XlaOp green = xla::SliceInDim(input, /*start_index=*/1,
                                       /*limit_index=*/2, /*stride=*/1,
                                       /*dimno=*/channel_dim);
    xla::XlaOp blue = xla::SliceInDim(input, /*start_index=*/2,
                                      /*limit_index=*/3, /*stride=*/1,
                                      /*dimno=*/channel_dim);
    TensorShape channel_shape = input_shape;
    channel_shape.set_dim(channel_dim, 1);
    auto hsv =
        RGBToHSV(context, b, {red, green, blue}, DT_FLOAT, channel_shape);

    auto zero = XlaHelpers::Zero(b, DT_FLOAT);
    auto one = XlaHelpers::One(b, DT_FLOAT);

    auto& hue = hsv[0];
    hue = xla::Rem(xla::Add(hsv[0], delta), one);
    hue =
        xla::Select(xla::Lt(hue, zero), xla::Rem(xla::Add(one, hue), one), hue);

    auto rgb = HSVToRGB(context->builder(), hsv, DT_FLOAT);

    auto output = XlaHelpers::ConvertElementType(
        xla::ConcatInDim(b, rgb, channel_dim), type);
    context->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("AdjustHue"), AdjustHueOp);

struct WhileCondFn {
  const int64_t num_boxes;
  const int64_t output_size;

  explicit WhileCondFn(int64_t num_boxes, int64_t output_size)
      : num_boxes(num_boxes), output_size(output_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_10(mht_10_v, 544, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "WhileCondFn");
}

  StatusOr<xla::XlaOp> operator()(absl::Span<const xla::XlaOp> values,
                                  xla::XlaBuilder* cond_builder) const {
    xla::XlaOp row_idx = values[0];
    xla::XlaOp row_in_bounds =
        xla::Lt(row_idx, xla::ConstantR0<int32>(cond_builder, num_boxes));
    xla::XlaOp num_outputs_so_far = values[1];
    xla::XlaOp results_not_full = xla::Lt(
        num_outputs_so_far, xla::ConstantR0<int32>(cond_builder, output_size));
    return xla::And(row_in_bounds, results_not_full);
  }
};

// Process the boxes one-by-one using the iou matrix mask.
// This implementation uses a correct, but greedy, sequential algorithm
// to ensure that suppressed boxes cannot themselves suppress other
// boxes.
struct SuppressBodyFn {
  const int64_t num_boxes;

  explicit SuppressBodyFn(int64_t num_boxes) : num_boxes(num_boxes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_11(mht_11_v, 568, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "SuppressBodyFn");
}

  StatusOr<std::vector<xla::XlaOp>> operator()(
      absl::Span<const xla::XlaOp> values, xla::XlaBuilder* builder) const {
    auto row_idx = values[0];
    auto num_outputs_so_far = values[1];
    auto iou_mask = values[2];
    auto included_iou = values[3];
    auto zero = xla::ConstantR0<int32>(builder, 0);
    // Determine if current elem is active using a slice.
    // TODO(b/118437727): The only reason we need an explicit vector is because
    // some old GCCs can't deduce the right type for MakeConstSpan, and
    // providing a single-value initializer list directly uses the wrong
    // overload. Delete this once the deprecated overload is gone.
    std::vector<xla::XlaOp> row_idx_vector = {row_idx};
    auto active_elem = xla::DynamicSlice(included_iou, row_idx_vector, {1});
    active_elem = xla::Reshape(active_elem, {});
    // Increment output count iff current elem is not suppressed.
    num_outputs_so_far = xla::Select(
        active_elem, num_outputs_so_far + xla::ConstantR0<int32>(builder, 1),
        num_outputs_so_far);
    // Slice out the row_idx.
    auto row_iou = xla::DynamicSlice(iou_mask, {row_idx, zero}, {1, num_boxes});
    // Remove the diagonal from consideration. An elem cannot suppress
    // itself.
    row_iou = xla::DynamicUpdateSlice(
        row_iou, xla::ConstantR2FromArray2D<bool>(builder, {{false}}),
        {zero, row_idx});
    // Create a suppression by inverting polarity.
    row_iou = xla::Reshape(row_iou, {num_boxes});
    auto supp_mask = xla::Not(row_iou);
    // Update mask iff current elem is not suppressed.
    included_iou = xla::Select(xla::Broadcast(active_elem, {num_boxes}),
                               xla::And(included_iou, supp_mask), included_iou);
    row_idx = row_idx + xla::ConstantR0<int32>(builder, 1);
    return std::vector<xla::XlaOp>{row_idx, num_outputs_so_far, iou_mask,
                                   included_iou};
  }
};

class NonMaxSuppressionOp : public XlaOpKernel {
 public:
  explicit NonMaxSuppressionOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_12(mht_12_v, 614, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "NonMaxSuppressionOp");

    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_13(mht_13_v, 622, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    // TODO(b/111646731): Improve scalability of this op, using blocking.
    OP_REQUIRES(context, pad_to_max_output_size_,
                errors::Unimplemented(
                    "XLA compilation requires pad_to_max_output_size == True"));

    xla::XlaOp selected_indices, num_valid;
    ComputeResult(context, pad_to_max_output_size_);
  }
  static void ComputeResult(XlaOpKernelContext* context,
                            bool pad_to_max_output_size = false) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_14(mht_14_v, 635, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "ComputeResult");

    const TensorShape& boxes_shape = context->InputShape("boxes");
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(boxes_shape),
        errors::InvalidArgument("boxes must be 2-D, currently: [",
                                std::to_string(boxes_shape.dim_size(0)), ",",
                                std::to_string(boxes_shape.dim_size(1)), "]"));
    const int64_t num_boxes = boxes_shape.dim_size(0);
    OP_REQUIRES(
        context, boxes_shape.dim_size(1) == 4,
        errors::InvalidArgument("boxes must have 4 columns, currently: ",
                                std::to_string(boxes_shape.dim_size(1))));
    const TensorShape& scores_shape = context->InputShape("scores");
    OP_REQUIRES(context, TensorShapeUtils::IsVector(scores_shape),
                errors::InvalidArgument("scores must be 1-D, currently: ",
                                        scores_shape.DebugString()));
    OP_REQUIRES(context, scores_shape.dim_size(0) == num_boxes,
                errors::InvalidArgument(
                    "scores size ", std::to_string(scores_shape.dim_size(0)),
                    " must equal number of boxes ", std::to_string(num_boxes)));
    OP_REQUIRES(context, num_boxes <= kint32max,
                errors::InvalidArgument("XLA compilation requires number of "
                                        "boxes to be <= kint32max, got ",
                                        num_boxes));
    xla::PrimitiveType boxes_xla_type = context->InputXlaType("boxes");
    xla::PrimitiveType scores_xla_type = context->InputXlaType("scores");
    const xla::XlaOp boxes_input = context->Input("boxes");
    const xla::XlaOp scores_input = context->Input("scores");
    int64_t output_size;
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(context->InputShape("max_output_size")),
        errors::InvalidArgument("Max Output Size isn't a scalar"));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(context->InputShape("iou_threshold")),
        errors::InvalidArgument("IOU Threshold isn't a scalar"));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(2, &output_size));
    OP_REQUIRES(
        context, output_size >= 0,
        errors::InvalidArgument("Need output_size >= 0, got ", output_size));
    OP_REQUIRES(context, output_size <= kint32max,
                errors::InvalidArgument("Need output_size <= kint32Max, got ",
                                        output_size));
    const xla::XlaOp score_thresh = context->Input("score_threshold");
    const xla::XlaOp iou_thresh = context->Input("iou_threshold");
    xla::XlaBuilder* const builder = context->builder();

    // Choose a more convenient layout.
    const xla::XlaOp boxes = xla::Transpose(boxes_input, {1, 0});
    const xla::XlaOp boxes_sorted = xla::GetTupleElement(
        xla::Sort({xla::Broadcast(scores_input, {4}), boxes},
                  xla::CreateScalarGtComputation(
                      {scores_xla_type, boxes_xla_type}, builder),
                  /*dimension=*/1),
        1);
    // Track the mapping of indices into sorted domain.
    const xla::XlaOp iota_indices = xla::Iota(builder, xla::S32, num_boxes);
    const xla::XlaOp indices_sort = xla::Sort(
        {scores_input, iota_indices},
        xla::CreateScalarGtComputation({scores_xla_type, xla::S32}, builder));
    const xla::XlaOp indices_sorted = xla::GetTupleElement(indices_sort, 1);
    const xla::XlaOp scores = xla::GetTupleElement(indices_sort, 0);

    // Shapes are henceforth [1, num_boxes]. 'c_y0' denotes 'coordinate' y0.
    const xla::XlaOp c_y0 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                         /*start_index=*/0,
                                                         /*limit_index=*/1,
                                                         /*stride=*/1,
                                                         /*dimno=*/0),
                                         {num_boxes});
    const xla::XlaOp c_x0 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                         /*start_index=*/1,
                                                         /*limit_index=*/2,
                                                         /*stride=*/1,
                                                         /*dimno=*/0),
                                         {num_boxes});
    const xla::XlaOp c_y1 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                         /*start_index=*/2,
                                                         /*limit_index=*/3,
                                                         /*stride=*/1,
                                                         /*dimno=*/0),
                                         {num_boxes});
    const xla::XlaOp c_x1 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                         /*start_index=*/3,
                                                         /*limit_index=*/4,
                                                         /*stride=*/1,
                                                         /*dimno=*/0),
                                         {num_boxes});

    xla::XlaOp y1 = xla::Select(xla::Le(c_y0, c_y1), c_y0, c_y1);
    xla::XlaOp y2 = xla::Select(xla::Le(c_y0, c_y1), c_y1, c_y0);
    xla::XlaOp x1 = xla::Select(xla::Le(c_x0, c_x1), c_x0, c_x1);
    xla::XlaOp x2 = xla::Select(xla::Le(c_x0, c_x1), c_x1, c_x0);
    xla::XlaOp area = (y2 - y1) * (x2 - x1);

    // Shapes are henceforth [1, num_boxes].
    y1 = xla::Broadcast(y1, {1});
    y2 = xla::Broadcast(y2, {1});
    x1 = xla::Broadcast(x1, {1});
    x2 = xla::Broadcast(x2, {1});
    area = xla::Broadcast(area, {1});

    // Shapes are henceforth [num_boxes, num_boxes].
    xla::XlaOp i_xmin = xla::Max(x1, xla::Transpose(x1, {1, 0}));
    xla::XlaOp i_ymin = xla::Max(y1, xla::Transpose(y1, {1, 0}));
    xla::XlaOp i_xmax = xla::Min(x2, xla::Transpose(x2, {1, 0}));
    xla::XlaOp i_ymax = xla::Min(y2, xla::Transpose(y2, {1, 0}));
    auto square_zero = xla::ZerosLike(i_xmin);

    xla::XlaOp i_area = xla::Max(i_xmax - i_xmin, square_zero) *
                        xla::Max(i_ymax - i_ymin, square_zero);
    xla::XlaOp u_area = area + xla::Transpose(area, {1, 0}) - i_area;
    xla::XlaOp iou = i_area / u_area;

    xla::XlaOp iou_thresh_mask = xla::Gt(iou, iou_thresh + square_zero);
    xla::XlaOp included_iou =
        xla::Broadcast(xla::ConstantR0<bool>(builder, true), {num_boxes});

    std::vector<xla::XlaOp> init_values;
    init_values.reserve(4);
    init_values.push_back(xla::ConstantR0<int32>(builder, 0));  // col_idx
    init_values.push_back(xla::ConstantR0<int32>(builder, 0));  // num_outputs
    init_values.push_back(iou_thresh_mask);
    init_values.push_back(included_iou);

    auto suppress_loop_result =
        xla::WhileLoopHelper(WhileCondFn(num_boxes, output_size),
                             SuppressBodyFn(num_boxes), init_values,
                             "suppress_loop", builder)
            .ValueOrDie();

    xla::XlaOp included_score =
        xla::Gt(scores, xla::Broadcast(score_thresh, {num_boxes}));
    xla::XlaOp included = xla::And(included_score, suppress_loop_result[3]);

    // Only consider boxes over which we have iterated. This allows for accurate
    // counting. DynamicSlice would require knowledge of the size of the output.
    auto valid_elem = xla::Lt(
        iota_indices, xla::Broadcast(suppress_loop_result[0], {num_boxes}));
    included = xla::And(included, valid_elem);

    xla::XlaOp neg_inf =
        xla::Broadcast(xla::MinValue(builder, boxes_xla_type), {num_boxes});
    xla::XlaOp scores_included = xla::Select(included, scores, neg_inf);
    xla::XlaOp output_tuple = TopK(scores_included, output_size);
    xla::XlaOp selected_indices_sorted = xla::GetTupleElement(output_tuple, 1);
    // Calculate num_valid.
    // Note: num_valid cannot be taken from the loop outputs, because outputs
    // can be suppressed by score threshold.
    xla::XlaOp ones_included = xla::Select(
        included,
        xla::Broadcast(xla::ConstantR0<int32>(builder, 1), {num_boxes}),
        xla::Broadcast(xla::ConstantR0<int32>(builder, 0), {num_boxes}));
    // num_valid is scalar. Value should be bound by output_size.

    xla::XlaOp num_valid_total = xla::Reduce(
        ones_included,
        /*init_value=*/xla::ConstantR0<int>(builder, 0),
        /*computation=*/CreateScalarAddComputation(xla::S32, builder),
        /*dimensions_to_reduce=*/{0});
    xla::XlaOp num_valid =
        xla::Min(num_valid_total, xla::ConstantR0<int32>(builder, output_size));

    // Re-index into the original scores input tensor, using a Gather.
    // Boxes were suppressed in the sorted domain.
    xla::XlaOp selected_indices;
    DataType gather_type = context->expected_output_dtype(0);
    OP_REQUIRES_OK(
        context,
        XlaGather(indices_sorted, scores_shape, selected_indices_sorted,
                  TensorShape({output_size}),
                  /*axis=*/0,
                  /*indices_are_nd=*/false,
                  /*dtype=*/gather_type, DT_INT32, builder, &selected_indices));

    if (!pad_to_max_output_size) {
      StatusOr<xla::XlaOp> rebounded_result = xla::SetDimensionSizeWithRebound(
          &context->value_inference(), selected_indices, num_valid, 0);
      if (rebounded_result.ok()) {
        selected_indices = *rebounded_result;
      } else {
        // TODO(b/207187072): Remove special handling once dynamic reshape
        // can also be handled.
        selected_indices =
            xla::SetDimensionSize(selected_indices, num_valid, 0);
      }
    }
    context->SetOutput(0, selected_indices);
    if (pad_to_max_output_size) context->SetOutput(1, num_valid);
  }

 private:
  bool pad_to_max_output_size_;
};

REGISTER_XLA_OP(
    Name("NonMaxSuppressionV4").CompileTimeConstantInput("max_output_size"),
    NonMaxSuppressionOp);

class NonMaxSuppressionV3Op : public XlaOpKernel {
 public:
  explicit NonMaxSuppressionV3Op(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_15(mht_15_v, 841, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "NonMaxSuppressionV3Op");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSimage_opsDTcc mht_16(mht_16_v, 846, "", "./tensorflow/compiler/tf2xla/kernels/image_ops.cc", "Compile");

    xla::XlaOp selected_indices, num_valid;
    NonMaxSuppressionOp::ComputeResult(context);
  }
};

REGISTER_XLA_OP(
    Name("NonMaxSuppressionV3").CompileTimeConstantInput("max_output_size"),
    NonMaxSuppressionV3Op);

}  // namespace
}  // namespace tensorflow
