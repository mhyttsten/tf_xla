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
class MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"

// Implements legalization and post-legalization optimization helper functions

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "buildRescale");

  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getI32ArrayAttr({multiplier}), rewriter.getI32ArrayAttr({shift}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, double input_scale,
                          int64_t input_zp) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "buildRescaleToInt32");

  // Output is always int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
                      input_zp, 0, false, true);
}

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter& rewriter, Operation* op,
                            ShapedType output_type, Value input_val,
                            double output_scale, int64_t output_zp) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "buildRescaleFromInt32");

  // Input should be int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  (void)input_type;
  assert(input_type && input_type.getElementType().isInteger(32) &&
         "expected rescale input element type to be i32");

  // Potentially check input_shape == output_shape here
  return buildRescale(rewriter, op, output_type, input_val, output_scale, 0,
                      output_zp, true, true);
}

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter& rewriter, Operation* op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "buildRescaleOpConvOutput");

  auto input_qtype =
      input_type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto output_qtype = output_type.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();

  double input_scale = input_qtype.getScale();

  int64_t output_zp = output_qtype.getZeroPoint();
  double output_scale = output_qtype.getScale();

  bool scale32 = isScale32(output_qtype);
  int32_t scale_width = scale32 ? 32 : 16;
  // Only use double round if we are doing 32 bit scaling
  bool double_round = scale32;

  if (auto weight_per_tensor_qtype =
          weight_type.getElementType()
              .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    // Per-tensor quantization
    double weight_scale = weight_per_tensor_qtype.getScale();

    int32_t multiplier;
    int32_t shift;

    double op_tensor_scale = (input_scale * weight_scale) / output_scale;

    computeMultiplierAndShift(op_tensor_scale, multiplier, shift, scale_width);

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getI32ArrayAttr({multiplier}),
        rewriter.getI32ArrayAttr({shift}), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(double_round), rewriter.getBoolAttr(false));

    return rescale_op.getResult();

  } else if (auto weight_per_channel_qtype =
                 weight_type.getElementType()
                     .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
    // Per-channel quantization
    SmallVector<int32_t> multiplier_arr;
    SmallVector<int32_t> shift_arr;

    SmallVector<double> weight_scale_arr(
        weight_per_channel_qtype.getScales().begin(),
        weight_per_channel_qtype.getScales().end());

    int64_t output_zp = output_qtype.getZeroPoint();
    double output_scale = output_qtype.getScale();

    for (double weight_scale : weight_scale_arr) {
      int32_t multiplier;
      int32_t shift;

      double op_channel_scale = (input_scale * weight_scale) / output_scale;

      computeMultiplierAndShift(op_channel_scale, multiplier, shift,
                                scale_width);

      multiplier_arr.push_back(multiplier);
      shift_arr.push_back(shift);
    }

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getI32ArrayAttr(multiplier_arr),
        rewriter.getI32ArrayAttr(shift_arr), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(double_round), rewriter.getBoolAttr(true));

    return rescale_op.getResult();

  } else {
    op->emitOpError("buildConvRescaleOp: unknown weight quantized type");
    return nullptr;
  }
}

// Create a 8-bit TOSA TABLE constant tensor with int8[256] array.
// Follow PopulateLookupTable() tensorflow/lite/kernels/activations.cc
Value getTosaConst8bitTable(PatternRewriter& rewriter, Operation* op,
                            double input_scale, int32_t input_zp,
                            double output_scale, int32_t output_zp,
                            std::function<double(double)> func) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_4(mht_4_v, 349, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTosaConst8bitTable");

  SmallVector<int8_t, 256> table;

  for (int32_t i = -128; i < 128; i++) {
    double dequantized = input_scale * (i - input_zp);
    double transformed = func(dequantized);
    int32_t rescaled = std::llround(transformed / output_scale);
    int32_t quantized = static_cast<int32_t>(rescaled + output_zp);
    table.push_back(
        static_cast<int8_t>(std::min(std::max(quantized, -128), 127)));
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(8),
                                rewriter.getF32Type(), 1.0f, 0, -128, 127);
  auto const_type = RankedTensorType::get({256}, element_qtype);
  auto storage_type =
      RankedTensorType::get({256}, element_qtype.getStorageType());
  auto const_attr =
      DenseElementsAttr::get(storage_type, llvm::makeArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 16-bit TOSA TABLE constant tensor with int16[513] array.
// Output is restricted to [-1.0, 1.0].
// Follow gen_lut() tensorflow/lite/kernels/internal/common.h
Value getTosaConst16bitTable(PatternRewriter& rewriter, Operation* op,
                             std::function<double(double)> func, double min,
                             double max) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_5(mht_5_v, 383, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTosaConst16bitTable");

  SmallVector<int16_t, 513> table;

  double step = (max - min) / 512.0f;
  double half_step = step / 2.0f;
  for (int32_t i = 0; i < 512; i++) {
    int32_t sample_val = std::llround(func(min + (i * step)) * 32768.0);
    double midpoint_interp_val =
        std::round(((func(min + (i + 1) * step) * 32768.0) +
                    std::round(func(min + (i * step)) * 32768.0)) /
                   2.0);
    double midpoint_val =
        std::round(func(min + (i * step) + half_step) * 32768.0);
    double midpoint_err = midpoint_interp_val - midpoint_val;
    int32_t bias = std::llround(midpoint_err / 2.0);

    table.push_back(static_cast<int16_t>(
        std::min(std::max(sample_val - bias, -32768), 32767)));
  }

  int32_t max_val = std::llround(func(max) * 32768.0);
  table.push_back(
      static_cast<int16_t>(std::min(std::max(max_val, -32768), 32767)));

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(16),
                                rewriter.getF32Type(), 1.0f, 0, -32768, 32767);
  auto const_type = RankedTensorType::get({513}, element_qtype);
  auto storage_type =
      RankedTensorType::get({513}, element_qtype.getStorageType());
  auto const_attr =
      DenseElementsAttr::get(storage_type, llvm::makeArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 32-bit TOSA TABLE constant tensor with int16[513] array.
// Output is restricted to [-1.0, 1.0] as s0.31 format.
void getTosaConst32bitTable(PatternRewriter& rewriter, Operation* op,
                            double input_scale, int32_t input_zp,
                            std::function<double(double)> func,
                            Value& upper_const, Value& lower_const) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_6(mht_6_v, 429, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTosaConst32bitTable");

  SmallVector<int16_t, 513> upper_table, lower_table;

  double output_inv_scale = static_cast<double>(1L << 31);

  for (int32_t i = -256; i <= 256; i++) {
    double dequantized = input_scale * (i - input_zp);
    double transformed = func(dequantized);
    double truncated = std::min(std::max(transformed, -1.0), 1.0);
    int64_t rescaled =
        static_cast<int64_t>(std::round(truncated * output_inv_scale));

    // 2^31 is not representable in int32_t, so store as 2^31 - 1 instead
    if (rescaled == static_cast<int64_t>(1L << 31)) {
      rescaled = static_cast<int64_t>(1L << 31) - 1;
    }

    int32_t upper = (rescaled >> 16) & 0xFFFF;
    // TABLE output is signed 16 bits with range [-32768, 32767]
    // Lower 16 bits are unsigned and ranges [0, 65536]
    // Need to adjust value with offset 0x8000 in table generation
    // Legalization should add this back before recovering 32-bit value
    int32_t lower = (rescaled & 0xFFFF) - 0x8000;

    upper_table.push_back(upper);
    lower_table.push_back(lower);
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(16),
                                rewriter.getF32Type(), 1.0f, 0, -32768, 32767);
  auto const_type = RankedTensorType::get({513}, element_qtype);
  auto storage_type =
      RankedTensorType::get({513}, element_qtype.getStorageType());

  auto upper_const_attr =
      DenseElementsAttr::get(storage_type, llvm::makeArrayRef(upper_table));
  auto lower_const_attr =
      DenseElementsAttr::get(storage_type, llvm::makeArrayRef(lower_table));

  upper_const =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, upper_const_attr)
          .getResult();
  lower_const =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, lower_const_attr)
          .getResult();
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter& rewriter, Operation* op,
                                  float val) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_7(mht_7_v, 482, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTosaConstTensorSingleF32");

  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 32-bit integer constant operator from an int
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_8(mht_8_v, 496, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTosaConstTensorSingleI32");

  auto const_type = RankedTensorType::get({}, rewriter.getIntegerType(32));
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a vector from a 32-bit value tensor.  Returns the size of
// the new vector or -1 on error.
LogicalResult getVectorFromValue32(Value val, SmallVectorImpl<int32_t>& vec) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_9(mht_9_v, 510, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getVectorFromValue32");

  int i = 0;

  ElementsAttr elems;

  vec.clear();

  if (!matchPattern(val, m_Constant(&elems))) return failure();

  for (auto idx : elems.getValues<IntegerAttr>()) {
    vec.push_back(idx.getInt());
    i++;
  }

  return success();
}

// Calculates the TOSA padding values based on TF operators padded with
// SAME/VALID.
//
// This could pass tensorflow::FilterTensorFormat and do
// GetFilterTensorSpatialDimIndex but the current TF core libs do not support
// FORMAT_OHWI parsing by that function in core/util/tensor_format.h
bool getPaddingValuesFromPadType(tensorflow::Padding tf_pad,
                                 tensorflow::TensorFormat data_format_tf,
                                 uint32_t first_filter_spatial_dim,
                                 ShapedType input_type, ShapedType filter_type,
                                 ArrayAttr strides, ArrayAttr dilations,
                                 PatternRewriter& rewriter,
                                 ArrayAttr& explicit_padding) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_10(mht_10_v, 542, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getPaddingValuesFromPadType");

  assert(tf_pad != tensorflow::Padding::EXPLICIT);
  if (!input_type.hasRank() || !filter_type.getRank()) return false;

  // Storing the numeric padding values is useful for TOSA codegen, as opposed
  // to holding the padding regime mnemonic, i.e. SAME, VALID, FULL, ...
  SmallVector<int64_t> computed_paddings;

  int64_t pad_before, pad_after;
  for (int i = 0; i < 2; i++) {  // Two spatial dimensions X&Y
    int64_t ifm_dim = GetTensorSpatialDimIndex(
        4, data_format_tf, i);  // 4D tensor, NHWC/NCHW format
    int64_t filter_dim = first_filter_spatial_dim + i;

    int64_t dim_dilation = dilations[i].template cast<IntegerAttr>().getInt();
    int64_t dim_stride = strides[i].template cast<IntegerAttr>().getInt();

    int64_t ip_size = input_type.getDimSize(ifm_dim);
    int64_t f_size = filter_type.getDimSize(filter_dim);
    // If we have a dynamic shape we should assume it is wide enough.
    ip_size = ip_size < 0 ? f_size * dim_dilation : ip_size;
    int64_t op_size, pad_before_tf,
        pad_after_tf;  // Complains if using int64_T
    tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
        ip_size, f_size, dim_dilation, dim_stride, tf_pad, &op_size,
        &pad_before_tf, &pad_after_tf);
    if (!status.ok()) return false;

    pad_before = pad_before_tf;
    pad_after = pad_after_tf;
    computed_paddings.push_back(pad_before);
    computed_paddings.push_back(pad_after);
  }

  explicit_padding = rewriter.getI64ArrayAttr(computed_paddings);
  return true;
}

// Calculates the TOSA padding values for explicit-padded TF operators.
//
// This function only handles the TF padding array explicit_padding, which is
// only present in certain TF ops. All others encode padding using the string
// SAME/VALID, which is interpreted using the getPaddingValuesFromPadString
// function below.

// The explicit padding array in TF holds 2 pad values for every
// dimension, even those that are not the 2 spatial ones. Just extract the
// 2x pad values for the XY dims.
ArrayAttr getPaddingValuesFromExplicitPadAttr(
    ArrayAttr explicit_pad, tensorflow::TensorFormat data_format_tf,
    PatternRewriter& rewriter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_11(mht_11_v, 595, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getPaddingValuesFromExplicitPadAttr");

  SmallVector<int64_t> computed_paddings;

  int64_t pad_before, pad_after;
  for (int i = 0; i < 2; i++) {  // Two spatial dimensions X&Y
    int64_t dim = GetTensorSpatialDimIndex(4, data_format_tf,
                                           i);  // 4D tensor, NHWC/NCHW format

    pad_before = explicit_pad[dim * 2].template cast<IntegerAttr>().getInt();
    pad_after = explicit_pad[dim * 2 + 1].template cast<IntegerAttr>().getInt();
    computed_paddings.push_back(pad_before);
    computed_paddings.push_back(pad_after);
  }

  return rewriter.getI64ArrayAttr(computed_paddings);
}

// Calculates the TOSA padding values for transposeConv2d
bool getTransposeConv2dPaddingValues(
    tensorflow::Padding tf_pad, tensorflow::TensorFormat data_format_tf,
    uint32_t first_filter_spatial_dim, ShapedType input_type,
    ShapedType filter_type, ShapedType output_type, ArrayAttr strides,
    ArrayAttr dilations, PatternRewriter& rewriter,
    ArrayAttr& explicit_padding) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_12(mht_12_v, 621, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "getTransposeConv2dPaddingValues");

  assert(tf_pad != tensorflow::Padding::EXPLICIT);
  if (!input_type.hasRank() || !filter_type.hasRank() || !output_type.hasRank())
    return false;

  // Storing the numeric padding values is useful for TOSA codegen, as opposed
  // to holding the padding regime mnemonic, i.e. SAME, VALID, FULL, ...

  SmallVector<int64_t> computed_paddings;

  int64_t pad_before, pad_after;
  for (int i = 0; i < 2; i++) {  // Two spatial dimensions X&Y
    int64_t ifm_dim = GetTensorSpatialDimIndex(
        4, data_format_tf, i);  // 4D tensor, NHWC/NCHW format
    int64_t ofm_dim = GetTensorSpatialDimIndex(
        4, data_format_tf, i);  // 4D tensor, NHWC/NCHW format
    int64_t filter_dim = first_filter_spatial_dim + i;

    int64_t ifm_size = input_type.getDimSize(ifm_dim);
    int64_t filter_size = filter_type.getDimSize(filter_dim);
    int64_t ofm_size = output_type.getDimSize(ofm_dim);
    int64_t dim_dilation = dilations[i].template cast<IntegerAttr>().getInt();
    int64_t dim_stride = strides[i].template cast<IntegerAttr>().getInt();

    // These dimensions need to be static to legalize.
    if (ShapedType::isDynamic(filter_size) || ShapedType::isDynamic(ifm_size) ||
        ShapedType::isDynamic(ofm_size)) {
      return false;
    }

    int effective_filter_size = (filter_size - 1) * dim_dilation + 1;
    int total_padding =
        ((ifm_size - 1) * dim_stride + effective_filter_size - ofm_size);
    total_padding = total_padding > 0 ? total_padding : 0;

    pad_before = total_padding / 2;
    pad_after = total_padding - pad_before;

    computed_paddings.push_back(pad_before);
  }

  explicit_padding = rewriter.getI64ArrayAttr(computed_paddings);
  return true;
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter& rewriter, Operation* op,
                                     ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
llvm::Optional<Value> getConstTensor<APInt>(PatternRewriter& rewriter,
                                            Operation* op, ArrayRef<APInt> vec,
                                            ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(PatternRewriter& rewriter,
                                            Operation* op, ArrayRef<float> vec,
                                            ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(PatternRewriter&,
                                                       Operation*,
                                                       ArrayRef<int32_t> vec,
                                                       ArrayRef<int64_t> shape);

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_13(mht_13_v, 748, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "isScale32");

  return (output_element_type.getStorageTypeIntegralWidth() == 8);
}

LogicalResult ApplyPatternsWithShapeResolution(
    FuncOp func, const FrozenRewritePatternSet& patterns) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSlegalize_utilsDTcc mht_14(mht_14_v, 756, "", "./tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc", "ApplyPatternsWithShapeResolution");

  // We use top-down traversal so that shape inference can fully infer types
  // during pattern rewrite.
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
    return failure();
  }

  // Check that constant attributes types and op types match up. If the lowering
  // needs to change a type (e.g. fp16 -> fp32) its possible the return type
  // could be incorrect.
  //
  // This should be investigate for whether it is still necessary due to quant
  // type stripping changing.
  func.walk([&](tosa::ConstOp op) {
    auto ety = op.value().getType().getElementType();
    auto new_ty = op.getType().cast<ShapedType>().clone(ety);
    op.getResult().setType(new_ty);
  });

  // Insert UnrealizedConversionCasts to guarantee ReturnOp agrees with
  // the FuncOp type.
  IRRewriter rewriter(func.getContext());
  func.walk([&](func::ReturnOp op) {
    FuncOp parent = dyn_cast<FuncOp>(op->getParentOp());
    if (parent != func) return;

    rewriter.setInsertionPoint(op);
    FunctionType func_ty = func.getFunctionType();
    auto result_tys = func_ty.getResults();

    bool cast_added = false;
    SmallVector<Value> return_values;
    for (auto it : llvm::zip(op->getOperands(), result_tys)) {
      Value operand = std::get<0>(it);
      Type current_ty = operand.getType();
      Type cast_ty = std::get<1>(it);
      if (current_ty == cast_ty) {
        return_values.push_back(operand);
        continue;
      }

      return_values.push_back(
          rewriter.create<tensor::CastOp>(op.getLoc(), cast_ty, operand)
              .getResult());

      cast_added = true;
    }

    if (cast_added) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, return_values);
    }
  });

  return success();
}

}  // namespace tosa
}  // namespace mlir
