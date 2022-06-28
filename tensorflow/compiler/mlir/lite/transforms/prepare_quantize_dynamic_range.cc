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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_dynamic_range_per_channel_quantization(
    "tfl-enable-dynamic-range-per-channel-quantization",
    llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether enable per-channel quantized weights."),
    llvm::cl::init(true));

// NOLINTNEXTLINE
static llvm::cl::opt<int64_t> min_elements_for_weights(
    "tfl-min-elements-for-weights", llvm::cl::value_desc("int64_t"),
    llvm::cl::desc("The minimum number of elements in a weights array required "
                   "to apply quantization."),
    llvm::cl::init(1024));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_float16_quantization(
    "tfl-enable-float16-quantization", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether apply float16 quantization. If false, int8 "
                   "quantization is applied."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> enable_custom_op_quantization(
    "tfl-enable-custom-op-quantization",
    llvm::cl::desc(
        "Specifies which pairs of a custom op and indicies are "
        "quantizable where the indicies are separated with a space."),
    llvm::cl::ZeroOrMore);

//===----------------------------------------------------------------------===//
// The prepare-dynamic-range-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {

// A boolean attribute used to describe whether input activations need to be
// asymmetrically quantized.
constexpr char kAsymmetricQuantizeInputsAttr[] = "asymmetric_quantize_inputs";

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;

// Applies prepare dynamic range quantization on the model in TFL dialect.
// This pass runs before the quantization pass and apply preprocess if
// applicable.
class PrepareDynamicRangeQuantizePass
    : public PassWrapper<PrepareDynamicRangeQuantizePass,
                         OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_0(mht_0_v, 250, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "getDependentDialects");

    registry
        .insert<TensorFlowLiteDialect, ::mlir::quant::QuantizationDialect>();
  }

 public:
  // Constructor used by the PassRegistration. This is only used by test.
  explicit PrepareDynamicRangeQuantizePass() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_1(mht_1_v, 260, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "PrepareDynamicRangeQuantizePass");

    quant_specs_.inference_type = enable_float16_quantization
                                      ? tensorflow::DT_HALF
                                      : tensorflow::DT_QINT8;
    quant_specs_.weight_quantization = true;
    quant_specs_.enable_mlir_dynamic_range_quantizer = true;
    quant_specs_.disable_per_channel =
        !enable_dynamic_range_per_channel_quantization;
    quant_specs_.minimum_elements_for_weights = min_elements_for_weights;
    ParseCustomOpSpecs(enable_custom_op_quantization,
                       quant::CustomOpUpdateOptions::kINputIndices,
                       quant_specs_.custom_map);
  }

  // Constructor used by manually creating the pass.
  explicit PrepareDynamicRangeQuantizePass(
      const quant::QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_2(mht_2_v, 280, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "PrepareDynamicRangeQuantizePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_3(mht_3_v, 285, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "getArgument");

    return "tfl-prepare-quantize-dynamic-range";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "getDescription");

    return "Prepare TFL dialect for dynamic range quantization";
  }

  // The function might contain stats ops which are redundant for processing
  // dynamic range quantization. And stats ops may cause conflict while
  // processing the function for dynamic range quantization. Therefore, this
  // method preprocess the function to remove all stats ops.
  void removeAllStatsOp(FuncOp func);

  void runOnOperation() override;

 private:
  quant::QuantizationSpecs quant_specs_;
};

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

// If the weight is applicable to dynamic range quantization, insert Quantize
// and Dequantize ops with either per-axis or per-tensor scale.
class PrepareDynamicRangeQuantizableOp
    : public OpRewritePattern<arith::ConstantOp> {
 public:
  explicit PrepareDynamicRangeQuantizableOp(
      MLIRContext* context, const quant::QuantizationSpecs& quant_specs)
      : OpRewritePattern<arith::ConstantOp>(context),
        quant_specs_(quant_specs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_5(mht_5_v, 320, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "PrepareDynamicRangeQuantizableOp");
}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_6(mht_6_v, 326, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "matchAndRewrite");

    QuantizationUnits quantizable_ops;

    // 1. Collect quantizable ops.
    if (!(getQuantizableOps(op, quantizable_ops))) {
      return failure();
    }

    // 2. Quantize collected ops. It is immediatly quantized by inserting Q-DQ
    // pair for int8 while it is lazily applied for float16 by inserting CastOp.
    if (!(quantizeOps(rewriter, op, quantizable_ops))) {
      return failure();
    }

    // 3. Apply post-processing required for each inference type.
    // TODO(b/212514817): refactor mode checking to improve code quality
    if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
        (setAsymmetricQuantizeInputAttr(rewriter, quantizable_ops))) {
      return failure();
    }
    if (quant_specs_.inference_type == tensorflow::DT_HALF &&
        (convertToFloat16Constant(rewriter, op))) {
      return failure();
    }

    return success();
  }

 private:
  // Check if the operand_index is included in the quantizable_indices.
  bool isQuantizableIndex(const int operand_index,
                          const std::vector<int>& quantizable_indices) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_7(mht_7_v, 360, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "isQuantizableIndex");

    return std::find(std::begin(quantizable_indices),
                     std::end(quantizable_indices),
                     operand_index) != std::end(quantizable_indices);
  }

  // Check if any specific operand and its index pair is supported for int8
  // quantization. For dynamic range quantizable ops, it refers to the op
  // specification for checking the support. For custom ops, it checks the
  // provided map.
  bool hasInt8QuantizableOperandAt(Operation* op, int operand_index) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_8(mht_8_v, 373, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "hasInt8QuantizableOperandAt");

    if (auto custom_op = llvm::dyn_cast_or_null<CustomOp>(op)) {
      std::string op_name = custom_op.custom_code().str();
      auto custom_map_iter = quant_specs_.custom_map.find(op_name);
      if (custom_map_iter != quant_specs_.custom_map.end())
        return isQuantizableIndex(
            operand_index, custom_map_iter->second.quantizable_input_indices);
    } else if (auto quantizable_op =
                   llvm::dyn_cast<DynamicRangeQuantizedOpInterface>(op)) {
      const auto& quantizable_indices =
          quantizable_op.GetQuantizableOperandIndices();
      return isQuantizableIndex(operand_index, quantizable_indices);
    }
    return false;
  }

  // Insert CastOp which is used to for converting float32 ConstantOp into
  // float16 quantization. If there is an existing CastOp connected to the
  // ConstantOp, the quantize_op will be rewired to the existing CastOp. This
  // guarentees at most one CastOp is created for float32 to float16 conversion.
  void quantizeOpAsFloat16(PatternRewriter& rewriter, arith::ConstantOp op,
                           std::pair<Operation*, int> quant_op) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_9(mht_9_v, 397, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "quantizeOpAsFloat16");

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    // If the constant is an output tensor, do nothing.
    if (llvm::dyn_cast_or_null<func::ReturnOp>(quantize_op)) {
      return;
    }

    // Get types
    TensorType old_result_type =
        op.getResult().getType().template dyn_cast<TensorType>();
    FloatType quantized_type = FloatType::getF16(op.getContext());
    ShapedType new_result_type = old_result_type.clone(quantized_type);

    // Insert CastOp if it does not exist yet. Otherwise, just rewire without
    // creating a CastOp.
    for (auto& connected_op : op.getResult().getUses()) {
      auto cast_op = llvm::dyn_cast_or_null<CastOp>(connected_op.getOwner());
      if (cast_op && cast_op.getType() == new_result_type) {
        quantize_op->setOperand(quantize_operand_num, cast_op);
        return;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto new_cast_op =
        rewriter.create<CastOp>(op->getLoc(), new_result_type, op.getResult());
    quantize_op->setOperand(quantize_operand_num, new_cast_op.getResult());
  }

  // Apply per-axis quantization if applicable. Otherwise, apply per-tensor
  // quantization for int8 dynamic range quantization.
  bool quantizeOpAsInt8(PatternRewriter& rewriter, arith::ConstantOp op,
                        std::pair<Operation*, int> quant_op) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_10(mht_10_v, 433, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "quantizeOpAsInt8");

    bool is_narrow_range = true;
    bool is_legacy_float = quant_specs_.legacy_float_scale;
    bool is_signed = quant_specs_.IsSignedInferenceType();
    int bit_width = quant_specs_.GetQuantizationTypeWidth();

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    auto affine_user = dyn_cast<AffineQuantizedOpInterface>(quantize_op);

    bool op_with_per_axis_support = false;

    if (!llvm::dyn_cast_or_null<CustomOp>(quantize_op)) {
      bool op_with_narrow_range =
          affine_user &&
          affine_user.GetAffineOperandIndex() == quantize_operand_num &&
          affine_user.RequiredNarrowRangeAffineOperand();

      op_with_per_axis_support = op_with_narrow_range &&
                                 affine_user.GetQuantizationDimIndex() != -1 &&
                                 !quant_specs_.disable_per_channel;
    }

    QuantizedType quant_type = nullptr;
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) return false;

    if (attr.dyn_cast<DenseFPElementsAttr>().size() <
        quant_specs_.minimum_elements_for_weights) {
      op->emitRemark("Quantization is skipped for ")
          << quantize_op->getName().getStringRef().str() << " because it has "
          << attr.dyn_cast<DenseFPElementsAttr>().size()
          << " elements which is fewer than the threshold("
          << quant_specs_.minimum_elements_for_weights << " elements).";
      return false;
    }

    if (op_with_per_axis_support) {
      quant_type = quant::GetUniformQuantizedPerAxisTypeForWeight(
                       attr, affine_user.GetQuantizationDimIndex(),
                       /*symmetric=*/true, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    } else {
      quant_type = quant::GetUniformQuantizedTypeForWeight(
                       attr, is_narrow_range && is_signed, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    }
    return insertQDQ(rewriter, op, quant_type, quant_op);
  }

  // Insert Quantize and Dequantize ops.
  bool insertQDQ(PatternRewriter& rewriter, arith::ConstantOp op,
                 QuantizedType quant_type,
                 std::pair<Operation*, int> quant_op) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_11(mht_11_v, 492, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "insertQDQ");

    if (!quant_type) return false;

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    Type expressed_type = op.getResult().getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);

    // Insert DQ-op if it does not exist yet. Otherwise, just rewire without
    // creating a new DQ-op.
    for (auto connected_op : op->getUsers()) {
      auto q_op = llvm::dyn_cast_or_null<Q>(connected_op);
      if (q_op && q_op.getType() == cast_type) {
        auto dq_op = llvm::cast<DQ>(q_op.getResult().use_begin()->getOwner());
        quantize_op->setOperand(quantize_operand_num, dq_op);
        return false;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<Q>(op->getLoc(), cast_type, op.getResult());
    auto dq = rewriter.create<DQ>(op->getLoc(), expressed_type, q);
    quantize_op->setOperand(quantize_operand_num, dq.getResult());
    return true;
  }

  // Mark users that are applicable for dynamic range quantization where the
  // criteria for determining quantizable ops differs by the inferentce type.
  bool getQuantizableOps(arith::ConstantOp op,
                         QuantizationUnits& quantizable_ops) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_12(mht_12_v, 524, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "getQuantizableOps");

    // Non-float tensors do not need quantization.
    auto type = op.getType().dyn_cast<ShapedType>();
    if (!type || !type.getElementType().isF32()) return false;

    Value value = op.getResult();

    // Check whether dynamic range quantization can be applied.
    for (auto& use : value.getUses()) {
      Operation* user = use.getOwner();
      int operand_num = use.getOperandNumber();

      // TODO(b/212514817): refactor mode checking to improve code quality
      if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
          hasInt8QuantizableOperandAt(user, operand_num)) {
        quantizable_ops.insert({user, operand_num});
      } else if (quant_specs_.inference_type == tensorflow::DT_HALF) {
        quantizable_ops.insert({user, operand_num});
      }
    }
    return !quantizable_ops.empty();
  }

  // For each filtered user, apply quantization.
  bool quantizeOps(PatternRewriter& rewriter, arith::ConstantOp op,
                   QuantizationUnits& quantizable_ops) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_13(mht_13_v, 552, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "quantizeOps");

    bool quantized = false;

    // TODO(b/212514817): refactor mode checking to improve code quality
    for (auto& quant_op : quantizable_ops) {
      if (quant_specs_.inference_type == tensorflow::DT_QINT8) {
        quantized |= quantizeOpAsInt8(rewriter, op, quant_op);
      } else if (quant_specs_.inference_type == tensorflow::DT_HALF) {
        quantizeOpAsFloat16(rewriter, op, quant_op);
        quantized = true;
      }
    }
    return quantized;
  }

  // Add asymmetric input quantization attribute. MLIR dynamic quantization
  // supports only the case that the value of the attribute equals to true. For
  // details, see tensorflow/compiler/mlir/lite/quantization/quantization.td
  bool setAsymmetricQuantizeInputAttr(
      PatternRewriter& rewriter, QuantizationUnits& quantizable_ops) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_14(mht_14_v, 574, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "setAsymmetricQuantizeInputAttr");

    bool changed = false;
    for (auto& quant_op : quantizable_ops) {
      auto dynamic_range_quantized_user =
          dyn_cast<DynamicRangeQuantizedOpInterface>(quant_op.first);
      if (dynamic_range_quantized_user &&
          dynamic_range_quantized_user.RequireAsymmetricQuantizeInputsAttr()) {
        // At runtime, this flag will be used in the kernels to decide whether
        // input activations need to be asymmetrically quantized. Refer to the
        // implementation for fully-connected as an example in
        // tensorflow/lite/kernels/fully_connected.cc. The kernels will handle
        // the asymmetric_quantize_inputs attribute in the builtin option.
        dynamic_range_quantized_user->setAttr(
            kAsymmetricQuantizeInputsAttr,
            BoolAttr::get(rewriter.getContext(), true));
        changed = true;
      }
    }
    return changed;
  }

  // Convert ConstantOp-CastOp-Operation sequence into new ConstantOp
  // -Dequantize-Operation where the new ConstantOp has float16 data type.
  bool convertToFloat16Constant(PatternRewriter& rewriter,
                                arith::ConstantOp op) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_15(mht_15_v, 601, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "convertToFloat16Constant");

    for (auto connected_op : op.getResult().getUsers()) {
      auto cast_op = dyn_cast_or_null<CastOp>(connected_op);
      if (!cast_op || cast_op.getResult().use_empty()) continue;

      // Get types
      Type old_result_type = op.getResult().getType();
      ShapedType new_result_type =
          cast_op.getType().template dyn_cast<ShapedType>();

      // Proceeds only if the casting is to float16
      if (!new_result_type.getElementType().isF16()) continue;

      // Cast values
      std::vector<Eigen::half> new_values;
      DenseFPElementsAttr value_attr =
          op.getValue().cast<DenseFPElementsAttr>();
      new_values.reserve(value_attr.getNumElements());
      for (auto value : value_attr.template getValues<float>()) {
        new_values.push_back(Eigen::half(value));
      }
      DenseElementsAttr new_value_attr = DenseFPElementsAttr::get(
          new_result_type, ArrayRef<Eigen::half>(new_values));

      // Create new ConstantOp-Dequantize-Operation sequences. At this moment,
      // old ConstantOp is guaranteed to have one F32->F16 cast regardless of
      // its number of users.
      rewriter.setInsertionPointAfter(op);
      auto new_const = rewriter.create<arith::ConstantOp>(
          op->getLoc(), new_result_type, new_value_attr);
      auto dq = rewriter.create<DQ>(op->getLoc(), old_result_type, new_const);
      cast_op->replaceAllUsesWith(dq);

      // Return without scanning for the next CastOp as only one CastOp is
      // connected to all quantizable ops.
      return true;
    }
    return false;
  }

 protected:
  quant::QuantizationSpecs quant_specs_;
};

// Remove all the stats ops which are redundant for dynamic range quantizaiton.
void PrepareDynamicRangeQuantizePass::removeAllStatsOp(FuncOp func) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_16(mht_16_v, 649, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "PrepareDynamicRangeQuantizePass::removeAllStatsOp");

  func.walk([&](quant::StatisticsOp stats_op) {
    stats_op.replaceAllUsesWith(stats_op.arg());
    stats_op.erase();
  });
}

void PrepareDynamicRangeQuantizePass::runOnOperation() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantize_dynamic_rangeDTcc mht_17(mht_17_v, 659, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc", "PrepareDynamicRangeQuantizePass::runOnOperation");

  FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  ConvertTFLQuantOpsToMlirQuantOps(func);
  removeAllStatsOp(func);

  RewritePatternSet patterns(&getContext());
  patterns.add<PrepareDynamicRangeQuantizableOp>(ctx, quant_specs_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  ConvertMlirQuantOpsToTFLQuantOps(func);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect
// PrepareDynamicRangeQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareDynamicRangeQuantizePass(
    const quant::QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareDynamicRangeQuantizePass>(quant_specs);
}

static PassRegistration<PrepareDynamicRangeQuantizePass> pass;

}  // namespace TFL
}  // namespace mlir
