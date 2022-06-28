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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies quantization on TFLite dialect.

#include <cstddef>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_numeric_verify(
    "tfl-numeric-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals at runtime."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<float> error_tolerance(
    "tfl-error-tolerance", llvm::cl::value_desc("float"),
    llvm::cl::desc("Error tolerance for numeric verify. Valid when "
                   "`-tfl-numeric-verify` is set."),
    llvm::cl::init(5.0));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_whole_model_verify(
    "tfl-whole-model-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals layer by layer or whole model. "
                   "Valid when `-tfl-numeric-verify` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_log_if_failed(
    "tfl-log-if-failed", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals with thresholding "
                   "tolerance. Valid when `-tfl-numeric-verify` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_dynamic_range_quantization(
    "tfl-enable-dynamic-range-quantization", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether run post-training dynamic range quantization pass"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_weight_only_quantization(
    "tfl-enable-weight-only-quantization", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether to run weight-only for post-training dynamic range "
                   "quantization pass"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_legacy_quantize(
    "tfl-legacy-quantize", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Use legacy quantize mode in test. Valid when"
                   "`-tfl-legacy-quantize` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::list<std::string> ops_blocklist_flag(
    "tfl-ops-blocklist",
    llvm::cl::desc("Names of ops to blocklist from quantization"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

// NOLINTNEXTLINE
static llvm::cl::list<std::string> nodes_blocklist_flag(
    "tfl-locs-blocklist",
    llvm::cl::desc("Names of location to blocklist from quantization"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> enable_custom_op_weight_only(
    "tfl-enable-custom-op-weight-only",
    llvm::cl::desc("Specifies which custom ops are weight-only."),
    llvm::cl::ZeroOrMore);

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcretTy,
          typename RootOp = DequantizeOp>
struct TFLQuantizationBase
    : public quant::QuantizationPattern<ConcretTy, QuantizeOp, DequantizeOp,
                                        NumericVerifyOp, RootOp> {
  explicit TFLQuantizationBase(MLIRContext* ctx,
                               const quant::QuantPassSpec& quant_params)
      : quant::QuantizationPattern<ConcretTy, QuantizeOp, DequantizeOp,
                                   NumericVerifyOp, RootOp>(ctx, quant_params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_0(mht_0_v, 300, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "TFLQuantizationBase");

  }

  static bool IsQuantizableCustomOp(Operation* op,
                                    const quant::CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_1(mht_1_v, 307, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "IsQuantizableCustomOp");

    // In some cases, ops may need to be quantized even though their op trait is
    // not quantizable. For example, for the case of custom op various ops can
    // be categorized as cusom ops despite each of them may require different
    // behaviors. In that case, these ops can be marked in the custom map and
    // treated separately in this pass.

    auto custom_op = llvm::dyn_cast_or_null<TFL::CustomOp>(op);
    if (!custom_op) return false;

    // Custom op which is marked in the custom op map is quantizable.
    std::string op_name = custom_op.custom_code().str();
    return (custom_op_map.find(op_name) == custom_op_map.end()) ? false : true;
  }

  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const quant::CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_2(mht_2_v, 326, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "AllowDynamicRangeQuantizedOperand");

    // Collect the input if dynamic range quantization is on and the op supports
    // it.

    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool AllowDynamicRangeQuantizedResult(
      Operation* quantized_op, const quant::CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_3(mht_3_v, 339, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "AllowDynamicRangeQuantizedResult");

    // Collect the output if dynamic range quantization is on and the op
    // supports it.

    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool IsWeightOnlyOp(Operation* quantized_op, StringSet& ops_blocklist,
                             bool weight_only_quantization,
                             const quant::CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_4(mht_4_v, 353, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "IsWeightOnlyOp");

    // Check whether the quantized_op needs to be quantized in weight-only
    // manner.
    bool is_blocklisted = false;

    if (auto custom_op = dyn_cast_or_null<CustomOp>(quantized_op)) {
      std::string custom_op_name = custom_op.custom_code().str();
      auto custom_map_iter = custom_op_map.find(custom_op_name);

      is_blocklisted =
          ops_blocklist.find(custom_op_name) != ops_blocklist.end();

      bool weight_only_custom_op = custom_map_iter != custom_op_map.end()
                                       ? custom_map_iter->second.is_weight_only
                                       : false;

      return is_blocklisted || weight_only_custom_op ||
             weight_only_quantization;
    } else {
      auto dynamic_range_op =
          dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op);

      const auto op_name = quantized_op->getName().getStringRef().str();
      is_blocklisted = ops_blocklist.find(op_name) != ops_blocklist.end();

      bool kernel_support =
          dynamic_range_op.GetDynamicRangeQuantKernelSupport();

      return is_blocklisted || !kernel_support || weight_only_quantization;
    }
  }
};

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFLFullQuantization
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantization> {
  explicit TFLFullQuantization(MLIRContext* ctx,
                               const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantization>(
            ctx, quant_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_5(mht_5_v, 395, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "TFLFullQuantization");
}
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFLFullQuantizationReverse
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                                 QuantizeOp> {
  explicit TFLFullQuantizationReverse(MLIRContext* ctx,
                                      const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                            QuantizeOp>(ctx, quant_params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_6(mht_6_v, 409, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "TFLFullQuantizationReverse");
}
};

// Dynamic range quantization rewrite pattern using DQ as the root op.
struct TFLDynamicRangeQuantization
    : public TFLQuantizationBase<kDynamicRangeQuantization,
                                 TFLDynamicRangeQuantization> {
  explicit TFLDynamicRangeQuantization(MLIRContext* ctx,
                                       const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kDynamicRangeQuantization,
                            TFLDynamicRangeQuantization>(ctx, quant_params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_7(mht_7_v, 422, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "TFLDynamicRangeQuantization");
}
};

class QuantizeConstPattern : public OpRewritePattern<QuantizeOp> {
 public:
  explicit QuantizeConstPattern(MLIRContext* context, bool legacy_float_scale)
      : OpRewritePattern<QuantizeOp>(context),
        legacy_float_scale_(legacy_float_scale) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_8(mht_8_v, 432, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "QuantizeConstPattern");
}
  LogicalResult matchAndRewrite(QuantizeOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_9(mht_9_v, 437, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "matchAndRewrite");

    DenseFPElementsAttr attr;
    if (matchPattern(op.input(), m_Constant(&attr))) {
      auto qtype = op.qtypeAttr();
      Attribute quantized_attr;
      if (legacy_float_scale_) {
        quantized_attr = quant::QuantizeLegacy(attr, qtype.getValue());
      } else {
        quantized_attr = quant::Quantize(attr, qtype.getValue());
      }
      if (quantized_attr) {
        rewriter.replaceOpWithNewOp<QConstOp>(op, qtype, quantized_attr);
        return success();
      }
    }
    return failure();
  }

 private:
  bool legacy_float_scale_;
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public PassWrapper<QuantizePass, OperationPass<FuncOp>> {
 public:
  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_10(mht_10_v, 466, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "QuantizePass");

    quant_specs.inference_type = tensorflow::DT_QINT8;
    quant_specs.verify_numeric = enable_numeric_verify;
    quant_specs.whole_model_verify = enable_whole_model_verify;
    quant_specs.legacy_float_scale = enable_legacy_quantize;
    quant_specs.weight_quantization = enable_dynamic_range_quantization;
    quant_specs.weight_only_quantization = enable_weight_only_quantization;
    quant_specs.ops_blocklist =
        StringSet(ops_blocklist_flag.begin(), ops_blocklist_flag.end());
    quant_specs.nodes_blocklist =
        StringSet(nodes_blocklist_flag.begin(), nodes_blocklist_flag.end());
    ParseCustomOpSpecs(enable_custom_op_weight_only,
                       quant::CustomOpUpdateOptions::kWeightOnly,
                       quant_specs.custom_map);
  }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const quant::QuantizationSpecs& quant_specs)
      : quant_specs(quant_specs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_11(mht_11_v, 487, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "QuantizePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_12(mht_12_v, 492, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-quantize";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSquantizeDTcc mht_13(mht_13_v, 500, "", "./tensorflow/compiler/mlir/lite/transforms/quantize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Apply quantization on models in TensorFlow Lite dialect";
  }

  void runOnOperation() override;

 private:
  quant::QuantizationSpecs quant_specs;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();

  const quant::QuantPassSpec quant_params = {
      {quant_specs.verify_numeric, error_tolerance,
       quant_specs.whole_model_verify, enable_log_if_failed},
      quant_specs};

  TFL::populateWithGenerated(patterns);

  if (quant_specs.weight_quantization || quant_specs.use_fake_quant_num_bits) {
    patterns.add<TFLDynamicRangeQuantization>(ctx, quant_params);
  } else {
    patterns.add<TFLFullQuantization, TFLFullQuantizationReverse>(ctx,
                                                                  quant_params);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<QuantizeConstPattern>(ctx, quant_specs.legacy_float_scale);
  if (quant_params.numeric_verify_spec.whole_model_verify) {
    patterns_2.add<quant::RemoveDebugAttrPattern>(ctx);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    const quant::QuantizationSpecs& quant_specs, const StringSet& ops_blocklist,
    const StringSet& nodes_blocklist) {
  quant::QuantizationSpecs updated_quant_specs;
  updated_quant_specs = quant_specs;
  // If there's new blocklists given, update quant_specs to use the new one.
  if (!ops_blocklist.empty()) {
    updated_quant_specs.ops_blocklist = ops_blocklist;
  }
  if (!nodes_blocklist.empty()) {
    updated_quant_specs.nodes_blocklist = nodes_blocklist;
  }
  return std::make_unique<QuantizePass>(updated_quant_specs);
}

std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    bool verify_numeric, bool whole_model_verify, bool legacy_float_scale,
    const StringSet& ops_blocklist, const StringSet& nodes_blocklist) {
  quant::QuantizationSpecs quant_specs;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = ops_blocklist;
  quant_specs.nodes_blocklist = nodes_blocklist;
  return std::make_unique<QuantizePass>(quant_specs);
}

static PassRegistration<QuantizePass> pass;

}  // namespace TFL
}  // namespace mlir
