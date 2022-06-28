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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
// Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/transforms/quantize.cc
// This transformation pass applies quantization on TF dialect.
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/quant_spec.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcretTy,
          typename RootOp = DequantizeCastOp>
struct TFQuantizationBase
    : public QuantizationPattern<ConcretTy, QuantizeCastOp, DequantizeCastOp,
                                 /*VERIFIER=*/void, RootOp> {
  explicit TFQuantizationBase(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : QuantizationPattern<ConcretTy, QuantizeCastOp, DequantizeCastOp,
                            /*VERIFIER=*/void, RootOp>(ctx, quant_params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_0(mht_0_v, 237, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "TFQuantizationBase");
}

  // Custom op quantization is not supported.
  static bool IsQuantizableCustomOp(Operation* op,
                                    const CustomMap& custom_op_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_1(mht_1_v, 244, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "IsQuantizableCustomOp");

    return false;
  }

  // Dynamic range quantization is not supported.
  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const CustomMap& custom_op_map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "AllowDynamicRangeQuantizedOperand");

    return false;
  }

  // Dynamic range quantization is not supported.
  static bool AllowDynamicRangeQuantizedResult(Operation* quantized_op,
                                               const CustomMap& custom_op_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "AllowDynamicRangeQuantizedResult");

    return false;
  }

  // Weight-only quantization is not supported.
  static bool IsWeightOnlyOp(Operation* quantized_op, StringSet& ops_blocklist,
                             bool weight_only_quantization,
                             const CustomMap& custom_op_map) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "IsWeightOnlyOp");

    return false;
  }
};

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFFullQuantization
    : public TFQuantizationBase<kFullQuantization, TFFullQuantization> {
  explicit TFFullQuantization(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantization>(
            ctx, quant_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_5(mht_5_v, 286, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "TFFullQuantization");
}
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFFullQuantizationReverse
    : public TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                                QuantizeCastOp> {
  explicit TFFullQuantizationReverse(MLIRContext* ctx,
                                     const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                           QuantizeCastOp>(ctx, quant_params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_6(mht_6_v, 300, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "TFFullQuantizationReverse");
}
};

// Removes quantize-dequantize pairs that are not used in the quantization.
// The benefit of this pattern is set to lower value than other patterns, so
// that the other patterns can work on quantize/dequantize ops first.
class RemoveUnusedQdqPattern : public OpRewritePattern<QuantizeCastOp> {
 public:
  explicit RemoveUnusedQdqPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_7(mht_7_v, 312, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "RemoveUnusedQdqPattern");
}
  LogicalResult matchAndRewrite(QuantizeCastOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_8(mht_8_v, 317, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "matchAndRewrite");

    if (!op->hasOneUse() ||
        !llvm::isa<DequantizeCastOp>(*op->getUsers().begin())) {
      return failure();
    }
    op->getUsers().begin()->getResult(0).replaceAllUsesWith(op.arg());
    return success();
  }
};

class QuantizeSameScaleOpsPattern : public OpRewritePattern<DequantizeCastOp> {
 public:
  explicit QuantizeSameScaleOpsPattern(
      MLIRContext* context, OpQuantScaleSpecGetter op_quant_scale_spec_getter)
      // Set the score to a large number so it is always preferred, after
      // quantization patterns.
      : OpRewritePattern<DequantizeCastOp>(context, /*benefit=*/200),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_9(mht_9_v, 337, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "QuantizeSameScaleOpsPattern");
}

  LogicalResult matchAndRewrite(DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_10(mht_10_v, 343, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "matchAndRewrite");

    llvm::SmallVector<Operation*, 4> quantizing_ops;
    auto users = op.getResult().getUsers();
    quantizing_ops.append(users.begin(), users.end());

    bool changed = false;
    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* quantizing_op : quantizing_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<QuantizeCastOp, DequantizeCastOp>(quantizing_op)) {
        return failure();
      }

      // If the op is terminator, not quantizable or any ops from the mlir quant
      // ops dialect, we shouldn't rewrite.
      if (quantizing_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }

      if (!op_quant_scale_spec_getter_(quantizing_op)
               ->has_same_scale_requirement) {
        continue;
      }

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantizing_op->getNumOperands());
      for (const auto& operand : quantizing_op->getOperands()) {
        Type operand_type = operand.getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }

        Type elem_type = operand_type.cast<TensorType>().getElementType();
        if (auto dq_op =
                dyn_cast_or_null<DequantizeCastOp>(operand.getDefiningOp())) {
          auto dq_arg_type = dq_op.arg().getType().cast<TensorType>();
          auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
          auto scast_op = rewriter.create<StorageCastOp>(
              dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
              dq_op.arg());
          inputs.push_back(scast_op.getResult());
        } else if (!elem_type.isF32()) {
          // If the operand is an integer tensor, then it doesn't require the
          // DQ op in the pattern.
          inputs.push_back(operand);
        } else {
          return failure();
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantizing_op->getNumResults());
      for (const auto& enumerated_result :
           llvm::enumerate(quantizing_op->getResults())) {
        Value result = enumerated_result.value();
        Type result_type = result.getType();
        if (result_type.isa<NoneType>()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        auto result_tensor_type = result_type.cast<TensorType>();
        // If the user is the Quantize op, it must be the only user.
        if (result.hasOneUse() &&
            llvm::isa<QuantizeCastOp>(*result.user_begin())) {
          auto user = llvm::cast<QuantizeCastOp>(*result.user_begin());
          outputs_replaced.insert(
              {user.getResult(), enumerated_result.index()});
          auto qtype = user.getType()
                           .cast<TensorType>()
                           .getElementType()
                           .cast<QuantizedType>();
          output_types.push_back(
              result_tensor_type.clone(qtype.getStorageType()));
        } else if (!result_tensor_type.getElementType().isF32()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          // TODO(b/224691264): separate matching and rewriting clearly.
          return failure();
        }
      }

      rewriter.setInsertionPointAfter(quantizing_op);
      OperationState new_state(quantizing_op->getLoc(),
                               quantizing_op->getName().getStringRef(), inputs,
                               output_types, quantizing_op->getAttrs());
      for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* quantized_op = rewriter.create(new_state);
      if (quantizing_op->getNumRegions() != 0) {
        for (const auto& indexed_regions :
             llvm::enumerate(quantizing_op->getRegions())) {
          BlockAndValueMapping mapping;
          indexed_regions.value().cloneInto(
              &quantized_op->getRegion(indexed_regions.index()), mapping);
        }
      }
      for (const auto& output_index_pair : outputs_replaced) {
        Value output = output_index_pair.getFirst();
        int output_index = output_index_pair.getSecond();
        auto scast_op = rewriter.create<StorageCastOp>(
            output.getLoc(), output.getType(),
            quantized_op->getResult(output_index));
        output.replaceAllUsesWith(scast_op);
      }
      changed = true;
    }
    return success(changed);
  }

 private:
  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;
};

// Applies quantization on the model in TF dialect.
struct QuantizePass : public PassWrapper<QuantizePass, OperationPass<FuncOp>> {
 public:
  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_11(mht_11_v, 475, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "QuantizePass");
 quant_specs.inference_type = tensorflow::DT_QINT8; }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs(quant_specs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_12(mht_12_v, 482, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "QuantizePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_13(mht_13_v, 487, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_14(mht_14_v, 495, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Apply quantization on models in TensorFlow dialect";
  }

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs;
};

void QuantizePass::runOnOperation() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantizeDTcc mht_15(mht_15_v, 509, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize.cc", "QuantizePass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();

  const QuantPassSpec quant_params = {
      {quant_specs.verify_numeric, /*error_tolerance=*/5.0f,
       quant_specs.whole_model_verify, /*enable_log_if_failed=*/false},
      quant_specs};

  patterns.add<TFFullQuantization, TFFullQuantizationReverse>(ctx,
                                                              quant_params);
  patterns.add<QuantizeSameScaleOpsPattern>(ctx, GetTfQuantScaleSpec);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<RemoveUnusedQdqPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow dialect Quantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass() {
  QuantizationSpecs quant_specs;
  return std::make_unique<QuantizePass>(quant_specs);
}

static PassRegistration<QuantizePass> pass;

}  // namespace quant
}  // namespace mlir
