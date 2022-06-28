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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc() {
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

// This transformation pass applies some clean up steps after quantization.

#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> enable_custom_op_no_side_effect(
    "tfl-enable-no-side-effect",
    llvm::cl::desc("Specifies which custom ops are NoSideEffect."),
    llvm::cl::ZeroOrMore);

//===----------------------------------------------------------------------===//
// The post-quantize Passes.
//
namespace mlir {
namespace TFL {
namespace {

// Applies all the clean up steps after quantization.
class PostQuantizePass
    : public PassWrapper<PostQuantizePass, OperationPass<FuncOp>> {
 public:
  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PostQuantizePass() : emit_quant_adaptor_ops_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "PostQuantizePass");

    ParseCustomOpSpecs(enable_custom_op_no_side_effect,
                       quant::CustomOpUpdateOptions::kNoSideEffect,
                       custom_op_map_);
  }

  // Constructor used by manually creating the pass.
  explicit PostQuantizePass(bool emit_quant_adaptor_ops,
                            const quant::CustomOpMap& custom_op_map)
      : emit_quant_adaptor_ops_(emit_quant_adaptor_ops),
        custom_op_map_(custom_op_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "PostQuantizePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-post-quantize";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_3(mht_3_v, 244, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Apply post quantization clean up after quantization";
  }

  void runOnOperation() override;

 private:
  // Set this flag to true if the inputs and outputs are in floating point. The
  // quant adaptor ops convert them to fixed point values (i.e. quantize) before
  // feeding them to the model and convert them back to floating point
  // (i.e. dequantize) as the output.
  bool emit_quant_adaptor_ops_;
  quant::CustomOpMap custom_op_map_;
};

// Cleans up unnecessary QDQ pattern for input/output ops.
class PostQuantizeRemoveQDQPass
    : public PassWrapper<PostQuantizeRemoveQDQPass, OperationPass<FuncOp>> {
 public:
  // Constructor used by the PassRegistration. This will remove QDQ ops.
  explicit PostQuantizeRemoveQDQPass() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "PostQuantizeRemoveQDQPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-post-quantize-remove-qdq";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_6(mht_6_v, 281, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Remove qdq from input and output nodes after quantization";
  }

  void runOnOperation() override;
};

// TODO(fengliuai): migrate to use modify_io_nodes pass.
void RemoveQuantizationAdaptorOps(FuncOp func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_7(mht_7_v, 293, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "RemoveQuantizationAdaptorOps");

  mlir::OpBuilder builder(func.getBody());
  auto& bb = func.front();
  auto loc = func.getLoc();

  int num_args = bb.getNumArguments();
  llvm::SmallVector<Type, 4> input_types;
  input_types.reserve(num_args);
  // Edit the block arguments and create the new input ops in place to replace
  // the old input ops and quantize ops.
  for (int i = 0; i != num_args; ++i) {
    // Previous loop iteration may invalidate the insertion point so we have to
    // reset insertion point each iteration.
    builder.setInsertionPointToStart(&bb);

    // In each iteration, a new argument is appended to the end of the list
    // and the current argument is erased, so here we always process the first
    // argument in the list.
    auto arg = bb.getArgument(0);

    auto remove_quantize_op = [&](QuantizeOp quantize_op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_8(mht_8_v, 316, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "lambda");

      auto quantize_output = quantize_op.output();
      auto quantize_type = quantize_output.getType();
      input_types.push_back(quantize_type);
      auto new_arg = bb.addArgument(quantize_type, loc);
      quantize_output.replaceAllUsesWith(new_arg);
      quantize_op.erase();
      arg.dropAllUses();
      bb.eraseArgument(0);
    };

    // This is looking for a pattern: arg -> tfl.quantize
    if (arg.hasOneUse() && llvm::isa<QuantizeOp>(*arg.user_begin())) {
      auto quantize_op = llvm::cast<QuantizeOp>(*arg.user_begin());
      remove_quantize_op(quantize_op);
      continue;
    }

    // Make a copy of current argument and append it to the end of the list if
    // the pattern isn't found.
    Type arg_type = arg.getType();
    input_types.push_back(arg_type);
    auto new_arg = bb.addArgument(arg_type, loc);
    arg.replaceAllUsesWith(new_arg);
    arg.dropAllUses();
    bb.eraseArgument(0);
  }

  // Edit the return ops and remove the dequantize ops in place.
  auto* terminator = bb.getTerminator();
  int num_return_operands = terminator->getNumOperands();
  llvm::SmallVector<Type, 4> output_types;
  output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto returned_value = terminator->getOperand(i);
    Operation* returned_op = returned_value.getDefiningOp();
    if (returned_op && returned_op->hasOneUse() &&
        llvm::isa<DequantizeOp>(returned_op)) {
      auto dequantize_op = llvm::cast<DequantizeOp>(returned_op);
      Value dequantized_result = dequantize_op.input();
      output_types.push_back(dequantized_result.getType());
      terminator->setOperand(i, dequantized_result);
      returned_op->erase();
    } else {
      output_types.push_back(returned_value.getType());
    }
  }
  auto new_func_type = builder.getFunctionType(input_types, output_types);
  func.setType(new_func_type);
}

enum RemoveVolatileOpsType {
  // Remove all volatile quant-dequant ops.
  kPreserveNone,
  // Preserve volatile quant-dequants for input and output ops.
  kPreserveInputsAndOutputs,
};

// Remove the back-to-back quantize and dequantize ops with volatile attribute.
template <RemoveVolatileOpsType remove_volatile_ops_type>
struct RemoveVolatileOps : public OpRewritePattern<DequantizeOp> {
  explicit RemoveVolatileOps(MLIRContext* context)
      : OpRewritePattern<DequantizeOp>(context, 1) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_9(mht_9_v, 381, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "RemoveVolatileOps");
}

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_10(mht_10_v, 387, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "matchAndRewrite");

    auto input_op = op.input().getDefiningOp();
    if (auto q = llvm::dyn_cast_or_null<QuantizeOp>(input_op)) {
      if (!q->getAttr(mlir::quant::kVolatileOpAttrName)) return failure();

      if (remove_volatile_ops_type == kPreserveInputsAndOutputs) {
        // Don't remove leading and trailing QDQ for PTQ workflow, so the io
        // modifying lib can work correctly.
        if (!q.input().getDefiningOp()) return failure();
        if (op->hasOneUse() &&
            op->user_begin()->hasTrait<OpTrait::IsTerminator>())
          return failure();
      }
      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, moving dequantize op before
      // the requantize op to remove the unnecessary requantize op.
      if (auto qtype = quant::QuantizedType::getQuantizedElementType(
              q.input().getType())) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<DequantizeOp>(op, op.output().getType(),
                                                  q.input());
        return success();
      }

      op.replaceAllUsesWith(q.input());
      return success();
    }
    return failure();
  }
};

// Removes operations with side effect (i.e. LSTM, SVDF) that have dangling
// output.
template <typename OpTy>
struct PruneUnusedOpsWithSideEffect : public OpRewritePattern<OpTy> {
 public:
  explicit PruneUnusedOpsWithSideEffect(
      MLIRContext* context, const quant::CustomOpMap& custom_op_map = {})
      : OpRewritePattern<OpTy>(context), custom_op_map(custom_op_map) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_11(mht_11_v, 431, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "matchAndRewrite");

    if (op.getOperation()->template hasTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
    for (auto result : op.getOperation()->getOpResults()) {
      if (!result.use_empty()) {
        return failure();
      }
    }
    // Remove if the custom op is in the provided map and is NoSideEffect.
    auto custom_op = llvm::isa<CustomOp>(op);
    if (custom_op) {
      auto q = llvm::cast<CustomOp>(op);
      std::string op_name = q.custom_code().str();
      if ((custom_op_map.find(op_name) == custom_op_map.end()) ||
          !custom_op_map.find(op_name)->second.no_side_effect)
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
  quant::CustomOpMap custom_op_map;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_post_quantize.inc"

void PostQuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<quant::FoldTrivalRequantizeOp<QuantizeOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::LSTMOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::UnidirectionalSequenceLSTMOp>>(
      ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::SVDFOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::CustomOp>>(ctx,
                                                            custom_op_map_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  if (!emit_quant_adaptor_ops_) {
    RemoveQuantizationAdaptorOps(getOperation());
  }

  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<quant::FoldTrivalRequantizeOp<QuantizeOp>,
                       RemoveVolatileOps<kPreserveInputsAndOutputs>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}

void PostQuantizeRemoveQDQPass::runOnOperation() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSpost_quantizeDTcc mht_12(mht_12_v, 485, "", "./tensorflow/compiler/mlir/lite/transforms/post_quantize.cc", "PostQuantizeRemoveQDQPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<RemoveVolatileOps<kPreserveNone>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops, const quant::CustomOpMap& custom_op_map) {
  return std::make_unique<PostQuantizePass>(emit_quant_adaptor_ops,
                                            custom_op_map);
}

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizeRemoveQDQPass() {
  return std::make_unique<PostQuantizeRemoveQDQPass>();
}

static PassRegistration<PostQuantizePass> pass;

static PassRegistration<PostQuantizeRemoveQDQPass> remove_qdq_pass;

}  // namespace TFL
}  // namespace mlir
