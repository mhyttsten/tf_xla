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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc() {
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

#include <string>
#include <utility>

#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework ::JITCompileFromStrOp::kJITEntryFunctionName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr int64_t i32BitLimit = 4294967296;
using shape::ShapeOfOp;

bool IsTFOperation(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "IsTFOperation");

  return op != nullptr &&
         op->getDialect() ==
             op->getContext()->getLoadedDialect<TF::TensorFlowDialect>();
}

bool IsUnaryTFOperation(Operation *op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "IsUnaryTFOperation");

  return IsTFOperation(op) && op->getNumOperands() == 1;
}

struct ModuleParameters {
  llvm::ArrayRef<int64_t> tile_sizes;
  llvm::ArrayRef<int64_t> unroll_factors;
  int64_t max_supported_rank;
  bool index_64bit;
  bool cpu_codegen;
};

struct TFToJITInvocationsPattern : public RewritePattern {
  explicit TFToJITInvocationsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "TFToJITInvocationsPattern");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_3(mht_3_v, 255, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "matchAndRewrite");

    // Apply to all TF ops except those that are already in a JIT-compiled
    // region.
    if (!IsTFOperation(op) || op->getParentOfType<tf_framework::JITCompileOp>())
      return failure();

    // Find last TF op.
    while (IsTFOperation(op->getNextNode())) op = op->getNextNode();

    // Find JIT compile region operands and results.
    SmallVector<Operation *, 16> cluster;
    llvm::SmallPtrSet<Value, 16> operand_set, result_set;
    Operation *it = op;
    while (IsTFOperation(it)) {
      // Find results that escape the JIT compile region.
      for (auto &use : it->getUses()) {
        if (!llvm::is_contained(cluster, use.getOwner()))
          result_set.insert(use.get());
      }

      // Update JIT region operands and results.
      for (Value v : it->getResults()) operand_set.erase(v);
      for (Value v : it->getOperands()) operand_set.insert(v);

      cluster.push_back(it);
      it = it->getPrevNode();
    }

    // Introduce order to the operands and results.
    auto operands = llvm::to_vector<16>(operand_set);
    auto results = llvm::to_vector<16>(result_set);
    auto operand_types = llvm::to_vector<16>(
        llvm::map_range(operands, [](Value v) { return v.getType(); }));
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(results, [](Value v) { return v.getType(); }));

    // Create the JIT compile op.
    auto loc = op->getLoc();
    auto jit_compile_op = rewriter.create<tf_framework::JITCompileOp>(
        loc, rewriter.getType<tf_framework::JITCallableType>(), llvm::None);

    // Move the TF operations into the new op's body.
    BlockAndValueMapping bvm;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block =
          rewriter.createBlock(&jit_compile_op.body(), {}, operand_types,
                               SmallVector<Location>(operands.size(), loc));
      for (auto it : llvm::zip(operands, block->getArguments()))
        bvm.map(std::get<0>(it), std::get<1>(it));
      rewriter.setInsertionPointToStart(block);
      for (Operation *it : llvm::reverse(cluster)) rewriter.clone(*it, bvm);
      auto mapped_results = llvm::to_vector<16>(
          llvm::map_range(results, [&](Value v) { return bvm.lookup(v); }));
      rewriter.create<tf_framework::JITCompileYieldOp>(loc, TypeRange{},
                                                       mapped_results);
    }

    // Create JIT execute op.
    auto jit_execute_op = rewriter.create<tf_framework::JITExecuteOp>(
        loc, result_types, Value(), jit_compile_op.result(), operands);

    // Replace old TF ops with the new results.
    for (auto it : llvm::zip(results, jit_execute_op.results()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    for (Operation *it : cluster) {
      if (it->getUses().empty()) {
        rewriter.eraseOp(it);
        continue;
      }
      auto replacements = llvm::to_vector<16>(llvm::map_range(
          it->getResults(), [&](Value v) { return bvm.lookup(v); }));
      rewriter.replaceOp(it, replacements);
    }

    return success();
  }
};

struct PackJITCompileOpPattern
    : public OpRewritePattern<tf_framework::JITCompileOp> {
  using OpRewritePattern<tf_framework::JITCompileOp>::OpRewritePattern;

  explicit PackJITCompileOpPattern(MLIRContext *ctx,
                                   llvm::ArrayRef<int64_t> tile_sizes,
                                   llvm::ArrayRef<int64_t> unroll_factors,
                                   int64_t max_supported_rank, bool enable_ftz,
                                   bool index_64bit_if_jit_compiling,
                                   bool cpu_codegen)
      : OpRewritePattern<tf_framework::JITCompileOp>(ctx),
        tile_sizes(tile_sizes),
        unroll_factors(unroll_factors),
        max_supported_rank(max_supported_rank),
        enable_ftz(enable_ftz),
        index_64bit_if_jit_compiling(index_64bit_if_jit_compiling),
        cpu_codegen(cpu_codegen) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_4(mht_4_v, 353, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "PackJITCompileOpPattern");
}

  LogicalResult matchAndRewrite(tf_framework::JITCompileOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_5(mht_5_v, 359, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "matchAndRewrite");

    Block *body = op.getBody();
    auto yield_op =
        llvm::cast<tf_framework::JITCompileYieldOp>(body->getTerminator());

    // Temporarily, build the module that would be JIT-compiled. This is only to
    // obtain the serialized code attribute.
    auto loc = op->getLoc();
    OpBuilder tmp_module_builder(getContext(), rewriter.getListener());
    auto jit_module = tmp_module_builder.create<ModuleOp>(loc);
    tmp_module_builder.setInsertionPointToStart(jit_module.getBody());
    auto jit_function = tmp_module_builder.create<FuncOp>(
        loc, tf_framework::JITCompileFromStrOp::kJITEntryFunctionName,
        tmp_module_builder.getFunctionType(body->getArgumentTypes(),
                                           yield_op->getOperandTypes()));
    jit_function->setAttr(tf_framework::TFFrameworkDialect::kTFEntryAttrName,
                          tmp_module_builder.getUnitAttr());
    jit_function.getBody().takeBody(op.getBodyRegion());
    tmp_module_builder.setInsertionPointToEnd(&jit_function.getBody().front());
    tmp_module_builder.create<func::ReturnOp>(loc, yield_op.result());
    rewriter.eraseOp(yield_op);

    // Serialize JIT module.
    std::string code;
    llvm::raw_string_ostream ss(code);
    jit_module.print(ss);

    // Finally, create the new JIT compile op.
    rewriter.replaceOpWithNewOp<tf_framework::JITCompileFromStrOp>(
        op, op->getResultTypes(), op.ctx(), rewriter.getStringAttr(code),
        rewriter.getI64ArrayAttr(tile_sizes),
        rewriter.getI64ArrayAttr(unroll_factors),
        rewriter.getI64IntegerAttr(max_supported_rank),
        rewriter.getBoolAttr(enable_ftz),
        rewriter.getBoolAttr(index_64bit_if_jit_compiling),
        rewriter.getBoolAttr(cpu_codegen));

    return success();
  }

 private:
  llvm::ArrayRef<int64_t> tile_sizes;
  llvm::ArrayRef<int64_t> unroll_factors;
  int64_t max_supported_rank;
  bool enable_ftz;
  bool index_64bit_if_jit_compiling;
  bool cpu_codegen;
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct TFToJITInvocationPass
    : public TFToJITInvocationPassBase<TFToJITInvocationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_6(mht_6_v, 416, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "getDependentDialects");

    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect,
                    scf::SCFDialect, shape::ShapeDialect>();
  }
  explicit TFToJITInvocationPass(llvm::ArrayRef<int64_t> tile_sizes,
                                 llvm::ArrayRef<int64_t> unroll_factors,
                                 int64_t max_supported_rank, bool enable_ftz,
                                 bool index_64bit, bool cpu_codegen,
                                 bool jit_i64_indexed_for_large_tensors) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_7(mht_7_v, 427, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "TFToJITInvocationPass");

    tile_sizes_ = tile_sizes;
    unroll_factors_ = unroll_factors;
    max_supported_rank_ = max_supported_rank;
    enable_ftz_ = enable_ftz;
    index_64bit_ = index_64bit;
    cpu_codegen_ = cpu_codegen;
    jit_i64_indexed_for_large_tensors_ = jit_i64_indexed_for_large_tensors;
  }

  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_8(mht_8_v, 440, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "runOnOperation");

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateTFToJITInvocationPatterns(ctx, &patterns, tile_sizes_,
                                      unroll_factors_, max_supported_rank_,
                                      enable_ftz_, index_64bit_, cpu_codegen_,
                                      jit_i64_indexed_for_large_tensors_);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

struct TFToI64JITInvocationForLargeTensorsPattern : public RewritePattern {
  explicit TFToI64JITInvocationForLargeTensorsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_9(mht_9_v, 459, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "TFToI64JITInvocationForLargeTensorsPattern");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_10(mht_10_v, 465, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "matchAndRewrite");

    if (!IsUnaryTFOperation(op) || !llvm::isa<FuncOp>(op->getParentOp())) {
      return failure();
    }

    auto results = llvm::to_vector<16>(op->getResults());
    auto operand_types = llvm::to_vector<16>(llvm::map_range(
        op->getOperands(), [](Value v) { return v.getType(); }));
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(results, [](Value v) { return v.getType(); }));

    // Create the JIT compile op.
    auto loc = op->getLoc();
    Value shape_size_limit =
        rewriter.create<arith::ConstantIndexOp>(loc, i32BitLimit);
    auto arg = op->getOperands().front();
    auto shape = rewriter.create<shape::ShapeOfOp>(loc, arg);
    auto num_elems = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value coniditon_check_main = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, num_elems, shape_size_limit);

    Value conditional_path =
        rewriter
            .create<scf::IfOp>(
                loc, op->getResultTypes(), coniditon_check_main,
                [&](OpBuilder &b, Location l) {
                  auto jit_compile_op =
                      rewriter.create<tf_framework::JITCompileOp>(
                          loc,
                          rewriter.getType<tf_framework::JITCallableType>(),
                          llvm::None);
                  BlockAndValueMapping bvm;
                  {
                    OpBuilder::InsertionGuard guard(rewriter);
                    Block *block = rewriter.createBlock(
                        &jit_compile_op.body(), {}, operand_types,
                        SmallVector<Location>(operand_types.size(), loc));
                    for (auto it :
                         llvm::zip(op->getOperands(), block->getArguments()))
                      bvm.map(std::get<0>(it), std::get<1>(it));
                    rewriter.setInsertionPointToStart(block);
                    rewriter.clone(*op, bvm);
                    auto new_op = rewriter.clone(*op, bvm);
                    rewriter.create<tf_framework::JITCompileYieldOp>(
                        loc, TypeRange{}, new_op->getResults());
                  }
                  auto jit_execute_op =
                      rewriter.create<tf_framework::JITExecuteOp>(
                          loc, result_types, Value(), jit_compile_op.result(),
                          op->getOperands());
                  b.create<scf::YieldOp>(l, jit_execute_op.results());
                },
                [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_11(mht_11_v, 520, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "lambda");

                  auto new_op = rewriter.clone(*op);
                  b.create<scf::YieldOp>(l, new_op->getResult(0));
                })
            .getResult(0);

    rewriter.replaceOp(op, conditional_path);
    return success();
  }
};
}  // namespace

void PopulateTFToJITInvocationPatterns(
    MLIRContext *ctx, RewritePatternSet *patterns,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool enable_ftz, bool index_64bit,
    bool cpu_codegen, bool jit_i64_indexed_for_large_tensors) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_to_jit_invocationsDTcc mht_12(mht_12_v, 539, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_to_jit_invocations.cc", "PopulateTFToJITInvocationPatterns");

  if (jit_i64_indexed_for_large_tensors) {
    patterns->add<TFToI64JITInvocationForLargeTensorsPattern>(ctx);
  } else {
    patterns->add<TFToJITInvocationsPattern>(ctx);
  }

  bool index_64bit_if_jit_compiling =
      jit_i64_indexed_for_large_tensors ? true : index_64bit;
  patterns->add<PackJITCompileOpPattern>(
      ctx, tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
      index_64bit_if_jit_compiling, cpu_codegen);
}

std::unique_ptr<OperationPass<FuncOp>> CreateTFToJITInvocationPass(
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool enable_ftz, bool index_64bit,
    bool cpu_codegen, bool jit_i64_indexed_for_large_tensors) {
  return std::make_unique<TFToJITInvocationPass>(
      tile_sizes, unroll_factors, max_supported_rank, enable_ftz, index_64bit,
      cpu_codegen, jit_i64_indexed_for_large_tensors);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
