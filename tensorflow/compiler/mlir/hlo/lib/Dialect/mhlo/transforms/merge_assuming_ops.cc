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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc() {
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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

struct ShapeReificationPattern : public OpRewritePattern<shape::ShapeOfOp> {
  explicit ShapeReificationPattern(MLIRContext *context)
      : OpRewritePattern<shape::ShapeOfOp>(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "ShapeReificationPattern");

    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Only reify shape computation if operand allows for it.
    auto shape_origin = op.getArg().getDefiningOp<InferShapedTypeOpInterface>();
    if (!shape_origin) return failure();

    llvm::SmallVector<Value, 1> reifications;
    if (failed(shape_origin.reifyReturnTypeShapes(
            rewriter, shape_origin->getOperands(), reifications)))
      return failure();
    assert(reifications.size() == 1);
    Value reified_shape = reifications.front();

    // Insert cast if needed.
    if (reified_shape.getType() != op.getType()) {
      reified_shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                      reified_shape);
    }

    rewriter.replaceOp(op, reified_shape);
    return success();
  }
};

template <typename OpTy>
struct InlineBroadcastedShapeOperandsPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Find all the shape operands, direct and indirect.
    SmallVector<Value, 8> inlined_operands;
    for (Value direct : op->getOperands()) {
      if (auto bcast_op = direct.getDefiningOp<shape::BroadcastOp>()) {
        for (Value indirect : bcast_op->getOperands())
          inlined_operands.push_back(indirect);
      } else {
        inlined_operands.push_back(direct);
      }
    }

    // Only rewrite if it makes a difference.
    if (inlined_operands.size() == op.getNumOperands()) return failure();

    // Inline shape operands.
    rewriter.replaceOpWithNewOp<OpTy>(op, op->getResultTypes(),
                                      inlined_operands, op->getAttrs());
    return success();
  }
};

bool IsMovable(Operation *op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_3(mht_3_v, 280, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "IsMovable");

  return MemoryEffectOpInterface::hasNoEffect(op) ||
         llvm::isa<shape::CstrBroadcastableOp>(op);
}

LogicalResult MoveUpIntoAssumingOpMatchAndRewrite(Operation *op,
                                                  PatternRewriter &rewriter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "MoveUpIntoAssumingOpMatchAndRewrite");

  // Only implemented for single-result ops.
  if (op->getNumResults() != 1) return failure();

  // Find a preceding `assuming` op.
  auto *the_block = op->getBlock();
  Operation *prev = op->getPrevNode();
  while (prev != nullptr && !llvm::isa<shape::AssumingOp>(prev))
    prev = prev->getPrevNode();
  auto assuming_op = llvm::dyn_cast_or_null<shape::AssumingOp>(prev);
  if (!assuming_op) return failure();
  assert(assuming_op->getBlock() == the_block && op->getBlock() == the_block &&
         "expect assuming op and root op to be in the same block");

  // Make sure that all operands will be available after moving.
  auto is_available = [&](Value v) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_5(mht_5_v, 307, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "lambda");

    Operation *def = v.getDefiningOp();
    return def == nullptr || def->getBlock() != the_block ||
           !assuming_op->isBeforeInBlock(def);
  };
  if (!llvm::all_of(op->getOperands(), is_available)) return failure();

  Block *body = assuming_op.getBody();
  auto yield_op = llvm::cast<shape::AssumingYieldOp>(body->getTerminator());

  // Find the operands to use if the op was within the assuming region. We
  // will later use their copies, as we copy the assuming op and its body.
  SmallVector<Value, 8> new_operands_unmapped =
      llvm::to_vector<8>(llvm::map_range(op->getOperands(), [&](Value v) {
        for (const auto &result : llvm::enumerate(assuming_op->getResults())) {
          if (result.value() == v) return yield_op->getOperand(result.index());
        }
        return v;
      }));

  // Insert the rewritten assuming op right before the old one.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(assuming_op);
  auto new_assuming_op = rewriter.create<shape::AssumingOp>(
      assuming_op.getLoc(), assuming_op.getWitness(),
      [&](OpBuilder &b, Location) {
        // Copy body.
        BlockAndValueMapping mapping;
        for (auto &nested : body->without_terminator())
          b.clone(nested, mapping);

        // Copy op into the new body and use the mapped operands.
        for (auto it : llvm::zip(op->getOperands(), new_operands_unmapped)) {
          Value old_operand, new_operand_unmapped;
          std::tie(old_operand, new_operand_unmapped) = it;
          mapping.map(old_operand,
                      mapping.lookupOrDefault(new_operand_unmapped));
        }
        Operation *new_op = b.clone(*op, mapping);

        // Yield the previous results and also the new ones.
        auto mapped_results = llvm::to_vector<8>(llvm::map_range(
            yield_op.getOperands(),
            [&](Value v) { return mapping.lookupOrDefault(v); }));
        mapped_results.append(new_op->getResults().begin(),
                              new_op->getResults().end());
        return mapped_results;
      });

  // Replace the assuming op and the root op with the corresponding result
  // values.
  ValueRange new_assuming_op_results = new_assuming_op->getResults();
  rewriter.replaceOp(assuming_op, new_assuming_op_results.drop_back());
  rewriter.replaceOp(op, new_assuming_op_results.back());
  return success();
}

/// Move operation into a preceding assuming op. This allows to process
/// operations that depend on the assuming op's results. It will eventually
/// allow to make assuming regions' constraints independent from each other.
template <typename OpTy>
struct MoveUpIntoAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_6(mht_6_v, 375, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    return MoveUpIntoAssumingOpMatchAndRewrite(op.getOperation(), rewriter);
  }
};

// Move elementwise operations into a preceding assuming op. This will
// eventually allow for more fusion opportunities.
struct MoveElementwiseOpsUpIntoAssumingOpPattern : public RewritePattern {
  explicit MoveElementwiseOpsUpIntoAssumingOpPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_7(mht_7_v, 387, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "MoveElementwiseOpsUpIntoAssumingOpPattern");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_8(mht_8_v, 393, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Apply to all elementwise and broadcasting elementwise operations with no
    // side effects.
    if (!op->hasTrait<mlir::OpTrait::Elementwise>() &&
        !op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>()) {
      return failure();
    }
    if (!MemoryEffectOpInterface::hasNoEffect(op)) return failure();

    return MoveUpIntoAssumingOpMatchAndRewrite(op, rewriter);
  }
};

// Move operation into an assuming region if all uses are within its body.
LogicalResult MoveDownIntoAssumingOpMatchAndRewrite(Operation *op,
                                                    PatternRewriter &rewriter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_9(mht_9_v, 411, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "MoveDownIntoAssumingOpMatchAndRewrite");

  auto users = op->getUsers();
  auto it = users.begin();
  auto end = users.end();
  if (it == end) return failure();

  // Find candidate assuming op.
  auto assuming_op = (it++)->getParentOfType<shape::AssumingOp>();
  if (!assuming_op || assuming_op->isProperAncestor(op)) return failure();

  // Make sure all uses are within the unique assuming op's body.
  while (it != end) {
    auto hopefully_same_assuming_op =
        (it++)->getParentOfType<shape::AssumingOp>();
    if (!hopefully_same_assuming_op ||
        hopefully_same_assuming_op != assuming_op) {
      return failure();
    }
  }

  // Move op into the assuming region.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(assuming_op.getBody());
  Operation *new_op = rewriter.clone(*op);
  rewriter.replaceOp(op, new_op->getResults());
  return success();
}

// Move elementwise operations into succeeding assuming regions. This will
// eventually allow for more fusion opportunities.
struct MoveElementwiseOpsDownIntoAssumingOpPattern : public RewritePattern {
  explicit MoveElementwiseOpsDownIntoAssumingOpPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_10(mht_10_v, 446, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "MoveElementwiseOpsDownIntoAssumingOpPattern");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_11(mht_11_v, 452, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Apply to all elementwise and broadcasting elementwise operations with no
    // side effects.
    if (!op->hasTrait<mlir::OpTrait::Elementwise>() &&
        !op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>()) {
      return failure();
    }
    if (!MemoryEffectOpInterface::hasNoEffect(op)) return failure();

    return MoveDownIntoAssumingOpMatchAndRewrite(op, rewriter);
  }
};

/// Move operation out of assuming op. This is only valid for
/// constraint-independent ops, like `cstr_broadcastable` and `shape_of`. It
/// will eventually allow to make assuming regions' constraints independent from
/// each other.
template <typename OpTy>
struct MoveUpOutOfAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_12(mht_12_v, 477, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Must be inside of an assuming op.
    auto assuming_op = op->template getParentOfType<shape::AssumingOp>();
    if (!assuming_op) return failure();

    // Operands must not be defined within the assuming op.
    Block *body = assuming_op.getBody();
    auto is_available = [&](Value v) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_13(mht_13_v, 487, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "lambda");

      Operation *def = v.getDefiningOp();
      return def == nullptr || def->getBlock() != body;
    };
    if (!llvm::all_of(op->getOperands(), is_available)) return failure();

    // Move op before the assuming region.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(assuming_op);
    Operation *new_op = rewriter.clone(*op);
    rewriter.replaceOp(op, new_op->getResults());

    // If the assuming region yields none of the new op's results, these values
    // are exclusively used in the assuming op's body. In these cases there is
    // no need for further rewrites.
    auto is_new_op_result = [&](Value v) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_14(mht_14_v, 505, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "lambda");

      return llvm::is_contained(new_op->getResults(), v);
    };
    auto yield_op = cast<shape::AssumingYieldOp>(body->getTerminator());
    if (llvm::none_of(yield_op.getOperands(), is_new_op_result))
      return success();

    // If the assuming region yields any of the new op's results, these values
    // can instead bypass the assuming region. There is no need to yield them
    // explicitly as they are assumed to be independent. The assuming op is
    // rewritten accordingly.
    SmallVector<Value, 2> replacement_values;
    auto new_assuming_op = rewriter.create<shape::AssumingOp>(
        assuming_op.getLoc(), assuming_op.getWitness(),
        [&](OpBuilder &b, Location) {
          // Copy body.
          BlockAndValueMapping mapping;
          for (Operation &nested : body->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect new yield operands.
          SmallVector<Value, 2> new_yield_operands;
          for (Value result : yield_op.getOperands()) {
            if (is_new_op_result(result)) {
              replacement_values.push_back(result);
            } else {
              new_yield_operands.push_back(mapping.lookupOrDefault(result));
              replacement_values.push_back(nullptr);
            }
          }
          return new_yield_operands;
        });

    // Use the assuming op's results for the missing replacement values.
    auto src = new_assuming_op.getResults().begin();
    for (auto &dst : replacement_values) {
      if (dst) continue;
      dst = *src++;
    }

    rewriter.replaceOp(assuming_op, replacement_values);
    return success();
  }
};

/// Merge assuming regions if their constraints are independent from each other.
struct MergeAssumingOpsPattern : public OpRewritePattern<shape::AssumingOp> {
  using OpRewritePattern<shape::AssumingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::AssumingOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_15(mht_15_v, 559, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Merge assuming op with directly preceding one if both witnesses are
    // availiable.
    auto preceding_op =
        llvm::dyn_cast_or_null<shape::AssumingOp>(op->getPrevNode());
    if (!preceding_op) return failure();
    if (op.getWitness().getDefiningOp() == preceding_op) return failure();

    // Merge witnesses.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(preceding_op);
    Value new_witness = rewriter.create<shape::AssumingAllOp>(
        op.getWitness().getDefiningOp()->getLoc(),
        ValueRange{preceding_op.getWitness(), op.getWitness()});

    // Merge assuming ops.
    Block *body_a = preceding_op.getBody();
    Block *body_b = op.getBody();
    auto new_assuming_op = rewriter.create<shape::AssumingOp>(
        preceding_op.getLoc(), new_witness, [&](OpBuilder &b, Location) {
          // Copy preceding op's body.
          BlockAndValueMapping mapping;
          for (auto &nested : body_a->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Map result values of preceding assuming op.
          auto yield_op_a =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_a->getTerminator());
          for (auto pair : llvm::zip(preceding_op->getResults(),
                                     yield_op_a.getOperands())) {
            mapping.map(std::get<0>(pair),
                        mapping.lookupOrDefault(std::get<1>(pair)));
          }

          // Copy op's body.
          for (auto &nested : body_b->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect merged assuming op's results.
          SmallVector<Value, 4> mapped_results;
          auto yield_op_b =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_b->getTerminator());
          for (Value v : yield_op_a.getOperands()) {
            mapped_results.push_back(mapping.lookupOrDefault(v));
          }
          for (Value v : yield_op_b.getOperands()) {
            mapped_results.push_back(mapping.lookupOrDefault(v));
          }
          return mapped_results;
        });

    // Replace the two assuming ops with the new corresponding results.
    ValueRange new_results = new_assuming_op->getResults();
    size_t split_at = preceding_op->getNumResults();
    rewriter.replaceOp(preceding_op, new_results.take_front(split_at));
    rewriter.replaceOp(op, new_results.drop_front(split_at));
    return success();
  }
};

struct EliminateDuplicateCstrBroadcastableOps
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern<shape::CstrBroadcastableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_16(mht_16_v, 629, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "matchAndRewrite");

    // Search for previous occurence of the same constraint.
    Operation *it = op->getPrevNode();
    while (it != nullptr) {
      if (auto candidate = llvm::dyn_cast<shape::CstrBroadcastableOp>(it)) {
        if (candidate.getShapes() == op.getShapes()) {
          rewriter.replaceOp(op, candidate.getResult());
          return success();
        }
      }
      it = it->getPrevNode();
    }

    return failure();
  }
};

struct MergeAssumingOpsPass
    : public MergeAssumingOpsPassBase<MergeAssumingOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_17(mht_17_v, 651, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "getDependentDialects");

    registry.insert<shape::ShapeDialect, mhlo::MhloDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_18(mht_18_v, 658, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "runOnOperation");

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::PopulateMergeAssumingOpsPatterns(ctx, &patterns);
    GreedyRewriteConfig config;
    config.maxIterations = GreedyRewriteConfig::kNoIterationLimit;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateMergeAssumingOpsPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmerge_assuming_opsDTcc mht_19(mht_19_v, 677, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/merge_assuming_ops.cc", "PopulateMergeAssumingOpsPatterns");

  // clang-format off
  patterns->add<
      EliminateDuplicateCstrBroadcastableOps,
      InlineBroadcastedShapeOperandsPattern<shape::CstrBroadcastableOp>,
      MergeAssumingOpsPattern,
      MoveElementwiseOpsDownIntoAssumingOpPattern,
      MoveElementwiseOpsUpIntoAssumingOpPattern,
      MoveUpIntoAssumingOpPattern<shape::AssumingAllOp>,
      MoveUpIntoAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveUpIntoAssumingOpPattern<shape::ShapeOfOp>,
      MoveUpOutOfAssumingOpPattern<shape::AssumingAllOp>,
      MoveUpOutOfAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveUpOutOfAssumingOpPattern<shape::ShapeOfOp>,
      ShapeReificationPattern>(context);
  // clang-format on
  mhlo::DynamicBroadcastInDimOp::getCanonicalizationPatterns(*patterns,
                                                             context);
  mhlo::DynamicReshapeOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingAllOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingOp::getCanonicalizationPatterns(*patterns, context);
  shape::BroadcastOp::getCanonicalizationPatterns(*patterns, context);
  shape::CstrBroadcastableOp::getCanonicalizationPatterns(*patterns, context);
  tensor::CastOp::getCanonicalizationPatterns(*patterns, context);
}

std::unique_ptr<OperationPass<FuncOp>> createMergeAssumingOpsPass() {
  return std::make_unique<MergeAssumingOpsPass>();
}

}  // namespace mhlo
}  // namespace mlir
