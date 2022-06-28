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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc() {
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

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// This function takes ForOps that contain AffineMinOps and possibly peels off
// the last iteration of the loop. This is done in cases where it is provable
// that the AffineMinOp is deterministic in all cases except the possible last
// iteration. Some additional cleanup is done to simplify the IR that is correct
// through knowledge of what this transformation is doing but would generally be
// unwieldy in a canonicalization-like pattern.
//
// This pass is only necessary due to inefficiencies in VectorTransferSplit that
// is unlikely to be fixed upstream. If that changes, this pass can be fully
// removed.
//
// Example:
// scf.for %i = %c0 to %c11 step %c2
//   %a = affine.min(%c2, %c11-%i)
//
// Becomes:
// scf.for %i = %c0 to %c10 step %c2
//   %a = %c2
// scf.if %one_more_iter
//   %a = affine.min(2, %c11-%i)
//
// This is possible because we can determine that the min will always be 2
// except for the last iteration.
void SplitSCFForOp(scf::ForOp scf_for) {
  // The set of following steps is:
  // 1. Validate that there are min_ops to be modified in this function.
  // 2. Create the boundary that decides whether the min_op evaluates to the
  // loop's step value or to the computed value based upon the iteration value.
  // 3. Create the primary loop that does all the work except for possibly the
  // last iteration of the loop, and replace all relevant min_ops with the step.
  // 4. Create the final iteration, remove the step from relevant min_ops, and
  // additionally modify related if/else ops to have a constant condition based
  // on what we know about this loop structure.

  // Match only when the lower bound is zero and the step is constant.
  // TODO(TPOPP): Requiring constant steps and lower bound simplifies things
  // but isn't necesarilly needed
  auto lower_bound_op = llvm::dyn_cast<arith::ConstantOp>(
      scf_for.getLowerBound().getDefiningOp());
  if (!lower_bound_op) {
    return;
  }
  auto lower_bound_value = lower_bound_op.getValue().dyn_cast<IntegerAttr>();
  if (!lower_bound_value || lower_bound_value.getInt() != 0) {
    return;
  }

  auto step_bound_op =
      llvm::dyn_cast<arith::ConstantOp>(scf_for.getStep().getDefiningOp());
  if (!step_bound_op) {
    return;
  }
  auto step_bound_value = step_bound_op.getValue().dyn_cast<IntegerAttr>();
  if (!step_bound_value) {
    return;
  }

  auto loc = scf_for.getLoc();
  ImplicitLocOpBuilder b(loc, scf_for);

  // This function will determine if the min_op is an operation that can be
  // transformed after loop splitting. This relies on the function that the op
  // represents relative to the induction variable in its loop and the
  // bounds of the original for loop.
  auto is_op_of_interest = [&](AffineMinOp min_op, Value iv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_0(mht_0_v, 278, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "lambda");

    bool min_by_step = false;
    for (auto i : min_op.getAffineMap().getResults()) {
      if (i == b.getAffineConstantExpr(step_bound_value.getInt())) {
        min_by_step = true;
        continue;
      }
      if (i == b.getAffineSymbolExpr(0) - b.getAffineDimExpr(0) &&
          min_op.getDimOperands().front() == iv &&
          min_op.getSymbolOperands().front() == scf_for.getUpperBound())
        continue;
      if (i == b.getAffineDimExpr(0) - b.getAffineDimExpr(1) &&
          min_op.getDimOperands().drop_front().front() == iv &&
          min_op.getDimOperands().front() == scf_for.getUpperBound())
        continue;
      if (auto idx_op =
              scf_for.getUpperBound().getDefiningOp<arith::ConstantIndexOp>()) {
        auto val = idx_op.value();
        if (i == b.getAffineConstantExpr(val) - b.getAffineDimExpr(0) &&
            min_op.getDimOperands().front() == iv)
          continue;
      }
      return false;
    }
    return min_by_step;
  };

  // Determine if the loop should be split based on the existence of
  // AffineMinOps of an expected form.
  llvm::SmallVector<AffineMinOp, 1> min_ops;
  scf_for->walk([&](AffineMinOp min_op) {
    if (is_op_of_interest(min_op, scf_for.getInductionVar()))
      min_ops.push_back(min_op);
  });
  if (min_ops.empty()) {
    return;
  }

  // Split the loop just before a possible last iteration.
  b.setInsertionPoint(scf_for);
  Value split_point = b.create<arith::SubIOp>(
      scf_for.getUpperBound(),
      b.create<arith::RemUIOp>(b.create<arith::SubIOp>(scf_for.getUpperBound(),
                                                       scf_for.getLowerBound()),
                               scf_for.getStep()));

  // New primary loop with relevant min ops replaced with their constant value
  BlockAndValueMapping mapper;
  auto new_loop = llvm::cast<scf::ForOp>(b.clone(*scf_for, mapper));
  new_loop.setUpperBound(split_point);

  new_loop->walk([&](AffineMinOp min_op) {
    if (is_op_of_interest(min_op, new_loop.getInductionVar()))
      min_op->replaceAllUsesWith(llvm::makeArrayRef(scf_for.getStep()));
  });

  // Peeled loop iteration (or nothing if perfectly aligned data and step sizes)
  BlockAndValueMapping tail_mapper;
  tail_mapper.map(scf_for.getRegionIterArgs(), new_loop.getResults());
  tail_mapper.map(scf_for.getInductionVar(), split_point);
  auto tail_if = b.create<scf::IfOp>(
      scf_for.getResultTypes(),
      b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, split_point,
                              scf_for.getUpperBound()),
      [&](OpBuilder &then_b, Location loc) {
        for (auto &op : *scf_for.getBody()) {
          then_b.clone(op, tail_mapper);
        }
      }, scf_for->getNumResults() ?
      [&](OpBuilder &else_b, Location loc) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_1(mht_1_v, 350, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "lambda");

        else_b.clone(scf_for.getBody()->back(), tail_mapper);
      } : static_cast<function_ref<void(OpBuilder &, Location)>>(nullptr));

  tail_if->walk([&](AffineMinOp min_op) {
    SmallVector<AffineExpr> exprs;

    if (!is_op_of_interest(min_op, split_point)) return;

    ImplicitLocOpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(min_op);

    // This function is to be called on comparisons that use the min_ops of
    // interest in the last loop iteration. Through loop splitting, we know that
    // the min result is strictly less than the step value. Therefore, we can
    // take the predicate and a statement regarding the location of the min_op
    // (and the implied position of the step value) to evaluate the cmpi.
    auto is_true_cmp = [](arith::CmpIPredicate pred, bool min_is_op_0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_2(mht_2_v, 370, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "lambda");

      switch (pred) {
        // This loop splitting guarantees the step is not equal to the min on
        // the last iteration.
        case arith::CmpIPredicate::eq:
        case arith::CmpIPredicate::ne:
          return false;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ule:
        case arith::CmpIPredicate::ult:
          return min_is_op_0;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::uge:
        case arith::CmpIPredicate::ugt:
          return !min_is_op_0;
      }
    };

    for (auto user : min_op->getUsers()) {
      if (auto cmp = dyn_cast<arith::CmpIOp>(user)) {
        if (cmp.getOperand(0) == min_op.getResult() &&
            cmp.getOperand(1) == step_bound_op) {
          cmp.replaceAllUsesWith(b.create<arith::ConstantIntOp>(
                                      is_true_cmp(cmp.getPredicate(), true), 1)
                                     .getResult());
          cmp.erase();
        } else if (cmp.getOperand(0) == step_bound_op &&
                   cmp.getOperand(1) == min_op.getResult()) {
          cmp.replaceAllUsesWith(b.create<arith::ConstantIntOp>(
                                      is_true_cmp(cmp.getPredicate(), false), 1)
                                     .getResult());
        }
      }
    }

    // Replace the min_op with a simplified min_op that removes the constant
    // step option. This will be further simplified after affine ops are
    // lowered.
    auto map = min_op.getAffineMap();
    for (auto i : map.getResults()) {
      if (i != b.getAffineConstantExpr(step_bound_value.getInt()))
        exprs.push_back(i);
    }

    Value new_min = b.createOrFold<AffineMinOp>(
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs,
                       b.getContext()),
        min_op.operands());

    min_op->replaceAllUsesWith(llvm::makeArrayRef(new_min));
  });

  scf_for->replaceAllUsesWith(tail_if.getResults());
  scf_for.erase();
}

// A pass to remove memref::AllocOps and other ops interacting with the memrefs
// if it is provable that this will not change the results of the program. This
// is determined by confirming all consumers of all aliases are only creating an
// alias or writing data to an alias but never reading from or interacting with
// the memref in other ways.
void RemoveDeadMemrefCode(FuncOp func) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_3(mht_3_v, 436, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "RemoveDeadMemrefCode");

  BufferViewFlowAnalysis baa(func);
  llvm::SmallSet<Operation *, 8> to_remove;

  // Gather all operations interacting with memrefs guaranteed to never be read
  // from.
  func->walk([&](memref::AllocaOp op) {
    llvm::SmallVector<Operation *> maybe_to_remove;
    for (auto &alias : baa.resolve(op.getResult())) {
      for (auto user : alias.getUsers()) {
        if (!(isa<ViewLikeOpInterface>(user) ||
              (isa<memref::CopyOp>(user) &&
               alias == cast<memref::CopyOp>(user).target()) ||
              (isa<linalg::FillOp>(user) &&
               alias == cast<linalg::FillOp>(user).output()))) {
          return;
        }
        maybe_to_remove.push_back(user);
      }
    }
    to_remove.insert(maybe_to_remove.begin(), maybe_to_remove.end());
    to_remove.insert(op);
  });

  // Erase after the walk to avoid corrupting data being traversed.
  for (auto *op : to_remove) {
    op->dropAllUses();
    op->erase();
  }
}

struct VectorizationPass : public VectorizationPassBase<VectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_4(mht_4_v, 471, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "getDependentDialects");

    registry.insert<vector::VectorDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_5(mht_5_v, 479, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "runOnOperation");

    // This functions in 2 passes:
    // 1. Tile, promote, and vectorize to create elementwise operations on
    //    <(1x)*4xty> memrefs
    // 2. cast <(1x)*4xty> memrefs to <4xty>
    auto f = getOperation();

    // Stage 1: Vectorize to form static shaped computations
    auto tiling_options =
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder b, Operation *op) {
              auto num_loops = llvm::cast<linalg::LinalgOp>(op).getNumLoops();
              SmallVector<Value> tiles(
                  num_loops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
              if (!tiles.empty())
                tiles.back() =
                    b.create<arith::ConstantIndexOp>(op->getLoc(), 4);
              return tiles;
            });
    auto alignment = 16;

    mlir::linalg::CodegenStrategy strategy;
    strategy.tile(mlir::linalg::GenericOp::getOperationName(), tiling_options)
        .promote(mlir::linalg::GenericOp::getOperationName(),
                 mlir::linalg::LinalgPromotionOptions()
                     .setAlignment(alignment)
                     .setUseFullTileBuffersByDefault(true)
                     .setUseAlloca(true))
        .vectorize(mlir::linalg::GenericOp::getOperationName())
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .enableTransferLowering(false)
                .enableTransferPartialRewrite()
                .setVectorTransformsOptions(
                    mlir::vector::VectorTransformsOptions()
                        .setVectorTransferSplit(
                            mlir::vector::VectorTransferSplit::VectorTransfer))
                .enableTransferToSCFConversion()
                .setVectorTransferToSCFOptions(
                    mlir::VectorTransferToSCFOptions().enableFullUnroll())
                .enableContractionLowering());

    // Created a nested OpPassManager, populate the strategy and run.
    OpPassManager dynamicPM("func.func");
    strategy.configurePassPipeline(dynamicPM, f.getContext());
    if (failed(runPipeline(dynamicPM, f))) return signalPassFailure();

    // Stage 2: Remove extent 1 dims to ensure correct 1-ranked vectorization
    auto ctx = f.getContext();
    RewritePatternSet patterns(ctx);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

struct VectorizationCleanupPass
    : public VectorizationCleanupPassBase<VectorizationCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_6(mht_6_v, 546, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "getDependentDialects");

    registry.insert<memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSvectorization_passDTcc mht_7(mht_7_v, 554, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/vectorization_pass.cc", "runOnOperation");

    getOperation().walk([](scf::ForOp op) { SplitSCFForOp(op); });

    RemoveDeadMemrefCode(getOperation());
  }
};

std::unique_ptr<OperationPass<FuncOp>> CreateVectorizationCleanupPass() {
  return std::make_unique<VectorizationCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
