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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc() {
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

// This file implements logic for fusing linalg ops obtained after LHLO
// lowering.

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/lhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/lhlo/transforms/passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {
namespace {

using linalg::LinalgOp;

class LhloFuseLinalgPass : public LhloFuseLinalgPassBase<LhloFuseLinalgPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc", "getDependentDialects");

    registry.insert<AffineDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

 public:
  LhloFuseLinalgPass() = default;
  LhloFuseLinalgPass(const LhloFuseLinalgPass&) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc", "LhloFuseLinalgPass");
}
  LhloFuseLinalgPass(bool use_parallel_loops,
                     llvm::ArrayRef<unsigned> tile_sizes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc", "LhloFuseLinalgPass");

    tile_sizes_ = tile_sizes;
    use_parallel_loops_.setValue(use_parallel_loops);
  }

  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc mht_3(mht_3_v, 235, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc", "runOnOperation");

    auto func = getOperation();

    // TODO(pifon): Remove assumption that the function has a single block.
    if (!llvm::hasSingleElement(func)) {
      emitError(func.getLoc(), "The function needs to have a single block.");
      signalPassFailure();
      return;
    }

    // The fusion in Linalg is currently possible only when the consumer op is
    // tiled. In order to greedily fuse the ops, we have to start from the tiled
    // root linalg ops, i.e. linalg ops that write to output buffers of the
    // function or are returned in case of escaping allocations.
    llvm::SmallDenseSet<Value> result_buffers;
    for (auto func_arg : func.getArguments()) {
      result_buffers.insert(func_arg);
    }
    for (auto& block : func) {
      auto returnOp =
          mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
      if (!returnOp) continue;
      for (auto operand : returnOp.getOperands()) {
        result_buffers.insert(operand);
      }
    }
    // Resolve aliasing operations (like casts) on the result to identify
    // results. This only handles escaping results.
    // TODO(herhut): Use BufferizeAliasAnalysis for this.
    llvm::SmallVector<Value, 4> worklist(result_buffers.begin(),
                                         result_buffers.end());
    while (!worklist.empty()) {
      Value result = worklist.pop_back_val();
      auto* definingOp = result.getDefiningOp();
      if (!definingOp) {
        continue;
      }

      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(definingOp)) {
        auto alias = viewLike.getViewSource();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto to_tensor = dyn_cast<bufferization::ToTensorOp>(definingOp)) {
        auto alias = to_tensor.memref();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto to_memref = dyn_cast<bufferization::ToMemrefOp>(definingOp)) {
        auto alias = to_memref.tensor();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto tensor_cast = dyn_cast<tensor::CastOp>(definingOp)) {
        auto alias = tensor_cast.source();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto regionInterface =
              dyn_cast<RegionBranchOpInterface>(definingOp)) {
        for (Region& region : regionInterface.getOperation()->getRegions()) {
          // Only consider regions that can return to the parent region.
          SmallVector<RegionSuccessor, 2> successorRegions;
          regionInterface.getSuccessorRegions(region.getRegionNumber(),
                                              successorRegions);
          if (llvm::none_of(successorRegions, [&](auto successorRegion) {
                return successorRegion.isParent();
              }))
            continue;

          // Iterate over all immediate terminators and record the values
          // corresponding to result_buffers of interest.
          for (Block& block : region) {
            if (block.empty()) continue;
            Operation& operation = block.back();
            if (!operation.hasTrait<OpTrait::ReturnLike>()) continue;
            auto idx = result.dyn_cast<OpResult>().getResultNumber();
            if (result_buffers.insert(operation.getOperand(idx)).second) {
              worklist.push_back(operation.getOperand(idx));
            }
          }
        }
      }
    }

    MLIRContext* ctx = func.getContext();
    OpBuilder b(func);
    func.walk([&](linalg::GenericOp generic_op) {
      SmallVector<int64_t, 2> tile_sizes(tile_sizes_.begin(),
                                         tile_sizes_.end());
      if (tile_sizes.empty()) {
        tile_sizes = SmallVector<int64_t, 2>(generic_op.getNumLoops(), 1);
      }
      auto op = cast<LinalgOp>(generic_op.getOperation());
      for (OpOperand* op_operand : op.getOutputBufferOperands()) {
        if (!result_buffers.count(op_operand->get())) continue;
        if (tileGenericOp(op, tile_sizes, &b)) {
          generic_op.erase();
          return;
        }
      }
    });
    auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();

    // Fuse producers of tiled linalg ops.
    llvm::SmallDenseSet<Operation*> erase_set;
    SmallVector<LinalgOp, 8> linalg_ops;
    func.walk([&](LinalgOp op) { linalg_ops.push_back(op); });
    for (LinalgOp op : llvm::reverse(linalg_ops)) {
      for (OpOperand* inputOperand : op.getInputOperands()) {
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalg_ops);
        auto info = fuseProducerOfBuffer(b, *inputOperand, graph);
        if (failed(info)) continue;
        auto* originalOp = info->originalProducer.getOperation();
        erase_set.insert(originalOp);
        auto* originalOpInLinalgOpsVector =
            std::find_if(linalg_ops.begin(), linalg_ops.end(),
                         [&](const Operation* op) { return op == originalOp; });
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
      }

      auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
        return signalPassFailure();
    }
    for (auto* e : erase_set) e->erase();
  }

 private:
  bool tileGenericOp(LinalgOp op, ArrayRef<int64_t> tile_sizes, OpBuilder* b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_fuse_linalgDTcc mht_4(mht_4_v, 382, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc", "tileGenericOp");

    auto loopType = use_parallel_loops_
                        ? linalg::LinalgTilingLoopType::ParallelLoops
                        : linalg::LinalgTilingLoopType::Loops;
    IRRewriter rewriter(*b);
    return succeeded(linalg::tileLinalgOp(rewriter, op,
                                          linalg::LinalgTilingOptions()
                                              .setTileSizes(tile_sizes)
                                              .setLoopType(loopType)));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLhloFuseLinalgPass(
    bool use_parallel_loops, ArrayRef<unsigned> tile_sizes) {
  return std::make_unique<LhloFuseLinalgPass>(use_parallel_loops, tile_sizes);
}

}  // namespace lmhlo
}  // namespace mlir
