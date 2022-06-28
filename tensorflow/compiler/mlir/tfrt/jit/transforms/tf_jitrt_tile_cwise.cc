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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc() {
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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;

struct TileCWisePattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileCWisePattern(LinalgTilingOptions options,
                   LinalgTransformationFilter filter, MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_0(mht_0_v, 226, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "TileCWisePattern");
}

  LogicalResult matchAndRewrite(LinalgOp linalg_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "matchAndRewrite");

    // Check if it is cwise on tensors.
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    auto tiled_linalg_op =
        mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_linalg_op) || tiled_linalg_op.getValue().loops.empty())
      return failure();

    LoopOp tiled_loop =
        mlir::dyn_cast<LoopOp>(*tiled_linalg_op.getValue().loops.front());
    if (!tiled_loop) return failure();

    tiled_loop->walk([&](LinalgOp tiledOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    });

    rewriter.replaceOp(linalg_op, tiled_loop->getResults());
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isNonTiledCwiseGeneric(Operation *op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "isNonTiledCwiseGeneric");

  if (op->getParentOfType<LoopOp>()) return false;
  auto linalg_op = mlir::dyn_cast<GenericOp>(op);
  if (linalg_op) {
    if (!linalg_op.hasTensorSemantics()) return false;
    return llvm::all_of(linalg_op.iterator_types(), [](auto type) {
      return mlir::isParallelIterator(type);
    });
  }
  if (auto fill_op = mlir::dyn_cast<FillOp>(op)) {
    return fill_op.hasTensorSemantics();
  }
  return false;
}

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isNonTiledFill(Operation *op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_3(mht_3_v, 283, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "isNonTiledFill");

  if (op->getParentOfType<LoopOp>()) return false;
  if (auto fill_op = mlir::dyn_cast<FillOp>(op)) {
    return fill_op.hasTensorSemantics();
  }
  return false;
}

static constexpr llvm::StringRef kTiledId = "tiled";

void Tile(mlir::func::FuncOp func, int64_t tile_size,
          LinalgTransformationFilter &filter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_4(mht_4_v, 297, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "Tile");

  LinalgTilingOptions tiling_options;
  // Tile the innermost dimension by `tile_size` for vectorization and scalarize
  // the other dimensions.
  tiling_options.setTileSizeComputationFunction(
      [&](OpBuilder b, Operation *op) {
        auto num_loops = llvm::cast<LinalgOp>(op).getNumLoops();
        SmallVector<Value> tiles(num_loops,
                                 b.create<ConstantIndexOp>(op->getLoc(), 1));
        if (!tiles.empty())
          tiles.back() = b.create<ConstantIndexOp>(op->getLoc(), tile_size);
        return tiles;
      });

  mlir::RewritePatternSet patterns(func.getContext());
  patterns.add<TileCWisePattern>(tiling_options, filter, patterns.getContext());
  (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Ensure we drop the marker in the end.
  func.walk([](LinalgOp op) {
    op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
  });
}

struct TileCWisePass : public TileCWiseBase<TileCWisePass> {
  TileCWisePass() = default;
  explicit TileCWisePass(int64_t tile_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_5(mht_5_v, 326, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "TileCWisePass");
 cwise_tile_size = tile_size; }

  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_6(mht_6_v, 331, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "runOnOperation");

    auto func = getOperation();
    auto filter = LinalgTransformationFilter(
                      llvm::ArrayRef<mlir::StringAttr>{},
                      {mlir::StringAttr::get(func.getContext(), kTiledId)})
                      .addFilter([](Operation *op) {
                        return success(isNonTiledCwiseGeneric(op));
                      });
    Tile(func, cwise_tile_size, filter);
  }
};

struct TileFillPass : public TileFillBase<TileFillPass> {
  TileFillPass() = default;
  explicit TileFillPass(int64_t tile_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_7(mht_7_v, 348, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "TileFillPass");
 cwise_tile_size = tile_size; }

  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_cwiseDTcc mht_8(mht_8_v, 353, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_cwise.cc", "runOnOperation");

    auto func = getOperation();
    auto filter = LinalgTransformationFilter(
                      llvm::ArrayRef<mlir::StringAttr>{},
                      {mlir::StringAttr::get(func.getContext(), kTiledId)})
                      .addFilter([](Operation *op) {
                        return success(isNonTiledFill(op));
                      });
    Tile(func, cwise_tile_size, filter);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileCWisePass() {
  return std::make_unique<TileCWisePass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileCWisePass(
    int64_t cwise_tile_size) {
  return std::make_unique<TileCWisePass>(cwise_tile_size);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileFillPass() {
  return std::make_unique<TileFillPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileFillPass(
    int64_t cwise_tile_size) {
  return std::make_unique<TileFillPass>(cwise_tile_size);
}

}  // namespace tensorflow
