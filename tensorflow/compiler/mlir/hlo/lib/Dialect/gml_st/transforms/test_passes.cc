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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc() {
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

#include "mlir-hlo/Dialect/gml_st/transforms/test_passes.h"

#include <string>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_CLASSES
#include "mlir-hlo/Dialect/gml_st/transforms/test_passes.h.inc"

static constexpr char kPeeledLoopsLabel[] = "__peeled_loops__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

/// Peel LoopOps, i.e., split them into two loops: One loop where the
/// `idx`-th loop contains only "full" iterations and a second loop for the
/// remaining partial iteration (if any).
struct TiledLoopPeelingPattern : public OpRewritePattern<LoopOp> {
  TiledLoopPeelingPattern(MLIRContext *ctx, int64_t idx, bool skip_partial)
      : OpRewritePattern<LoopOp>(ctx), idx(idx), skip_partial(skip_partial) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "TiledLoopPeelingPattern");
}

  LogicalResult matchAndRewrite(LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "matchAndRewrite");

    SmallVector<int64_t> peeledLoops;
    if (loopOp->hasAttr(kPeeledLoopsLabel)) {
      auto attr = loopOp->getAttr(kPeeledLoopsLabel).cast<ArrayAttr>();
      peeledLoops =
          llvm::to_vector<4>(llvm::map_range(attr, [](Attribute attr) {
            return attr.cast<IntegerAttr>().getInt();
          }));
      // Check if the loop was already peeled.
      if (llvm::find(peeledLoops, idx) != peeledLoops.end()) return failure();
    }
    if (skip_partial && loopOp->hasAttr(kPartialIterationLabel))
      // No peeling of loop nests with a partial iteration.
      return failure();

    if (static_cast<int64_t>(loopOp.iterator_types().size()) <= idx)
      return failure();

    // Peel loop and canonicalize.
    LoopOp result;
    if (failed(peelAndCanonicalizeGmlStLoop(rewriter, loopOp, idx, result)))
      return failure();

    // Apply label, so that the same loop is not rewritten a second time.
    peeledLoops.push_back(idx);
    rewriter.updateRootInPlace(loopOp, [&]() {
      loopOp->setAttr(kPeeledLoopsLabel, rewriter.getI64ArrayAttr(peeledLoops));
    });
    result->setAttr(kPeeledLoopsLabel, rewriter.getI64ArrayAttr(peeledLoops));
    result->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());

    return success();
  }

  /// Index of loop to peel.
  int64_t idx;

  /// If set to true, do not peel LoopOps with a partial iteration.
  bool skip_partial;
};

class TestGmlStLoopPeelingPass
    : public TestGmlStLoopPeelingBase<TestGmlStLoopPeelingPass> {
  void runOnOperation() final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_2(mht_2_v, 261, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "runOnOperation");

    auto funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    for (unsigned idx : dims)
      patterns.add<TiledLoopPeelingPattern>(ctx, idx, skip_partial);

    (void)(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)));

    // Drop the markers.
    funcOp.walk([](LoopOp op) {
      op->removeAttr(kPeeledLoopsLabel);
      op->removeAttr(kPartialIterationLabel);
    });
  }
};

struct LinalgTilingPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      linalg::LinalgTransformationFilter f,
                      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        filter(std::move(f)),
        options(std::move(options)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "LinalgTilingPattern");
}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_4(mht_4_v, 294, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "matchAndRewrite");

    if (failed(filter.checkAndNotify(rewriter, op))) return failure();

    FailureOr<linalg::TiledLinalgOp> res =
        gml_st::tileLinalgOp(rewriter, op, options);
    if (failed(res)) return failure();

    filter.replaceLinalgTransformationFilter(rewriter, res->op);

    if (res->tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, res->tensorResults);

    return success();
  }

 private:
  linalg::LinalgTransformationFilter filter;
  linalg::LinalgTilingOptions options;
};

struct TestGmlStLoopTilingPass
    : public TestGmlStLoopTilingBase<TestGmlStLoopTilingPass> {
  TestGmlStLoopTilingPass() = default;
  TestGmlStLoopTilingPass(ArrayRef<int64_t> tileSizes,
                          ArrayRef<StringRef> distributionTypes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_5(mht_5_v, 323, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "TestGmlStLoopTilingPass");

    this->tile_sizes = tileSizes;
    this->distribution_types = llvm::to_vector<2>(llvm::map_range(
        distributionTypes, [](StringRef ref) { return ref.str(); }));
  }

  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSgml_stPStransformsPStest_passesDTcc mht_6(mht_6_v, 332, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/gml_st/transforms/test_passes.cc", "runOnOperation");

    FuncOp funcOp = getOperation();

    auto distTypes = llvm::to_vector<2>(llvm::map_range(
        distribution_types, [](std::string &str) { return StringRef(str); }));
    auto options = linalg::LinalgTilingOptions()
                       .setTileSizes(tile_sizes)
                       .setDistributionTypes(distTypes);
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    linalg::LinalgTransformationFilter f(ArrayRef<StringAttr>{},
                                         StringAttr::get(ctx, "tile"));
    patterns.add<LinalgTilingPattern>(ctx, options, f);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    funcOp.walk([](linalg::LinalgOp op) {
      op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createTestGmlStLoopPeelingPass() {
  return std::make_unique<TestGmlStLoopPeelingPass>();
}

std::unique_ptr<OperationPass<FuncOp>> createTestGmlStLoopTilingPass() {
  return std::make_unique<TestGmlStLoopTilingPass>();
}
}  // namespace gml_st
}  // namespace mlir
