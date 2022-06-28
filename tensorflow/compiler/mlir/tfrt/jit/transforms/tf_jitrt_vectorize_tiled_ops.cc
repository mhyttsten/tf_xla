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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc() {
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

#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::failure;
using mlir::success;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::CodegenStrategy;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;

// The upper limit for vectorization of untiled `linalg.fill`. If a tensor has a
// static shape with more elements, then `linalg.fill` won't be vectorized. It
// is expected that such operations are tiled to get to small static shapes.
constexpr int64_t kNumElementsThreshold = 1024;

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransferReadOp vector_read,
      mlir::PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_vectorize_tiled_ops.cc", "matchAndRewrite");

    auto expand = vector_read.getSource().getDefiningOp<ExpandShapeOp>();
    if (!expand) return failure();

    auto expand_src = expand.src();
    auto expand_src_type = expand.getSrcType();
    auto expand_dst_type = expand.getResultType();
    if (expand_src_type.getRank() != 1 || expand_dst_type.getRank() != 2)
      return failure();

    auto result_type = vector_read.getType().dyn_cast<mlir::ShapedType>();
    if (!result_type || result_type.getShape() != expand_dst_type.getShape())
      return failure();

    auto zero = rewriter.create<ConstantIndexOp>(vector_read.getLoc(), 0);
    auto map = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                    vector_read.getContext());
    // TODO(pifon): Also support canonicalization in case the map is not an
    // identity.
    if (!map.isIdentity()) return failure();

    auto new_read = rewriter.create<TransferReadOp>(
        vector_read.getLoc(),
        mlir::VectorType::get(expand_src_type.getShape(),
                              expand_src_type.getElementType()),
        expand_src, mlir::ValueRange{zero}, mlir::AffineMapAttr::get(map),
        vector_read.getPadding(),
        /*mask=*/mlir::Value(), rewriter.getBoolArrayAttr({true}));
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        vector_read, vector_read.getType(), new_read);
    return success();
  }
};

struct VectorizeTiledOpsPass
    : public VectorizeTiledOpsBase<VectorizeTiledOpsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc mht_1(mht_1_v, 264, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_vectorize_tiled_ops.cc", "getDependentDialects");

    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_vectorize_tiled_opsDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_vectorize_tiled_ops.cc", "runOnOperation");

    auto funcOp = getOperation();

    // Vectorize linalg.fill and linalg.generic operations.
    mlir::OpPassManager dynamicPM("func.func");
    CodegenStrategy strategy;
    strategy.vectorize(FillOp::getOperationName(), [](mlir::Operation *op) {
      auto fill = mlir::dyn_cast<FillOp>(op);
      if (!fill) return failure();

      if (op->getParentOfType<LoopOp>()) return success();

      // Allow vectorization for static shapes with low number of elements.
      auto output_type = fill.output().getType().cast<mlir::RankedTensorType>();
      if (output_type.hasStaticShape() &&
          output_type.getNumElements() < kNumElementsThreshold)
        return success();

      return failure();
    });

    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();

    mlir::OpPassManager dynamicPM2("func.func");
    CodegenStrategy strategy2;
    strategy2.vectorize(GenericOp::getOperationName(), [](mlir::Operation *op) {
      auto generic = mlir::dyn_cast<GenericOp>(op);
      if (!generic) return failure();

      if (op->getParentOfType<LoopOp>()) return success();

      // Allow vectorization of 1D reductions.
      return success(generic.getNumLoops() == 1 &&
                     generic.getNumReductionLoops() == 1);
    });

    strategy2.configurePassPipeline(dynamicPM2, funcOp.getContext());
    if (failed(runPipeline(dynamicPM2, funcOp))) return signalPassFailure();

    // Vectorize padding.
    mlir::RewritePatternSet patterns(funcOp.getContext());
    mlir::linalg::populatePadOpVectorizationPatterns(patterns);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
        patterns);
    patterns.add<TransferReadOfOneDimExpandShape>(funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVectorizeTiledOpsPass() {
  return std::make_unique<VectorizeTiledOpsPass>();
}

}  // namespace tensorflow
