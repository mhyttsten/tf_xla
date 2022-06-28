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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc() {
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

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::makeArrayRef;
using mlir::BlockAndValueMapping;
using mlir::dyn_cast;
using mlir::failure;
using mlir::FailureOr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::InitTensorOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingLoopType;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::PadOp;

// Tiles a GenericOp that models a 2D row or column reduction.
struct RowOrColumnReductionTilingPattern : public OpRewritePattern<GenericOp> {
  RowOrColumnReductionTilingPattern(const LinalgTilingOptions &options,
                                    const LinalgTransformationFilter &filter,
                                    MLIRContext *context,
                                    mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        filter(filter),
        options(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_0(mht_0_v, 245, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "RowOrColumnReductionTilingPattern");
}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "matchAndRewrite");

    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    if (linalg_op.getNumOutputs() != 1) return failure();
    if (linalg_op.getNumLoops() != 2) return failure();

    auto tiled_op = mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_op)) return failure();

    tiled_op->loops.front()->walk([&](LinalgOp tOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tOp);
    });

    rewriter.replaceOp(linalg_op, tiled_op->tensorResults);
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Rewrites a 1D reduction for vectorization. Matches `linalg.generic` that
// combines elements of tensor<?xELEM_TYPE> into tensor<ELEM_TYPE> and then
// creates a loop to reduce tensor<?xELEM_TYPE> -> tensor<VECTOR_SIZExELEM_TYPE>
// and an additional `linalg.generic` that reduces tensor<VECTOR_SIZExELEM_TYPE>
// to tensor<ELEM_TYPE>.
//
// Example:
//
// %sum = linalg.generic {
//   indexing_maps = [affine_map<(d0) -> (d0)>,
//                    affine_map<(d0) -> ()>],
//   iterator_types = ["reduction"]}
//   ins(%input : tensor<?xf32>)
//   outs(%fill : tensor<f32>) {
// ^bb0(%in: f32, %out: f32):
//   %add = arith.addf %in, %out : f32
//   linalg.yield %add : f32
// } -> tensor<f32>
//
// will be rewritten as
//
// %vector_result = linalg.tiled_loop (%i)
//     = (%c0) to (%INPUT_SIZE) step (%vector_size)
//     ins (%input_ = %input: tensor<?xf32>)
//     outs (%tmp_result_ = %tmp_result: tensor<VECTOR_SIZExf32>)
//     iterators["reduction"] {
//   %tile = tensor.extract_slice %arg2[%i] [%TILE_SIZE] [1]
//     : tensor<?xf32> to tensor<?xf32>
//   %tile_pad = linalg.pad_tensor %tile
//     : tensor<?xf32> to tensor<VECTOR_SIZExf32>
//   %tile_reshape = tensor.expand_shape %tile_pad [[0, 1]]
//     : tensor<VECTOR_SIZExf32> into tensor<1xVECTOR_SIZExf32>
//   %combine = linalg.generic ins(%tile_reshape : tensor<1xVECTOR_SIZExf32>)
//     outs(%tmp_result_ : tensor<VECTOR_SIZExf32>) -> tensor<VECTOR_SIZExf32>
//   linalg.yield %combine : tensor<VECTOR_SIZExf32>
//   }
// %result = linalg.generic ins(%vector_result : tensor<VECTOR_SIZExf32>)
//   outs(%fill : tensor<f32>) -> tensor<f32>
//
// This is necessary to push horizontal reduction to the later stage.
struct OneDimReductionTilingPattern : public OpRewritePattern<GenericOp> {
  OneDimReductionTilingPattern(int64_t vector_size, int64_t tile_size,
                               const LinalgTransformationFilter &filter,
                               mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        filter(filter),
        vector_size(vector_size),
        tile_size(tile_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_2(mht_2_v, 324, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "OneDimReductionTilingPattern");
}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_3(mht_3_v, 330, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "matchAndRewrite");

    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();
    if (linalg_op.getNumOutputs() != 1) return failure();

    // Check if all inputs have a 1D identity map.
    if (linalg_op.getNumLoops() != 1) return failure();
    auto indexing_maps = linalg_op.getIndexingMaps();
    for (auto affine_map : makeArrayRef(indexing_maps).drop_back()) {
      if (!affine_map.isIdentity()) return failure();
    }

    Location loc = linalg_op.getLoc();
    Value input = linalg_op.getInputOperand(0)->get();
    // All inputs have the same size because of identity maps for indexing.
    SmallVector<Value> inputs = linalg_op.inputs();
    Value input_size = rewriter.create<mlir::tensor::DimOp>(loc, input, 0);

    auto fill_op = linalg_op.outputs().front().getDefiningOp<FillOp>();
    auto init_op = fill_op.output().getDefiningOp<InitTensorOp>();

    auto neutral_value = fill_op.value();
    auto element_type = init_op.getType().getElementType();

    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value tile_size_value = rewriter.create<ConstantIndexOp>(loc, tile_size);
    Value new_init = rewriter.create<InitTensorOp>(loc, ValueRange{},
                                                   vector_size, element_type);
    Value new_fill =
        rewriter.create<FillOp>(loc, fill_op.value(), new_init).result();

    GenericOp tiled_reduction;
    auto tiled_loop_op = rewriter.create<LoopOp>(
        loc, makeArrayRef(zero), makeArrayRef(input_size),
        makeArrayRef(tile_size_value), inputs, makeArrayRef(new_fill),
        rewriter.getStrArrayAttr(mlir::getReductionIteratorTypeName()),
        [&](OpBuilder &b, Location nested_loc, ValueRange ivs,
            ValueRange inputs, ValueRange outputs) {
          SmallVector<Value, 2> reshaped_tiled_inputs =
              TileAndReshapeInputTensors(b, nested_loc, ivs, inputs,
                                         neutral_value, input_size,
                                         tile_size_value);
          // Create `linalg.generic` to combine
          // `tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE> input with
          // the `tensor<VECTOR_SIZExELEM_TYPE>` output.
          SmallVector<mlir::StringRef, 2> iter_types{
              mlir::getReductionIteratorTypeName(),
              mlir::getParallelIteratorTypeName()};
          SmallVector<mlir::AffineMap, 2> indexing_maps(
              inputs.size(), rewriter.getMultiDimIdentityMap(2));
          indexing_maps.push_back(
              mlir::AffineMap::get(2, 0, b.getAffineDimExpr(1)));
          tiled_reduction = b.create<GenericOp>(
              nested_loc, outputs[0].getType(), reshaped_tiled_inputs,
              makeArrayRef({outputs[0]}), indexing_maps, iter_types,
              /*bodyBuild=*/nullptr);
          mlir::Region &region = tiled_reduction.region();
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.cloneRegionBefore(linalg_op.region(), region, region.end());
          b.create<mlir::gml_st::YieldOp>(nested_loc,
                                          tiled_reduction.getResult(0));
        });
    // Create `linalg.generic` to reduce
    // tensor<VECTOR_SIZExELEM_TYPE>->tensor<ELEM_TYPE>.
    auto final_reduction_or =
        ReduceVectorIntoOutput(rewriter, linalg_op, tiled_loop_op.getResult(0));
    if (failed(final_reduction_or)) return failure();
    auto final_reduction = final_reduction_or.getValue();
    rewriter.replaceOp(linalg_op, final_reduction->getResults());

    tiled_loop_op->walk([&](GenericOp op) {
      filter.replaceLinalgTransformationFilter(rewriter, op);
      filter.replaceLinalgTransformationFilter(rewriter, final_reduction);
    });
    return success();
  }

  // Tiles, pads and reshapes every input argument of type tensor<?xELEM_TYPE>
  // into tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
  SmallVector<Value, 2> TileAndReshapeInputTensors(
      OpBuilder &b, Location nested_loc, ValueRange ivs, ValueRange inputs,
      Value neutral_value, Value input_size, Value tile_size_value) const {
    SmallVector<Value, 2> reshaped_tiled_inputs;

    SmallVector<mlir::ReassociationIndices> indices = {{0, 1}};
    auto identity_1d_map = b.getMultiDimIdentityMap(1);
    auto iv = ivs.front();

    auto tile_sizes = mlir::linalg::computeTileSizes(
        b, nested_loc, ivs, tile_size_value, input_size);
    for (auto input : inputs) {
      // Extract slice of input.
      Value slice = mlir::linalg::makeTiledShape(
          b, nested_loc, input, tile_size_value, identity_1d_map, iv,
          input_size, tile_sizes);
      auto element_type = slice.getType().cast<ShapedType>().getElementType();

      // Pad input tile.
      Value pad = mlir::tensor::createPadHighOp(
          RankedTensorType::get({tile_size}, element_type), slice,
          neutral_value, false, nested_loc, b);

      // Reshape input tile to
      // tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
      Value expand_shape = b.create<ExpandShapeOp>(
          nested_loc,
          RankedTensorType::get({tile_size / vector_size, vector_size},
                                element_type),
          pad, indices);
      reshaped_tiled_inputs.push_back(expand_shape);
    }
    return reshaped_tiled_inputs;
  }

  // Creates `linalg.generic` to reduce
  // tensor<VECTOR_SIZExELEM_TYPE>->tensor<ELEM_TYPE>. To perform that we match
  // the combiner in the original "untiled" linalg_op.
  FailureOr<GenericOp> ReduceVectorIntoOutput(PatternRewriter &rewriter,
                                              LinalgOp linalg_op,
                                              Value partial_result) const {
    SmallVector<mlir::StringRef, 3> reduction_iter_type(
        1, mlir::getReductionIteratorTypeName());
    auto map = mlir::AffineMap::get(1, 0, llvm::None, rewriter.getContext());

    auto combiner_or = DetectCombiner(linalg_op);
    if (failed(combiner_or)) return failure();
    Operation *combiner = combiner_or.getValue();

    auto accumulator = rewriter.create<GenericOp>(
        linalg_op.getLoc(), linalg_op->getResultTypes(),
        makeArrayRef(partial_result),
        makeArrayRef(linalg_op.getOutputOperand(0)->get()),
        makeArrayRef({rewriter.getMultiDimIdentityMap(1), map}),
        reduction_iter_type,
        [&](OpBuilder &b, Location nested_loc, ValueRange args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_4(mht_4_v, 466, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "lambda");

          BlockAndValueMapping bvm;
          bvm.map(combiner->getOperands(), args);
          Value result_val = b.clone(*combiner, bvm)->getResult(0);
          b.create<mlir::linalg::YieldOp>(nested_loc, result_val);
        });
    return accumulator;
  }

 private:
  LinalgTransformationFilter filter;
  int64_t vector_size;
  int64_t tile_size;
};

// Match 1D or 2D reduction.
bool isCanonicalizedReduction(Operation *op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_5(mht_5_v, 485, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "isCanonicalizedReduction");

  auto reduction = mlir::dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumLoops() > 2) return false;
  return reduction.getNumReductionLoops() == 1;
}

struct TileReductionPass : public TileReductionBase<TileReductionPass> {
  TileReductionPass() = default;
  TileReductionPass(int64_t vector_size, int64_t reduction_1d_tile,
                    llvm::ArrayRef<int64_t> reduction_2d_tiles) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_6(mht_6_v, 499, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "TileReductionPass");

    reduction_vector_size = vector_size;
    reduction_1d_tile_size = reduction_1d_tile;
    reduction_2d_tile_sizes = reduction_2d_tiles;
  }
  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_tile_reductionDTcc mht_7(mht_7_v, 507, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_tile_reduction.cc", "runOnOperation");

    auto func = getOperation();
    auto context = func.getContext();

    auto filter = LinalgTransformationFilter(
                      llvm::None, {mlir::StringAttr::get(context, "tiled")})
                      .addFilter([](Operation *op) {
                        return success(isCanonicalizedReduction(op));
                      });
    assert(reduction_1d_tile_size % reduction_vector_size == 0 &&
           "Tile size for 1D reduction should be a multiple of vector size");
    auto patterns =
        mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.add<OneDimReductionTilingPattern>(reduction_vector_size,
                                               reduction_1d_tile_size, filter,
                                               patterns.getContext());

    assert(reduction_2d_tile_sizes.size() == 2 &&
           "Tiling sizes for 2D reductions should have two elements");
    patterns.add<RowOrColumnReductionTilingPattern>(
        LinalgTilingOptions{}.setTileSizes(reduction_2d_tile_sizes), filter,
        patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileReductionPass() {
  return std::make_unique<TileReductionPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileReductionPass(int64_t reduction_vector_size,
                        int64_t reduction_1d_tile_size,
                        llvm::ArrayRef<int64_t> reduction_2d_tile_sizes) {
  return std::make_unique<TileReductionPass>(
      reduction_vector_size, reduction_1d_tile_size, reduction_2d_tile_sizes);
}

}  // namespace tensorflow
