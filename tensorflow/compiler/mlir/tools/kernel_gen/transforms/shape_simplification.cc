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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc() {
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

// This file contains the patterns to simplify shape ops that were deemed not
// suitable for shape op canonicalization in MLIR Core.

#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

using shape::BroadcastOp;
using shape::ConstShapeOp;
using shape::ShapeOfOp;

// Try to remove operands from broadcasts that don't contribute to the final
// result.
struct BroadcastRemoveSubsumedOperandsPattern
    : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/shape_simplification.cc", "matchAndRewrite");

    // First collect the static components when joining all shapes. The
    // resulting vector contains a static dimension if any operand has a static
    // non-1 dimension in that position. The remaining dimensions are set to
    // dynamic size.
    SmallVector<int64_t> known_extents;
    SmallVector<SmallVector<int64_t, 4>, 4> operand_extents;
    for (Value shape : op.getShapes()) {
      auto &extents = operand_extents.emplace_back();
      if (failed(shape::getShapeVec(shape, extents))) return failure();

      // Prepend dynamic dims if sizes don't match.
      if (extents.size() > known_extents.size()) {
        known_extents.insert(known_extents.begin(),
                             extents.size() - known_extents.size(),
                             ShapedType::kDynamicSize);
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        if (extent != ShapedType::kDynamicSize && extent != 1) {
          int64_t &known_extent = known_extents[known_extents.size() - i - 1];
          // A dynamic dimension is subsumed by a static one, but bail out for
          // known conflicting shapes.
          if (known_extent != extent &&
              known_extent != ShapedType::kDynamicSize)
            return failure();
          known_extent = extent;
        }
      }
    }

    // If we've figured out all shapes to be constants we're done.
    if (!llvm::is_contained(known_extents, ShapedType::kDynamicSize)) {
      rewriter.replaceOpWithNewOp<ConstShapeOp>(
          op, op->getResultTypes(), rewriter.getIndexTensorAttr(known_extents));
      return success();
    }

    // If only some dimensions are known see if any of the operands can be
    // removed without affecting the result.
    SmallVector<Value, 4> filtered_operands;
    for (auto tuple : llvm::zip(op.getShapes(), operand_extents)) {
      Value shape = std::get<0>(tuple);
      auto &extents = std::get<1>(tuple);

      // An operand can't be dead if it's the only operand of the maximum rank.
      // Removing it would reduce the rank of the output.
      if (llvm::count_if(operand_extents, [&](ArrayRef<int64_t> op) {
            return op.size() >= extents.size();
          }) <= 1) {
        filtered_operands.push_back(shape);
        continue;
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        // A dimension of an operand can be subsumed if it's
        //   - a 1 dimension. All other operands will have 1 dims or better.
        if (extent == 1) continue;

        //   - a dynamic dim but the result is known to be constant.
        int64_t known_extent = known_extents[known_extents.size() - i - 1];
        assert(known_extent != 1);
        if (known_extent != ShapedType::kDynamicSize &&
            extent == ShapedType::kDynamicSize)
          continue;

        //   - a constant non-1 dimension equal to the "known" dim.
        // In this case we also have to check whether this operand is the only
        // contributor of that constant.
        if (known_extent != ShapedType::kDynamicSize &&
            extent == known_extent &&
            llvm::count_if(
                operand_extents, [&](ArrayRef<int64_t> operand_shape) {
                  return i < operand_shape.size() &&
                         operand_shape[operand_shape.size() - i - 1] ==
                             known_extent;
                }) > 1)
          continue;

        filtered_operands.push_back(shape);
        break;
      }
    }
    if (filtered_operands.size() != op.getShapes().size()) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op->getResultTypes(),
                                               filtered_operands);
      return success();
    }
    return failure();
  }
};

// Convert cases like:
// ```
//  %1 = shape.shape_of %arg0 : tensor<?x?x?xf64> -> tensor<3xindex>
//  %2 = shape.shape_of %arg1 : tensor<?x?x1xf64> -> tensor<3xindex>
//  %3 = shape.broadcast %1, %2 : tensor<3xindex>, tensor<3xindex>
//                                -> tensor<3xindex>
//  %result = tensor.extract %3[%c2] : tensor<3xindex>
// ```
// to
//
// ```
//  %result = tensor.dim %arg0[%c2] : tensor<?x?x2048xf64>
// ```
struct ExtractFromBroadcastedTensorCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc mht_1(mht_1_v, 331, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/shape_simplification.cc", "matchAndRewrite");

    auto broadcast_op = op.tensor().getDefiningOp<BroadcastOp>();
    if (!broadcast_op) return failure();

    // Confirm that there is a constant index. This is required, so we can
    // confirm the DimOp's input will define the resulting broadcasted shape in
    // that dimension.
    auto index = op.indices().front().getDefiningOp<arith::ConstantIndexOp>();
    if (!index) return failure();
    auto idx = index.value();

    // Iterate through the operands with 3 considerations in this order:
    // 1. If a static, non-1 dimension is seen, we know this to be the
    // broadcasted result
    // 2. If a single dynamic dimension is seen, we know this to be the
    // broadcasted result (with a possibly 1 or non-1 result)
    // 3. If no dynamic dimensions and no non-1 static dimensions are seen, we
    // know the result to be 1
    //
    // Iterate through all operands, keeping track of dynamic dimensions and
    // returning immediately if a non-1 static dimension is seen.
    ShapeOfOp dynamic_shape;
    int64_t num_dynamic = 0;
    for (auto shape : broadcast_op.getShapes()) {
      auto shape_of_op = shape.getDefiningOp<ShapeOfOp>();
      if (!shape_of_op) return failure();
      auto shaped_type =
          shape_of_op->getOperandTypes().front().cast<ShapedType>();

      // Abort on the existence of unranked shapes as they require more logic.
      if (!shaped_type.hasRank()) return failure();
      if (shaped_type.getRank() <= idx) continue;

      // Only consider dynamic dimensions after the loop because any non-1
      // static dimension takes precedence.
      if (shaped_type.isDynamicDim(idx)) {
        dynamic_shape = shape_of_op;
        num_dynamic++;
        continue;
      }

      if (shaped_type.getDimSize(idx) == 1) continue;

      // Return as soon as we see a non-1 static dim.
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, shaped_type.getDimSize(idx));
      return success();
    }
    if (num_dynamic > 1) return failure();

    // Replace with the single dynamic dimension or 1.
    if (dynamic_shape) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, dynamic_shape.getArg(),
                                                 index);
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 1);
    }
    return success();
  }
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct ShapeSimplification
    : public ShapeSimplificationBase<ShapeSimplification> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc mht_2(mht_2_v, 400, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/shape_simplification.cc", "getDependentDialects");

    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSshape_simplificationDTcc mht_3(mht_3_v, 411, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/shape_simplification.cc", "runOnOperation");

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());

    for (auto op : context->getRegisteredOperations()) {
      if (isa<shape::ShapeDialect, mhlo::MhloDialect>(op.getDialect()))
        op.getCanonicalizationPatterns(patterns, context);
    }

    patterns.add<BroadcastRemoveSubsumedOperandsPattern,
                 ExtractFromBroadcastedTensorCanonicalizationPattern>(context);

    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateShapeSimplification() {
  return std::make_unique<ShapeSimplification>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
