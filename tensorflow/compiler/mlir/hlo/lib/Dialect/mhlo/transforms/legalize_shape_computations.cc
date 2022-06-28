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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc() {
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

// This file implements logic for lowering HLO/LHLO dialect to scalar shape
// operations.

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

// We assume that if one of the operands is a FromElements operation that means
// it is a shape computation.
bool OpIsShapeComputation(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "OpIsShapeComputation");

  bool found_from_elements = false;
  for (auto operand : op->getOperands()) {
    auto shaped_ty = operand.getType().template cast<ShapedType>();
    if (!shaped_ty.hasRank() || shaped_ty.getRank() > 1) return false;
    if (auto from_elements =
            operand.template getDefiningOp<tensor::FromElementsOp>()) {
      found_from_elements = true;
      continue;
    }
  }
  return found_from_elements;
}

template <typename OpTy>
class MhloElementwiseConverter : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "matchAndRewrite");

    if (!OpIsShapeComputation(op)) return failure();

    auto result_ty = op.getType().template cast<ShapedType>();

    Location loc = op.getLoc();
    SmallVector<Value> operands;
    for (int i = 0, s = result_ty.getNumElements(); i < s; i++) {
      SmallVector<Value> extracts;
      for (auto operand : op->getOperands()) {
        ShapedType operand_ty = operand.getType().template cast<ShapedType>();
        if (operand_ty.getRank() == 0) {
          Value extract =
              rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
          extracts.push_back(extract);
        } else {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          extracts.push_back(extract);
        }
      }

      Value scalar_op = mhlo::MhloOpToStdScalarOp::map<OpTy>(
          op, result_ty.getElementType(), extracts, &rewriter);
      operands.push_back(scalar_op);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty,
                                                        operands);

    return success();
  }
};

class ConcatenateConverter : public OpRewritePattern<mhlo::ConcatenateOp> {
 public:
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_2(mht_2_v, 292, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "matchAndRewrite");

    if (!OpIsShapeComputation(op)) return failure();

    Location loc = op.getLoc();
    auto result_ty = op.getType().cast<ShapedType>();
    llvm::SmallVector<Value> elements;
    elements.reserve(result_ty.getNumElements());

    for (auto operand : op->getOperands()) {
      ShapedType operand_ty = operand.getType().template cast<ShapedType>();
      if (operand_ty.getRank() == 0) {
        Value extract =
            rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
        elements.push_back(extract);
      } else {
        for (int i = 0, s = operand_ty.getNumElements(); i < s; i++) {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          elements.push_back(extract);
        }
      }
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty,
                                                        elements);
    return success();
  }
};

class GetDimSizeConverter : public OpRewritePattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpRewritePattern<mhlo::GetDimensionSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_3(mht_3_v, 329, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "matchAndRewrite");

    Location loc = op.getLoc();
    auto result_ty = op.getType();
    auto element_ty = getElementTypeOrSelf(result_ty);
    auto dim_attr = rewriter.getIndexAttr(op.dimension());
    auto dim_const = rewriter.create<arith::ConstantOp>(loc, dim_attr);

    Value dim_op = rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(),
                                                  op.operand(), dim_const);

    // Cast to the correct element type and convert to a tensor.
    Value cast = rewriter.create<arith::IndexCastOp>(loc, element_ty, dim_op);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty, cast);
    return success();
  }
};

class ReshapeConverter : public OpRewritePattern<mhlo::ReshapeOp> {
 public:
  using OpRewritePattern<mhlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_4(mht_4_v, 354, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "matchAndRewrite");

    auto operand = op.operand();
    auto shaped_ty = operand.getType().template cast<ShapedType>();
    if (!shaped_ty.hasRank() || shaped_ty.getRank() > 1) return failure();

    auto result_ty = op.getType().cast<ShapedType>();

    auto from_elements = op.operand().getDefiningOp<tensor::FromElementsOp>();
    if (!from_elements) return failure();

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, result_ty, from_elements.getOperands());
    return success();
  }
};

struct HloLegalizeShapeComputationsPass
    : public mhlo::HloLegalizeShapeComputationsPassBase<
          HloLegalizeShapeComputationsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_5(mht_5_v, 376, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "getDependentDialects");

    registry.insert<arith::ArithmeticDialect, math::MathDialect,
                    func::FuncDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_6(mht_6_v, 384, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "runOnOperation");

    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);

    auto func = getOperation();
    mhlo::populateShapeComputationPatterns(&ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mhlo {

void populateShapeComputationPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_shape_computationsDTcc mht_7(mht_7_v, 404, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_shape_computations.cc", "populateShapeComputationPatterns");

  patterns->add<MhloElementwiseConverter<mhlo::AbsOp>,
                MhloElementwiseConverter<mhlo::AddOp>,
                MhloElementwiseConverter<mhlo::AndOp>,
                MhloElementwiseConverter<mhlo::CeilOp>,
                MhloElementwiseConverter<mhlo::ConvertOp>,
                MhloElementwiseConverter<mhlo::DivOp>,
                MhloElementwiseConverter<mhlo::FloorOp>,
                MhloElementwiseConverter<mhlo::MaxOp>,
                MhloElementwiseConverter<mhlo::MinOp>,
                MhloElementwiseConverter<mhlo::MulOp>,
                MhloElementwiseConverter<mhlo::NegOp>,
                MhloElementwiseConverter<mhlo::RoundOp>,
                MhloElementwiseConverter<mhlo::RsqrtOp>,
                MhloElementwiseConverter<mhlo::SqrtOp>,
                MhloElementwiseConverter<mhlo::SubOp>, ConcatenateConverter,
                GetDimSizeConverter, ReshapeConverter>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeShapeComputationsPass() {
  return std::make_unique<HloLegalizeShapeComputationsPass>();
}

}  // namespace mhlo
}  // namespace mlir
