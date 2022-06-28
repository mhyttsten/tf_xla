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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc() {
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

// This file implements logic for lowering MHLO dialect to Standard dialect.

#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {
#include "generated_legalize_to_standard.inc"
}  // end anonymous namespace
namespace mhlo {
namespace {

class CompareIConvert : public OpRewritePattern<mhlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "matchAndRewrite");

    auto lhs = op.lhs();
    auto rhs = op.rhs();
    auto lhs_type = lhs.getType().cast<TensorType>();
    auto rhs_type = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return failure();

    if (!lhs_type.getElementType().isSignlessInteger() ||
        !rhs_type.getElementType().isSignlessInteger())
      return failure();

    Optional<arith::CmpIPredicate> compare_predicate;
    switch (op.comparison_direction()) {
      case ComparisonDirection::EQ:
        compare_predicate = arith::CmpIPredicate::eq;
        break;
      case ComparisonDirection::NE:
        compare_predicate = arith::CmpIPredicate::ne;
        break;
      case ComparisonDirection::LT:
        compare_predicate = arith::CmpIPredicate::slt;
        break;
      case ComparisonDirection::LE:
        compare_predicate = arith::CmpIPredicate::sle;
        break;
      case ComparisonDirection::GT:
        compare_predicate = arith::CmpIPredicate::sgt;
        break;
      case ComparisonDirection::GE:
        compare_predicate = arith::CmpIPredicate::sge;
        break;
      default:
        compare_predicate = llvm::None;
    }

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, compare_predicate.getValue(),
                                               lhs, rhs);
    return success();
  }
};

class CompareFConvert : public OpRewritePattern<mhlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_1(mht_1_v, 265, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "matchAndRewrite");

    auto lhs = op.lhs();
    auto rhs = op.rhs();
    auto lhs_type = lhs.getType().cast<TensorType>();
    auto rhs_type = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return failure();

    if (!lhs_type.getElementType().isa<FloatType>() ||
        !rhs_type.getElementType().isa<FloatType>())
      return failure();

    Optional<arith::CmpFPredicate> compare_predicate;
    switch (op.comparison_direction()) {
      case ComparisonDirection::EQ:
        compare_predicate = arith::CmpFPredicate::OEQ;
        break;
      case ComparisonDirection::NE:
        compare_predicate = arith::CmpFPredicate::UNE;
        break;
      case ComparisonDirection::LT:
        compare_predicate = arith::CmpFPredicate::OLT;
        break;
      case ComparisonDirection::LE:
        compare_predicate = arith::CmpFPredicate::OLE;
        break;
      case ComparisonDirection::GT:
        compare_predicate = arith::CmpFPredicate::OGT;
        break;
      case ComparisonDirection::GE:
        compare_predicate = arith::CmpFPredicate::OGE;
        break;
      default:
        compare_predicate = llvm::None;
    }

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, compare_predicate.getValue(),
                                               lhs, rhs);
    return success();
  }
};

// Replace IotaOp with an integer constant. A ConvertOp is added to
// convert the integer constant to iota result type. For complex types, the real
// part is replaced with the generated constant and the imaginary part is
// replaced with zero tensor.
class ConvertIotaOp : public OpRewritePattern<mhlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::IotaOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_2(mht_2_v, 322, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "matchAndRewrite");

    auto output_type = op.getType().cast<ShapedType>();
    auto output_size = output_type.getNumElements();
    auto dimension = op.iota_dimension();
    auto max_dim_size = output_type.getDimSize(dimension);

    auto element_type = output_type.getElementType();
    int bitwidth;

    auto complex_ty = element_type.dyn_cast<ComplexType>();
    Type int_or_float_ty = element_type;
    if (complex_ty) int_or_float_ty = complex_ty.getElementType();

    bitwidth = int_or_float_ty.getIntOrFloatBitWidth();
    llvm::SmallVector<APInt, 10> values;
    values.reserve(output_size);

    int64_t increase_stride = output_size;
    for (uint64_t i = 0; i <= dimension; i++) {
      increase_stride /= output_type.getDimSize(i);
    }

    int64_t current_value = 0;
    for (int i = 0; i < output_size; i++) {
      int64_t value = (current_value / increase_stride) % max_dim_size;
      values.push_back(APInt(bitwidth, value));
      ++current_value;
    }

    auto int_shape_type = RankedTensorType::get(
        output_type.getShape(),
        IntegerType::get(rewriter.getContext(), bitwidth));
    auto loc = op.getLoc();
    auto integer_const = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(int_shape_type, values));

    auto int_or_float_shape_ty =
        RankedTensorType::get(output_type.getShape(), int_or_float_ty);

    auto iota_const =
        rewriter.create<ConvertOp>(loc, int_or_float_shape_ty, integer_const);

    // For int/float types we are done, replace op and return.
    if (!complex_ty) {
      rewriter.replaceOp(op, iota_const.getResult());
      return success();
    }

    // For complex types, generate a constant tensor of zeroes for the imaginary
    // part and use iota_const for real part.
    auto zeroes = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(int_shape_type, APInt(bitwidth, 0)));
    auto imag_zeroes =
        rewriter.create<ConvertOp>(loc, int_or_float_shape_ty, zeroes);
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, iota_const, imag_zeroes);
    return success();
  }
};

}  // end anonymous namespace

namespace {
struct LegalizeToStandardPass
    : public LegalizeToStandardPassBase<LegalizeToStandardPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_3(mht_3_v, 389, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "getDependentDialects");

    registry.insert<arith::ArithmeticDialect, math::MathDialect,
                    func::FuncDialect>();
  }

  /// Perform the lowering to Standard dialect.
  void runOnOperation() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeToStdPass() {
  return std::make_unique<LegalizeToStandardPass>();
}

void PopulateMhloToStdPatterns(RewritePatternSet *patterns,
                               mlir::MLIRContext *ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_4(mht_4_v, 408, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "PopulateMhloToStdPatterns");

  mlir::populateWithGenerated(*patterns);
  patterns->add<CompareFConvert, CompareIConvert, ConvertIotaOp>(ctx);
}

/// Perform the lowering to standard dialect.
void LegalizeToStandardPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_to_standardDTcc mht_5(mht_5_v, 417, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_to_standard.cc", "LegalizeToStandardPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  mlir::mhlo::PopulateMhloToStdPatterns(&patterns, &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

}  // end namespace mhlo
}  // end namespace mlir
