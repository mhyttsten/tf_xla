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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements the lowering for trigonometric standard ops to
// approximations.

#include <utility>

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
namespace mhlo {
namespace {

template <typename OpTy>
class ApproximateOnExtendedF32Lowering : public OpRewritePattern<OpTy> {
 public:
  explicit ApproximateOnExtendedF32Lowering(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx, /*benefit=*/100) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "ApproximateOnExtendedF32Lowering");
}

  virtual Value emitApproximation(ValueRange, Location,
                                  PatternRewriter &) const = 0;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "matchAndRewrite");

    Location loc = op.getLoc();
    auto raw_args = op.getOperation()->getOperands();

    // Supports only f16 and f32 for now.
    if (!op.getType().isF16() && !op.getType().isF32()) return failure();

    // Extend operands to f32 if needed and possible.
    SmallVector<Value, 2> f32_args;
    f32_args.reserve(raw_args.size());
    for (Value arg : raw_args) {
      // Similar to XLA, do not rewrite f64 as precision might matter.
      Type arg_ty = arg.getType();
      if (arg_ty.isF64()) return failure();

      if (arg_ty.isF16())
        arg = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), arg);

      // If we still do not have f32, fail.
      if (!arg.getType().isF32()) return failure();

      f32_args.push_back(arg);
    }

    Value result = emitApproximation(f32_args, loc, rewriter);
    assert(result.getType().isF32() && "Expect f32 intermediate result.");

    // Truncate back if needed.
    if (op.getType().isF16())
      result =
          rewriter.create<arith::TruncFOp>(loc, rewriter.getF16Type(), result);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// This approximation resembles Eigen and realizes a constant approximation for
// the +/-1 limits on top.
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/MathFunctionsImpl.h
class ApproximateTanhLowering
    : public ApproximateOnExtendedF32Lowering<math::TanhOp> {
 public:
  explicit ApproximateTanhLowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<math::TanhOp>(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "ApproximateTanhLowering");
}

  // Emits the fast tanh approximation that is also used by XLA.
  Value emitApproximation(ValueRange args, Location loc,
                          PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "emitApproximation");

    Value input = args.front();
    assert(input.getType().isF32());
    static constexpr std::array<float, 7> numerator_coeffs{
        -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
        5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
        4.89352455891786e-03f};
    static constexpr std::array<float, 4> denominator_coeffs{
        1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
        4.89352518554385e-03f};

    // Materialize polynomial approximation.
    Value input_squared = rewriter.create<arith::MulFOp>(loc, input, input);
    Value numerator = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(numerator_coeffs[0]));
    for (int i = 1; i < numerator_coeffs.size(); i++) {
      numerator = rewriter.create<arith::AddFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, input_squared, numerator),
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF32FloatAttr(numerator_coeffs[i])));
    }
    numerator = rewriter.create<arith::MulFOp>(loc, input, numerator);
    Value denominator = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(denominator_coeffs[0]));
    for (int i = 1; i < denominator_coeffs.size(); i++) {
      denominator = rewriter.create<arith::AddFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, input_squared, denominator),
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF32FloatAttr(denominator_coeffs[i])));
    }
    Value approx = rewriter.create<arith::DivFOp>(loc, numerator, denominator);

    // For small values of |x|, we can approximate tanh(x) = x. For extremely
    // small values of x (|x| < 1e-37), the other approximation would evaluate
    // tanh(x) = 0.
    constexpr float kUseIdentityApprox = 0.0004;
    Value abs_input = rewriter.create<math::AbsOp>(loc, input);
    Value use_identity_approx = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, abs_input,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(kUseIdentityApprox)));
    approx = rewriter.create<arith::SelectOp>(loc, use_identity_approx, input,
                                              approx);

    // For very small/large values, use a constant approximation -1/1.
    Value too_large_input = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UGT, input,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(7.90531110763549805f)));
    Value too_small_input = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ULT, input,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(-7.90531110763549805f)));
    Value input_is_nan = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNE, input, input);
    approx = rewriter.create<arith::SelectOp>(
        loc, too_large_input,
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0)),
        approx);
    approx = rewriter.create<arith::SelectOp>(
        loc, too_small_input,
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(-1.0)),
        approx);
    approx = rewriter.create<arith::SelectOp>(loc, input_is_nan, input, approx);

    return approx;
  }
};

struct LegalizeTrigonometricToApproximationPass
    : public LegalizeTanhToApproximationPassBase<
          LegalizeTrigonometricToApproximationPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnOperation() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_4(mht_4_v, 347, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    PopulateTrigonometricToApproximationPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeTrigonometricToApproximationPass() {
  return std::make_unique<LegalizeTrigonometricToApproximationPass>();
}

void PopulateTrigonometricToApproximationPatterns(mlir::MLIRContext *context,
                                                  RewritePatternSet *patterns) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_trigonometric_to_approximationDTcc mht_5(mht_5_v, 368, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc", "PopulateTrigonometricToApproximationPatterns");

  // clang-format off
  patterns->add<ApproximateTanhLowering>(context);
  // clang-format on
}

}  // namespace mhlo
}  // namespace mlir
