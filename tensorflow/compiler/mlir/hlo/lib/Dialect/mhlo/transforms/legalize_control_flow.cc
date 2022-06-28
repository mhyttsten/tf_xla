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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc() {
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

// This file implements logic for lowering MHLO dialect to SCF dialect.
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

// All transformations in this file take mhlo blocks which end with
// mhlo::ReturnOp and lower to SCF ops which end with scf::YieldOp. Inline an
// entire block with the only change being return -> yield.
void inlineMhloRegionIntoSCFRegion(PatternRewriter& rewriter, Region& mhlo,
                                   Region& scf) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "inlineMhloRegionIntoSCFRegion");

  // Remove an existing block, then move the region over.
  if (!scf.empty()) rewriter.eraseBlock(&scf.back());
  rewriter.inlineRegionBefore(mhlo, scf, scf.end());
  // Fix up the terminator.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  auto* terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                            terminator->getOperands());
}

// mhlo ops need inputs to be tensors, but scalar values can be a scalar tensor
// or a 1 element tensor. To handle this, collapse shape before extracting the
// scalar value when necessary.
Value extractTensorValue(OpBuilder& b, Value tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "extractTensorValue");

  auto loc = tensor.getLoc();
  if (tensor.getType().cast<TensorType>().hasRank() &&
      tensor.getType().cast<TensorType>().getRank() != 0) {
    tensor = b.create<tensor::CollapseShapeOp>(
        loc, tensor, SmallVector<ReassociationIndices>());
  }
  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

// Create a memref descriptor given a pointer and memref type information.
struct WhileOpPattern : public OpConversionPattern<mhlo::WhileOp> {
  using OpConversionPattern<WhileOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::WhileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "matchAndRewrite");

    auto loc = op.getLoc();

    auto new_while_op = rewriter.create<scf::WhileOp>(loc, op.getResultTypes(),
                                                      adaptor.getOperands());

    // Inline while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.inlineRegionBefore(op.cond(), new_while_op.getBefore(),
                                new_while_op.getBefore().end());
    auto condition_return =
        cast<mhlo::ReturnOp>(new_while_op.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&new_while_op.getBefore().front());
    Value i1 = extractTensorValue(rewriter, condition_return->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        condition_return, i1, new_while_op.getBeforeArguments());

    // Inline while body, and only replace the mhlo.return with an scf.yield.
    inlineMhloRegionIntoSCFRegion(rewriter, op.body(), new_while_op.getAfter());

    rewriter.replaceOp(op, new_while_op.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct IfOpPattern : public OpConversionPattern<mhlo::IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "matchAndRewrite");

    auto scf_if =
        rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                   extractTensorValue(rewriter, adaptor.pred()),
                                   /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(rewriter, op.true_branch(),
                                  scf_if.getThenRegion());
    inlineMhloRegionIntoSCFRegion(rewriter, op.false_branch(),
                                  scf_if.getElseRegion());
    rewriter.replaceOp(op, scf_if.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct CaseOpPattern : public OpConversionPattern<mhlo::CaseOp> {
  using OpConversionPattern<CaseOp>::OpConversionPattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int current_idx, CaseOp op, OpAdaptor adaptor,
                              PatternRewriter& outer_builder) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_4(mht_4_v, 315, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "createNestedCases");

    Location loc = op.getLoc();
    Value idx_value = adaptor.index();
    auto final_idx = op.branches().size() - 2;

    // Determine if the current index matches the case index.
    auto scalar_type = idx_value.getType();
    auto const_attr = DenseElementsAttr::get(
        scalar_type,
        {outer_builder.getI32IntegerAttr(current_idx).cast<mlir::Attribute>()});
    Value current_idx_val = outer_builder.create<mhlo::ConstOp>(
        loc, idx_value.getType(), const_attr);

    auto scf_if = outer_builder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outer_builder, outer_builder.create<mhlo::CompareOp>(
                                              loc, idx_value, current_idx_val,
                                              ComparisonDirection::EQ)),
        /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(outer_builder, op.branches()[current_idx],
                                  scf_if.getThenRegion());
    int next_idx = current_idx + 1;
    // Don't recurse for the final default block.
    if (current_idx == final_idx) {
      inlineMhloRegionIntoSCFRegion(outer_builder, op.branches()[next_idx],
                                    scf_if.getElseRegion());
    } else {
      PatternRewriter::InsertionGuard guard(outer_builder);
      outer_builder.setInsertionPointToEnd(&scf_if.getElseRegion().back());
      auto inner_if = createNestedCases(next_idx, op, adaptor, outer_builder);
      outer_builder.create<scf::YieldOp>(op.getLoc(), inner_if.getResults());
    }
    return scf_if;
  }

  LogicalResult matchAndRewrite(
      mhlo::CaseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_5(mht_5_v, 355, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "matchAndRewrite");

    // Inline the op if there is only a default block.
    if (op.branches().size() == 1) {
      Block& block = op.branches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the mhlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.mergeBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
                                /*argValues=*/{});
      rewriter.replaceOp(op, results);
      return success();
    }

    // Begin recursion with case 0.
    rewriter.replaceOp(
        op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

struct LegalizeControlFlowPass
    : public LegalizeControlFlowPassBase<LegalizeControlFlowPass> {
  // Perform the lowering to MLIR control flow.
  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "runOnOperation");

    FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();

    RewritePatternSet patterns(&getContext());
    patterns.add<WhileOpPattern, IfOpPattern, CaseOpPattern>(&getContext());

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<mhlo::IfOp, mhlo::WhileOp, mhlo::CaseOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createLegalizeControlFlowPass() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_control_flowDTcc mht_7(mht_7_v, 406, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_control_flow.cc", "mlir::mhlo::createLegalizeControlFlowPass");

  return std::make_unique<LegalizeControlFlowPass>();
}
