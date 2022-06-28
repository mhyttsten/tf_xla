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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc() {
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

#include <cstdint>
#include <iterator>
#include <memory>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"
#include "tensorflow/compiler/mlir/tfr/passes/passes.h"

//===----------------------------------------------------------------------===//
// Canonicalization patterns for the scf.for and scf.if ops. They are used to
// optimize the control flow in the tfr function. Technically, both patterns
// should be upstreamed to be part of the op definition.
// TODO(fengliuai): sync with the llvm upstream for both patterns.
//
namespace mlir {
namespace TFR {

namespace {

class UnrollSCFForOp : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/tfr/passes/canonicalize.cc", "matchAndRewrite");

    Location loc = for_op.getLoc();
    APInt lower_bound, upper_bound, step;
    if (!matchPattern(for_op.getLowerBound(), m_ConstantInt(&lower_bound)) ||
        !matchPattern(for_op.getUpperBound(), m_ConstantInt(&upper_bound)) ||
        !matchPattern(for_op.getStep(), m_ConstantInt(&step))) {
      return failure();
    }
    uint64_t trip_count = (upper_bound - lower_bound).sdiv(step).getZExtValue();
    if (trip_count <= 0) return failure();

    // TODO(fengliuai): use loopUnrollByFactor once the iter_arg is supported

    Block *single_block = for_op.getBody();
    BlockAndValueMapping mapping;
    Value iv = for_op.getInductionVar();
    for (auto iter_op :
         llvm::zip(for_op.getRegionIterArgs(), for_op.getInitArgs())) {
      mapping.map(std::get<0>(iter_op), std::get<1>(iter_op));
    }
    mapping.map(iv, for_op.getLowerBound());
    for (auto i = 0; i < trip_count; ++i) {
      if (!iv.use_empty()) {
        // iv' = iv + step * i;
        Value iter = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value step_cst =
            rewriter.create<arith::ConstantIndexOp>(loc, step.getSExtValue());
        Value stride = rewriter.create<arith::MulIOp>(loc, step_cst, iter);
        Value iv_unroll =
            rewriter.create<arith::AddIOp>(loc, mapping.lookup(iv), stride);
        mapping.map(iv, iv_unroll);
      }

      Operation *terminator_op;
      for (auto it = single_block->begin(); it != single_block->end(); ++it) {
        terminator_op = rewriter.clone(*it, mapping);
      }
      // Map the block arguments to the yield results.
      for (auto iter_op : llvm::zip(for_op.getRegionIterArgs(),
                                    terminator_op->getOperands())) {
        mapping.map(std::get<0>(iter_op), std::get<1>(iter_op));
      }
      rewriter.eraseOp(terminator_op);
    }
    SmallVector<Value, 4> returned;
    for (Value arg : for_op.getRegionIterArgs()) {
      returned.push_back(mapping.lookup(arg));
    }
    rewriter.replaceOp(for_op, returned);
    return success();
  }
};

// TODO(fengliuai): up stream this pattern.
class SimplifySCFIfOp : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(scf::IfOp if_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc mht_1(mht_1_v, 285, "", "./tensorflow/compiler/mlir/tfr/passes/canonicalize.cc", "matchAndRewrite");

    // Then branch
    if (matchPattern(if_op.getCondition(), m_NonZero())) {
      return InlineRegion(if_op.getLoc(), rewriter, if_op,
                          &if_op.getThenRegion());
    }

    // Else branch
    if (matchPattern(if_op.getCondition(), m_Zero())) {
      if (if_op.getElseRegion().empty()) {
        // Remove the op
        rewriter.eraseOp(if_op);
        return success();
      } else {
        return InlineRegion(if_op.getLoc(), rewriter, if_op,
                            &if_op.getElseRegion());
      }
    }

    // Not a constant condition
    return failure();
  }

 private:
  LogicalResult InlineRegion(Location loc, PatternRewriter &rewriter,
                             Operation *inline_point, Region *region) const;
};

LogicalResult SimplifySCFIfOp::InlineRegion(Location loc,
                                            PatternRewriter &rewriter,
                                            Operation *inline_point,
                                            Region *region) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc mht_2(mht_2_v, 319, "", "./tensorflow/compiler/mlir/tfr/passes/canonicalize.cc", "SimplifySCFIfOp::InlineRegion");

  InlinerInterface interface(loc.getContext());
  if (failed(inlineRegion(interface, region, inline_point, {},
                          inline_point->getResults(), loc,
                          /*shouldCloneInlinedRegion=*/true))) {
    return failure();
  }

  // If the inlining was successful then erase the scf.if op.
  rewriter.eraseOp(inline_point);
  return success();
}

}  // namespace

void populateCanonicalizationPatterns(FuncOp func,
                                      RewritePatternSet &patterns) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSpassesPScanonicalizeDTcc mht_3(mht_3_v, 338, "", "./tensorflow/compiler/mlir/tfr/passes/canonicalize.cc", "populateCanonicalizationPatterns");

  MLIRContext *context = func.getContext();
  mlir::Dialect *tf = context->getLoadedDialect<mlir::TF::TensorFlowDialect>();
  // Load all official canonicalization patterns. Here we skip the
  // canonicalization of the ops in the tf dialect, because they couldn't
  // propagate the attributes correctly. These optimization will be played by
  // bridge.
  func->walk([&](Operation *op) {
    if (op->getDialect() != tf) {
      op->getRegisteredInfo()->getCanonicalizationPatterns(patterns, context);
    }
  });
  patterns.add<UnrollSCFForOp, SimplifySCFIfOp>(context);
}

}  // namespace TFR
}  // namespace mlir
