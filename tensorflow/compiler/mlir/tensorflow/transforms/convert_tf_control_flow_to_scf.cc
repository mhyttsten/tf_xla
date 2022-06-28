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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc() {
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

#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {

namespace {

/// Move the ops of `source_block` into `destination_block`, keeping the later's
/// block arguments' type as `block_arguments_type`.
static void moveBlock(Block* source_block, Block* destination_block,
                      TypeRange block_arguments_type,
                      PatternRewriter& rewriter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "moveBlock");

  // If `destination_block` isn't empty, erase its terminator to ensure that it
  // never contains two terminator-like ops after merging.
  if (!destination_block->empty())
    rewriter.eraseOp(destination_block->getTerminator());

  destination_block->addArguments(
      block_arguments_type,
      SmallVector<Location>(block_arguments_type.size(),
                            source_block->getParent()->getLoc()));
  rewriter.mergeBlocks(source_block, destination_block,
                       destination_block->getArguments());
}

/// Convert the `tf.IfRegion` op to the `scf.if` op.
class ConvertIfRegionOp : public OpRewritePattern<IfRegionOp> {
 public:
  using OpRewritePattern<IfRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfRegionOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "matchAndRewrite");

    // Creates the `then` or `else` region of the `scf.if` op. Note that
    // `tf_then_or_else_region` is the `then` or `else` region of the
    // `tf.IfRegion` op and `scf_then_or_else_region` is the `then` or `else`
    // region of the new `scf.if` op. Further, `tf_if_region_return_type` is the
    // list of return types of the `tf.IfRegion` op.
    auto createScfThenOrElse = [](Region& tf_then_or_else_region,
                                  Region& scf_then_or_else_region,
                                  TypeRange tf_if_region_return_type,
                                  PatternRewriter& rewriter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "lambda");

      // Move the first block of `tf_then_or_else_region` into the first block
      // of `scf_then_or_else_region` and do not add any arguments to the block.
      moveBlock(&tf_then_or_else_region.front(),
                &scf_then_or_else_region.front(), TypeRange(), rewriter);

      // Replace the current terminator (a `tf.Yield` op) with an `scf.yield`
      // op. The input of the `scf.yield` op is a list of results of `tf.Cast`
      // ops, each of which casts an operand of the current terminator to the
      // corresponding result type of the `tf.IfRegion` op.
      Operation* current_terminator =
          scf_then_or_else_region.front().getTerminator();
      rewriter.setInsertionPoint(current_terminator);
      SmallVector<Value, 4> scf_yield_input;
      for (auto it : llvm::zip(tf_if_region_return_type,
                               current_terminator->getOperands())) {
        scf_yield_input.push_back(rewriter.create<CastOp>(
            current_terminator->getLoc(), std::get<0>(it), std::get<1>(it)));
      }

      rewriter.replaceOpWithNewOp<scf::YieldOp>(current_terminator,
                                                scf_yield_input);
    };

    Location loc = op.getLoc();

    // The condition of an `scf.if` op is a 1-bit signless integer. Whereas, the
    // condition of the `tf.IfRegion` op is a 0-D tensor of 1-bit signless
    // integers. Thus, we use the `tensor.extract` op to compute the condition
    // of `scf.if` from that of `tf.IfRegion`.
    auto scf_if_condition = rewriter.create<tensor::ExtractOp>(loc, op.cond());

    TypeRange tf_if_region_return_type = op.getResultTypes();

    // Create the `scf.if` op.
    auto scf_if_op =
        rewriter.create<scf::IfOp>(loc, tf_if_region_return_type,
                                   scf_if_condition, /*withElseRegion=*/true);

    Region& then_region = op.then_branch();
    Region& else_region = op.else_branch();

    // Create the `then` and `else` regions of the `scf.if` op.
    createScfThenOrElse(then_region, scf_if_op.getThenRegion(),
                        tf_if_region_return_type, rewriter);
    createScfThenOrElse(else_region, scf_if_op.getElseRegion(),
                        tf_if_region_return_type, rewriter);

    // Replace the `tf.IfRegion` op with the results of the `scf.if` op.
    rewriter.replaceOp(op, scf_if_op.getResults());
    return success();
  }
};

/// Convert the `tf.WhileRegion` op to the `scf.while` op.
class ConvertWhileRegionOp : public OpRewritePattern<WhileRegionOp> {
 public:
  using OpRewritePattern<WhileRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileRegionOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_3(mht_3_v, 299, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "matchAndRewrite");

    // Creates the `before` or `after` region of the `scf.while` op. Note that
    // `tf_cond_or_body_region` is the `cond` or `body` region of the
    // `tf.WhileRegion` op. `scf_before_or_after_region` is the `before` or
    // `after` region of the new `scf.while` op. `scf_block_arguments_type` is
    // the type of arguments that need to be in the first block of
    // `scf_before_or_after_region`.
    auto createScfCondOrBody =
        [](Region& tf_cond_or_body_region, Region& scf_before_or_after_region,
           TypeRange scf_block_arguments_type, PatternRewriter& rewriter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_4(mht_4_v, 311, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "lambda");

          // Move the first block of `tf_cond_or_body_region` into the first
          // block of `scf_before_or_after_region` and keep the later's
          // arguments' type as `scf_block_arguments_type`.
          moveBlock(&tf_cond_or_body_region.front(),
                    &scf_before_or_after_region.front(),
                    scf_block_arguments_type, rewriter);

          Operation* cond_or_body_terminator =
              scf_before_or_after_region.front().getTerminator();
          rewriter.setInsertionPoint(cond_or_body_terminator);
          return cond_or_body_terminator;
        };

    ValueRange opInput = op.input();
    TypeRange scf_block_arguments_type = opInput.getType();

    // Create the `scf.while` op.
    auto scf_while_op = rewriter.create<scf::WhileOp>(
        op.getLoc(), op.getResultTypes(), opInput);

    // Create the `before` block of the `scf.while` op (with an `scf.condition`
    // op as the terminator). Note that the arguments' type of this block is
    // kept as `opInput`'s type. Note that the input of an `scf.condition` op is
    // a 1-bit signless integer. But, the condition of the `tf.WhileRegion` op
    // is a 0-D tensor of 1-bit signless integers. Thus, we use the
    // `tensor.extract` op to compute the input of `scf.condition`.
    rewriter.createBlock(&scf_while_op.getBefore());
    Operation* cond_terminator =
        createScfCondOrBody(op.cond(), scf_while_op.getBefore(),
                            scf_block_arguments_type, rewriter);
    auto scf_condition_input = rewriter.create<tensor::ExtractOp>(
        cond_terminator->getLoc(), cond_terminator->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        cond_terminator, scf_condition_input.getResult(),
        scf_while_op.getBefore().front().getArguments());

    // Create the `after` block of the `scf.while` op (with an `scf.yield` op as
    // the terminator). Note that the arguments' type of this block is kept as
    // `opInput`'s type.
    rewriter.createBlock(&scf_while_op.getAfter());
    Operation* body_terminator = createScfCondOrBody(
        op.body(), scf_while_op.getAfter(), scf_block_arguments_type, rewriter);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(body_terminator,
                                              body_terminator->getOperands());

    // Replace the `tf.WhileRegion` op with the `scf.while` op.
    rewriter.replaceOp(op, scf_while_op.getResults());

    return success();
  }
};

}  // end anonymous namespace

void populateTfControlFlowToScfPatterns(MLIRContext* context,
                                        RewritePatternSet* patterns) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_5(mht_5_v, 370, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "populateTfControlFlowToScfPatterns");

  patterns->add<ConvertIfRegionOp, ConvertWhileRegionOp>(context);
}

struct ConvertTfControlFlowToScf
    : public ConvertTfControlFlowToScfPassBase<ConvertTfControlFlowToScf> {
  void runOnOperation() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_tf_control_flow_to_scfDTcc mht_6(mht_6_v, 379, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_tf_control_flow_to_scf.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    populateTfControlFlowToScfPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertTfControlFlowToScfPass() {
  return std::make_unique<ConvertTfControlFlowToScf>();
}

}  // namespace TF
}  // end namespace mlir
