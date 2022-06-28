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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc() {
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

#include <memory>

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace {

class ConvertResultsBroadcastableShapeOp : public RewritePattern {
 public:
  ConvertResultsBroadcastableShapeOp(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "ConvertResultsBroadcastableShapeOp");
}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 private:
  template <typename Op>
  LogicalResult RewriteEqOp(Operation* op, PatternRewriter& rewriter) const;

  LogicalResult RewriteOp(
      Operation* op, PatternRewriter& rewriter,
      const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                               SmallVectorImpl<int64_t>&)>&
          get_broadcasted_shape) const;

  LogicalResult RewriteBatchMatMulV2Op(Operation* op,
                                       PatternRewriter& rewriter) const;
};

class BroadcastFoldPass : public TF::BroadcastFoldPassBase<BroadcastFoldPass> {
 public:
  void runOnOperation() override;
};

LogicalResult ConvertResultsBroadcastableShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "ConvertResultsBroadcastableShapeOp::matchAndRewrite");

  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);

  // tf.Equal and tf.NotEqual ops only satisfy ResultsBroadcastableShape when
  // incompatible_shape_error is `true` (what is also checked by the verifier).
  if (succeeded(RewriteEqOp<TF::EqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteEqOp<TF::NotEqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteBatchMatMulV2Op(op, rewriter))) return success();

  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteBatchMatMulV2Op(
    Operation* op, PatternRewriter& rewriter) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_2(mht_2_v, 254, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "ConvertResultsBroadcastableShapeOp::RewriteBatchMatMulV2Op");

  auto matmul_op = llvm::dyn_cast<TF::BatchMatMulV2Op>(op);
  if (!matmul_op) return failure();

  // Gets the broadcasted output shape for tf.BatchMatMulV2Op. `shape_x` is the
  // shape of op's first/left-hand-side operand and `shape_y` is the shape of
  // op's second/right-hand-side operand.
  const auto get_broadcasted_shape =
      [&](ArrayRef<int64_t> shape_x, ArrayRef<int64_t> shape_y,
          SmallVectorImpl<int64_t>& result_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "lambda");

        if (shape_x.size() < 2 || shape_y.size() < 2) {
          return false;
        }

        // Checks outer dimensions (i.e., the dimensions higher than 2D) are
        // broadcastable. If true, then get the broadcasted shape for outer
        // dimension.
        if (!OpTrait::util::getBroadcastedShape(
                shape_x.drop_back(2), shape_y.drop_back(2), result_shape)) {
          return false;
        }

        const int x_row =
            matmul_op.adj_x() ? shape_x.back() : *(shape_x.rbegin() + 1);
        const int x_col =
            !matmul_op.adj_x() ? shape_x.back() : *(shape_x.rbegin() + 1);

        const int y_row =
            matmul_op.adj_y() ? shape_y.back() : *(shape_y.rbegin() + 1);
        const int y_col =
            !matmul_op.adj_y() ? shape_y.back() : *(shape_y.rbegin() + 1);

        // Checks that matrix multiply can perform a valid contraction.
        if (x_col != y_row) {
          result_shape.clear();
          return false;
        }

        result_shape.push_back(x_row);
        result_shape.push_back(y_col);
        return true;
      };

  return RewriteOp(op, rewriter, get_broadcasted_shape);
}

template <typename Op>
LogicalResult ConvertResultsBroadcastableShapeOp::RewriteEqOp(
    Operation* op, PatternRewriter& rewriter) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_4(mht_4_v, 308, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "ConvertResultsBroadcastableShapeOp::RewriteEqOp");

  auto eq_op = llvm::dyn_cast_or_null<Op>(op);
  if (eq_op && eq_op.incompatible_shape_error())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);
  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteOp(
    Operation* op, PatternRewriter& rewriter,
    const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                             SmallVectorImpl<int64_t>&)>& get_broadcasted_shape)
    const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_5(mht_5_v, 322, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "ConvertResultsBroadcastableShapeOp::RewriteOp");

  if (op->getNumOperands() != 2 || op->getResultTypes().size() != 1)
    return failure();

  // Check that the result shape is fully defined.
  auto result_type =
      op->getResultTypes().front().dyn_cast_or_null<RankedTensorType>();
  if (!result_type || !result_type.hasStaticShape()) return failure();

  bool changed = false;
  for (uint64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    // Check that the i'th operand is a broadcast.
    auto broadcast = llvm::dyn_cast_or_null<TF::BroadcastToOp>(
        op->getOpOperand(i).get().getDefiningOp());
    if (!broadcast) continue;

    // Check that the operand of the broadcast has fully defined shape.
    auto broadcast_arg_type =
        broadcast.input().getType().dyn_cast_or_null<RankedTensorType>();
    if (!broadcast_arg_type || !broadcast_arg_type.hasStaticShape()) continue;

    // Check that the other argument has fully defined shape.
    auto argument_type = op->getOpOperand(1 - i)
                             .get()
                             .getType()
                             .dyn_cast_or_null<RankedTensorType>();
    if (!argument_type || !argument_type.hasStaticShape()) continue;

    // Get the unbroadcasted shapes in the operand order.
    std::array<llvm::ArrayRef<int64_t>, 2> operand_shapes;
    operand_shapes[i] = broadcast_arg_type.getShape();
    operand_shapes[1 - i] = argument_type.getShape();

    // Check that the input of the broadcast and the other operand is broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcasted_shape;
    if (!get_broadcasted_shape(operand_shapes[0], operand_shapes[1],
                               broadcasted_shape))
      continue;

    // Check that an implicit broadcast between the operand of the broadcast and
    // the other argument would result in the same type as the result type.
    if (broadcasted_shape != result_type.getShape()) continue;

    // Update the operand of the op to be the operand of the broadcast.
    rewriter.updateRootInPlace(
        op, [&]() { op->getOpOperand(i).set(broadcast.input()); });
    changed = true;
  }
  return success(changed);
}

void BroadcastFoldPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfold_broadcastDTcc mht_6(mht_6_v, 377, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fold_broadcast.cc", "BroadcastFoldPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertResultsBroadcastableShapeOp>(func.getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<FuncOp>> CreateBroadcastFoldPass() {
  return absl::make_unique<BroadcastFoldPass>();
}
}  // namespace TF

}  // namespace mlir
