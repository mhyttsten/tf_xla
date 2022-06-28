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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgpu_fusionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgpu_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgpu_fusionDTcc() {
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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "tf-gpu-op-fusion"

namespace mlir {
namespace TF {

namespace {

// GpuOpFusionPass is a pass performing fusion specific to GPU targets.
// This is an ad-hoc pass for now, but should be integrated with some notion
// of "target" in the MLIR pipeline in the future.
class GpuOpFusionPass : public TensorflowGPUFusionBase<GpuOpFusionPass> {
 public:
  void runOnOperation() final;
};

//   %y:6 = "tf.FusedBatchNormV3"(%x, %scale, %offset, %mean, %variance)
//   %0 = "tf.Relu"(%y#0)
// ->
//   %y:6 = "tf._FusedBatchNormEx"(%x, %scale, %offset, %mean, %variance)
//
// Or:
//   %y:6 = "tf.FusedBatchNormV3"(%x, %scale, %offset, %mean, %variance)
//   %0 = "tf.AddV2"(%y#0, %side_input)
//   %1 = "tf.Relu"(%0)
// ->
//  %y:6 = "tf._FusedBatchNormEx"(%x, %scale, %offset, %mean, %variance,
//                                %side_input)
// TODO(aminim): we should revisit this as a declarative pattern.
// For the second pattern, there is not good way in the framework to handle the
// commutativity of the AddV2: we want the FusedBatchNormV3 on any side.
// Also we need some native calls to handle the "hasOneUse" aspects and the
// optional extra operands for the AddV2 case.
struct ReluToFusedBatchNorm : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgpu_fusionDTcc mht_0(mht_0_v, 236, "", "./tensorflow/compiler/mlir/tensorflow/transforms/gpu_fusion.cc", "matchAndRewrite");

    Operation *relu_input = relu_op.features().getDefiningOp();
    if (!relu_input) return failure();
    auto batch_norm = dyn_cast_or_null<FusedBatchNormV3Op>(relu_input);
    AddV2Op add_op;
    Value side_input;
    if (!batch_norm) {
      // We don't have a FusedBatchNorm as input to the ReLu, but we can get
      // through an AddV2 as well.
      add_op = dyn_cast_or_null<AddV2Op>(relu_input);
      if (!add_op) return failure();

      batch_norm =
          dyn_cast_or_null<FusedBatchNormV3Op>(add_op.x().getDefiningOp());
      if (batch_norm) {
        side_input = add_op.y();
      } else {
        // Didn't get a FusedBatchNorm on the LHS of the AddV2, try the RHS.
        batch_norm =
            dyn_cast_or_null<FusedBatchNormV3Op>(add_op.y().getDefiningOp());
        if (!batch_norm) return failure();
        side_input = add_op.x();
      }
    }
    assert(batch_norm);
    if (batch_norm.is_training()) return failure();
    if (!batch_norm.y().hasOneUse()) return failure();

    // Build the newly fused operation to replace the batch norm
    OperationState state(batch_norm.getLoc(),
                         _FusedBatchNormExOp::getOperationName());
    state.addOperands(batch_norm.getOperands());
    if (side_input) state.operands.push_back(side_input);
    state.addTypes(batch_norm.getResultTypes());
    state.addAttributes(batch_norm->getAttrs());
    Operation *op = rewriter.create(state);
    rewriter.replaceOp(batch_norm, op->getResults());

    // Depending on the case, we may fuse the add, the relu, or both.
    if (!add_op || add_op.z().hasOneUse()) {
      // We fuse the Relu only if the add has a single use, otherwise we only
      // fuse the add itself.
      op->setAttr("activation_mode", rewriter.getStringAttr("Relu"));
      rewriter.replaceOp(relu_op, op->getResult(0));
    }
    if (add_op) {
      rewriter.replaceOp(add_op, op->getResult(0));
    }

    return success();
  }
};

void GpuOpFusionPass::runOnOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgpu_fusionDTcc mht_1(mht_1_v, 292, "", "./tensorflow/compiler/mlir/tensorflow/transforms/gpu_fusion.cc", "GpuOpFusionPass::runOnOperation");

  FuncOp func = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<ReluToFusedBatchNorm>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateGpuOpFusionPass() {
  return std::make_unique<GpuOpFusionPass>();
}

}  // namespace TF
}  // namespace mlir
