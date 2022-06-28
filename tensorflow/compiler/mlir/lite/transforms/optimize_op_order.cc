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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc() {
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

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

// Dequantize ops will produce 3x larger tensors, so we want to move it after
// some passthrough ops to reduce the memory consumption.
struct PushDownDequantize : public OpRewritePattern<DequantizeOp> {
  explicit PushDownDequantize(MLIRContext* context)
      : OpRewritePattern<DequantizeOp>(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "PushDownDequantize");
}

  LogicalResult matchAndRewrite(DequantizeOp dequantize_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "matchAndRewrite");

    if (!dequantize_op->hasOneUse()) return failure();

    auto use = dequantize_op->use_begin();
    Operation* passthrough_op = use->getOwner();
    unsigned operand_index = use->getOperandNumber();
    if (passthrough_op->hasTrait<OpTrait::IsTerminator>()) return failure();

    auto get_num_elements = [](RankedTensorType tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "lambda");

      return tensor.getNumElements();
    };

    // If the op is the pass-through op with (3x) smaller output, the dequantize
    // op can be pushed down to the single result of this op.
    if (!llvm::dyn_cast<mlir::SameScalesOpInterface>(passthrough_op) ||
        passthrough_op->getNumResults() != 1) {
      return failure();
    }
    // Only push down the dequantize op when the output is smaller, so that it
    // can have smaller memory usage.
    auto input_type =
        dequantize_op.output().getType().dyn_cast<RankedTensorType>();
    auto output_type =
        passthrough_op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!input_type || !output_type || !input_type.hasStaticShape() ||
        !output_type.hasStaticShape() ||
        get_num_elements(input_type) <= get_num_elements(output_type)) {
      return failure();
    }

    // Set the output type of the dequantize op and push it down.
    dequantize_op.output().setType(output_type);
    passthrough_op->replaceAllUsesWith(dequantize_op);

    // Set the input type of the passthrough op and pull it up.
    Type new_output_type =
        QuantizedType::getQuantizedElementType(dequantize_op.input().getType())
            .castFromExpressedType(output_type);
    passthrough_op->getResult(0).setType(new_output_type);
    passthrough_op->setOperand(operand_index, dequantize_op.input());

    // Set the input of the dequantize to the result of the passthrough op.
    // And switch the order of the ops.
    dequantize_op->setOperand(0, passthrough_op->getResult(0));
    dequantize_op->moveAfter(passthrough_op);
    return success();
  }
};

// This transformation pass optimizes the op execution order of the ops in the
// model.
struct OptimizeOpOrderPass
    : public PassWrapper<OptimizeOpOrderPass, OperationPass<FuncOp>> {
  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_3(mht_3_v, 269, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-optimize-op-order";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "getDescription");

    // This is a brief description of the pass.
    return "Optimize the execution order of the ops.";
  }
};

void OptimizeOpOrderPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_op_orderDTcc mht_5(mht_5_v, 286, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_op_order.cc", "OptimizeOpOrderPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  patterns.add<PushDownDequantize>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite optimize op order pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeOpOrderPass() {
  return std::make_unique<OptimizeOpOrderPass>();
}

static PassRegistration<OptimizeOpOrderPass> pass;

}  // namespace TFL
}  // namespace mlir
