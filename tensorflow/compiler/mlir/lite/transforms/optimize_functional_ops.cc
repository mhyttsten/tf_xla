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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

// Module pass to optimize TensorFlow functional ops.
struct OptimizeFunctionalOpsPass
    : public PassWrapper<OptimizeFunctionalOpsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-optimize-functional-ops";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "getDescription");

    // This is a brief description of the pass.
    return "Optimize TensorFlow functional ops";
  }
};

// Updates function return type of the given functions to match the terminator
// op operands' types.
//
// Requires the function has exactly one block.
void UpdateFuncType(FuncOp func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "UpdateFuncType");

  Operation* terminator = func.front().getTerminator();
  auto return_types = llvm::to_vector<4>(terminator->getOperandTypes());

  FunctionType func_type = func.getFunctionType();
  if (llvm::makeArrayRef(return_types) == func_type.getResults()) return;

  auto updated_type =
      FunctionType::get(func.getContext(), func_type.getInputs(), return_types);
  func.setType(updated_type);
}

// TODO(jpienaar): Remove when recursive side-effect modeling is added.
bool IsSideEffectFree(FuncOp func) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "IsSideEffectFree");

  return !func.getBody()
              .walk([&](Operation* op) {
                if (!MemoryEffectOpInterface::hasNoEffect(op) &&
                    !op->hasTrait<OpTrait::IsTerminator>())
                  return WalkResult::interrupt();
                return WalkResult::advance();
              })
              .wasInterrupted();
}

// Folds TensorFlow If op with constant conditional operand by inlining the
// function body based on the conditional value.
class FoldIfOp : public OpRewritePattern<TF::IfOp> {
 public:
  explicit FoldIfOp(MLIRContext* context)
      : OpRewritePattern<TF::IfOp>(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "FoldIfOp");
}

  LogicalResult matchAndRewrite(TF::IfOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_5(mht_5_v, 272, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "matchAndRewrite");

    // This pattern is restricted to if ops in functions with exactly one block
    // and therefore one terminator op. So, that function return type can be
    // updated if operands' shapes change after inlining. Without this
    // restriction, it would require tensor cast ops.
    FuncOp parent_op = op->getParentOfType<FuncOp>();
    if (!llvm::hasSingleElement(parent_op)) return failure();

    // Find the then and else branch functions.
    FuncOp then_func = op.then_function();
    FuncOp else_func = op.else_function();

    // If the If has no uses and its functions are side-effect free, then
    // remove.
    // TODO(jpienaar): Remove once recusive side-effects are supported.
    if (op.use_empty() &&
        (op.is_stateless() ||
         (IsSideEffectFree(then_func) && IsSideEffectFree(else_func)))) {
      rewriter.eraseOp(op.getOperation());
      return success();
    }

    // Extract the constant cond value.
    DenseElementsAttr cond;
    if (!matchPattern(op.cond(), m_Constant(&cond))) return failure();

    // TODO(hinsu): Handle constants that are not scalar booleans.
    auto cond_type = cond.getType().dyn_cast<RankedTensorType>();
    if (!cond_type || !cond_type.getShape().equals({}) ||
        !cond_type.getElementType().isInteger(/*width=*/1))
      return failure();

    // Identify the branch to inline.
    bool cond_value = (*cond.value_begin<APInt>()).getSExtValue();
    FuncOp func = cond_value ? then_func : else_func;

    // Make sure that the function has exactly one block to simplify inlining.
    // TFLite doesn't use control flow with blocks so functions with more than
    // one blocks are not encountered in practice.
    if (!llvm::hasSingleElement(func)) return failure();

    BlockAndValueMapping mapper;
    for (int i = 0, e = func.getNumArguments(); i != e; ++i)
      mapper.map(func.getArgument(i), op.getOperand(i + 1));

    llvm::SmallVector<Value, 4> updated_results;
    for (auto& op_to_inline : func.front()) {
      // If this is a terminator, identify the values to use to replace the
      // original If op.
      if (op_to_inline.hasTrait<OpTrait::IsTerminator>()) {
        updated_results.reserve(op_to_inline.getNumOperands());
        for (Value operand : op_to_inline.getOperands())
          updated_results.push_back(mapper.lookup(operand));
        break;
      }

      // Otherwise, clone the op here.
      rewriter.clone(op_to_inline, mapper);
    }
    rewriter.replaceOp(op, updated_results);

    // Here, shapes of the updated_results may not match the original values. If
    // any of the values are operands of the terminator op, then the function
    // return type should be updated.
    UpdateFuncType(parent_op);

    return success();
  }
};

void OptimizeFunctionalOpsPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimize_functional_opsDTcc mht_6(mht_6_v, 345, "", "./tensorflow/compiler/mlir/lite/transforms/optimize_functional_ops.cc", "OptimizeFunctionalOpsPass::runOnOperation");

  RewritePatternSet patterns(&getContext());

  patterns.add<FoldIfOp>(&getContext());

  ModuleOp module = getOperation();
  (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

PassRegistration<OptimizeFunctionalOpsPass> pass;
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeFunctionalOpsPass() {
  return std::make_unique<OptimizeFunctionalOpsPass>();
}

}  // namespace TFL
}  // namespace mlir
