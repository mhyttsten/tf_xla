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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc() {
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

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// This pass is used to fold tfl.const ops to each subgraph (FuncOp):
// See the example below:
//
// In main:
// %0 = tfl.const...
// %1 = tfl.const...
// %2 = call func_1(..., %0,...)
// %3 = call func_2(..., %0, ..., %1...)
// ...
//
// Then those consts will be copied into each function and replace their usage.
// func_1:
//   %0 = tfl.const...
// func_2:
//   %0 = tfl.const...
//   %1 = tfl.const...
class FoldConstantsToSubgraphPass
    : public mlir::PassWrapper<FoldConstantsToSubgraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "getArgument");

    return "tfl-fold-constants-to-subgraph";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "getDescription");

    return "Fold constants into each subgraph.";
  }
  FoldConstantsToSubgraphPass() = default;
  FoldConstantsToSubgraphPass(const FoldConstantsToSubgraphPass& other) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_2(mht_2_v, 245, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "FoldConstantsToSubgraphPass");

    this->fold_all_constants_flag_ = other.fold_all_constants_flag_;
  }
  explicit FoldConstantsToSubgraphPass(bool fold_all_constants) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "FoldConstantsToSubgraphPass");

    fold_all_constants_flag_ = fold_all_constants;
  }

 private:
  void runOnOperation() override;

  Option<bool> fold_all_constants_flag_{
      *this, "fold-all-constants",
      llvm::cl::desc("Whether to fold all constants or just i32."),
      llvm::cl::init(false)};
};

void CopyConstantIntoFunc(int argument_index, Operation* const_op,
                          FuncOp func) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "CopyConstantIntoFunc");

  assert((llvm::isa<TFL::ConstOp, TFL::QConstOp>(const_op)) &&
         "Expect QConst or Const op.");
  OpBuilder builder(func.getBody());
  auto cloned_const_op = const_op->clone();
  cloned_const_op->setLoc(func.getBody().getLoc());
  builder.insert(cloned_const_op);
  // Rewire the usage.
  func.getArgument(argument_index)
      .replaceAllUsesWith(cloned_const_op->getResult(0));
}

bool IsConstOrQConstInt(Operation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_5(mht_5_v, 283, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "IsConstOrQConstInt");

  if (!llvm::isa<TFL::ConstOp, TFL::QConstOp>(op)) return false;

  if (auto const_op = dyn_cast_or_null<TFL::ConstOp>(op)) {
    // ConstOp path.
    auto type = const_op.getType()
                    .dyn_cast_or_null<RankedTensorType>()
                    .getElementType();
    if (!type.isInteger(32) && !type.isInteger(64)) return false;
  } else {
    // QConstOp path.
    auto qconst_op = dyn_cast<TFL::QConstOp>(op);
    auto type =
        quant::QuantizedType::getQuantizedElementType(qconst_op.getType());
    if (type.getStorageTypeIntegralWidth() != 32) {
      return false;
    }
  }
  return true;
}

void FoldConstantsToSubgraphPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSfold_constants_to_subgraphDTcc mht_6(mht_6_v, 307, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/fold_constants_to_subgraph.cc", "FoldConstantsToSubgraphPass::runOnOperation");

  auto module = getOperation();

  for (auto fn : module.getOps<FuncOp>()) {
    fn.walk([&](Operation* op) {
      if (!llvm::isa<TFL::ConstOp, TFL::QConstOp>(op)) return;

      // We only fold int32/int64 for Const and i32 for QConst if not specify
      // all constants flag. (Since they're more like "configs" or i32 biases.)
      // We will fold every const ops (and q_const ops) if we speicfy the
      // fold_all_constants_flag.
      if (!fold_all_constants_flag_) {
        if (!IsConstOrQConstInt(op)) return;
      }

      for (auto consumer : op->getResult(0).getUsers()) {
        auto consumer_call = llvm::dyn_cast_or_null<func::CallOp>(consumer);

        if (!consumer_call) continue;

        auto function_name = consumer_call.getCallee();

        // Locate the argument position of the use.
        int argument_index = -1;
        for (int i = 0; i < consumer_call.getNumOperands(); ++i) {
          if (consumer_call.getOperand(i) == op->getResult(0)) {
            argument_index = i;
            break;
          }
        }

        // Copy the const into the consumer func and replace their usages.
        FuncOp func = module.lookupSymbol<FuncOp>(function_name);

        CopyConstantIntoFunc(argument_index, op, func);
      }
    });
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateFoldConstantsToSubgraphPass(
    bool fold_all_constants) {
  return std::make_unique<FoldConstantsToSubgraphPass>(fold_all_constants);
}

static PassRegistration<FoldConstantsToSubgraphPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
