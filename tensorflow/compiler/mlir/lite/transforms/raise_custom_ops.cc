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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc() {
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

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// NOLINTNEXTLINE
static llvm::cl::list<std::string> target_ops(
    "tfl-test-raise-tf-targets", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of target op names to be wrapped. Only"
                   " used in tests"),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace TFL {
namespace {
// This transformation pass takes an operation with unknown op properties and
// wrap it by a TFL::CustomTfOp.
struct RaiseCustomOpsPass
    : public PassWrapper<RaiseCustomOpsPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "getDependentDialects");

    registry.insert<TensorFlowLiteDialect>();
  }

 public:
  explicit RaiseCustomOpsPass()
      : target_op_names(target_ops.begin(), target_ops.end()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "RaiseCustomOpsPass");
}
  explicit RaiseCustomOpsPass(const std::vector<std::string> &target_ops)
      : target_op_names(target_ops.begin(), target_ops.end()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "RaiseCustomOpsPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-raise-custom-ops";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_4(mht_4_v, 244, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "getDescription");

    // This is a brief description of the pass.
    return "Raise custom ops into tflite dialect.";
  }

  void runOnOperation() override;

 private:
  // If this set is empty, then all the qualified ops will be wrapped.
  const absl::flat_hash_set<std::string> target_op_names;
};

void RaiseCustomOpsPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSraise_custom_opsDTcc mht_5(mht_5_v, 259, "", "./tensorflow/compiler/mlir/lite/transforms/raise_custom_ops.cc", "RaiseCustomOpsPass::runOnOperation");

  auto fn = getOperation();
  OpBuilder builder(fn.getContext());

  llvm::SmallVector<Operation *, 4> custom_ops;
  fn.walk([&](Operation *op) {
    // Skips already imported ops that are imported as CustomTfOp.
    if (op->getParentOfType<CustomTfOp>()) return;
    if (llvm::isa<TFL::CustomTfOp>(op) || llvm::isa<TFL::CustomOp>(op)) return;

    std::string op_name = op->getName().getIdentifier().str();
    // Wrap the operation, if
    // - the op is targeted explicitly, or
    // - the op isn't registered when there are no target list.
    if (target_op_names.contains(op_name) ||
        (target_op_names.empty() && !op->isRegistered())) {
      custom_ops.push_back(op);
    }
  });

  for (auto *op : custom_ops) {
    builder.setInsertionPoint(op);
    Location loc = op->getLoc();
    auto custom_op = builder.create<CustomTfOp>(loc, op->getResultTypes(),
                                                op->getOperands());
    Region region;
    Block *new_block = new Block;
    region.push_back(new_block);

    builder.setInsertionPointToEnd(&region.front());
    Operation *inner_op = builder.clone(*op);

    new_block->addArguments(op->getOperandTypes(),
                            SmallVector<Location>(op->getNumOperands(), loc));
    for (auto &idx_args : llvm::enumerate(new_block->getArguments())) {
      inner_op->setOperand(idx_args.index(), idx_args.value());
    }
    custom_op->setAttrs(inner_op->getAttrs());
    builder.create<YieldOp>(loc, inner_op->getResults());
    custom_op.body().takeBody(region);

    op->replaceAllUsesWith(custom_op);
    op->erase();
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect raise custom op pass.
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseCustomOpsPass(
    const std::vector<std::string> &target_ops) {
  return std::make_unique<RaiseCustomOpsPass>(target_ops);
}

static PassRegistration<RaiseCustomOpsPass> pass;

}  // namespace TFL
}  // namespace mlir
