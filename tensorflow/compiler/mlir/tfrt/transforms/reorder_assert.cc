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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc() {
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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// Reorder tf.Assert ops or tf.If ops that contains only tf.Assert ops to the
// end of the function, in order to avoid unnecessary control dependencies
// between tf.Assert and other ops.
class ReorderTfAssertPass
    : public mlir::PassWrapper<ReorderTfAssertPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "getArgument");
 return "tfrt-reorder-tf-assert"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "getDescription");

    return "Move tf.Assert to the end of the function to avoid unnecessary "
           "control dependencies";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_2(mht_2_v, 212, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "runOnOperation");

    auto module = getOperation();
    for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
      ProcessFunction(func_op);
    }
  }

  void ProcessFunction(mlir::func::FuncOp func_op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_3(mht_3_v, 222, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "ProcessFunction");

    auto& block = func_op.front();

    llvm::SmallVector<mlir::Operation*, 2> assert_ops;
    for (mlir::Operation& op : block) {
      if (auto assert_op = llvm::dyn_cast<mlir::TF::AssertOp>(&op)) {
        assert_ops.push_back(assert_op);
      }

      if (auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(&op)) {
        if (IsAssertOnlyIfOp(if_op)) {
          assert_ops.push_back(if_op);
        }
      }
    }

    auto& return_op = block.back();

    for (auto assert_op : assert_ops) {
      assert_op->moveBefore(&return_op);
    }
  }

  bool IsAssertOnlyIfOp(mlir::TF::IfOp op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_4(mht_4_v, 248, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "IsAssertOnlyIfOp");

    // If the results of the if op are used by some other ops, we cannot reorder
    // it.
    if (!op->use_empty()) return false;

    // Only reorder if both branches are non-side-effecting or containing only
    // Assert ops.
    if (IsFunctionNonSideEffectingOrAssert(op.then_function()) &&
        IsFunctionNonSideEffectingOrAssert(op.else_function()))
      return true;

    return false;
  }

  bool IsFunctionNonSideEffectingOrAssert(mlir::func::FuncOp func_op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSreorder_assertDTcc mht_5(mht_5_v, 265, "", "./tensorflow/compiler/mlir/tfrt/transforms/reorder_assert.cc", "IsFunctionNonSideEffectingOrAssert");

    auto& block = func_op.front();
    for (mlir::Operation& op : block) {
      if (!llvm::isa<mlir::TF::AssertOp>(&op) &&
          !mlir::MemoryEffectOpInterface::hasNoEffect(&op))
        return false;
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateReorderTfAssertPass() {
  return std::make_unique<ReorderTfAssertPass>();
}

static mlir::PassRegistration<ReorderTfAssertPass> register_pass(
    CreateReorderTfAssertPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
