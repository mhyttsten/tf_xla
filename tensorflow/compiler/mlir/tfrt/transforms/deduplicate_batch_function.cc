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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc() {
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

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

using ::mlir::ArrayRef;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::SymbolTable;
using ::mlir::SymbolTableCollection;
using ::mlir::SymbolUserMap;
using ::mlir::func::FuncOp;

// This only includes some preliminary checks as this is a short term solution.
bool AreEquivalent(FuncOp& lhs, FuncOp& rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tfrt/transforms/deduplicate_batch_function.cc", "AreEquivalent");

  if (lhs.getFunctionType() != rhs.getFunctionType()) return false;

  for (auto arg_pair : llvm::zip(lhs.getArguments(), rhs.getArguments())) {
    auto& lhs_arg = std::get<0>(arg_pair);
    auto& rhs_arg = std::get<1>(arg_pair);
    if (lhs_arg.getType() != rhs_arg.getType()) return false;
  }

  auto lhs_ops = lhs.getBody().getOps();
  auto rhs_ops = rhs.getBody().getOps();
  if (std::distance(lhs_ops.begin(), lhs_ops.end()) !=
      std::distance(rhs_ops.begin(), rhs_ops.end()))
    return false;

  for (auto op_pair : llvm::zip(lhs_ops, rhs_ops)) {
    auto& lhs_op = std::get<0>(op_pair);
    auto& rhs_op = std::get<1>(op_pair);
    if (lhs_op.getName() != rhs_op.getName()) return false;
    if (lhs_op.getNumRegions() != rhs_op.getNumRegions()) return false;
    if (lhs_op.getNumSuccessors() != rhs_op.getNumSuccessors()) return false;
    if (!std::equal(lhs_op.getOperandTypes().begin(),
                    lhs_op.getOperandTypes().end(),
                    rhs_op.getOperandTypes().begin()))
      return false;
    if (!std::equal(lhs_op.getResultTypes().begin(),
                    lhs_op.getResultTypes().end(),
                    rhs_op.getResultTypes().begin()))
      return false;
  }

  return true;
}

// Deduplicate the functions if all users are BatchFunctionOp and have the same
// shared_name.
//
// TODO(b/192463730): this is the short term solution and not needed anymore
// after the shape inference pass is revamped with ideal solution
// (b/192463730#comment11).
class DeduplicateFunctionsInovkedByBatchFunction
    : public mlir::PassWrapper<DeduplicateFunctionsInovkedByBatchFunction,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc mht_1(mht_1_v, 265, "", "./tensorflow/compiler/mlir/tfrt/transforms/deduplicate_batch_function.cc", "getArgument");

    return "tfrt-deduplicate-functions-invoked-by-batch-function";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tfrt/transforms/deduplicate_batch_function.cc", "getDescription");

    return "Deduplicate the functions invoked by tf.BatchFunction with the "
           "same shared_name";
  }
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/mlir/tfrt/transforms/deduplicate_batch_function.cc", "runOnOperation");

    if (failed(Run())) {
      signalPassFailure();
    }
  }

  mlir::LogicalResult Run();
};

mlir::LogicalResult DeduplicateFunctionsInovkedByBatchFunction::Run() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSdeduplicate_batch_functionDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/mlir/tfrt/transforms/deduplicate_batch_function.cc", "DeduplicateFunctionsInovkedByBatchFunction::Run");

  ModuleOp module = getOperation();
  SymbolTableCollection symbol_table_collection;
  SymbolTable& symbol_table = symbol_table_collection.getSymbolTable(module);
  SymbolUserMap symbol_users(symbol_table_collection, module);

  // Categorize the functions invoked by BatchFunctionOp by its shared_name.
  llvm::StringMap<llvm::SmallVector<FuncOp, 2>> shared_name_to_func_ops;

  for (auto func : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    ArrayRef<Operation*> users = symbol_users.getUsers(func);
    llvm::StringRef shared_name;
    // Deduplicate the function only if all users are BatchFunctionOp and have
    // the same shared_name
    if (!users.empty() && llvm::all_of(users, [&shared_name](Operation* user) {
          auto op = llvm::dyn_cast_or_null<mlir::TF::BatchFunctionOp>(user);
          // User is not a BatchFunctionOp
          if (!op) return false;
          if (shared_name.empty()) {
            shared_name = op.shared_name();
            return true;
          }
          return shared_name == op.shared_name();
        })) {
      shared_name_to_func_ops[shared_name].push_back(func);
    }
  }

  for (auto& it : shared_name_to_func_ops) {
    auto& func_ops = it.second;
    FuncOp& func_op_to_keep = func_ops.front();
    for (FuncOp& func_op_to_remove : llvm::drop_begin(func_ops)) {
      if (!AreEquivalent(func_op_to_keep, func_op_to_remove)) {
        return func_op_to_remove.emitError(
            "func_ops for BatchFunctionOp with the same shared name are "
            "different");
      }
      if (failed(SymbolTable::replaceAllSymbolUses(
              func_op_to_remove, func_op_to_keep.getSymNameAttr(), module))) {
        return func_op_to_remove.emitError("unable to replace the symbol use");
      }
      symbol_table.erase(func_op_to_remove);
    }
  }

  return mlir::success();
}
}  // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateDeduplicateFunctionsInovkedByBatchFunctionPass() {
  return std::make_unique<DeduplicateFunctionsInovkedByBatchFunction>();
}

static mlir::PassRegistration<DeduplicateFunctionsInovkedByBatchFunction>
    register_pass;

}  // namespace tfrt_compiler
}  // namespace tensorflow
