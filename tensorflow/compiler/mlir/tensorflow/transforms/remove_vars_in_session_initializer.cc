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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc() {
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

#include <algorithm>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

namespace mlir {
namespace tf_saved_model {
namespace {
using mlir::Operation;
using mlir::TF::VarHandleOp;

class RemoveVariablesInSessionInitializerPass
    : public RemoveVariablesInSessionInitializerPassBase<
          RemoveVariablesInSessionInitializerPass> {
 public:
  void runOnOperation() override;
};

void RecursiveRemove(Operation* op,
                     llvm::SmallVectorImpl<Operation*>& erase_list,
                     llvm::SmallPtrSetImpl<Operation*>& dead_ops) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/tensorflow/transforms/remove_vars_in_session_initializer.cc", "RecursiveRemove");

  for (mlir::Value res : op->getResults()) {
    for (Operation* user : res.getUsers()) {
      if (!dead_ops.insert(user).second) continue;
      RecursiveRemove(user, erase_list, dead_ops);
    }
  }

  erase_list.push_back(op);

  for (auto& use : op->getOpOperands()) {
    if (auto op_result = use.get().dyn_cast<mlir::OpResult>()) {
      Operation* def = op_result.getDefiningOp();
      if (!dead_ops.insert(def).second) continue;
      RecursiveRemove(def, erase_list, dead_ops);
    }
  }
}

void RemoveVariables(llvm::ArrayRef<VarHandleOp> vars) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/tensorflow/transforms/remove_vars_in_session_initializer.cc", "RemoveVariables");

  // TODO(b/160906885): Repalce the following code with an non-recursive one.
  llvm::SmallVector<Operation*, 4> erase_list;
  llvm::SmallPtrSet<Operation*, 4> dead_ops;

  // Marks all the variables dead.
  dead_ops.insert(vars.begin(), vars.end());

  // Removes relevant ops in topological order.
  for (auto& op : vars) RecursiveRemove(op, erase_list, dead_ops);

  // Erases the ops.
  for (auto op : erase_list) op->erase();
}

void RemoveVariablesInSessionInitializerPass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSremove_vars_in_session_initializerDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/mlir/tensorflow/transforms/remove_vars_in_session_initializer.cc", "RemoveVariablesInSessionInitializerPass::runOnOperation");

  ModuleOp module = getOperation();
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);

  if (!session_init_op) return;

  SymbolTable symbol_table(module);

  for (auto sym_ref : session_init_op.initializers()) {
    FuncOp init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        sym_ref.cast<FlatSymbolRefAttr>().getValue());

    if (!init_func_op) {
      module.emitError("no session initializer function found");
      return signalPassFailure();
    }

    if (init_func_op.getBlocks().size() != 1) {
      init_func_op.emitError("expects exactly one block in the MLIR function");
      return signalPassFailure();
    }

    auto var_handle_ops =
        init_func_op.getBlocks().front().getOps<VarHandleOp>();
    llvm::SmallVector<VarHandleOp, 4> init_vars(var_handle_ops.begin(),
                                                var_handle_ops.end());
    RemoveVariables(init_vars);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateRemoveVariablesInSessionInitializerPass() {
  return std::make_unique<RemoveVariablesInSessionInitializerPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
