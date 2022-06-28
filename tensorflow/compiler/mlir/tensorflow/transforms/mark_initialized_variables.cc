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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc() {
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Threading.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/session_utils.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

// Returns true if the variable 'var_handle_op' is initialized in 'session'.
bool IsVariableInitialized(mlir::TF::VarHandleOp var_handle_op,
                           llvm::StringRef device_name,
                           const tensorflow::DeviceMgr* mgr,
                           tensorflow::Session* session) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.cc", "IsVariableInitialized");

  auto* var_ptr = GetVariableFromSession(var_handle_op, device_name, mgr);
  if (!var_ptr) return false;
  auto* tensor = var_ptr->tensor();
  bool is_initialized = tensor && tensor->IsInitialized();
  var_ptr->Unref();
  return is_initialized;
}

LogicalResult MarkInitializedVariablesInFunction(FuncOp function,
                                                 tensorflow::Session* session) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.cc", "MarkInitializedVariablesInFunction");

  if (!session || !llvm::hasSingleElement(function)) return success();
  Block& block = function.front();

  const tensorflow::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok())
    return function->emitError("failed to fetch device manager: " +
                               status.error_message());

  // Fetch all varHandleOp in the function.
  llvm::SmallVector<TF::VarHandleOp, 4> var_ops;
  for (auto var_handle_op : block.getOps<TF::VarHandleOp>())
    var_ops.emplace_back(var_handle_op);

  // Get resources from Session.
  auto resource_tensors_or = GetResourcesFromSession(var_ops, session);
  if (!resource_tensors_or.ok())
    return function->emitError(resource_tensors_or.status().message().data());

  MLIRContext* context = function.getContext();
  for (auto var_and_tensor : llvm::zip(var_ops, resource_tensors_or.value())) {
    auto& var_op = std::get<0>(var_and_tensor);
    auto& resource_tensor = std::get<1>(var_and_tensor);
    bool is_variable_initialized = false;
    if (resource_tensor.dtype() != tensorflow::DT_RESOURCE) {
      is_variable_initialized = true;
    } else {
      auto handle = resource_tensor.scalar<tensorflow::ResourceHandle>()();
      is_variable_initialized =
          IsVariableInitialized(var_op, handle.device(), mgr, session);
    }
    var_op->setAttr("_is_initialized",
                    BoolAttr::get(context, is_variable_initialized));
  }
  return success();
}

LogicalResult MarkInitializedVariablesInFunction(ModuleOp module,
                                                 tensorflow::Session* session) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_initialized_variablesDTcc mht_2(mht_2_v, 266, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.cc", "MarkInitializedVariablesInFunction");

  auto functions_range = module.getOps<FuncOp>();
  return mlir::failableParallelForEach(
      module.getContext(), functions_range.begin(), functions_range.end(),
      [&](FuncOp function) {
        return MarkInitializedVariablesInFunction(function, session);
      });
}

}  // namespace tf_saved_model
}  // namespace mlir
