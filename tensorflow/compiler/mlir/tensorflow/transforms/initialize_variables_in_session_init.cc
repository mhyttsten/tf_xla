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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/session_utils.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {

void InitializeVariable(TF::VarHandleOp var_handle_op,
                        tensorflow::Tensor* tensor, FuncOp session_init_func,
                        OpBuilder builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.cc", "InitializeVariable");

  tensorflow::StatusOr<ElementsAttr> tensor_attr_or =
      tensorflow::ConvertTensor(*tensor, &builder);
  assert(tensor_attr_or.ok() && "Expect valid tensor");
  ElementsAttr tensor_attr = tensor_attr_or.ValueOrDie();

  builder.setInsertionPointToStart(&session_init_func.getBlocks().front());
  auto var_handle_op_in_init = var_handle_op->clone();
  builder.insert(var_handle_op_in_init);
  auto const_op = builder.create<mlir::arith::ConstantOp>(
      session_init_func.getLoc(), tensor_attr.getType(), tensor_attr);

  builder.create<TF::AssignVariableOp>(
      session_init_func.getLoc(), llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{var_handle_op_in_init->getResult(0),
                                  const_op.getResult()});
}

constexpr char kTfSavedModelExportedNameAttr[] =
    "tf_saved_model.exported_names";

FuncOp CreateSessionInitFunc(ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.cc", "CreateSessionInitFunc");

  constexpr char kSessionInitFuncName[] = "SessionInitializerFunction";

  mlir::OpBuilder builder(module.getBodyRegion());
  auto func_type =
      FunctionType::get(module.getContext(), /*inputs=*/{}, /*results=*/{});
  auto func =
      builder.create<FuncOp>(module->getLoc(), kSessionInitFuncName, func_type);
  func->setAttr(kTfSavedModelExportedNameAttr,
                builder.getStrArrayAttr({kSessionInitFuncName}));
  func.setVisibility(mlir::func::FuncOp::Visibility::Public);
  auto func_builder = OpBuilder::atBlockBegin(func.addEntryBlock());
  func_builder.create<mlir::func::ReturnOp>(func.getLoc());
  // In cases where there is a session initializer op with empty initializer,
  // replace the session initializer with the new one that points to the session
  // initializer func.
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  auto new_session_init_op =
      builder.create<tf_saved_model::SessionInitializerOp>(
          module->getLoc(), builder.getArrayAttr(SymbolRefAttr::get(
                                builder.getContext(), kSessionInitFuncName)));
  if (session_init_op) {
    session_init_op->replaceAllUsesWith(new_session_init_op);
    session_init_op->erase();
  }
  return func;
}

FuncOp GetOrCreateSessionInitFunc(ModuleOp module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.cc", "GetOrCreateSessionInitFunc");

  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  if (!session_init_op) return CreateSessionInitFunc(module);

  SymbolTable symbol_table(module);
  if (!session_init_op.initializers().empty()) {
    FuncOp init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        session_init_op.initializers()[0].cast<FlatSymbolRefAttr>().getValue());
    return init_func_op;
  }
  return CreateSessionInitFunc(module);
}

}  // namespace

LogicalResult InitializeVariablesInSessionInitializer(
    ModuleOp module, tensorflow::Session* session) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSinitialize_variables_in_session_initDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.cc", "InitializeVariablesInSessionInitializer");

  const tensorflow::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok()) {
    module->emitError("failed to fetch device manager: " +
                      status.error_message());
    return failure();
  }

  // Fetch all VarHandleOp.
  llvm::StringSet<> variable_names;
  llvm::SmallVector<TF::VarHandleOp, 4> var_ops;
  for (auto func_op : module.getOps<FuncOp>()) {
    for (auto var_handle_op : func_op.getOps<TF::VarHandleOp>()) {
      auto variable_name = GetVariableName(var_handle_op);
      if (variable_names.count(variable_name)) continue;
      var_ops.emplace_back(var_handle_op);
      variable_names.insert(variable_name);
    }
  }

  // Get resources from Session.
  auto resource_tensors_or = GetResourcesFromSession(var_ops, session);
  if (!resource_tensors_or.ok()) {
    module->emitError(resource_tensors_or.status().message().data());
    return failure();
  }

  auto session_init_func = GetOrCreateSessionInitFunc(module);
  OpBuilder builder(session_init_func.getContext());

  for (auto var_and_tensor : llvm::zip(var_ops, resource_tensors_or.value())) {
    auto& var_op = std::get<0>(var_and_tensor);
    auto& resource_tensor = std::get<1>(var_and_tensor);
    if (resource_tensor.dtype() != tensorflow::DT_RESOURCE) {
      InitializeVariable(var_op, &resource_tensor, session_init_func, builder);
      continue;
    }

    auto handle = resource_tensor.scalar<tensorflow::ResourceHandle>()();
    auto* var_ptr = GetVariableFromSession(var_op, handle.device(), mgr);
    if (!var_ptr) {
      // If no value in session, then just skip this variable.
      // This can happen if the variable is not saved in checkpoint.
      // For example, when the variable is created on every call.
      continue;
    }
    tensorflow::core::RefCountPtr<tensorflow::Var> var(var_ptr);
    auto* tensor = var_ptr->tensor();

    InitializeVariable(var_op, tensor, session_init_func, builder);
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
