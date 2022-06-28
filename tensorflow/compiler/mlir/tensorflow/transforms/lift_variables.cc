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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variablesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variablesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variablesDTcc() {
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

using llvm::SmallSet;
using ::tensorflow::Device;
using ::tensorflow::DeviceMgr;
using ::tensorflow::mutex_lock;
using ::tensorflow::ResourceHandle;
using ::tensorflow::Session;
using ::tensorflow::Status;
using ::tensorflow::StatusOr;
using ::tensorflow::Tensor;
using ::tensorflow::Var;

namespace {

constexpr char kResourceNameArgAttr[] = "tf.resource_name";
constexpr char kSavedModelArgAttr[] = "tf_saved_model.bound_input";

LogicalResult LiftVariablesFromSession(
    ModuleOp module, Session* session,
    const SmallSet<StringRef, 4>& resource_names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variablesDTcc mht_0(mht_0_v, 238, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.cc", "LiftVariablesFromSession");

  OpBuilder builder(module.getBodyRegion());

  if (!session) return module.emitOpError() << "no session provided";

  // Read all resource variables from the session.
  std::vector<std::string> variable_names;
  variable_names.reserve(resource_names.size());
  for (StringRef name : resource_names) variable_names.push_back(name.str());

  std::vector<Tensor> resource_tensors;
  Status status = session->Run(
      /*inputs=*/{}, variable_names,
      /*target_node_names=*/{}, &resource_tensors);
  if (!status.ok()) {
    return module.emitOpError()
           << "failed to run the provided session: " << status.error_message();
  }

  const DeviceMgr* device_manager;
  if (!(session->LocalDeviceManager(&device_manager).ok())) {
    return module.emitOpError() << "failed to get local device manager";
  }

  // Read all underlying tensors of the variables from the session.
  std::vector<Tensor> tensors;
  tensors.reserve(resource_tensors.size());
  for (const Tensor& resource_tensor : resource_tensors) {
    if (resource_tensor.dtype() != tensorflow::DT_RESOURCE) {
      tensors.push_back(resource_tensor);
      continue;
    }

    const ResourceHandle& resource_handle =
        resource_tensor.scalar<ResourceHandle>()();

    Device* device;
    if (!(device_manager->LookupDevice(resource_handle.device(), &device)
              .ok())) {
      return module.emitOpError() << "failed to look up device";
    }

    tensorflow::Var* var_ptr;
    if (!(device->resource_manager()
              ->Lookup(resource_handle.container(), resource_handle.name(),
                       &var_ptr)
              .ok())) {
      return module.emitOpError() << "failed to look up resource value";
    }
    tensorflow::core::RefCountPtr<Var> var(var_ptr);

    // The variable tensor is already loaded into corresponding device's
    // resource manager when we load the saved model using LoadSavedModel().
    // Here we just read its value.
    mutex_lock ml(*var->mu());
    tensors.push_back(*var->tensor());
  }

  for (const auto iter : llvm::zip(resource_names, tensors)) {
    const StringRef name = std::get<0>(iter);
    const Tensor& tensor = std::get<1>(iter);

    // Create tensor attribute for this variable.
    StatusOr<ElementsAttr> tensor_attr_or = ConvertTensor(tensor, &builder);
    if (!tensor_attr_or.ok()) {
      return module.emitOpError()
             << "failed to convert tensor (name: " << name.str() << ")";
    }
    ElementsAttr tensor_attr = tensor_attr_or.ValueOrDie();

    builder.create<tf_saved_model::GlobalTensorOp>(
        NameLoc::get(builder.getStringAttr(name.str())),
        builder.getStringAttr(name), tensor_attr,
        TypeAttr::get(tensor_attr.getType()), builder.getUnitAttr());
  }

  return success();
}

}  // namespace

LogicalResult LiftVariables(ModuleOp module, Session* session) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSlift_variablesDTcc mht_1(mht_1_v, 322, "", "./tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.cc", "LiftVariables");

  MLIRContext* context = module.getContext();
  mlir::Builder builder(context);
  StringAttr resource_name_id = builder.getStringAttr(kResourceNameArgAttr);

  SmallSet<StringRef, 4> resource_names;

  for (FuncOp func : module.getOps<FuncOp>()) {
    for (int i = 0, e = func.getNumArguments(); i < e; ++i) {
      auto resource_arg =
          func.getArgAttrOfType<StringAttr>(i, kResourceNameArgAttr);
      if (!resource_arg) continue;

      StringRef resource_name = resource_arg.getValue();
      auto flat_symbol_ref_attr =
          FlatSymbolRefAttr::get(context, resource_name);

      // Add the corresponding `tf_saved_model.bound_input` attribute.
      func.setArgAttr(i, kSavedModelArgAttr, flat_symbol_ref_attr);

      resource_names.insert(flat_symbol_ref_attr.getValue());

      // Remove the existing `tf.resource_name` attribute.
      func.removeArgAttr(i, resource_name_id);
    }
  }

  if (resource_names.empty()) return success();

  if (failed(LiftVariablesFromSession(module, session, resource_names)))
    return failure();

  // Now that we have all global tensors created, we set the corresponding
  // bound_inputs' types correctly.
  SymbolTable symbol_table(module);
  for (auto func : module.getOps<FuncOp>()) {
    for (auto arg : func.getArguments()) {
      unsigned arg_number = arg.getArgNumber();
      auto global_tensor = LookupBoundInputOfType<GlobalTensorOp>(
          func, arg_number, symbol_table);
      if (!global_tensor) continue;

      auto arg_type = arg.getType().cast<RankedTensorType>();
      assert(arg_type.getRank() == 0);
      llvm::ArrayRef<TensorType> underlying_type =
          arg_type.getElementType().cast<TF::ResourceType>().getSubtypes();

      // If the arg type already matches the global_tensor type, we don't need
      // to do anything.
      if (!underlying_type.empty() &&
          underlying_type[0] == global_tensor.type()) {
        assert(underlying_type.size() == 1);
        continue;
      }

      // Otherwise, set this argument's type to the global_tensor's type.
      auto new_arg_type = mlir::RankedTensorType::get(
          /*shape=*/{},
          mlir::TF::ResourceType::get(
              /*subtypes=*/{global_tensor.type().cast<TensorType>()},
              module.getContext()));

      arg.setType(new_arg_type);
    }

    // Update the function type.
    func.setType(mlir::FunctionType::get(module.getContext(),
                                         func.getBody().getArgumentTypes(),
                                         func.getFunctionType().getResults()));
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
