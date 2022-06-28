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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc() {
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

// This pass converts each tfrt_dist.remote_execute_func op into a combination
// of tfrt_dist.register_tfrt_function op and tfrt_dist.remote_execute op. The
// function to be executed in the remote host will be serialized as a string
// attribute of the tfrt_dist.register_tfrt_function op.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/opdefs/kernels.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/test_kernels/opdefs/test_kernels.h"  // from @tf_runtime

namespace tensorflow {

namespace {

constexpr const char* kHost = "host";
constexpr const char* kTFRTDevice = "tfrt.device";

struct DistRemoteRunEncapsulatePass
    : public PassWrapper<DistRemoteRunEncapsulatePass,
                         OperationPass<ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/mlir/tfrt/transforms/remote_run_encapsulate.cc", "getArgument");

    return "tfrt-dist-remote-run-encapsulate";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/tfrt/transforms/remote_run_encapsulate.cc", "getDescription");

    return "This pass looks for a remote_run_func and serialize the callee to "
           "a string attribute attached to a remote_register operation, "
           "followed by a remote_execute invocation.";
  }
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/tfrt/transforms/remote_run_encapsulate.cc", "getDependentDialects");

    registry.insert<tfrt::dist::DistributedDialect>();
  }
};

LogicalResult EncapsulateFuncAndSerialize(FuncOp entry_func,
                                          std::string* serialized_func_module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/mlir/tfrt/transforms/remote_run_encapsulate.cc", "EncapsulateFuncAndSerialize");

  ModuleOp module = entry_func->getParentOfType<ModuleOp>();
  SymbolTable entry_module_table(module);
  SmallVector<FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningOpRef<mlir::ModuleOp> module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()));
  SymbolTable symbol_table(module_for_func.get());

  while (!referenced.empty()) {
    FuncOp func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (symbol_table.lookup<FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    Optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(func);
    assert(uses && "expected to be able to collect symbol uses");
    for (SymbolTable::SymbolUse use : *uses) {
      FuncOp referenced_func = entry_module_table.lookup<FuncOp>(
          use.getSymbolRef().cast<FlatSymbolRefAttr>().getValue());

      // Skip Symbols that do not map to a function.
      if (!referenced_func) continue;

      referenced.emplace_back(referenced_func);
    }

    FuncOp clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      clone.setPublic();
    } else {
      clone.setPrivate();
    }
    symbol_table.insert(clone);
  }

  *serialized_func_module =
      tensorflow::SerializeMlirModule(module_for_func.get());
  return success();
}

void DistRemoteRunEncapsulatePass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremote_run_encapsulateDTcc mht_4(mht_4_v, 298, "", "./tensorflow/compiler/mlir/tfrt/transforms/remote_run_encapsulate.cc", "DistRemoteRunEncapsulatePass::runOnOperation");

  mlir::TF::RuntimeDevices devices;
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  Type chain_type = tfrt::compiler::ChainType::get(&getContext());
  Type remote_object_id_ty = tfrt::dist::RemoteObjectIdType::get(&getContext());
  Type tensor_handle_ty = tfrt::corert::TensorHandleType::get(&getContext());
  module.walk([&](tfrt::dist::RemoteExecuteFuncOp remote_exec_op) {
    FlatSymbolRefAttr callee_sym = remote_exec_op.calleeAttr();
    FuncOp callee = symtab.lookup<FuncOp>(callee_sym.getValue());
    if (!callee) {
      remote_exec_op.emitOpError("callee function ")
          << callee_sym.getValue() << " is not found";
      signalPassFailure();
      return WalkResult::interrupt();
    }
    std::string txt_module;
    if (failed(EncapsulateFuncAndSerialize(callee, &txt_module))) {
      remote_exec_op.emitOpError("failed to serialize the callee function ")
          << callee.getName();
      signalPassFailure();
      return WalkResult::interrupt();
    }
    Location loc = remote_exec_op.getLoc();
    StringAttr callee_name =
        StringAttr::get(&getContext(), callee_sym.getValue());
    OpBuilder builder(remote_exec_op);
    auto register_op = builder.create<tfrt::dist::RegisterTFRTFunctionOp>(
        loc, chain_type, remote_exec_op.in_op_chain(), remote_exec_op.context(),
        remote_exec_op.remote_task(),
        StringAttr::get(&getContext(), txt_module), callee_name);

    // Build the device assignment for the results
    // TODO(tfrt-devs): Define properly MLIR types and operations
    SmallVector<Attribute, 8> result_devices;
    for (const auto& result : llvm::enumerate(remote_exec_op.results())) {
      StringAttr device =
          callee.getResultAttrOfType<StringAttr>(result.index(), kTFRTDevice);
      if (!device) {
        // The result might not have the device attribute if it is added by
        // the tf-to-tfrt pass. Use the first CPU on the remote host as the
        // device of this result.
        DeviceNameUtils::ParsedName parsed_name;
        if (StringAttr host_attr = callee->getAttrOfType<StringAttr>(kHost)) {
          auto host = host_attr.getValue();
          DeviceNameUtils::ParseFullName({host.data(), host.size()},
                                         &parsed_name);
        }
        parsed_name.has_type = true;
        parsed_name.type = "CPU";
        parsed_name.has_id = true;
        parsed_name.id = 0;
        device = StringAttr::get(
            &getContext(), DeviceNameUtils::ParsedNameToString(parsed_name));
      }
      result_devices.push_back(std::move(device));
    }
    // IDEA(donglin): Update the create_remote_execute_spec kernel to use Device
    // object instead of Device string.
    Type remote_spec_ty = tfrt::dist::RemoteExecuteSpecType::get(&getContext());
    auto result_devices_attr = ArrayAttr::get(&getContext(), result_devices);
    auto remote_spec = builder.create<tfrt::dist::CreateRemoteExecuteSpecOp>(
        loc, remote_spec_ty, remote_exec_op.context(), result_devices_attr);
    // If original argument is already tfrt_dist.remote_object_id, use it
    // directly. If it is TensorHandle, insert an op to extract the
    // tfrt_dist.remote_object_id from it. Otherwise, emit an error.
    SmallVector<Value, 4> arguments;
    for (Value value : remote_exec_op.callee_args()) {
      if (value.getType().isa<tfrt::dist::RemoteObjectIdType>()) {
        arguments.push_back(value);
      } else if (value.getType().isa<tfrt::corert::TensorHandleType>()) {
        auto new_op = builder.create<tfrt::dist::GetRemoteObjectIdFromTHOp>(
            loc, remote_object_id_ty, value);
        arguments.push_back(new_op.result());
      } else {
        remote_exec_op.emitOpError(
            "callee argument type should be either "
            "TensorHandle or RemoteObjectId");
        signalPassFailure();
        return WalkResult::interrupt();
      }
    }
    // Result types are 1 chain, followed by `num_th_results + 1`
    // tfrt_dist.remote_object_id results, followed by `num_th_results`
    // corert.tensorhandle results.
    int32_t num_th_results = remote_exec_op.results().size() - 1;
    SmallVector<Type, 8> result_types;
    result_types.push_back(chain_type);
    for (int count : llvm::seq<int>(0, num_th_results + 1)) {
      (void)count;
      result_types.push_back(remote_object_id_ty);
    }
    for (int count : llvm::seq<int>(0, num_th_results)) {
      (void)count;
      result_types.push_back(tensor_handle_ty);
    }
    auto new_remote_exec_th_op = builder.create<tfrt::dist::RemoteExecuteTHOp>(
        loc, result_types, register_op.out_op_chain(), remote_exec_op.context(),
        remote_exec_op.remote_task(), remote_spec, num_th_results,
        callee_name.getValue(), std::move(arguments));
    // The part of the new results to replace the original results are 2 chains,
    // followed `num_th_results` corert.tesnorhandle results from the callee
    // function.
    SmallVector<Value, 4> new_results;
    new_results.push_back(new_remote_exec_th_op.getResult(0));
    new_results.push_back(new_remote_exec_th_op.getResult(1));
    for (int i : llvm::seq<int>(0, num_th_results)) {
      new_results.push_back(
          new_remote_exec_th_op.getResult(i + 2 + num_th_results));
    }
    remote_exec_op.replaceAllUsesWith(new_results);
    remote_exec_op.erase();

    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateDistRemoteRunEncapsulatePass() {
  return std::make_unique<DistRemoteRunEncapsulatePass>();
}

static PassRegistration<DistRemoteRunEncapsulatePass> pass;

}  // namespace tensorflow
