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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc() {
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

// This pass clusters the TensorFlow ops by host. The program generated by this
// pass will have one function per host where all operations in the same
// function are placed on the same host. Each result of the per-host function
// will have a "tf.device" attribute which specifies the device assignment of
// the result.
//
// The pass currently assumes that there is no circular dependency among the
// per-host functions. For example, if there exists an operation placed on
// host_A that consumes the result of an operation placed on host_B, then there
// does not exist any operation placed on host_B that conumes any result of any
// operation placed on host_A.

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TF {
namespace {

using DeviceNameUtils = ::tensorflow::DeviceNameUtils;
using ParsedName = ::tensorflow::DeviceNameUtils::ParsedName;

constexpr const char *kHostAttr = "host";
constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFDeviceAttr = "tf.device";
// TODO(donglin): Handle the case where the address of localhost is different
// from /job:localhost/replica:0/task:0.
constexpr const char *kLocalhost = "/job:localhost/replica:0/task:0";
constexpr const char *kErrorMessage =
    "The operation that uses the operand is on a different host than the "
    "operation that defines the op. This pass does not support cross-host data "
    "transfer yet";

// The host address is identified by the job/replicate/task in the device name.
std::string GetHost(llvm::StringRef device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "GetHost");

  ParsedName parsed_name;
  DeviceNameUtils::ParseFullName(device.str(), &parsed_name);
  std::string result = DeviceNameUtils::ParsedNameToString(
      DeviceNameUtils::AddressSpace(parsed_name));
  return result.empty() ? kLocalhost : result;
}

std::string GetHost(Operation *op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "GetHost");

  std::string device = "";
  if (StringAttr attr = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
    device = attr.getValue().str();
  }
  return GetHost(device);
}

// The device is considered to be on the localhost iff one of the following is
// true:
// 1) None of the job/replica/task is specified in the device name.
// 2) The job/replica/task in the device name are explicitly specified as
//    /job:localhost/replica:0/task:0.
bool IsOnLocalHost(llvm::StringRef device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_2(mht_2_v, 254, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "IsOnLocalHost");

  std::string host = GetHost(device);
  return host == kLocalhost;
}

// This structure contains the metadata of the per-host function. All operations
// in this function should be on the same host.
struct FunctionMetadata {
  // The original function name before partition.
  llvm::StringRef original_name;
  // The insertion point of partition functions.
  Block::iterator insertion_point;
  // The partitioned function name.
  llvm::StringRef partition_name;
  // The input values of the function.
  llvm::SmallVector<Value, 4> inputs;
  // The result values of the function.
  llvm::SmallVector<Value, 4> results;
  // The devices of the input values. It should have the same size as inputs.
  llvm::SmallVector<std::string, 4> input_devices;
  // The devices of the result values. It should have the same size as results.
  llvm::SmallVector<std::string, 4> result_devices;
  // The operations to be included in the body of the function.
  llvm::SmallVector<Operation *, 4> ops;

  FuncOp partition_op;
};

// Returns a map that maps the host address to the metadata of the function
// for that remote host. The metadata of the function specifies the input
// values, result values, result devices and the operations to be included in
// the function body.
llvm::Optional<llvm::StringMap<FunctionMetadata>> GetFunctionMetadatas(
    FuncOp func_op) {
  llvm::StringMap<FunctionMetadata> metadatas;
  WalkResult result = func_op.getBody().walk([&](Operation *op) {
    std::string op_host = GetHost(op);
    FunctionMetadata &func_metadata = metadatas[op_host];
    func_metadata.original_name = func_op.getName();
    func_metadata.insertion_point = ++Block::iterator(func_op);
    func_metadata.ops.push_back(op);

    for (Value value : op->getOperands()) {
      std::string value_device = "";

      // If the value is defined as an argument of the func_op, adds it to
      // the argument list of the function that uses this op.
      if (BlockArgument block_arg = value.dyn_cast<BlockArgument>()) {
        if (StringAttr attr = func_op.getArgAttrOfType<StringAttr>(
                block_arg.getArgNumber(), kTFDeviceAttr)) {
          value_device = attr.getValue().str();
        }

        if (GetHost(value_device) != op_host) {
          op->emitOpError() << kErrorMessage;
          return WalkResult::interrupt();
        }

        if (llvm::find(func_metadata.inputs, value) ==
            func_metadata.inputs.end()) {
          func_metadata.inputs.push_back(value);
          func_metadata.input_devices.push_back(value_device);
        }
        continue;
      }

      Operation *defining_op = value.getDefiningOp();
      std::string defining_op_host = GetHost(defining_op);
      FunctionMetadata &defining_func_metadata = metadatas[defining_op_host];

      if (StringAttr attr =
              defining_op->getAttrOfType<StringAttr>(kDeviceAttr)) {
        value_device = attr.getValue().str();
      }

      // If the value is used as an operand of the terminator op, adds it to
      // the result list of function that defines this op.
      if (op->hasTrait<OpTrait::IsTerminator>()) {
        if (llvm::find(defining_func_metadata.results, value) ==
            defining_func_metadata.results.end()) {
          defining_func_metadata.results.push_back(value);
          defining_func_metadata.result_devices.push_back(value_device);
        }
        continue;
      }

      if (defining_op_host != op_host) {
        op->emitOpError() << kErrorMessage;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return llvm::None;

  return metadatas;
}

// Creates functions in the given module using the given FunctionMetadatas.
void CreateFunctions(ModuleOp module_op,
                     llvm::StringMap<FunctionMetadata> &metadatas) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_3(mht_3_v, 358, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "CreateFunctions");

  MLIRContext *context = module_op.getContext();
  SymbolTable symbol_table(module_op);
  for (auto &iter : metadatas) {
    llvm::StringRef host = iter.first();
    FunctionMetadata &metadata = iter.second;

    // Do not create any new function for the operations on the localhost.
    if (IsOnLocalHost(host)) continue;

    llvm::SmallVector<mlir::Type, 4> input_types;
    llvm::SmallVector<mlir::Type, 4> result_types;
    for (Value input : metadata.inputs) {
      input_types.push_back(input.getType());
    }
    for (Value result : metadata.results) {
      result_types.push_back(result.getType());
    }

    // Replaces ':' and '/' with '_' in the host name and uses the resulting
    // string as the function name.
    std::string func_name =
        absl::StrCat(iter.second.original_name.str(), ":", host.str());
    std::replace(func_name.begin(), func_name.end(), ':', '_');
    std::replace(func_name.begin(), func_name.end(), '/', '_');

    FunctionType func_type =
        FunctionType::get(context, input_types, result_types);
    Location loc = metadata.ops.front()->getLoc();
    FuncOp func_op = FuncOp::create(loc, func_name, func_type);
    // Sets the device attribute for every input and every result of the
    // function.
    for (int i : llvm::seq<int>(0, metadata.input_devices.size())) {
      func_op.setArgAttr(i, kTFDeviceAttr,
                         StringAttr::get(context, metadata.input_devices[i]));
    }
    for (int i : llvm::seq<int>(0, metadata.result_devices.size())) {
      func_op.setResultAttr(
          i, kTFDeviceAttr,
          StringAttr::get(context, metadata.result_devices[i]));
    }

    func_op->setAttr(kHostAttr, StringAttr::get(context, host));
    func_op.setPublic();
    Block *block = func_op.addEntryBlock();

    // Clones and moves the operations into the function's body. And the cloned
    // operation should use the arguments of the newly created func_op as
    // appropriate.
    OpBuilder builder(block, block->end());
    BlockAndValueMapping mapping;
    for (int i : llvm::seq<int>(0, metadata.inputs.size())) {
      Value original_value = metadata.inputs[i];
      Value new_value = func_op.getArgument(i);
      mapping.map(original_value, new_value);
    }
    for (Operation *op : metadata.ops) {
      builder.clone(*op, mapping);
    }
    // Creates the ReturnOp so that the per-host function returns the
    // correct values of the cloned operations.
    llvm::SmallVector<Value, 4> results_after_mapping;
    for (Value result : metadata.results) {
      results_after_mapping.push_back(mapping.lookupOrDefault(result));
    }
    builder.create<func::ReturnOp>(loc, results_after_mapping);
    symbol_table.insert(func_op, metadata.insertion_point++);
    // Record the actual name. The symbol table might rename the FuncOp if there
    // is name collision.
    metadata.partition_name = func_op.getName();
  }
}

// Creates a tf_device.remote_run call for every remote function. And replaces
// usages of the results of the original operations with the results of the
// tf_device.remote_run calls.
void CreateRemoteRunCalls(MLIRContext *context,
                          const llvm::StringMap<FunctionMetadata> &metadatas) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_4(mht_4_v, 438, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "CreateRemoteRunCalls");

  BlockAndValueMapping mapping;
  for (auto &iter : metadatas) {
    llvm::StringRef host = iter.first();
    const FunctionMetadata &metadata = iter.second;

    // Do not create tf_device.remote_run call for the operations already placed
    // on the localhost.
    if (IsOnLocalHost(host)) continue;

    // Creates the tf_device.remote_run operation.
    OpBuilder builder(metadata.ops.back());
    llvm::SmallVector<Type, 4> result_types;
    for (Value result : metadata.results) {
      result_types.push_back(result.getType());
    }
    Location loc = metadata.ops.front()->getLoc();
    llvm::SmallVector<Value, 4> inputs_after_mapping;
    for (Value input : metadata.inputs) {
      inputs_after_mapping.push_back(mapping.lookupOrDefault(input));
    }

    tf_device::RemoteRunOp remote_run_op =
        builder.create<tf_device::RemoteRunOp>(loc, result_types, host,
                                               metadata.partition_name,
                                               inputs_after_mapping);
    // Clones the tf_device.remote_run operation to replace its callee args with
    // the results of the other tf_device.remote_run operations using the
    // `mapping` as appropriate.
    Operation *cloned_remote_run_op =
        builder.clone(*remote_run_op.getOperation(), mapping);
    remote_run_op.erase();

    // Replaces usages of the results of the original operations with the
    // results of the tf_device.remote_run operations.
    for (int i : llvm::seq<int>(0, metadata.results.size())) {
      Value original_value = metadata.results[i];
      Value new_value = cloned_remote_run_op->getResult(i);
      original_value.replaceAllUsesWith(new_value);
      mapping.map(original_value, new_value);
    }
  }
}

class ClusterTFOpsByHostPass
    : public ClusterTFOpsByHostPassBase<ClusterTFOpsByHostPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_tf_ops_passDTcc mht_5(mht_5_v, 487, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_tf_ops_pass.cc", "runOnOperation");

    MLIRContext *context = &getContext();
    ModuleOp module_op = getOperation();
    SmallVector<FuncOp, 4> original_func;
    for (auto func_op : module_op.getOps<FuncOp>()) {
      original_func.push_back(func_op);
    }
    for (auto func_op : original_func) {
      llvm::Optional<llvm::StringMap<FunctionMetadata>> metadatas =
          GetFunctionMetadatas(func_op);
      if (!metadatas) {
        signalPassFailure();
        return;
      }

      CreateFunctions(module_op, *metadatas);
      CreateRemoteRunCalls(context, *metadatas);

      // Erases the original operations which have been cloned in the remote
      // functions.
      for (auto &iter : *metadatas) {
        llvm::StringRef host = iter.first();
        FunctionMetadata &metadata = iter.second;
        // Do not erase operations placed on the localhost.
        if (IsOnLocalHost(host)) continue;

        for (int i = metadata.ops.size() - 1; i >= 0; i--) {
          metadata.ops[i]->erase();
        }
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateClusterTFOpsByHostPass() {
  return std::make_unique<ClusterTFOpsByHostPass>();
}

}  // namespace TF
}  // namespace mlir
