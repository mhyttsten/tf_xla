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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc() {
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

// This pass inserts corert.transfer op to make sure any argument of any op is
// on the same device of the op itself.

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {

namespace {

using DeviceNameUtils = ::tensorflow::DeviceNameUtils;

constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFRTDeviceAttr = "tfrt.device";
// TODO(b/175480458): Do not assign default device once every op in the TF
// dialect has the device attribute.
constexpr const char *kDefaultDevice =
    "/job:localhost/replica:0/task:0/device:CPU:0";

// This method canonicalizes the device name so that we can use string
// comparison to see if two devices are the same. It does the following
// transformations:
// 1) Set device ID to 0 if device ID is not already specified.
// 2) Change the device type to uppercase string.
static std::string CanonicalizeDeviceName(const std::string &device) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "CanonicalizeDeviceName");

  if (device.empty()) return kDefaultDevice;

  DeviceNameUtils::ParsedName parsed_name;
  if (!device.empty() && device.at(0) == '/') {
    DeviceNameUtils::ParseFullName(device, &parsed_name);
  } else {
    DeviceNameUtils::ParseFullName("/device:" + device, &parsed_name);
  }

  if (!parsed_name.has_id) {
    parsed_name.has_id = true;
    parsed_name.id = 0;
  }

  if (parsed_name.type == "cpu")
    parsed_name.type = "CPU";
  else if (parsed_name.type == "gpu")
    parsed_name.type = "GPU";
  else if (parsed_name.type == "tpu")
    parsed_name.type = "TPU";
  return DeviceNameUtils::ParsedNameToString(parsed_name);
}

// Return the device of the given operation.
static std::string GetDevice(Operation *op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "GetDevice");

  std::string device = "";
  if (StringAttr device_attr = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
    device = device_attr.getValue().str();
  } else if (auto execute_op = llvm::dyn_cast<tfrt::corert::ExecuteOp>(op)) {
    SmallVector<std::pair<StringRef, Attribute>, 4> attrs;
    execute_op.getOpAttrs(&attrs);
    for (std::pair<StringRef, Attribute> entry : attrs) {
      if (entry.first == kDeviceAttr && entry.second.isa<StringAttr>()) {
        device = entry.second.cast<StringAttr>().getValue().str();
        break;
      }
    }
  }

  return CanonicalizeDeviceName(device);
}

// Return the device of the given value.
static std::string GetDevice(mlir::Value value, FuncOp parent_func_op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "GetDevice");

  std::string device = "";
  if (BlockArgument block_arg = value.dyn_cast<BlockArgument>()) {
    if (StringAttr device_attr = parent_func_op.getArgAttrOfType<StringAttr>(
            block_arg.getArgNumber(), kTFRTDeviceAttr)) {
      device = device_attr.getValue().str();
    }
  } else {
    device = GetDevice(value.getDefiningOp());
  }

  return CanonicalizeDeviceName(device);
}

struct CrossDeviceTransferPass
    : public PassWrapper<CrossDeviceTransferPass, OperationPass<FuncOp>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "getArgument");

    return "tfrt-cross-device-transfer";
  }

  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_4(mht_4_v, 299, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "getDescription");

    return "This pass inserts corert.transfer op to make sure any argument of "
           "any op is on the same device of the op itself.";
  }
};

void CrossDeviceTransferPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScross_device_transferDTcc mht_5(mht_5_v, 308, "", "./tensorflow/compiler/mlir/tfrt/transforms/cross_device_transfer.cc", "CrossDeviceTransferPass::runOnOperation");

  FuncOp func_op = getOperation();
  llvm::DenseMap<mlir::Value, llvm::StringMap<mlir::Value>>
      transferred_value_by_value_and_device;

  func_op.getBody().walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) return WalkResult::advance();
    // Do not transfer the argument of corert.transfer op.
    if (llvm::isa<tfrt::corert::TransferOp>(op)) return WalkResult::advance();

    OpBuilder builder(op);
    std::string dst_device = GetDevice(op);
    mlir::Type tensor_type_type =
        builder.getType<::tfrt::compiler::TensorTypeType>();
    mlir::Type device_type = builder.getType<::tfrt::compiler::DeviceType>();

    for (mlir::Value arg : op->getOperands()) {
      // Do not transfer non-TensorHandle values.
      if (!arg.getType().isa<tfrt::corert::TensorHandleType>()) continue;

      // Do not transfer the result of corert.transfer op.
      if (OpResult op_result = arg.dyn_cast<OpResult>()) {
        Operation *defining_op = arg.getDefiningOp();
        if (llvm::isa<tfrt::corert::TransferOp>(defining_op)) continue;
      }

      std::string src_device = GetDevice(arg, func_op);

      if (DeviceNameUtils::LocalName(src_device) ==
          DeviceNameUtils::LocalName(dst_device))
        continue;

      // Re-use the value already transferred to the given device.
      llvm::StringMap<mlir::Value> &transferred_value_by_device =
          transferred_value_by_value_and_device[arg];
      auto iter = transferred_value_by_device.find(dst_device);
      if (iter != transferred_value_by_device.end()) {
        op->replaceUsesOfWith(arg, iter->second);
        continue;
      }

      mlir::Value chain_in = func_op.getArgument(0);
      auto get_device_op = builder.create<tfrt::compiler::GetDeviceOp>(
          op->getLoc(), device_type, chain_in, dst_device);
      auto get_tensor_type_op =
          builder.create<tfrt::corert::GetDstTensorTypeOp>(
              op->getLoc(), tensor_type_type, arg, get_device_op.getResult());
      auto transfer_op = builder.create<tfrt::corert::TransferOp>(
          op->getLoc(), arg.getType(), arg, get_device_op.getResult(),
          get_tensor_type_op.getResult());
      mlir::Value new_arg = transfer_op.getResult();
      transferred_value_by_device[dst_device] = new_arg;
      op->replaceUsesOfWith(arg, new_arg);
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateCrossDeviceTransferPass() {
  return std::make_unique<CrossDeviceTransferPass>();
}

static PassRegistration<CrossDeviceTransferPass> pass;

}  // namespace tensorflow
