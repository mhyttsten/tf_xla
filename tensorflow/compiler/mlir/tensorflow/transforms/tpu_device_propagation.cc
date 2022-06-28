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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc() {
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

#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// Checks if a function only contains a tf_executor.graph.
bool IsSupportedGraph(FuncOp func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "IsSupportedGraph");

  if (!llvm::hasSingleElement(func)) return false;

  Block& block = func.front();
  if (!llvm::hasSingleElement(block.without_terminator())) return false;

  auto graph = llvm::dyn_cast<tf_executor::GraphOp>(block.front());
  if (!graph) return false;

  Operation* terminator = block.getTerminator();
  if (graph.getNumResults() != terminator->getNumOperands()) return false;
  for (auto result : llvm::zip(graph.results(), terminator->getOperands()))
    if (std::get<0>(result) != std::get<1>(result)) return false;

  return true;
}

// Checks if an operation of the tf_executor dialect can have TPU devices
// propagated through.
bool IsSupportedExecutorOp(Operation& op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "IsSupportedExecutorOp");

  auto ops_have_same_device = [](Operation* lhs, Operation* rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "lambda");

    auto lhs_device_attr = lhs->getAttrOfType<StringAttr>(kDeviceAttr);
    auto rhs_device_attr = rhs->getAttrOfType<StringAttr>(kDeviceAttr);
    return (!lhs_device_attr && !rhs_device_attr) ||
           (lhs_device_attr && rhs_device_attr &&
            lhs_device_attr.getValue() == rhs_device_attr.getValue());
  };

  // Check if tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink
  // pair has matching devices or no devices.
  if (auto source = llvm::dyn_cast<tf_executor::NextIterationSourceOp>(op)) {
    return ops_have_same_device(source, source.GetSink());
  } else if (auto sink = llvm::dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    return ops_have_same_device(sink.GetSource(), sink);
  }

  return llvm::isa<tf_executor::EnterOp, tf_executor::ExitOp,
                   tf_executor::IslandOp, tf_executor::MergeOp,
                   tf_executor::SwitchOp>(op);
}

// Assigns all data results to a specified device.
void PopulateDeviceForOpResults(
    Operation& op, llvm::StringRef device,
    llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "PopulateDeviceForOpResults");

  Operation* op_to_update = &op;
  // Use tf_executor.island op if present as non v1 control flow op results are
  // forwarded by a parent tf_executor.island op.
  if (llvm::isa<tf_executor::IslandOp>(op_to_update->getParentOp()))
    op_to_update = op_to_update->getParentOp();

  for (Value result : op_to_update->getResults()) {
    if (result.getType().isa<tf_executor::TokenType>()) continue;
    if (result.getType().isa<tf_executor::ControlType>()) break;

    value_to_device.insert({result, device});
  }
}

// Checks if an operation can have TPU devices propagated through.
bool IsSupportedOpToSetDevice(Operation& op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "IsSupportedOpToSetDevice");

  return IsSupportedExecutorOp(op) ||
         isa<TF::IdentityOp, TF::IdentityNOp, TF::ShapeOp>(op);
}

// Finds nonconflicting TPU device for an operation from its operands. If an
// operand has no device or a non TPU device, or if there are conflicting
// devices, and empty StringRef will be returned. Control dependencies,
// NextIteration.Source -> NextIteration.Sink token dependencies, and
// LoopCond -> Switch data dependencies are ignored.
llvm::StringRef FindDeviceFromOperands(
    Operation& op,
    const llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_5(mht_5_v, 300, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "FindDeviceFromOperands");

  llvm::StringRef new_device;
  const bool is_switch = llvm::isa<tf_executor::SwitchOp>(op);
  for (Value operand : op.getOperands()) {
    if (operand.getType().isa<tf_executor::TokenType>()) continue;
    if (operand.getType().isa<tf_executor::ControlType>()) break;

    if (is_switch &&
        llvm::isa_and_nonnull<tf_executor::LoopCondOp>(operand.getDefiningOp()))
      continue;

    auto it = value_to_device.find(operand);
    if (it == value_to_device.end()) return llvm::StringRef();

    if (new_device.empty()) {
      new_device = it->getSecond();
      continue;
    }

    if (new_device != it->getSecond()) return llvm::StringRef();
  }

  return new_device;
}

// Propagates devices from function arguments.
void PropagateDevicesFromArguments(
    FuncOp func, llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_6(mht_6_v, 330, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "PropagateDevicesFromArguments");

  for (BlockArgument& arg : func.getArguments()) {
    auto arg_device_attr =
        func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), kFuncDeviceAttr);
    if (!arg_device_attr || arg_device_attr.getValue().empty() ||
        !tensorflow::IsTPUDevice(arg_device_attr.getValue()))
      continue;
    value_to_device.insert({arg, arg_device_attr.getValue()});
  }
}

// Propagates devices from operation operands to results. Updating the device of
// a tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink will result
// in multiple passes over the tf_executor.graph to propagate devices in loops.
void PropagateDevicesInGraph(
    tf_executor::GraphOp graph,
    llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_7(mht_7_v, 349, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "PropagateDevicesInGraph");

  auto ops = graph.GetBody().without_terminator();

  bool updated_next_iteration = false;
  do {
    updated_next_iteration = false;
    for (Operation& op : ops) {
      if (!IsSupportedExecutorOp(op)) continue;

      Operation* op_to_update = &op;
      // Unpack inner op of tf_executor.island.
      if (auto island_op =
              llvm::dyn_cast<tf_executor::IslandOp>(op_to_update)) {
        if (!island_op.WrapsSingleOp()) continue;
        op_to_update = &island_op.GetBody().front();
      }

      // If op already has a TPU device set, simply propagate its device.
      auto device_attr = op_to_update->getAttrOfType<StringAttr>(kDeviceAttr);
      const bool has_device = device_attr && !device_attr.getValue().empty();
      if (has_device && tensorflow::IsTPUDevice(device_attr.getValue())) {
        PopulateDeviceForOpResults(*op_to_update, device_attr.getValue(),
                                   value_to_device);
        continue;
      }

      // Op has an unsupported device.
      if (has_device) continue;

      if (!IsSupportedOpToSetDevice(*op_to_update)) continue;

      llvm::StringRef new_device =
          FindDeviceFromOperands(*op_to_update, value_to_device);
      if (new_device.empty()) continue;

      auto new_device_attr =
          mlir::StringAttr::get(op_to_update->getContext(), new_device);
      op_to_update->setAttr(kDeviceAttr, new_device_attr);
      PopulateDeviceForOpResults(*op_to_update, new_device_attr.getValue(),
                                 value_to_device);

      if (auto sink =
              llvm::dyn_cast<tf_executor::NextIterationSinkOp>(op_to_update)) {
        auto source = sink.GetSource();
        source->setAttr(kDeviceAttr, new_device_attr);
        PopulateDeviceForOpResults(*source, new_device_attr.getValue(),
                                   value_to_device);
        updated_next_iteration = true;
      }
    }
  } while (updated_next_iteration);
}

// Propagates devices to function results.
void PropagateDevicesToResults(
    FuncOp func, tf_executor::FetchOp fetch,
    const llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_8(mht_8_v, 408, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "PropagateDevicesToResults");

  for (OpOperand& operand : fetch.getOperation()->getOpOperands()) {
    if (operand.get().getType().isa<tf_executor::ControlType>()) break;
    auto it = value_to_device.find(operand.get());
    if (it != value_to_device.end()) {
      auto device_attr = func.getResultAttrOfType<StringAttr>(
          operand.getOperandNumber(), kFuncDeviceAttr);
      if (device_attr && !device_attr.getValue().empty()) continue;
      func.setResultAttr(operand.getOperandNumber(), kFuncDeviceAttr,
                         StringAttr::get(func.getContext(), it->getSecond()));
    }
  }
}

struct TPUDevicePropagation
    : public TF::TPUDevicePropagationPassBase<TPUDevicePropagation> {
  void runOnOperation() override;
};

void TPUDevicePropagation::runOnOperation() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_device_propagationDTcc mht_9(mht_9_v, 430, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_device_propagation.cc", "TPUDevicePropagation::runOnOperation");

  FuncOp func = getOperation();
  if (!IsSupportedGraph(func)) return;

  llvm::DenseMap<Value, llvm::StringRef> value_to_device;
  PropagateDevicesFromArguments(func, value_to_device);
  auto graph = llvm::cast<tf_executor::GraphOp>(func.front().front());
  PropagateDevicesInGraph(graph, value_to_device);
  PropagateDevicesToResults(func, graph.GetFetch(), value_to_device);
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTPUDevicePropagationPass() {
  return std::make_unique<TPUDevicePropagation>();
}

}  // namespace TFTPU
}  // namespace mlir
