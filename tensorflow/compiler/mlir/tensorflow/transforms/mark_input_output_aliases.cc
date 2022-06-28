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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc() {
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "tf-device-mark-input-output-aliases"

namespace mlir {
namespace TFDevice {

namespace {
struct MarkInputOutputAliasesPass
    : public TF::MarkInputOutputAliasesPassBase<MarkInputOutputAliasesPass> {
  void runOnOperation() override;
};

constexpr char kAliasingAttr[] = "tf.aliasing_output";
constexpr int kUnassigned = -1;

struct AliasInfo {
  AliasInfo() : input_index(kUnassigned), output_index(kUnassigned) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_input_output_aliases.cc", "AliasInfo");
}
  int input_index;
  int output_index;
};

// Idenitfy tf_device.cluster_func input-output alias pairs.
// This is currently conservative, and handles the following simple case:
// ```
// %value = tf.ReadVariableOp(%resource_var)
// %output:N = tf_device.cluster_func(..., /*input index = a*/ %value, ...)
// tf.AssignVariableOp(%resource_var, %output#b) // write output #b to resource
// ```
// where `%value` and `%output#b` have only one use. (a, b) would be added as
// input-output alias pair for `%resource_var`.
//
// TODO(b/184420848): Explore relaxing these constraints.
LogicalResult BuildAliasingInfo(
    tf_device::ClusterFuncOp cluster_func,
    llvm::DenseMap<Value, AliasInfo>& resource_alias_info_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_input_output_aliases.cc", "BuildAliasingInfo");

  for (auto result : cluster_func.getResults()) {
    if (!result.hasOneUse()) continue;
    auto assign_op = llvm::dyn_cast_or_null<TF::AssignVariableOp>(
        result.use_begin()->getOwner());
    if (!assign_op) continue;
    AliasInfo& alias_info = resource_alias_info_map[assign_op.resource()];
    // TODO(b/184420848): We may not need to skip aliasing for entire function
    // in case of multiple assigns.
    if (alias_info.output_index != kUnassigned) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip adding aliasing information because of multiple assigns to "
             "the same resource from tf_device.cluster_func outputs. This can "
             "lead to poor memory management on device.\n");

      return failure();
    }
    alias_info.output_index = result.getResultNumber();
  }

  for (auto& operand : cluster_func->getOpOperands()) {
    auto read_op = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.get().getDefiningOp());
    if (!read_op) continue;
    if (!read_op->hasOneUse()) continue;
    auto it = resource_alias_info_map.find(read_op.resource());
    if (it == resource_alias_info_map.end()) continue;
    AliasInfo& alias_info = it->getSecond();
    // TODO(b/184420848): We may not need to skip aliasing for entire function
    // in case of multiple reads from same resource variable.
    if (alias_info.input_index != kUnassigned) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip adding aliasing information because of multiple reads of "
             "the same resource in tf_device.cluster_func inputs. This can "
             "lead to poor memory management on device.\n");
      return failure();
    }

    alias_info.input_index = operand.getOperandNumber();
  }
  return success();
}

void AddAliasingAttributeToDeviceFunc(
    FuncOp device_func,
    llvm::DenseMap<Value, AliasInfo>& resource_alias_info_map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc mht_2(mht_2_v, 277, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_input_output_aliases.cc", "AddAliasingAttributeToDeviceFunc");

  OpBuilder builder(device_func.getContext());
  for (const auto& resource_alias_entry : resource_alias_info_map) {
    const AliasInfo& alias_info = resource_alias_entry.second;
    if (alias_info.input_index == kUnassigned ||
        alias_info.output_index == kUnassigned)
      continue;
    auto aliasing_attr = device_func.getArgAttrOfType<mlir::IntegerAttr>(
        alias_info.input_index, kAliasingAttr);

    // Set only if aliasing attribute does not exist.
    if (!aliasing_attr) {
      device_func.setArgAttr(
          alias_info.input_index, kAliasingAttr,
          builder.getI64IntegerAttr(alias_info.output_index));
      continue;
    }
    // If aliasing attribute already exists, it must match the new value.
    assert(aliasing_attr.getInt() == alias_info.output_index);
  }
}

void MarkInputOutputAliasesPass::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_input_output_aliasesDTcc mht_3(mht_3_v, 302, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_input_output_aliases.cc", "MarkInputOutputAliasesPass::runOnOperation");

  SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  ModuleOp module = getOperation();
  module.walk([&](tf_device::ClusterFuncOp cluster_func) {
    // Map resource values to pair of input-output indices.
    llvm::DenseMap<Value, AliasInfo> resource_alias_info_map;
    if (failed(BuildAliasingInfo(cluster_func, resource_alias_info_map)) ||
        resource_alias_info_map.empty()) {
      return;
    }

    FlatSymbolRefAttr func_attr = cluster_func.funcAttr();
    FuncOp device_func = module.lookupSymbol<FuncOp>(func_attr.getValue());
    AddAliasingAttributeToDeviceFunc(device_func, resource_alias_info_map);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateMarkInputOutputAliasesPass() {
  return std::make_unique<MarkInputOutputAliasesPass>();
}

}  // namespace TFDevice
}  // namespace mlir
