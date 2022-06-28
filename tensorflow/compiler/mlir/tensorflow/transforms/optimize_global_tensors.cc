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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This pass optimizes tf_saved_model.global_tensor ops.

#include <cstddef>
#include <map>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

namespace mlir {
namespace tf_saved_model {
namespace {
struct OptimizeGlobalTensorsPass
    : public OptimizeGlobalTensorsPassBase<OptimizeGlobalTensorsPass> {
  void runOnOperation() override;
};

// A global tensor is bound to arguments of multiple funcs.
// This struct tracks which funcs (and which argument to that func) the global
// tensor is bound to.
struct GlobalTensorUse {
  mutable FuncOp func;
  size_t arg_index;
};

using GlobalTensorUsesMap =
    std::map<GlobalTensorOp, std::vector<GlobalTensorUse>>;

bool IsImmutable(GlobalTensorOp global_tensor,
                 ArrayRef<GlobalTensorUse> global_tensor_uses,
                 const TF::ResourceAnalyzer& resource_analyzer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_0(mht_0_v, 231, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "IsImmutable");

  // Global tensor is already known to be immutable.
  if (!global_tensor.is_mutable()) {
    return false;
  }
  // An exported global tensor that is not already known to be immutable might
  // be externally mutated.
  if (IsExported(global_tensor)) {
    return false;
  }

  // A global tensor is immutable if the resource analyzer deems it so.
  for (auto& global_tensor_use : global_tensor_uses) {
    auto arg = global_tensor_use.func.getArgument(global_tensor_use.arg_index);
    if (resource_analyzer.IsPotentiallyWritten(arg)) {
      return false;
    }
  }
  return true;
}

GlobalTensorUsesMap CreateGlobalTensorUsesMap(ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_1(mht_1_v, 255, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "CreateGlobalTensorUsesMap");

  GlobalTensorUsesMap global_tensor_uses;

  SymbolTable symbol_table(module);
  for (auto func : module.getOps<FuncOp>()) {
    for (size_t i = 0, e = func.getNumArguments(); i < e; i++) {
      auto sym =
          func.getArgAttrOfType<SymbolRefAttr>(i, "tf_saved_model.bound_input");
      if (!sym) {
        continue;
      }
      auto global_tensor = symbol_table.lookup<GlobalTensorOp>(
          sym.cast<FlatSymbolRefAttr>().getValue());
      if (!global_tensor) {
        continue;
      }
      global_tensor_uses[global_tensor].push_back({func, i});
    }
  }

  return global_tensor_uses;
}

// Removes `is_mutable` attribute from tf_saved_model.global_tensor ops where we
// can prove it is safe to do so.
void MarkGlobalTensorsImmutable(
    ModuleOp module, const GlobalTensorUsesMap& global_tensor_uses_map,
    const TF::ResourceAnalyzer& resource_analyzer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_2(mht_2_v, 285, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "MarkGlobalTensorsImmutable");

  for (const auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (IsImmutable(global_tensor, global_tensor_uses, resource_analyzer)) {
      global_tensor->removeAttr("is_mutable");
    }
  }
}

void EraseUnusedGlobalTensors(ModuleOp module,
                              const GlobalTensorUsesMap& global_tensor_uses) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_3(mht_3_v, 299, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "EraseUnusedGlobalTensors");

  for (auto global_tensor :
       llvm::make_early_inc_range(module.getOps<GlobalTensorOp>())) {
    // If the tensor is exported, then it is used.
    if (IsExported(global_tensor)) {
      continue;
    }
    // If the tensor is bound to an argument, then it is used.
    if (global_tensor_uses.find(global_tensor) != global_tensor_uses.end()) {
      continue;
    }
    // Erase it.
    global_tensor.erase();
  }
}

void EraseUnusedBoundInputs(ModuleOp module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_4(mht_4_v, 318, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "EraseUnusedBoundInputs");

  for (auto func : module.getOps<FuncOp>()) {
    llvm::BitVector args_to_erase(func.getNumArguments());
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      if (func.getArgAttr(i, "tf_saved_model.bound_input") &&
          func.getArgument(i).use_empty()) {
        args_to_erase.set(i);
      }
    }
    func.eraseArguments(args_to_erase);
  }
}

void OptimizeGlobalTensorsPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSoptimize_global_tensorsDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc", "OptimizeGlobalTensorsPass::runOnOperation");

  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) {
    return;
  }

  EraseUnusedBoundInputs(module);

  TF::ResourceAnalyzer resource_analyzer(module);

  GlobalTensorUsesMap global_tensor_uses = CreateGlobalTensorUsesMap(module);

  MarkGlobalTensorsImmutable(module, global_tensor_uses, resource_analyzer);

  EraseUnusedGlobalTensors(module, global_tensor_uses);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeGlobalTensorsPass() {
  return std::make_unique<OptimizeGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
