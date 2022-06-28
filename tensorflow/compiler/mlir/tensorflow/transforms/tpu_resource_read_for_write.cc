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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc() {
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

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {

// A pass that finds TPU clusters with write only resource access and adds an
// associated resource read, so the resource can later be fused into TPUExecute.
namespace {
struct TPUResourceReadForWritePass
    : public TF::TPUResourceReadForWritePassBase<TPUResourceReadForWritePass> {
  void runOnOperation() override;
};

// Helper struct holding a resource value and its associated type.
struct ResourceValueAndSubtype {
  Value resource;
  Type subtype;
};

// Finds resource handle and type for result if result writes to a resource.
ResourceValueAndSubtype GetResourceWriteResult(
    tf_device::ClusterFuncOp cluster_func, Value result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_read_for_write.cc", "GetResourceWriteResult");

  ResourceValueAndSubtype resource;
  if (!result.hasOneUse()) return resource;
  Operation* result_user = *result.getUsers().begin();
  auto assign_var = dyn_cast<TF::AssignVariableOp>(result_user);
  if (!assign_var) return resource;

  auto handle = assign_var.resource();
  // Skip result if cluster writes to the same variable via multiple results.
  for (Operation* handle_user : handle.getUsers()) {
    if (handle_user == assign_var) continue;
    auto assign_var_user = dyn_cast<TF::AssignVariableOp>(handle_user);
    if (!assign_var_user) continue;
    if (assign_var_user.value().getDefiningOp() == cluster_func)
      return resource;
  }

  resource.resource = assign_var.resource();
  resource.subtype = assign_var.value().getType();
  return resource;
}

// Checks if resource is read by TPU cluster.
bool ClusterFuncHasResourceRead(tf_device::ClusterFuncOp cluster_func,
                                Value resource) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_read_for_write.cc", "ClusterFuncHasResourceRead");

  for (Operation* resource_user : resource.getUsers())
    if (auto read = dyn_cast<TF::ReadVariableOp>(resource_user))
      for (Operation* read_user : read.value().getUsers())
        if (read_user == cluster_func) return true;

  return false;
}

void TPUResourceReadForWritePass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_read_for_writeDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_read_for_write.cc", "TPUResourceReadForWritePass::runOnOperation");

  SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func) {
    cluster_funcs.push_back(cluster_func);
  });

  OpBuilder builder(&getContext());
  // Add resource reads for resource writes from TPU cluster where for such
  // resources the TPU cluster does not read from.
  for (tf_device::ClusterFuncOp cluster_func : cluster_funcs) {
    builder.setInsertionPoint(cluster_func);

    SmallVector<Value, 4> read_operands;
    for (Value result : cluster_func.getResults()) {
      // TODO(lyandy): Update pass to use resource alias analysis.
      auto resource_and_type = GetResourceWriteResult(cluster_func, result);
      if (!resource_and_type.resource) continue;
      if (ClusterFuncHasResourceRead(cluster_func, resource_and_type.resource))
        continue;
      auto new_read = builder.create<TF::ReadVariableOp>(
          resource_and_type.resource.getLoc(), resource_and_type.subtype,
          resource_and_type.resource);
      read_operands.push_back(new_read.value());
    }

    if (read_operands.empty()) continue;

    // Update caller and function types with new read operands.
    auto operands = llvm::to_vector<4>(cluster_func.getOperands());
    operands.append(read_operands.begin(), read_operands.end());

    auto loc = cluster_func.getLoc();
    auto new_cluster_func = builder.create<tf_device::ClusterFuncOp>(
        loc, cluster_func.getResultTypes(), operands, cluster_func->getAttrs());
    cluster_func.replaceAllUsesWith(new_cluster_func);
    FuncOp func = cluster_func.getFunc();
    Block& block = func.front();
    for (Value read_operand : read_operands)
      block.addArgument(read_operand.getType(), loc);

    func.setType(FunctionType::get(&getContext(), block.getArgumentTypes(),
                                   func.getCallableResults()));
    cluster_func.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUResourceReadForWritePass() {
  return std::make_unique<TPUResourceReadForWritePass>();
}

}  // namespace TFTPU
}  // namespace mlir
