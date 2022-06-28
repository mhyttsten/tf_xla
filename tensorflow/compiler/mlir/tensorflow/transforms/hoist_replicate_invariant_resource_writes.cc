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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc() {
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "tf-hoist-replicate-invariant-resource-writes"

namespace mlir {
namespace TF {

namespace {

struct HoistReplicateInvariantResourceWritesPass
    : public TF::HoistReplicateInvariantResourceWritesPassBase<
          HoistReplicateInvariantResourceWritesPass> {
  void runOnOperation() override;
};

// TODO(prakalps): This is a common utility and other passes use something
// similar. Move to common utils.
bool IsResourceType(Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/tensorflow/transforms/hoist_replicate_invariant_resource_writes.cc", "IsResourceType");

  return type.isa<TF::ResourceType>() ||
         (type.isa<TensorType>() &&
          type.cast<TensorType>().getElementType().isa<TF::ResourceType>());
}

SmallVector<Value> GetAccessedResources(Operation& op) {
  SmallVector<Value, 4> accessed_resources;
  for (auto operand : op.getOperands()) {
    if (!IsResourceType(operand.getType())) continue;
    accessed_resources.push_back(operand);
  }
  return std::move(accessed_resources);
}

// Lifts the tail writes outside of tf_device.replicate. The written value is
// added to the values returned by tf_device.replicate op. Modify the assign
// variable ops to use the value from first replica.
void MoveTailWritesAfterReplicate(
    tf_device::ReplicateOp replicate_op,
    llvm::ArrayRef<TF::AssignVariableOp> tail_assign_variable_ops) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/tensorflow/transforms/hoist_replicate_invariant_resource_writes.cc", "MoveTailWritesAfterReplicate");

  const auto num_replicas = replicate_op.n();
  auto return_op = llvm::dyn_cast<tf_device::ReturnOp>(
      replicate_op.getRegion().front().getTerminator());

  // Get the new result types.
  // TODO(prakalps): Do not add a value to returned values if it is already
  // returned.
  auto new_result_types = llvm::to_vector<4>(replicate_op->getResultTypes());
  for (auto assign : tail_assign_variable_ops) {
    return_op->insertOperands(return_op->getNumOperands(), assign.value());
    new_result_types.insert(new_result_types.end(), num_replicas,
                            assign.value().getType());
  }

  OpBuilder builder(replicate_op);
  // Clone this old replicate op but with new result types.
  auto new_replicate_op = builder.create<tf_device::ReplicateOp>(
      replicate_op->getLoc(), new_result_types, replicate_op->getOperands(),
      replicate_op->getAttrs());

  // Move region to the new op.
  new_replicate_op.getRegion().takeBody(replicate_op.getRegion());

  // Replace all old uses with new op results.
  int old_num_results = replicate_op->getNumResults();
  replicate_op->replaceAllUsesWith(
      new_replicate_op->getResults().take_front(old_num_results));

  // Move assign ops after replicate and use the output of first replica.
  for (auto indexed_assign : llvm::enumerate(tail_assign_variable_ops)) {
    auto assign_op = indexed_assign.value();
    auto index = indexed_assign.index();
    assign_op->moveAfter(new_replicate_op);
    assign_op->setOperand(
        1, new_replicate_op->getResult(old_num_results + num_replicas * index));
  }
  replicate_op->erase();
}

// Looks for AssignVariable ops from the end of the tf_device.replicate op. It
// returns all the last writes to replicate invariant resource variables
// (resource handles defined outside the tf_device.replicate op).
SmallVector<TF::AssignVariableOp> GetTailWritesToReplicateInvariantResourceVars(
    tf_device::ReplicateOp replicate_op) {
  SmallVector<TF::AssignVariableOp, 16> tail_assign_variable_ops;
  llvm::SmallDenseSet<Value, 16> visited_resources;
  for (auto& op :
       llvm::reverse(replicate_op.getRegion().front().getOperations())) {
    SmallVector<Value> op_accessed_resources = GetAccessedResources(op);
    if (op_accessed_resources.empty()) continue;

    if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(op)) {
      Value resource_var = assign.resource();
      if (visited_resources.contains(resource_var) ||
          !resource_var.getParentRegion()->isProperAncestor(
              &replicate_op.getRegion()))
        continue;
      tail_assign_variable_ops.push_back(assign);
    }

    for (Value resource : op_accessed_resources)
      visited_resources.insert(resource);
  }
  return std::move(tail_assign_variable_ops);
}

void HoistReplicateInvariantResourceWritesPass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShoist_replicate_invariant_resource_writesDTcc mht_2(mht_2_v, 302, "", "./tensorflow/compiler/mlir/tensorflow/transforms/hoist_replicate_invariant_resource_writes.cc", "HoistReplicateInvariantResourceWritesPass::runOnOperation");

  SmallVector<tf_device::ReplicateOp, 2> replicate_ops;
  getOperation().walk([&](tf_device::ReplicateOp replicate_op) {
    replicate_ops.push_back(replicate_op);
  });
  for (auto replicate_op : replicate_ops) {
    SmallVector<TF::AssignVariableOp> tail_writes =
        GetTailWritesToReplicateInvariantResourceVars(replicate_op);

    if (tail_writes.empty()) continue;
    MoveTailWritesAfterReplicate(replicate_op, tail_writes);
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateHoistReplicateInvariantResourceWritesPass() {
  return std::make_unique<HoistReplicateInvariantResourceWritesPass>();
}

}  // namespace TF
}  // namespace mlir
