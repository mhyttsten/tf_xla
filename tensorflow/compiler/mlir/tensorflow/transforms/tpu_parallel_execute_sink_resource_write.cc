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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc() {
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

#include <algorithm>
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {

namespace {

// A pass that moves `tf.AssignVariableOp` into a `tf_device.parallel_execute`
// region if the `tf.AssignVariableOp` is the only consumer of a
// `tf_device.parallel_execute` result. This will allow
// TPUMergeVariablesWithExecute to merge resource writes without special
// handling for `tf_device.parallel_execute`.
struct TPUParallelExecuteSinkResourceWrite
    : public TF::TPUParallelExecuteSinkResourceWritePassBase<
          TPUParallelExecuteSinkResourceWrite> {
  void runOnOperation() override;
};

// Finds an AssignVariableOp that can be moved into the parallel_execute region.
// These AssignVariableOps must be the only consumer of the respective
// parallel_execute result, and the resource handle producer must be from an op
// before or above the parallel_execute.
TF::AssignVariableOp GetSingleUseResourceWrite(
    tf_device::ParallelExecuteOp parallel_execute, Value result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_parallel_execute_sink_resource_write.cc", "GetSingleUseResourceWrite");

  if (!result.hasOneUse()) return nullptr;

  OpOperand& use = *result.getUses().begin();
  auto assign_var = dyn_cast<TF::AssignVariableOp>(use.getOwner());
  if (!assign_var) return nullptr;

  if (use.get() != assign_var.value()) return nullptr;

  auto* resource_handle_op = assign_var.resource().getDefiningOp();
  if (resource_handle_op == parallel_execute) return nullptr;

  if (resource_handle_op &&
      resource_handle_op->getBlock() ==
          parallel_execute.getOperation()->getBlock() &&
      parallel_execute.getOperation()->isBeforeInBlock(resource_handle_op))
    return nullptr;

  return assign_var;
}

// Finds AssignVariableOps that can be moved into a parallel_execute region and
// moves them. Leftover parallel_execute results that were used by the
// such AssignVariableOp are also pruned.
void SinkResourceWritesIntoParallelExecute(
    tf_device::ParallelExecuteOp parallel_execute) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_parallel_execute_sink_resource_write.cc", "SinkResourceWritesIntoParallelExecute");

  bool rewrite = false;
  const int num_regions = parallel_execute.getNumRegions();
  llvm::SmallVector<Value, 4> results_to_remap;

  // Go through each region and find AssignVariableOps that can be moved into
  // the parallel_execute region. Result indices by region index are collected,
  // so they can be removed afterwards.
  llvm::SmallVector<llvm::SmallVector<int, 4>, 4> results_to_remove_by_region;
  results_to_remove_by_region.resize(num_regions);
  for (int i = 0; i < num_regions; ++i) {
    Block& block = parallel_execute.GetRegionBlockWithIndex(i);
    auto results = parallel_execute.GetRegionOutputs(i);
    auto& results_to_remove = results_to_remove_by_region[i];
    results_to_remove.reserve(results.size());
    Operation* terminator = block.getTerminator();
    for (auto result : llvm::enumerate(results)) {
      TF::AssignVariableOp assign_var =
          GetSingleUseResourceWrite(parallel_execute, result.value());
      if (!assign_var) {
        results_to_remap.push_back(result.value());
        continue;
      }

      // Move AssignVariableOp and update the value to be written to the
      // resource variable to be the non forwarded value from within the
      // parallel_execute region.
      assign_var.getOperation()->moveBefore(terminator);
      assign_var.valueMutable().assign(terminator->getOperand(result.index()));
      results_to_remove.push_back(result.index());
    }

    rewrite |= !results_to_remove.empty();
  }

  if (!rewrite) return;

  // Remove leftover unused results (terminator operands) from moving
  // AssignVariabeOps into the parallel_execute region.
  for (auto results_to_remove : llvm::enumerate(results_to_remove_by_region)) {
    Block& block =
        parallel_execute.GetRegionBlockWithIndex(results_to_remove.index());
    Operation* terminator = block.getTerminator();
    for (int index_to_remove : llvm::reverse(results_to_remove.value()))
      terminator->eraseOperand(index_to_remove);
  }

  // Replace old parallel_execute with new parallel_execute by moving the
  // regions to a new parallel_execute and remapping the results.
  llvm::SmallVector<Type, 4> new_result_types;
  new_result_types.reserve(results_to_remap.size());
  for (Value old_result : results_to_remap)
    new_result_types.push_back(old_result.getType());

  OpBuilder builder(parallel_execute);
  auto new_parallel_execute = builder.create<tf_device::ParallelExecuteOp>(
      parallel_execute.getLoc(), num_regions, new_result_types);

  for (auto region : llvm::zip(new_parallel_execute.getRegions(),
                               parallel_execute.getRegions()))
    std::get<0>(region)->takeBody(*std::get<1>(region));

  for (auto result :
       llvm::zip(results_to_remap, new_parallel_execute.getResults()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  parallel_execute.erase();
}

void TPUParallelExecuteSinkResourceWrite::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_parallel_execute_sink_resource_writeDTcc mht_2(mht_2_v, 321, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_parallel_execute_sink_resource_write.cc", "TPUParallelExecuteSinkResourceWrite::runOnOperation");

  llvm::SmallVector<tf_device::ParallelExecuteOp, 4> parallel_executes;
  getOperation().walk([&](tf_device::ParallelExecuteOp parallel_execute) {
    parallel_executes.push_back(parallel_execute);
  });

  for (tf_device::ParallelExecuteOp parallel_execute : parallel_executes)
    SinkResourceWritesIntoParallelExecute(parallel_execute);
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUParallelExecuteSinkResourceWritePass() {
  return std::make_unique<TPUParallelExecuteSinkResourceWrite>();
}

}  // namespace TFTPU
}  // namespace mlir
