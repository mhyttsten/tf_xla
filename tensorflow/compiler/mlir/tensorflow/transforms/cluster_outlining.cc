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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc() {
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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kFuncAttr[] = "func";

struct ClusterOutliningPass
    : public TF::ClusterOutliningPassBase<ClusterOutliningPass> {
  void runOnOperation() override;
};

struct LaunchOutliningPass
    : public TF::LaunchOutliningPassBase<LaunchOutliningPass> {
  void runOnOperation() override;
};

void ReplaceClusterReturnWithReturn(tf_device::ReturnOp cluster_return_op,
                                    OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "ReplaceClusterReturnWithReturn");

  builder->create<func::ReturnOp>(cluster_return_op.getLoc(),
                                  cluster_return_op.getOperands());
  cluster_return_op.erase();
}

// Builds a function that outlines region attached to cluster_op or launch_op,
// and inserts built function into given module.
template <typename ClusterOrLaunchOp>
FuncOp BuildFunction(llvm::ArrayRef<Value> live_ins, ClusterOrLaunchOp op,
                     SymbolTable* symbol_table, OpBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "BuildFunction");

  llvm::SmallVector<Type, 4> operand_types;
  operand_types.reserve(live_ins.size());
  for (Value v : live_ins) operand_types.emplace_back(v.getType());

  auto func_type = builder->getFunctionType(operand_types, op.getResultTypes());

  // TODO(lyandy): Define better name for outlined function. Potentially some
  // name can be added during cluster formation.
  FuncOp outlined_func = FuncOp::create(op.getLoc(), "_func", func_type);

  // This function is not externally visible and marking it private would allow
  // symbol-dce pass to remove it when it is not referenced anymore.
  outlined_func.setPrivate();

  // Create function body.
  Block* outlined_func_block = outlined_func.addEntryBlock();

  // Replace uses of live-in values within cluster_op region with function
  // arguments.
  Region& op_region = op.body();
  for (auto p : llvm::zip(live_ins, outlined_func_block->getArguments())) {
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p), op_region);
  }

  // Move all instructions in cluster_op into outlined_function's only block.
  auto& op_body = op.GetBody().getOperations();
  outlined_func_block->getOperations().splice(
      outlined_func_block->end(), op_body, op_body.begin(), op_body.end());

  // Replace `tf_device.return` terminator with `std.return` in function
  // body.
  auto return_op =
      cast<tf_device::ReturnOp>(outlined_func_block->getTerminator());
  builder->setInsertionPoint(return_op);
  ReplaceClusterReturnWithReturn(return_op, builder);

  symbol_table->insert(outlined_func);
  return outlined_func;
}

// Outlines body of `tf_device.cluster` into a function and create a
// `tf_device.cluster_func` to invoke that function. `tf_device.cluster` is
// removed afterwards.`
void OutlineCluster(tf_device::ClusterOp cluster_op, SymbolTable* symbol_table,
                    OpBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_2(mht_2_v, 279, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "OutlineCluster");

  llvm::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(cluster_op.body(), cluster_op.body(), live_ins);

  FuncOp outlined_func =
      BuildFunction(live_ins.getArrayRef(), cluster_op, symbol_table, builder);
  cluster_op->setAttr(
      builder->getStringAttr(kFuncAttr),
      mlir::SymbolRefAttr::get(builder->getContext(), outlined_func.getName()));

  builder->setInsertionPoint(cluster_op);
  auto cluster_func_op = builder->create<tf_device::ClusterFuncOp>(
      cluster_op.getLoc(), outlined_func.getFunctionType().getResults(),
      live_ins.getArrayRef(), cluster_op->getAttrs());

  cluster_op.replaceAllUsesWith(cluster_func_op);
  cluster_op.erase();
}

// Outlines body of `tf_device.launch` into a function and create a
// `tf_device.launch_func` to invoke that function. `tf_device.launch` is
// removed afterwards.`
void OutlineLaunch(tf_device::LaunchOp launch_op, SymbolTable* symbol_table,
                   OpBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_3(mht_3_v, 305, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "OutlineLaunch");

  llvm::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(launch_op.body(), launch_op.body(), live_ins);

  FuncOp outlined_func =
      BuildFunction(live_ins.getArrayRef(), launch_op, symbol_table, builder);
  launch_op->setAttr(
      builder->getStringAttr(kFuncAttr),
      mlir::SymbolRefAttr::get(builder->getContext(), outlined_func.getName()));

  builder->setInsertionPoint(launch_op);
  auto cluster_func_op = builder->create<tf_device::LaunchFuncOp>(
      launch_op.getLoc(), outlined_func.getFunctionType().getResults(),
      live_ins.getArrayRef(), launch_op->getAttrs());

  launch_op.replaceAllUsesWith(cluster_func_op);
  launch_op.erase();
}

void ClusterOutliningPass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_4(mht_4_v, 327, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "ClusterOutliningPass::runOnOperation");

  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);
  OpBuilder builder(module.getContext());
  module.walk([&](tf_device::ClusterOp cluster) {
    OutlineCluster(cluster, &symbol_table, &builder);
  });
}

void LaunchOutliningPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_outliningDTcc mht_5(mht_5_v, 339, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_outlining.cc", "LaunchOutliningPass::runOnOperation");

  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);
  OpBuilder builder(module.getContext());
  module.walk([&](tf_device::LaunchOp launch) {
    OutlineLaunch(launch, &symbol_table, &builder);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateClusterOutliningPass() {
  return std::make_unique<ClusterOutliningPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateLaunchOutliningPass() {
  return std::make_unique<LaunchOutliningPass>();
}

}  // namespace TFDevice
}  // namespace mlir
