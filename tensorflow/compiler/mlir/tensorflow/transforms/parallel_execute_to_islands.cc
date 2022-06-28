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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc() {
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

// This pass forms `tf_executor.island` per region of
// `tf_device.parallel_execute`.
//
// For example, the following:
//
//  %0 = tf_executor.island {
//    tf_executor.yield
//  }
//  %1:2 = tf_executor.island {
//    %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//      tf_executor.yield %2 : tensor<i1>
//  }
//  %3:2 = tf_executor.island(%0) {
//    %4 = "tf_device.parallel_execute"() ({
//      %5 = "tf.opB"() : () -> tensor<i1>
//      tf_device.return %5 : tensor<i1>
//    }, {
//      %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
//      tf_device.return
//    }) {} : () -> (tensor<i1>)
//    tf_executor.yield %4 : tensor<i1>
//  }
//  tf_executor.fetch %3#0 : tensor<i1>
//
// gets lowered to:
//
//  %0 = tf_executor.island {
//    tf_executor.yield
//  }
//  %1:2 = tf_executor.island {
//    %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//    tf_executor.yield %2 : tensor<i1>
//  }
//
//  // Island for the first region of above parallel_execute.
//  %3:2 = tf_executor.island(%0) {
//    %4 = "tf.opB"() : () -> tensor<i1>
//    tf_executor.yield %4 : tensor<i1>
//  }
//
//  // Island for the second region of above parallel_execute.
//  %5 = tf_executor.island(%0) {
//    %6 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
//    tf_executor.yield
//  }
//
//  tf_executor.fetch %3#0, %5 : tensor<i1>, !tf_executor.control
//
//  When tf_device.parallel_execute op is enclosed after tf_device.replicate,
//  then this pass will run following `replicate-to-island` pass and
//  `tf-executor-break-up-islands` pass.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFDevice {
namespace {

struct ParallelExecuteToIslandsPass
    : public TF::ParallelExecuteToIslandsPassBase<
          ParallelExecuteToIslandsPass> {
  void runOnOperation() override;
};

// Convert parallel_execute op to a set of islands where each region of
// parallel_execute op becomes a separate island. This ensures that the regions
// of the parallel_execute op gets executed concurrently.
void ExpandParallelExecuteToIslands(
    tf_executor::IslandOp island_op,
    tf_device::ParallelExecuteOp parallel_execute_op, OpBuilder* builder,
    llvm::SmallVectorImpl<tf_executor::IslandOp>& executes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc mht_0(mht_0_v, 264, "", "./tensorflow/compiler/mlir/tensorflow/transforms/parallel_execute_to_islands.cc", "ExpandParallelExecuteToIslands");

  const int num_regions = parallel_execute_op.getOperation()->getNumRegions();
  executes.reserve(num_regions);

  for (int i : llvm::seq<int>(0, num_regions)) {
    Block& execute_block = parallel_execute_op.GetRegionBlockWithIndex(i);

    // Replace terminator with tf_executor.YieldOp.
    Operation* terminator = execute_block.getTerminator();
    builder->setInsertionPoint(terminator);
    auto yield = builder->create<tf_executor::YieldOp>(
        terminator->getLoc(), terminator->getOperands());
    terminator->erase();

    // Create new island for each region.
    builder->setInsertionPoint(island_op);
    auto execute_island = builder->create<tf_executor::IslandOp>(
        island_op.getLoc(), yield.getOperandTypes(),
        island_op.control().getType(), island_op.controlInputs());

    // Move over tf_device.parallel_execute body region into newly the created
    // island.
    execute_island.body().takeBody(*execute_block.getParent());
    executes.push_back(execute_island);
  }
}

void CreateIslandsFromParallelExecute(
    tf_executor::IslandOp island_op,
    tf_device::ParallelExecuteOp parallel_execute_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc mht_1(mht_1_v, 296, "", "./tensorflow/compiler/mlir/tensorflow/transforms/parallel_execute_to_islands.cc", "CreateIslandsFromParallelExecute");

  OpBuilder builder(island_op);

  // Create islands for each region of the parallel_execute op.
  llvm::SmallVector<tf_executor::IslandOp, 4> executes;
  ExpandParallelExecuteToIslands(island_op, parallel_execute_op, &builder,
                                 executes);

  // Remap all results of parallel_execute op with outputs from newly created
  // islands.
  llvm::SmallVector<Value, 8> parallel_execute_outputs;
  parallel_execute_outputs.reserve(
      parallel_execute_op.getOperation()->getNumResults());

  for (auto& execute : executes)
    parallel_execute_outputs.append(execute.outputs().begin(),
                                    execute.outputs().end());

  for (auto result : llvm::zip(island_op.outputs(), parallel_execute_outputs))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  // Add sink island to pin all islands as a control dependency if there is a
  // control dependency leading from the parallel_execute originally.
  if (!island_op.control().use_empty()) {
    llvm::SmallVector<Value, 8> island_operands;
    for (auto& execute : executes) island_operands.push_back(execute.control());

    builder.setInsertionPoint(island_op);
    auto island_sink = builder.create<tf_executor::IslandOp>(
        island_op.getLoc(), llvm::ArrayRef<Type>{},
        island_op.control().getType(), island_operands);
    island_sink.body().push_back(new Block);
    builder.setInsertionPointToEnd(&island_sink.GetBody());
    builder.create<tf_executor::YieldOp>(island_op.getLoc(),
                                         llvm::ArrayRef<Value>{});
    island_op.control().replaceAllUsesWith(island_sink.control());
  }

  // Islands with no uses should be pinned to a graph fetch so they still
  // execute.
  llvm::SmallVector<Value, 8> unused_execute_controls;
  for (auto& execute : executes)
    if (execute.use_empty())
      unused_execute_controls.push_back(execute.control());

  if (!unused_execute_controls.empty()) {
    auto graph_op = island_op->getParentOfType<tf_executor::GraphOp>();
    tf_executor::FetchOp fetch = graph_op.GetFetch();
    auto fetches = llvm::to_vector<8>(fetch.getOperands());
    fetches.append(unused_execute_controls.begin(),
                   unused_execute_controls.end());
    builder.setInsertionPoint(fetch);
    builder.create<tf_executor::FetchOp>(fetch.getLoc(), fetches);
    fetch.erase();
  }

  island_op.erase();
}

void ParallelExecuteToIslandsPass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSparallel_execute_to_islandsDTcc mht_2(mht_2_v, 358, "", "./tensorflow/compiler/mlir/tensorflow/transforms/parallel_execute_to_islands.cc", "ParallelExecuteToIslandsPass::runOnOperation");

  // Find islands with a single `tf_device.parallel_execute` and create
  // individual islands per execute region of the parallel_execute.
  llvm::SmallVector<tf_executor::IslandOp, 4> parallel_execute_op_islands;
  getOperation().walk([&](tf_executor::GraphOp graph_op) {
    for (auto island_op : graph_op.getOps<tf_executor::IslandOp>()) {
      if (!island_op.WrapsSingleOp()) continue;

      if (isa<tf_device::ParallelExecuteOp>(&island_op.GetBody().front()))
        parallel_execute_op_islands.push_back(island_op);
    }
  });

  for (tf_executor::IslandOp island_op : parallel_execute_op_islands) {
    auto parallel_execute_op =
        cast<tf_device::ParallelExecuteOp>(island_op.GetBody().front());
    CreateIslandsFromParallelExecute(island_op, parallel_execute_op);
  }
}
}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateParallelExecuteToIslandsPass() {
  return std::make_unique<ParallelExecuteToIslandsPass>();
}

}  // namespace TFDevice
}  // namespace mlir
