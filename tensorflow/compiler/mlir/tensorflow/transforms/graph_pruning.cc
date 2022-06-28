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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc() {
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace tf_executor {
namespace {

// This transformation pass prunes a TF graph eliminating dead-nodes.
class GraphPruningPass
    : public TF::ExecutorGraphPruningPassBase<GraphPruningPass> {
 public:
  GraphPruningPass() = default;
  explicit GraphPruningPass(llvm::ArrayRef<std::string> ops_to_preserve);
  void runOnOperation() override;

 private:
  bool ShouldPreserveOp(Operation* op);
  bool ShouldPreserveIsland(IslandOp island);
  void PruneGraph(GraphOp graph);

  llvm::SmallDenseSet<mlir::StringAttr, 4> ops_to_preserve_ids_;
};

// Checks if a tf_executor.Graph can be pruned.
// For TensorFlow V1.0 compatibility: when importing a graph without providing
// feeds/fetches/targets we should not attempt to prune. The best approximation
// here is to check if the graph is of the "main" function and does not have the
// "tf.entry_function" attribute defined.
bool CanPruneGraph(FuncOp func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "CanPruneGraph");

  return func.getName() != "main" ||
         func->getAttrOfType<DictionaryAttr>("tf.entry_function") != nullptr;
}

// Visits an op's operand if it is an output of an Operation in the same
// tf_executor.graph.
void VisitOpOperand(GraphOp graph, Value operand,
                    llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
                    llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "VisitOpOperand");

  Operation* def = operand.getDefiningOp();
  if (def && def->getParentOp() == graph && reachable_ops->insert(def).second) {
    // Op has not been visited, add to queue to visit later.
    ops_to_visit->push_back(def);
  }
}

// Visits all operands of an op where each operand is an output of an Operation
// in the same tf_executor.graph.
void VisitOpOperands(GraphOp graph, Operation* op,
                     llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
                     llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "VisitOpOperands");

  for (Value operand : op->getOperands())
    VisitOpOperand(graph, operand, reachable_ops, ops_to_visit);
}

// Visits an op and it's associated operands. IslandOps are handled differently
// where it's regions op operands are also visited as values may be implicitly
// captured within. NextIterationSourceOp will also visit it's associated
// NextIterationSinkOp.
void VisitOp(GraphOp graph, Operation* op,
             llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
             llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_3(mht_3_v, 269, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "VisitOp");

  if (auto island = llvm::dyn_cast<IslandOp>(op)) {
    mlir::visitUsedValuesDefinedAbove(
        island.body(), island.body(), [&](OpOperand* operand) {
          VisitOpOperand(graph, operand->get(), reachable_ops, ops_to_visit);
        });
  }

  VisitOpOperands(graph, op, reachable_ops, ops_to_visit);

  // If op is a `tf_executor.NextIteration.Source`, visit its associated
  // `tf_executor.NextIteration.Sink` op.
  if (auto source_op = llvm::dyn_cast<NextIterationSourceOp>(op)) {
    Operation* sink_op = source_op.GetSink().getOperation();
    if (reachable_ops->insert(sink_op).second) ops_to_visit->push_back(sink_op);
  }
}

GraphPruningPass::GraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "GraphPruningPass::GraphPruningPass");

  ops_to_preserve_ = ops_to_preserve;
}

void GraphPruningPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "GraphPruningPass::runOnOperation");

  for (const auto& op_name : ops_to_preserve_) {
    ops_to_preserve_ids_.insert(mlir::StringAttr::get(&getContext(), op_name));
  }
  if (!CanPruneGraph(getOperation())) return;
  getOperation().walk(
      [this](tf_executor::GraphOp graph) { PruneGraph(graph); });
}

// An op should be preserved if either its identifier is contained in
// `ops_to_preserve_ids_` or if it has a `MustExecute` effect.
bool GraphPruningPass::ShouldPreserveOp(Operation* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_6(mht_6_v, 312, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "GraphPruningPass::ShouldPreserveOp");

  if (ops_to_preserve_ids_.contains(op->getName().getIdentifier())) return true;

  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (interface) interface.getEffects(effects);

  for (const auto& effect : effects) {
    if (llvm::isa<TF::ResourceEffects::MustExecute>(effect.getResource())) {
      return true;
    }
  }
  return false;
}

// An island should be preserved if any of its inner ops should be preserved.
bool GraphPruningPass::ShouldPreserveIsland(IslandOp island) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_7(mht_7_v, 331, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "GraphPruningPass::ShouldPreserveIsland");

  auto result = island.walk([this](Operation* inner_op) {
    if (ShouldPreserveOp(inner_op)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Prunes unreachable operations of a tf_executor.graph operation.
void GraphPruningPass::PruneGraph(GraphOp graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSgraph_pruningDTcc mht_8(mht_8_v, 343, "", "./tensorflow/compiler/mlir/tensorflow/transforms/graph_pruning.cc", "GraphPruningPass::PruneGraph");

  // A graph has a single block which forms a DAG: operations that aren't
  // reachable from the `fetch` operands can be eliminated.

  llvm::SmallPtrSet<Operation*, 8> reachable_ops;
  llvm::SmallVector<Operation*, 8> ops_to_visit;

  // Visit fetches first to create a starting point for ops that are reachable.
  reachable_ops.insert(graph.GetFetch());
  VisitOpOperands(graph, graph.GetFetch(), &reachable_ops, &ops_to_visit);

  // Find and visit ops that should be preserved regardless of being reachable
  // from a fetch.
  for (Operation& op : graph.GetBody().without_terminator()) {
    auto island = llvm::dyn_cast<IslandOp>(op);
    if (!island) continue;
    if (ShouldPreserveIsland(island)) {
      reachable_ops.insert(&op);
      VisitOp(graph, &op, &reachable_ops, &ops_to_visit);
    }
  }

  // Visit transitive ops until no there are no reachable ops left that have not
  // been visited.
  while (!ops_to_visit.empty()) {
    Operation* op = ops_to_visit.pop_back_val();
    VisitOp(graph, op, &reachable_ops, &ops_to_visit);
  }

  // Erase unreachable ops in reverse order so references don't need to be
  // dropped before removing an op. Going in reverse order will guarantee that
  // when an op to be erased is reached, there are no users left.
  for (Operation& op :
       llvm::make_early_inc_range(llvm::reverse(graph.GetBody())))
    if (!reachable_ops.contains(&op)) op.erase();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTFExecutorGraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve) {
  return std::make_unique<GraphPruningPass>(ops_to_preserve);
}

}  // namespace tf_executor
}  // namespace mlir
