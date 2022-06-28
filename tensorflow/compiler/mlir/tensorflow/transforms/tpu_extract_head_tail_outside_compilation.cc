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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc() {
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
#include <tuple>
#include <type_traits>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

// This pass extracts a CPU computation cluster with `_xla_outside_compilation`
// annotation from the head or tail of a TPU cluster.

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

bool HasOutsideCompilationAttribute(Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_0(mht_0_v, 226, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "HasOutsideCompilationAttribute");

  return op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr) != nullptr;
}

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "GetOpOfValue");

  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

// Checks if `op` is nested in `block`.
bool OpInBlock(Operation* op, Block* block) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "OpInBlock");

  Block* op_block = op->getBlock();
  while (op_block) {
    if (op_block == block) return true;
    if (auto* parent_op = op_block->getParentOp()) {
      op_block = parent_op->getBlock();
    } else {
      break;
    }
  }
  return false;
}

// Wraps block in a Launch. External uses of ops in the block will be return
// values of the Launch and remapped to the Launch results. If `before` is set
// to true, the Launch is created before `op`. Otherwise the Launch is created
// after `op`.
tf_device::LaunchOp CreateLaunchForBlock(OpBuilder* builder, Operation* op,
                                         bool before, Block* launch_block,
                                         llvm::StringRef host_device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_3(mht_3_v, 268, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "CreateLaunchForBlock");

  // Find results and result types of ops in block that needs to returned.
  llvm::SmallVector<Value, 4> launch_results;
  llvm::SmallVector<Type, 4> launch_result_types;
  for (Operation& head_outside_compiled_op : *launch_block) {
    for (Value result : head_outside_compiled_op.getResults()) {
      bool has_external_uses = false;
      for (Operation* user : result.getUsers()) {
        if (OpInBlock(user, launch_block)) continue;
        has_external_uses = true;
        break;
      }
      if (has_external_uses) {
        launch_results.push_back(result);
        launch_result_types.push_back(result.getType());
      }
    }
  }

  before ? builder->setInsertionPoint(op) : builder->setInsertionPointAfter(op);
  auto launch = builder->create<tf_device::LaunchOp>(
      op->getLoc(), builder->getStringAttr(host_device), launch_result_types);
  launch.body().push_back(launch_block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(op->getLoc(), launch_results);

  return launch;
}

// Checks if an operation is a supported TPU embedding op.
bool IsEmbeddingOp(Operation* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_4(mht_4_v, 302, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "IsEmbeddingOp");

  return isa<TF::EnqueueTPUEmbeddingRaggedTensorBatchOp,
             TF::EnqueueTPUEmbeddingSparseTensorBatchOp,
             TF::EnqueueTPUEmbeddingArbitraryTensorBatchOp,
             TF::RecvTPUEmbeddingActivationsOp,
             TF::SendTPUEmbeddingGradientsOp>(op);
}

// Returns a set of ops that are outside compiled and can be extracted to before
// the TPU computation. These ops are either connected to the inputs of the TPU
// computation or other ops that can be extracted, and have no operands from
// other ops in the TPU computation that cannot be extracted.
llvm::SmallVector<Operation*, 4> FindOutsideCompiledOpsAtHead(
    const TF::SideEffectAnalysis& side_effect_analysis,
    tf_device::ClusterOp cluster) {
  const auto& analysis = side_effect_analysis.GetAnalysisForFunc(
      cluster->getParentOfType<FuncOp>());
  Region* cluster_region = &cluster.body();
  llvm::SmallSetVector<Operation*, 4> head_outside_compiled_ops;

  auto cluster_ops = cluster.GetBody().without_terminator();
  for (Operation& cluster_op : cluster_ops) {
    if (!HasOutsideCompilationAttribute(&cluster_op)) continue;
    // An outside compiled op can be extracted if its operands are not from
    // other ops in the cluster that cannot be extracted.

    // Check if the side effecting op right before this side effecting op, if
    // it is side effecting, can be head extracted. Because of op ordering due
    // to side effects, if this is not true, this op cannot be head extracted.
    // TODO(lyandy): Remove special handling of embedding ops. Currently the IR
    // is in a topological sort order and depending on that ordering, embedding
    // ops may prevent other ops from being head extracted.
    auto predecessors = analysis.DirectControlPredecessors(&cluster_op);
    if (!predecessors.empty() && !IsEmbeddingOp(&cluster_op)) {
      bool skip = false;
      for (Operation* predecessor : llvm::reverse(predecessors)) {
        if (IsEmbeddingOp(predecessor)) continue;
        skip = !head_outside_compiled_ops.contains(predecessor);
        break;
      }
      if (skip) continue;
    }

    auto walk_result = cluster_op.walk([&](Operation* op) {
      for (Value operand : op->getOperands()) {
        Operation* operand_op = GetOpOfValue(operand);
        if (head_outside_compiled_ops.count(operand_op) ||
            operand_op == &cluster_op)
          continue;

        if (operand_op->getParentRegion() == cluster_region)
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!walk_result.wasInterrupted())
      head_outside_compiled_ops.insert(&cluster_op);
  }

  return head_outside_compiled_ops.takeVector();
}

// Moves head outside compiled ops into its own `tf_device.LaunchOp`
// computation before the cluster.
void CreateHeadComputation(OpBuilder* builder, tf_device::ClusterOp cluster,
                           llvm::ArrayRef<Operation*> head_outside_compiled_ops,
                           llvm::StringRef host_device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_5(mht_5_v, 372, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "CreateHeadComputation");

  Block* launch_block = new Block;
  for (Operation* head_outside_compiled_op : head_outside_compiled_ops) {
    head_outside_compiled_op->removeAttr(kXlaOutsideCompilationAttr);
    head_outside_compiled_op->moveBefore(launch_block, launch_block->end());
  }

  tf_device::LaunchOp launch = CreateLaunchForBlock(
      builder, cluster, /*before=*/true, launch_block, host_device);

  for (auto result : llvm::zip(launch.GetBody().getTerminator()->getOperands(),
                               launch.getResults()))
    replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                               cluster.body());
}

// Extracts and move outside compiled ops that have no dependencies in the
// cluster to before the cluster.
mlir::LogicalResult LiftHeadOutsideCompiledOps(
    OpBuilder* builder, const TF::SideEffectAnalysis& side_effect_analysis,
    const mlir::TF::RuntimeDevices& devices, tf_device::ClusterOp cluster,
    std::string* host_device, bool* cluster_updated) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_6(mht_6_v, 396, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "LiftHeadOutsideCompiledOps");

  llvm::SmallVector<Operation*, 4> head_outside_compiled_ops =
      FindOutsideCompiledOpsAtHead(side_effect_analysis, cluster);
  if (head_outside_compiled_ops.empty()) return success();
  if (failed(tensorflow::GetHostDeviceOutsideComputation(devices, cluster,
                                                         host_device)))
    return failure();

  CreateHeadComputation(builder, cluster, head_outside_compiled_ops,
                        *host_device);

  *cluster_updated = true;
  return success();
}

// Fills `tail_outside_compiled_ops` with ops that are outside compiled and
// can be extracted to after the TPU computation, and `cluster_results` with new
// results of the cluster. These ops are either connected to the output of the
// TPU computation or other ops that can be extracted, and have no results used
// by other ops in the TPU computation that cannot be extracted.
void FindOutsideCompiledOpsAtTailAndClusterResults(
    const TF::SideEffectAnalysis& side_effect_analysis,
    tf_device::ClusterOp cluster,
    llvm::SmallVectorImpl<Operation*>* tail_outside_compiled_ops,
    llvm::SmallVectorImpl<Value>* cluster_results) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_7(mht_7_v, 423, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "FindOutsideCompiledOpsAtTailAndClusterResults");

  const auto& analysis = side_effect_analysis.GetAnalysisForFunc(
      cluster->getParentOfType<FuncOp>());
  Region* cluster_region = &cluster.body();
  llvm::SmallSetVector<Operation*, 4> tail_outside_compiled_ops_set;
  Operation* terminator = cluster.GetBody().getTerminator();
  llvm::SmallSetVector<Value, 4> cluster_results_set;
  cluster_results_set.insert(terminator->getOperands().begin(),
                             terminator->getOperands().end());

  auto cluster_ops = llvm::reverse(cluster.GetBody().without_terminator());
  for (Operation& cluster_op : cluster_ops) {
    if (!HasOutsideCompilationAttribute(&cluster_op)) continue;

    // Check if the side effecting op right after this side effecting op, if
    // it is side effecting, can be tail extracted. Because of op ordering due
    // to side effects, if this is not true, this op cannot be tail extracted.
    // TODO(lyandy): Remove special handling of embedding ops. Currently the IR
    // is in a topological sort order and depending on that ordering, embedding
    // ops may prevent other ops from being tail extracted.
    auto successors = analysis.DirectControlSuccessors(
        &cluster_op, [&terminator](Operation* op) { return op != terminator; });
    if (!successors.empty() && !IsEmbeddingOp(&cluster_op)) {
      bool skip = false;
      for (Operation* successor : successors) {
        if (IsEmbeddingOp(successor)) continue;
        skip = !tail_outside_compiled_ops_set.contains(successor);
        break;
      }
      if (skip) continue;
    }

    llvm::SmallVector<int, 4> results_to_forward;
    bool can_be_extracted =
        llvm::all_of(cluster_op.getUsers(), [&](Operation* op) {
          return op == terminator || tail_outside_compiled_ops_set.count(op);
        });
    if (!can_be_extracted) continue;

    // Collect operands of cluster op that are generated within the cluster.
    // These values should be returned by the cluster.
    cluster_op.walk([&](Operation* op) {
      for (Value operand : op->getOperands()) {
        Operation* operand_op = GetOpOfValue(operand);
        if (operand_op->getParentRegion() == cluster_region)
          cluster_results_set.insert(operand);
      }
    });

    // Remove results of op to be extracted as there are no uses in the cluster.
    for (Value result : cluster_op.getResults())
      cluster_results_set.remove(result);
    // Insert all ops including nested ops for checking outputs/side effects.
    cluster_op.walk(
        [&](Operation* op) { tail_outside_compiled_ops_set.insert(op); });

    // Only add top level ops to output vector.
    tail_outside_compiled_ops->push_back(&cluster_op);
  }

  *cluster_results = cluster_results_set.takeVector();
}

// Moves tail outside compiled ops into its own `tf_device.LaunchOp`
// computation after the cluster.
void CreateTailComputation(OpBuilder* builder, tf_device::ClusterOp cluster,
                           llvm::ArrayRef<Operation*> tail_outside_compiled_ops,
                           llvm::StringRef host_device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_8(mht_8_v, 493, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "CreateTailComputation");

  Block* launch_block = new Block;
  for (Operation* tail_outside_compiled_op : tail_outside_compiled_ops) {
    tail_outside_compiled_op->removeAttr(kXlaOutsideCompilationAttr);
    tail_outside_compiled_op->moveBefore(launch_block, launch_block->begin());
  }

  tf_device::LaunchOp launch = CreateLaunchForBlock(
      builder, cluster, /*before=*/false, launch_block, host_device);

  auto operand_not_in_launch = [&](OpOperand& operand) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_9(mht_9_v, 506, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "lambda");

    return !launch.getOperation()->isProperAncestor(operand.getOwner());
  };
  for (auto result : llvm::zip(launch.GetBody().getTerminator()->getOperands(),
                               launch.getResults()))
    std::get<0>(result).replaceUsesWithIf(std::get<1>(result),
                                          operand_not_in_launch);
}

// Updates cluster with updated cluster results after extracting tail outside
// compiled ops.
tf_device::ClusterOp UpdateClusterResults(
    OpBuilder* builder, tf_device::ClusterOp cluster,
    llvm::ArrayRef<Value> new_cluster_results) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_10(mht_10_v, 522, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "UpdateClusterResults");

  Operation* old_terminator = cluster.GetBody().getTerminator();
  builder->setInsertionPoint(old_terminator);
  builder->create<tf_device::ReturnOp>(old_terminator->getLoc(),
                                       new_cluster_results);
  old_terminator->erase();

  builder->setInsertionPoint(cluster);
  llvm::SmallVector<Type, 4> new_cluster_result_types;
  new_cluster_result_types.reserve(new_cluster_results.size());
  for (const auto& new_cluster_result : new_cluster_results)
    new_cluster_result_types.push_back(new_cluster_result.getType());

  auto new_cluster = builder->create<tf_device::ClusterOp>(
      cluster.getLoc(), new_cluster_result_types,
      /*operands=*/llvm::ArrayRef<Value>{}, cluster->getAttrs());
  new_cluster.body().takeBody(cluster.body());

  auto operand_not_in_cluster = [&](OpOperand& operand) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_11(mht_11_v, 543, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "lambda");

    return !new_cluster.getOperation()->isProperAncestor(operand.getOwner());
  };
  for (auto result :
       llvm::zip(new_cluster.GetBody().getTerminator()->getOperands(),
                 new_cluster.getResults()))
    std::get<0>(result).replaceUsesWithIf(std::get<1>(result),
                                          operand_not_in_cluster);

  cluster.erase();
  return new_cluster;
}

// Extracts and move outside compiled ops that do not create dependencies in the
// cluster to after the cluster.
mlir::LogicalResult LiftTailOutsideCompiledOps(
    OpBuilder* builder, const TF::SideEffectAnalysis& side_effect_analysis,
    const mlir::TF::RuntimeDevices& devices, std::string host_device,
    tf_device::ClusterOp* cluster, bool* cluster_updated) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("host_device: \"" + host_device + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_12(mht_12_v, 565, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "LiftTailOutsideCompiledOps");

  llvm::SmallVector<Operation*, 4> tail_outside_compiled_ops;
  llvm::SmallVector<Value, 4> cluster_results;
  FindOutsideCompiledOpsAtTailAndClusterResults(side_effect_analysis, *cluster,
                                                &tail_outside_compiled_ops,
                                                &cluster_results);
  if (tail_outside_compiled_ops.empty()) return success();

  if (host_device.empty())
    if (failed(tensorflow::GetHostDeviceOutsideComputation(devices, *cluster,
                                                           &host_device)))
      return failure();

  // Forward all results of cluster first. These results will be remapped once
  // a new cluster is formed.
  cluster->replaceAllUsesWith(
      cluster->GetBody().getTerminator()->getOperands());

  CreateTailComputation(builder, *cluster, tail_outside_compiled_ops,
                        host_device);

  *cluster = UpdateClusterResults(builder, *cluster, cluster_results);

  *cluster_updated = true;
  return success();
}

// Removes aliased outputs in cluster from ops outside of cluster.
void RemoveClusterAliasedOutputs(OpBuilder* builder,
                                 tf_device::ClusterOp cluster) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_13(mht_13_v, 597, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "RemoveClusterAliasedOutputs");

  llvm::SmallVector<Value, 4> used_old_cluster_results;
  llvm::SmallVector<Value, 4> new_cluster_results;
  llvm::SmallVector<Type, 4> new_cluster_result_types;
  Operation* cluster_terminator = cluster.GetBody().getTerminator();
  for (auto result :
       llvm::zip(cluster_terminator->getOperands(), cluster.getResults())) {
    Value cluster_terminator_operand = std::get<0>(result);
    if (cluster_terminator_operand.getDefiningOp() &&
        cluster.getOperation()->isProperAncestor(
            cluster_terminator_operand.getDefiningOp())) {
      new_cluster_results.push_back(cluster_terminator_operand);
      new_cluster_result_types.push_back(cluster_terminator_operand.getType());
      used_old_cluster_results.push_back(std::get<1>(result));
    } else {
      std::get<1>(result).replaceAllUsesWith(cluster_terminator_operand);
    }
  }

  if (new_cluster_results.size() == cluster.getNumResults()) return;

  builder->setInsertionPoint(cluster);
  auto new_cluster = builder->create<tf_device::ClusterOp>(
      cluster.getLoc(), new_cluster_result_types,
      /*operands=*/llvm::ArrayRef<Value>{}, cluster->getAttrs());
  new_cluster.body().takeBody(cluster.body());
  new_cluster.GetBody().getTerminator()->setOperands(new_cluster_results);

  for (auto result :
       llvm::zip(used_old_cluster_results, new_cluster.getResults()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  cluster.erase();
}

struct TPUExtractHeadTailOutsideCompilationPass
    : public TF::TPUExtractHeadTailOutsideCompilationPassBase<
          TPUExtractHeadTailOutsideCompilationPass> {
  void runOnOperation() override;
};

void TPUExtractHeadTailOutsideCompilationPass::runOnOperation() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_extract_head_tail_outside_compilationDTcc mht_14(mht_14_v, 641, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_extract_head_tail_outside_compilation.cc", "TPUExtractHeadTailOutsideCompilationPass::runOnOperation");

  auto& side_effect_analysis = getAnalysis<TF::SideEffectAnalysis>();
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  OpBuilder builder(&getContext());
  llvm::SmallVector<tf_device::ClusterOp, 4> clusters;
  module.walk(
      [&](tf_device::ClusterOp cluster) { clusters.push_back(cluster); });

  for (tf_device::ClusterOp cluster : clusters) {
    std::string host_device;
    bool cluster_updated = false;
    if (failed(LiftHeadOutsideCompiledOps(&builder, side_effect_analysis,
                                          devices, cluster, &host_device,
                                          &cluster_updated)) ||
        failed(LiftTailOutsideCompiledOps(&builder, side_effect_analysis,
                                          devices, host_device, &cluster,
                                          &cluster_updated)))
      return signalPassFailure();
    if (cluster_updated) RemoveClusterAliasedOutputs(&builder, cluster);
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUExtractHeadTailOutsideCompilationPass() {
  return std::make_unique<TPUExtractHeadTailOutsideCompilationPass>();
}

}  // namespace TFTPU
}  // namespace mlir
