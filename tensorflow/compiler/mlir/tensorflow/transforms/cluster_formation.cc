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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc() {
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

// This transformation forms clusters from instructions in same island and
// assigned to save devices. Clusters are represented as regions.
// Note that side-effecting ops are not correctly handled yet.

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFDevice {

namespace {

struct ClusterFormationPass
    : public TF::ClusterFormationPassBase<ClusterFormationPass> {
  void runOnOperation() override;
};

// Cluster structure captures all the operations that are assigned to same
// device and can form a legal strict cluster.
// Ops must follow same ordering in their parent block. We rely on this
// assumption to perform analysis.
struct Cluster {
  llvm::SmallVector<Operation*, 4> ops;
  StringRef device;
};

StringRef GetDevice(Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_0(mht_0_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "GetDevice");

  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return device_attr ? device_attr.getValue() : "";
}

// An op can be merged into cluster if all of its operands are one of the
// following:
//  1) A block argument
//  2) A value produced by other islands
//  1) Defined before the cluster
//  2) Defined by an operation in the cluster
// TODO(ycao): This is not optimal as it doesn't consider the situation of
// defining_op's operands all meet the requirements above. In that case, the
// defining_op can be moved and to_merge op would be legal to absorb.
// TODO(ycao): Take op side-effects into consideration since they can not be
// re-ordered but forming clusters of non-continuous ops is effectively
// re-ordering them..
bool CanMergeIntoCluster(const Cluster& c, Operation* to_merge) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_1(mht_1_v, 244, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "CanMergeIntoCluster");

  return llvm::all_of(to_merge->getOperands(), [&](Value operand) {
    // Block arguments.
    if (operand.isa<BlockArgument>()) return true;

    Operation* defining_op = operand.getDefiningOp();

    // Operand produced by other islands.
    if (defining_op->getBlock() != c.ops.front()->getBlock()) return true;

    // Defining op is before the cluster.
    if (defining_op->isBeforeInBlock(c.ops.front())) return true;

    // Defining op is between first and last operation in cluster. Note that
    // cluster may contain operations that are non-continuous in their original
    // block, thus we also need to check defining_op is also assigned to
    // cluster's device to be sure. This is a faster check than linearly
    // searching through all ops in cluster.
    if (defining_op->isBeforeInBlock(c.ops.back()->getNextNode()) &&
        GetDevice(defining_op) == c.device)
      return true;

    // Other cases, operand is generated after or outside the cluster, this
    // means it is illegal to merge operation.
    return false;
  });
}

void ReplaceLiveOutExternalUses(llvm::ArrayRef<Value> live_outs,
                                tf_device::LaunchOp launch_op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_2(mht_2_v, 276, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "ReplaceLiveOutExternalUses");

  Region* launch_op_region = &launch_op.body();
  for (const auto& p : llvm::zip(live_outs, launch_op.getResults())) {
    Value from = std::get<0>(p);
    // TODO(jingpu): move this to RegionUtils.h in MLIR core.
    for (auto& use : llvm::make_early_inc_range(from.getUses())) {
      if (launch_op_region->isAncestor(use.getOwner()->getParentRegion()))
        continue;
      use.set(std::get<1>(p));
    }
  }
}

// Get all escaped live-out values of a region.
void GetLiveOuts(Region* region, llvm::SmallVectorImpl<Value>* live_outs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_3(mht_3_v, 293, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "GetLiveOuts");

  live_outs->clear();

  for (Operation& op : region->front()) {
    for (Value v : op.getResults()) {
      // A value is live-out if any of its users are not inside value producer's
      // region.
      bool is_live_out = llvm::any_of(v.getUsers(), [&](Operation* user) {
        return !region->isAncestor(user->getParentRegion());
      });

      if (is_live_out) live_outs->emplace_back(v);
    }
  }
}

// Build a `tf_device.launch` op with a region that contains all the operations
// in given cluster. Then all ops in cluster are replaced by `tf_device.launch`.
void BuildLaunchForCluster(const Cluster& c, OpBuilder* builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_4(mht_4_v, 314, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "BuildLaunchForCluster");

  // Set insertion point to right after all operations in cluster.
  builder->setInsertionPoint(c.ops.back()->getNextNode());

  // Create a stand-alone region to hold all instructions in the cluster.
  Region region;
  region.push_back(new Block);

  // Move all operations in cluster to newly created region, stripping their
  // "device" attribute since launch op already carries device information.
  Block* block = &region.front();
  for (Operation* op : c.ops) {
    op->moveBefore(block, block->end());
    op->removeAttr(builder->getStringAttr("device"));
  }

  // Get all escaped live-out values of region, they are used later to determine
  // return values and types of launch op.
  llvm::SmallVector<Value, 4> live_outs;
  GetLiveOuts(&region, &live_outs);

  // Build a `tf_device.return` op at end of region, with all live-out values
  // as operand.
  OpBuilder return_builder(builder->getContext());
  return_builder.setInsertionPointToEnd(block);
  return_builder.create<tf_device::ReturnOp>(return_builder.getUnknownLoc(),
                                             live_outs);

  llvm::SmallVector<Type, 4> live_out_types;
  live_out_types.reserve(live_outs.size());
  for (Value v : live_outs) {
    live_out_types.emplace_back(v.getType());
  }

  tf_device::LaunchOp launch_op = builder->create<tf_device::LaunchOp>(
      builder->getUnknownLoc(), builder->getStringAttr(c.device),
      live_out_types);

  // Attach the region to launch_op.
  launch_op.body().takeBody(region);

  // Replace any external uses of live-out values with return values of launch
  // op. So live-out values no longer escape the region.
  ReplaceLiveOutExternalUses(live_outs, launch_op);
}

void BuildClusters(Block* block, OpBuilder builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_5(mht_5_v, 363, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "BuildClusters");

  // Iteratively find clusters of different devices within an island.
  // Whenever we see an operation that is assigned to an accelerator device
  // (ie. device != ""), we try to merge it into the last cluster of same
  // device. If that is infeasible (say because of violating def-before-use),
  // create a new cluster with that operation and move on.
  llvm::MapVector<StringRef, Cluster> nearest_clusters;
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    auto device = GetDevice(&op);
    if (device.empty()) continue;

    // If no cluster of same device has been formed yet, create a new cluster
    // with op alone.
    auto it = nearest_clusters.find(device);
    if (it == nearest_clusters.end()) {
      nearest_clusters[device] = Cluster{{&op}, device};
      continue;
    }

    // Check if it is legal to merge op into nearest cluster of same device.
    // If positive, update cluster and move on to next operation.
    Cluster& nearest_cluster = it->second;
    if (CanMergeIntoCluster(nearest_cluster, &op)) {
      nearest_cluster.ops.emplace_back(&op);
      continue;
    }

    // If nearest cluster of same device can not absorb `op`, then that
    // cluster needs to be finalized by building a `tf_device.launch` op with
    // a region that contains all operations in clusters.
    BuildLaunchForCluster(nearest_cluster, &builder);

    // Create a new cluster to hold op alone and update nearest_clusters.
    nearest_clusters[device] = Cluster{{&op}, device};
  }

  // At the end, there might be left-over found clusters that need to be
  // built.
  for (auto& device_cluster : nearest_clusters)
    BuildLaunchForCluster(device_cluster.second, &builder);
}

void ClusterFormationPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPScluster_formationDTcc mht_6(mht_6_v, 408, "", "./tensorflow/compiler/mlir/tensorflow/transforms/cluster_formation.cc", "ClusterFormationPass::runOnOperation");

  auto func = getOperation();
  if (func.isExternal()) return;
  OpBuilder builder(func.getContext());

  // Operates on individual blocks independently of if they are directly in the
  // function body or if they are nested in individual `tf_executor.island`.
  for (Block& block : func.getBody()) BuildClusters(&block, builder);
  func.walk([&](tf_executor::IslandOp island) {
    BuildClusters(&island.GetBody(), builder);
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateClusterFormationPass() {
  return std::make_unique<ClusterFormationPass>();
}

}  // namespace TFDevice
}  // namespace mlir
