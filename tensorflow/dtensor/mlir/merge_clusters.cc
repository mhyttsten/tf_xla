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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kMissingMeshErrorMsg[] =
    "Failed to extract mesh for DTensorMergeCluster pass. "
    "All clusters must have specified mesh.";

constexpr char kSendRecvKeyPrefix[] = "SendRecvKeyForControlflow_";

// Extracts mesh from `cluster`.
mlir::LogicalResult ExtractMeshFromCluster(mlir::tf_device::ClusterOp cluster,
                                           Mesh* mesh_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_0(mht_0_v, 233, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "ExtractMeshFromCluster");

  auto mesh_or_status = ExtractDeviceMeshFromOp(cluster);
  if (!mesh_or_status.ok()) return cluster.emitOpError(kMissingMeshErrorMsg);

  const absl::optional<Mesh>& mesh_or_null = *mesh_or_status;
  if (!mesh_or_null.has_value())
    return cluster.emitOpError(kMissingMeshErrorMsg);

  *mesh_output = mesh_or_null.value();
  return mlir::success();
}

// Returns all tf_device.ClusterOps nested inside `op`.
llvm::SmallVector<mlir::tf_device::ClusterOp, 4> FindAllDeviceClusters(
    mlir::Operation* op) {
  llvm::SmallVector<mlir::tf_device::ClusterOp, 4> nested_clusters;
  op->walk([&](mlir::tf_device::ClusterOp nested_cluster) {
    nested_clusters.emplace_back(nested_cluster);
  });
  return nested_clusters;
}

mlir::LogicalResult MergeAttributes(
    mlir::Operation* op, mlir::DenseIntElementsAttr indices_attr,
    mlir::ArrayAttr layout_attr, mlir::DenseIntElementsAttr indices_attr2,
    mlir::ArrayAttr layout_attr2, llvm::SmallVector<int, 4>* merged_indices,
    llvm::SmallVector<mlir::Attribute, 4>* merged_layout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_1(mht_1_v, 262, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "MergeAttributes");

  llvm::SmallDenseMap<llvm::APInt, mlir::Attribute> attr_map;
  attr_map.reserve(indices_attr.size() + indices_attr2.size());
  for (const auto& data : llvm::zip(indices_attr, layout_attr))
    attr_map.try_emplace(std::get<0>(data), std::get<1>(data));

  for (const auto& data : llvm::zip(indices_attr2, layout_attr2)) {
    const auto& index = std::get<0>(data);
    const auto& layout = std::get<1>(data);
    auto result = attr_map.try_emplace(index, layout);

    if (!result.second && layout != result.first->getSecond()) {
      return op->emitOpError(
          "Found conflicting metadata attributes while merging clusters");
    }
  }

  merged_indices->reserve(attr_map.size());
  merged_layout->reserve(attr_map.size());
  for (const auto& it : attr_map) {
    merged_indices->emplace_back(it.first.getSExtValue());
    merged_layout->emplace_back(it.second);
  }
  return mlir::success();
}

// Merges metadata attribute from `src_cluster` to `target_cluster`. If metadata
// attribute exists for both clusters, merge the attributes and verify that
// there are no conflicing attributes.
mlir::LogicalResult MergeClusterMetadata(
    mlir::tf_device::ClusterOp src_cluster,
    mlir::tf_device::ClusterOp target_cluster) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_2(mht_2_v, 296, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "MergeClusterMetadata");

  if (mlir::failed(ValidateMetadataAttributes(src_cluster)) ||
      mlir::failed(ValidateMetadataAttributes(target_cluster)))
    return mlir::failure();

  mlir::OpBuilder builder(target_cluster);

  // Extract resource metadata from src/target clusters.
  auto src_resource_handle_indices_metadata =
      src_cluster->getAttrOfType<mlir::DenseIntElementsAttr>(
          kNewResourceLayoutIndices);
  auto src_inferred_resource_handle_layouts_metadata =
      src_cluster->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);

  auto target_resource_handle_indices_metadata =
      target_cluster->getAttrOfType<mlir::DenseIntElementsAttr>(
          kNewResourceLayoutIndices);
  auto target_inferred_resource_handle_layouts_metadata =
      target_cluster->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);
  const bool should_merge_resource_metadata =
      (src_inferred_resource_handle_layouts_metadata &&
       src_resource_handle_indices_metadata &&
       target_inferred_resource_handle_layouts_metadata &&
       target_resource_handle_indices_metadata);
  // If only source cluster has metadata, then simply copy the metadata to
  // target  cluster.
  if (src_inferred_resource_handle_layouts_metadata &&
      !target_inferred_resource_handle_layouts_metadata) {
    target_cluster->setAttr(kNewResourceLayoutIndices,
                            src_resource_handle_indices_metadata);
    target_cluster->setAttr(kNewResourceArgLayouts,
                            src_inferred_resource_handle_layouts_metadata);
  } else if (should_merge_resource_metadata) {
    // If both src cluster and target cluster has metadata, merge the metadata
    // and check if there are no conflicts.
    llvm::SmallVector<int, 4> merged_resource_indices;
    llvm::SmallVector<mlir::Attribute, 4> merged_resource_layouts;
    if (mlir::failed(MergeAttributes(
            src_cluster, src_resource_handle_indices_metadata,
            src_inferred_resource_handle_layouts_metadata,
            target_resource_handle_indices_metadata,
            target_inferred_resource_handle_layouts_metadata,
            &merged_resource_indices, &merged_resource_layouts)))
      return mlir::failure();

    target_cluster->setAttr(
        kNewResourceArgLayouts,
        builder.getArrayAttr(
            llvm::ArrayRef<mlir::Attribute>(merged_resource_layouts)));

    target_cluster->setAttr(
        kNewResourceLayoutIndices,
        builder.getI32VectorAttr(llvm::ArrayRef<int>(merged_resource_indices)));
  }

  // Extract shape metadata from src/target clusters.
  auto src_shape_layouts =
      src_cluster->getAttrOfType<mlir::ArrayAttr>(kShapeOpInputLayout);
  auto src_shape_op_indices =
      src_cluster->getAttrOfType<mlir::DenseIntElementsAttr>(
          kShapeOpInputLayoutIndices);
  auto target_shape_layouts =
      target_cluster->getAttrOfType<mlir::ArrayAttr>(kShapeOpInputLayout);
  auto target_shape_op_indices =
      target_cluster->getAttrOfType<mlir::DenseIntElementsAttr>(
          kShapeOpInputLayoutIndices);

  const bool should_merge_shape_metadata =
      (src_shape_layouts && src_shape_op_indices && target_shape_layouts &&
       target_shape_op_indices);

  // If only src cluster has shape metadata, copy shape metadata to target
  // cluster.
  if (src_shape_layouts && !target_shape_layouts) {
    target_cluster->setAttr(kShapeOpInputLayoutIndices, src_shape_op_indices);
    target_cluster->setAttr(kShapeOpInputLayout, src_shape_layouts);
  } else if (should_merge_shape_metadata) {
    // If both src/target clusters have shape metadata, merge the shape metadata
    // and set the merged metadata to target cluster.
    llvm::SmallVector<int, 4> merged_shape_indices;
    llvm::SmallVector<mlir::Attribute, 4> merged_shape_layouts;
    if (mlir::failed(MergeAttributes(
            src_cluster, src_shape_op_indices, src_shape_layouts,
            target_shape_op_indices, target_shape_layouts,
            &merged_shape_indices, &merged_shape_layouts)))
      return mlir::failure();

    target_cluster->setAttr(
        kShapeOpInputLayout,
        builder.getArrayAttr(
            llvm::ArrayRef<mlir::Attribute>(merged_shape_layouts)));

    target_cluster->setAttr(
        kShapeOpInputLayoutIndices,
        builder.getI32VectorAttr(llvm::ArrayRef<int>(merged_shape_indices)));
  }

  return mlir::success();
}

// Removes tf_device.Cluster ops if tf_device.Cluster is nested inside another
// cluster and it has same mesh specification as parent cluster.
mlir::LogicalResult InlineNestedDeviceClusters(mlir::ModuleOp module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_3(mht_3_v, 401, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "InlineNestedDeviceClusters");

  auto clusters = FindAllDeviceClusters(module);
  for (mlir::tf_device::ClusterOp cluster : clusters) {
    auto parent_cluster =
        cluster->getParentOfType<mlir::tf_device::ClusterOp>();
    if (!parent_cluster) continue;

    Mesh cluster_mesh;
    if (mlir::failed(ExtractMeshFromCluster(cluster, &cluster_mesh)))
      return mlir::failure();

    Mesh parent_cluster_mesh;
    if (mlir::failed(
            ExtractMeshFromCluster(parent_cluster, &parent_cluster_mesh)))
      return mlir::failure();

    if (parent_cluster_mesh != cluster_mesh) continue;

    // Found a tf_device.cluster that has same mesh specification as parent
    // enclosing cluster. Remove the child cluster and move all ops to parent
    // cluster instead.
    for (auto it : llvm::zip(cluster.GetBody().getTerminator()->getOperands(),
                             cluster.results())) {
      mlir::Value new_value = std::get<0>(it);
      mlir::Value value_to_replace = std::get<1>(it);
      value_to_replace.replaceAllUsesWith(new_value);
    }
    for (mlir::Operation& op :
         llvm::make_early_inc_range(cluster.GetBody().without_terminator())) {
      op.moveBefore(cluster);
    }

    if (mlir::failed(MergeClusterMetadata(cluster, parent_cluster)))
      return mlir::failure();

    cluster.erase();
  }
  return mlir::success();
}

// Clones an IfRegionOp 'if_region' and attributes and creates then/else regions
// with yield op and an empty block.
void CloneEmptyIfWithPredicate(mlir::TF::IfRegionOp if_region, const Mesh& mesh,
                               mlir::OpBuilder& builder, int* num_send_recvs,
                               mlir::MLIRContext* context,
                               mlir::TF::IfRegionOp* cloned_if_region_op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_4(mht_4_v, 449, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "CloneEmptyIfWithPredicate");

  // Create DTensorSend just before tf.If op before creating new cluster. The
  // DTensorSend op sends the predicate to `mesh` cluster with replicated
  // layout.
  mlir::TensorType predicate_tensor_type =
      if_region.cond().getType().cast<mlir::TensorType>();
  const std::string send_recv_key =
      absl::StrCat(kSendRecvKeyPrefix, *num_send_recvs);
  *num_send_recvs += 1;

  const Layout target_layout = Layout::ReplicatedOnMesh(mesh, 0);
  builder.create<mlir::TF::DTensorSend>(
      if_region.getLoc(), if_region.cond(),
      builder.getStringAttr(send_recv_key),
      mlir::dtensor::LayoutAttr::get(context, target_layout));

  // Create new cluster op that contains cloned if operation.
  auto new_cluster = builder.create<mlir::tf_device::ClusterOp>(
      if_region.getLoc(), llvm::SmallVector<mlir::Type, 4>{});
  new_cluster.body().push_back(new mlir::Block);
  builder.setInsertionPointToEnd(&new_cluster.GetBody());
  auto return_op = builder.create<mlir::tf_device::ReturnOp>(
      if_region.getLoc(), llvm::SmallVector<mlir::Value, 4>{});

  // Add DTensorRecv op inside new cluster that receives the cluster.
  builder.setInsertionPoint(return_op);
  auto recv_op = builder.create<mlir::TF::DTensorRecv>(
      if_region.getLoc(), predicate_tensor_type,
      builder.getStringAttr(send_recv_key),
      mlir::TF::ShapeAttr::get(context, predicate_tensor_type),
      mlir::dtensor::LayoutAttr::get(context, target_layout));

  // Clone tf.IfRegion op inside newly created cluster and make sure
  // that the predicate tensor is from DTensorRecv op created above.
  auto host_side_if = builder.create<mlir::TF::IfRegionOp>(
      if_region.getLoc(), llvm::SmallVector<mlir::Type, 4>{}, recv_op.output(),
      if_region.is_stateless(),
      GetUniqueControlflowFnName("cloned_if_then", builder),
      GetUniqueControlflowFnName("cloned_if_else", builder));
  *cloned_if_region_op = host_side_if;

  // Create empty then branch region.
  auto& then_branch = host_side_if.then_branch();
  then_branch.push_back(new mlir::Block);
  builder.setInsertionPointToEnd(&then_branch.front());
  builder.create<mlir::TF::YieldOp>(if_region.getLoc(),
                                    /*operands=*/llvm::ArrayRef<mlir::Value>{});

  // Create empty else branch region.
  auto& else_branch = host_side_if.else_branch();
  else_branch.push_back(new mlir::Block);
  builder.setInsertionPointToEnd(&else_branch.front());
  builder.create<mlir::TF::YieldOp>(if_region.getLoc(),
                                    /*operands=*/llvm::ArrayRef<mlir::Value>{});
  new_cluster->setAttr(kMeshAttr, builder.getStringAttr(mesh.ToString()));
}

// Verifies that send/recv ops are used for input output of cluster. That is,
// cluster should not have any input/output edges.
mlir::LogicalResult VerifyClusterInputOutput(
    mlir::tf_device::ClusterOp cluster) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_5(mht_5_v, 512, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "VerifyClusterInputOutput");

  if (cluster.getNumResults() > 0)
    return cluster->emitOpError(
        "found nested tf_device.Cluster op with outputs. Nested cluster must "
        "use send/recv instead.");

  mlir::LogicalResult result = mlir::success();
  mlir::visitUsedValuesDefinedAbove(
      cluster.body(), cluster.body(), [&](mlir::OpOperand* input) {
        if (!input->get().isa<mlir::BlockArgument>()) {
          result = cluster.emitOpError(
              "found nested tf_device.Cluster op with inputs. Nested cluster "
              "must use send/recv instead.");
          return;
        }
      });
  return result;
}

// Returns whether `cluster` is inside then branch of `if_op`.
bool IsInsideIfThenBranch(mlir::TF::IfRegionOp if_op,
                          mlir::tf_device::ClusterOp cluster) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_6(mht_6_v, 536, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "IsInsideIfThenBranch");

  assert(if_op->isProperAncestor(cluster));
  return if_op.then_branch().isAncestor(cluster->getParentRegion());
}

// Decomposes multi-mesh computation nested inside tf_if operations. See
// comments for `DecomposeControlflow()` function for details.
mlir::LogicalResult DecomposeIf(mlir::TF::IfRegionOp if_op,
                                mlir::MLIRContext* context,
                                int* num_control_flow_send_recvs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_7(mht_7_v, 548, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "DecomposeIf");

  auto nested_clusters = FindAllDeviceClusters(if_op);
  if (nested_clusters.empty()) return mlir::success();

  for (mlir::tf_device::ClusterOp nested_cluster : nested_clusters) {
    if (mlir::failed(VerifyClusterInputOutput(nested_cluster)))
      return mlir::failure();

    Mesh nested_mesh;
    if (mlir::failed(ExtractMeshFromCluster(nested_cluster, &nested_mesh)))
      return mlir::failure();

    mlir::OpBuilder builder(if_op);
    mlir::TF::IfRegionOp cloned_if;
    CloneEmptyIfWithPredicate(if_op, nested_mesh, builder,
                              num_control_flow_send_recvs, context, &cloned_if);

    // Find nested clusters in then/else branch of original `if_op` and
    // move all inner ops inside nested cluster to `tf_cloned` in
    // corresponding branch.
    if (IsInsideIfThenBranch(if_op, nested_cluster)) {
      mlir::Operation* then_branch_terminator =
          cloned_if.then_branch().begin()->getTerminator();
      auto& nested_cluster_operations =
          nested_cluster.GetBody().getOperations();
      cloned_if.then_branch().begin()->getOperations().splice(
          then_branch_terminator->getIterator(), nested_cluster_operations,
          nested_cluster_operations.begin(),
          std::prev(nested_cluster_operations.end()));
    } else {
      mlir::Operation* else_branch_terminator =
          cloned_if.else_branch().begin()->getTerminator();
      auto& nested_cluster_operations =
          nested_cluster.GetBody().getOperations();
      cloned_if.else_branch().begin()->getOperations().splice(
          else_branch_terminator->getIterator(), nested_cluster_operations,
          nested_cluster_operations.begin(),
          std::prev(nested_cluster_operations.end()));
    }
    nested_cluster.erase();
  }
  return mlir::success();
}

// Decomposes controlflows with nested mesh computations. When multi-mesh
// computation exists inside control flow operations like tf.If, then
// the control flow operations should be replicated to ensure correct execution
// semantics.
// For example:
//
// "tf_device.cluster"() ( {
//    %1 = "tf.G"() : () -> (tensor<i1>)
//    "tf.IfRegion"(%1) ({
//      "tf_device.cluster"() ( {
//        "tf.D"() {} : () -> ()
//        tf_device.return
//      }) {_mesh = "TPU|x=1|0|0|TPU:0"} : () -> ()
//
//      "tf.Yield"() : () -> ()
//    }, {
//    }) {is_stateless = false} : (tensor<i1>) -> ()
//    tf_device.return
//  }) {_mesh = "CPU|x=1|0|0|CPU:0"} : () -> ()
//
// Above computation includes TPU device computation that exists inside
// tf.If op in CPU mesh. In this case, tf.If op should be replicated to TPU
// device computation so that `tf.D` op is executed in sync with CPU side
// computation. After transformation in this function, above IR is changed to:
//
// "tf_device.cluster"() ( {
//      %1 = "tf.DTensorRecv"() : () -> tensor<i1>
//      "tf.IfRegion"(%1) ( {
//        "tf.D"() : () -> ()
//        "tf.Yield"() : () -> ()
//      },  {
//        "tf.Yield"() : () -> ()
//      }) {is_stateless = false} : (tensor<i1>) -> ()
//      tf_device.return
//    }) {_mesh = "TPU|x=1|0|0|TPU:0"} : () -> ()
//
// "tf_device.cluster"() ( {
//   %1 = "tf.G"() : () -> tensor<i1>
//   "tf.DTensorSend"(%1) : (tensor<i1>) -> ()
//   "tf.IfRegion"(%1) ( {
//     "tf.Yield"() : () -> ()
//    },  {
//     "tf.Yield"() : () -> ()
//    }) {is_stateless = false} : (tensor<i1>) -> ()
//    tf_device.return
// }) {_mesh = "CPU|x=1|0|0|CPU:0"} : () -> ()
//
// Note that:
//  1) Control flow is replicated.
//  2) DTensorSend/Recv ops are added to transfer predicate tensors for
//     control flow operations
mlir::LogicalResult DecomposeControlflow(mlir::MLIRContext* context,
                                         int* num_control_flow_send_recvs,
                                         mlir::ModuleOp module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_8(mht_8_v, 648, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "DecomposeControlflow");

  llvm::SmallVector<mlir::tf_device::ClusterOp, 4> clusters;
  // Identify all clusters in topological order.
  module.walk([&](mlir::tf_device::ClusterOp cluster) {
    clusters.emplace_back(cluster);
  });

  for (mlir::tf_device::ClusterOp cluster : clusters) {
    mlir::WalkResult walk_result = cluster->walk([&](mlir::Operation* op) {
      if (auto if_op = mlir::dyn_cast<mlir::TF::IfRegionOp>(op)) {
        if (mlir::failed(
                DecomposeIf(if_op, context, num_control_flow_send_recvs)))
          return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) return mlir::failure();
  }

  return mlir::success();
}

// Merges multiple tf_device.clusters with same mesh specification to a single
// mesh cluster.
mlir::LogicalResult MergeClusters(mlir::ModuleOp module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_9(mht_9_v, 675, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "MergeClusters");

  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");

  // Create global cluster for each mesh in entire computation.
  auto clusters = FindAllDeviceClusters(main_func);
  mlir::Block& func_block = *main_func.getBody().begin();
  mlir::OpBuilder builder(&func_block.front());
  std::map<Mesh, llvm::SmallVector<mlir::tf_device::ClusterOp, 4>> cluster_map;
  std::vector<Mesh> meshes;
  for (mlir::tf_device::ClusterOp cluster : clusters) {
    Mesh mesh;
    if (mlir::failed(ExtractMeshFromCluster(cluster, &mesh)))
      return mlir::failure();

    if (cluster_map.find(mesh) != cluster_map.end()) {
      cluster_map[mesh].emplace_back(cluster);
    } else {
      cluster_map[mesh] =
          llvm::SmallVector<mlir::tf_device::ClusterOp, 4>{cluster};
      meshes.push_back(std::move(mesh));
    }
  }

  // Reevaluate if this sort is necessary after b/186804270 is closed.
  std::sort(meshes.begin(), meshes.end(), [](const Mesh& a, const Mesh& b) {
    if (a.device_type() != b.device_type()) {
      return a.device_type() < b.device_type();
    }
    return a < b;
  });
  for (const Mesh& mesh : meshes) {
    const auto& mesh_cluster_list = cluster_map[mesh];
    llvm::SmallVector<mlir::Value, 4> merged_cluster_outputs;
    llvm::SmallVector<mlir::Value, 4> merged_return_values;
    llvm::SmallVector<mlir::Type, 4> merged_return_types;

    for (mlir::tf_device::ClusterOp cluster : mesh_cluster_list) {
      merged_cluster_outputs.insert(merged_cluster_outputs.end(),
                                    cluster.results().begin(),
                                    cluster.results().end());

      auto return_values = cluster.GetBody().getTerminator()->getOperands();
      merged_return_values.insert(merged_return_values.end(),
                                  return_values.begin(), return_values.end());

      auto return_type = cluster->getResultTypes();
      merged_return_types.insert(merged_return_types.end(), return_type.begin(),
                                 return_type.end());
    }

    // Create a single cluster op contains merged computations for `mesh`.
    builder.setInsertionPoint(&func_block.front());
    auto new_cluster = builder.create<mlir::tf_device::ClusterOp>(
        module.getLoc(), merged_return_types);
    new_cluster.body().push_back(new mlir::Block);
    new_cluster->setAttr(kMeshAttr, builder.getStringAttr(mesh.ToString()));

    // Move all ops inside clusters in cluster mesh to `new_cluster`.
    for (mlir::tf_device::ClusterOp cluster : mesh_cluster_list) {
      mlir::Block& cluster_body = cluster.GetBody();
      for (mlir::Operation& op_to_move :
           llvm::make_early_inc_range(cluster_body.without_terminator())) {
        for (mlir::OpOperand& use : op_to_move.getUses()) {
          auto return_op =
              llvm::dyn_cast<mlir::tf_device::ReturnOp>(use.getOwner());
          if (!return_op) continue;

          mlir::Value output = cluster.getResult(use.getOperandNumber());
          output.replaceUsesWithIf(use.get(), [](mlir::OpOperand& operand) {
            return operand.getOwner()
                       ->getParentOfType<mlir::tf_device::ClusterOp>() !=
                   nullptr;
          });
        }
        op_to_move.moveBefore(new_cluster.getBody(),
                              new_cluster.getBody()->end());
      }
    }

    builder.setInsertionPointToEnd(&new_cluster.GetBody());
    builder.create<mlir::tf_device::ReturnOp>(new_cluster.getLoc(),
                                              merged_return_values);

    // Replace return value usages.
    for (auto it :
         llvm::zip(merged_cluster_outputs, new_cluster.getResults())) {
      mlir::Value value_to_replace = std::get<0>(it);
      mlir::Value new_result_value = std::get<1>(it);
      value_to_replace.replaceAllUsesWith(new_result_value);
    }

    // Erase clusters in cluster_map now that all ops are moved.
    for (mlir::tf_device::ClusterOp cluster : mesh_cluster_list) {
      if (mlir::failed(MergeClusterMetadata(cluster, new_cluster)))
        return mlir::failure();

      cluster.erase();
    }
  }

  return mlir::success();
}

// Pass that merges multiple tf_device.Cluster ops for multi-mesh computation
// into a single cluster. After this pass, exactly one tf_device.Cluster op
// exists for each device mesh.
struct DTensorMergeClusters
    : public DTensorMergeClustersBase<DTensorMergeClusters> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_10(mht_10_v, 787, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "getDependentDialects");

    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSmerge_clustersDTcc mht_11(mht_11_v, 794, "", "./tensorflow/dtensor/mlir/merge_clusters.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    auto module = getOperation();
    if (mlir::failed(InlineNestedDeviceClusters(module)))
      return signalPassFailure();

    int num_controlflow_send_recv = 0;
    if (mlir::failed(
            DecomposeControlflow(&context, &num_controlflow_send_recv, module)))
      return signalPassFailure();

    if (mlir::failed(MergeClusters(module))) return signalPassFailure();

    llvm::SmallVector<mlir::tf_device::ClusterOp, 4> clusters;
    module.walk([&](mlir::tf_device::ClusterOp cluster) {
      clusters.emplace_back(cluster);
    });

    for (mlir::tf_device::ClusterOp cluster : clusters) {
      RemoveUnusedClusterResults(cluster);
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMergeClustersPass() {
  return std::make_unique<DTensorMergeClusters>();
}

}  // namespace dtensor
}  // namespace tensorflow
