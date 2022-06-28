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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc() {
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
#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"

#include <queue>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

// A helper for rewriting nodes assigned to a virtual composite device.
class ReplicateHelper {
 public:
  // Initialize replicated nodes with nullptr.
  Status InitializeNode(const Node* node, int num_allowed_devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "InitializeNode");

    if (replicated_nodes_map_.find(node) != replicated_nodes_map_.end()) {
      return errors::InvalidArgument("Node ", node->name(),
                                     " has been replicated.");
    }
    std::vector<Node*> replicated_nodes(num_allowed_devices, nullptr);
    replicated_nodes_map_.emplace(node, std::move(replicated_nodes));
    return Status::OK();
  }

  // Replicate the given node to an allowed device.
  Status ReplicateNode(const Node* node,
                       const std::vector<string>& allowed_devices,
                       int allowed_device_index, Graph* graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicateNode");

    auto& replicated_nodes = replicated_nodes_map_.at(node);
    if (replicated_nodes[allowed_device_index] != nullptr) {
      return Status::OK();
    }
    const auto& device = allowed_devices.at(allowed_device_index);
    NodeDef node_def = node->def();
    const string suffix = strings::StrCat("/R", allowed_device_index);
    node_def.set_name(graph->NewName(strings::StrCat(node_def.name(), suffix)));
    TF_ASSIGN_OR_RETURN(Node * replicated_node, graph->AddNode(node_def));
    replicated_node->set_assigned_device_name(device);
    if (replicated_node->IsArg()) {
      replicated_node->AddAttr("sub_index", allowed_device_index);
    }
    replicated_nodes[allowed_device_index] = replicated_node;
    return Status::OK();
  }

  // Replace an edge (a regular device -> composite device) with
  // N edges (a regular device -> allowed devices).
  void ReplicateFromRegularDeviceToCompositeDevice(const Edge* edge,
                                                   Graph* graph) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicateFromRegularDeviceToCompositeDevice");

    Node* src = edge->src();
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    for (Node* dst : dst_replicated_nodes) {
      // Skip a replicated dst node without any consumer.
      if (dst == nullptr) {
        continue;
      }
      graph->AddEdge(src, edge->src_output(), dst, edge->dst_input());
    }
  }

  // Replace an edge (composite device -> composite device) with
  // N edges (allowed devices -> allowed devices).
  Status ReplicateFromCompositeDeviceToCompositeDevice(
      const Edge* edge, const std::vector<string>& allowed_devices,
      Graph* graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicateFromCompositeDeviceToCompositeDevice");

    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    if (src_replicated_nodes.size() != dst_replicated_nodes.size()) {
      return errors::InvalidArgument(
          "Nodes assigned to the same composite device should have the "
          "same number of replicated nodes. Found an edge from node ",
          edge->src()->name(), " (", src_replicated_nodes.size(),
          " replicated nodes) to node ", edge->dst()->name(), " (",
          dst_replicated_nodes.size(), " replicated nodes).");
    }
    for (int i = 0; i < src_replicated_nodes.size(); ++i) {
      Node* dst = dst_replicated_nodes.at(i);
      // Skip a replicated dst node without any consumer.
      if (dst == nullptr) {
        continue;
      }
      TF_RETURN_IF_ERROR(ReplicateNode(edge->src(), allowed_devices, i, graph));
      graph->AddEdge(src_replicated_nodes.at(i), edge->src_output(), dst,
                     edge->dst_input());
    }
    return Status::OK();
  }

  // Data edge: replace an edge (composite device -> a regular device) with
  // one edge (one allowed device -> a regular device).
  // Control edge: replace an edge (composite device -> a regular device) with
  // N edges (allowed devices -> a regular device).
  Status ReplicateFromCompositeDeviceToRegularDevice(
      const Edge* edge, const std::vector<string>& allowed_devices,
      Graph* graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicateFromCompositeDeviceToRegularDevice");

    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    Node* dst = edge->dst();
    const string& dst_device = dst->assigned_device_name();
    bool found_src_node = false;
    for (int i = 0; i < allowed_devices.size(); ++i) {
      if (allowed_devices.at(i) == dst_device) {
        TF_RETURN_IF_ERROR(
            ReplicateNode(edge->src(), allowed_devices, i, graph));
        graph->AddEdge(src_replicated_nodes.at(i), edge->src_output(), dst,
                       edge->dst_input());
        found_src_node = true;
        break;
      }
    }
    if (!found_src_node) {
      for (int i = 0; i < allowed_devices.size(); ++i) {
        TF_RETURN_IF_ERROR(
            ReplicateNode(edge->src(), allowed_devices, i, graph));
      }
      if (edge->IsControlEdge()) {
        for (Node* replicated_node : src_replicated_nodes) {
          // Duplication check in `Graph::AddControlEdge` is expensive for the
          // dst node with a lot of input edges. Here each (src, dst) pair
          // will only occur once so it is safe to skip the duplication check.
          graph->AddControlEdge(replicated_node, dst,
                                /*allow_duplicates=*/true);
        }
        return Status::OK();
      }
      if (edge->src()->type_string() == "_Arg") {
        // This happens when the dst node runs on a host CPU and
        // captures a function with an arg node assigned to the same
        // composite device (e.g. ScanDataset).
        // For this case, we insert a PackOp between replicated nodes and the
        // dst node. The dst node is responsible for unpacking the packed
        // tensor.
        // Add '/Packed' as a substring to the name of the new node, which
        // could be helpful when debugging the graph.
        NodeDefBuilder pack_builder(
            graph->NewName(absl::StrCat(edge->src()->name(), "/Packed")),
            "Pack");
        const int num_replicas = src_replicated_nodes.size();
        pack_builder.Attr("N", num_replicas);
        const DataType dtype = edge->src()->output_type(edge->src_output());
        pack_builder.Attr("T", dtype);
        std::vector<NodeDefBuilder::NodeOut> inputs;
        inputs.reserve(src_replicated_nodes.size());
        for (Node* replicated_node : src_replicated_nodes) {
          inputs.emplace_back(NodeDefBuilder::NodeOut{
              replicated_node->name(), edge->src_output(), dtype});
        }
        pack_builder.Input(inputs);
        NodeDef pack_def;
        TF_RETURN_IF_ERROR(pack_builder.Finalize(&pack_def));
        TF_ASSIGN_OR_RETURN(Node * pack_node, graph->AddNode(pack_def));
        pack_node->set_assigned_device_name(dst->assigned_device_name());
        for (int i = 0; i < src_replicated_nodes.size(); ++i) {
          graph->AddEdge(src_replicated_nodes[i], edge->src_output(), pack_node,
                         i);
        }
        graph->AddEdge(pack_node, /*x=*/0, dst, edge->dst_input());
      } else {
        return errors::InvalidArgument(
            "Dst node should be assigned to an allowed device. Found an "
            "edge from node ",
            edge->src()->name(), " assigned to ",
            edge->src()->assigned_device_name(), " to node ", dst->name(),
            " assigned to ", dst_device);
      }
    }
    return Status::OK();
  }

 private:
  // Map from original nodes to corresponding replicated nodes.
  absl::flat_hash_map<const Node*, std::vector<Node*>> replicated_nodes_map_;
};

// Replicate the nodes in cluster_nodes and update edges.
Status ReplicateNodesAndEdges(const std::vector<string>& allowed_devices,
                              absl::flat_hash_map<Node*, int>* cluster_nodes,
                              ReplicateHelper* helper, Graph* graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_5(mht_5_v, 380, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicateNodesAndEdges");

  // Contains nodes in cluster_nodes whose out nodes are all on physical
  // devices.
  std::queue<Node*> nodes_ready_to_delete;
  for (auto& pair : *cluster_nodes) {
    Node* node = pair.first;
    for (const Edge* edge : node->out_edges()) {
      Node* dst = edge->dst();
      if (dst->assigned_device_name() != node->assigned_device_name()) {
        // The dst node is assigned to a different device.
        TF_RETURN_IF_ERROR(helper->ReplicateFromCompositeDeviceToRegularDevice(
            edge, allowed_devices, graph));
        --pair.second;
      }
    }
    // Node is ready to delete when all its consumer nodes are assigned to a
    // physical device.
    if (cluster_nodes->at(node) == 0) {
      nodes_ready_to_delete.push(node);
    }
  }

  while (!nodes_ready_to_delete.empty()) {
    Node* node = nodes_ready_to_delete.front();
    nodes_ready_to_delete.pop();

    // Update input edges.
    for (const Edge* edge : node->in_edges()) {
      Node* src = edge->src();
      if (src->assigned_device_name() != node->assigned_device_name()) {
        // The source node is assigned to a different device.
        helper->ReplicateFromRegularDeviceToCompositeDevice(edge, graph);
      } else {
        // The source node is assigned to the same composite device.
        TF_RETURN_IF_ERROR(
            helper->ReplicateFromCompositeDeviceToCompositeDevice(
                edge, allowed_devices, graph));
        if (--(*cluster_nodes)[src] == 0) {
          nodes_ready_to_delete.push(src);
        }
      }
    }

    // Remove the original node.
    cluster_nodes->erase(node);
    graph->RemoveNode(node);
  }
  return Status::OK();
}

}  // namespace

Status ReplicatePerReplicaNodesInFunctionGraph(
    const absl::flat_hash_map<string, const std::vector<string>*>&
        composite_devices,
    Graph* graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodesDTcc mht_6(mht_6_v, 438, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes.cc", "ReplicatePerReplicaNodesInFunctionGraph");

  std::set<string> composite_device_names;
  for (const auto& it : composite_devices) {
    composite_device_names.insert(it.first);
  }
  // Map from a composite device to a cluster of nodes assigned to the
  // composite device and the numbers of their out edges to process.
  absl::flat_hash_map<string, absl::flat_hash_map<Node*, int>>
      composite_device_to_cluster_nodes;
  for (Node* n : graph->op_nodes()) {
    if (composite_device_names.find(n->assigned_device_name()) !=
        composite_device_names.end()) {
      // TODO(b/145922293): Validate that an _Arg node assigned to a
      // CompositeDevice should have an attribute indicating that the _Arg node
      // represents a packed input.
      composite_device_to_cluster_nodes[n->assigned_device_name()].emplace(
          n, n->out_edges().size());
    }
  }

  for (auto& it : composite_device_to_cluster_nodes) {
    const std::vector<string>& allowed_devices =
        *composite_devices.at(it.first);
    if (allowed_devices.empty()) {
      return errors::InvalidArgument("No allowed device of composite device: ",
                                     it.first);
    }
    absl::flat_hash_map<Node*, int>& cluster_nodes = it.second;
    if (allowed_devices.size() == 1) {
      // Reuse the original nodes if there is only one allowed device.
      for (const auto& pair : it.second) {
        Node* n = pair.first;
        n->set_assigned_device_name(allowed_devices.at(0));
        if (n->IsArg()) {
          n->AddAttr("sub_index", 0);
        }
      }
      continue;
    }
    ReplicateHelper helper;
    for (const auto& pair : cluster_nodes) {
      TF_RETURN_IF_ERROR(
          helper.InitializeNode(pair.first, allowed_devices.size()));
    }

    TF_RETURN_IF_ERROR(ReplicateNodesAndEdges(allowed_devices, &cluster_nodes,
                                              &helper, graph));

    if (!cluster_nodes.empty()) {
      return errors::InvalidArgument(
          "There are still ", cluster_nodes.size(),
          " nodes on CompositiveDevice ",
          cluster_nodes.begin()->first->assigned_device_name());
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
