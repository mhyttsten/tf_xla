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
class MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_clusterDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_clusterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_clusterDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {

using tensorflow::GraphDef;
using tensorflow::NodeDef;

void AddNodeToGraph(const NodeDef& node,
                    const std::vector<std::string>& cluster_names,
                    GraphDef* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_clusterDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.cc", "AddNodeToGraph");

  NodeDef* new_node = graph->add_node();
  new_node->set_op(node.op());
  new_node->set_name(node.name());
  new_node->set_device(node.device());
  // If the inputs are coming from a node which belongs to another cluster, then
  // those inputs are renamed to the source cluster name. Otherwise the original
  // input name is used.
  for (const std::string& node_input : node.input()) {
    bool input_from_cluster = false;
    for (const std::string& cluster_name : cluster_names) {
      if (StrContains(node_input, cluster_name) &&
          !StrContains(node.name(), cluster_name)) {
        new_node->add_input(cluster_name);
        input_from_cluster = true;
        break;
      }
    }
    if (!input_from_cluster) {
      new_node->add_input(node_input);
    }
  }
  for (const auto& attr : node.attr()) {
    (*new_node->mutable_attr())[attr.first] = attr.second;
  }
}

bool FindCluster(const ClusterFactoryInterface& cluster_factory,
                 const GraphDef& graph_def,
                 std::unordered_map<std::string, bool>* is_node_in_cluster,
                 std::vector<std::unique_ptr<Cluster>>* clusters) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSresolve_clusterDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.cc", "FindCluster");

  for (const NodeDef& node : graph_def.node()) {
    // If the node is not assigned to any cluster, then we check if it belong to
    // the cluster_factory.
    bool node_in_cluster = (*is_node_in_cluster)[node.name()];
    if (!node_in_cluster) {
      std::unique_ptr<Cluster> cluster =
          cluster_factory.CreateCluster(node, graph_def);
      if (cluster) {
        // Label all the nodes in is_node_in_cluster which are in this cluster
        // as belonged to this cluster.
        for (const NodeDef* cluster_node : cluster->GetNodes()) {
          (*is_node_in_cluster)[cluster_node->name()] = true;
        }
        clusters->push_back(std::move(cluster));
      }
    }
  }
  return (!clusters->empty());
}

std::unique_ptr<GraphDef> MaybeResolveClusters(
    const GraphDef& graph_def,
    const std::vector<ClusterFactoryInterface*>& cluster_factories) {
  std::unique_ptr<GraphDef> pruned_graph(new GraphDef);
  // The structure to keep track of which cluster each node is assigned to, and
  // to initialize them to all un-assigned,
  std::unordered_map<std::string, bool> is_node_in_cluster;
  for (const NodeDef& node : graph_def.node()) {
    is_node_in_cluster[node.name()] = false;
  }

  std::vector<std::string> cluster_names;
  std::vector<std::unique_ptr<Cluster>> all_clusters;
  // Find the clusters for all available cluster factories.
  for (const ClusterFactoryInterface* cluster_factory : cluster_factories) {
    std::vector<std::unique_ptr<Cluster>> clusters;
    if (FindCluster(*cluster_factory, graph_def, &is_node_in_cluster,
                    &clusters)) {
      for (auto itr = clusters.begin(); itr != clusters.end(); ++itr) {
        cluster_names.push_back((*itr)->GetName());
        (*itr)->CreateNodes();
        all_clusters.push_back(std::move(*itr));
      }
    }
  }

  for (const std::unique_ptr<Cluster>& cluster : all_clusters) {
    for (const std::unique_ptr<tensorflow::NodeDef>& src_node :
         cluster->GetNewNodes()) {
      // Add it to the output GraphDef.
      AddNodeToGraph(*src_node, cluster_names, pruned_graph.get());
    }
  }

  // Add any node which is not part of a cluster.
  for (const NodeDef& node : graph_def.node()) {
    bool node_in_cluster = is_node_in_cluster[node.name()];
    if (!node_in_cluster) {
      AddNodeToGraph(node, cluster_names, pruned_graph.get());
    }
  }

  if (pruned_graph->node_size() == 0) {
    return nullptr;
  } else {
    return pruned_graph;
  }
}

std::unique_ptr<GraphDef> MaybeReplaceCompositeSubgraph(
    const GraphDef& tf_graph) {
  SvdfClusterFactory svdf_cluster_factory;

  std::vector<ClusterFactoryInterface*> cluster_factories;
  cluster_factories.push_back(&svdf_cluster_factory);

  std::unique_ptr<GraphDef> pruned_graph =
      MaybeResolveClusters(tf_graph, cluster_factories);

  // Copy function definitions
  if (pruned_graph) {
    *(pruned_graph->mutable_library()) = tf_graph.library();
  }
  return pruned_graph;
}

}  // end namespace toco
