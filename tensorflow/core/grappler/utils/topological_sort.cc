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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc() {
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

#include "tensorflow/core/grappler/utils/topological_sort.h"

#include <algorithm>
#include <deque>
#include <unordered_map>

#include "absl/types/span.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

namespace {

std::vector<GraphView::Edge> MakeEphemeralEdges(
    const absl::Span<const TopologicalDependency> extra_dependencies) {
  std::vector<GraphView::Edge> ephemeral_edges;
  ephemeral_edges.reserve(extra_dependencies.size());
  for (const auto& dep : extra_dependencies) {
    ephemeral_edges.emplace_back(
        GraphView::OutputPort(dep.from, Graph::kControlSlot),
        GraphView::InputPort(dep.to, Graph::kControlSlot));
  }
  return ephemeral_edges;
}

// Kahn's algorithm is implemented.
// For details, see https://en.wikipedia.org/wiki/Topological_sorting
Status ComputeTopologicalOrder(
    const GraphDef& graph,
    const absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<int>* ready_nodes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/grappler/utils/topological_sort.cc", "ComputeTopologicalOrder");

  GraphTopologyView graph_view;
  TF_RETURN_IF_ERROR(graph_view.InitializeFromGraph(
      graph, MakeEphemeralEdges(extra_dependencies)));

  // Keep track of how many inputs are ready for the given node.
  std::vector<int> num_ready_inputs(graph.node_size(), 0);

  // We'll push index of ready nodes to this output vector.
  ready_nodes->reserve(graph.node_size());

  int front = 0;
  int back = 0;

  for (int i = 0; i < graph.node_size(); i++) {
    if (graph_view.GetFanin(i).empty()) {
      ready_nodes->push_back(i);
      back++;
    }
    if (IsMerge(graph.node(i))) {
      for (int input : graph_view.GetFanin(i)) {
        if (IsNextIteration(graph.node(input))) {
          num_ready_inputs[i]++;
        }
      }
    }
  }

  while (front != back) {
    int ready_node = (*ready_nodes)[front];
    for (int fanout : graph_view.GetFanout(ready_node)) {
      ++num_ready_inputs[fanout];
      const int max_size = graph_view.GetFanin(fanout).size();
      if (num_ready_inputs[fanout] == max_size) {
        ready_nodes->push_back(fanout);
        ++back;
      }
    }
    ++front;
  }

  if (back != graph_view.num_nodes()) {
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "The graph couldn't be sorted in topological order. Stalled "
                 "at node = "
              << graph.node(back).DebugString();
      for (int i = 0; i < graph_view.num_nodes(); ++i) {
        const int max_size = graph_view.GetFanin(i).size();
        if (num_ready_inputs[i] != max_size) {
          VLOG(1) << "Node not ready: " << graph.node(i).DebugString();
        }
      }
    }
    return errors::InvalidArgument(
        "The graph couldn't be sorted in topological order.");
  }
  return Status::OK();
}

}  // namespace

Status ComputeTopologicalOrder(
    const GraphDef& graph,
    const absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<const NodeDef*>* topo_order) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc mht_1(mht_1_v, 288, "", "./tensorflow/core/grappler/utils/topological_sort.cc", "ComputeTopologicalOrder");

  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(
      ComputeTopologicalOrder(graph, extra_dependencies, &ready_nodes));

  topo_order->reserve(ready_nodes.size());
  for (int ready_node_idx : ready_nodes) {
    topo_order->emplace_back(&graph.node(ready_node_idx));
  }

  return Status::OK();
}

Status ComputeTopologicalOrder(const GraphDef& graph,
                               std::vector<const NodeDef*>* topo_order) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc mht_2(mht_2_v, 305, "", "./tensorflow/core/grappler/utils/topological_sort.cc", "ComputeTopologicalOrder");

  return ComputeTopologicalOrder(graph, {}, topo_order);
}

Status ReversedTopologicalSort(GraphDef* graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc mht_3(mht_3_v, 312, "", "./tensorflow/core/grappler/utils/topological_sort.cc", "ReversedTopologicalSort");

  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(*graph, {}, &ready_nodes));
  std::reverse(ready_nodes.begin(), ready_nodes.end());
  PermuteNodesInPlace(graph, &ready_nodes, /*invert_permutation=*/true);
  return Status::OK();
}

Status TopologicalSort(GraphDef* graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sortDTcc mht_4(mht_4_v, 323, "", "./tensorflow/core/grappler/utils/topological_sort.cc", "TopologicalSort");

  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(*graph, {}, &ready_nodes));
  PermuteNodesInPlace(graph, &ready_nodes, /*invert_permutation=*/true);
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
