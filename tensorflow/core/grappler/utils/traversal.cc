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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/traversal.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"

namespace tensorflow {
namespace grappler {

namespace {

struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/grappler/utils/traversal.cc", "DfsStackElem");
}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/grappler/utils/traversal.cc", "DfsStackElem");
}

  // Index of the node in the graph ∊ [0, num_nodes).
  int node;
  // `True` if visited all the input/output nodes (pushed all input/output nodes
  // to the stack).
  bool children_visited;
  // Index of the node in the graph, from which we entered the `node`.
  int src;
};

enum class NodeState { kNotVisited, kVisiting, kDone };

}  // namespace

void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  const TraversalDirection direction,
                  const DfsPredicates& predicates,
                  const DfsCallbacks& callbacks) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/grappler/utils/traversal.cc", "DfsTraversal");

  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());

  for (const NodeDef* node : from) {
    const absl::optional<int> node_idx = graph_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value()) << "Illegal start node: " << node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }

  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();

    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;

    // Skip nodes that we should not enter.
    if (predicates.enter && !predicates.enter(graph_view.GetNode(w.node))) {
      state = NodeState::kDone;
      continue;
    }

    // We've processed all the children of this node.
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(graph_view.GetNode(w.node));
      }
      continue;
    }

    // Loop detected.
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(graph_view.GetNode(w.src),
                               graph_view.GetNode(w.node));
      }
      continue;
    }

    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(graph_view.GetNode(w.node));
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.emplace_back(w.node, true, w.src);

    // Check if we can continue traversal from the current node.
    if (predicates.advance && !predicates.advance(graph_view.GetNode(w.node))) {
      continue;
    }

    // Now enqueue the fanin/fanout nodes.
    if (direction == TraversalDirection::kFollowInputs) {
      for (const int fanin : graph_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    } else {
      for (const int fanout : graph_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}

void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsCallbacks& callbacks) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStraversalDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/grappler/utils/traversal.cc", "DfsTraversal");

  DfsTraversal(graph_view, from, direction, {}, callbacks);
}

}  // namespace grappler
}  // namespace tensorflow
