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
class MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/algorithm.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
template <typename T>
void DFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                   const std::function<void(T)>& enter,
                   const std::function<void(T)>& leave,
                   const NodeComparator& stable_comparator,
                   const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/graph/algorithm.cc", "DFSFromHelper");

  // Stack of work to do.
  struct Work {
    T node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    auto add_work = [&visited, &stack](Node* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/graph/algorithm.cc", "lambda");

      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          nodes_sorted.emplace_back(out_edge->dst());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* out : nodes_sorted) {
        add_work(out);
      }
    } else {
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          add_work(out_edge->dst());
        }
      }
    }
  }
}
}  // namespace

void DFS(const Graph& g, const std::function<void(Node*)>& enter,
         const std::function<void(Node*)>& leave,
         const NodeComparator& stable_comparator,
         const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/graph/algorithm.cc", "DFS");

  DFSFromHelper(g, {g.source_node()}, enter, leave, stable_comparator,
                edge_filter);
}

void DFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
             const std::function<void(Node*)>& enter,
             const std::function<void(Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/graph/algorithm.cc", "DFSFrom");

  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void DFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
             const std::function<void(const Node*)>& enter,
             const std::function<void(const Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/graph/algorithm.cc", "DFSFrom");

  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter,
                const std::function<void(Node*)>& leave,
                const NodeComparator& stable_comparator,
                const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/graph/algorithm.cc", "ReverseDFS");

  ReverseDFSFrom(g, {g.sink_node()}, enter, leave, stable_comparator,
                 edge_filter);
}

namespace {

template <typename T>
void ReverseDFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                          const std::function<void(T)>& enter,
                          const std::function<void(T)>& leave,
                          const NodeComparator& stable_comparator,
                          const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_6(mht_6_v, 315, "", "./tensorflow/core/graph/algorithm.cc", "ReverseDFSFromHelper");

  // Stack of work to do.
  struct Work {
    T node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    auto add_work = [&visited, &stack](T out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_7(mht_7_v, 347, "", "./tensorflow/core/graph/algorithm.cc", "lambda");

      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<T> nodes_sorted;
      for (const Edge* in_edge : n->in_edges()) {
        if (!edge_filter || edge_filter(*in_edge)) {
          nodes_sorted.emplace_back(in_edge->src());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (T in : nodes_sorted) {
        add_work(in);
      }
    } else {
      for (const Edge* in_edge : n->in_edges()) {
        if (!edge_filter || edge_filter(*in_edge)) {
          add_work(in_edge->src());
        }
      }
    }
  }
}

}  // namespace

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
                    const std::function<void(const Node*)>& enter,
                    const std::function<void(const Node*)>& leave,
                    const NodeComparator& stable_comparator,
                    const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_8(mht_8_v, 384, "", "./tensorflow/core/graph/algorithm.cc", "ReverseDFSFrom");

  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
                    const std::function<void(Node*)>& enter,
                    const std::function<void(Node*)>& leave,
                    const NodeComparator& stable_comparator,
                    const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_9(mht_9_v, 395, "", "./tensorflow/core/graph/algorithm.cc", "ReverseDFSFrom");

  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void GetPostOrder(const Graph& g, std::vector<Node*>* order,
                  const NodeComparator& stable_comparator,
                  const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_10(mht_10_v, 404, "", "./tensorflow/core/graph/algorithm.cc", "GetPostOrder");

  order->clear();
  DFS(g, nullptr, [order](Node* n) { order->push_back(n); }, stable_comparator,
      edge_filter);
}

void GetReversePostOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator,
                         const EdgeFilter& edge_filter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_11(mht_11_v, 415, "", "./tensorflow/core/graph/algorithm.cc", "GetReversePostOrder");

  GetPostOrder(g, order, stable_comparator, edge_filter);
  std::reverse(order->begin(), order->end());
}

bool PruneForReverseReachability(Graph* g,
                                 std::unordered_set<const Node*> start) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_12(mht_12_v, 424, "", "./tensorflow/core/graph/algorithm.cc", "PruneForReverseReachability");

  // Compute set of nodes that we need to traverse in order to reach
  // the nodes in "start" by performing a breadth-first search from those
  // nodes, and accumulating the visited nodes.
  std::vector<bool> visited(g->num_node_ids());
  for (auto node : start) {
    visited[node->id()] = true;
  }
  std::deque<const Node*> queue(start.begin(), start.end());
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    for (const Node* in : n->in_nodes()) {
      if (!visited[in->id()]) {
        visited[in->id()] = true;
        queue.push_back(in);
        VLOG(2) << "Reverse reach : " << n->name() << " from " << in->name();
      }
    }
  }

  // Make a pass over the graph to remove nodes not in "visited".
  bool any_removed = false;
  for (int i = 0; i < visited.size(); ++i) {
    if (!visited[i]) {
      Node* n = g->FindNodeId(i);
      if (n != nullptr && !n->IsSource() && !n->IsSink()) {
        g->RemoveNode(n);
        any_removed = true;
      }
    }
  }
  return any_removed;
}

bool FixupSourceAndSinkEdges(Graph* g) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithmDTcc mht_13(mht_13_v, 462, "", "./tensorflow/core/graph/algorithm.cc", "FixupSourceAndSinkEdges");

  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.
  bool changed = false;
  for (Node* n : g->nodes()) {
    if (!n->IsSource() && n->in_edges().empty()) {
      g->AddControlEdge(g->source_node(), n,
                        true /* skip test for duplicates */);
      changed = true;
    }
    if (!n->IsSink() && n->out_edges().empty()) {
      g->AddControlEdge(n, g->sink_node(), true /* skip test for duplicates */);
      changed = true;
    }
  }
  return changed;
}

}  // namespace tensorflow
