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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc() {
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

#include "tensorflow/core/grappler/utils/scc.h"
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

// Data structure used to store data for Tarjan's Strongly Connected
// Components algorithm.
struct SCCNodeData {
  SCCNodeData()
      : node(nullptr),
        index(-1),
        lowlink(-1),
        onstack(false),
        caller(nullptr),
        caller_loop_location(-1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/grappler/utils/scc.cc", "SCCNodeData");
}
  void ResetStack(int new_index, SCCNodeData* new_caller) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/grappler/utils/scc.cc", "ResetStack");

    index = new_index;
    lowlink = new_index;
    onstack = true;
    caller = new_caller;
    caller_loop_location = 0;
  }
  const NodeDef* node;
  int index;
  int lowlink;
  bool onstack;
  std::vector<SCCNodeData*> children;
  // StrongConnect "call stack" storage.
  SCCNodeData* caller;       // Node calling StrongConnect
  int caller_loop_location;  // Index in parent StrongConnect for loop
};

// Core DFS step of Tarjan's Strongly Connected Component algorithm
// (implemented using iteration instead of recursion).
void StrongConnect(SCCNodeData* v, std::stack<SCCNodeData*>* stack, int* index,
                   std::unordered_map<const NodeDef*, int>* components,
                   int* scc_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/grappler/utils/scc.cc", "StrongConnect");

  // Iterative version of Tarjan's StrongConnect function.
  // The "call stack" state is composed of a SCCNodeData's caller and
  // caller_loop_location properties.
  v->ResetStack(*index /* index */, nullptr /* caller */);
  ++*index;
  stack->push(v);

  // No one put v on a StrongConnect call stack, reset caller values.
  v->caller = nullptr;
  v->caller_loop_location = 0;

  SCCNodeData* last = v;
  while (true) {
    if (last->caller_loop_location < last->children.size()) {
      // Recursive equivalent: Looping over the children of v (possibly
      // continuing at v->caller_loop_location after having finished a
      // recursive call.
      SCCNodeData* w = last->children[last->caller_loop_location];
      ++(last->caller_loop_location);  // For loop iterator increment
      if (w->index == -1) {
        w->ResetStack(*index /* index */, last /* caller */);
        ++*index;
        stack->push(w);
        last = w;
      } else if (w->onstack == true) {
        last->lowlink = std::min(last->lowlink, w->index);
      }
    } else {
      // At the end of v's children
      if (last->lowlink == last->index) {
        // v is the root of a strongly connected component
        SCCNodeData* top;
        while (true) {
          top = stack->top();
          stack->pop();
          top->onstack = false;
          (*components)[top->node] = *scc_index;
          if (top == last) {
            break;
          }
        }
        ++*scc_index;
      }

      // Go up the recursive call stack
      SCCNodeData* next_last = last->caller;
      if (next_last == nullptr) {
        // All nodes have been seen; finished.
        break;
      } else {
        next_last->lowlink = std::min(next_last->lowlink, last->lowlink);
        last = next_last;
      }
    }
  }
}

// This is an implementation of Tarjan's Strongly Connected Components
// DFS algorithm.  Most of the hard work is done in the function
// StrongConnect, which is an iterative reimplementation of the
// recursive version described here:
//   https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//
// The edges for the purpose of this algorithm are directed from input
// to op (the reverse of the declarations of the NodeDef, which
// contain in-edges)
void StronglyConnectedComponents(
    const GraphDef& graph, std::unordered_map<const NodeDef*, int>* components,
    int* num_components) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc mht_3(mht_3_v, 306, "", "./tensorflow/core/grappler/utils/scc.cc", "StronglyConnectedComponents");

  std::stack<SCCNodeData*> stack;
  std::unordered_map<string, SCCNodeData*> name_to_data;
  std::vector<SCCNodeData> node_data_container;
  node_data_container.reserve(graph.node_size());
  std::unordered_map<const NodeDef*, SCCNodeData*> node_to_data;

  for (const NodeDef& node : graph.node()) {
    SCCNodeData node_data;
    node_data.node = &node;
    node_data_container.push_back(node_data);
    name_to_data[node.name()] = &(*node_data_container.rbegin());
    node_to_data[&node] = &(*node_data_container.rbegin());
  }

  // Create a list of top-level parents (add them to object queue)
  // Also create a mapping from nodes to their children.
  // Inputs might not be present if called on a subgraph.
  for (const NodeDef& node : graph.node()) {
    for (const string& input : node.input()) {
      auto it = name_to_data.find(NodeName(input));
      if (it != name_to_data.end()) {
        it->second->children.push_back(node_to_data[&node]);
      }
    }
  }

  components->clear();
  *num_components = 0;
  int index = 0;
  for (auto& v : node_data_container) {
    if (v.index == -1) {
      // Node has not yet been visited.  Start a DFS at v.
      StrongConnect(&v, &stack, &index, components, num_components);
    }
  }

  std::vector<int> counts_per_component(*num_components, 0);
  for (auto& component : *components) {
    DCHECK(component.second >= 0);
    DCHECK(component.second < *num_components);
    counts_per_component[component.second]++;
  }
  bool has_single_element_component = false;
  for (auto& component : *components) {
    if (counts_per_component[component.second] == 1) {
      component.second = -1;
      (*num_components)--;
      has_single_element_component = true;
    }
  }
  if (has_single_element_component) {
    (*num_components) += 1;
  }
}

int IdentifyLoops(const GraphDef& graph,
                  std::unordered_map<const NodeDef*, std::vector<int>>* loops) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsccDTcc mht_4(mht_4_v, 366, "", "./tensorflow/core/grappler/utils/scc.cc", "IdentifyLoops");

  int num_components = 0;
  std::unordered_map<const NodeDef*, int> components;
  StronglyConnectedComponents(graph, &components, &num_components);
  if (num_components <= 1) {
    if (!components.empty() && components.begin()->second == -1) {
      return 0;
    }
  }

  std::unordered_map<int, std::vector<const NodeDef*>> component_ids;
  for (const auto it : components) {
    int id = it.second;
    if (id < 0) {
      continue;
    }
    component_ids[id].push_back(it.first);
  }

  int loop_id = 0;
  for (const auto& component : component_ids) {
    const std::vector<const NodeDef*>& component_nodes = component.second;
    std::vector<std::pair<NodeDef*, string>> next_iter_nodes;
    GraphDef subgraph;
    std::unordered_map<const NodeDef*, const NodeDef*> subgraph_mapping;

    for (const auto& component_node : component_nodes) {
      NodeDef* node = subgraph.add_node();
      *node = *component_node;
      subgraph_mapping[node] = component_node;
      if (IsNextIteration(*node)) {
        CHECK_EQ(1, node->input_size());
        next_iter_nodes.emplace_back(node, node->input(0));
      }
    }
    if (next_iter_nodes.size() == 1) {
      for (const auto& component_node : component_nodes) {
        (*loops)[component_node].push_back(loop_id);
      }
      ++loop_id;
    } else {
      for (int i = 0; i < next_iter_nodes.size(); ++i) {
        for (int j = 0; j < next_iter_nodes.size(); ++j) {
          next_iter_nodes[j].first->clear_input();
          if (i == j) {
            *next_iter_nodes[j].first->add_input() = next_iter_nodes[j].second;
          }
        }
        int num_components = 0;
        std::unordered_map<const NodeDef*, int> components;
        StronglyConnectedComponents(subgraph, &components, &num_components);
        CHECK_GE(num_components, 1);
        for (const auto it : components) {
          int id = it.second;
          if (id < 0) {
            continue;
          }
          (*loops)[subgraph_mapping[it.first]].push_back(loop_id);
        }
        ++loop_id;
      }
    }
  }

  return loop_id;
}

}  // namespace grappler
}  // namespace tensorflow
