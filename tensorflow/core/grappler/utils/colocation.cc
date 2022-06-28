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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc() {
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

#include "tensorflow/core/grappler/utils/colocation.h"

#include <cstring>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {

// Find root node of the colocation group.
// The map is mapping from one node name to its parent. node_name is the
// starting node to search. By iteratively following the path from child to
// parent, we can find the root node for the colocation group that node_name
// belongs to.
string GetColocationGroupRoot(std::unordered_map<string, string>* map,
                              const string& node_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/grappler/utils/colocation.cc", "GetColocationGroupRoot");

  if (map->find(node_name) == map->end()) {
    // If node_name is not in the map, we create a new root node which points
    // to itself.
    map->insert({node_name, node_name});
    return node_name;
  }
  std::list<string> nodes_to_root;
  string cur = node_name;
  while ((*map)[cur] != cur) {
    // Backtracing the map until we reach the root node.
    nodes_to_root.push_back(cur);
    cur = (*map)[cur];
  }

  // Update the nodes on the path to the root node to point to the root as well,
  // so the further lookups can be faster.
  if (!nodes_to_root.empty()) {
    nodes_to_root.pop_back();
    for (const string& node : nodes_to_root) {
      (*map)[node] = cur;
    }
  }
  return cur;
}

// Merge two colocation groups into one.
// left and right is the root node of two colocation groups respectively.
void MergeColocationGroup(std::unordered_map<string, string>* map,
                          const string& left, const string& right) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("left: \"" + left + "\"");
   mht_1_v.push_back("right: \"" + right + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/grappler/utils/colocation.cc", "MergeColocationGroup");

  // Do nothing if left or right node is not in the map.
  if (map->find(left) == map->end() || map->find(right) == map->end()) {
    return;
  }
  if (left != right) {
    // Make the right node a child of the left node, which merges the two
    // groups.
    map->at(right) = left;
  }
}
}  // namespace

// Use of disjoint set algorithm to build the colocation groups from the input
// graph. The core data structure in use is a hash map from one node to its
// parent node. Whenever we see two nodes colocate with each other, we merge
// their colocation groups together. After we traverse all colocation pairs
// in the graph, we will have several disjoint sets. Then we pick the root node
// of each disjoint set as the representative node, and let all other nodes in
// the group colocate with the representative node.
void ReassignColocation(GraphDef* graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocationDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/grappler/utils/colocation.cc", "ReassignColocation");

  constexpr char kClassAttr[] = "_class";
  constexpr char kColocPrefix[] = "loc:@";

  // A hashmap that maps from a node name to its parent node name.
  std::unordered_map<string, string> coloc_groups;
  NodeMap node_map(graph);
  for (const auto& node : graph->node()) {
    auto iter = node.attr().find(kClassAttr);
    if (iter != node.attr().end() && iter->second.has_list()) {
      for (const auto& str : iter->second.list().s()) {
        size_t pos = str.find(kColocPrefix);
        if (pos == 0) {
          // After we find a colocation, update the colocation groups.
          string colocate_node = str.substr(pos + strlen(kColocPrefix));
          MergeColocationGroup(
              &coloc_groups, GetColocationGroupRoot(&coloc_groups, node.name()),
              GetColocationGroupRoot(&coloc_groups, colocate_node));
        }
      }
    }
  }

  // We use the root node of each colocation groups as its representative
  // node. For each node in one group, colocate with the representative node
  // if the node is in the graph.
  for (const auto& pair : coloc_groups) {
    if (pair.first != pair.second) {
      // This is a child node.
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {
        // Colocate this node with the root node.
        AttrValue new_value;
        new_value.mutable_list()->add_s(
            kColocPrefix + GetColocationGroupRoot(&coloc_groups, pair.first));
        node->mutable_attr()->erase(kClassAttr);
        node->mutable_attr()->insert({kClassAttr, new_value});
      }
    } else {
      // This is a root node. Clear the _class attribute.
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {  // root node should always exist in the graph as guaranteed
                   // by order of merging. Just put check here to ensure safety.
        node->mutable_attr()->erase(kClassAttr);
      }
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
