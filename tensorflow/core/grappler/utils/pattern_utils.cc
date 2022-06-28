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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc() {
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

#include "tensorflow/core/grappler/utils/pattern_utils.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"

namespace tensorflow {
namespace grappler {
namespace utils {

inline const bool IsCommutativeOp(const string& op) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/grappler/utils/pattern_utils.cc", "IsCommutativeOp");

  // TODO(intel-tf): Add more ops to this list if needed.
  static const auto* commutative_ops =
      new absl::flat_hash_set<string>({"Add", "AddV2", "Mul"});
  return commutative_ops->contains(op);
}

// op1 is an op name in the pattern and it could be wildcard `*` or some
// registered op in tensorflow. op2 is an op name in the computation graph and
// is always one of the registered ops in tensorflow.
inline bool IsSame(string op1, string op2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op1: \"" + op1 + "\"");
   mht_1_v.push_back("op2: \"" + op2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/grappler/utils/pattern_utils.cc", "IsSame");
 return op1 == "*" || op1 == op2; }

// A subgraph pattern syntax implicitly defines a DAG having a single root. We
// traverse the syntax DAG in DFS manner. This function finds a match for
// current root of the pattern with the current node and recursively matches
// children subpatterns with the children of current node.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::DoesOpTypePatternMatch(
    const OpTypePattern& pattern, MutableNodeView* node_view,
    NodeViewMatch* match) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/grappler/utils/pattern_utils.cc", "SubGraphMatcher<MatchingDirection::kFollowInputs>::DoesOpTypePatternMatch");

  // Currently no control inputs and outputs are allowed.
  if (node_view->NumControllingFanins() > 0 ||
      node_view->NumControlledFanouts() > 0)
    return false;

  bool op_type_matched = false;
  if (pattern.op == "*") {
    op_type_matched = true;
  } else {
    // The op field string of current pattern might express an op among multiple
    // op types (mutually exclusive) separated by '|'.
    std::vector<string> op_list = str_util::Split(pattern.op, '|');
    for (const string& op : op_list) {
      if (node_view->node()->op() == op) {
        op_type_matched = true;
        break;
      }
    }
  }
  if (op_type_matched) {
    // If op type matches and current node is visited first time, insert current
    // node to node_label_to_index_ map with the current label as the key.
    // Multiple occurances of same label in the pattern syntax indicates that
    // the same node needs to be visited for each of such occurances. Hence
    // subsequent visits should find the corresponding label in the map as a key
    // and the current node should be the value for that key.
    if (node_label_to_index_.find(pattern.label) ==
        node_label_to_index_.end()) {
      node_label_to_index_[pattern.label] = node_view->node_index();
      // Bookkeeping
      matched_node_indices_.insert(node_view->node_index());
      if (pattern.node_status == NodeStatus::kRemove) {
        remove_node_indices_.insert(node_view->node_index());
      }
    } else if (node_label_to_index_[pattern.label] != node_view->node_index()) {
      return false;  // label constraint could not be satisfied.
    } else {
      DCHECK(node_label_to_index_[pattern.label] == node_view->node_index());
    }
  } else {
    return false;
  }
  // Current root of the pattern syntax is matched with the current node.
  match->node_view = node_view;

  // Go for matching child subpattern.
  if (!pattern.children.empty()) {
    // Currently only direction toward inputs is implemented.
    auto graph_children = node_view->GetRegularFanins();
    int num_children = graph_children.size();
    if (num_children != pattern.children.size()) {
      return false;
    } else {
      // A pattern is a graph that we would like to match with a subgraph of
      // a tensorflow computation graph. We travese both pattern-graph and the
      // given graph in DFS manner and try to find one-to-one mapping between
      // the nodes. However, commutative binary ops (e.g., Add, AddV2, Mul
      // etc.) in the computation graph can have their inputs in different order
      // than the pattern syntax graph. To allow such input permutation in a
      // limited manner, we employ a heuristic of looking one level ahead in
      // both graphs, whether visiting the right child of pattern is likely to
      // match left child of the given graph. In that case, we simply swap the
      // left subtree with right subtree of pattern syntax graph and continue
      // matching children of pattern with the children of given computation
      // graph. Note, we do not change anything in the computation graph during
      // pattern matching, only the pattern graph is changed. By looking ahead
      // one step in case of commutative ops, we keep the time comlexity of
      // pattern matching linear. Since it is only a heuristic and we look only
      // one step ahead it is not guranteed that all possible permutations will
      // be matched. For example, when both the input ops to the commutative op
      // are same, we cannot anticipate which of the permutation is likely to
      // match unless we look two level down the graphs.
      std::vector<int> pattern_child_indices(num_children);
      std::iota(pattern_child_indices.begin(), pattern_child_indices.end(), 0);
      string op_name = pattern.op;
      if (IsCommutativeOp(op_name) && num_children == 2) {
        MutableNodeView* graph_child0_node_view =
            graph_view_->GetNode(graph_children[0].node_index());
        if (!IsSame(pattern.children[0].op, graph_child0_node_view->GetOp()) &&
            IsSame(pattern.children[1].op, graph_child0_node_view->GetOp()))
          std::swap(pattern_child_indices[0], pattern_child_indices[1]);
      }
      for (int i = 0; i < num_children; ++i) {
        auto child_node_index = graph_children[i].node_index();
        // TODO (mdfaijul): Is it guaranted that GetNode will reuturn non null
        // pointer.
        MutableNodeView* child_node_view =
            graph_view_->GetNode(child_node_index);
        const OpTypePattern& child_pattern =
            pattern.children[pattern_child_indices[i]];
        match->children.push_back(NodeViewMatch());
        NodeViewMatch* child_match = &(match->children.back());
        if (!DoesOpTypePatternMatch(child_pattern, child_node_view,
                                    child_match)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Current implementation supports pattern maching toward node's inputs only.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::GetMatchedNodes(
    const OpTypePattern& pattern,
    const std::unordered_set<string>& nodes_to_preserve,
    MutableNodeView* node_view, std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTcc mht_3(mht_3_v, 335, "", "./tensorflow/core/grappler/utils/pattern_utils.cc", "SubGraphMatcher<MatchingDirection::kFollowInputs>::GetMatchedNodes");

  bool found_match = false;
  match_.reset(new NodeViewMatch());
  if (DoesOpTypePatternMatch(pattern, node_view, match_.get())) {
    if (IsSafeNodesToRemove(nodes_to_preserve)) {
      found_match = true;
      *matched_nodes_map = this->node_label_to_index_;
      *remove_node_indices = this->remove_node_indices_;
    }
  } else {
    found_match = false;
  }

  // Clear all bookkeeping data
  match_->Clear();
  match_.reset(nullptr);
  matched_node_indices_.clear();
  node_label_to_index_.clear();
  remove_node_indices_.clear();

  return found_match;
}

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
