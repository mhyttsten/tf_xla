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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh() {
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


#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {
namespace utils {

//------------------------------------------------------------------------------
// A pattern can be defined by the following grammar. Here, op_type is any valid
// op name in the TensorFlow.
//
//    leaf_pattern ::= `{` op_type `}`
//    pattern ::= leaf_pattern |
//                `{` op_type `,` `{` pattern `,` ... `,` pattern `}` `}`
//
// (1) For example, the following pattern syntax describes a pattern for
// _FusedConv2D (Conv2D + BiasAdd + Relu). Note that "*" means any type of op.
//
//  {"Relu",
//    {
//      "BiasAdd",
//      {
//        {"Conv2D"},
//        {"*"}
//      }
//    }
//  }
//
// The syntax above has a root ("Relu") and children (inputs), where each child
// is a sub-pattern. Graph pattern matcher finds a match for the given pattern
// syntax in a graph and returns a set of matched nodes.
//
// (2) In order to match a DAG with a given root, we extend pattern syntax with
// labels. For example, a frequently found pattern in Deep Learning models is a
// residual block like below.
//
//    Placeholder  Const
//          |        |
//    +-----+-----+  |
//    |           |  |
//    |           v  v
//    |          Conv2D   Const
//    |            |        |
//    |            v  v-----+
//    |          BiasAdd
//    |            |
//    v v----------+
//   AddV2
//
// As shown above, it is the same input node (Placeholder) consumed by both
// AddV2 and and Conv2D. This constrained can be put as labels in the following
// augmented pattern syntax.
//
//  {"AddV2", "my_add",
//    {
//      {"*", "my_residual_input"},
//      {"BiasAdd", "my_bias_add",
//        {
//          {"Conv2D", "my_conv",
//            {
//              {"*", "my_residual_input"},
//              {"*", "my_filter"}
//            }
//          },
//          {"*", my_bias"}
//        }
//      }
//    }
//  }
//
// Note that the same label "my_residual_input" is used to tell that it is a
// child of both "AddV2" and "Conv2D". Labels are arbitrary strings to associate
// with the nodes to be matched as well as to uniquely identify those nodes.
//
// (3) The motivatation for a grammar based pattern matching in grappler is to
// make easy for finding fusion pattern in the remapper. A subgraph that
// matches a given pattern, however, is not fusable if any of the matched node,
// that will be removed as a part of fusion, has a consumer outside the matched
// subgraph. In order to check for such type of external dependencies, we
// further extend pattern syntax by prospective action (NodeStatus) on the
// matched nodes as shown below. This helps cross checking the nodes to be
// removed with the nodes matched intially.
//
//  {"AddV2", "my_add", NodeStatus::kReplace,
//    {
//      {"*", "my_residual_input", NodeStatus::kRemain},
//      {"BiasAdd", "my_bias_add", NodeStatus::kRemove,
//        {
//          {"Conv2D", "my_conv", NodeStatus::kRemove,
//            {
//              {"*", "my_residual_input", NodeStatus::kRemain},
//              {"*", "my_filter", NodeStatus::Remain}
//            }
//          },
//          {"*", my_bias", NodeStatus::kRemain}
//        }
//      }
//    }
//  }
//------------------------------------------------------------------------------

// Pattern matcher recursively matches child subpatterns. The direction
// for children could be toward node's input (fanins) or outputs (fanouts).
enum class MatchingDirection { kFollowInputs, kFollowOutputs };

// Action for each node in the set of matched nodes for a given pattern.
enum class NodeStatus { kRemain, kRemove, kReplace };

// TODO (intel-tf): Support multiple roots by making them children of a single
// virtual root.
struct OpTypePattern {
  string op;
  string label;
  NodeStatus node_status;
  std::vector<OpTypePattern> children;

  string DebugString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh mht_0(mht_0_v, 303, "", "./tensorflow/core/grappler/utils/pattern_utils.h", "DebugString");

    string result = "{(op: " + op + ", " + "label: " + label + "), {";
    for (const OpTypePattern& child : children) {
      result += child.DebugString() + ",";
    }
    result += "}}";
    return result;
  }
};

// This is a helpful recursive structure that keeps one-to-one mapping of
// pattern syntax to the matched nodes. User can call DebugString to see what
// has been matched so far and where is the failing point.
struct NodeViewMatch {
  MutableNodeView* node_view = nullptr;
  std::vector<NodeViewMatch> children;

  string DebugString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh mht_1(mht_1_v, 323, "", "./tensorflow/core/grappler/utils/pattern_utils.h", "DebugString");

    string result = "{";
    if (node_view == nullptr) {
      result += "Non-Matched-Node}";
      return result;
    } else {
      result += node_view->node()->DebugString();
      result += ", {";
      for (const NodeViewMatch& child : children) {
        result += child.DebugString() + ",";
      }
      result += "}}";
      return result;
    }
  }

  void Clear() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh mht_2(mht_2_v, 342, "", "./tensorflow/core/grappler/utils/pattern_utils.h", "Clear");

    for (NodeViewMatch& child : children) {
      child.Clear();  // child is an object.
    }
    children.clear();  // children is a vector.
    if (node_view != nullptr) {
      node_view = nullptr;
    }
  }
};

template <MatchingDirection DIRECTION = MatchingDirection::kFollowInputs>
class SubGraphMatcher {
 public:
  SubGraphMatcher(MutableGraphView* graph_view) : graph_view_(graph_view){};

  // If a given pattern is matched, this function returns true as well as the
  // matched node and remove node info is populated.
  bool GetMatchedNodes(const OpTypePattern& pattern,
                       const std::unordered_set<string>& nodes_to_preserve,
                       MutableNodeView* node_view,
                       std::map<string, int>* matched_nodes_map,
                       std::set<int>* remove_node_indices);

 private:
  MutableGraphView* graph_view_;
  std::map<string, int> node_label_to_index_;
  std::set<int> matched_node_indices_;
  std::set<int> remove_node_indices_;
  std::unique_ptr<NodeViewMatch> match_ = nullptr;

  bool DoesOpTypePatternMatch(const OpTypePattern& pattern,
                              MutableNodeView* node_view, NodeViewMatch* match);

  // This function should be called after the pattern matcher has found
  // potential matched nodes (i.e. when DoesOpTypePatternMatch returns "true").
  // It performs a sanity check if the candidate nodes for removal in subgraph
  // fusion is indeed safe to remove.
  bool IsSafeNodesToRemove(
      const std::unordered_set<string>& nodes_to_preserve) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utilsDTh mht_3(mht_3_v, 384, "", "./tensorflow/core/grappler/utils/pattern_utils.h", "IsSafeNodesToRemove");

    for (const auto& node_idx : remove_node_indices_) {
      auto node_view = graph_view_->GetNode(node_idx);
      // Check if the node to be removed is in the nodes to be preserved.
      string node_name = node_view->GetName();
      if (nodes_to_preserve.count(node_name) > 0) return false;
      // Traverse all the Regular Fanouts. Fanouts are stored as vector of
      // vector, std::vector<std::vector<MutableFaninView>>. Note that
      // a MutableNodeView's fanouts are stored in a nested vector of
      // MutableFaninView type.
      auto fanouts_by_ports = node_view->GetRegularFanouts();
      for (const auto& fanouts : fanouts_by_ports) {
        for (const auto& fanout : fanouts) {
          if (!matched_node_indices_.count(fanout.node_index())) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_
