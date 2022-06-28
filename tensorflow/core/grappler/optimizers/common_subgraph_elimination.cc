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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc() {
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

#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/canonicalizer.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {
class Cluster;
}  // namespace grappler
}  // namespace tensorflow

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {

class UniqueNodes {
 public:
  // Warning: This is conservative and may fail to find an identical node in
  // some cases. This happens if the node has large attribute tensor values that
  // have different proto encoding but identical tensor value.
  NodeDef* FindOrAddRepresentative(NodeDef* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "FindOrAddRepresentative");

    uint64 sig = ComputeSignature(*node);
    std::vector<NodeDef*>& candidates = rep_[sig];
    for (auto& candidate : candidates) {
      if ((candidate == node) || SameNode(*candidate, *node)) {
        return candidate;
      }
    }
    candidates.push_back(node);
    return node;
  }

  void RemoveRepresentative(NodeDef* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "RemoveRepresentative");

    auto it = memoized_signatures_.find(node);
    if (it == memoized_signatures_.end()) return;

    std::vector<NodeDef*>& candidates = rep_[it->second];
    for (int i = 0, end = candidates.size(); i < end; ++i) {
      if (candidates[i] == node) {
        std::swap(candidates[i], candidates[candidates.size() - 1]);
        candidates.resize(candidates.size() - 1);
        break;
      }
    }
    memoized_signatures_.erase(node);
  }

 private:
  uint64 ComputeSignature(const NodeDef& node);
  bool SameNode(const NodeDef& node1, const NodeDef& node2) const;

  absl::flat_hash_map<uint64, std::vector<NodeDef*>> rep_;
  absl::flat_hash_map<const NodeDef*, uint64> memoized_signatures_;
};

uint64 UniqueNodes::ComputeSignature(const NodeDef& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "UniqueNodes::ComputeSignature");

  auto it = memoized_signatures_.find(&node);
  if (it != memoized_signatures_.end()) return it->second;

  uint64 h = Hash64(node.op());
  h = Hash64Combine(Hash64(node.device()), h);

  for (const auto& input : node.input()) {
    const TensorId input_tensor = ParseTensorName(input);
    uint64 input_hash = Hash64Combine(
        Hash64(input_tensor.node().data(), input_tensor.node().size()),
        std::hash<int>()(input_tensor.index()));
    h = Hash64CombineUnordered(input_hash, h);
  }
  for (const auto& attr : node.attr()) {
    uint64 attr_hash =
        Hash64Combine(Hash64(attr.first), FastAttrValueHash(attr.second));
    h = Hash64CombineUnordered(attr_hash, h);
  }
  memoized_signatures_.emplace(&node, h);
  return h;
}

// PRECONDITION:
//  Node input orders are assumed to be canonicalized, i.e. control inputs for
//  all nodes as well as regular inputs for commutative nodes must be sorted.
bool UniqueNodes::SameNode(const NodeDef& node1, const NodeDef& node2) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "UniqueNodes::SameNode");

  if (node1.op() != node2.op()) {
    return false;
  }
  if (node1.device() != node2.device()) {
    return false;
  }
  if (node1.input_size() != node2.input_size()) {
    return false;
  }
  if (node1.attr_size() != node2.attr_size()) {
    return false;
  }

  // Compare inputs.
  auto it1 = node1.input().begin();
  auto it2 = node2.input().begin();
  for (; it1 != node1.input().end(); ++it1, ++it2) {
    if (*it1 != *it2) return false;
  }

  // Compare attributes.
  for (const auto& attr1 : node1.attr()) {
    auto it = node2.attr().find(attr1.first);
    if (it == node2.attr().end()) return false;
    if (!AreAttrValuesEqual(attr1.second, it->second,
                            /*allow_false_negatives=*/true)) {
      return false;
    }
  }

  return true;
}

bool CommonSubgraphElimination::CanDedup(const NodeDef& node) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_4(mht_4_v, 338, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "CommonSubgraphElimination::CanDedup");

  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (IsEnter(node) || IsExit(node)) {
    return false;
  }
  if (node.device().find("SPU") != string::npos) {
    return false;
  }
  // Workaround for Assert and Print mistakenly being labeled as stateful.
  if (IsAssert(node) || IsPrint(node)) {
    return true;
  }
  return IsFreeOfSideEffect(node);
}

Status CommonSubgraphElimination::DedupComputations(GraphDef* optimized_graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_5(mht_5_v, 358, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "CommonSubgraphElimination::DedupComputations");

  CanonicalizeGraph(optimized_graph);

  GraphTopologyView graph_view;
  if (!graph_view.InitializeFromGraph(*optimized_graph).ok()) {
    LOG(WARNING) << "Failed to initialize GraphTopologyView.";
    return Status::OK();
  }

  // If either node or rep feeds an inplace op, deduping them may cause data
  // races. For example: If we dedup nodes initializing two independent
  // inplace accumulations, they will write to the same buffer, clobbering
  // each other's results.
  absl::flat_hash_set<const NodeDef*> feeds_inplace_op;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& root = optimized_graph->node(i);
    if (feeds_inplace_op.find(&root) != feeds_inplace_op.end()) continue;
    if (ModifiesInputsInPlace(root)) {
      const auto is_continue_traversal = [&](const NodeDef* node) -> bool {
        return node->op() == root.op() || !NeverForwardsInputs(*node);
      };

      DfsTraversal(graph_view, {&root}, TraversalDirection::kFollowInputs,
                   DfsPredicates::Advance(is_continue_traversal),
                   DfsCallbacks::PreOrder([&](const NodeDef* node) {
                     feeds_inplace_op.insert(node);
                   }));
    }
  }

  std::vector<bool> can_dedup(optimized_graph->node_size());
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& node = optimized_graph->node(i);
    can_dedup[i] = (feeds_inplace_op.find(&node) == feeds_inplace_op.end()) &&
                   CanDedup(node);
  }

  bool stop = true;
  std::set<int> duplicates;
  UniqueNodes nodes;
  NodeMap node_map(optimized_graph);
  do {
    stop = true;
    for (int i = 0; i < optimized_graph->node_size(); ++i) {
      if (!can_dedup[i] || duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph->mutable_node(i);
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      // Make a copy since we mutate the set below.
      const auto fanouts = node_map.GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        // Update consumers of node.
        bool updated_fanout = false;
        for (int i = 0; i < fanout->input_size(); ++i) {
          string* fanout_input = fanout->mutable_input(i);

          const int position =
              NodePositionIfSameNode(*fanout_input, node->name());
          // Update name in-place.
          if (position < -1) {
            continue;
          } else {
            if (!updated_fanout) {
              // The signature of the fanout node will change. Remove it from
              // nodes.
              nodes.RemoveRepresentative(fanout);
            }
            updated_fanout = true;
            if (position > 0) {
              *fanout_input = StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              *fanout_input = rep->name();
            } else {
              *fanout_input = StrCat("^", rep->name());
            }
          }
        }
        if (updated_fanout) {
          node_map.UpdateInput(fanout->name(), node->name(), rep->name());
          CanonicalizeNode(fanout);
        }
      }
      if (fetch_nodes_known_) {
        node->Clear();
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (fetch_nodes_known_ && !duplicates.empty()) {
    EraseNodesFromGraph(duplicates, optimized_graph);
  }

  return Status::OK();
}

Status CommonSubgraphElimination::Optimize(Cluster* /*cluster*/,
                                           const GrapplerItem& item,
                                           GraphDef* optimized_graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_eliminationDTcc mht_6(mht_6_v, 465, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination.cc", "CommonSubgraphElimination::Optimize");

  // Set up helper data structures.
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  *optimized_graph = item.graph;

  // Perform topological sort on the graph in order to help DedupComputations
  // optimize larger subgraphs starting from the roots with more inputs.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  return DedupComputations(optimized_graph);
}

}  // namespace grappler
}  // namespace tensorflow
