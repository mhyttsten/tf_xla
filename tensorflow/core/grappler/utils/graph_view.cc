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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/graph_view.h"

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view_internal.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace utils {

FaninView::FaninView(NodeView* node_view, int index)
    : NodeIndexAndPortIndex(node_view->graph_view_, node_view->node_index_,
                            index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/grappler/utils/graph_view.cc", "FaninView::FaninView");
}

FanoutView::FanoutView(NodeView* node_view, int index)
    : NodeIndexAndPortIndex(node_view->graph_view_, node_view->node_index_,
                            index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/grappler/utils/graph_view.cc", "FanoutView::FanoutView");
}

const NodeDef* NodeView::node() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/grappler/utils/graph_view.cc", "NodeView::node");

  return &graph_view_->graph()->node(node_index_);
}

bool NodeView::HasFanin(const FanoutView& fanin) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_3(mht_3_v, 226, "", "./tensorflow/core/grappler/utils/graph_view.cc", "NodeView::HasFanin");

  if (fanin.index() < Graph::kControlSlot || graph_view_ != fanin.graph_view_) {
    return false;
  }
  return fanins_set_.contains(
      {&graph_view_->graph_->node(fanin.node_index_), fanin.index()});
}

bool NodeView::HasFanout(const FaninView& fanout) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/grappler/utils/graph_view.cc", "NodeView::HasFanout");

  if (fanout.index() < Graph::kControlSlot ||
      graph_view_ != fanout.graph_view_) {
    return false;
  }
  NodeView* view = fanout.node_view();
  if (view == nullptr) {
    return false;
  } else if (fanout.index() == Graph::kControlSlot) {
    return view->fanins_set_.contains({this->node(), Graph::kControlSlot});
  } else if (fanout.index() >= static_cast<int>(view->regular_fanins_.size())) {
    return false;
  }
  return view->regular_fanins_[fanout.index()].node_index_ == node_index_;
}

inline const FanoutView& NodeView::GetMissingFanin() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/grappler/utils/graph_view.cc", "NodeView::GetMissingFanin");

  return graph_view_->missing_fanin_;
}

inline const std::vector<FaninView>& NodeView::GetMissingFanout() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_6(mht_6_v, 263, "", "./tensorflow/core/grappler/utils/graph_view.cc", "NodeView::GetMissingFanout");

  return graph_view_->missing_fanout_;
}

namespace {
const char kGraphViewError[] = "GraphView::GraphView error: ";
}  // namespace

GraphView::GraphView(const GraphDef* graph, Status* status)
    : GraphViewInternal(graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_7(mht_7_v, 275, "", "./tensorflow/core/grappler/utils/graph_view.cc", "GraphView::GraphView");

  const int num_nodes = graph->node_size();
  node_index_by_name_.reserve(num_nodes);
  nodes_.reserve(num_nodes);
  for (const NodeDef& node : graph->node()) {
    if (!AddUniqueNodeInternal(&node)) {
      *status = errors::InvalidArgument(
          kGraphViewError, "graph has multiple nodes with the name '",
          node.name(), "'.");
      Reset();
      return;
    }
  }
  Status s;
  for (NodeView& node_view : nodes_) {
    s = CheckAndAddFaninsInternal(&node_view);
    if (!s.ok()) {
      *status = s;
      Reset();
      return;
    }
  }
  *status = Status::OK();
}

bool GraphView::AddUniqueNodeInternal(const NodeDef* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_8(mht_8_v, 303, "", "./tensorflow/core/grappler/utils/graph_view.cc", "GraphView::AddUniqueNodeInternal");

  const int node_index = node_index_by_name_.size();
  auto it = node_index_by_name_.emplace(node->name(), node_index);
  if (it.second) {
    nodes_.emplace_back(this, node_index);
    return true;
  }
  return false;
}

Status GraphView::CheckAndAddFaninsInternal(NodeView* node_view) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/grappler/utils/graph_view.cc", "GraphView::CheckAndAddFaninsInternal");

  bool has_observed_control = false;
  const NodeDef* node = node_view->node();
  const string& node_name = node->name();
  const int node_index = node_view->node_index_;
  node_view->fanins_set_.reserve(node->input_size());
  for (const string& input : node->input()) {
    TensorId fanin_id = ParseTensorName(input);
    if (fanin_id.node() == node_name) {
      return errors::InvalidArgument(kGraphViewError, "node '", node_name,
                                     "' has self cycle fanin '", input, "'.");
    }
    bool is_control = IsTensorIdControl(fanin_id);
    if (!is_control && has_observed_control) {
      return errors::InvalidArgument(kGraphViewError, "node '", node_name,
                                     "' has regular fanin '", input,
                                     "' after controlling fanins.");
    }
    auto it = node_index_by_name_.find(fanin_id.node());
    if (it == node_index_by_name_.end()) {
      return errors::InvalidArgument(kGraphViewError, "node '", node_name,
                                     "' has missing fanin '", input, "'.");
    }
    const int fanin_node_index = it->second;
    NodeView& fanin_node_view = nodes_[fanin_node_index];

    if (is_control) {
      fanin_node_view.controlled_fanouts_.emplace_back(this, node_index,
                                                       Graph::kControlSlot);
      node_view->controlling_fanins_.emplace_back(this, fanin_node_index,
                                                  Graph::kControlSlot);
      node_view->fanins_set_.emplace(fanin_node_view.node(),
                                     Graph::kControlSlot);
      has_observed_control = true;
    } else {
      int fanin_node_view_regular_fanouts_by_port_size =
          fanin_node_view.regular_fanouts_by_port_.size();
      if (fanin_node_view_regular_fanouts_by_port_size < fanin_id.index() + 1) {
        fanin_node_view.regular_fanouts_by_port_.resize(fanin_id.index() + 1);
      }
      fanin_node_view.regular_fanouts_by_port_[fanin_id.index()].emplace_back(
          this, node_index, node_view->regular_fanins_.size());
      ++fanin_node_view.num_regular_fanouts_;
      node_view->regular_fanins_.emplace_back(this, fanin_node_index,
                                              fanin_id.index());
      node_view->fanins_set_.emplace(fanin_node_view.node(), fanin_id.index());
    }
  }
  return Status::OK();
}

MutableFaninView::MutableFaninView(MutableNodeView* node_view, int index)
    : NodeIndexAndPortIndex(node_view->graph_view_, node_view->node_index_,
                            index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_10(mht_10_v, 372, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableFaninView::MutableFaninView");
}

MutableFanoutView::MutableFanoutView(MutableNodeView* node_view, int index)
    : NodeIndexAndPortIndex(node_view->graph_view_, node_view->node_index_,
                            index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_11(mht_11_v, 379, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableFanoutView::MutableFanoutView");
}

NodeDef* MutableNodeView::node() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_12(mht_12_v, 384, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableNodeView::node");

  return graph_view_->graph()->mutable_node(node_index_);
}

bool MutableNodeView::HasFanin(const MutableFanoutView& fanin) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_13(mht_13_v, 391, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableNodeView::HasFanin");

  if (fanin.index() < Graph::kControlSlot || graph_view_ != fanin.graph_view_) {
    return false;
  }
  return fanins_count_.contains(
      {&graph_view_->graph_->node(fanin.node_index_), fanin.index()});
}

bool MutableNodeView::HasFanout(const MutableFaninView& fanout) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_14(mht_14_v, 402, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableNodeView::HasFanout");

  if (fanout.index() < Graph::kControlSlot ||
      graph_view_ != fanout.graph_view_) {
    return false;
  }
  MutableNodeView* view = fanout.node_view();
  if (view == nullptr) {
    return false;
  } else if (fanout.index() == Graph::kControlSlot) {
    return view->fanins_count_.contains({this->node(), Graph::kControlSlot});
  } else if (fanout.index() >= static_cast<int>(view->regular_fanins_.size())) {
    return false;
  }
  return view->regular_fanins_[fanout.index()].node_index_ == node_index_;
}

const MutableFanoutView& MutableNodeView::GetMissingFanin() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_15(mht_15_v, 421, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableNodeView::GetMissingFanin");

  return graph_view_->missing_fanin_;
}

const std::vector<MutableFaninView>& MutableNodeView::GetMissingFanout() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_16(mht_16_v, 428, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableNodeView::GetMissingFanout");

  return graph_view_->missing_fanout_;
}

namespace {
const char kMutationAddNodeError[] = "Mutation::AddNode error: ";

bool IsTensorIdRegular(const TensorId& tensor_id) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_17(mht_17_v, 438, "", "./tensorflow/core/grappler/utils/graph_view.cc", "IsTensorIdRegular");

  return tensor_id.index() >= 0;
}
}  // namespace

Mutation::Mutation(MutableGraphView* graph_view) : graph_view_(graph_view) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_18(mht_18_v, 446, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::Mutation");
}

MutationNewNode Mutation::AddNode(NodeDef&& node, Status* status) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_19(mht_19_v, 451, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddNode");

  bool has_observed_control = false;
  const string& node_name = node.name();
  std::vector<SafeTensorId> regular_fanins;
  absl::flat_hash_set<string> controlling_fanins;
  const int num_fanins = node.input_size();
  for (int i = 0; i < num_fanins; ++i) {
    const string& input = node.input(i);
    TensorId fanin_id = ParseTensorName(input);
    if (fanin_id.node() == node_name) {
      *status =
          errors::InvalidArgument(kMutationAddNodeError, "node '", node_name,
                                  "' has self cycle fanin '", input, "'.");
      return MutationNewNode(this, mutation_counter_, internal::kMissingIndex);
    }
    bool is_control = IsTensorIdControl(fanin_id);
    if (is_control) {
      has_observed_control = true;
      controlling_fanins.emplace(fanin_id.node());
    } else if (has_observed_control) {
      *status = errors::InvalidArgument(kMutationAddNodeError, "node '",
                                        node_name, "' has regular fanin '",
                                        input, "' after controlling fanins.");
      return MutationNewNode(this, mutation_counter_, internal::kMissingIndex);
    } else {
      regular_fanins.emplace_back(fanin_id);
    }
  }

  node.mutable_input()->Clear();
  new_nodes_.emplace_back(graph_view_, std::move(node));
  MutationNewNodeHolder& mutation_node = new_nodes_.back();
  mutation_node.regular_fanins = std::move(regular_fanins);
  mutation_node.num_regular_fanins = mutation_node.regular_fanins.size();
  mutation_node.controlling_fanins = std::move(controlling_fanins);
  *status = Status::OK();
  return MutationNewNode(this, mutation_counter_, new_nodes_.size() - 1);
}

void Mutation::AddMutation(
    MutableNodeView* node,
    std::function<bool(MutableNodeViewDiff*)> mutate_fn) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_20(mht_20_v, 495, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddMutation");

  DCHECK(node->graph_view_ == graph_view_);
  if (node->update_index_ == internal::kMissingIndex) {
    MutableNodeViewDiff diff(graph_view_, node->node_index_);
    // If mutation is a no-op return and do not add it to the `updated_nodes_`.
    if (!mutate_fn(&diff)) return;
    node->update_index_ = updated_nodes_.size();
    updated_nodes_.push_back(std::move(diff));
  } else if (!removed_nodes_.contains(node->node_index_)) {
    MutableNodeViewDiff& diff = updated_nodes_[node->update_index_];
    mutate_fn(&diff);
  }
}

void Mutation::RemoveNode(MutableNodeView* node) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_21(mht_21_v, 512, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveNode");

  auto& update_index = node->update_index_;
  if (update_index != internal::kMissingIndex) {
    int updated_nodes_size = updated_nodes_.size();
    if (update_index < updated_nodes_size - 1) {
      graph_view_->nodes_[updated_nodes_.back().node_index].update_index_ =
          update_index;
      std::swap(updated_nodes_[update_index], updated_nodes_.back());
    }
    updated_nodes_.pop_back();
    update_index = internal::kMissingIndex;
  }
  removed_nodes_.insert(node->node_index_);
}

void Mutation::UpdateNodeName(MutableNodeView* node, absl::string_view name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_22(mht_22_v, 531, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeName");

  AddMutation(node, [name](MutableNodeViewDiff* diff) {
    return internal::UpdateName(diff, name);
  });
}

void Mutation::UpdateNodeName(const MutationNewNode& node,
                              absl::string_view name) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_23(mht_23_v, 542, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeName");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::UpdateName(&new_nodes_[node.index_], name);
}

void Mutation::UpdateNodeOp(MutableNodeView* node, absl::string_view op) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_24(mht_24_v, 551, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeOp");

  AddMutation(node, [op](MutableNodeViewDiff* diff) {
    return internal::UpdateOp(diff, op);
  });
}

void Mutation::UpdateNodeOp(const MutationNewNode& node, absl::string_view op) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_25(mht_25_v, 561, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeOp");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::UpdateOp(&new_nodes_[node.index_], op);
}

void Mutation::UpdateNodeDevice(MutableNodeView* node,
                                absl::string_view device) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_26(mht_26_v, 571, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeDevice");

  AddMutation(node, [device](MutableNodeViewDiff* diff) {
    return internal::UpdateDevice(diff, device);
  });
}

void Mutation::UpdateNodeDevice(const MutationNewNode& node,
                                absl::string_view device) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_27(mht_27_v, 582, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::UpdateNodeDevice");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::UpdateDevice(&new_nodes_[node.index_], device);
}

void Mutation::AddOrUpdateRegularFanin(MutableNodeView* node, int index,
                                       const TensorId& fanin) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_28(mht_28_v, 591, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddOrUpdateRegularFanin");

  AddMutation(node, [index, fanin](MutableNodeViewDiff* diff) {
    return internal::AddOrUpdateRegularFanin(diff, index, fanin);
  });
}

void Mutation::AddOrUpdateRegularFanin(const MutationNewNode& node, int index,
                                       const TensorId& fanin) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_29(mht_29_v, 601, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddOrUpdateRegularFanin");

  DCHECK(node.mutation_ == this &&
         node.mutation_counter_ == mutation_counter_ && index >= 0 &&
         IsTensorIdRegular(fanin));
  internal::AddOrUpdateRegularFanin(&new_nodes_[node.index_], index, fanin);
}

void Mutation::RemoveRegularFanin(MutableNodeView* node, int index) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_30(mht_30_v, 611, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveRegularFanin");

  AddMutation(node, [index](MutableNodeViewDiff* diff) {
    return internal::RemoveRegularFanin(diff, index);
  });
}

void Mutation::RemoveRegularFanin(const MutationNewNode& node, int index) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_31(mht_31_v, 620, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveRegularFanin");

  DCHECK(node.mutation_ == this &&
         node.mutation_counter_ == mutation_counter_ && index >= 0);
  internal::RemoveRegularFanin(&new_nodes_[node.index_], index);
}

void Mutation::AddControllingFanin(MutableNodeView* node,
                                   absl::string_view fanin_node_name) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_32(mht_32_v, 631, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddControllingFanin");

  AddMutation(node, [node, fanin_node_name](MutableNodeViewDiff* diff) {
    auto it = node->controlling_fanins_index_.find(fanin_node_name);
    const int control_index = it != node->controlling_fanins_index_.end()
                                  ? it->second
                                  : internal::kMissingIndex;
    return internal::AddControllingFanin(diff, control_index, fanin_node_name);
  });
}

void Mutation::AddControllingFanin(const MutationNewNode& node,
                                   absl::string_view fanin_node_name) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_33(mht_33_v, 646, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddControllingFanin");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::AddControllingFanin(&new_nodes_[node.index_], fanin_node_name);
}

void Mutation::RemoveControllingFanin(MutableNodeView* node,
                                      absl::string_view fanin_node_name) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_34(mht_34_v, 656, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveControllingFanin");

  AddMutation(node, [node, fanin_node_name](MutableNodeViewDiff* diff) {
    auto it = node->controlling_fanins_index_.find(fanin_node_name);
    const int control_index = it != node->controlling_fanins_index_.end()
                                  ? it->second
                                  : internal::kMissingIndex;
    return internal::RemoveControllingFanin(diff, control_index,
                                            fanin_node_name);
  });
}

void Mutation::RemoveControllingFanin(const MutationNewNode& node,
                                      absl::string_view fanin_node_name) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_35(mht_35_v, 672, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveControllingFanin");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::RemoveControllingFanin(&new_nodes_[node.index_], fanin_node_name);
}

void Mutation::AddOrUpdateNodeAttr(MutableNodeView* node,
                                   absl::string_view attr_name,
                                   const AttrValue& attr_value) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_36(mht_36_v, 683, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddOrUpdateNodeAttr");

  AddMutation(node, [attr_name, attr_value](MutableNodeViewDiff* diff) {
    return internal::AddOrUpdateAttribute(diff, attr_name, attr_value);
  });
}

void Mutation::AddOrUpdateNodeAttr(const MutationNewNode& node,
                                   absl::string_view attr_name,
                                   const AttrValue& attr_value) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_37(mht_37_v, 695, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::AddOrUpdateNodeAttr");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::AddOrUpdateAttribute(&new_nodes_[node.index_], attr_name,
                                 attr_value);
}

void Mutation::RemoveNodeAttr(MutableNodeView* node,
                              absl::string_view attr_name) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_38(mht_38_v, 706, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveNodeAttr");

  AddMutation(node, [attr_name](MutableNodeViewDiff* diff) {
    return internal::RemoveAttribute(diff, attr_name);
  });
}

void Mutation::RemoveNodeAttr(const MutationNewNode& node,
                              absl::string_view attr_name) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_39(mht_39_v, 717, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::RemoveNodeAttr");

  DCHECK(node.mutation_ == this && node.mutation_counter_ == mutation_counter_);
  internal::RemoveAttribute(&new_nodes_[node.index_], attr_name);
}

void Mutation::ResetInternal() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_40(mht_40_v, 725, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::ResetInternal");

  updated_nodes_.clear();
  removed_nodes_.clear();
  new_nodes_.clear();
}

void Mutation::Reset() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_41(mht_41_v, 734, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::Reset");

  for (const auto& update : updated_nodes_) {
    graph_view_->nodes_[update.node_index].update_index_ =
        internal::kMissingIndex;
  }
  ResetInternal();
}

Status Mutation::Apply() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_42(mht_42_v, 745, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Mutation::Apply");
 return graph_view_->ApplyMutationInternal(); }

namespace {
const char kMutableGraphViewError[] =
    "MutableGraphView::MutableGraphView error: ";

const char kMutableGraphViewApplyError[] = "Mutation::Apply error: ";

inline void IncrementFaninCount(
    absl::flat_hash_map<internal::NodeDefAndPortIndex, int>* fanins_count,
    const internal::NodeDefAndPortIndex& fanin) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_43(mht_43_v, 758, "", "./tensorflow/core/grappler/utils/graph_view.cc", "IncrementFaninCount");

  ++(*fanins_count)[fanin];
}

inline void DecrementFaninCount(
    absl::flat_hash_map<internal::NodeDefAndPortIndex, int>* fanins_count,
    const internal::NodeDefAndPortIndex& fanin) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_44(mht_44_v, 767, "", "./tensorflow/core/grappler/utils/graph_view.cc", "DecrementFaninCount");

  auto it = fanins_count->find(fanin);
  if (it != fanins_count->end()) {
    if (it->second <= 1) {
      fanins_count->erase(it);
    } else {
      --it->second;
    }
  }
}
}  // namespace

MutableGraphView::MutableGraphView(GraphDef* graph, Status* status)
    : GraphViewInternal(graph), mutation_(Mutation(this)) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_45(mht_45_v, 783, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::MutableGraphView");

  const int num_nodes = graph->node_size();
  node_index_by_name_.reserve(num_nodes);
  nodes_.reserve(num_nodes);
  for (NodeDef& node : *graph->mutable_node()) {
    if (!AddUniqueNodeInternal(&node)) {
      *status = errors::InvalidArgument(
          kMutableGraphViewError, "graph has multiple nodes with the name '",
          node.name(), "'.");
      Reset();
      return;
    }
  }
  std::vector<std::vector<TensorId>> fanins;
  Status s = CheckFaninsInternal(&fanins);
  if (!s.ok()) {
    *status = s;
    Reset();
    return;
  }
  AddFaninsInternal(&fanins);
  mutation_.ResetInternal();
  *status = Status::OK();
}

Mutation* MutableGraphView::GetMutationBuilder() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_46(mht_46_v, 811, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::GetMutationBuilder");
 return &mutation_; }

bool MutableGraphView::AddUniqueNodeInternal(NodeDef* node) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_47(mht_47_v, 816, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::AddUniqueNodeInternal");

  const int node_index = node_index_by_name_.size();
  auto it = node_index_by_name_.emplace(node->name(), node_index);
  if (it.second) {
    nodes_.emplace_back(this, node_index);
    return true;
  }
  return false;
}

Status MutableGraphView::CheckFaninsInternal(
    std::vector<std::vector<TensorId>>* fanins) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_48(mht_48_v, 830, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::CheckFaninsInternal");

  const int num_nodes = nodes_.size();
  fanins->reserve(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    bool has_observed_control = false;
    const NodeDef* node = nodes_[i].node();
    const string& node_name = node->name();
    std::vector<TensorId> node_fanins;
    node_fanins.reserve(node->input_size());
    for (const string& input : node->input()) {
      TensorId fanin_id = ParseTensorName(input);
      if (fanin_id.node() == node_name) {
        return errors::InvalidArgument(kMutableGraphViewError, "node '",
                                       node_name, "' has self cycle fanin '",
                                       input, "'.");
      }
      bool is_control = IsTensorIdControl(fanin_id);
      if (!is_control && has_observed_control) {
        return errors::InvalidArgument(kMutableGraphViewError, "node '",
                                       node_name, "' has regular fanin '",
                                       input, "' after controlling fanins.");
      }
      if (!node_index_by_name_.contains(fanin_id.node())) {
        return errors::InvalidArgument(kMutableGraphViewError, "node '",
                                       node_name, "' has missing fanin '",
                                       input, "'.");
      }
      if (is_control) {
        has_observed_control = true;
      }
      node_fanins.push_back(std::move(fanin_id));
    }
    fanins->push_back(std::move(node_fanins));
  }
  return Status::OK();
}

void MutableGraphView::AddFaninsInternal(
    std::vector<std::vector<TensorId>>* fanins) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_49(mht_49_v, 871, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::AddFaninsInternal");

  const int num_nodes = nodes_.size();
  for (int i = 0; i < num_nodes; ++i) {
    MutableNodeView& node_view = nodes_[i];
    NodeDef* node = node_view.node();
    std::vector<TensorId>& node_fanins = fanins->at(i);
    absl::flat_hash_set<absl::string_view> observed_controls;
    int pos = 0;
    const int last_idx = node_fanins.size() - 1;
    int last_pos = last_idx;
    node_view.fanins_count_.reserve(node->input_size());
    node_view.controlling_fanins_index_.reserve(node->input_size());
    while (pos <= last_pos) {
      const TensorId& fanin_id = node_fanins[pos];
      bool is_control = IsTensorIdControl(fanin_id);
      const int fanin_node_index = node_index_by_name_[fanin_id.node()];
      MutableNodeView& fanin_node_view = nodes_[fanin_node_index];

      if (is_control) {
        if (gtl::InsertIfNotPresent(&observed_controls, fanin_id.node())) {
          fanin_node_view.controlled_fanouts_.emplace_back(
              this, i, Graph::kControlSlot,
              node_view.controlling_fanins_.size());
          node_view.controlling_fanins_.emplace_back(
              this, fanin_node_index, Graph::kControlSlot,
              fanin_node_view.controlled_fanouts_.size() - 1);
          IncrementFaninCount(
              &node_view.fanins_count_,
              {&graph_->node(fanin_node_index), Graph::kControlSlot});
          node_view.controlling_fanins_index_.emplace(
              fanin_id.node(), pos - node_view.NumRegularFanins());
          ++pos;
        } else {
          node->mutable_input()->SwapElements(pos, last_pos);
          std::swap(node_fanins[pos], node_fanins[last_pos]);
          --last_pos;
        }
      } else {
        int fanin_node_view_regular_fanouts_by_port_size =
            fanin_node_view.regular_fanouts_by_port_.size();
        if (fanin_node_view_regular_fanouts_by_port_size <
            fanin_id.index() + 1) {
          fanin_node_view.regular_fanouts_by_port_.resize(fanin_id.index() + 1);
        }
        auto& fanin_regular_fanouts =
            fanin_node_view.regular_fanouts_by_port_[fanin_id.index()];
        fanin_regular_fanouts.emplace_back(this, i,
                                           node_view.regular_fanins_.size(),
                                           node_view.regular_fanins_.size());
        ++fanin_node_view.num_regular_fanouts_;
        node_view.regular_fanins_.emplace_back(
            this, fanin_node_index, fanin_id.index(),
            fanin_regular_fanouts.size() - 1);
        IncrementFaninCount(
            &node_view.fanins_count_,
            {&graph_->node(fanin_node_index), fanin_id.index()});
        ++pos;
      }
    }
    if (last_pos < last_idx) {
      node->mutable_input()->DeleteSubrange(last_pos + 1, last_idx - last_pos);
    }
  }
}

Status MutableGraphView::GetNodeNamesAndPartitionUpdatedNodes(
    absl::flat_hash_map<absl::string_view, int>* node_names,
    std::vector<RenamedOrOverwrittenNode>* renamed_nodes,
    std::vector<int>* inplace_nodes,
    std::vector<int>* empty_diff_node_indices) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_50(mht_50_v, 943, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::GetNodeNamesAndPartitionUpdatedNodes");

  // For all nodes to be removed and renamed, mark their original names as
  // missing and put associated node index in graph.
  for (const auto& diff : mutation_.updated_nodes_) {
    if (diff.update_name) {
      const int index = diff.node_index;
      const string& node_name = nodes_[index].GetName();
      node_names->emplace(node_name, index);
    }
  }

  for (int node_index : mutation_.removed_nodes_) {
    const string& node_name = nodes_[node_index].GetName();
    node_names->emplace(node_name, node_index);
  }

  auto name_conflict = [](const absl::string_view node_name) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_51(mht_51_v, 963, "", "./tensorflow/core/grappler/utils/graph_view.cc", "lambda");

    return errors::InvalidArgument(kMutableGraphViewApplyError,
                                   "multiple nodes with the name: '", node_name,
                                   "' exists in Mutation.");
  };

  // Partition updated nodes by if they will be renamed or not.
  const int num_updated_nodes = mutation_.updated_nodes_.size();
  renamed_nodes->reserve(num_updated_nodes);
  inplace_nodes->reserve(num_updated_nodes);
  empty_diff_node_indices->reserve(num_updated_nodes);
  for (int i = 0; i < num_updated_nodes; ++i) {
    auto& diff = mutation_.updated_nodes_[i];
    if (internal::IsEmpty(&diff)) {
      empty_diff_node_indices->emplace_back(diff.node_index);
      continue;
    }
    // Get name of updated node after potential mutation.
    const string& node_name =
        diff.update_name ? diff.name : nodes_[diff.node_index].GetName();
    auto it = node_names->insert({node_name, internal::kNodeNamePresent});
    if (!it.second) {
      if (it.first->second == internal::kNodeNamePresent) {
        // Another node in the mutation is already using this name, which will
        // result in a conflict.
        return name_conflict(node_name);
      } else {
        // Mark name as present (node was marked missing from either being
        // removed or renamed).
        it.first->second = internal::kNodeNamePresent;
      }
    }
    if (diff.update_name) {
      // Lookup new name of node in current graph. If a node has such name,
      // store its index for later lookups as this node will be overwritten.
      auto node_name_it = node_index_by_name_.find(node_name);
      const int overwritten_node_index =
          node_name_it != node_index_by_name_.end() ? node_name_it->second
                                                    : internal::kMissingIndex;
      renamed_nodes->emplace_back(i, overwritten_node_index);
    } else {
      inplace_nodes->push_back(i);
    }
  }

  // Get names of new nodes after potential mutation.
  for (const auto& new_node : mutation_.new_nodes_) {
    const string& node_name = new_node.node.name();
    auto it = node_names->insert({node_name, internal::kNodeNamePresent});
    if (it.second) {
      continue;
    }
    if (it.first->second == internal::kNodeNamePresent) {
      // Another node in the mutation is already using this name, which will
      // result in a conflict.
      return name_conflict(node_name);
    } else {
      // Mark name as present (node was marked missing from either being removed
      // or renamed).
      it.first->second = internal::kNodeNamePresent;
    }
  }

  return Status::OK();
}

Status MutableGraphView::RemovedOrMissingNodeFanoutsWellFormed(
    const absl::flat_hash_map<absl::string_view, int>& node_names,
    const std::vector<RenamedOrOverwrittenNode>& renamed_nodes) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_52(mht_52_v, 1034, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemovedOrMissingNodeFanoutsWellFormed");

  auto bad_fanout = [](absl::string_view fanout_node_name,
                       absl::string_view node_name) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("fanout_node_name: \"" + std::string(fanout_node_name.data(), fanout_node_name.size()) + "\"");
   mht_53_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_53(mht_53_v, 1041, "", "./tensorflow/core/grappler/utils/graph_view.cc", "lambda");

    return errors::InvalidArgument(
        kMutableGraphViewApplyError, "fanout '", fanout_node_name,
        "' exist for missing node '", node_name, "'.");
  };

  // Lookup nodes to be overwritten.
  std::vector<bool> overwritten_nodes(NumNodes());
  for (auto& renamed_node : renamed_nodes) {
    if (renamed_node.overwritten_node_index_ == internal::kMissingIndex) {
      continue;
    }
    overwritten_nodes[renamed_node.overwritten_node_index_] = true;
  }

  // Check if removed nodes and previous state of renamed nodes have no fanouts.
  for (const auto& node_name_state : node_names) {
    if (node_name_state.second == internal::kNodeNamePresent) {
      continue;
    }
    const MutableNodeView& node_view = nodes_[node_name_state.second];
    for (const auto& regular_fanouts : node_view.GetRegularFanouts()) {
      for (const auto& regular_fanout : regular_fanouts) {
        // Check all fanouts of a single port.
        MutableNodeView* fanout_view = regular_fanout.node_view();
        if (fanout_view->update_index_ == internal::kMissingIndex) {
          if (mutation_.removed_nodes_.contains(fanout_view->node_index_)) {
            // Fanout node will be removed, this can be ignored.
            continue;
          } else if (!overwritten_nodes[fanout_view->node_index_]) {
            // Fanout is not updated or removed/overwritten.
            return bad_fanout(fanout_view->GetName(), node_name_state.first);
          }
        } else {
          auto& diff = mutation_.updated_nodes_[fanout_view->update_index_];
          const int last_index = fanout_view->NumRegularFanins() -
                                 diff.num_regular_inputs_to_remove - 1;
          if (regular_fanout.index() > last_index) {
            // Fanin of fanout is removed, this can be ignored.
            continue;
          }
          // Check if fanin is updated.
          if (diff.regular_inputs_to_update.find(regular_fanout.index()) ==
              diff.regular_inputs_to_update.end()) {
            return bad_fanout(fanout_view->GetName(), node_name_state.first);
          }
        }
      }
    }
    for (const auto& controlled_fanout : node_view.GetControlledFanouts()) {
      MutableNodeView* fanout_view = controlled_fanout.node_view();
      if (fanout_view->update_index_ == internal::kMissingIndex) {
        if (mutation_.removed_nodes_.contains(fanout_view->node_index_)) {
          // Fanout node will be removed, this can be ignored.
          continue;
        } else if (!overwritten_nodes[fanout_view->node_index_]) {
          // Fanout is not updated or removed/overwritten.
          return bad_fanout(fanout_view->GetName(), node_name_state.first);
        }
      } else {
        auto& diff = mutation_.updated_nodes_[fanout_view->update_index_];
        // Check if controlling fanin is removed.
        if (diff.controlling_inputs_to_remove.find(
                controlled_fanout.fanin_index_) ==
            diff.controlling_inputs_to_remove.end()) {
          return bad_fanout(fanout_view->GetName(), node_name_state.first);
        }
      }
    }
  }

  return Status::OK();
}

Status MutableGraphView::CheckNodeNamesAndFanins(
    const absl::flat_hash_map<absl::string_view, int>& node_names,
    const std::vector<RenamedOrOverwrittenNode>& renamed_nodes,
    const std::vector<int>& inplace_nodes) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_54(mht_54_v, 1121, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::CheckNodeNamesAndFanins");

  // Check if removed/missing node fanouts are valid.
  TF_RETURN_IF_ERROR(
      RemovedOrMissingNodeFanoutsWellFormed(node_names, renamed_nodes));

  // Check if updated nodes and their fanins are valid.
  for (auto& inplace_node : inplace_nodes) {
    auto& diff = mutation_.updated_nodes_[inplace_node];
    if (!internal::IsWellFormed(&diff, node_names)) {
      return errors::InvalidArgument(
          kMutableGraphViewApplyError, "inplace updated node '",
          nodes_[diff.node_index].GetName(), "' is ill-formed.");
    }
  }
  for (auto& renamed_node : renamed_nodes) {
    auto& diff = mutation_.updated_nodes_[renamed_node.renamed_update_index_];
    if (!internal::IsWellFormed(&diff, node_names)) {
      return errors::InvalidArgument(
          kMutableGraphViewApplyError, "renamed updated node '", diff.name,
          "' ('", nodes_[diff.node_index].GetName(), "') is ill-formed.");
    }
  }

  // Check if new nodes and their fanins are valid.
  for (auto& new_node : mutation_.new_nodes_) {
    if (!internal::IsWellFormed(&new_node, node_names)) {
      return errors::InvalidArgument(kMutableGraphViewApplyError, "new node '",
                                     new_node.node.name(), "' is ill-formed.");
    }
  }

  return Status::OK();
}

Status MutableGraphView::CheckKernelRegisteredForNodes() {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_55(mht_55_v, 1158, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::CheckKernelRegisteredForNodes");

  Status s;
  for (auto& diff : mutation_.updated_nodes_) {
    if (internal::IsEmpty(&diff)) {
      continue;
    }

    NodeDef* node = nodes_[diff.node_index].node();
    diff.processed_attrs =
        AttrValueMap(node->attr().begin(), node->attr().end());
    for (const auto& attr_to_remove : diff.attrs_to_remove) {
      (*diff.processed_attrs).erase(attr_to_remove);
    }
    for (const auto& attr_to_add : diff.attrs_to_add) {
      gtl::InsertOrUpdate(&(*diff.processed_attrs), attr_to_add.first,
                          attr_to_add.second);
    }
    const string& device = diff.update_device ? diff.device : node->device();
    DeviceNameUtils::ParsedName name;
    if (device.empty() || !DeviceNameUtils::ParseFullName(device, &name) ||
        !name.has_type) {
      continue;
    }
    s = IsKernelRegisteredForNode(diff.update_name ? diff.name : node->name(),
                                  node->has_experimental_debug_info(),
                                  node->experimental_debug_info(),
                                  diff.update_op ? diff.op : node->op(), device,
                                  AttrSlice(&(*diff.processed_attrs)));
    if (!s.ok()) {
      LOG(WARNING) << s.error_message();
    }
  }
  for (const auto& new_node_holder : mutation_.new_nodes_) {
    const auto& new_node_def = new_node_holder.node;
    DeviceNameUtils::ParsedName name;
    if (new_node_def.device().empty() ||
        !DeviceNameUtils::ParseFullName(new_node_def.device(), &name) ||
        !name.has_type) {
      continue;
    }
    s = IsKernelRegisteredForNode(new_node_def);
    if (!s.ok()) {
      LOG(WARNING) << s.error_message();
    }
  }
  return Status::OK();
}

template <typename T>
void MutableGraphView::ReplaceNodeFanouts(MutableNodeView* node, T* fanouts) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_56(mht_56_v, 1210, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::ReplaceNodeFanouts");

  node->num_regular_fanouts_ = fanouts->num_regular_fanouts_;
  node->regular_fanouts_by_port_ = std::move(fanouts->regular_fanouts_by_port_);
  for (int i = 0, i_max = node->regular_fanouts_by_port_.size(); i < i_max;
       ++i) {
    for (int j = 0, j_max = node->regular_fanouts_by_port_[i].size(); j < j_max;
         ++j) {
      auto& fanout = node->regular_fanouts_by_port_[i][j];
      auto* fanout_node_view = fanout.node_view();
      auto& fanout_fanin = fanout_node_view->regular_fanins_[fanout.index()];
      auto* fanout_fanins_count = &fanout_node_view->fanins_count_;
      DecrementFaninCount(
          fanout_fanins_count,
          {&graph_->node(fanout_fanin.node_index_), fanout_fanin.index()});
      fanout_fanin.node_index_ = node->node_index_;
      IncrementFaninCount(
          fanout_fanins_count,
          {&graph_->node(node->node_index_), fanout_fanin.index()});
    }
  }
  node->controlled_fanouts_ = std::move(fanouts->controlled_fanouts_);
  for (int i = 0, i_max = node->controlled_fanouts_.size(); i < i_max; ++i) {
    auto& fanout = node->controlled_fanouts_[i];
    auto* fanout_node_view = fanout.node_view();
    auto& fanout_fanin =
        fanout_node_view->controlling_fanins_[fanout.fanin_index_];
    auto* fanout_fanins_count = &fanout_node_view->fanins_count_;
    DecrementFaninCount(
        fanout_fanins_count,
        {&graph_->node(fanout_fanin.node_index_), Graph::kControlSlot});
    fanout_fanin.node_index_ = node->node_index_;
    fanout_fanin.fanout_index_ = i;
    IncrementFaninCount(fanout_fanins_count, {&graph_->node(node->node_index_),
                                              Graph::kControlSlot});
  }
}

void MutableGraphView::FixRenamedNodes(
    std::vector<RenamedOrOverwrittenNode>* renamed_nodes,
    absl::flat_hash_map<string, NodeViewFanouts>* renamed_fanouts,
    std::vector<bool>* overwritten_name_removed_nodes) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_57(mht_57_v, 1253, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::FixRenamedNodes");

  // Extract all renamed node fanouts.
  renamed_fanouts->reserve(renamed_nodes->size());
  for (auto& renamed : *renamed_nodes) {
    auto& diff = mutation_.updated_nodes_[renamed.renamed_update_index_];
    // Remove node index by name from graph.
    node_index_by_name_.erase(nodes_[diff.node_index].GetName());
    MutableNodeView& renamed_node = nodes_[diff.node_index];
    renamed_fanouts->try_emplace(
        renamed_node.GetName(),
        std::move(renamed_node.regular_fanouts_by_port_),
        renamed_node.num_regular_fanouts_,
        std::move(renamed_node.controlled_fanouts_));
  }

  // Replace renamed node fanouts with fanouts associated with updated name.
  for (auto& renamed : *renamed_nodes) {
    auto& diff = mutation_.updated_nodes_[renamed.renamed_update_index_];
    MutableNodeView& renamed_node = nodes_[diff.node_index];
    auto fanouts_it = renamed_fanouts->find(diff.name);
    if (fanouts_it != renamed_fanouts->end()) {
      // Another renamed node's fanout.
      auto& fanouts = fanouts_it->second;
      ReplaceNodeFanouts(&renamed_node, &fanouts);
      renamed_fanouts->erase(fanouts_it);
      // Node to be overwritten is being renamed, so it won't be overwritten.
      renamed.overwritten_node_index_ = internal::kMissingIndex;
    } else if (renamed.overwritten_node_index_ != internal::kMissingIndex) {
      // Existing node in graph.
      MutableNodeView& node_to_overwrite =
          nodes_[renamed.overwritten_node_index_];
      ReplaceNodeFanouts(&renamed_node, &node_to_overwrite);
      node_index_by_name_.erase(node_to_overwrite.GetName());
      if (mutation_.removed_nodes_.contains(node_to_overwrite.node_index_)) {
        (*overwritten_name_removed_nodes)[node_to_overwrite.node_index_] = true;
      }
    } else {
      // No existing fanouts.
      renamed_node.num_regular_fanouts_ = 0;
    }

    // Update node name.
    renamed_node.node()->set_name(diff.name);
    diff.update_name = false;
    diff.name.clear();
    // Rehash renamed nodes with updated name.
    node_index_by_name_.emplace(renamed_node.GetName(), diff.node_index);
  }
}

void MutableGraphView::AddNewNodes(
    absl::flat_hash_map<string, NodeViewFanouts>* renamed_fanouts,
    std::vector<int>* new_node_indices) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_58(mht_58_v, 1308, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::AddNewNodes");

  new_node_indices->reserve(mutation_.new_nodes_.size());
  for (auto& new_node : mutation_.new_nodes_) {
    int node_index;
    auto graph_it = node_index_by_name_.find(new_node.node.name());
    if (graph_it != node_index_by_name_.end()) {
      // Overwrite existing node.
      node_index = graph_it->second;
      MutableNodeView& node_view = nodes_[node_index];
      RemoveAllFaninFanoutInternal(&node_view);
      auto* node_def = graph_->mutable_node(node_index);
      node_def->mutable_op()->swap(*new_node.node.mutable_op());
      node_def->mutable_device()->swap(*new_node.node.mutable_device());
      node_def->mutable_input()->Clear();
      node_def->mutable_attr()->swap(*new_node.node.mutable_attr());
      mutation_.removed_nodes_.erase(node_index);
    } else {
      // New node.
      auto* new_node_def = graph_->add_node();
      *new_node_def = std::move(new_node.node);
      node_index = nodes_.size();
      nodes_.emplace_back(this, node_index);
      MutableNodeView& new_node_view = nodes_.back();
      auto it = renamed_fanouts->find(new_node_view.GetName());
      if (it != renamed_fanouts->end()) {
        // Reuse fanouts of renamed node.
        NodeViewFanouts& fanouts = it->second;
        ReplaceNodeFanouts(&new_node_view, &fanouts);
        renamed_fanouts->erase(it);
      }
      node_index_by_name_.emplace(new_node_view.GetName(), node_index);
    }
    new_node_indices->emplace_back(node_index);
  }
}

void MutableGraphView::FixRenamedFanouts(
    const absl::flat_hash_map<string, NodeViewFanouts>& renamed_fanouts) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_59(mht_59_v, 1348, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::FixRenamedFanouts");

  // Leftover fanouts in renamed_fanouts are due to nodes not existing anymore
  // or a node being renamed without another node taking its place. For these
  // leftover fanouts, mark their respective fanin fanout_index_ to
  // internal::kMissingIndex as an indicator so when it comes to updating or
  // removing fanins inplace, nodes with the same index don't get affected and
  // other fanouts are accidentally removed.
  for (auto& renamed_fanout : renamed_fanouts) {
    for (auto& regular_fanouts :
         renamed_fanout.second.regular_fanouts_by_port_) {
      for (auto& fanout : regular_fanouts) {
        auto* fanout_node_view = fanout.node_view();
        auto& fanin = fanout_node_view->regular_fanins_[fanout.index()];
        fanout_node_view->fanins_count_.erase(
            {fanin.node_view()->node(), fanin.index()});
        fanin.fanout_index_ = internal::kMissingIndex;
      }
    }
    for (auto& fanout : renamed_fanout.second.controlled_fanouts_) {
      auto* fanout_node_view = fanout.node_view();
      auto& fanin = fanout_node_view->controlling_fanins_[fanout.fanin_index_];
      fanout_node_view->fanins_count_.erase(
          {fanin.node_view()->node(), Graph::kControlSlot});
      fanout_node_view->controlling_fanins_index_.erase(renamed_fanout.first);
      fanin.fanout_index_ = internal::kMissingIndex;
    }
  }
}

inline void MutableGraphView::RemoveRegularFaninFanoutInternal(
    MutableNodeView* node_view, int i) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_60(mht_60_v, 1381, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemoveRegularFaninFanoutInternal");

  MutableFanoutView& fanin = node_view->regular_fanins_[i];
  // Fanin was marked as removed via FixRenamedFanouts.
  if (fanin.fanout_index_ == internal::kMissingIndex) {
    return;
  }

  DecrementFaninCount(&node_view->fanins_count_,
                      {&graph_->node(fanin.node_index_), fanin.index()});
  auto* fanin_node_view = fanin.node_view();
  auto& fanouts = fanin_node_view->regular_fanouts_by_port_[fanin.index()];
  int fanouts_size = fanouts.size();
  if (fanin.fanout_index_ < fanouts_size - 1) {
    // Swap fanout with last fanout in vector, and update it's associated fanin
    // index.
    MutableFaninView& last_fanout = fanouts.back();
    last_fanout.node_view()
        ->regular_fanins_[last_fanout.index()]
        .fanout_index_ = fanin.fanout_index_;
    std::swap(last_fanout, fanouts[fanin.fanout_index_]);
  }
  // Remove fanout.
  fanouts.pop_back();
  --fanin.node_view()->num_regular_fanouts_;

  // Resize fanouts. Fanouts may not be removed sequentially in relation to
  // output port, so trailing empty output ports may be left behind. It is
  // necessary to loop through all of the output ports to determine the maximum
  // output port before resizing.
  int last_fanout_index = fanin_node_view->regular_fanouts_by_port_.size();
  for (int i = fanin_node_view->regular_fanouts_by_port_.size() - 1; i >= 0;
       --i) {
    if (fanin_node_view->regular_fanouts_by_port_[i].empty()) {
      last_fanout_index = i;
    } else {
      break;
    }
  }
  int fanin_node_view_regular_fanouts_by_port_size =
      fanin_node_view->regular_fanouts_by_port_.size();
  if (last_fanout_index < fanin_node_view_regular_fanouts_by_port_size) {
    fanin_node_view->regular_fanouts_by_port_.resize(last_fanout_index);
  }
}

inline void MutableGraphView::AddRegularFaninInternal(
    MutableNodeView* node_view, const SafeTensorId& fanin_id) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_61(mht_61_v, 1430, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::AddRegularFaninInternal");

  MutableNodeView* fanin_node_view = GetNode(fanin_id.node());
  // Resize fanouts to include new output port index.
  int fanin_node_view_regular_fanouts_by_port_size =
      fanin_node_view->regular_fanouts_by_port_.size();
  if (fanin_node_view_regular_fanouts_by_port_size < fanin_id.index() + 1) {
    fanin_node_view->regular_fanouts_by_port_.resize(fanin_id.index() + 1);
  }

  // Add node as fanout to fanin.
  auto& fanouts = fanin_node_view->regular_fanouts_by_port_[fanin_id.index()];
  fanouts.emplace_back(this, node_view->node_index(),
                       node_view->regular_fanins_.size(),
                       node_view->regular_fanins_.size());
  ++fanin_node_view->num_regular_fanouts_;

  // Add fanin to node.
  node_view->regular_fanins_.emplace_back(this, fanin_node_view->node_index(),
                                          fanin_id.index(), fanouts.size() - 1);
  IncrementFaninCount(
      &node_view->fanins_count_,
      {&graph_->node(fanin_node_view->node_index()), fanin_id.index()});
}

inline void MutableGraphView::UpdateRegularFaninInternal(
    MutableNodeView* node_view, const int i, const SafeTensorId& fanin_id) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_62(mht_62_v, 1458, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::UpdateRegularFaninInternal");

  // Remove fanin.
  RemoveRegularFaninFanoutInternal(node_view, i);

  MutableNodeView* fanin_node_view = GetNode(fanin_id.node());
  // Resize fanouts to include new output port index.
  int fanin_node_view_regular_fanouts_by_port_size =
      fanin_node_view->regular_fanouts_by_port_.size();
  if (fanin_node_view_regular_fanouts_by_port_size < fanin_id.index() + 1) {
    fanin_node_view->regular_fanouts_by_port_.resize(fanin_id.index() + 1);
  }

  // Add node as fanout to fanin.
  auto& fanouts = fanin_node_view->regular_fanouts_by_port_[fanin_id.index()];
  fanouts.emplace_back(this, node_view->node_index(), i, i);
  ++fanin_node_view->num_regular_fanouts_;

  // Replace fanin in node.
  node_view->regular_fanins_[i] =
      MutableFanoutView(this, fanin_node_view->node_index(), fanin_id.index(),
                        fanouts.size() - 1);
  IncrementFaninCount(
      &node_view->fanins_count_,
      {&graph_->node(fanin_node_view->node_index()), fanin_id.index()});
}

inline void MutableGraphView::RemoveControllingFaninFanoutInternal(
    MutableNodeView* node_view, int i) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_63(mht_63_v, 1488, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemoveControllingFaninFanoutInternal");

  auto& control_to_remove = node_view->controlling_fanins_[i];
  if (control_to_remove.fanout_index_ != internal::kMissingIndex) {
    // Update internal state associated with node.
    node_view->fanins_count_.erase(
        {control_to_remove.node_view()->node(), Graph::kControlSlot});
    node_view->controlling_fanins_index_.erase(
        control_to_remove.node_view()->GetName());

    // Remove controlled fanout from controlling fanin, via swapping last
    // controlled fanout in controlling fanin with controlled fanout to be
    // removed.
    auto* control_to_remove_view = control_to_remove.node_view();
    int control_to_remove_view_controlled_fanouts_size =
        control_to_remove_view->controlled_fanouts_.size();
    if (control_to_remove.fanout_index_ <
        control_to_remove_view_controlled_fanouts_size - 1) {
      auto& control_to_remove_view_last_control =
          control_to_remove_view->controlled_fanouts_.back();
      control_to_remove_view_last_control.node_view()
          ->controlling_fanins_[control_to_remove_view_last_control
                                    .fanin_index_]
          .fanout_index_ = control_to_remove.fanout_index_;
      std::swap(control_to_remove_view_last_control,
                control_to_remove_view
                    ->controlled_fanouts_[control_to_remove.fanout_index_]);
    }
    control_to_remove_view->controlled_fanouts_.pop_back();
  }
}

inline void MutableGraphView::RemoveControllingFaninInternal(
    MutableNodeView* node_view, const std::set<int>& indices_to_remove) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_64(mht_64_v, 1523, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemoveControllingFaninInternal");

  const int num_regular_fanins = node_view->NumRegularFanins();
  auto* mutable_input = node_view->node()->mutable_input();
  // Iterate in descending order so indices stay consistent.
  for (auto rit = indices_to_remove.rbegin(); rit != indices_to_remove.rend();
       ++rit) {
    const int control_index = *rit;
    RemoveControllingFaninFanoutInternal(node_view, control_index);

    // Swap last controlling fanin in node with controlling fanin to be removed.
    int node_view_controlling_fanins_size =
        node_view->controlling_fanins_.size();
    if (control_index < node_view_controlling_fanins_size - 1) {
      auto& last_control = node_view->controlling_fanins_.back();
      auto* last_control_view = last_control.node_view();
      last_control_view->controlled_fanouts_[last_control.fanout_index_]
          .fanin_index_ = control_index;
      node_view->controlling_fanins_index_.find(last_control_view->GetName())
          ->second = control_index;
      mutable_input->SwapElements(
          num_regular_fanins + control_index,
          num_regular_fanins + node_view->NumControllingFanins() - 1);
      std::swap(last_control, node_view->controlling_fanins_[control_index]);
    }
    mutable_input->RemoveLast();
    node_view->controlling_fanins_.pop_back();
  }
}

inline void MutableGraphView::AddControllingFaninInternal(
    MutableNodeView* node_view, absl::string_view fanin_node_name) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_65(mht_65_v, 1557, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::AddControllingFaninInternal");

  NodeDef* node = node_view->node();
  // Add controlling fanin to NodeDef.
  node->add_input(AsControlDependency(string(fanin_node_name)));
  MutableNodeView* fanin_node_view = GetNode(fanin_node_name);
  const int index = node_view->controlling_fanins_.size();
  fanin_node_view->controlled_fanouts_.emplace_back(
      this, node_view->node_index(), Graph::kControlSlot, index);
  node_view->controlling_fanins_.emplace_back(
      this, fanin_node_view->node_index(), Graph::kControlSlot,
      fanin_node_view->controlled_fanouts_.size() - 1);
  IncrementFaninCount(
      &node_view->fanins_count_,
      {&graph_->node(fanin_node_view->node_index()), Graph::kControlSlot});
  // Parse new fanin string for node name.
  TensorId tensor_id = ParseTensorName(node->input(node->input_size() - 1));
  node_view->controlling_fanins_index_.emplace(tensor_id.node(), index);
}

void MutableGraphView::ApplyNodeUpdates() {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_66(mht_66_v, 1579, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::ApplyNodeUpdates");

  for (auto& diff : mutation_.updated_nodes_) {
    if (internal::IsEmpty(&diff)) {
      continue;
    }
    MutableNodeView& node_view = nodes_[diff.node_index];
    diff.node_index = internal::kMissingIndex;
    // Clean up node view.
    node_view.update_index_ = internal::kMissingIndex;

    NodeDef* node_def = node_view.node();

    // Set updated fields and attributes of node.
    if (diff.update_op) {
      node_def->set_op(diff.op);
    }
    if (diff.update_device) {
      node_def->set_device(diff.device);
    }
    node_def->mutable_attr()->swap((*diff.processed_attrs));

    // Updated fanins. Only one of `regular_inputs_to_remove_` or
    // `regular_inputs_to_add_` can be set.
    if (diff.num_regular_inputs_to_remove > 0) {
      // Truncate trailing regular fanins.
      const int first_index =
          node_view.NumRegularFanins() - diff.num_regular_inputs_to_remove;
      for (int i = first_index; i < node_view.NumRegularFanins(); ++i) {
        RemoveRegularFaninFanoutInternal(&node_view, i);
      }
      node_view.regular_fanins_.resize(first_index);
      node_def->mutable_input()->DeleteSubrange(
          node_view.NumRegularFanins(), diff.num_regular_inputs_to_remove);
    } else if (diff.num_regular_inputs_to_add > 0) {
      // Append regular fanins.
      node_def->mutable_input()->Reserve(node_def->mutable_input()->size() +
                                         diff.num_regular_inputs_to_add);
      int curr_index = node_view.NumRegularFanins();
      int curr_control_start = curr_index;
      for (const SafeTensorId& fanin : diff.regular_inputs_to_add) {
        AddRegularFaninInternal(&node_view, fanin);
        node_def->add_input(SafeTensorIdToString(fanin));
        node_def->mutable_input()->SwapElements(curr_index,
                                                node_def->input_size() - 1);
        if (curr_control_start == curr_index) {
          curr_control_start = node_def->input_size() - 1;
        }
        ++curr_index;
      }
      // Rotate shifted controlling fanins to match up with
      // `node_view.controlling_fanins_` as `num_regular_inputs_to_add_` may not
      // be a multiple of `num_regular_inputs_to_add_`. This is to prevent
      // rehashing controlling fanins in `node_view.controlling_fanins_index_`.
      if (node_view.NumControllingFanins() > 1 &&
          curr_control_start != node_view.NumRegularFanins()) {
        std::rotate(
            node_def->mutable_input()->begin() + node_view.NumRegularFanins(),
            node_def->mutable_input()->begin() + curr_control_start,
            node_def->mutable_input()->end());
      }
    }

    for (const auto& update_fanin : diff.regular_inputs_to_update) {
      UpdateRegularFaninInternal(&node_view, update_fanin.first,
                                 update_fanin.second);
      node_def->set_input(update_fanin.first,
                          SafeTensorIdToString(update_fanin.second));
    }

    RemoveControllingFaninInternal(&node_view,
                                   diff.controlling_inputs_to_remove);

    node_def->mutable_input()->Reserve(node_def->mutable_input()->size() +
                                       diff.controlling_inputs_to_add.size());
    for (const auto& control_to_add : diff.controlling_inputs_to_add) {
      AddControllingFaninInternal(&node_view, control_to_add);
    }
  }
}

void MutableGraphView::SetNewNodesFanins(
    const std::vector<int>& new_node_indices) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_67(mht_67_v, 1663, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::SetNewNodesFanins");

  auto new_node = mutation_.new_nodes_.begin();
  for (const int new_node_index : new_node_indices) {
    MutableNodeView& new_node_view = nodes_[new_node_index];
    NodeDef* new_node_def = new_node_view.node();
    new_node_def->mutable_input()->Reserve(new_node->num_regular_fanins +
                                           new_node->controlling_fanins.size());
    for (const SafeTensorId& fanin : new_node->regular_fanins) {
      AddRegularFaninInternal(&new_node_view, fanin);
      new_node_def->add_input(SafeTensorIdToString(fanin));
    }
    for (const string& control_to_add : new_node->controlling_fanins) {
      AddControllingFaninInternal(&new_node_view, control_to_add);
    }
    ++new_node;
  }
}

inline void MutableGraphView::RemoveAllFaninFanoutInternal(
    MutableNodeView* node_view) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_68(mht_68_v, 1685, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemoveAllFaninFanoutInternal");

  const int num_regular_fanins = node_view->NumRegularFanins();
  for (int i = 0; i < num_regular_fanins; ++i) {
    RemoveRegularFaninFanoutInternal(node_view, i);
  }
  std::vector<MutableFanoutView>().swap(node_view->regular_fanins_);
  const int num_controlling_fanins = node_view->NumControllingFanins();
  for (int i = 0; i < num_controlling_fanins; ++i) {
    RemoveControllingFaninFanoutInternal(node_view, i);
  }
  std::vector<MutableFanoutView>().swap(node_view->controlling_fanins_);
}

void MutableGraphView::RemoveNodesInternal(
    const std::vector<RenamedOrOverwrittenNode>& renamed_nodes,
    const std::vector<bool>& overwritten_name_removed_nodes) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_69(mht_69_v, 1703, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::RemoveNodesInternal");

  // Get all nodes overwritten by renamed nodes and remove their fanins.
  std::vector<int> overwritten_nodes;
  overwritten_nodes.reserve(renamed_nodes.size());
  for (const auto& renamed : renamed_nodes) {
    if (renamed.overwritten_node_index_ != internal::kMissingIndex) {
      auto& node = nodes_[renamed.overwritten_node_index_];
      RemoveAllFaninFanoutInternal(&node);
      overwritten_nodes.emplace_back(renamed.overwritten_node_index_);
    }
  }

  // Get all nodes explicitly marked for removal and remove their fanins.
  std::vector<int> node_indices_to_remove;
  node_indices_to_remove.reserve(mutation_.updated_nodes_.size() +
                                 overwritten_nodes.size());
  for (int node_index : mutation_.removed_nodes_) {
    auto& node = nodes_[node_index];
    RemoveAllFaninFanoutInternal(&node);
    node_indices_to_remove.push_back(node_index);
    if (!overwritten_name_removed_nodes[node_index]) {
      node_index_by_name_.erase(node.GetName());
    }
  }
  node_indices_to_remove.insert(node_indices_to_remove.end(),
                                overwritten_nodes.begin(),
                                overwritten_nodes.end());
  std::set<int> sorted_node_indices_to_remove(node_indices_to_remove.begin(),
                                              node_indices_to_remove.end());

  // Iterate in descending order so indices stay consistent.
  for (auto rit = sorted_node_indices_to_remove.rbegin();
       rit != sorted_node_indices_to_remove.rend(); ++rit) {
    const int removed_node_index = *rit;
    MutableNodeView& last_node = nodes_.back();
    if (last_node.node_index_ > removed_node_index) {
      last_node.node_index_ = removed_node_index;
      for (auto& regular_fanin : last_node.regular_fanins_) {
        // Update fanouts of regular fanins with new index.
        regular_fanin.node_view()
            ->regular_fanouts_by_port_[regular_fanin.index()]
                                      [regular_fanin.fanout_index_]
            .node_index_ = removed_node_index;
      }
      for (auto& controlling_fanin : last_node.controlling_fanins_) {
        // Update fanouts of controlling fanins with new index.
        controlling_fanin.node_view()
            ->controlled_fanouts_[controlling_fanin.fanout_index_]
            .node_index_ = removed_node_index;
      }
      for (auto& regular_fanouts : last_node.regular_fanouts_by_port_) {
        for (auto& regular_fanout : regular_fanouts) {
          // Update fanins of regular fanouts.
          MutableNodeView* fanout_node_view = regular_fanout.node_view();
          fanout_node_view->regular_fanins_[regular_fanout.fanin_index_]
              .node_index_ = removed_node_index;
        }
      }
      for (auto& controlled_fanout : last_node.controlled_fanouts_) {
        // Update fanins of controlled fanouts.
        MutableNodeView* fanout_node_view = controlled_fanout.node_view();
        fanout_node_view->controlling_fanins_[controlled_fanout.fanin_index_]
            .node_index_ = removed_node_index;
      }

      const int last_node_index = nodes_.size() - 1;
      std::swap(nodes_[last_node_index], nodes_[removed_node_index]);
      graph()->mutable_node()->SwapElements(last_node_index,
                                            removed_node_index);
      node_index_by_name_.find(nodes_[removed_node_index].GetName())->second =
          removed_node_index;
    }
    nodes_.pop_back();
  }
  if (!sorted_node_indices_to_remove.empty()) {
    const int current_size = graph()->node_size();
    const int num_to_remove = sorted_node_indices_to_remove.size();
    graph()->mutable_node()->DeleteSubrange(current_size - num_to_remove,
                                            num_to_remove);
  }
}

namespace {
constexpr int kTopologicalSortDone = -1;

const char kMutableGraphViewSortTopologicallyError[] =
    "MutableGraphView::SortTopologically error: ";

// TraversalState is an enum representing the state of a node when it is being
// traversed via DFS.
enum TraversalState : uint8_t { PENDING, PROCESSING, PROCESSED };

// RecursionStackState is an enum representing the recursion stack state
// when using DFS iteratively. `ENTER` is the state representing entering into
// a recursive call, while `EXIT` is the state representing exiting a
// recursive call.
enum RecursionStackState : bool { ENTER, EXIT };

// RecursionStackEntry is a helper struct representing an instance of a
// recursive call in the iterative DFS simulating a recursive ordering.
struct RecursionStackEntry {
  RecursionStackEntry(int node_index, RecursionStackState recursion_state)
      : node_index(node_index), recursion_state(recursion_state) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_70(mht_70_v, 1808, "", "./tensorflow/core/grappler/utils/graph_view.cc", "RecursionStackEntry");
}

  const int node_index;
  const RecursionStackState recursion_state;
};

// Edge is a helper struct representing an edge in the graph.
struct Edge {
  Edge(int from, int to) : from(from), to(to) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_71(mht_71_v, 1819, "", "./tensorflow/core/grappler/utils/graph_view.cc", "Edge");
}

  const int from;
  const int to;
};
}  // namespace

Status MutableGraphView::SortTopologically(
    bool ignore_cycles,
    absl::Span<const TopologicalDependency> extra_dependencies) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_72(mht_72_v, 1831, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::SortTopologically");

  if (!mutation_.updated_nodes_.empty() || !mutation_.new_nodes_.empty()) {
    // Cannot sort when there is an active mutation due to indices possibly
    // being changed or invalidated.
    return errors::InvalidArgument(kMutableGraphViewSortTopologicallyError,
                                   "active mutation exists.");
  }

  const int num_nodes = nodes_.size();

  // Group extra dependencies by `from` node.
  absl::flat_hash_map<int, std::vector<int>> extra_dependencies_by_parent;
  for (const auto& extra_dependency : extra_dependencies) {
    if (extra_dependency.graph_view_ != this ||
        extra_dependency.from_ == extra_dependency.to_ ||
        extra_dependency.from_ < 0 || extra_dependency.from_ >= num_nodes ||
        extra_dependency.to_ < 0 || extra_dependency.to_ >= num_nodes) {
      return errors::InvalidArgument(kMutableGraphViewSortTopologicallyError,
                                     "invalid extra dependencies.");
    }
    extra_dependencies_by_parent[extra_dependency.from_].push_back(
        extra_dependency.to_);
  }

  // Reversed colored post-order DFS traversal. This does not fail on cycles,
  // but there are no guarantees on ordering within a cycle.
  std::vector<TraversalState> traversal_state(num_nodes, PENDING);
  int curr_pos = num_nodes - 1;
  std::vector<int> order(num_nodes);
  std::vector<Edge> edges_in_cycle;

  auto push_onto_stack = [this](
                             const int curr_index, const int fanout_index,
                             std::vector<RecursionStackEntry>* recursion_stack,
                             std::vector<TraversalState>* traversal_state,
                             std::vector<Edge>* edges_in_cycle) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_73(mht_73_v, 1869, "", "./tensorflow/core/grappler/utils/graph_view.cc", "lambda");

    // Ignore NextIteration -> Merge connections to break control flow cycles.
    if (IsNextIteration(graph_->node(curr_index)) &&
        IsMerge(graph_->node(fanout_index))) {
      return;
    }
    auto& fanout_traversal_state = (*traversal_state)[fanout_index];
    if (fanout_traversal_state == PROCESSING) {
      // Cycle detected.
      edges_in_cycle->push_back({curr_index, fanout_index});
    } else if (fanout_traversal_state == PENDING) {
      // Unvisited node, simply add to stack for future traversal.
      recursion_stack->push_back({fanout_index, ENTER});
    }
  };

  auto process_fanouts = [this, &extra_dependencies_by_parent,
                          &push_onto_stack](
                             const int curr_index,
                             std::vector<RecursionStackEntry>* recursion_stack,
                             std::vector<TraversalState>* traversal_state,
                             std::vector<Edge>* edges_in_cycle) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_74(mht_74_v, 1893, "", "./tensorflow/core/grappler/utils/graph_view.cc", "lambda");

    const auto& node_view = nodes_[curr_index];
    // Regular fanouts.
    for (const auto& regular_fanouts_port_i : node_view.GetRegularFanouts()) {
      for (const auto& regular_fanout : regular_fanouts_port_i) {
        push_onto_stack(curr_index, regular_fanout.node_index_, recursion_stack,
                        traversal_state, edges_in_cycle);
      }
    }
    // Controlled fanouts.
    for (const auto& controlled_fanout : node_view.GetControlledFanouts()) {
      push_onto_stack(curr_index, controlled_fanout.node_index_,
                      recursion_stack, traversal_state, edges_in_cycle);
    }
    // Extra dependencies.
    auto it = extra_dependencies_by_parent.find(curr_index);
    if (it != extra_dependencies_by_parent.end()) {
      for (const auto& extra_fanout : it->second) {
        push_onto_stack(curr_index, extra_fanout, recursion_stack,
                        traversal_state, edges_in_cycle);
      }
    }
  };

  auto reversed_postorder_dfs =
      [&process_fanouts](const MutableNodeView& root_node_view,
                         std::vector<int>* order,
                         std::vector<TraversalState>* traversal_state,
                         int* curr_pos, std::vector<Edge>* edges_in_cycle) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_75(mht_75_v, 1924, "", "./tensorflow/core/grappler/utils/graph_view.cc", "lambda");

        std::vector<RecursionStackEntry> recursion_stack;
        // Add the root to stack to start the traversal.
        const int root_index = root_node_view.node_index_;
        auto& root_traversal_state = (*traversal_state)[root_index];
        if (root_traversal_state == PENDING) {
          recursion_stack.push_back({root_index, ENTER});
        }
        while (!recursion_stack.empty()) {
          auto curr_entry = recursion_stack.back();
          recursion_stack.pop_back();
          const int curr_index = curr_entry.node_index;
          auto& curr_traversal_state = (*traversal_state)[curr_index];
          if (curr_traversal_state == PROCESSED) {
            // Node already processed which can be ignored.
            continue;
          } else if (curr_entry.recursion_state == EXIT) {
            // Node from recursion stack where all fanouts were visited.
            // Instead of adding node index to a vector, simply set what its
            // index would be, so there will not be a need for inversion later
            // on. The value set is in decending order so the reversed
            // post-order is returned.
            (*order)[curr_index] = *curr_pos;
            curr_traversal_state = PROCESSED;
            --(*curr_pos);
          } else {
            // Process current node and fanouts.
            curr_traversal_state = PROCESSING;
            recursion_stack.push_back({curr_index, EXIT});
            process_fanouts(curr_index, &recursion_stack, traversal_state,
                            edges_in_cycle);
          }
        }
      };

  // Determine sources to start DFS (nodes with no inputs) and unique fanout
  // nodes.
  for (int i = num_nodes - 1; i >= 0; --i) {
    auto& node = nodes_[i];
    if (node.NumRegularFanins() + node.NumControllingFanins() == 0) {
      reversed_postorder_dfs(node, &order, &traversal_state, &curr_pos,
                             &edges_in_cycle);
    }
  }

  if (!ignore_cycles && !edges_in_cycle.empty()) {
    std::vector<string> edges_formatted;
    edges_formatted.reserve(edges_in_cycle.size());
    for (const auto& edge : edges_in_cycle) {
      edges_formatted.push_back(
          absl::StrCat("'", graph_->node(edge.from).name(), "' -> '",
                       graph_->node(edge.to).name(), "'"));
    }
    const string edges_str =
        absl::StrCat("{", absl::StrJoin(edges_formatted, ", "), "}");
    return errors::InvalidArgument(kMutableGraphViewSortTopologicallyError,
                                   "detected edge(s) creating cycle(s) ",
                                   edges_str, ".");
  }
  if (curr_pos != kTopologicalSortDone) {
    // Not all nodes were processed.
    if (!ignore_cycles) {
      return errors::InvalidArgument(
          kMutableGraphViewSortTopologicallyError,
          "was not able to sort all nodes topologically.");
    }
    // Otherwise process all nodes regardless of cycles.
    for (const auto& node : nodes_) {
      reversed_postorder_dfs(node, &order, &traversal_state, &curr_pos,
                             &edges_in_cycle);
    }
  }

  // Permute nodes by reversed post-order DFS.
  std::vector<MutableNodeView> permuted_nodes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    permuted_nodes[order[i]] = std::move(nodes_[i]);
  }
  nodes_.swap(permuted_nodes);

  // Fix up indices of MutableNodeViews.
  for (MutableNodeView& node_view : nodes_) {
    const int prev_node_index = node_view.node_index_;
    if (prev_node_index != order[prev_node_index]) {
      const string& node_name = graph_->node(prev_node_index).name();
      node_view.node_index_ = order[prev_node_index];
      node_index_by_name_.find(node_name)->second = node_view.node_index_;
    }
    for (MutableFanoutView& regular_fanin : node_view.regular_fanins_) {
      regular_fanin.node_index_ = order[regular_fanin.node_index_];
    }
    for (MutableFanoutView& controlling_fanin : node_view.controlling_fanins_) {
      controlling_fanin.node_index_ = order[controlling_fanin.node_index_];
    }
    for (std::vector<MutableFaninView>& regular_fanouts_port_i :
         node_view.regular_fanouts_by_port_) {
      for (MutableFaninView& regular_fanout : regular_fanouts_port_i) {
        regular_fanout.node_index_ = order[regular_fanout.node_index_];
      }
    }
    for (MutableFaninView& controlled_fanout : node_view.controlled_fanouts_) {
      controlled_fanout.node_index_ = order[controlled_fanout.node_index_];
    }
  }

  // Permute graph NodeDefs.
  PermuteNodesInPlace(graph_, &order, /*invert_permutation=*/false);

  return Status::OK();
}

inline Status MutableGraphView::ValidateInternal(
    absl::flat_hash_map<absl::string_view, int>* node_names,
    std::vector<RenamedOrOverwrittenNode>* renamed_nodes,
    std::vector<int>* inplace_nodes,
    std::vector<int>* empty_diff_node_indices) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_76(mht_76_v, 2042, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::ValidateInternal");

  // Get node names and partition updated_nodes_ by if they are renamed or not,
  // skipping empty MutableNodeViewDiff.
  TF_RETURN_IF_ERROR(GetNodeNamesAndPartitionUpdatedNodes(
      node_names, renamed_nodes, inplace_nodes, empty_diff_node_indices));

  // Check existence of fanins and validity (i.e. no self loops).
  TF_RETURN_IF_ERROR(
      CheckNodeNamesAndFanins(*node_names, *renamed_nodes, *inplace_nodes));

  // Check if nodes after mutation have kernels registered.
  TF_RETURN_IF_ERROR(CheckKernelRegisteredForNodes());

  return Status::OK();
}

Status MutableGraphView::ApplyMutationInternal() {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_viewDTcc mht_77(mht_77_v, 2061, "", "./tensorflow/core/grappler/utils/graph_view.cc", "MutableGraphView::ApplyMutationInternal");

  // Node name -> node index mapping. If a node index is -1, the associated node
  // with key node name exists. Otherwise the node index is the node's index in
  // the graph.
  absl::flat_hash_map<absl::string_view, int> node_names;
  // Indices of MutableNodeViewDiff in Mutation::updated_nodes_ where nodes are
  // renamed (and possibly have other fields mutated).
  std::vector<RenamedOrOverwrittenNode> renamed_nodes;
  // Indices of MutableNodeViewDiff in Mutation::updated_nodes_ where nodes are
  // not renamed but have fields mutated.
  std::vector<int> inplace_nodes;
  // Indices of nodes in graph where MutableNodeViewDiff are empty.
  // `update_index_` of nodes associated to empty MutableNodeViewDiff should be
  // cleared after validation success.
  std::vector<int> empty_diff_node_indices;

  // Check if this mutation is valid before applying, and partition
  // updated_nodes_ into inplace mutated nodes and renamed nodes.
  TF_RETURN_IF_ERROR(ValidateInternal(
      &node_names, &renamed_nodes, &inplace_nodes, &empty_diff_node_indices));

  // Clear `update_index_` of MutableNodeView with empty associated
  // MutableNodeViewDiff.
  for (const int empty_diff_node_index : empty_diff_node_indices) {
    nodes_[empty_diff_node_index].update_index_ = internal::kMissingIndex;
  }

  // Node name and associated fanouts.
  absl::flat_hash_map<string, NodeViewFanouts> renamed_fanouts;
  // Removed nodes where name was overwritten by a renamed node.
  std::vector<bool> overwritten_name_removed_nodes(nodes_.size());
  // Fix renaming of existing nodes by swapping fanouts and rehashing names.
  // This will also overwrite removed or unmodified nodes.
  FixRenamedNodes(&renamed_nodes, &renamed_fanouts,
                  &overwritten_name_removed_nodes);

  // Indices of nodes in graph where new nodes were inserted/appended. These
  // will be corresponding to `new_nodes_` in order.
  std::vector<int> new_node_indices;
  // Add new nodes, overwriting removed or unmodified nodes.
  AddNewNodes(&renamed_fanouts, &new_node_indices);

  // For abandoned fanouts, mark their respective fanins so the original node
  // associated will not have their fanouts removed and be left in an
  // inconsistent state.
  FixRenamedFanouts(renamed_fanouts);

  // Apply mutations to updated nodes (renamed nodes are treated as inplace
  // nodes as they have already been renamed). Removed nodes are ignored.
  ApplyNodeUpdates();

  // Set fanins of new nodes.
  SetNewNodesFanins(new_node_indices);

  // Remove overwritten nodes and updated nodes set to be removed.
  RemoveNodesInternal(renamed_nodes, overwritten_name_removed_nodes);

  mutation_.ResetInternal();

  mutation_.mutation_counter_++;

  return Status::OK();
}

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
