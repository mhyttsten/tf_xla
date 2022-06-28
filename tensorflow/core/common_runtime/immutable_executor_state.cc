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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc() {
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

#include "tensorflow/core/common_runtime/immutable_executor_state.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {
bool IsInitializationOp(const Node* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "IsInitializationOp");

  return node->op_def().allows_uninitialized_input();
}
}  // namespace

ImmutableExecutorState::~ImmutableExecutorState() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ImmutableExecutorState::~ImmutableExecutorState");

  for (int32_t i = 0; i < gview_.num_nodes(); i++) {
    NodeItem* item = gview_.node(i);
    if (item != nullptr) {
      params_.delete_kernel(item->kernel);
    }
  }
}

namespace {
void GetMaxPendingCounts(const Node* n, size_t* max_pending,
                         size_t* max_dead_count) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "GetMaxPendingCounts");

  const size_t num_in_edges = n->in_edges().size();
  size_t initial_count;
  if (IsMerge(n)) {
    // merge waits all control inputs so we initialize the pending
    // count to be the number of control edges.
    int32_t num_control_edges = 0;
    for (const Edge* edge : n->in_edges()) {
      if (edge->IsControlEdge()) {
        num_control_edges++;
      }
    }
    // Use bit 0 to indicate if we are waiting for a ready live data input.
    initial_count = 1 + (num_control_edges << 1);
  } else {
    initial_count = num_in_edges;
  }

  *max_pending = initial_count;
  *max_dead_count = num_in_edges;
}
}  // namespace

ImmutableExecutorState::FrameInfo* ImmutableExecutorState::EnsureFrameInfo(
    const string& fname) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ImmutableExecutorState::EnsureFrameInfo");

  auto iter = frame_info_.find(fname);
  if (iter != frame_info_.end()) {
    return iter->second.get();
  } else {
    auto frame_info = absl::make_unique<FrameInfo>(fname);
    absl::string_view fname_view = frame_info->name;
    auto emplace_result =
        frame_info_.emplace(fname_view, std::move(frame_info));
    return emplace_result.first->second.get();
  }
}

Status ImmutableExecutorState::Initialize(const Graph& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ImmutableExecutorState::Initialize");

  TF_RETURN_IF_ERROR(gview_.Initialize(&graph));

  // Build the information about frames in this subgraph.
  ControlFlowInfo cf_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(&graph, &cf_info));

  for (auto& it : cf_info.unique_frame_names) {
    EnsureFrameInfo(it)->nodes =
        absl::make_unique<std::vector<const NodeItem*>>();
  }
  root_frame_info_ = frame_info_[""].get();

  pending_ids_.resize(gview_.num_nodes());

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  requires_control_flow_ = false;
  for (const Node* n : graph.nodes()) {
    if (IsSink(n)) continue;
    if (IsSwitch(n) || IsMerge(n) || IsEnter(n) || IsExit(n)) {
      requires_control_flow_ = true;
    } else if (IsRecv(n)) {
      // A Recv node from a different device may produce dead tensors from
      // non-local control-flow nodes.
      //
      // TODO(mrry): Track whether control flow was present in the
      // pre-partitioned graph, and enable the caller (e.g.
      // `DirectSession`) to relax this constraint.
      string send_device;
      string recv_device;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "send_device", &send_device));
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "recv_device", &recv_device));
      if (send_device != recv_device) {
        requires_control_flow_ = true;
      }
    }

    const int id = n->id();
    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);

    NodeItem* item = gview_.node(id);
    item->node_id = id;

    item->input_start = frame_info->total_inputs;
    frame_info->total_inputs += n->num_inputs();

    Status s = params_.create_kernel(n->properties(), &item->kernel);
    if (!s.ok()) {
      params_.delete_kernel(item->kernel);
      item->kernel = nullptr;
      s = AttachDef(s, *n);
      return s;
    }
    CHECK(item->kernel);
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);
    item->is_any_consumer_merge_or_control_trigger = false;
    for (const Node* consumer : n->out_nodes()) {
      if (IsMerge(consumer) || IsControlTrigger(consumer)) {
        item->is_any_consumer_merge_or_control_trigger = true;
        break;
      }
    }
    const Tensor* const_tensor = item->kernel->const_tensor();
    if (const_tensor) {
      // Hold onto a shallow copy of the constant tensor in `*this` so that the
      // reference count does not drop to 1. This prevents the constant tensor
      // from being forwarded, and its buffer reused.
      const_tensors_.emplace_back(*const_tensor);
    }
    item->const_tensor = const_tensor;
    item->is_noop = (item->kernel->type_string_view() == "NoOp");
    item->is_enter = IsEnter(n);
    if (item->is_enter) {
      bool is_constant_enter;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "is_constant", &is_constant_enter));
      item->is_constant_enter = is_constant_enter;

      string frame_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &frame_name));
      FrameInfo* frame_info = frame_info_[frame_name].get();

      int parallel_iterations;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "parallel_iterations", &parallel_iterations));

      if (frame_info->parallel_iterations == -1) {
        frame_info->parallel_iterations = parallel_iterations;
      } else if (frame_info->parallel_iterations != parallel_iterations) {
        LOG(WARNING) << "Loop frame \"" << frame_name
                     << "\" had two different values for parallel_iterations: "
                     << frame_info->parallel_iterations << " vs. "
                     << parallel_iterations << ".";
      }

      if (enter_frame_info_.size() <= id) {
        enter_frame_info_.resize(id + 1);
      }
      enter_frame_info_[id] = frame_info;
    } else {
      item->is_constant_enter = false;
    }
    item->is_exit = IsExit(n);
    item->is_control_trigger = IsControlTrigger(n);
    item->is_source = IsSource(n);
    item->is_enter_exit_or_next_iter =
        (IsEnter(n) || IsExit(n) || IsNextIteration(n));
    item->is_transfer_node = IsTransferNode(n);
    item->is_initialization_op = IsInitializationOp(n);
    item->is_recv_or_switch = IsRecv(n) || IsSwitch(n);
    item->is_next_iteration = IsNextIteration(n);
    item->is_distributed_communication = IsDistributedCommunication(n);

    // Compute the maximum values we'll store for this node in the
    // pending counts data structure, and allocate a handle in
    // that frame's pending counts data structure that has enough
    // space to store these maximal count values.
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    pending_ids_[id] =
        frame_info->pending_counts_layout.CreateHandle(max_pending, max_dead);

    // See if this node is a root node, and if so, add item to root_nodes_.
    if (n->in_edges().empty()) {
      root_nodes_.push_back(item);
    }

    // Initialize static information about the frames in the graph.
    frame_info->nodes->push_back(item);
    if (item->is_enter) {
      string enter_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &enter_name));
      EnsureFrameInfo(enter_name)->input_count++;
    }

    // Record information about whether each output of the op is used.
    std::unique_ptr<bool[]> outputs_required(new bool[n->num_outputs()]);
    std::fill(&outputs_required[0], &outputs_required[n->num_outputs()], false);
    int32_t unused_outputs = n->num_outputs();
    for (const Edge* e : n->out_edges()) {
      if (IsSink(e->dst())) continue;
      if (e->src_output() >= 0) {
        if (!outputs_required[e->src_output()]) {
          --unused_outputs;
          outputs_required[e->src_output()] = true;
        }
      }
    }
    if (unused_outputs > 0) {
      for (int i = 0; i < n->num_outputs(); ++i) {
        if (!outputs_required[i]) {
          metrics::RecordUnusedOutput(n->type_string());
        }
      }
      item->outputs_required = std::move(outputs_required);
    }
  }

  // Rewrite each `EdgeInfo::input_slot` member to refer directly to the input
  // location.
  for (const Node* n : graph.nodes()) {
    if (IsSink(n)) continue;
    const int id = n->id();
    NodeItem* item = gview_.node(id);

    for (EdgeInfo& e : item->mutable_output_edges()) {
      const int dst_id = e.dst_id;
      NodeItem* dst_item = gview_.node(dst_id);
      e.input_slot += dst_item->input_start;
    }
  }

  // Initialize PendingCounts only after pending_ids_[node.id] is initialized
  // for all nodes.
  InitializePending(&graph, cf_info);
  return gview_.SetAllocAttrs(&graph, params_.device);
}

namespace {
// If a Node has been marked to use a ScopedAllocator x for output i, then
// sc_attr will contain the subsequence (i, x) at an even offset.  This function
// extracts and transfers that ScopedAllocator id to alloc_attr.  For now, we
// only allow one ScopedAllocator use per Node.
bool ExtractScopedAllocatorAttr(const std::vector<int>& sc_attr,
                                int output_index,
                                AllocatorAttributes* alloc_attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_5(mht_5_v, 457, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ExtractScopedAllocatorAttr");

  DCHECK_LE(2, sc_attr.size());
  for (int i = 0; i < sc_attr.size(); i += 2) {
    if (sc_attr[i] == output_index) {
      CHECK_EQ(alloc_attr->scope_id, 0);
      alloc_attr->scope_id = sc_attr[i + 1];
      return true;
    }
  }
  return false;
}
}  // namespace

Status ImmutableExecutorState::BuildControlFlowInfo(const Graph* g,
                                                    ControlFlowInfo* cf_info) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_6(mht_6_v, 474, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ImmutableExecutorState::BuildControlFlowInfo");

  const int num_nodes = g->num_node_ids();
  cf_info->frame_names.resize(num_nodes);
  std::vector<Node*> parent_nodes;
  parent_nodes.resize(num_nodes);
  std::vector<bool> visited;
  visited.resize(num_nodes);

  string frame_name;
  std::deque<Node*> ready;

  // Initialize with the root nodes.
  for (Node* n : g->nodes()) {
    if (n->in_edges().empty()) {
      visited[n->id()] = true;
      cf_info->unique_frame_names.insert(frame_name);
      ready.push_back(n);
    }
  }

  while (!ready.empty()) {
    Node* curr_node = ready.front();
    int curr_id = curr_node->id();
    ready.pop_front();

    Node* parent = nullptr;
    if (IsEnter(curr_node)) {
      // Enter a child frame.
      TF_RETURN_IF_ERROR(
          GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
      parent = curr_node;
    } else if (IsExit(curr_node)) {
      // Exit to the parent frame.
      parent = parent_nodes[curr_id];
      if (!parent) {
        return errors::InvalidArgument(
            "Invalid Exit op: Cannot find a corresponding Enter op.");
      }
      frame_name = cf_info->frame_names[parent->id()];
      parent = parent_nodes[parent->id()];
    } else {
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[curr_id];
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      Node* out = out_edge->dst();
      if (IsSink(out)) continue;
      const int out_id = out->id();

      // Add to ready queue if not visited.
      bool is_visited = visited[out_id];
      if (!is_visited) {
        ready.push_back(out);
        visited[out_id] = true;

        // Process the node 'out'.
        cf_info->frame_names[out_id] = frame_name;
        parent_nodes[out_id] = parent;
        cf_info->unique_frame_names.insert(frame_name);
      }
    }
  }

  return Status::OK();
}

void ImmutableExecutorState::InitializePending(const Graph* graph,
                                               const ControlFlowInfo& cf_info) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSimmutable_executor_stateDTcc mht_7(mht_7_v, 545, "", "./tensorflow/core/common_runtime/immutable_executor_state.cc", "ImmutableExecutorState::InitializePending");

  for (auto& it : cf_info.unique_frame_names) {
    FrameInfo* finfo = EnsureFrameInfo(it);
    DCHECK_EQ(finfo->pending_counts.get(), nullptr);
    finfo->pending_counts =
        absl::make_unique<PendingCounts>(finfo->pending_counts_layout);
  }

  if (!requires_control_flow_) {
    atomic_pending_counts_.reset(new std::atomic<int32>[gview_.num_nodes()]);
    std::fill(atomic_pending_counts_.get(),
              atomic_pending_counts_.get() + gview_.num_nodes(), 0);
  }

  for (const Node* n : graph->nodes()) {
    if (IsSink(n)) continue;
    const int id = n->id();
    const string& name = cf_info.frame_names[id];
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    auto& counts = EnsureFrameInfo(name)->pending_counts;
    counts->set_initial_count(pending_ids_[id], max_pending);
    if (!requires_control_flow_) {
      atomic_pending_counts_[id] = max_pending;
    }
  }
}
}  // namespace tensorflow
