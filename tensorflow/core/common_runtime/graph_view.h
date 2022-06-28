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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh() {
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


#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
class Graph;
class Node;
class OpKernel;
class Tensor;

// Represents a single data edge in a `NodeItem`.
struct EdgeInfo {
  // The node ID of the destination in the containing `GraphView`.
  int dst_id;
  // The index of the output that produces values on this edge.
  int output_slot : 31;
  // true if this is the last info for output_slot in the EdgeInfo list.
  bool is_last : 1;
  // The index of the input that consumes values on this edge.
  int input_slot;
};

// Represents a single control edge in a `NodeItem`.
struct ControlEdgeInfo {
  // The node ID of the destination in the containing `GraphView`.
  int dst_id;
};

// Compact structure representing a graph node and its associated kernel.
//
// Each NodeItem is an element of exactly one GraphView.
struct NodeItem {
  // The index of this node's item in its GraphView.
  int node_id = -1;

  // Cached attributes of this node for fast lookup.
  bool kernel_is_async : 1;     // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;            // True iff IsMerge(node)
  bool is_enter : 1;            // True iff IsEnter(node)
  bool is_constant_enter : 1;   // True iff IsEnter(node) and
                                // node->GetAttr("is_constant") == true.
  bool is_exit : 1;             // True iff IsExit(node)
  bool is_control_trigger : 1;  // True iff IsControlTrigger(node)
  bool is_source : 1;           // True iff IsSource(node)
  // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
  bool is_enter_exit_or_next_iter : 1;
  bool is_transfer_node : 1;      // True iff IsTransferNode(node)
  bool is_initialization_op : 1;  // True iff IsInitializationOp(node)
  bool is_recv_or_switch : 1;     // True iff IsRecv(node) || IsSwitch(node)
  bool is_next_iteration : 1;     // True iff IsNextIteration(node)
  bool is_noop : 1;  // True iff item->kernel->type_string_view() == "NoOp")
  bool
      is_any_consumer_merge_or_control_trigger : 1;  // True iff the destination
                                                     // of any output edge is a
                                                     // merge or control trigger
                                                     // node.
  bool is_any_input_ref_typed : 1;  // True iff any IsRefType(dt) for dt in this
                                    // node's input types.
  bool is_distributed_communication : 1;  // True iff the op is registered to
                                          // use distributed communication.

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  // If the kernel is a Const op, this containts points to the constant tensor.
  const Tensor* const_tensor = nullptr;

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // Number of output edges, excluding control edges.
  int32 num_output_edges;

  // Number of output control edges.
  int32 num_output_control_edges;

  // If non-null, contains an array of num_outputs bools, where the ith bool
  // is true if and only if the ith output is consumed by another node.
  std::unique_ptr<bool[]> outputs_required;

  gtl::MutableArraySlice<EdgeInfo> mutable_output_edges() {
    return gtl::MutableArraySlice<EdgeInfo>(output_edge_base(),
                                            num_output_edges);
  }

  gtl::ArraySlice<EdgeInfo> output_edges() const {
    return gtl::ArraySlice<EdgeInfo>(output_edge_base(), num_output_edges);
  }

  gtl::ArraySlice<ControlEdgeInfo> output_control_edges() const {
    return gtl::ArraySlice<const ControlEdgeInfo>(output_control_edge_base(),
                                                  num_output_control_edges);
  }

  DataType input_type(int i) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_0(mht_0_v, 297, "", "./tensorflow/core/common_runtime/graph_view.h", "input_type");

    DCHECK_LT(i, num_inputs);
    return static_cast<DataType>(input_type_base()[i]);
  }
  DataType output_type(int i) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_1(mht_1_v, 304, "", "./tensorflow/core/common_runtime/graph_view.h", "output_type");

    DCHECK_LT(i, num_outputs);
    return static_cast<DataType>(output_type_base()[i]);
  }

  // Return array of per-output allocator attributes.
  const AllocatorAttributes* output_attrs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_2(mht_2_v, 313, "", "./tensorflow/core/common_runtime/graph_view.h", "output_attrs");
 return output_attr_base(); }

  // Return array of expected input index from which each output should
  // be forwarded:
  // kNeverForward (-2) for DO NOT FORWARD (must allocate).
  // kNoReservation (-1) for no expected forwarding.
  // 0... for forward from that input.
  const int* forward_from() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_3(mht_3_v, 323, "", "./tensorflow/core/common_runtime/graph_view.h", "forward_from");
 return forward_from_base(); }

  string DebugString() const;

 private:
  friend class GraphView;

  NodeItem() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_4(mht_4_v, 333, "", "./tensorflow/core/common_runtime/graph_view.h", "NodeItem");
}

  // Variable length section starts immediately after *this
  // (uint8 is enough for DataType).
  //   EdgeInfo            out_edges[num_output_edges];
  //   ControlEdgeInfo     out_control_edges[num_output_control_edges];
  //   AllocatorAttributes output_attr[num_outputs];
  //   int                 forward_from[num_outputs];
  //   uint8               input_type[num_inputs];
  //   uint8               output_type[num_outputs];

  // Return pointer to variable length section.
  char* var() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_5(mht_5_v, 348, "", "./tensorflow/core/common_runtime/graph_view.h", "var");

    return const_cast<char*>(reinterpret_cast<const char*>(this) +
                             sizeof(NodeItem));
  }

  EdgeInfo* output_edge_base() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_6(mht_6_v, 356, "", "./tensorflow/core/common_runtime/graph_view.h", "output_edge_base");

    return reinterpret_cast<EdgeInfo*>(var());
  }

  ControlEdgeInfo* output_control_edge_base() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_7(mht_7_v, 363, "", "./tensorflow/core/common_runtime/graph_view.h", "output_control_edge_base");

    return reinterpret_cast<ControlEdgeInfo*>(var() + sizeof(EdgeInfo) *
                                                          num_output_edges);
  }

  AllocatorAttributes* output_attr_base() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_8(mht_8_v, 371, "", "./tensorflow/core/common_runtime/graph_view.h", "output_attr_base");

    return reinterpret_cast<AllocatorAttributes*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges);
  }
  int* forward_from_base() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_9(mht_9_v, 379, "", "./tensorflow/core/common_runtime/graph_view.h", "forward_from_base");

    return reinterpret_cast<int*>(var() + sizeof(EdgeInfo) * num_output_edges +
                                  sizeof(ControlEdgeInfo) *
                                      num_output_control_edges +
                                  sizeof(AllocatorAttributes) * num_outputs);
  }
  uint8* input_type_base() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_10(mht_10_v, 388, "", "./tensorflow/core/common_runtime/graph_view.h", "input_type_base");

    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs);
  }
  uint8* output_type_base() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_11(mht_11_v, 397, "", "./tensorflow/core/common_runtime/graph_view.h", "output_type_base");

    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(ControlEdgeInfo) * num_output_control_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs +
        sizeof(uint8) * num_inputs);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};

// Immutable view of a Graph organized for efficient execution.
//
// TODO(b/152651962): Add independent unit tests for this class.
class GraphView {
 public:
  GraphView() : space_(nullptr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_12(mht_12_v, 416, "", "./tensorflow/core/common_runtime/graph_view.h", "GraphView");
}
  ~GraphView();

  Status Initialize(const Graph* g);
  Status SetAllocAttrs(const Graph* g, const Device* device);
  void SetScopedAllocatorAttrs(const std::vector<const Node*>& sa_nodes);

  // Returns a mutable pointer to the `NodeItem` with the given `id` if it
  // exists in the graph, or `nullptr` if it does not.
  NodeItem* node(int32_t id) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_13(mht_13_v, 428, "", "./tensorflow/core/common_runtime/graph_view.h", "node");

    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    return ((offset == kuint32max)
                ? nullptr
                : reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));
  }

  // Returns the `NodeItem` with the given `id`.
  //
  // REQUIRES: `id` must be the ID of a valid node in the graph.
  const NodeItem& node_ref(int32_t id) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_14(mht_14_v, 443, "", "./tensorflow/core/common_runtime/graph_view.h", "node_ref");

    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    DCHECK_NE(offset, kuint32max);
    return *reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]);
  }

  int32 num_nodes() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_viewDTh mht_15(mht_15_v, 454, "", "./tensorflow/core/common_runtime/graph_view.h", "num_nodes");
 return num_nodes_; }

 private:
  char* InitializeNode(char* ptr, const Node* n);
  size_t NodeItemBytes(const Node* n);

  int32 num_nodes_ = 0;
  uint32* node_offsets_ = nullptr;  // array of size "num_nodes_"
  // node_offsets_[id] holds the byte offset for node w/ "id" in space_

  char* space_;  // NodeItem objects are allocated here

  TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_VIEW_H_
