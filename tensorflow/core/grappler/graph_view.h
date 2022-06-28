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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh() {
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


#include <unordered_map>
#include <unordered_set>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// Map a node/op's input/output port_id to arg_id.
//
// The port_id refers to the n-th tensor of the node, while the arg_id refers to
// the n-th arg of the op. These two can be different if an op's arg is a list
// of tensors.
//
// We return -1 for any invalid port_id (i.e., no corresponding arg_id).
int OpOutputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id);
int OpInputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id);

namespace internal {

// GraphViewInternal is a helper class to simplify graph traversal. It creates
// an immutable view of the nodes and edges represented by a GraphDef protocol
// buffer.
//
// There are two public classes implementing GraphViewInternal:
//
// - GraphView: constructed from the `const GraphDef` and doesn't allow
//   to mutate underlying graph via input/output ports lookup functions (ports
//   have const pointers to nodes).
//
// - MutableGraphView: constructed from the 'GraphDef` and allows to mutate
//   the graph via input/output ports lookup functions (ports have non-const
//   pointers to nodes), and also have couple additional functions to
//   add/remove/replace nodes in the graph.
//
// --------------------------- !!! WARNING !!! ---------------------------------
//     Removing nodes from the graph outside of MutableGraphView will
//     lead to segfaults! Guaranteed by absl::string_view!
// -----------------------------------------------------------------------------
//
template <typename GraphDefT, typename NodeDefT>
class GraphViewInternal {
 public:
  struct Port {
    Port() : node(nullptr), port_id(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_0(mht_0_v, 241, "", "./tensorflow/core/grappler/graph_view.h", "Port");
}
    Port(NodeDefT* n, int port) : node(n), port_id(port) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_1(mht_1_v, 245, "", "./tensorflow/core/grappler/graph_view.h", "Port");
}

    bool operator==(const Port& other) const {
      return node == other.node && port_id == other.port_id;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Port& p) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/grappler/graph_view.h", "AbslHashValue");

      return H::combine(std::move(h), p.node, p.port_id);
    }

    NodeDefT* node;
    int port_id;
  };

  struct InputPort : public Port {
    using Port::Port;
  };

  struct OutputPort : public Port {
    using Port::Port;
  };

  struct Edge {
    Edge(OutputPort s, InputPort d) : src(s), dst(d) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_3(mht_3_v, 275, "", "./tensorflow/core/grappler/graph_view.h", "Edge");
}

    bool operator==(const Edge& other) const {
      return src == other.src && dst == other.dst;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Edge& e) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_4(mht_4_v, 285, "", "./tensorflow/core/grappler/graph_view.h", "AbslHashValue");

      return H::combine(std::move(h), e.src, e.dst);
    }

    OutputPort src;
    InputPort dst;
  };

  GraphDefT* graph() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_5(mht_5_v, 296, "", "./tensorflow/core/grappler/graph_view.h", "graph");
 return graph_; }

  // Finds a node by name or return `nullptr` if it's not in the graph view.
  NodeDefT* GetNode(absl::string_view node_name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_6(mht_6_v, 303, "", "./tensorflow/core/grappler/graph_view.h", "GetNode");

    return gtl::FindWithDefault(nodes_, node_name, nullptr);
  }

  // Checks if a node by name is in the graph view.
  bool HasNode(absl::string_view node_name) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_7(mht_7_v, 312, "", "./tensorflow/core/grappler/graph_view.h", "HasNode");

    return GetNode(node_name) != nullptr;
  }

  // Gets the specified input port. Note that the special '-1' port_id can be
  // used to access the controlling nodes (i.e. the nodes connected to node_name
  // through an incoming control dependency).
  InputPort GetInputPort(absl::string_view node_name, int port_id) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_8(mht_8_v, 323, "", "./tensorflow/core/grappler/graph_view.h", "GetInputPort");

    return InputPort(GetNode(node_name), port_id);
  }

  // Gets the specified output port. Note that the special '-1' port_id can be
  // used to access the controlled nodes (i.e. the nodes connected to node_name
  // through an outgoing control dependency).
  OutputPort GetOutputPort(absl::string_view node_name, int port_id) const {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_9(mht_9_v, 334, "", "./tensorflow/core/grappler/graph_view.h", "GetOutputPort");

    return OutputPort(GetNode(node_name), port_id);
  }

  // Gets the input port(s) in the immediate fanout of an output port.
  const absl::flat_hash_set<InputPort>& GetFanout(
      const OutputPort& port) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_10(mht_10_v, 343, "", "./tensorflow/core/grappler/graph_view.h", "GetFanout");

    return gtl::FindWithDefault(fanouts_, port, fanout_not_found_value_);
  }

  // Gets the output port(s) in the immediate fanin of an input port.
  absl::flat_hash_set<OutputPort> GetFanin(const InputPort& port) const {
    if (port.port_id >= 0) {
      OutputPort regular_fanin = GetRegularFanin(port);
      if (regular_fanin.node == nullptr) {
        return {};
      }
      return {regular_fanin};
    }

    // Collect fanin for the control input.
    absl::flat_hash_set<OutputPort> result;
    const int first_control_port =
        gtl::FindWithDefault(max_regular_input_port_, port.node, -1) + 1;
    for (int i = first_control_port; i < port.node->input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(port.node->input(i));

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) result.emplace(it->second, tensor_id.index());
    }
    return result;
  }

  // Special case: regular (i.e. non-control) input ports can only have one
  // fanin. If port.port_id is out of range or is a control dependency, then an
  // empty OutputPort is returned.
  const OutputPort GetRegularFanin(const InputPort& port) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_11(mht_11_v, 376, "", "./tensorflow/core/grappler/graph_view.h", "GetRegularFanin");

    if (port.port_id < 0 ||
        port.port_id >
            gtl::FindWithDefault(max_regular_input_port_, port.node, -1)) {
      return OutputPort();
    }

    TensorId tensor_id = ParseTensorName(port.node->input(port.port_id));
    return GetOutputPort(tensor_id.node(), tensor_id.index());
  }

  // Checks if a tensor id is a fanin of the node.
  bool HasFanin(const NodeDefT& node, const TensorId& fanin) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_12(mht_12_v, 391, "", "./tensorflow/core/grappler/graph_view.h", "HasFanin");

    int end = node.input_size();
    if (end == 0 || fanin.index() < -1) {
      return false;
    }

    const int num_regular_fanins =
        gtl::FindWithDefault(max_regular_input_port_, &node, -1) + 1;
    int start = 0;
    if (fanin.index() > -1) {
      end = num_regular_fanins;
    } else {
      start = num_regular_fanins;
    }
    for (int i = start; i < end; ++i) {
      if (ParseTensorName(node.input(i)) == fanin) {
        return true;
      }
    }
    return false;
  }

  // Gets all the input ports in the immediate fanout of a node. Include the
  // controlled nodes iff include_controlled_nodes is true.
  absl::flat_hash_set<InputPort> GetFanouts(
      const NodeDefT& node, bool include_controlled_nodes) const {
    absl::flat_hash_set<InputPort> result;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlled_nodes ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, &node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) {
        result.insert(it->second.begin(), it->second.end());
      }
    }
    return result;
  }

  // Gets all the output ports in the immediate fanin of a node. Include the
  // controlling nodes iff include_controlling_nodes is true.
  absl::flat_hash_set<OutputPort> GetFanins(
      const NodeDefT& node, bool include_controlling_nodes) const {
    absl::flat_hash_set<OutputPort> result;
    const int max_input_port =
        include_controlling_nodes
            ? node.input_size() - 1
            : gtl::FindWithDefault(max_regular_input_port_, &node, -1);
    for (int i = 0; i <= max_input_port; ++i) {
      TensorId tensor_id = ParseTensorName(node.input(i));

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) result.emplace(it->second, tensor_id.index());
    }
    return result;
  }

  // Gets the number of ports in the immediate fanin of a node. Count the
  // controlling nodes iff include_controlling_nodes is true.
  int NumFanins(const NodeDefT& node, bool include_controlling_nodes) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_13(mht_13_v, 458, "", "./tensorflow/core/grappler/graph_view.h", "NumFanins");

    if (include_controlling_nodes) {
      return node.input_size();
    }
    return gtl::FindWithDefault(max_regular_input_port_, &node, -1) + 1;
  }

  // Gets the number of ports in the immediate fanout of a node. Count the
  // controlled nodes iff include_controlled_nodes is true.
  int NumFanouts(const NodeDefT& node, bool include_controlled_nodes) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_14(mht_14_v, 470, "", "./tensorflow/core/grappler/graph_view.h", "NumFanouts");

    int count = 0;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlled_nodes ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, &node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) count += it->second.size();
    }

    return count;
  }

  // Gets all the edges in the immediate fanout of a node. Include the
  // controlled edges iff include_controlled_edges is true.
  absl::flat_hash_set<Edge> GetFanoutEdges(
      const NodeDefT& node, bool include_controlled_edges) const {
    absl::flat_hash_set<Edge> result;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlled_edges ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, &node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) {
        for (auto itr = it->second.begin(); itr != it->second.end(); ++itr) {
          result.emplace(/*src=*/port, /*dst=*/*itr);
        }
      }
    }
    return result;
  }

  // Gets all the edges in the immediate fanin of a node. Include the
  // controlling edges iff include_controlling_edges is true.
  absl::flat_hash_set<Edge> GetFaninEdges(
      const NodeDefT& node, bool include_controlling_edges) const {
    absl::flat_hash_set<Edge> result;
    const int max_input_port =
        include_controlling_edges
            ? node.input_size() - 1
            : gtl::FindWithDefault(max_regular_input_port_, &node, -1);
    for (int i = 0; i <= max_input_port; ++i) {
      TensorId tensor_id = ParseTensorName(node.input(i));

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) {
        result.emplace(/*src=*/OutputPort(it->second, tensor_id.index()),
                       /*dst=*/InputPort(const_cast<NodeDefT*>(&node), i));
      }
    }
    return result;
  }

 protected:
  explicit GraphViewInternal(GraphDefT* graph) : graph_(graph) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_15(mht_15_v, 537, "", "./tensorflow/core/grappler/graph_view.h", "GraphViewInternal");
}

  Status AddUniqueNode(NodeDefT* node) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_16(mht_16_v, 542, "", "./tensorflow/core/grappler/graph_view.h", "AddUniqueNode");

    auto inserted = nodes_.emplace(node->name(), node);
    return inserted.second
               ? Status::OK()
               : errors::InvalidArgument("Non unique node name detected: ",
                                         node->name());
  }

  // TODO(ezhulenev): Remove this function.
  void AddUniqueNodeOrDie(NodeDefT* node) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_17(mht_17_v, 554, "", "./tensorflow/core/grappler/graph_view.h", "AddUniqueNodeOrDie");

    Status st = AddUniqueNode(node);
    CHECK(st.ok()) << st.error_message();
  }

  // TODO(lyandy): Checks for self loops, Switch control dependencies, fanins
  // exist, and all regular fanins come before controlling fanins.
  void AddFanouts(NodeDefT* node) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_18(mht_18_v, 564, "", "./tensorflow/core/grappler/graph_view.h", "AddFanouts");

    int max_input_port = -1;
    for (int i = 0; i < node->input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node->input(i));
      OutputPort output(nodes_[tensor_id.node()], tensor_id.index());

      if (output.port_id < 0) {
        fanouts_[output].emplace(node, -1);
      } else {
        max_input_port = i;
        max_regular_output_port_[output.node] =
            std::max(max_regular_output_port_[output.node], output.port_id);
        fanouts_[output].emplace(node, i);
      }
    }
    if (max_input_port > -1) {
      max_regular_input_port_[node] = max_input_port;
    }
  }

  // Access to the mutable internal state for MutableGraphView.
  absl::flat_hash_map<absl::string_view, NodeDefT*>& nodes() { return nodes_; }

  absl::flat_hash_map<OutputPort, absl::flat_hash_set<InputPort>>& fanouts() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_19(mht_19_v, 590, "", "./tensorflow/core/grappler/graph_view.h", "fanouts");

    return fanouts_;
  }

  absl::flat_hash_map<const NodeDefT*, int>& max_regular_input_port() {
    return max_regular_input_port_;
  }

  absl::flat_hash_map<const NodeDefT*, int>& max_regular_output_port() {
    return max_regular_output_port_;
  }

 private:
  GraphDefT* graph_;  // must outlive the graph view

  // A mapping from the node name to the node itself.
  absl::flat_hash_map<absl::string_view, NodeDefT*> nodes_;

  // A mapping from the output port to all inputs that read from it.
  absl::flat_hash_map<OutputPort, absl::flat_hash_set<InputPort>> fanouts_;

  // Keep a maximum index of input tensors of the node.
  absl::flat_hash_map<const NodeDefT*, int> max_regular_input_port_;

  // Keep a maximum index of tensor fetched from the node. It doesn't guarantee
  // that all tensors in the [0, max_regular_output_port] range are actually
  // fetched by other nodes.
  absl::flat_hash_map<const NodeDefT*, int> max_regular_output_port_;

  // If the node has no fanouts at given output port (output tensor consumers)
  // we return a reference to this set from `GetFanout` (we can't construct new
  // empty set every time, because we need a non-dangling reference).
  absl::flat_hash_set<InputPort> fanout_not_found_value_;
};

}  // namespace internal

// Immutable GraphView that keeps the constness of the GraphDef. If you need to
// mutate the graph or the nodes via the graph view lookup functions, see
// MutableGraphView.
class GraphView
    : public internal::GraphViewInternal<const GraphDef, const NodeDef> {
 public:
  explicit GraphView(const GraphDef* graph) : GraphViewInternal(graph) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTh mht_20(mht_20_v, 636, "", "./tensorflow/core/grappler/graph_view.h", "GraphView");

    for (const NodeDef& node : graph->node()) AddUniqueNodeOrDie(&node);
    for (const NodeDef& node : graph->node()) AddFanouts(&node);
  }
};

// Returns true if node has one (or zero) fanout nodes at given output port.
bool HasSingleFanoutNode(const GraphView& graph_view, const NodeDef* node,
                         int port = 0);

// Returns true if node has at least one fanout node at given output port.
bool HasFanouts(const GraphView& graph_view, const NodeDef* node, int port = 0);
// Returns true if the node has at least one input control dependency.
bool HasControlFanin(const GraphView& graph_view, const NodeDef* node);
// Returns true if the node has at least one output control dependency.
bool HasControlFanout(const GraphView& graph_view, const NodeDef* node);
// Returns true if the node has at least one input or output control dependency.
bool HasControlFaninOrFanout(const GraphView& graph_view, const NodeDef* node);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
