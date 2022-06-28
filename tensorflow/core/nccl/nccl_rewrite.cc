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
class MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc {
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
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc() {
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

#include "tensorflow/core/lib/strings/str_util.h"
#if GOOGLE_CUDA

#include <forward_list>
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

// Replaces NcclReduce node with _NcclReduceRecv reusing one input of same
// device, adds one _NcclReduceSend for each other input.
Status ReplaceReduce(Graph* graph, Node* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "ReplaceReduce");

  string reduction;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "reduction", &reduction));
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
  int num_devices = node->num_inputs();
  string shared_name = node->name();
  auto make_builder = [&](StringPiece op_name, StringPiece suffix) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "lambda");

    return NodeBuilder(strings::StrCat(shared_name, suffix), op_name)
        .Attr("reduction", reduction)
        .Attr("num_devices", num_devices)
        .Attr("shared_name", shared_name)
        .Attr("T", dtype);
  };
  std::vector<Node*> control_inputs;
  for (const auto& edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      control_inputs.push_back(edge->src());
    }
  }
  std::vector<NodeBuilder::NodeOut> out_nodes;
  for (const auto& edge : node->out_edges()) {
    out_nodes.emplace_back(edge->dst(), edge->dst_input());
  }
  int recv_dev = node->assigned_device_name_index();
  NodeBuilder recv_builder =
      make_builder("_NcclReduceRecv", "Recv").ControlInputs(control_inputs);
  bool recv_input_set = false;
  int send_counter = 0;
  for (const auto& edge : node->in_edges()) {
    Node* src_node = edge->src();
    if (edge->IsControlEdge()) {
      continue;
    }
    int send_dev = src_node->assigned_device_name_index();
    if (!recv_input_set && send_dev == recv_dev) {
      recv_builder.Input(src_node);
      recv_input_set = true;
      continue;
    }
    auto send_builder = make_builder("_NcclReduceSend",
                                     strings::StrCat("Send_", ++send_counter))
                            .Input(src_node)
                            .ControlInputs(control_inputs);
    Node* send_node = nullptr;
    TF_RETURN_IF_ERROR(send_builder.Finalize(graph, &send_node));
    send_node->set_assigned_device_name_index(send_dev);
    // Send nodes don't have any outputs and therefore have no data dependencies
    // to the outputs of the graph. We add a control dependency to the receive
    // node so that those 'dangling' nodes are run.
    // TODO(b/67027412): Avoid these cross-device control edges.
    for (const auto& out_node : out_nodes) {
      graph->AddControlEdge(send_node, out_node.node);
    }
  }
  if (!recv_input_set) {
    return errors::InvalidArgument(
        "No input tensor uses the same device as the NcclReduce op");
  }
  Node* recv_node = nullptr;
  TF_RETURN_IF_ERROR(recv_builder.Finalize(graph, &recv_node));
  recv_node->set_assigned_device_name_index(recv_dev);
  graph->RemoveNode(node);
  for (const auto& out_node : out_nodes) {
    if (out_node.index == Graph::kControlSlot) {
      graph->AddControlEdge(recv_node, out_node.node);
    } else {
      graph->AddEdge(recv_node, 0, out_node.node, out_node.index);
    }
  }
  return Status::OK();
}

TensorProto TensorFromShape(const TensorShapeProto& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_2(mht_2_v, 279, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "TensorFromShape");

  TensorProto result;
  result.set_dtype(DT_INT32);
  for (const auto& dim : shape.dim()) {
    result.add_int_val(dim.size());
  }
  result.mutable_tensor_shape()->add_dim()->set_size(shape.dim_size());
  return result;
}

// Replaces NcclBroadcast node with _NcclBroadcastSend, connects the input to
// all outputs of same device, adds one _NcclBroadcastRecv for each other output
// device.
Status ReplaceBroadcast(Graph* graph, Node* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_3(mht_3_v, 295, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "ReplaceBroadcast");

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
  int send_dev = node->assigned_device_name_index();
  int num_devices = 0;  // Number of distinct devices, incremented below.
  std::vector<int> recv_index_map;  // Map device name index to stable index.

  // Map device name index to nodes that take the broadcast as input.
  std::vector<std::forward_list<NodeBuilder::NodeOut>> out_nodes_map;
  for (const auto& edge : node->out_edges()) {
    int dst_dev = edge->IsControlEdge()
                      ? send_dev
                      : edge->dst()->assigned_device_name_index();
    if (out_nodes_map.size() <= dst_dev) {
      out_nodes_map.resize(dst_dev + 1);
      recv_index_map.resize(dst_dev + 1);
    }
    auto it = out_nodes_map.begin() + dst_dev;
    if (it->empty()) {
      recv_index_map[dst_dev] = num_devices;
      ++num_devices;
    }
    it->emplace_front(NodeBuilder::NodeOut(edge->dst(), edge->dst_input()));
  }

  if (num_devices <= 1) {
    // Only one participating device, skip NCCL op.
    const Edge* in_edge = nullptr;
    TF_RETURN_IF_ERROR(node->input_edge(0, &in_edge));
    Node* in_node = in_edge->src();
    int in_index = in_edge->src_output();
    graph->RemoveNode(node);
    for (const auto& out_nodes : out_nodes_map) {
      for (const auto& out_node : out_nodes) {
        if (out_node.index == Graph::kControlSlot) {
          graph->AddControlEdge(in_node, out_node.node);
        } else {
          graph->AddEdge(in_node, in_index, out_node.node, out_node.index);
        }
      }
    }
    return Status::OK();
  }

  string shared_name = node->name();
  auto make_builder = [&](StringPiece op_name, StringPiece suffix) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_4(mht_4_v, 343, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "lambda");

    return NodeBuilder(strings::StrCat(shared_name, suffix), op_name)
        .Attr("num_devices", num_devices)
        .Attr("shared_name", shared_name)
        .Attr("T", dtype);
  };

  // Create broadcast send node and replace the original broadcast node.
  NodeBuilder::NodeOut in_node;
  NodeBuilder send_builder = make_builder("_NcclBroadcastSend", "Send");
  for (const auto& edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      send_builder.ControlInput(edge->src());
    } else {
      in_node = NodeBuilder::NodeOut(edge->src(), edge->src_output());
      send_builder.Input(in_node);
    }
  }
  Node* send_node = nullptr;
  TF_RETURN_IF_ERROR(send_builder.Finalize(graph, &send_node));
  send_node->set_assigned_device_name_index(send_dev);

  TensorShapeProto shape_proto;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape_proto));

  // Delete the original node before reconnecting to outputs.
  graph->RemoveNode(node);

  // Connect all outputs on the device of broadcast send.
  for (const auto& out_node : out_nodes_map[send_dev]) {
    if (out_node.index == Graph::kControlSlot) {
      graph->AddControlEdge(send_node, out_node.node);
    } else {
      graph->AddEdge(in_node.node, in_node.index, out_node.node,
                     out_node.index);
      // Add control edge so send node is run.
      graph->AddControlEdge(send_node, out_node.node);
    }
  }
  out_nodes_map[send_dev].clear();

  TensorProto tensor_proto = TensorFromShape(shape_proto);
  bool is_fully_defined = TensorShape(shape_proto).IsFullyDefined();
  string shape_name = strings::StrCat(in_node.node->name(), "/Shape");
  Node* shape_node = nullptr;
  if (!is_fully_defined) {
    NodeBuilder shape_builder(shape_name, "Shape");
    shape_builder.Input(in_node).Attr("out_type", DT_INT32).Attr("T", dtype);
    TF_RETURN_IF_ERROR(shape_builder.Finalize(graph, &shape_node));
    shape_node->set_assigned_device_name_index(send_dev);
  }

  // For all other devices, create a broadcast receive and connect outputs.
  for (int recv_dev = 0; recv_dev < out_nodes_map.size(); ++recv_dev) {
    if (out_nodes_map[recv_dev].empty()) {
      continue;
    }
    int recv_index = recv_index_map[recv_dev];
    if (is_fully_defined) {
      // If the shape is fully defined, define one const node per device.
      NodeBuilder shape_builder(strings::StrCat(shape_name, recv_index),
                                "Const");
      shape_builder.Attr("value", tensor_proto).Attr("dtype", DT_INT32);
      TF_RETURN_IF_ERROR(shape_builder.Finalize(graph, &shape_node));
      shape_node->set_assigned_device_name_index(recv_dev);
    }
    Node* recv_node;
    TF_RETURN_IF_ERROR(
        make_builder("_NcclBroadcastRecv", strings::StrCat("Recv_", recv_index))
            .Input(shape_node)
            .Finalize(graph, &recv_node));
    recv_node->set_assigned_device_name_index(recv_dev);
    for (const auto& out_node : out_nodes_map[recv_dev]) {
      graph->AddEdge(recv_node, 0, out_node.node, out_node.index);
    }
  }

  return Status::OK();
}

// Replaces occurrences of Nccl{Reduce, Broadcast}Input/Output with their
// _Nccl...Send/Recv counterparts and removes data dependencies between them.
class NcclReplacePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_rewriteDTcc mht_5(mht_5_v, 430, "", "./tensorflow/core/nccl/nccl_rewrite.cc", "Run");

    if (options.graph == nullptr) {
      return Status::OK();
    }
    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal(
          "NCCL replacement should happen before partitioning and a "
          "graph should be available.");
    }
    // Find reduction and broadcast ops and replace them with Send/Recv ops.
    for (Node* node : graph->op_nodes()) {
      StringPiece type = node->type_string();
      if (!absl::StartsWith(type, "Nccl")) {
        continue;
      }
      if (type == "NcclReduce") {
        TF_RETURN_IF_ERROR(ReplaceReduce(graph, node));
      }
      if (type == "NcclBroadcast") {
        TF_RETURN_IF_ERROR(ReplaceBroadcast(graph, node));
      }
    }
    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      NcclReplacePass);

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
