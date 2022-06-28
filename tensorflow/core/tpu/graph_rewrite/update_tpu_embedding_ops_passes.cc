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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Configuration for TPU Embedding.

#include "tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.h"

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kTPUEmbeddingOps[] = {
    "EnqueueTPUEmbeddingBatch",
    "EnqueueTPUEmbeddingIntegerBatch",
    "EnqueueTPUEmbeddingSparseBatch",
    "EnqueueTPUEmbeddingSparseTensorBatch",
    "EnqueueTPUEmbeddingRaggedTensorBatch",
    "EnqueueTPUEmbeddingArbitraryTensorBatch"};

constexpr absl::string_view kTPURecvOps[] = {"RecvTPUEmbeddingActivations",
                                             "_RecvTPUEmbeddingActivations"};

constexpr absl::string_view kTPUGradientSendOps[] = {
  "SendTPUEmbeddingGradients", "_SendTPUEmbeddingGradients"};

}  // namespace

Status UpdateTPUEmbeddingEnqueueOrdinalPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingEnqueueOrdinalPass::Run");

  VLOG(1) << "UpdateTPUEmbeddingEnqueueOrdinalPass::Run";

  // Need the device set to get the number of devices per host.
  TF_RET_CHECK(options.device_set != nullptr);

  std::vector<Device*> tpu_devices;
  DeviceNameUtils::ParsedName tpu_device_spec;
  tpu_device_spec.has_type = true;
  tpu_device_spec.type = "TPU";
  options.device_set->FindMatchingDevices(tpu_device_spec, &tpu_devices);
  if (tpu_devices.empty()) {
    // If there are no TPUs don't run this pass.
    return Status::OK();
  }

  TF_RET_CHECK(options.graph != nullptr);
  Graph* graph = options.graph->get();

  std::vector<Node*> embedding_nodes;
  for (Node* node : graph->op_nodes()) {
    if (absl::c_linear_search(kTPUEmbeddingOps, node->type_string())) {
      embedding_nodes.emplace_back(node);
    }
  }

  // Only run if there are embedding nodes.
  if (embedding_nodes.empty()) {
    return Status::OK();
  }

  DeviceNameUtils::ParsedName single_tpu_device_spec =
      tpu_devices[0]->parsed_name();

  TF_RET_CHECK(single_tpu_device_spec.has_job);

  // Note that TPUEmbedding is only supported on system with a single TPU slice
  // (as determined by the 'job' portion of the device spec). Check for that
  // here just to be sure.
  for (const auto* tpu_device : tpu_devices) {
    TF_RET_CHECK(tpu_device->parsed_name().has_job);
    TF_RET_CHECK(tpu_device->parsed_name().job == single_tpu_device_spec.job)
        << "Multiple TPU jobs detected. This is not supported for now.";
  }

  std::vector<Device*> task_devices;
  single_tpu_device_spec.has_id = false;
  options.device_set->FindMatchingDevices(single_tpu_device_spec,
                                          &task_devices);
  int64 num_tpus_per_task = task_devices.size();

  for (Node* node : embedding_nodes) {
    int64 replica_id;
    if (TryGetNodeAttr(node->attrs(), kXlaReplicaIdAttrName, &replica_id)) {
      node->AddAttr("device_ordinal", replica_id % num_tpus_per_task);
    }
  }

  VLOG(1) << "UpdateTPUEmbeddingEnqueueOrdinalPass::Run() finished";
  return Status::OK();
}

template <typename A, typename N>
Status UpdateMapsForModeOverride(
    const std::string& op, const A& attrs, const N node_identifier,
    std::map<std::string, N>* enqueue_op,
    std::map<std::string, bool>* found_recv_op,
    std::map<std::string, bool>* found_grad_send_op) {
  string layer_call_index;
  if (TryGetNodeAttr(attrs, "_tpu_embedding_layer", &layer_call_index)) {
    if ((op == kTPURecvOps[0]) || (op == kTPURecvOps[1])) {
      // We will prevent users from creating multiple copies of the
      // TPUEmbedding layer so this should never happen.
      TF_RET_CHECK(!(*found_recv_op)[layer_call_index])
          << "Found second receive op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*found_recv_op)[layer_call_index] = true;
    } else if ((op == kTPUGradientSendOps[0]) ||
               (op == kTPUGradientSendOps[1])) {
      TF_RET_CHECK(!(*found_grad_send_op)[layer_call_index])
          << "Found second send op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*found_grad_send_op)[layer_call_index] = true;
    } else if (absl::c_linear_search(kTPUEmbeddingOps, op)) {
      TF_RET_CHECK(enqueue_op->find(layer_call_index) == enqueue_op->end())
          << "Found second enqueue op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*enqueue_op)[layer_call_index] = node_identifier;
    }
  }
  return Status::OK();
}

template <typename M, typename N>
Status ComputeEnqueueTrainingStatus(
    const std::map<std::string, N>& enqueue_op,
    const std::map<std::string, bool>& found_recv_op,
    const std::map<std::string, bool>& found_grad_send_op, M* enqueue) {
  TF_RET_CHECK(enqueue_op.size() == found_recv_op.size())
      << "Enqueue and recv ops should be in a one-to-one corresondence."
      << "Found " << enqueue_op.size() << " enqueue(s) and "
      << found_recv_op.size() << " receive(s).";
  for (const auto& node : enqueue_op) {
    TF_RET_CHECK(found_recv_op.find(node.first) != found_recv_op.end())
        << "No receive for enqueue call " << node.first;
    bool send_exists =
        (found_grad_send_op.find(node.first) != found_grad_send_op.end());
    VLOG(1) << "Found call " << node.first
        << (send_exists ? " with " : " without ") << " send op(s).";
    // If we have found a send gradient op for that is in the same cluster as
    // the enqueue op, then this is a training call so set the output to true
    // for this
    (*enqueue)[node.second] = send_exists;
  }
  return Status::OK();
}

// Get the enqueue ops and their status (training or eval) from a graph.
// enqueue is a map from a Graph Node* for an enqueue op to a bool which is true
// when the enqueue is part of a TPUEmbedding layer call that contains a send
// gradients.
Status UpdateTPUEmbeddingModePass::GetEnqueueOpsFromGraph(
    Graph* graph, absl::flat_hash_map<Node*, bool>* enqueue) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_1(mht_1_v, 349, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingModePass::GetEnqueueOpsFromGraph");

  // Maps are index by the TPUEmbedding layer's call number.
  std::map<std::string, Node*> enqueue_op;
  std::map<std::string, bool> found_recv_op;
  std::map<std::string, bool> found_grad_send_op;

  for (Node* node : graph->op_nodes()) {
    TF_RETURN_IF_ERROR(UpdateMapsForModeOverride(
        node->type_string(), node->attrs(), node, &enqueue_op, &found_recv_op,
        &found_grad_send_op));
    // Clear attribute so any further executions of this pass don't activate
    // pass.
    node->ClearAttr("_tpu_embedding_layer");
  }

  return ComputeEnqueueTrainingStatus(enqueue_op, found_recv_op,
                                      found_grad_send_op, enqueue);
}

// Update the graph for a specific enqueue op.
Status UpdateTPUEmbeddingModePass::UpdateGraphEnqueueOp(bool training,
                                                     Graph* graph,
                                                     Node* enqueue) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_2(mht_2_v, 374, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingModePass::UpdateGraphEnqueueOp");

  // When using the layer, the mode override input is a SelectV2 op (unless this
  // pass has already run), which takes a training and eval op as input. We will
  // simply short circut the SelectV2 and take input from the correct op.
  const Edge* select_edge;
  TF_RETURN_IF_ERROR(
      enqueue->input_edge(enqueue->num_inputs() - 1, &select_edge));
  if (select_edge->src()->type_string() == "SelectV2") {
    TF_RET_CHECK(select_edge->src()->num_inputs() == 3);
    Node* mode;
    TF_RETURN_IF_ERROR(select_edge->src()->input_node(training ? 1 : 2, &mode));
    graph->AddEdge(mode, 0, enqueue, enqueue->num_inputs() - 1);
    graph->RemoveEdge(select_edge);
  }

  return Status::OK();
}

// Get the enqueue ops and their status (training or eval) from a function def.
// The enqueue map is indexed by the position of the enqueue op in the
// function's node_def array.
Status UpdateTPUEmbeddingModePass::GetEnqueueOpsFromFunctionDef(
    FunctionDef* function, std::map<int, bool>* enqueue) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_3(mht_3_v, 399, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingModePass::GetEnqueueOpsFromFunctionDef");

  std::map<std::string, int> enqueue_op;
  std::map<std::string, bool> found_recv_op;
  std::map<std::string, bool> found_grad_send_op;

  std::string cluster;
  for (int i = 0; i < function->node_def_size(); ++i) {
    const NodeDef& node = function->node_def(i);
    TF_RETURN_IF_ERROR(UpdateMapsForModeOverride(
        node.op(), node, i, &enqueue_op, &found_recv_op, &found_grad_send_op));
    // Clear attribute so any further executions of this pass don't activate
    // pass.
    function->mutable_node_def(i)->mutable_attr()->erase(
        "_tpu_embedding_layer");
  }

  return ComputeEnqueueTrainingStatus(enqueue_op, found_recv_op,
                                      found_grad_send_op, enqueue);
}

// Update the function def for a specific enqueue op.
Status UpdateTPUEmbeddingModePass::UpdateFunctionDefEnqueueOp(
    int enqueue, bool training, FunctionDef* function, bool* updated) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_4(mht_4_v, 424, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingModePass::UpdateFunctionDefEnqueueOp");

  // When using the layer, the mode override input is a SelectV2 op,
  // which takes a training and eval op as input. We will simply short circut
  // the SelectV2 and take input from the correct op.
  NodeDef* node = function->mutable_node_def(enqueue);
  int mode_override = node->input_size() - 1;
  while ((mode_override >= 0) && (node->input(mode_override).empty() ||
                                  (node->input(mode_override)[0] == '^'))) {
    mode_override--;
  }
  TF_RET_CHECK(mode_override >= 0) << "Can't find non-control input to "
                                   << "enqueue.";
  TF_RET_CHECK(!node->input(mode_override).empty());

  // Find input node
  string select_name = std::vector<std::string>(
      absl::StrSplit(node->input(mode_override), ':'))[0];
  int select = 0;
  while ((select < function->node_def_size()) &&
         (function->node_def(select).name() != select_name)) {
    select++;
  }
  TF_RET_CHECK(select < function->node_def_size())
      << "Unable to find enqueue input node " << select_name << " in function "
      << function->signature().name();
  if (function->node_def(select).op() == "SelectV2") {
    // Make the mode override input the same as the correct input of the
    // select v2.
    (*node->mutable_input(mode_override)) =
        function->node_def(select).input(training ? 1 : 2);
    *updated = true;
  }

  return Status::OK();
}

Status UpdateTPUEmbeddingModePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSupdate_tpu_embedding_ops_passesDTcc mht_5(mht_5_v, 464, "", "./tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.cc", "UpdateTPUEmbeddingModePass::Run");

  // Updates the Enqueue ops when using a layer to set the mode override
  // behavior depending on the existence of send gradients ops.
  // Note we only do this when a layer is used (all BC ops with an integer
  // attribute "_tpu_embedding_layer" that is incremented per call, so we can
  // easily associate the various ops).
  //
  // Note that the BC ops can be in the Graph or in the FunctionDef.
  // If they are in the graph at stage 0, this means that there as no control
  // flow containing them (i.e. a host loop). In this case, we group together
  // ops with the same "_tpu_embedding_layer" tag.
  //
  // We also search all FunctionDefs. Note that as the ops are all created in
  // the layer's call, a cluster of TPUEmbedding ops won't be split across
  // different FunctionDefs.

  VLOG(1) << "UpdateTPUEmbeddingModePass::Run";

  TF_RET_CHECK(options.graph != nullptr);

  // First process the graph
  Graph* graph = options.graph->get();
  absl::flat_hash_map<Node*, bool> enqueue_nodes;
  TF_RETURN_IF_ERROR(GetEnqueueOpsFromGraph(graph, &enqueue_nodes));
  for (const auto& enqueue : enqueue_nodes) {
    TF_RETURN_IF_ERROR(
        UpdateGraphEnqueueOp(enqueue.second, graph, enqueue.first));
  }

  for (const auto& fname : options.flib_def->ListFunctionNames()) {
    FunctionDef fdef_copy(*options.flib_def->Find(fname));
    std::map<int, bool> enqueue_nodes;
    TF_RETURN_IF_ERROR(
        GetEnqueueOpsFromFunctionDef(&fdef_copy, &enqueue_nodes));
    bool updated = false;
    for (const auto& enqueue : enqueue_nodes) {
      TF_RETURN_IF_ERROR(UpdateFunctionDefEnqueueOp(
          enqueue.first, enqueue.second, &fdef_copy, &updated));
    }

    if (updated) {
      TF_RETURN_IF_ERROR(options.flib_def->ReplaceFunction(fname, fdef_copy));
    }
  }

  VLOG(1) << "UpdateTPUEmbeddingModePass::Run() finished";
  return Status::OK();
}

}  // namespace tensorflow
