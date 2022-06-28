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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc() {
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
#include "tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.h"

#include <iterator>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/graph_rewrite/tpu_embedding_rewrite_pass_utils.h"
#include "tensorflow/core/tpu/tpu_embedding_configuration_utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

// Check the number of outputs for RecvActivationsNode or for number of inputs
// For SendGradientsNode.
xla::Status CheckNumInputsOrOutputs(
    const int32 num_input_or_outputs, const std::string& attribute_name,
    const std::string& node_name,
    const tpu::TPUEmbeddingConfiguration& tpu_embedding_config) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("attribute_name: \"" + attribute_name + "\"");
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "CheckNumInputsOrOutputs");

  if (tpu_embedding_config.feature_descriptor_size() == 0 &&
      num_input_or_outputs != tpu_embedding_config.table_descriptor_size()) {
    return errors::InvalidArgument(absl::StrFormat(
        "Number of tables in the TPU embedding config: %d does not match the "
        "%s attribute: %d in the %s node.",
        tpu_embedding_config.table_descriptor_size(), attribute_name,
        num_input_or_outputs, node_name));
  }

  if (tpu_embedding_config.feature_descriptor_size() > 0 &&
      num_input_or_outputs != tpu_embedding_config.feature_descriptor_size()) {
    return errors::InvalidArgument(absl::StrFormat(
        "Feature descriptor is set in tpu embedding config. But number of "
        "features in the TPU embedding config: %d does not match the "
        "%s attribute: %d in the %s node.",
        tpu_embedding_config.feature_descriptor_size(), attribute_name,
        num_input_or_outputs, node_name));
  }
  return Status::OK();
}

// Constructs a NodeDef proto for the _RecvTPUEmbeddingDeduplicationData node to
// be added to the graph or function def.
xla::StatusOr<NodeDef> MakeRecvDeduplicationDataNodeDef(
    absl::string_view device_name, absl::string_view tpu_replicate_attr,
    absl::string_view tpu_embedding_config_str,
    absl::Span<const std::string> control_inputs) {
  DeviceNameUtils::ParsedName parsed_name;
  TF_RET_CHECK(DeviceNameUtils::ParseFullName(device_name, &parsed_name));

  NodeDefBuilder builder(
      absl::StrFormat("RecvTPUEmbeddingDeduplicationData_%s_%s_%d",
                      tpu_replicate_attr, parsed_name.type, parsed_name.id),
      "_RecvTPUEmbeddingDeduplicationData");
  if (!device_name.empty()) {
    builder.Device(device_name);
  }
  if (!tpu_replicate_attr.empty()) {
    builder.Attr("_tpu_replicate", tpu_replicate_attr);
  }
  builder.Attr("config", tpu_embedding_config_str);
  for (const std::string& control_input : control_inputs) {
    builder.ControlInput(control_input);
  }

  NodeDef deduplication_data_node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&deduplication_data_node_def));
  VLOG(1) << "Created new _RecvTPUEmbeddingDeduplicationData node def: "
          << deduplication_data_node_def.DebugString();
  return deduplication_data_node_def;
}

// Constructs a NodeDef proto for the _RecvTPUEmbeddingActivations node to be
// added to the graph or function def.
xla::StatusOr<NodeDef> MakeRecvActivationsNodeDef(
    const NodeDef& old_activations_node_def,
    absl::string_view deduplication_data_node_name,
    absl::string_view device_name, absl::string_view tpu_replicate_attr,
    absl::string_view tpu_embedding_config_str,
    absl::Span<const NodeDefBuilder::NodeOut> data_inputs,
    absl::Span<const std::string> control_inputs) {
  if (!data_inputs.empty()) {
    return errors::InvalidArgument(
        absl::StrFormat("Expected to have zero inputs for "
                        "RecvTPUEmbeddingActivations node, found %d inputs.",
                        data_inputs.size()));
  }

  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  if (!tpu_embedding_config.ParseFromString(
          std::string(tpu_embedding_config_str))) {  // NOLINT
    return errors::InvalidArgument(
        "Malformed config attribute in the RecvTPUEmbeddingActivations node.");
  }

  int32 num_outputs;
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(old_activations_node_def),
                                 "num_outputs", &num_outputs));

  TF_RETURN_IF_ERROR(CheckNumInputsOrOutputs(num_outputs, "num_outputs",
                                             "RecvTPUEmbeddingActivations",
                                             tpu_embedding_config));

  NodeDefBuilder builder(old_activations_node_def.name(),
                         "_RecvTPUEmbeddingActivations");
  if (!device_name.empty()) {
    builder.Device(device_name);
  }
  // The num tables here can be interpreted as num features if the feature
  // descriptor is present in the config.
  builder.Attr("num_tables", num_outputs)
      .Attr("config", tpu_embedding_config_str);
  if (!tpu_replicate_attr.empty()) {
    builder.Attr("_tpu_replicate", tpu_replicate_attr);
  }
  std::string embedding_layer;
  if (TryGetNodeAttr(AttrSlice(old_activations_node_def),
                     "_tpu_embedding_layer", &embedding_layer)) {
    builder.Attr("_tpu_embedding_layer", embedding_layer);
  }

  builder.Input(absl::StrCat(deduplication_data_node_name, ":output"),
                /*src_index=*/0, DT_VARIANT);
  for (const std::string& control_input : control_inputs) {
    builder.ControlInput(control_input);
  }

  NodeDef activations_node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&activations_node_def));
  *activations_node_def.mutable_experimental_debug_info() =
      old_activations_node_def.experimental_debug_info();
  VLOG(1) << "Created new _RecvTPUEmbeddingActivations node def: "
          << activations_node_def.DebugString();
  return activations_node_def;
}

// Constructs a NodeDef proto for the _RecvTPUEmbeddingDeduplicationData node to
// be added to the graph or function def.
xla::StatusOr<NodeDef> MakeSendGradientsNodeDef(
    const NodeDef& old_gradients_node_def,
    absl::string_view deduplication_data_node_name,
    absl::string_view device_name, absl::string_view tpu_replicate_attr,
    absl::string_view tpu_embedding_config_str,
    absl::Span<const NodeDefBuilder::NodeOut> data_inputs,
    absl::Span<const std::string> control_inputs) {
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  if (!tpu_embedding_config.ParseFromString(
          std::string(tpu_embedding_config_str))) {  // NOLINT
    return errors::InvalidArgument(
        "Malformed config attribute in the SendTPUEmbeddingGradients node.");
  }

  int32 num_inputs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(old_gradients_node_def), "N", &num_inputs));

  TF_RETURN_IF_ERROR(CheckNumInputsOrOutputs(num_inputs, "num_inputs",
                                             "SendTPUEmbeddingGradients",
                                             tpu_embedding_config));

  int32 learning_rate_tag_count = 0;
  if (!GetNodeAttr(AttrSlice(old_gradients_node_def), "NN",
                   &learning_rate_tag_count)
           .ok()) {
    LOG(INFO)
        << "Missing the NN attribute (number of dynamic learning rate tags) in "
           "the SendTPUEmbeddingGradients node. Setting the value to 0.";
  }

  auto status_or_lr_tag_count =
      tpu::ComputeTotalTagCountForDynamicLearningRates(tpu_embedding_config);
  if (!status_or_lr_tag_count.ok()) {
    return errors::InvalidArgument(status_or_lr_tag_count.status().message());
  }

  const int32 expected_learning_rate_tag_count = status_or_lr_tag_count.value();

  if (learning_rate_tag_count != expected_learning_rate_tag_count) {
    return errors::InvalidArgument(absl::StrFormat(
        "Number of dynamic learning rate tags in the TPU embedding config: %d "
        "does not match the NN attribute: %d in the SendTPUEmbeddingGradients "
        "node.",
        expected_learning_rate_tag_count, learning_rate_tag_count));
  }

  if (data_inputs.size() !=
      static_cast<uint64>(num_inputs + learning_rate_tag_count)) {
    return errors::InvalidArgument(absl::StrFormat(
        "Mismatch in the number of inputs for SendTPUEmbeddingGradients node, "
        "expected: %d, actual: %d",
        num_inputs + learning_rate_tag_count, data_inputs.size()));
  }

  NodeDefBuilder builder(old_gradients_node_def.name(),
                         "_SendTPUEmbeddingGradients");
  if (!device_name.empty()) {
    builder.Device(device_name);
  }

  // The Numtables here can be interpreted as num features if the feature
  //  descriptor is present in the config.
  builder.Attr("NumTables", num_inputs)
      .Attr("NumLearningRateTags", learning_rate_tag_count)
      .Attr("config", tpu_embedding_config_str);
  if (!tpu_replicate_attr.empty()) {
    builder.Attr("_tpu_replicate", tpu_replicate_attr);
  }
  std::string embedding_layer;
  if (TryGetNodeAttr(AttrSlice(old_gradients_node_def), "_tpu_embedding_layer",
                     &embedding_layer)) {
    builder.Attr("_tpu_embedding_layer", embedding_layer);
  }

  builder.Input(absl::MakeConstSpan(data_inputs.data(), num_inputs))
      .Input(absl::MakeConstSpan(data_inputs.data() + num_inputs,
                                 learning_rate_tag_count))
      .Input(absl::StrCat(deduplication_data_node_name, ":output"),
             /*src_index=*/0, DT_VARIANT);
  for (const std::string& control_input : control_inputs) {
    builder.ControlInput(control_input);
  }

  NodeDef gradients_node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&gradients_node_def));
  *gradients_node_def.mutable_experimental_debug_info() =
      old_gradients_node_def.experimental_debug_info();
  VLOG(1) << "Created new _SendTPUEmbeddingGradients node def: "
          << gradients_node_def.DebugString();
  return gradients_node_def;
}

// Key for the map that holds the RecvTPUEmbeddingActivations /
// SendTPUEmbeddingGradients nodes in the graph before they are rewritten.
struct SendRecvNodesMapKey {
  // _tpu_replicate attribute from the NodeDef.
  std::string tpu_replicate_attr;

  // Device name from the NodeDef.
  std::string requested_device;

  template <typename H>
  friend H AbslHashValue(H h, const SendRecvNodesMapKey& s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_1(mht_1_v, 439, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "AbslHashValue");

    return H::combine(std::move(h), s.tpu_replicate_attr,
                      s.requested_device);
  }

  const inline bool operator==(const SendRecvNodesMapKey& s) const {
    return (tpu_replicate_attr == s.tpu_replicate_attr &&
            requested_device == s.requested_device);
  }
};

// Pointers to the original RecvTPUEmbeddingActivations and
// SendTPUEmbeddingGradients nodes in the graph. If a node is not present in the
// graph, the pointer is set to nullptr.
struct SendRecvNodes {
  Node* activations_node = nullptr;
  Node* gradients_node = nullptr;
};

// Maps a _tpu_replicate attribute to the corresponding
// RecvTPUEmbeddingActivations and SendTPUEmbeddingGradients nodes in the graph.
using SendRecvNodesMap =
    absl::flat_hash_map<SendRecvNodesMapKey, SendRecvNodes>;

// Gets the src nodes for the incoming control edges of the node.
std::vector<Node*> GetControlInputNodes(const Node* node) {
  std::vector<Node*> control_inputs;
  for (const Edge* edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      control_inputs.push_back(edge->src());
    }
  }
  return control_inputs;
}

// Gets the src node names and output indices for the incoming data edges of
// node.
std::vector<NodeDefBuilder::NodeOut> GetDataInputs(const Node* node,
                                                   DataType dt) {
  std::vector<NodeDefBuilder::NodeOut> data_inputs;
  for (const Edge* edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      data_inputs.emplace_back(edge->src()->name(), edge->src_output(), dt);
    }
  }
  return data_inputs;
}

// Gets the TPUEmbeddingConfiguration proto, assigned device name, and index
// from the graph nodes (activations_node and gradients_node). If both nodes are
// present, ensure that the TPUEmbeddingConfiguration proto, assigned device
// name, and index are the same on both nodes.
Status ValidateAndGetTPUEmbeddingConfiguration(
    const Node* activations_node, const Node* gradients_node,
    absl::string_view tpu_replicate_attr,
    std::string* tpu_embedding_config_str) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tpu_replicate_attr: \"" + std::string(tpu_replicate_attr.data(), tpu_replicate_attr.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_2(mht_2_v, 498, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "ValidateAndGetTPUEmbeddingConfiguration");

  if (activations_node != nullptr && gradients_node != nullptr) {
    std::string activations_config_str;
    std::string gradients_config_str;
    TF_RETURN_IF_ERROR(GetNodeAttr(activations_node->def(), "config",
                                   &activations_config_str));
    TF_RETURN_IF_ERROR(
        GetNodeAttr(gradients_node->def(), "config", &gradients_config_str));

    if (activations_config_str != gradients_config_str) {
      return errors::InvalidArgument(absl::StrFormat(
          "TPU embedding config attributes of RecvTPUEmbeddingActivations and "
          "SendTPUEmbeddingGradients nodes with the same tpu_replicate attr: "
          "%s are not identical.",
          tpu_replicate_attr));
    }
    if (activations_node->assigned_device_name() !=
        gradients_node->assigned_device_name()) {
      return errors::InvalidArgument(absl::StrFormat(
          "Mismatch in assigned device names for the "
          "RecvTPUEmbeddingActivations (%s) and SendTPUEmbeddingGradients (%s) "
          "nodes with the same tpu_replicate attr: %s.",
          activations_node->assigned_device_name(),
          gradients_node->assigned_device_name(), tpu_replicate_attr));
    }
    if (activations_node->assigned_device_name_index() !=
        gradients_node->assigned_device_name_index()) {
      return errors::InvalidArgument(absl::StrFormat(
          "Mismatch in assigned device name indices for the "
          "RecvTPUEmbeddingActivations (%d) and SendTPUEmbeddingGradients (%d) "
          "nodes with the same tpu_replicate attr: %s.",
          activations_node->assigned_device_name_index(),
          gradients_node->assigned_device_name_index(), tpu_replicate_attr));
    }
  }

  if (activations_node == nullptr && gradients_node == nullptr) {
    return errors::Internal(absl::StrFormat(
        "Found tpu_replicate attr: %s with no corresponding "
        "RecvTPUEmbeddingActivations or SendTPUEmbeddingGradients nodes",
        tpu_replicate_attr));
  }

  const Node* compile_node =
      (activations_node != nullptr) ? activations_node : gradients_node;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(compile_node->def(), "config", tpu_embedding_config_str));

  return Status::OK();
}

// Adds a _RecvTPUEmbeddingDeduplicationNode to the graph assigning to the
// specified device and setting its attributes to tpu_replicate_attr and
// tpu_embedding_config_str. Control inputs for the old_activations_node
// (Op=RecvTPUEmbeddingActivations) and the old_gradients_node
// (Op=SendTPUEmbeddingGradients) are copied over to the newly inserted node to
// ensure that it has the same control frame.
Status AddRecvDeduplicationDataNode(const Node* old_activations_node,
                                    const Node* old_gradients_node,
                                    const std::string& requested_device,
                                    absl::string_view tpu_replicate_attr,
                                    absl::string_view tpu_embedding_config_str,
                                    Node** deduplication_data_node,
                                    Graph* graph) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("requested_device: \"" + requested_device + "\"");
   mht_3_v.push_back("tpu_replicate_attr: \"" + std::string(tpu_replicate_attr.data(), tpu_replicate_attr.size()) + "\"");
   mht_3_v.push_back("tpu_embedding_config_str: \"" + std::string(tpu_embedding_config_str.data(), tpu_embedding_config_str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_3(mht_3_v, 567, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "AddRecvDeduplicationDataNode");

  // Note that control inputs added later while constructing the Node are copied
  // over automatically to the NodeDef, so we don't need to specify any control
  // inputs here.
  TF_ASSIGN_OR_RETURN(const NodeDef deduplication_data_node_def,
                      MakeRecvDeduplicationDataNodeDef(
                          requested_device, tpu_replicate_attr,
                          tpu_embedding_config_str, /*control_inputs=*/{}));

  TF_RETURN_IF_ERROR(
      AddNode(deduplication_data_node_def, deduplication_data_node, graph));

  // Incoming control edges to the old_activations_node and old_gradients_node
  // need to be added to the newly created node to ensure that it has the same
  // control frame.
  std::vector<Node*> control_inputs;
  if (old_activations_node != nullptr) {
    absl::c_copy(GetControlInputNodes(old_activations_node),
                 std::back_inserter(control_inputs));
  }
  if (old_gradients_node != nullptr) {
    absl::c_copy(GetControlInputNodes(old_gradients_node),
                 std::back_inserter(control_inputs));
  }
  for (Node* control_input : control_inputs) {
    graph->AddControlEdge(control_input, *deduplication_data_node);
  }

  const Node* compile_node = (old_activations_node != nullptr)
                                 ? old_activations_node
                                 : old_gradients_node;
  (*deduplication_data_node)
      ->set_assigned_device_name(compile_node->assigned_device_name());
  (*deduplication_data_node)
      ->set_assigned_device_name_index(
          compile_node->assigned_device_name_index());

  VLOG(1) << "Inserted RecvDeduplicationData node: "
          << (*deduplication_data_node)->DebugString();
  return Status::OK();
}

// Replaces the old_activations_node (Op=RecvTPUEmbeddingActivations) with a new
// node (Op=_RecvTPUEmbeddingActivations) and initializes it with the specified
// tpu_replicate and tpu_embedding_config_str attributes. Connects the output of
// the deduplication_data_node to the input of the newly added node.
Status ReplaceRecvActivationsNodeAndAddDeduplicationInputs(
    absl::string_view tpu_replicate_attr,
    absl::string_view tpu_embedding_config_str, Node* old_activations_node,
    Node* deduplication_data_node, Graph* graph) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tpu_replicate_attr: \"" + std::string(tpu_replicate_attr.data(), tpu_replicate_attr.size()) + "\"");
   mht_4_v.push_back("tpu_embedding_config_str: \"" + std::string(tpu_embedding_config_str.data(), tpu_embedding_config_str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_4(mht_4_v, 621, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "ReplaceRecvActivationsNodeAndAddDeduplicationInputs");

  VLOG(1) << "Removing old RecvTPUEmbeddingActivations node: "
          << old_activations_node->DebugString();
  VLOG(1) << "Old RecvTPUEmbeddingActivations node def: "
          << old_activations_node->def().DebugString();

  // Note that control inputs added later while constructing the Node are copied
  // over automatically to the NodeDef, so we don't need to specify any control
  // inputs here.
  TF_ASSIGN_OR_RETURN(
      const NodeDef activations_node_def,
      MakeRecvActivationsNodeDef(old_activations_node->def(),
                                 deduplication_data_node->name(),
                                 old_activations_node->requested_device(),
                                 tpu_replicate_attr, tpu_embedding_config_str,
                                 GetDataInputs(old_activations_node, DT_FLOAT),
                                 /*control_inputs=*/{}));
  VLOG(1) << "Created new RecvTPUEmbeddingActivations node def: "
          << activations_node_def.ShortDebugString();

  Node* activations_node;
  TF_RETURN_IF_ERROR(ReplaceNode(activations_node_def, old_activations_node,
                                 &activations_node, graph));
  graph->AddEdge(deduplication_data_node, 0, activations_node, 0);

  activations_node->set_assigned_device_name(
      old_activations_node->assigned_device_name());
  activations_node->set_assigned_device_name_index(
      old_activations_node->assigned_device_name_index());
  VLOG(1) << "Inserted new RecvTPUEmbeddingActivations node: "
          << activations_node->DebugString();

  return Status::OK();
}

// Replaces the old_gradients_node (Op=SendTPUEmbeddingGradients) with a new
// node (Op=_SendTPUEmbeddingGradients) and initializes it with the specified
// tpu_replicate and tpu_embedding_config_str attributes. Connects the output of
// the deduplication_data_node to the last input of the newly added node.
Status ReplaceSendGradientsNodeAndAddDeduplicationInputs(
    absl::string_view tpu_replicate_attr,
    absl::string_view tpu_embedding_config_str, Node* old_gradients_node,
    Node* deduplication_data_node, Graph* graph) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tpu_replicate_attr: \"" + std::string(tpu_replicate_attr.data(), tpu_replicate_attr.size()) + "\"");
   mht_5_v.push_back("tpu_embedding_config_str: \"" + std::string(tpu_embedding_config_str.data(), tpu_embedding_config_str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_5(mht_5_v, 668, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "ReplaceSendGradientsNodeAndAddDeduplicationInputs");

  VLOG(1) << "Removing old SendTPUEmbeddingGradients node: "
          << old_gradients_node->DebugString();
  VLOG(1) << "Old SendTPUEmbeddingGradients node def: "
          << old_gradients_node->def().DebugString();

  const std::vector<NodeDefBuilder::NodeOut> data_inputs =
      GetDataInputs(old_gradients_node, DT_FLOAT);
  // Note that control inputs added later while constructing the Node are copied
  // over automatically to the NodeDef, so we don't need to specify any control
  // inputs here.
  TF_ASSIGN_OR_RETURN(
      const NodeDef gradients_node_def,
      MakeSendGradientsNodeDef(
          old_gradients_node->def(), deduplication_data_node->name(),
          old_gradients_node->requested_device(), tpu_replicate_attr,
          tpu_embedding_config_str, data_inputs, /*control_inputs=*/{}));

  VLOG(1) << "Created new SendTPUEmbeddingGradients node def: "
          << gradients_node_def.ShortDebugString();

  Node* gradients_node;
  TF_RETURN_IF_ERROR(ReplaceNode(gradients_node_def, old_gradients_node,
                                 &gradients_node, graph));
  graph->AddEdge(deduplication_data_node, 0, gradients_node,
                 data_inputs.size());

  gradients_node->set_assigned_device_name(
      old_gradients_node->assigned_device_name());
  gradients_node->set_assigned_device_name_index(
      old_gradients_node->assigned_device_name_index());
  VLOG(1) << "Inserted new SendTPUEmbeddingGradients node: "
          << gradients_node->DebugString();

  return Status::OK();
}

// Rewrites the graph for a particular _tpu_replicate attribute.
Status RewriteGraphForTpuReplicateAttrAndDevice(
    absl::string_view tpu_replicate_attr, const std::string& requested_device,
    Node* old_activations_node, Node* old_gradients_node, Graph* graph) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("tpu_replicate_attr: \"" + std::string(tpu_replicate_attr.data(), tpu_replicate_attr.size()) + "\"");
   mht_6_v.push_back("requested_device: \"" + requested_device + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_6(mht_6_v, 713, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "RewriteGraphForTpuReplicateAttrAndDevice");

  VLOG(1) << "Rewriting graph for _tpu_replicate attribute: "
          << tpu_replicate_attr << ", activations node: "
          << ((old_activations_node) ? old_activations_node->DebugString()
                                     : "NULL")
          << ", gradients node: "
          << ((old_gradients_node) ? old_gradients_node->DebugString()
                                   : "NULL");
  std::string tpu_embedding_config_str;
  TF_RETURN_IF_ERROR(ValidateAndGetTPUEmbeddingConfiguration(
      old_activations_node, old_gradients_node, tpu_replicate_attr,
      &tpu_embedding_config_str));

  Node* deduplication_data_node;
  TF_RETURN_IF_ERROR(AddRecvDeduplicationDataNode(
      old_activations_node, old_gradients_node, requested_device,
      tpu_replicate_attr, tpu_embedding_config_str, &deduplication_data_node,
      graph));

  if (old_activations_node != nullptr) {
    TF_RETURN_IF_ERROR(ReplaceRecvActivationsNodeAndAddDeduplicationInputs(
        tpu_replicate_attr, tpu_embedding_config_str, old_activations_node,
        deduplication_data_node, graph));
  }
  if (old_gradients_node != nullptr) {
    TF_RETURN_IF_ERROR(ReplaceSendGradientsNodeAndAddDeduplicationInputs(
        tpu_replicate_attr, tpu_embedding_config_str, old_gradients_node,
        deduplication_data_node, graph));
  }
  return Status::OK();
}

// Inserts a RecvTPUEmbeddingActivations node into the send_recv_nodes_map. This
// map temporarily holds the RecvTPUEmbeddingActivations and
// SendTPUEmbeddingGradients of the graph before they are rewritten.
Status InsertActivationsNodeIntoMap(Node* activations_node,
                                    SendRecvNodesMap* send_recv_nodes_map) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_7(mht_7_v, 752, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "InsertActivationsNodeIntoMap");

  std::string tpu_replicate_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(activations_node->def(), "_tpu_replicate",
                                 &tpu_replicate_attr));
  const std::string& requested_device = activations_node->requested_device();
  VLOG(1) << absl::StrFormat(
      "Inserting RecvTPUEmbeddingActivations node with _tpu_replicate "
      "attribute: %s and requested_device: %s",
      tpu_replicate_attr, requested_device);

  const SendRecvNodesMapKey key{.tpu_replicate_attr = tpu_replicate_attr,
                                .requested_device = requested_device};
  const SendRecvNodesMap::iterator it = send_recv_nodes_map->find(key);
  if (it != send_recv_nodes_map->end()) {
    if (it->second.activations_node != nullptr) {
      return errors::AlreadyExists(absl::StrFormat(
          "Found duplicate RecvTPUEmbeddingActivations node in graph with "
          "tpu_replicate attr: %s and requested_device: %s",
          tpu_replicate_attr, requested_device));
    }
    if (it->second.gradients_node == nullptr) {
      return errors::Internal(absl::StrFormat(
          "Found map object with no RecvTPUEmbeddingActivations or "
          "SendTPUEmbeddingGradients nodes and tpu_replicate attr: %s and "
          "requested_device: %s",
          tpu_replicate_attr, requested_device));
    }
    it->second.activations_node = activations_node;
  } else {
    send_recv_nodes_map->emplace(
        key, SendRecvNodes{.activations_node = activations_node,
                           .gradients_node = nullptr});
  }
  return Status::OK();
}

// Inserts a SendTPUEmbeddingGradients node into the send_recv_nodes_map. This
// map temporarily holds the RecvTPUEmbeddingActivations and
// SendTPUEmbeddingGradients of the graph before they are rewritten.
Status InsertGradientsNodeIntoMap(Node* gradients_node,
                                  SendRecvNodesMap* send_recv_nodes_map) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_8(mht_8_v, 795, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "InsertGradientsNodeIntoMap");

  std::string tpu_replicate_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(gradients_node->def(), "_tpu_replicate",
                                 &tpu_replicate_attr));
  const std::string& requested_device = gradients_node->requested_device();
  VLOG(1) << absl::StrFormat(
      "Inserting SendTPUEmbeddingGradients node with _tpu_replicate "
      "attribute: %s and requested_device: %s",
      tpu_replicate_attr, requested_device);

  const SendRecvNodesMapKey key{.tpu_replicate_attr = tpu_replicate_attr,
                                .requested_device = requested_device};
  const SendRecvNodesMap::iterator it = send_recv_nodes_map->find(key);
  if (it != send_recv_nodes_map->end()) {
    if (it->second.gradients_node != nullptr) {
      return errors::AlreadyExists(absl::StrFormat(
          "Found duplicate SendTPUEmbeddingGradients node in graph with "
          "tpu_replicate attr: %s and requested_device: %s",
          tpu_replicate_attr, requested_device));
    }
    if (it->second.activations_node == nullptr) {
      return errors::Internal(absl::StrFormat(
          "Found map object with no RecvTPUEmbeddingActivations or "
          "SendTPUEmbeddingGradients nodes and tpu_replicate attr: %s and "
          "requested_device: %s",
          tpu_replicate_attr, requested_device));
    }
    it->second.gradients_node = gradients_node;
  } else {
    send_recv_nodes_map->emplace(
        key, SendRecvNodes{.activations_node = nullptr,
                           .gradients_node = gradients_node});
  }
  return Status::OK();
}

// Groups the RecvTPUEmbeddingActivations and SendTPUEmbeddingGradients of the
// graph using their _tpu_replicate attribute and requested device.
Status GroupSendRecvNodesByTpuReplicateAttrAndDevice(
    const Graph* graph, SendRecvNodesMap* send_recv_nodes_map) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_9(mht_9_v, 837, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "GroupSendRecvNodesByTpuReplicateAttrAndDevice");

  VLOG(1) << "Grouping nodes by _tpu_replicate attribute";
  for (Node* node : graph->nodes()) {
    if (node->IsOp()) {
      if (node->op_def().name() == "RecvTPUEmbeddingActivations") {
        TF_RETURN_IF_ERROR(
            InsertActivationsNodeIntoMap(node, send_recv_nodes_map));
      } else if (node->op_def().name() == "SendTPUEmbeddingGradients") {
        TF_RETURN_IF_ERROR(
            InsertGradientsNodeIntoMap(node, send_recv_nodes_map));
      }
    }
  }
  return Status::OK();
}

// Rewrites the graph in the specified GraphOptimizationPassOptions object for
// software deduplication.
Status RewriteGraph(Graph* graph) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_10(mht_10_v, 858, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "RewriteGraph");

  SendRecvNodesMap send_recv_nodes_map;
  TF_RETURN_IF_ERROR(GroupSendRecvNodesByTpuReplicateAttrAndDevice(
      graph, &send_recv_nodes_map));

  for (const auto& attr_send_recv_nodes_pair : send_recv_nodes_map) {
    const std::string& tpu_replicate_attr =
        attr_send_recv_nodes_pair.first.tpu_replicate_attr;
    const std::string& requested_device =
        attr_send_recv_nodes_pair.first.requested_device;
    Node* activations_node = attr_send_recv_nodes_pair.second.activations_node;
    Node* gradients_node = attr_send_recv_nodes_pair.second.gradients_node;
    TF_RETURN_IF_ERROR(RewriteGraphForTpuReplicateAttrAndDevice(
        tpu_replicate_attr, requested_device, activations_node, gradients_node,
        graph));
  }

  return Status::OK();
}

// Rewriter configuration for each function def. For function defs, only node
// defs are present and need to be rewritten.
struct RewriterConfig {
  // Name of a RecvTPUEmbeddingActivations node def if present in the function.
  std::string activations_node_def_name;

  // Name of a SendTPUEmbeddingGradients node def if present in the function.
  std::string gradients_node_def_name;

  // Device name for the RecvTPUEmbeddingActivations (and/or)
  // SendTPUEmbeddingGradients node defs. Must be the same value if both are
  // present.
  std::string device_name;

  // _tpu_replicate attribute for the RecvTPUEmbeddingActivations (and/or)
  // SendTPUEmbeddingGradients node defs. Must be the same value if both are
  // present.
  std::string tpu_replicate_attr;

  // config attribute for the RecvTPUEmbeddingActivations (and/or)
  // SendTPUEmbeddingGradients node defs. Must be the same value if both are
  // present.
  std::string tpu_embedding_config_str;

  // Union of control inputs for the RecvTPUEmbeddingActivations and
  // SendTPUEmbeddingGradients node defs.
  std::vector<std::string> control_inputs;
};

// Gets the src node names for the incoming control edges of the node_def.
std::vector<std::string> GetControlInputs(const NodeDef& node_def) {
  std::vector<std::string> control_inputs;
  for (const std::string& input : node_def.input()) {
    const TensorId tensor_id = ParseTensorName(input);
    // TF2 inserts additional control dependencies within a tf.function.
    // Filter out RecvTPUEmbeddingActivations so that we do not create a cycle.
    if (tensor_id.index() == Graph::kControlSlot &&
        tensor_id.first != "RecvTPUEmbeddingActivations") {
      control_inputs.emplace_back(tensor_id.node());
    }
  }
  return control_inputs;
}

// Gets the src node names and output indices for the incoming data edges of
// node_def.
std::vector<NodeDefBuilder::NodeOut> GetDataInputs(const NodeDef& node_def,
                                                   DataType dt) {
  std::vector<NodeDefBuilder::NodeOut> data_inputs;
  for (const std::string& input : node_def.input()) {
    const TensorId tensor_id = ParseTensorName(input);
    if (tensor_id.index() != Graph::kControlSlot) {
      data_inputs.emplace_back(tensor_id.node(), tensor_id.index(), dt);
    }
  }
  return data_inputs;
}

// Computes the RewriterConfig for the specified node_def.
xla::StatusOr<RewriterConfig> ComputeRewriterConfigForNodeDef(
    const NodeDef& node_def) {
  RewriterConfig rewriter_config;
  TF_RET_CHECK(!node_def.name().empty());
  if (node_def.op() == "RecvTPUEmbeddingActivations") {
    rewriter_config.activations_node_def_name = node_def.name();
  } else {
    TF_RET_CHECK(node_def.op() == "SendTPUEmbeddingGradients");
    rewriter_config.gradients_node_def_name = node_def.name();
  }
  rewriter_config.device_name = node_def.device();

  const AttrSlice attr_slice(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attr_slice, "config",
                                 &rewriter_config.tpu_embedding_config_str));
  {
    std::string test_tpu_replicate_attr;
    if (GetNodeAttr(attr_slice, "_tpu_replicate", &test_tpu_replicate_attr)
            .ok()) {
      rewriter_config.tpu_replicate_attr = test_tpu_replicate_attr;
    }
  }
  rewriter_config.control_inputs = GetControlInputs(node_def);
  return rewriter_config;
}

// Merges the specified rewriter_config and *final_rewriter_config into
// *final_rewriter_config. While merging, validates that the device names,
// tpu_replicate and tpu_embedding_config attributes are the same if the
// final_rewriter_config has been partially populated. Aggregates the
// control inputs of both configs as well.
Status MergeRewriterConfigs(const RewriterConfig& rewriter_config,
                            RewriterConfig* final_rewriter_config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_11(mht_11_v, 972, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "MergeRewriterConfigs");

  if (final_rewriter_config->activations_node_def_name.empty() &&
      final_rewriter_config->gradients_node_def_name.empty()) {
    final_rewriter_config->device_name = rewriter_config.device_name;
    final_rewriter_config->tpu_replicate_attr =
        rewriter_config.tpu_replicate_attr;
    final_rewriter_config->tpu_embedding_config_str =
        rewriter_config.tpu_embedding_config_str;
  } else {
    if (final_rewriter_config->device_name != rewriter_config.device_name) {
      return errors::InvalidArgument(absl::StrFormat(
          "Mismatch in device names for TPU embedding nodes: %s != %s",
          final_rewriter_config->device_name, rewriter_config.device_name));
    }
    if (final_rewriter_config->tpu_replicate_attr !=
        rewriter_config.tpu_replicate_attr) {
      return errors::InvalidArgument(
          absl::StrFormat("Mismatch in _tpu_replicate attributes for TPU "
                          "embedding nodes: %s != %s",
                          final_rewriter_config->tpu_replicate_attr,
                          rewriter_config.tpu_replicate_attr));
    }
    if (final_rewriter_config->tpu_embedding_config_str !=
        rewriter_config.tpu_embedding_config_str) {
      return errors::InvalidArgument(
          absl::StrFormat("Mismatch in config attributes for TPU "
                          "embedding nodes: %s != %s",
                          final_rewriter_config->tpu_embedding_config_str,
                          rewriter_config.tpu_embedding_config_str));
    }
  }
  absl::c_copy(rewriter_config.control_inputs,
               std::back_inserter(final_rewriter_config->control_inputs));

  if (!rewriter_config.activations_node_def_name.empty()) {
    if (!final_rewriter_config->activations_node_def_name.empty()) {
      return errors::InvalidArgument(
          absl::StrFormat("Found duplicate RecvTPUEmbeddingActivations nodes "
                          "%s and %s in function.",
                          rewriter_config.activations_node_def_name,
                          final_rewriter_config->activations_node_def_name));
    }
    final_rewriter_config->activations_node_def_name =
        rewriter_config.activations_node_def_name;
  }
  if (!rewriter_config.gradients_node_def_name.empty()) {
    if (!final_rewriter_config->gradients_node_def_name.empty()) {
      return errors::InvalidArgument(
          absl::StrFormat("Found duplicate SendTPUEmbeddingGradients nodes %s "
                          "and %s in function.",
                          rewriter_config.gradients_node_def_name,
                          final_rewriter_config->gradients_node_def_name));
    }
    final_rewriter_config->gradients_node_def_name =
        rewriter_config.gradients_node_def_name;
  }

  return Status::OK();
}

// Data type for map from device name to RewriterConfig.
using RewriterConfigsByDevice =
    absl::flat_hash_map<std::string, RewriterConfig>;

// Computes the RewriterConfigs for the specified FunctionDef (fdef) for each
// device (multiple devices can be used for model parallelism).
xla::StatusOr<RewriterConfigsByDevice> ComputeRewriterConfigsByDevice(
    const FunctionDef& fdef) {
  RewriterConfigsByDevice final_rewriter_configs_by_device;

  for (const NodeDef& node_def : fdef.node_def()) {
    if (node_def.op() == "RecvTPUEmbeddingActivations" ||
        node_def.op() == "SendTPUEmbeddingGradients") {
      TF_ASSIGN_OR_RETURN(const RewriterConfig rewriter_config,
                          ComputeRewriterConfigForNodeDef(node_def));

      // Merge a rewriter config if there is one for this device or create one
      // if there wasn't one before.
      TF_RETURN_IF_ERROR(MergeRewriterConfigs(
          rewriter_config,
          &final_rewriter_configs_by_device[node_def.device()]));
    }
  }
  return final_rewriter_configs_by_device;
}

// Determines whether any elements in a RewriterConfigsByDevice map contains any
// embedding operations.
bool RewriterConfigsByDeviceHasEmbeddingOperations(
    const RewriterConfigsByDevice& rewriter_configs_by_device) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_12(mht_12_v, 1064, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "RewriterConfigsByDeviceHasEmbeddingOperations");

  // Determine if there are any embedding operations in this function, and
  // skip it if there are none.
  bool any_embedding_operations = false;
  for (const auto& device_and_rewriter_config : rewriter_configs_by_device) {
    const RewriterConfig& rewriter_config = device_and_rewriter_config.second;
    if (rewriter_config.activations_node_def_name.empty() &&
        rewriter_config.gradients_node_def_name.empty()) {
      continue;
    }
    any_embedding_operations = true;
    break;
  }

  return any_embedding_operations;
}

// Rewrites the function defs in the specified GraphOptimizationPassOptions
// object for software deduplication.
Status RewriteFunctionDefs(FunctionLibraryDefinition* flib_def) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_13(mht_13_v, 1086, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "RewriteFunctionDefs");

  for (const std::string& fname : flib_def->ListFunctionNames()) {
    // The function def cannot be modified. Hence, make a copy, modify the copy
    // and then replace the original function def using the copy.
    const FunctionDef* fdef = flib_def->Find(fname);
    TF_ASSIGN_OR_RETURN(
        const RewriterConfigsByDevice rewriter_configs_by_device,
        ComputeRewriterConfigsByDevice(*fdef));

    // Determine if there are any embedding operations in this function, and
    // skip it if there are none.
    if (!RewriterConfigsByDeviceHasEmbeddingOperations(
            rewriter_configs_by_device)) {
      continue;
    }

    // Make a copy of the function def and clear the node defs.
    FunctionDef new_fdef = *fdef;
    new_fdef.clear_node_def();

    // Build a receive deduplication data node for each device that uses TPU
    // embeddings.
    absl::flat_hash_map<std::string, std::string>
        deduplication_data_node_names_by_device;
    for (const auto& device_and_rewriter_config : rewriter_configs_by_device) {
      const RewriterConfig& rewriter_config = device_and_rewriter_config.second;
      TF_RET_CHECK(device_and_rewriter_config.first ==
                   rewriter_config.device_name);

      TF_ASSIGN_OR_RETURN(
          const NodeDef deduplication_data_node_def,
          MakeRecvDeduplicationDataNodeDef(
              rewriter_config.device_name, rewriter_config.tpu_replicate_attr,
              rewriter_config.tpu_embedding_config_str,
              rewriter_config.control_inputs));
      *new_fdef.add_node_def() = deduplication_data_node_def;
      deduplication_data_node_names_by_device.emplace(
          rewriter_config.device_name, deduplication_data_node_def.name());
    }

    for (const NodeDef& node_def : fdef->node_def()) {
      if (node_def.op() == "RecvTPUEmbeddingActivations") {
        const RewriterConfig& rewriter_config =
            rewriter_configs_by_device.at(node_def.device());
        const std::string& deduplication_data_node_name =
            deduplication_data_node_names_by_device.at(
                rewriter_config.device_name);
        TF_ASSIGN_OR_RETURN(
            const NodeDef activations_node_def,
            MakeRecvActivationsNodeDef(
                node_def, deduplication_data_node_name,
                rewriter_config.device_name, rewriter_config.tpu_replicate_attr,
                rewriter_config.tpu_embedding_config_str,
                GetDataInputs(node_def, DT_FLOAT), GetControlInputs(node_def)));
        *new_fdef.add_node_def() = activations_node_def;
      } else if (node_def.op() == "SendTPUEmbeddingGradients") {
        const RewriterConfig& rewriter_config =
            rewriter_configs_by_device.at(node_def.device());
        const std::string& deduplication_data_node_name =
            deduplication_data_node_names_by_device.at(
                rewriter_config.device_name);
        TF_ASSIGN_OR_RETURN(
            const NodeDef gradients_node_def,
            MakeSendGradientsNodeDef(
                node_def, deduplication_data_node_name,
                rewriter_config.device_name, rewriter_config.tpu_replicate_attr,
                rewriter_config.tpu_embedding_config_str,
                GetDataInputs(node_def, DT_FLOAT), GetControlInputs(node_def)));
        *new_fdef.add_node_def() = gradients_node_def;
      } else {
        *new_fdef.add_node_def() = node_def;
      }
    }

    TF_RETURN_IF_ERROR(flib_def->ReplaceFunction(fname, new_fdef));
  }
  return Status::OK();
}

}  // namespace

Status TPUEmbeddingSoftwareDeduplicationRewritePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePStpu_embedding_software_deduplication_rewrite_passDTcc mht_14(mht_14_v, 1171, "", "./tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.cc", "TPUEmbeddingSoftwareDeduplicationRewritePass::Run");

  TF_RETURN_IF_ERROR(RewriteGraph(options.graph->get()));
  TF_RETURN_IF_ERROR(RewriteFunctionDefs(options.flib_def));
  return Status::OK();
}

}  // namespace tensorflow
