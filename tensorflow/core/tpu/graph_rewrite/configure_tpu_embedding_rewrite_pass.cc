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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.h"

#include <string>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"
#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

constexpr char kNoOp[] = "NoOp";
constexpr char kConfigureOp[] = "ConfigureTPUEmbedding";
constexpr char kExecuteEmbeddingPartitionerOp[] =
    "_ExecuteTPUEmbeddingPartitioner";
constexpr char kConfigureEmbeddingMemoryOp[] = "_ConfigureTPUEmbeddingMemory";
constexpr char kConfigureEmbeddingOp[] = "_ConfigureTPUEmbeddingHost";
constexpr char kConnectEmbeddingHostsOp[] =
    "_ConnectInterTPUEmbeddingCommunication";
constexpr char kFinalizeEmbeddingOp[] =
    "_FinalizeTPUEmbeddingSystemConfiguration";
constexpr char kEmbeddingConfigurationAttr[] = "config";

Status AddSynchronizationNode(
    const NodeDef& sync_node_def, const string& device_name,
    const std::vector<Node*>& global_array_id_nodes,
    const std::vector<DistributedTPURewriteHelpers::OutputDependency>
        output_dependencies,
    Graph* graph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_0(mht_0_v, 229, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddSynchronizationNode");

  NodeDef sync_def;
  sync_def.set_name(sync_node_def.name());
  sync_def.set_op(kNoOp);
  sync_def.set_device(device_name);
  MergeDebugInfo(NodeDebugInfo(sync_node_def), &sync_def);

  TF_ASSIGN_OR_RETURN(Node * sync_node, graph->AddNode(sync_def));
  sync_node->set_assigned_device_name(device_name);
  // Add control edges from the global array id nodes.
  for (auto node : global_array_id_nodes) {
    graph->AddControlEdge(node, sync_node);
  }
  // Replace the output edges.
  for (const DistributedTPURewriteHelpers::OutputDependency& dep :
       output_dependencies) {
    if (dep.dst_input == Graph::kControlSlot) {
      graph->AddControlEdge(sync_node, dep.dst);
    } else {
      graph->AddEdge(sync_node, dep.src_output, dep.dst, dep.dst_input);
    }
  }
  return Status::OK();
}

Status AddPartitionerEmbeddingNode(const string& configuration_device_name,
                                   const string& config,
                                   const std::vector<Node*>& input_dependencies,
                                   Graph* graph, Node** partitioner_node) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("configuration_device_name: \"" + configuration_device_name + "\"");
   mht_1_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddPartitionerEmbeddingNode");

  NodeDef partitioner_def;
  partitioner_def.set_name(graph->NewName("execute_embedding_partitioner"));
  partitioner_def.set_op(kExecuteEmbeddingPartitionerOp);
  partitioner_def.set_device(configuration_device_name);
  AddNodeAttr("config", config, &partitioner_def);

  TF_ASSIGN_OR_RETURN(*partitioner_node, graph->AddNode(partitioner_def));
  (*partitioner_node)->set_assigned_device_name(configuration_device_name);
  // Replace the input control edges.
  for (Node* src_node : input_dependencies) {
    graph->AddControlEdge(src_node, *partitioner_node);
  }

  return Status::OK();
}

Status AddMemoryConfigurationEmbeddingNode(const string& host_device_name,
                                           const string& config,
                                           Node* partitioner_node, Graph* graph,
                                           Node** embedding_node) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("host_device_name: \"" + host_device_name + "\"");
   mht_2_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_2(mht_2_v, 287, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddMemoryConfigurationEmbeddingNode");

  NodeDef embedding_def;
  embedding_def.set_name(graph->NewName("configure_tpu_embedding_memory"));
  embedding_def.set_op(kConfigureEmbeddingMemoryOp);
  embedding_def.set_device(host_device_name);
  AddNodeAttr("config", config, &embedding_def);

  TF_ASSIGN_OR_RETURN(*embedding_node, graph->AddNode(embedding_def));
  (*embedding_node)->set_assigned_device_name(host_device_name);
  graph->AddEdge(partitioner_node, 0, *embedding_node, 0);
  return Status::OK();
}

Status AddHostConfigurationEmbeddingNode(const string& host_device_name,
                                         const string& config,
                                         Node* partitioner_node,
                                         const std::vector<Node*>& memory_nodes,
                                         Graph* graph, Node** embedding_node) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("host_device_name: \"" + host_device_name + "\"");
   mht_3_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_3(mht_3_v, 309, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddHostConfigurationEmbeddingNode");

  NodeDef embedding_def;
  embedding_def.set_name(graph->NewName("configure_host_embedding"));
  embedding_def.set_op(kConfigureEmbeddingOp);
  embedding_def.set_device(host_device_name);
  AddNodeAttr("config", config, &embedding_def);
  AddNodeAttr("N", static_cast<int>(memory_nodes.size()), &embedding_def);
  if (!memory_nodes.empty()) {
    MergeDebugInfo(NodeDebugInfo(memory_nodes[0]->def()), &embedding_def);
  }

  TF_ASSIGN_OR_RETURN(*embedding_node, graph->AddNode(embedding_def));
  (*embedding_node)->set_assigned_device_name(host_device_name);
  // Add inputs from the partitioner node and all the memory nodes.
  graph->AddEdge(partitioner_node, 0, *embedding_node, 0);
  for (int i = 0; i < memory_nodes.size(); ++i) {
    graph->AddEdge(memory_nodes[i], 0, *embedding_node, i + 1);
  }
  return Status::OK();
}

Status AddSetupPropagationEmbeddingNode(
    const string& device_name, const string& node_name, const string& op_name,
    const std::vector<Node*>& embedding_nodes, Graph* graph, Node** node) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device_name: \"" + device_name + "\"");
   mht_4_v.push_back("node_name: \"" + node_name + "\"");
   mht_4_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_4(mht_4_v, 338, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddSetupPropagationEmbeddingNode");

  NodeDef node_def;
  node_def.set_name(node_name);
  node_def.set_op(op_name);
  node_def.set_device(device_name);
  AddNodeAttr("N", static_cast<int>(embedding_nodes.size()), &node_def);
  if (!embedding_nodes.empty()) {
    MergeDebugInfo(NodeDebugInfo(embedding_nodes[0]->def()), &node_def);
  }

  TF_ASSIGN_OR_RETURN(*node, graph->AddNode(node_def));
  (*node)->set_assigned_device_name(device_name);
  // Add inputs from the embedding nodes.
  for (int i = 0; i < embedding_nodes.size(); ++i) {
    graph->AddEdge(embedding_nodes[i], 0, *node, i);
  }
  return Status::OK();
}

Status AddConnectEmbeddingNode(const string& host_device_name,
                               const std::vector<Node*>& embedding_nodes,
                               Graph* graph, Node** connect_node) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("host_device_name: \"" + host_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_5(mht_5_v, 363, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddConnectEmbeddingNode");

  return AddSetupPropagationEmbeddingNode(
      host_device_name, graph->NewName("connect_tpu_embedding_hosts"),
      kConnectEmbeddingHostsOp, embedding_nodes, graph, connect_node);
}

Status AddFinalizeEmbeddingNode(const string& configuration_device_name,
                                const std::vector<Node*>& embedding_nodes,
                                Graph* graph, Node** finalize_node) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("configuration_device_name: \"" + configuration_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_6(mht_6_v, 375, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "AddFinalizeEmbeddingNode");

  return AddSetupPropagationEmbeddingNode(
      configuration_device_name, graph->NewName("finalize_host_embedding"),
      kFinalizeEmbeddingOp, embedding_nodes, graph, finalize_node);
}

}  // namespace

Status ConfigureTPUEmbeddingRewritePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSconfigure_tpu_embedding_rewrite_passDTcc mht_7(mht_7_v, 387, "", "./tensorflow/core/tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc", "ConfigureTPUEmbeddingRewritePass::Run");

  VLOG(1) << "ConfigureTPUEmbeddingRewritePass::Run";

  Graph* graph = options.graph->get();

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("configure_tpu_embedding_before", *graph, options.flib_def);
  }

  // This pass can only run in the session master, which should fill
  // in the device_set field to the options.
  TF_RET_CHECK(options.device_set != nullptr);

  TF_RETURN_IF_ERROR(
      DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType(
          kConfigureOp, graph, *options.device_set,
          [](const NodeDef& configuration_node_def,
             const std::string& configuration_device_name,
             const std::vector<Device*>& host_devices,
             const std::vector<Node*>& input_dependencies,
             const std::vector<DistributedTPURewriteHelpers::OutputDependency>&
                 output_dependencies,
             Graph* graph) -> Status {
            if (host_devices.empty()) {
              return errors::InvalidArgument("TPU job contains no CPU devices");
            }
            TF_RET_CHECK(!host_devices.empty());

            auto get_updated_device_name =
                [](absl::string_view initial_device_name)
                -> xla::StatusOr<std::string> {
              DeviceNameUtils::ParsedName device_spec;
              TF_RET_CHECK(DeviceNameUtils::ParseFullName(initial_device_name,
                                                          &device_spec));
              // Keep job, replica, and task information, but change the
              // '/device:TPU_SYSTEM:0' specification to '/device:CPU:0'.
              device_spec.type = "CPU";
              return DeviceNameUtils::ParsedNameToString(device_spec);
            };

            // Must not use embedding_attr_string beyond the lifetime of
            // configuration_node_def.
            const std::string& embedding_attr_string = GetNodeAttrString(
                AttrSlice(configuration_node_def), kEmbeddingConfigurationAttr);
            if (embedding_attr_string.empty()) {
              return errors::InvalidArgument("TPU embedding_config is empty.");
            } else {
              // Auto populate the feature descriptor so that we can make use
              // of these fields later.
              std::string updated_embedding_attr_string;
              tpu::TPUEmbeddingConfiguration tpu_embedding_config;
              tpu_embedding_config.ParseFromString(embedding_attr_string);
              TF_RETURN_IF_ERROR(PopulateMissingFieldsInTPUEmbeddingConfig(
                  &tpu_embedding_config));
              tpu_embedding_config.SerializeToString(
                  &updated_embedding_attr_string);

              // Execute the TPU embedding partitioner if configured to do so.
              Node* embedding_partitioner_node;
              TF_ASSIGN_OR_RETURN(
                  const std::string configuration_device_string,
                  get_updated_device_name(configuration_device_name));
              TF_RETURN_IF_ERROR(AddPartitionerEmbeddingNode(
                  configuration_device_string, updated_embedding_attr_string,
                  input_dependencies, graph, &embedding_partitioner_node));

              // Obtain the device strings for configuring the TPU embedding
              // core on each host.
              std::vector<std::string> host_device_strings(host_devices.size());
              for (int i = 0; i < host_devices.size(); ++i) {
                TF_ASSIGN_OR_RETURN(
                    host_device_strings[i],
                    get_updated_device_name(host_devices[i]->name()));
              }

              // Add nodes that configure the HBM memory at each host.
              std::vector<Node*> tpu_embedding_memory_nodes;
              for (int i = 0; i < host_devices.size(); ++i) {
                Node* tpu_embedding_memory_node;
                TF_RETURN_IF_ERROR(AddMemoryConfigurationEmbeddingNode(
                    host_device_strings[i], updated_embedding_attr_string,
                    embedding_partitioner_node, graph,
                    &tpu_embedding_memory_node));
                tpu_embedding_memory_nodes.push_back(tpu_embedding_memory_node);
              }

              // Add the nodes to configure the embeddings at each host.
              std::vector<Node*> host_embedding_nodes;
              for (int i = 0; i < host_devices.size(); ++i) {
                Node* host_embedding_node;
                TF_RETURN_IF_ERROR(AddHostConfigurationEmbeddingNode(
                    host_device_strings[i], updated_embedding_attr_string,
                    embedding_partitioner_node, tpu_embedding_memory_nodes,
                    graph, &host_embedding_node));
                host_embedding_nodes.push_back(host_embedding_node);
              }

              // Add the nodes to specify the ports to connect to on each each.
              // Note that each TPU worker needs to know how to connect to all
              // other TPU workers in the system, so these are all-to-all
              // communication links.
              std::vector<Node*> connect_embedding_nodes;
              for (int i = 0; i < host_devices.size(); ++i) {
                Node* connect_embedding_node;
                TF_RETURN_IF_ERROR(AddConnectEmbeddingNode(
                    host_device_strings[i], host_embedding_nodes, graph,
                    &connect_embedding_node));
                connect_embedding_nodes.push_back(connect_embedding_node);
              }

              // Add the finalize node that checks that the HBM base addresses
              // allocated are the same across all TPU worker tasks.
              Node* finalize_embedding_node;
              TF_RETURN_IF_ERROR(AddFinalizeEmbeddingNode(
                  configuration_device_string, host_embedding_nodes, graph,
                  &finalize_embedding_node));

              // Wait for the connect and finalize nodes to complete execution.
              std::vector<Node*> all_end_nodes(connect_embedding_nodes);
              all_end_nodes.push_back(finalize_embedding_node);

              TF_RETURN_IF_ERROR(AddSynchronizationNode(
                  configuration_node_def, configuration_device_string,
                  all_end_nodes, output_dependencies, graph));
            }

            return Status::OK();
          }));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("configure_tpu_embedding_after", *graph, options.flib_def);
  }

  VLOG(1) << "ConfigureTPUEmbeddingRewritePass::Run() finished";
  return Status::OK();
}

}  // namespace tensorflow
