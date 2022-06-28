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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc() {
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

// Configuration for distributed TPU jobs

#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.h"

#include <unordered_map>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_options.h"
#include "tensorflow/core/tpu/tpu_init_mode.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

constexpr char kIdentityOp[] = "Identity";
constexpr char kConfigureOp[] = "ConfigureDistributedTPU";
constexpr char kInternalConfigureOp[] = "_ConfigureDistributedTPU";
constexpr char kWaitOp[] = "_WaitForDistributedTPU";
constexpr char kHostConfigureOp[] = "_InitializeHostForDistributedTPU";
constexpr char kGlobalTPUArrayOp[] = "_SetGlobalTPUArray";
constexpr char kShutdownOp[] = "ShutdownDistributedTPU";
constexpr char kInternalShutdownOp[] = "_ShutdownDistributedTPU";
constexpr char kHostDisconnectOp[] = "_DisconnectHostFromDistributedTPUSystem";
constexpr char kEmbeddingConfigurationAttr[] = "embedding_config";
constexpr char kTpuCancellationClosesChipsAttr[] =
    "tpu_cancellation_closes_chips";
constexpr int kDefaultStartupTimeout = 20;

Status AddConfigurationNode(const string& configuration_device_name,
                            int number_of_hosts, Graph* graph,
                            bool enable_whole_mesh_compilations,
                            Node** configuration_node) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("configuration_device_name: \"" + configuration_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddConfigurationNode");

  NodeDef config_def;
  config_def.set_name(graph->NewName("configure_distributed_tpu"));
  config_def.set_op(kInternalConfigureOp);
  config_def.set_device(configuration_device_name);
  AddNodeAttr("N", number_of_hosts, &config_def);
  AddNodeAttr("enable_whole_mesh_compilations", enable_whole_mesh_compilations,
              &config_def);
  // TODO(shikharagarwal): Fill with appropriate original node debug info.

  TF_ASSIGN_OR_RETURN(*configuration_node, graph->AddNode(config_def));
  (*configuration_node)->set_assigned_device_name(configuration_device_name);
  return Status::OK();
}

Status AddHostConfigNode(const string& host_device_name,
                         Node* configuration_node, Graph* graph,
                         bool enable_whole_mesh_compilations,
                         int tpu_cancellation_closes_chips,
                         Node** host_configuration_node) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("host_device_name: \"" + host_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddHostConfigNode");

  NodeDef host_config_def;
  host_config_def.set_name(graph->NewName("configure_tpu_host"));
  host_config_def.set_op(kHostConfigureOp);
  host_config_def.set_device(host_device_name);
  AddNodeAttr("enable_whole_mesh_compilations", enable_whole_mesh_compilations,
              &host_config_def);
  AddNodeAttr(kTpuCancellationClosesChipsAttr, tpu_cancellation_closes_chips,
              &host_config_def);
  MergeDebugInfo(NodeDebugInfo(configuration_node->def()), &host_config_def);

  TF_ASSIGN_OR_RETURN(*host_configuration_node,
                      graph->AddNode(host_config_def));
  (*host_configuration_node)->set_assigned_device_name(host_device_name);
  graph->AddEdge(configuration_node, 0, *host_configuration_node, 0);
  return Status::OK();
}

Status AddWaitNode(const string& configuration_device_name,
                   const std::vector<Node*>& host_configuration_nodes,
                   Graph* graph, Node** wait_node) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("configuration_device_name: \"" + configuration_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_2(mht_2_v, 278, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddWaitNode");

  NodeDef wait_def;
  wait_def.set_name(graph->NewName("wait_for_distributed_tpu_system"));
  wait_def.set_op(kWaitOp);
  wait_def.set_device(configuration_device_name);
  AddNodeAttr("N", static_cast<int32>(host_configuration_nodes.size()),
              &wait_def);
  AddNodeAttr("startup_timeout_sec", kDefaultStartupTimeout, &wait_def);
  if (!host_configuration_nodes.empty()) {
    MergeDebugInfo(NodeDebugInfo(host_configuration_nodes[0]->def()),
                   &wait_def);
  }

  TF_ASSIGN_OR_RETURN(*wait_node, graph->AddNode(wait_def));
  (*wait_node)->set_assigned_device_name(configuration_device_name);
  // Get the inputs from the host configuration nodes.
  for (int i = 0; i < host_configuration_nodes.size(); ++i) {
    graph->AddEdge(host_configuration_nodes[i], 0, *wait_node, i);
  }
  return Status::OK();
}

Status AddGlobalTPUArrayNode(const string& host_device_name, Node* wait_node,
                             Graph* graph, Node** global_tpu_array_node) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("host_device_name: \"" + host_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_3(mht_3_v, 305, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddGlobalTPUArrayNode");

  NodeDef global_tpu_array_def;
  global_tpu_array_def.set_name(graph->NewName("set_global_tpu_array"));
  global_tpu_array_def.set_op(kGlobalTPUArrayOp);
  global_tpu_array_def.set_device(host_device_name);
  MergeDebugInfo(NodeDebugInfo(wait_node->def()), &global_tpu_array_def);

  TF_ASSIGN_OR_RETURN(*global_tpu_array_node,
                      graph->AddNode(global_tpu_array_def));
  (*global_tpu_array_node)->set_assigned_device_name(host_device_name);
  graph->AddEdge(wait_node, 0, *global_tpu_array_node, 0);
  return Status::OK();
}

Status AddSynchronizationNode(
    const NodeDef& sync_node_def, const string& device_name,
    const std::vector<Node*>& global_array_id_nodes, Node* wait_node,
    const std::vector<DistributedTPURewriteHelpers::OutputDependency>&
        output_dependencies,
    Graph* graph) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_4(mht_4_v, 328, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddSynchronizationNode");

  NodeDef sync_def;
  sync_def.set_name(sync_node_def.name());
  sync_def.set_op(kIdentityOp);
  sync_def.set_device(device_name);
  AddNodeAttr("T", DT_STRING, &sync_def);
  MergeDebugInfo(NodeDebugInfo(sync_node_def), &sync_def);

  TF_ASSIGN_OR_RETURN(Node * sync_node, graph->AddNode(sync_def));
  sync_node->set_assigned_device_name(device_name);
  // Add control edges from the global array id nodes.
  for (auto node : global_array_id_nodes) {
    graph->AddControlEdge(node, sync_node);
  }
  // Forward the data from the wait node.
  graph->AddEdge(wait_node, 0, sync_node, 0);
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


Status AddShutdownNode(
    const NodeDef& shutdown_node_def, const string& shutdown_device_name,
    const std::vector<DistributedTPURewriteHelpers::OutputDependency>&
        output_dependencies,
    Graph* graph, Node** shutdown_node) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("shutdown_device_name: \"" + shutdown_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_5(mht_5_v, 365, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddShutdownNode");

  NodeDef shutdown_def;
  shutdown_def.set_name(shutdown_node_def.name());
  shutdown_def.set_op(kInternalShutdownOp);
  shutdown_def.set_device(shutdown_device_name);
  MergeDebugInfo(NodeDebugInfo(shutdown_node_def), &shutdown_def);

  TF_ASSIGN_OR_RETURN(*shutdown_node, graph->AddNode(shutdown_def));
  (*shutdown_node)->set_assigned_device_name(shutdown_device_name);
  // Replace the output control edges.
  for (const DistributedTPURewriteHelpers::OutputDependency& dep :
       output_dependencies) {
    if (dep.dst_input != Graph::kControlSlot) {
      return errors::Internal("Shutdown node had non-control edge output");
    }
    graph->AddControlEdge(*shutdown_node, dep.dst);
  }
  return Status::OK();
}

Status AddHostDisconnectNode(const string& host_device_name,
                             const std::vector<Node*>& input_dependencies,
                             Node* post_disconnect_node, int output_index,
                             Graph* graph) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("host_device_name: \"" + host_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_6(mht_6_v, 392, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "AddHostDisconnectNode");

  NodeDef host_disconnect_def;
  host_disconnect_def.set_name(graph->NewName("disconnect_tpu_host"));
  host_disconnect_def.set_op(kHostDisconnectOp);
  host_disconnect_def.set_device(host_device_name);
  MergeDebugInfo(NodeDebugInfo(post_disconnect_node->def()),
                 &host_disconnect_def);

  TF_ASSIGN_OR_RETURN(Node * host_disconnect_node,
                      graph->AddNode(host_disconnect_def));
  host_disconnect_node->set_assigned_device_name(host_device_name);
  // Replace the input control edges.
  for (Node* src_node : input_dependencies) {
    graph->AddControlEdge(src_node, host_disconnect_node);
  }
  if (output_index == -1) {
    graph->AddControlEdge(host_disconnect_node, post_disconnect_node);
  } else {
    graph->AddEdge(host_disconnect_node, 0, post_disconnect_node, output_index);
  }
  return Status::OK();
}

}  // namespace

Status DistributedTPUConfigurationRewritePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_7(mht_7_v, 421, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "DistributedTPUConfigurationRewritePass::Run");

  VLOG(1) << "DistributedTPUConfigurationRewritePass::Run";

  Graph* graph = options.graph->get();

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("distributed_tpu_configuration_before", *graph,
                    options.flib_def);
  }

  // This pass can only run in the session master, which should fill
  // in the device_set field to the options.
  TF_RET_CHECK(options.device_set != nullptr);

  TF_RETURN_IF_ERROR(
      DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType(
          kConfigureOp, graph, *options.device_set,
          [](const NodeDef& configuration_node_def,
             const string& configuration_device_name,
             const std::vector<Device*>& host_devices,
             const std::vector<Node*>& input_dependencies,
             const std::vector<DistributedTPURewriteHelpers::OutputDependency>&
                 output_dependencies,
             Graph* graph) -> Status {
            const std::string& embedding_attr_string = GetNodeAttrString(
                AttrSlice(configuration_node_def), kEmbeddingConfigurationAttr);

            if (!embedding_attr_string.empty()) {
              return errors::InvalidArgument("embedding_config must be empty.");
            }

            bool is_global_init = false;
            bool enable_whole_mesh_compilations = false;
            TF_RETURN_IF_ERROR(GetNodeAttr(configuration_node_def,
                                           "is_global_init", &is_global_init));
            TryGetNodeAttr(configuration_node_def,
                           "enable_whole_mesh_compilations",
                           &enable_whole_mesh_compilations);
            TF_RETURN_IF_ERROR(SetTPUInitMode(
                is_global_init ? TPUInitMode::kGlobal : TPUInitMode::kRegular));

            bool compilation_failure_closes_chips;
            TF_RETURN_IF_ERROR(GetNodeAttr(configuration_node_def,
                                           "compilation_failure_closes_chips",
                                           &compilation_failure_closes_chips));
            internal::SetTpuCompilationFailureClosesChips(
                compilation_failure_closes_chips);

            int tpu_cancellation_closes_chips;
            TF_RETURN_IF_ERROR(GetNodeAttr(configuration_node_def,
                                           kTpuCancellationClosesChipsAttr,
                                           &tpu_cancellation_closes_chips));

            // Add the global TPU system configuration node.
            Node* configuration_node;
            TF_RETURN_IF_ERROR(AddConfigurationNode(
                configuration_device_name, host_devices.size(), graph,
                enable_whole_mesh_compilations, &configuration_node));

            // Add the host disconnect nodes.
            for (int i = 0; i < host_devices.size(); ++i) {
              const auto host_device = host_devices[i];
              TF_RETURN_IF_ERROR(
                  AddHostDisconnectNode(host_device->name(), input_dependencies,
                                        configuration_node, i, graph));
            }

            // Add the host configuration nodes.
            std::vector<Node*> host_configuration_nodes;
            for (const auto host_device : host_devices) {
              Node* host_configuration_node;
              TF_RETURN_IF_ERROR(AddHostConfigNode(
                  host_device->name(), configuration_node, graph,
                  enable_whole_mesh_compilations, tpu_cancellation_closes_chips,
                  &host_configuration_node));
              host_configuration_nodes.push_back(host_configuration_node);
            }

            // Add the node to wait for the system configuration to
            // stabilize. Use the name of the original dummy Op in case it was
            // the target of a Session::Run call.
            Node* wait_node;
            TF_RETURN_IF_ERROR(AddWaitNode(configuration_device_name,
                                           host_configuration_nodes, graph,
                                           &wait_node));

            // Add the nodes to set the global TPU ids at each host.
            std::vector<Node*> global_array_id_nodes;
            for (const auto host_device : host_devices) {
              Node* global_array_id_node;
              TF_RETURN_IF_ERROR(AddGlobalTPUArrayNode(host_device->name(),
                                                       wait_node, graph,
                                                       &global_array_id_node));
              global_array_id_nodes.push_back(global_array_id_node);
            }

            if (host_devices.empty()) {
              return errors::InvalidArgument("TPU job contains no CPU devices");
            }
            TF_RET_CHECK(!host_devices.empty());

            TF_RETURN_IF_ERROR(AddSynchronizationNode(
                configuration_node_def, host_devices.front()->name(),
                global_array_id_nodes, wait_node, output_dependencies, graph));

            return Status::OK();
          }));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("distributed_tpu_configuration_after", *graph,
                    options.flib_def);
  }

  VLOG(1) << "DistributedTPUConfigurationRewritePass::Run() finished";
  return Status::OK();
}

Status DistributedTPUShutdownRewritePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_configuration_rewrite_passDTcc mht_8(mht_8_v, 542, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.cc", "DistributedTPUShutdownRewritePass::Run");

  VLOG(1) << "DistributedTPUShutdownRewritePass::Run";

  Graph* graph = options.graph->get();

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("distributed_tpu_shutdown_before", *graph,
                    options.flib_def);
  }

  // This pass can only run in the session master, which should fill
  // in the device_set field to the options.
  TF_RET_CHECK(options.device_set != nullptr);

  TF_RETURN_IF_ERROR(
      DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType(
          kShutdownOp, graph, *options.device_set,
          [](const NodeDef& shutdown_node_def,
             const string& shutdown_device_name,
             const std::vector<Device*>& host_devices,
             const std::vector<Node*>& input_dependencies,
             const std::vector<DistributedTPURewriteHelpers::OutputDependency>&
                 output_dependencies,
             Graph* graph) -> Status {
            Node* shutdown_node;
            TF_RETURN_IF_ERROR(
                AddShutdownNode(shutdown_node_def, shutdown_device_name,
                                output_dependencies, graph, &shutdown_node));

            // Add the host disconnect nodes.
            for (const auto host_device : host_devices) {
              TF_RETURN_IF_ERROR(
                  AddHostDisconnectNode(host_device->name(), input_dependencies,
                                        shutdown_node, -1, graph));
            }

            return Status::OK();
          }));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("distributed_tpu_shutdown_after", *graph, options.flib_def);
  }

  VLOG(1) << "DistributedTPUShutdownRewritePass::Run() finished";
  return Status::OK();
}

}  // namespace tensorflow
