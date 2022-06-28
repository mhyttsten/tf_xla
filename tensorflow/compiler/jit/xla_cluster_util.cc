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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_cluster_util.h"

#include <string>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/xla_config_registry.h"

namespace tensorflow {

const char* const kXlaClusterAttr = "_XlaCluster";
const char* const kXlaOutsideCompilationAttr = "_XlaOutsideCompilation";
const char* const kXlaCompileTimeConstantInputsAttr =
    "_XlaCompileTimeConstantInputs";

namespace {
// Returns a string describing how an edge from src to dst would
// create a cycle.
string DescribeCycle(const GraphCycles* cycles, const Graph& graph, int src,
                     int dst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "DescribeCycle");

  int32_t max_path_size = graph.num_node_ids() + 1;
  std::vector<int32> path(max_path_size);
  int32_t path_size = cycles->FindPath(dst, src, max_path_size, path.data());
  if (path_size == 0) {
    return "";
  }

  auto node_name = [&graph](int node_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "lambda");

    if (!FastBoundsCheck(node_id, graph.num_node_ids())) {
      return string("(null)");
    }
    auto* node = graph.FindNodeId(node_id);
    if (node == nullptr) {
      return string("(null)");
    }
    return node->name();
  };

  string description;
  absl::StrAppend(&description, "Edge from ", node_name(src), " to ",
                  node_name(dst), " would create a cycle.\n");
  path.resize(path_size);
  for (int32_t node_id : path) {
    string ascii_art;
    if (node_id == dst) {
      ascii_art = "+-> ";
    } else if (node_id != src) {
      ascii_art = "|   ";
    } else {
      ascii_art = "+-- ";
    }
    absl::StrAppend(&description, ascii_art, node_name(node_id), "\n");
  }
  return description;
}

bool AlwaysForwardsRefInput(const Node& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "AlwaysForwardsRefInput");
 return node.IsIdentity(); }

}  // namespace

bool HasForwardedRefInput(const Node& node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "HasForwardedRefInput");

  if (AlwaysForwardsRefInput(node)) {
    for (const Edge* incoming_edge : node.in_edges()) {
      if (incoming_edge->IsControlEdge()) {
        continue;
      }

      Node* incoming_node = incoming_edge->src();
      if (IsRefType(incoming_node->output_type(incoming_edge->src_output()))) {
        VLOG(2) << "Node " << node.def().ShortDebugString() << " has ref input "
                << incoming_node->name() << " " << incoming_node->type_string();
        return true;
      }
    }
  }
  return false;
}

StatusOr<bool> CreateCycleDetectionGraph(const Graph* graph,
                                         GraphCycles* cycles) {
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    // We rely on the node IDs in the cycle detection graph being consecutive
    // integers starting from 0.
    CHECK_EQ(i, cycles->NewNode());
  }

  // Compute the loop structure of the graph.
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &control_flow_info));

  // The clustering code must avoid adding cycles to the graph to prevent
  // deadlock. However, the graph may contain loops, which would trigger the
  // cycle detection code. To handle loops, we alter the structure of the cycle
  // detection graph, disconnecting each loop from the enclosing graph.
  // Specifically, we:
  // * add a new "frame" node for each loop.
  // * replace edges to "Enter" nodes, and edges from "Exit" nodes with edges
  //   to/from the corresponding frame node. In essence, we collapse the loop
  //   into a single node for the purpose of cycle detection in the enclosing
  //   graph.
  // * the body of the loop should now be disconnected from the rest of the
  //   graph; we make it acyclic by breaking loop backedges (edges outgoing from
  //   "NextIteration" nodes.

  // Map from frame name strings to node IDs in the cycle detection graph.
  std::unordered_map<string, int> frame_nodes;

  // Get the cycle graph node ID for frame 'frame_name', or add one if none
  // exists.
  auto GetOrAddFrameNodeId = [&frame_nodes, cycles](const string& frame_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_4(mht_4_v, 324, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "lambda");

    int& frame_id = frame_nodes.emplace(frame_name, -1).first->second;
    if (frame_id < 0) {
      // The emplace succeeded; we have not allocated a frame node yet.
      frame_id = cycles->NewNode();
    }
    return frame_id;
  };

  for (Edge const* edge : graph->edges()) {
    if (edge->dst()->IsEnter() || edge->src()->IsExit()) {
      const char* src_type = "pre-enter";
      const char* dst_type = "post-exit";
      int src = edge->src()->id();
      int dst = edge->dst()->id();

      if (edge->dst()->IsEnter()) {
        // Lift edges to an "Enter" node to the corresponding frame node.
        const string& frame_name =
            control_flow_info[edge->dst()->id()].frame_name;
        dst = GetOrAddFrameNodeId(frame_name);
        dst_type = "frame";
      }

      if (edge->src()->IsExit()) {
        // Lift edges from an "Exit" node to the corresponding frame node.
        const string& frame_name =
            control_flow_info[edge->src()->id()].frame_name;
        src = GetOrAddFrameNodeId(frame_name);
        src_type = "frame";
      }

      if (!cycles->InsertEdge(src, dst)) {
        // TODO(b/127521408): We can probably handle this situation with a more
        // sophisticated SCC based algorithm, but for now we bail out.
        VLOG(1) << "Cycle detected when adding " << src_type << "->" << dst_type
                << " edge: " << DescribeCycle(cycles, *graph, src, dst);
        return false;
      }
      // Drop the original edge.
      continue;
    }
    if (edge->src()->IsNextIteration()) {
      // Break loop back-edges.
      continue;
    }
    if (!cycles->InsertEdge(edge->src()->id(), edge->dst()->id())) {
      // This should never happen. All cycles in the graph should contain
      // a control flow operator.
      return errors::Internal(
          "Found cycle in graph without control flow operator during XLA "
          "compilation: ",
          DescribeCycle(cycles, *graph, edge->src()->id(), edge->dst()->id()));
    }
  }

  return true;
}

absl::optional<absl::string_view> GetXlaClusterForNode(const Node& node) {
  const AttrValue* attr_value = node.attrs().Find(kXlaClusterAttr);
  if (attr_value == nullptr) {
    return absl::nullopt;
  }
  Status s = AttrValueHasType(*attr_value, "string");
  if (!s.ok()) {
    return absl::nullopt;
  }
  return attr_value->s();
}

bool HasResourceInputOrOutput(const Node& node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_5(mht_5_v, 398, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "HasResourceInputOrOutput");

  return std::find(node.input_types().begin(), node.input_types().end(),
                   DT_RESOURCE) != node.input_types().end() ||
         std::find(node.output_types().begin(), node.output_types().end(),
                   DT_RESOURCE) != node.output_types().end();
}

void RemoveFromXlaCluster(NodeDef* node_def) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_6(mht_6_v, 408, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "RemoveFromXlaCluster");

  node_def->mutable_attr()->erase(kXlaClusterAttr);
}

void RemoveFromXlaCluster(Node* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_7(mht_7_v, 415, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "RemoveFromXlaCluster");
 node->ClearAttr(kXlaClusterAttr); }

namespace {
typedef xla_config_registry::XlaGlobalJitLevel XlaGlobalJitLevel;

XlaGlobalJitLevel GetXlaGlobalJitLevel(
    const OptimizerOptions::GlobalJitLevel& jit_level_in_session_opts) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_8(mht_8_v, 424, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetXlaGlobalJitLevel");

  XlaGlobalJitLevel result;

  if (jit_level_in_session_opts == OptimizerOptions::DEFAULT) {
    // To set compilation to be on by default, change the following line.
    result.single_gpu = result.general = OptimizerOptions::OFF;
  } else {
    result.single_gpu = result.general = jit_level_in_session_opts;
  }

  // If the flag tf_xla_auto_jit is a valid, non-DEFAULT setting, it overrides
  // the setting in ConfigProto.
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (flags->xla_auto_jit_flag.optimization_level_single_gpu !=
      OptimizerOptions::DEFAULT) {
    result.single_gpu = static_cast<OptimizerOptions::GlobalJitLevel>(
        flags->xla_auto_jit_flag.optimization_level_single_gpu);
  }
  if (flags->xla_auto_jit_flag.optimization_level_general !=
      OptimizerOptions::DEFAULT) {
    result.general = static_cast<OptimizerOptions::GlobalJitLevel>(
        flags->xla_auto_jit_flag.optimization_level_general);
  }

  return result;
}

int GetGpuNumber(const string& device_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_9(mht_9_v, 455, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetGpuNumber");

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return -1;
  }

  return parsed_name.type == DEVICE_GPU ? parsed_name.id : -1;
}
}  // namespace

bool IsSingleGpuGraph(const Graph& g) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_10(mht_10_v, 468, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "IsSingleGpuGraph");

  int gpus_seen = 0;
  absl::flat_hash_set<string> devices_seen;

  for (Node* n : g.op_nodes()) {
    if (devices_seen.contains(n->assigned_device_name())) {
      continue;
    }

    int gpu_number = GetGpuNumber(n->assigned_device_name());
    if (gpu_number != -1) {
      if (++gpus_seen > 1) {
        return false;
      }
    }

    devices_seen.insert(n->assigned_device_name());
  }

  return gpus_seen == 1;
}

OptimizerOptions::GlobalJitLevel GetGlobalJitLevelForGraph(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_11(mht_11_v, 494, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetGlobalJitLevelForGraph");

  OptimizerOptions::GlobalJitLevel jit_level_in_session_opts =
      options.session_options->config.graph_options()
          .optimizer_options()
          .global_jit_level();
  XlaGlobalJitLevel xla_global_jit_level =
      GetXlaGlobalJitLevel(jit_level_in_session_opts);
  if (xla_global_jit_level.single_gpu == xla_global_jit_level.general) {
    VLOG(4) << "GetGlobalJitLevelForGraph returning "
            << xla_global_jit_level.single_gpu;
    return xla_global_jit_level.single_gpu;
  }
  OptimizerOptions::GlobalJitLevel result =
      IsSingleGpuGraph(**options.graph) ? xla_global_jit_level.single_gpu
                                        : xla_global_jit_level.general;
  VLOG(4) << "GetGlobalJitLevelForGraph returning " << result;
  return result;
}

bool MayCallFunction(const Node& n, const FunctionLibraryDefinition* flib_def) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_12(mht_12_v, 516, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "MayCallFunction");

  if (flib_def->Contains(n.type_string())) {
    return true;
  }

  // This is a conservative check: there may be nodes with a `func`
  // attribute that do not make function calls.
  return absl::c_any_of(n.def().attr(),
                        [](const std::pair<string, AttrValue>& name_attr_pair) {
                          return name_attr_pair.second.has_func();
                        });
}
bool IsShapeConsumerOp(const Node& node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_13(mht_13_v, 531, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "IsShapeConsumerOp");

  return node.type_string() == "Shape" || node.type_string() == "Rank" ||
         node.type_string() == "Size";
}

namespace {
struct ClusterInfo {
  int size;

  // Maps op names to the number of times they appear in the cluster.
  absl::flat_hash_map<absl::string_view, int> op_histogram;
};

void HistogramMapToRepeatedOpAndCount(
    protobuf::RepeatedPtrField<XlaAutoClusteringSummary::OpAndCount>* result,
    const absl::flat_hash_map<absl::string_view, int>& histogram) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_14(mht_14_v, 549, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "HistogramMapToRepeatedOpAndCount");

  for (const auto& pair : histogram) {
    XlaAutoClusteringSummary::OpAndCount* new_entry = result->Add();
    new_entry->set_op(std::string(pair.first));
    new_entry->set_count(pair.second);
  }

  absl::c_sort(*result, [](const XlaAutoClusteringSummary::OpAndCount& a,
                           const XlaAutoClusteringSummary::OpAndCount& b) {
    return a.op() < b.op();
  });
}

void ClusterInfoToProtobuf(XlaAutoClusteringSummary::Cluster* result,
                           absl::string_view name, const ClusterInfo& info) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_15(mht_15_v, 567, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "ClusterInfoToProtobuf");

  result->set_name(std::string(name));
  result->set_size(info.size);
  HistogramMapToRepeatedOpAndCount(result->mutable_op_histogram(),
                                   info.op_histogram);
}
}  // namespace

XlaAutoClusteringSummary GetXlaAutoClusteringSummary(const Graph& graph) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_16(mht_16_v, 578, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetXlaAutoClusteringSummary");

  absl::flat_hash_map<absl::string_view, ClusterInfo> cluster_name_to_info;
  XlaAutoClusteringSummary result;

  absl::flat_hash_map<absl::string_view, int> unclustered_op_histogram;

  for (Node* n : graph.nodes()) {
    absl::optional<absl::string_view> cluster_name = GetXlaClusterForNode(*n);
    if (cluster_name) {
      result.set_clustered_node_count(result.clustered_node_count() + 1);
      ClusterInfo* info = &cluster_name_to_info[*cluster_name];
      info->size++;
      info->op_histogram[n->type_string()]++;
    } else {
      result.set_unclustered_node_count(result.unclustered_node_count() + 1);
      unclustered_op_histogram[n->type_string()]++;
    }
  }

  for (const auto& pair : cluster_name_to_info) {
    XlaAutoClusteringSummary::Cluster* new_cluster = result.add_clusters();
    ClusterInfoToProtobuf(new_cluster, pair.first, pair.second);
  }

  absl::c_sort(*result.mutable_clusters(),
               [&](const XlaAutoClusteringSummary::Cluster& a,
                   const XlaAutoClusteringSummary::Cluster& b) {
                 return a.name() < b.name();
               });

  HistogramMapToRepeatedOpAndCount(result.mutable_unclustered_op_histogram(),
                                   unclustered_op_histogram);

  return result;
}

namespace {
using CallTargetListTy = absl::InlinedVector<NameAttrList, 2>;

CallTargetListTy GetCallTargetListFromNode(
    const Node& n, FunctionLibraryRuntime* lib_runtime) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_17(mht_17_v, 621, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetCallTargetListFromNode");

  const FunctionLibraryDefinition& flib_def =
      *lib_runtime->GetFunctionLibraryDefinition();
  if (flib_def.Find(n.type_string())) {
    NameAttrList callee;
    callee.set_name(n.type_string());
    *callee.mutable_attr() = n.def().attr();
    return {callee};
  }

  CallTargetListTy result;
  for (const auto& name_attr_pair : n.attrs()) {
    const AttrValue& attr_value = name_attr_pair.second;
    if (attr_value.value_case() == AttrValue::kFunc) {
      result.push_back(attr_value.func());
    } else if (attr_value.value_case() == AttrValue::kList) {
      result.insert(result.end(), attr_value.list().func().begin(),
                    attr_value.list().func().end());
    }
  }

  return result;
}

enum class Direction { kForward, kBackward };

Status GetNodesRelatedToRefVariablesInDirection(
    const Graph& graph, FunctionLibraryRuntime* lib_runtime,
    Direction direction, int depth, absl::flat_hash_set<Node*>* result);

StatusOr<bool> DoesAnyCalleeHaveRefNodes(
    const CallTargetListTy& call_target_list,
    FunctionLibraryRuntime* lib_runtime, Direction direction, int depth) {
  const int kMaxDepth = 10;

  if (depth == kMaxDepth && !call_target_list.empty()) {
    // Conservative answer to avoid recursing too much.
    return true;
  }

  absl::flat_hash_set<Node*> callee_ref_nodes;
  for (const NameAttrList& call_target : call_target_list) {
    const OpRegistrationData* op_reg;
    if (OpRegistry::Global()->LookUp(call_target.name(), &op_reg).ok()) {
      const OpDef& op = op_reg->op_def;
      if (absl::c_any_of(op.output_arg(), [](const OpDef::ArgDef arg) {
            return arg.is_ref();
          })) {
        return true;
      }
      continue;
    }

    callee_ref_nodes.clear();
    FunctionLibraryRuntime::Handle handle;
    if (!lib_runtime
             ->Instantiate(call_target.name(), AttrSlice(&call_target.attr()),
                           &handle)
             .ok()) {
      VLOG(2) << "Could not find " << call_target.name()
              << " in the function library.";
      // Since we don't know the semantic of `n` we don't know if this is an
      // error.  We return true to signal a conservative answer.
      return true;
    }

    auto release_handle_on_return = gtl::MakeCleanup(
        [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });

    const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
    TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(
        *fbody->graph, lib_runtime, direction, depth + 1, &callee_ref_nodes));

    // We could possibly use something cheaper than
    // GetNodesRelatedToRefVariablesInDirection since we only care about the
    // size of `callee_ref_nodes` but for now we don't ceare.
    if (!callee_ref_nodes.empty()) {
      return true;
    }
  }

  return false;
}

// Helper for GetNodesRelatedToRefVariables that traverses the graph in one
// direction.
Status GetNodesRelatedToRefVariablesInDirection(
    const Graph& graph, FunctionLibraryRuntime* lib_runtime,
    Direction direction, int depth, absl::flat_hash_set<Node*>* result) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_18(mht_18_v, 712, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "GetNodesRelatedToRefVariablesInDirection");

  std::vector<Node*> nodes_in_order;
  if (direction == Direction::kForward) {
    GetReversePostOrder(graph, &nodes_in_order,
                        /*stable_comparator=*/NodeComparatorName());
  } else {
    GetPostOrder(graph, &nodes_in_order,
                 /*stable_comparator=*/NodeComparatorName());
  }

  size_t old_result_size;
  int iterations = 0;

  const int kMaxIterations = 10 * 1000;

  std::vector<bool> callee_has_ref_nodes_cache;
  callee_has_ref_nodes_cache.resize(graph.num_node_ids());

  auto does_callee_have_ref_nodes = [&](Node* n) -> StatusOr<bool> {
    if (iterations == 1) {
      TF_ASSIGN_OR_RETURN(
          bool callee_has_ref_nodes,
          DoesAnyCalleeHaveRefNodes(GetCallTargetListFromNode(*n, lib_runtime),
                                    lib_runtime, direction, depth));
      callee_has_ref_nodes_cache[n->id()] = callee_has_ref_nodes;
      return callee_has_ref_nodes;
    } else {
      return {callee_has_ref_nodes_cache[n->id()]};
    }
  };

  do {
    TF_RET_CHECK(iterations++ < kMaxIterations) << "infinite loop?";

    old_result_size = result->size();
    for (Node* n : nodes_in_order) {
      if (n->IsSource() || n->IsSink()) {
        continue;
      }

      bool inserted_n = false;
      const EdgeSet& edges =
          direction == Direction::kForward ? n->in_edges() : n->out_edges();
      for (const Edge* e : edges) {
        if (result->contains(direction == Direction::kForward ? e->src()
                                                              : e->dst())) {
          result->insert(n);
          inserted_n = true;
          break;
        }
      }

      if (inserted_n) {
        continue;
      }

      if (direction == Direction::kForward &&
          absl::c_any_of(n->output_types(), IsRefType)) {
        result->insert(n);
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool callee_has_ref_nodes,
                          does_callee_have_ref_nodes(n));
      if (callee_has_ref_nodes) {
        result->insert(n);
        continue;
      }
    }

    // Loop until convergence.
  } while (result->size() != old_result_size);

  VLOG(2) << "# iterations = " << iterations;

  return Status::OK();
}

// Sorts control inputs of a graphdef so that they are deterministically
// ordered.
void SortControlInputs(GraphDef* gdef) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_utilDTcc mht_19(mht_19_v, 795, "", "./tensorflow/compiler/jit/xla_cluster_util.cc", "SortControlInputs");

  int64_t num_nodes = gdef->node_size();
  for (int64_t i = 0; i < num_nodes; ++i) {
    NodeDef* node = gdef->mutable_node(i);
    // Stable sort control inputs and leave the order of data inputs unchanged.
    std::stable_sort(node->mutable_input()->begin(),
                     node->mutable_input()->end(),
                     [](const string& a, const string& b) {
                       bool a_is_control = absl::StartsWith(a, "^");
                       bool b_is_control = absl::StartsWith(b, "^");
                       return (!a_is_control && b_is_control) ||
                              (a_is_control && b_is_control && a < b);
                     });
  }
}
}  // namespace

StatusOr<absl::flat_hash_set<Node*>> GetNodesRelatedToRefVariables(
    const Graph& graph, FunctionLibraryRuntime* lib_runtime) {
  absl::flat_hash_set<Node*> result;
  TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(
      graph, lib_runtime, Direction::kForward, 0, &result));
  TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(
      graph, lib_runtime, Direction::kBackward, 0, &result));

  VLOG(1) << "GetNodesRelatedToRefVariables() found " << result.size()
          << " nodes";
  return result;
}

StatusOr<std::string> SerializeGraphDeterministic(const Graph& graph) {
  GraphDef def;
  graph.ToGraphDef(&def);

  // Before serialization, sort each node's control inputs to achieve
  // determinism. Sorting control inputs could help (but not necessarily) create
  // a deterministic serialization and fingerprint. Other sources of
  // nondeterminism include unstable node ordering.
  SortControlInputs(&def);

  std::string s;
  if (!SerializeToStringDeterministic(def, &s)) {
    return errors::Internal("Failed to serialize graphdef.");
  }
  return s;
}

StatusOr<uint64> FingerprintGraph(const Graph& graph) {
  TF_ASSIGN_OR_RETURN(std::string serialized,
                      SerializeGraphDeterministic(graph));
  return Hash64(serialized.data(), serialized.size());
}

// Register a callback for querying XlaGlobalJitLevel.
REGISTER_XLA_CONFIG_GETTER(GetXlaGlobalJitLevel);

}  // namespace tensorflow
