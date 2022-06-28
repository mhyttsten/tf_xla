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
class MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc() {
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

#include "tensorflow/compiler/jit/partially_decluster_pass.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

bool NotBackedge(const Edge& edge) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "NotBackedge");
 return !edge.src()->IsNextIteration(); }

namespace reduce_device_to_host_copies {
Status FindNodesToDecluster(const Graph& graph,
                            absl::flat_hash_set<Node*>* result,
                            absl::Span<Node* const> post_order) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "FindNodesToDecluster");

  // Find nodes that have at least one user outside their cluster that expects
  // hostmem output.  These nodes should be cloned to outside the cluster to
  // avoid the device-host copy we'd otherwise need.

  MemoryTypeVector input_mtypes, output_mtypes;

  for (Node* n : post_order) {
    absl::optional<absl::string_view> from_cluster = GetXlaClusterForNode(*n);
    if (!from_cluster) {
      continue;
    }

    // Assume the benefit of not outputting a larger tensor outweighs the
    // benefit of this check.
    // TODO(tpopp): Only apply this if the value being consumed is not output
    // from the cluster to another consumer.
    // TODO(tpopp): See if XlaRun can be modified to avoid this issue
    // completely.
    if (IsShapeConsumerOp(*n)) {
      continue;
    }
    // We assume the only XLA-auto-clusterable operations with side effects are
    // resource variable updates.  We can't execute these twice.
    if (HasResourceInputOrOutput(*n)) {
      continue;
    }

    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
    TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                          n->def(), &input_mtypes,
                                          &output_mtypes));
    for (const Edge* e : n->out_edges()) {
      Node* dst = e->dst();

      if (e->IsControlEdge()) {
        continue;
      }

      bool edge_incurs_extra_device_to_host_copy;
      if (output_mtypes[e->src_output()] == DEVICE_MEMORY) {
        // If the output of the *TensorFlow* operation is in DEVICE_MEMORY then
        // keep the node clustered -- XLA will also produce the output in device
        // memory and we will get some benefit from clustering.
        edge_incurs_extra_device_to_host_copy = false;
      } else {
        MemoryTypeVector dst_input_mtypes, dst_output_mtypes;
        DeviceType dst_device_type("");
        TF_RETURN_IF_ERROR(DeviceNameToDeviceType(dst->assigned_device_name(),
                                                  &dst_device_type));
        TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                              dst->def(), &dst_input_mtypes,
                                              &dst_output_mtypes));
        edge_incurs_extra_device_to_host_copy =
            dst_input_mtypes[e->dst_input()] == HOST_MEMORY;
      }

      if (!edge_incurs_extra_device_to_host_copy) {
        continue;
      }

      // Check if `dst` is in a different cluster, unclustered, or about to be
      // partially declustered (here we rely on the post-order traversal order).
      // If yes, decluster `n` to avoid the device-to-host memcpy.
      absl::optional<absl::string_view> dst_cluster =
          result->count(dst) ? absl::nullopt : GetXlaClusterForNode(*dst);
      if (from_cluster != dst_cluster) {
        CHECK(result->insert(n).second);
        break;
      }
    }
  }
  return Status::OK();
}

Status PartiallyDeclusterNode(Graph* graph, Node* n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_2(mht_2_v, 294, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "PartiallyDeclusterNode");

  absl::string_view cluster_name = *GetXlaClusterForNode(*n);
  absl::InlinedVector<const Edge*, 6> out_edges_to_clone;
  for (const Edge* out_edge : n->out_edges()) {
    if (out_edge->IsControlEdge()) {
      continue;
    }

    Node* dst = out_edge->dst();
    absl::optional<absl::string_view> dst_cluster_name =
        GetXlaClusterForNode(*dst);
    if (dst_cluster_name != cluster_name) {
      out_edges_to_clone.push_back(out_edge);
    }
  }

  CHECK(!out_edges_to_clone.empty()) << n->DebugString();

  NodeDef ndef = n->def();
  ndef.set_name(absl::StrCat(n->name(), "/declustered"));
  MergeDebugInfo(NodeDebugInfo(n->def()), &ndef);
  RemoveFromXlaCluster(&ndef);
  TF_ASSIGN_OR_RETURN(Node * cloned_node, graph->AddNode(ndef));
  cloned_node->set_assigned_device_name(n->assigned_device_name());

  for (const Edge* in_edge : n->in_edges()) {
    graph->AddEdge(in_edge->src(), in_edge->src_output(), cloned_node,
                   in_edge->dst_input());
  }

  for (const Edge* out_edge_to_clone : out_edges_to_clone) {
    graph->AddEdge(cloned_node, out_edge_to_clone->src_output(),
                   out_edge_to_clone->dst(), out_edge_to_clone->dst_input());
    graph->RemoveEdge(out_edge_to_clone);
  }

  if (n->out_edges().empty()) {
    graph->RemoveNode(n);
  }

  return Status::OK();
}

// Clones nodes to outside their cluster to avoid device-to-host copies.  For
// instance, converts this:
//
//         .....
//           |
//           v
//      A_Clustered ====> C_Unclustered
//           |
//           v
//      B_Clustered
//
// to:
//
//         .....
//          | |
//          | +-------------+
//          |               |
//          v               v
//      A_Clustered   A_Unclustered ====> C_Unclustered
//           |
//           v
//      B_Clustered
//
// where the ===> arrow has a hostmem source and destination and would entail a
// device to host copy if the source and destination were not in the same XLA
// cluster.
Status PartiallyDeclusterGraph(Graph* graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_3(mht_3_v, 366, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "PartiallyDeclusterGraph");

  // When deciding whether to decluster a particular node, we base our decision
  // on if we've decided that some of its consumers have to be declustered too.
  // Iterating the graph in post-order guarantees that consumers have been
  // visited before producers.
  std::vector<Node*> post_order;
  GetPostOrder(*graph, &post_order, /*stable_comparator=*/NodeComparatorName(),
               /*edge_filter=*/NotBackedge);

  absl::flat_hash_set<Node*> nodes_to_partially_decluster;
  TF_RETURN_IF_ERROR(
      FindNodesToDecluster(*graph, &nodes_to_partially_decluster, post_order));

  if (VLOG_IS_ON(3)) {
    for (Node* n : post_order) {
      if (nodes_to_partially_decluster.count(n)) {
        VLOG(3) << n->DebugString();
      }
    }
  }

  for (Node* n : post_order) {
    if (nodes_to_partially_decluster.count(n)) {
      TF_RETURN_IF_ERROR(PartiallyDeclusterNode(graph, n));
    }
  }

  // Recompute post order since PartiallyDeclusterNode may have deleted nodes.
  post_order.clear();
  GetPostOrder(*graph, &post_order, /*stable_comparator=*/NodeComparatorName(),
               /*edge_filter=*/NotBackedge);
  nodes_to_partially_decluster.clear();
  TF_RETURN_IF_ERROR(
      FindNodesToDecluster(*graph, &nodes_to_partially_decluster, post_order));
  CHECK(nodes_to_partially_decluster.empty());

  return Status::OK();
}
}  // namespace reduce_device_to_host_copies

namespace reduce_recompilation {
bool IsIntraClusterEdge(const Edge& edge) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_4(mht_4_v, 410, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "IsIntraClusterEdge");

  absl::optional<absl::string_view> src_cluster_name =
      GetXlaClusterForNode(*edge.src());
  absl::optional<absl::string_view> dst_cluster_name =
      GetXlaClusterForNode(*edge.dst());
  return src_cluster_name.has_value() && src_cluster_name == dst_cluster_name;
}

bool IsMustCompileDevice(const DeviceType& device_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_5(mht_5_v, 421, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "IsMustCompileDevice");

  const XlaOpRegistry::DeviceRegistration* registration;
  if (XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    return registration->autoclustering_policy ==
           XlaOpRegistry::AutoclusteringPolicy::kAlways;
  }

  return false;
}

Status MustCompileNode(const Node* n, bool* must_compile) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_6(mht_6_v, 434, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "MustCompileNode");

  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceNameToDeviceType(n->assigned_device_name(), &device_type));

  if (IsMustCompileDevice(device_type)) {
    *must_compile = true;
    return Status::OK();
  }

  // We must compile `n` if it does not have a TensorFlow kernel.
  *must_compile = !FindKernelDef(device_type, n->def(), nullptr, nullptr).ok();
  return Status::OK();
}

// Declusters nodes to reduce the number of times we think we need to recompile
// a TensorFlow graph.
//
// Abstractly, if we have a cluster of this form:
//
//   x0 = arg0
//   x1 = arg1
//     ...
//   shape = f(x0, x1, ...)
//   result = Reshape(input=<something>, new_shape=shape)
//
// then pulling `f` out of the cluster may reduce the number of compilations and
// will never increase the number of compilations.
//
// We may reduce the number of compilations if f is many to one.  For instance
// if f(x,y) = x-y then x=3,y=1 and x=4,y=2 will generate two different
// compilations if f is in the cluster but only one compilation if f is outside
// the cluster.
//
// Declustering f will increase the number of compilations only if f is a
// one-to-many "function" i.e. isn't a function at all.  RNG is one possible
// example, depending on how we look at it.  But we never create clusters where
// such f's would be marked as must-be-constant.
//
// We assume here that the extra repeated (repeated compared to a clustered f
// where it will always be constant folded) host-side computation of f does not
// regress performance in any significant manner.  We will have to revisit this
// algorithm with a more complex cost model if this assumption turns out to be
// incorrect.
Status PartiallyDeclusterGraph(Graph* graph,
                               const FunctionLibraryDefinition* flib_def,
                               Env* env) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_7(mht_7_v, 483, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "PartiallyDeclusterGraph");

  std::vector<bool> compile_time_const_nodes(graph->num_node_ids());
  OptimizerOptions opts;
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, env, /*config=*/nullptr, TF_GRAPH_DEF_VERSION, flib_def, opts);
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(*graph, nullptr,
                                            &compile_time_const_nodes,
                                            lib_runtime, IsIntraClusterEdge));

  std::vector<Node*> rpo;
  GetReversePostOrder(*graph, &rpo, /*stable_comparator=*/NodeComparatorName(),
                      /*edge_filter=*/NotBackedge);
  for (Node* n : rpo) {
    if (!compile_time_const_nodes[n->id()]) {
      continue;
    }

    absl::string_view cluster_name = *GetXlaClusterForNode(*n);
    bool node_on_cluster_edge =
        absl::c_all_of(n->in_edges(), [&](const Edge* e) {
          absl::optional<absl::string_view> incoming_cluster =
              GetXlaClusterForNode(*e->src());
          return !incoming_cluster || *incoming_cluster != cluster_name;
        });

    // We don't want to decluster F in a graph like
    //
    //   Input -> OP -> Shape -> F -> Reshape
    //
    // Doing so will break up the cluster.  Even if we were okay with breaking
    // up the cluster we will at least have to relabel the two clusters to have
    // different cluster names.
    //
    // We may want to revisit this in the future: we may have cases where OP is
    // a small computation that does not benefit from XLA while XLA can optimize
    // everything that follows the Reshape.  In these cases it may be wise to
    // remove Input, OP, Shape and F from the cluster, if F is a many-to-one
    // function.
    //
    // Note that we do do the right thing for graphs like:
    //
    //   Input -> F0 -> F1 -> Reshape
    //
    // Since we iterate in RPO, we'll first encounter F0, decluster it, then
    // encounter F1, decluster it and so on.
    if (node_on_cluster_edge) {
      bool must_compile_node;
      TF_RETURN_IF_ERROR(MustCompileNode(n, &must_compile_node));
      if (!must_compile_node) {
        if (n->IsConstant()) {
          // We must decluster Const nodes that have an input control edge from
          // a different device, because this node may be part of the
          // co-ordination of while loops between devices.
          for (auto it : n->in_edges()) {
            if (!it->src()->assigned_device_name().empty() &&
                it->src()->assigned_device_name() !=
                    n->assigned_device_name()) {
              VLOG(3) << "Declustering Const with cross-device control input "
                      << n->name();
              RemoveFromXlaCluster(n);
              break;
            }
          }
        } else {
          VLOG(3) << "Declustering must-be-constant node " << n->name();
          RemoveFromXlaCluster(n);
        }
      }
    }
  }

  return Status::OK();
}
}  // namespace reduce_recompilation

namespace decluster_root_shape_consumers {

Status PartiallyDeclusterGraph(Graph* graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_8(mht_8_v, 565, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "PartiallyDeclusterGraph");

  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(*graph, &reverse_post_order,
                      /*stable_comparator=*/NodeComparatorName(),
                      /*edge_filter=*/NotBackedge);

  for (Node* n : reverse_post_order) {
    if (!IsShapeConsumerOp(*n)) {
      continue;
    }

    absl::optional<absl::string_view> cluster = GetXlaClusterForNode(*n);
    if (!cluster.has_value()) {
      continue;
    }

    auto input_belongs_to_same_cluster = [&](const Edge* e) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_9(mht_9_v, 584, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "lambda");

      return cluster == GetXlaClusterForNode(*e->src());
    };

    if (absl::c_any_of(n->in_edges(), input_belongs_to_same_cluster)) {
      continue;
    }

    VLOG(2) << "Declustering " << n->name()
            << " because it is a root shape consumer";
    RemoveFromXlaCluster(n);
  }
  return Status::OK();
}
}  // namespace decluster_root_shape_consumers
}  // namespace

Status PartiallyDeclusterPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSpartially_decluster_passDTcc mht_10(mht_10_v, 605, "", "./tensorflow/compiler/jit/partially_decluster_pass.cc", "PartiallyDeclusterPass::Run");

  // NB!  In this pass we assume the only XLA-auto-clusterable operations that
  // may have side effects are resource variable operations so we don't cluster
  // those.  The pass will have to be updated if this assumption becomes
  // invalid.

  Graph* graph = options.graph->get();

  TF_RETURN_IF_ERROR(
      reduce_device_to_host_copies::PartiallyDeclusterGraph(graph));
  if (options.flib_def == nullptr) {
    return errors::InvalidArgument(
        "GraphOptimizationPassOptions::flib_def must be set for "
        "PartiallyDeclusterPass.");
  }
  if (options.session_options == nullptr ||
      options.session_options->env == nullptr) {
    return errors::InvalidArgument(
        "GraphOptimizationPassOptions::session_options::env must be set for "
        "PartiallyDeclusterPass.");
  }
  TF_RETURN_IF_ERROR(reduce_recompilation::PartiallyDeclusterGraph(
      graph, options.flib_def, options.session_options->env));

  TF_RETURN_IF_ERROR(
      decluster_root_shape_consumers::PartiallyDeclusterGraph(graph));

  return Status::OK();
}
}  // namespace tensorflow
