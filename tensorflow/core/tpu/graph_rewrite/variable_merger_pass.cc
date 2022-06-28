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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc() {
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
#include "tensorflow/core/tpu/graph_rewrite/variable_merger_pass.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

// The name of a stateful op is semantically meaningful because ops with the
// same name will share the same kernel. We therefore form new op names using a
// deterministic function (a fingerprint) of the old names.
uint64 MergedOpFingerprint(absl::Span<Node* const> ops) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/tpu/graph_rewrite/variable_merger_pass.cc", "MergedOpFingerprint");

  std::vector<string> op_names;
  op_names.reserve(ops.size());
  for (const Node* node : ops) {
    op_names.push_back(node->name());
  }
  return Fingerprint64(absl::StrJoin(op_names, ","));
}

Status MergeVarHandleOps(const string& device, absl::Span<Node* const> nodes,
                         Graph* graph) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/tpu/graph_rewrite/variable_merger_pass.cc", "MergeVarHandleOps");

  int num_var_handles(nodes.size());
  if (num_var_handles <= 1) return Status::OK();

  std::vector<string> containers(num_var_handles);
  std::vector<string> names(num_var_handles);
  DataTypeVector dtypes(num_var_handles);
  std::vector<PartialTensorShape> shapes(num_var_handles);
  for (int i = 0; i < num_var_handles; ++i) {
    TF_RETURN_IF_ERROR(
        GetNodeAttr(nodes[i]->attrs(), "container", &containers[i]));
    TF_RETURN_IF_ERROR(
        GetNodeAttr(nodes[i]->attrs(), "shared_name", &names[i]));
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "dtype", &dtypes[i]));
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "shape", &shapes[i]));
  }
  NodeDefBuilder builder(graph->NewName(strings::StrCat(
                             "VarHandles_", MergedOpFingerprint(nodes))),
                         "_VarHandlesOp");
  builder.Attr("N", num_var_handles);
  builder.Attr("containers", containers);
  builder.Attr("shared_names", names);
  builder.Attr("dtypes", dtypes);
  builder.Attr("shapes", shapes);
  builder.Device(device);
  NodeDef node_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&node_def));
  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(node_def));
  node->set_assigned_device_name(device);

  graph->AddControlEdge(graph->source_node(), node);
  for (int i = 0; i < num_var_handles; ++i) {
    std::vector<std::pair<Node*, int>> consumers;
    for (const Edge* e : nodes[i]->out_edges()) {
      consumers.emplace_back(e->dst(), e->dst_input());
    }
    graph->RemoveNode(nodes[i]);
    for (const auto& t : consumers) {
      graph->AddEdge(node, t.second < 0 ? -1 : i, t.first, t.second);
    }
  }
  return Status::OK();
}

Status MergeReadVariableOps(Node* handle_op, Node* control_node,
                            absl::Span<Node* const> nodes, Graph* graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/tpu/graph_rewrite/variable_merger_pass.cc", "MergeReadVariableOps");

  int num_reads(nodes.size());
  if (num_reads <= 1) return Status::OK();

  DataTypeVector dtypes(num_reads);
  for (int i = 0; i < num_reads; ++i) {
    TF_RETURN_IF_ERROR(GetNodeAttr(nodes[i]->attrs(), "dtype", &dtypes[i]));
  }
  NodeDef node_def;
  node_def.set_name(graph->NewName(
      strings::StrCat("ReadVariables_", MergedOpFingerprint(nodes))));
  node_def.set_op("_ReadVariablesOp");
  AddNodeAttr("N", num_reads, &node_def);
  AddNodeAttr("dtypes", dtypes, &node_def);
  node_def.set_device(handle_op->requested_device());
  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(node_def));
  node->set_assigned_device_name(handle_op->assigned_device_name());
  if (control_node) graph->AddControlEdge(control_node, node);
  for (int i = 0; i < num_reads; ++i) {
    const Edge* handle_edge;
    TF_RETURN_IF_ERROR(nodes[i]->input_edge(0, &handle_edge));
    graph->AddEdge(handle_edge->src(), handle_edge->src_output(), node, i);

    std::vector<std::pair<Node*, int>> consumers;
    for (const Edge* e : nodes[i]->out_edges()) {
      consumers.emplace_back(e->dst(), e->dst_input());
    }
    graph->RemoveNode(nodes[i]);
    for (const auto& t : consumers) {
      graph->AddEdge(node, t.second < 0 ? -1 : i, t.first, t.second);
    }
  }
  return Status::OK();
}

}  // namespace

Status VariableMergerPass::Run(const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc mht_3(mht_3_v, 311, "", "./tensorflow/core/tpu/graph_rewrite/variable_merger_pass.cc", "VariableMergerPass::Run");

  Graph* graph = options.graph->get();

  VLOG(1) << DumpGraphToFile("variable_merger_pass_before", *graph);

  // Find VarHandleOps that are graph roots and group them by assigned device.
  // Also find any ReadVariableOps that are consumers of those handles.
  absl::flat_hash_map<string, std::vector<Node*>> var_handle_ops_by_device;
  absl::flat_hash_set<Node*> read_variable_ops;

  for (Node* m : graph->source_node()->out_nodes()) {
    // We check that the VarHandleOp has no control edges, other than the one we
    // followed from the source node.
    if (m->type_string() == "VarHandleOp" && m->in_edges().size() == 1) {
      var_handle_ops_by_device[m->assigned_device_name()].push_back(m);
      for (Node* n : m->out_nodes()) {
        // ReadVariableOp could have control edges, we will group them by
        // merged VarHandleOp and control dependency.
        if (n->type_string() == "ReadVariableOp" && n->in_edges().size() <= 2) {
          read_variable_ops.insert(n);
        }
      }
    }
  }

  auto node_name_comparator = [](Node* a, Node* b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSvariable_merger_passDTcc mht_4(mht_4_v, 339, "", "./tensorflow/core/tpu/graph_rewrite/variable_merger_pass.cc", "lambda");

    return a->name() < b->name();
  };

  // First merge the var handle ops.
  for (auto& vh : var_handle_ops_by_device) {
    // Sort the handles by name for determinism.
    std::sort(vh.second.begin(), vh.second.end(), node_name_comparator);
    TF_RETURN_IF_ERROR(MergeVarHandleOps(vh.first, vh.second, graph));
  }

  // ReadVariableOps by a pair of <VarHandleOp, ControlDependencyNode>.
  // ControlDependencyNode could be nullptr.
  absl::flat_hash_map<std::pair<Node*, Node*>, std::vector<Node*>> read_var_ops;

  for (Node* n : read_variable_ops) {
    Node* control_node = nullptr;
    Node* var_handle_op = nullptr;
    // Each ReadVariableOp has at most one control input since we only choose
    // ReadVariableOp with at most 2 input edges.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        control_node = e->src();
      } else {
        var_handle_op = e->src();
      }
    }
    TF_RET_CHECK(var_handle_op != nullptr);
    read_var_ops[std::pair<Node*, Node*>(var_handle_op, control_node)]
        .push_back(n);
  }

  for (auto& r : read_var_ops) {
    // Sort the reads by name for determinism.
    std::sort(r.second.begin(), r.second.end(), node_name_comparator);
    TF_RETURN_IF_ERROR(
        MergeReadVariableOps(r.first.first, r.first.second, r.second, graph));
  }

  VLOG(1) << DumpGraphToFile("variable_merger_pass_after", *graph);
  return Status::OK();
}

}  // namespace tensorflow
