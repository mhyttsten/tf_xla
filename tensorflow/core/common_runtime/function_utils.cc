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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc() {
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

#include "tensorflow/core/common_runtime/function_utils.h"

#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

static constexpr const char* const kNodeLabel = "Func";

// Represents the index-th output of a node.
struct Endpoint {
  Node* node;
  int index;

  // Returns the string name represents this endpoint.
  string name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/common_runtime/function_utils.cc", "name");

    if (index == 0) {
      return node->name();
    } else {
      return strings::StrCat(node->name(), ":", index);
    }
  }

  DataType dtype() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/common_runtime/function_utils.cc", "dtype");
 return node->output_type(index); }
};

// The following Add* routines are used to add a few graph nodes while
// functions are transformed.
static Node* AddNoOp(StringPiece name, Graph* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/common_runtime/function_utils.cc", "AddNoOp");

  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("NoOp");
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

static Node* AddIdentity(StringPiece name, Graph* g, Endpoint input) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/common_runtime/function_utils.cc", "AddIdentity");

  DCHECK_LT(0, input.dtype());
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("Identity");
  ndef.add_input(input.name());
  AddNodeAttr("T", BaseType(input.dtype()), &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

void DumpGraph(StringPiece label, const Graph* g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/common_runtime/function_utils.cc", "DumpGraph");

  // TODO(zhifengc): Change Graph to record #nodes.
  VLOG(2) << "Graph " << label << " #nodes " << g->num_nodes() << " #edges "
          << g->num_edges();
  if (VLOG_IS_ON(5)) {
    for (const auto& line : str_util::Split(DebugString(g), '\n')) {
      VLOG(5) << "|| " << line;
    }
  }
}

bool RemoveDeadNodes(Graph* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/common_runtime/function_utils.cc", "RemoveDeadNodes");

  VLOG(2) << "Removing dead nodes";
  std::unordered_set<const Node*> nodes;
  for (auto n : g->nodes()) {
    if (n->IsSource() || n->IsSink() || n->IsControlFlow() ||
        n->op_def().is_stateful()) {
      nodes.insert(n);
    }
  }
  return PruneForReverseReachability(g, std::move(nodes));
}

namespace {
// If 'edges' contains only 1 non-control edge, returns it. Otherwise,
// returns a nullptr.
const Edge* GetTheOnlyDataEdge(const EdgeSet& edges) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_6(mht_6_v, 289, "", "./tensorflow/core/common_runtime/function_utils.cc", "GetTheOnlyDataEdge");

  const Edge* ret = nullptr;
  for (const Edge* e : edges) {
    if (e->IsControlEdge() || ret) {
      // Don't touch it if there is a control edge.
      return nullptr;
    }
    if (IsRefType(e->src()->output_type(e->src_output()))) {
      // Don't touch it if the identity node is effectively de-reffing
      // a ref.
      return nullptr;
    }
    if (IsRecv(e->src()) || IsSwitch(e->src())) {
      // Don't touch it if the identity is introduced for control flow.
      // Recv disables all its successors if it receives a dead signal.
      // When Recv has an outgoing control edge, the current executor
      // would not disable the destination. The current solution (see
      // graph_partition.cc) is to add an identity after Recv and change
      // the control edge to be from this identity node. So the identity
      // can't be removed.
      return nullptr;
    }
    ret = e;
  }
  return ret;
}
}  // end namespace

bool RemoveIdentityNodes(Graph* g) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_7(mht_7_v, 320, "", "./tensorflow/core/common_runtime/function_utils.cc", "RemoveIdentityNodes");

  VLOG(2) << "Removing identity nodes";
  bool removed_any = false;
  gtl::InlinedVector<Node*, 8> matches;
  for (Node* n : g->nodes()) {
    if (!n->IsIdentity()) continue;
    if (!GetTheOnlyDataEdge(n->in_edges())) continue;

    // Some identity nodes are used as sink nodes to give names to output
    // tensors. These nodes are not going to be executed unless they are in the
    // fetch set. But if they are in the fetch set we don't want to remove them.
    if (n->out_edges().empty()) continue;

    matches.push_back(n);
  }
  if (!matches.empty()) {
    for (Node* n : matches) {
      const Edge* in = GetTheOnlyDataEdge(n->in_edges());
      for (const Edge* out : n->out_edges()) {
        if (out->IsControlEdge()) {
          g->AddControlEdge(in->src(), out->dst());
        } else {
          g->AddEdge(in->src(), in->src_output(), out->dst(), out->dst_input());
        }
      }
      VLOG(2) << "Remove Identity: " << n->DebugString();
      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

bool RemoveListArrayConverter(Graph* g) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_8(mht_8_v, 356, "", "./tensorflow/core/common_runtime/function_utils.cc", "RemoveListArrayConverter");

  VLOG(2) << "Removing list array converter";
  gtl::InlinedVector<Node*, 8> matches;
  for (Node* n : g->nodes()) {
    if ((n->type_string() == "_ListToArray") ||
        (n->type_string() == "_ArrayToList")) {
      matches.push_back(n);
    }
  }
  bool removed_any = false;
  if (!matches.empty()) {
    for (Node* n : matches) {
      if (n->num_inputs() != n->num_outputs()) {
        continue;  // Not expected. Skip.
      }
      gtl::InlinedVector<Node*, 8> identity_nodes(n->num_inputs(), nullptr);

      const auto no_op = [&](StringPiece name) -> Node* {
        return AddNoOp(absl::StrCat(n->name(), "/", name), g);
      };

      const auto identity = [&](StringPiece name, Endpoint input) -> Node* {
        Node* node = AddIdentity(absl::StrCat(n->name(), "/", name), g, input);
        node->set_requested_device(input.node->def().device());
        return node;
      };

      // Process input edges first.
      Node* input_control_node = nullptr;
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) {
          if (input_control_node == nullptr) {
            // If node "n" has any control dependencies, adds a no-op
            // node (input_control_node) which the additional Identity
            // nodes depends on and the input_control_node depends on
            // the node "n"s control dependencies.
            input_control_node = no_op("input_control_node");
          }
          g->AddControlEdge(e->src(), input_control_node);
        } else {
          const int index = e->dst_input();
          Node** id_node = &identity_nodes[index];
          if (*id_node != nullptr) {
            LOG(ERROR)
                << "RemoveListArrayConverter unexpected duplicated input: "
                << e->dst_input();
            return removed_any;
          }
          *id_node = identity("input", {e->src(), e->src_output()});
        }
      }

      // If node "n" has any control dependencies, the added identity
      // nodes should have control dependencies on input_control_node.
      if (input_control_node != nullptr) {
        for (Node* id : identity_nodes) {
          g->AddControlEdge(input_control_node, id);
        }
      }

      Node* output_control_node = nullptr;
      for (const Edge* e : n->out_edges()) {
        if (e->IsControlEdge()) {
          if (output_control_node == nullptr) {
            // If node "n" is control-depended upon by other nodes,
            // adds a no-op node (output_control_node) which those
            // nodes will depend on and output_control_node depends on
            // all Identity nodes.
            output_control_node = no_op("output_control_node");
          }
          g->AddControlEdge(output_control_node, e->dst());
        } else {
          Node* id_node = identity_nodes[e->src_output()];
          if (id_node == nullptr) {
            LOG(ERROR) << "RemoveListArrayConverter unexpected missing input: "
                       << e->src_output();
            return removed_any;
          }
          CHECK(id_node);
          g->AddEdge(id_node, 0, e->dst(), e->dst_input());
        }
      }

      // If any nodes have control dependencies on node "n", those
      // nodes should have control dependencies on
      // output_control_node.
      if (output_control_node != nullptr) {
        for (Node* id : identity_nodes) {
          g->AddControlEdge(id, output_control_node);
        }
      }

      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

Status NameAndAttrsFromFunctionCall(const NodeDef& call_def,
                                    NameAttrList* function) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_9(mht_9_v, 459, "", "./tensorflow/core/common_runtime/function_utils.cc", "NameAndAttrsFromFunctionCall");

  if (call_def.op() == "PartitionedCall" ||
      call_def.op() == "StatefulPartitionedCall") {
    TF_RETURN_IF_ERROR(GetNodeAttr(call_def, "f", function));
  } else {
    function->set_name(call_def.op());
    *function->mutable_attr() = call_def.attr();
  }
  return Status::OK();
}

Status InstantiateFunctionCall(const NodeDef& call_def,
                               FunctionLibraryRuntime* flr,
                               FunctionLibraryRuntime::Handle* handle) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_10(mht_10_v, 475, "", "./tensorflow/core/common_runtime/function_utils.cc", "InstantiateFunctionCall");

  NameAttrList function;
  TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(call_def, &function));
  return flr->Instantiate(function.name(), AttrSlice(&function.attr()), handle);
}

bool IsFunctionCall(const FunctionLibraryDefinition& lib_def,
                    const Node& node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_11(mht_11_v, 485, "", "./tensorflow/core/common_runtime/function_utils.cc", "IsFunctionCall");

  return node.IsFunctionCall();
}

string NewName(const Node* n, bool pretty) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_12(mht_12_v, 492, "", "./tensorflow/core/common_runtime/function_utils.cc", "NewName");

  if (pretty) {
    return strings::StrCat(n->type_string(), n->id());
  } else {
    return strings::StrCat("n", n->id());
  }
}

// TODO(zhifengc): Maybe this should be the default Graph::AsGraphDef.
// and stash the original NodeDef name as an attr for documentation
// purpose.
void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_13(mht_13_v, 506, "", "./tensorflow/core/common_runtime/function_utils.cc", "ToGraphDef");

  // We visit nodes in forward topological sort order, which is a
  // possible execution order of the graph.
  gtl::InlinedVector<const Edge*, 4> inputs;
  gdef->Clear();
  *gdef->mutable_versions() = g->versions();

  std::vector<Node*> start_nodes;
  for (Node* n : g->nodes()) {
    if (n->out_edges().empty()) {
      start_nodes.push_back(n);
    }
  }

  ReverseDFSFrom(*g, start_nodes, nullptr, [gdef, pretty, &inputs](Node* n) {
    if (!n->IsOp()) return;
    NodeDef* ndef = gdef->add_node();
    ndef->set_name(NewName(n, pretty));
    ndef->set_op(n->type_string());
    for (const auto& attr : n->attrs()) {
      (*ndef->mutable_attr())[attr.first] = attr.second;
    }

    if (!n->assigned_device_name().empty()) {
      ndef->set_device(n->assigned_device_name());
    } else {
      ndef->set_device(n->requested_device());
    }

    inputs.clear();
    inputs.resize(n->num_inputs());
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        inputs.push_back(e);
      } else {
        if (inputs[e->dst_input()] == nullptr) {
          inputs[e->dst_input()] = e;
        } else {
          LOG(WARNING) << "Malformed graph node. multiple input edges: "
                       << n->DebugString();
        }
      }
    }
    // node->name() is merely NodeDef::name, which are not guaranteed
    // to be unique and stable after optimization rewrites. Therefore,
    // we use "n<node id>" instead.
    for (const Edge* e : inputs) {
      if (e == nullptr) {
        ndef->add_input("unknown");
        continue;
      }
      const string srcname = NewName(e->src(), pretty);
      if (!e->src()->IsOp()) {
      } else if (e->IsControlEdge()) {
        ndef->add_input(strings::StrCat("^", srcname));
      } else if (e->src_output() == 0) {
        ndef->add_input(srcname);
      } else {
        ndef->add_input(strings::StrCat(srcname, ":", e->src_output()));
      }
    }
  });
}

string DebugString(const Graph* g) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_utilsDTcc mht_14(mht_14_v, 573, "", "./tensorflow/core/common_runtime/function_utils.cc", "DebugString");

  GraphDef gdef;
  ToGraphDef(g, &gdef);
  return DebugString(gdef);
}

}  // end namespace tensorflow
