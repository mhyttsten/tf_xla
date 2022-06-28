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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

bool IsFunctionCall(const Node& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "IsFunctionCall");

  // TODO(iga): Handle non-PCO functions when we add multi-device support
  // to regular function calls. Also, the GetFunctionDefAndAttrs assumes that
  // the function name is stored in the `f` attribute of the node. That code
  // will need to change as well.
  const string& op_type = node.op_def().name();
  return op_type == "PartitionedCall" || op_type == "StatefulPartitionedCall";
}

// Utility to set node's value in `cache` and `is_deep` to `value`.
Status Set(const Node& node, bool value, bool* is_deep,
           std::vector<absl::optional<bool>>* cache) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "Set");

  *is_deep = value;
  (*cache)[node.id()] = value;
  return Status::OK();
}

}  // namespace

PlacerInspectionRequiredOpChecker::PlacerInspectionRequiredOpChecker(
    const Graph* graph, const FunctionLibraryDefinition* flib_def)
    : graph_(*graph), flib_def_(*flib_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "PlacerInspectionRequiredOpChecker::PlacerInspectionRequiredOpChecker");

  cache_.resize(graph_.num_node_ids());
}

Status PlacerInspectionRequiredOpChecker::IsPlacerInspectionRequired(
    const Node& node, bool* is_deep) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "PlacerInspectionRequiredOpChecker::IsPlacerInspectionRequired");

  if (cache_[node.id()].has_value()) {
    *is_deep = cache_[node.id()].value();
    return Status::OK();
  }

  if (!IsFunctionCall(node)) {
    return Set(node, false, is_deep, &cache_);
  }
  const FunctionDef* fdef;
  NameAttrList func;
  TF_RETURN_IF_ERROR(GetFunctionDefAndAttrs(flib_def_, node, &fdef, &func));
  DataTypeVector types;
  TF_RETURN_IF_ERROR(
      OutputTypesForNode(AttrSlice(&func.attr()), fdef->signature(), &types));
  for (DataType type : types) {
    if (type == DT_RESOURCE) {
      return Set(node, true, is_deep, &cache_);
    }
  }
  return Set(node, false, is_deep, &cache_);
}

Status GetFunctionDefAndAttrs(const FunctionLibraryDefinition& flib_def,
                              const Node& node, const FunctionDef** fdef,
                              NameAttrList* func) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_4(mht_4_v, 263, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "GetFunctionDefAndAttrs");

  TF_RETURN_IF_ERROR(GetNodeAttr(node.def(), "f", func));
  const string& function_name = func->name();
  *fdef = flib_def.Find(function_name);
  if (*fdef == nullptr) {
    return errors::InvalidArgument(
        "Failed to find function \"", function_name,
        "\" in function library: ", flib_def.ToProto().DebugString());
  }
  return Status::OK();
}

FunctionStack::FunctionStack(const string& function_name)
    : current_function_name_(function_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "FunctionStack::FunctionStack");
}

FunctionStack FunctionStack::Push(const Node* node_in_current_function,
                                  const string& new_current_function) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("new_current_function: \"" + new_current_function + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_6(mht_6_v, 287, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "FunctionStack::Push");

  FunctionStack new_stack(new_current_function);
  new_stack.frames_ = frames_;
  new_stack.frames_.emplace_back(current_function_name_,
                                 node_in_current_function);
  return new_stack;
}

bool FunctionStack::HasFunction(const string& function_name) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_7(mht_7_v, 299, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "FunctionStack::HasFunction");

  if (current_function_name_ == function_name) {
    return true;
  }
  for (const Frame& frame : frames_) {
    if (frame.function_name == function_name) {
      return true;
    }
  }
  return false;
}

string FunctionStack::FormatForError() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_8(mht_8_v, 314, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "FunctionStack::FormatForError");

  std::vector<string> msgs;
  for (int i = 0; i < frames_.size(); ++i) {
    if (frames_[i].function_name.empty()) {
      // Empty function body should only happen at the top level, i.e. i = 0.
      // All internal frames should have valid function names.
      msgs.push_back(absl::StrCat("Graph contains node ",
                                  FormatNodeForError(*frames_[i].node)));

    } else {
      msgs.push_back(absl::StrCat(
          "Function ", errors::FormatFunctionForError(frames_[i].function_name),
          " contains node ", FormatNodeForError(*frames_[i].node)));
    }
    const string& fname = (i + 1 < frames_.size())
                              ? frames_[i + 1].function_name
                              : current_function_name_;
    msgs.push_back(absl::StrCat("Node ", FormatNodeForError(*frames_[i].node),
                                " calls function ",
                                errors::FormatFunctionForError(fname)));
  }
  return absl::StrJoin(msgs, "\n  ");
}

namespace {

using OutputEdgeMap = std::vector<std::vector<const Edge*>>;

constexpr char kIdentityOp[] = "Identity";

string Uniquify(const string& candidate_name,
                std::unordered_set<string>* node_names) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("candidate_name: \"" + candidate_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_9(mht_9_v, 349, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "Uniquify");

  if (node_names->find(candidate_name) == node_names->end()) {
    node_names->insert(candidate_name);
    return candidate_name;
  }

  for (int counter = 0;; ++counter) {
    string candidate = absl::StrCat(candidate_name, "_", counter);
    if (node_names->find(candidate) == node_names->end()) {
      node_names->insert(candidate);
      return candidate;
    }
  }
}

Status AddInputIdentity(Node* node, int input_idx, Graph* graph,
                        std::unordered_set<string>* node_names) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_10(mht_10_v, 368, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "AddInputIdentity");

  const Edge* edge;
  TF_RETURN_IF_ERROR(node->input_edge(input_idx, &edge));

  string identity_name = Uniquify(
      absl::StrCat(edge->src()->name(), "_", node->name()), node_names);

  NodeDefBuilder builder(identity_name, kIdentityOp);
  builder.Attr("T", node->input_type(input_idx));
  NodeDefBuilder::NodeOut input(edge->src()->name(), edge->src_output(),
                                node->input_type(input_idx));
  builder.Input(input);
  NodeDef identity_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&identity_def));
  MergeDebugInfo(NodeDebugInfo(*node), &identity_def);

  VLOG(6) << "Adding identity into " << edge->src()->name() << ":"
          << edge->src_output() << " -> " << edge->dst()->name() << ":"
          << input_idx << " \n"
          << identity_def.DebugString();

  TF_ASSIGN_OR_RETURN(Node * identity_node, graph->AddNode(identity_def));
  graph->AddEdge(edge->src(), edge->src_output(), identity_node, 0);

  // Replace node's `input_idx` input with the new identity's 0'th output
  TF_RETURN_IF_ERROR(graph->UpdateEdge(identity_node, 0, node, input_idx));

  VLOG(6) << "Successfully inserted identity. Modified node: \n"
          << node->DebugString();
  return Status::OK();
}

struct EdgePtrCompare {
  bool operator()(const Edge* lhs, const Edge* rhs) const {
    return lhs->id() < rhs->id();
  }
};

Status AddOutputIdentities(Node* node, Graph* graph,
                           std::unordered_set<string>* node_names) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_11(mht_11_v, 410, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "AddOutputIdentities");

  auto add_identity = [&](int src_output, const string& identity_name,
                          Node** identity_node) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("identity_name: \"" + identity_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_12(mht_12_v, 416, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "lambda");

    NodeDefBuilder builder(identity_name, kIdentityOp);
    builder.Attr("T", node->output_type(src_output));
    NodeDefBuilder::NodeOut input(node->name(), src_output,
                                  node->output_type(src_output));
    builder.Input(input);
    NodeDef identity_def;
    TF_RETURN_IF_ERROR(builder.Finalize(&identity_def));
    MergeDebugInfo(NodeDebugInfo(*node), &identity_def);

    TF_ASSIGN_OR_RETURN(*identity_node, graph->AddNode(identity_def));
    graph->AddEdge(node, src_output, *identity_node, 0);
    return Status::OK();
  };

  // output_used[i] == true iff `node`'s i'th output is used
  // in this graph
  std::vector<bool> output_used(node->num_outputs(), false);
  // Copy the set of edges since EdgeSet does not allow modifications
  // to graph edges during iteration.
  const EdgeSet& out_edges = node->out_edges();
  std::vector<const Edge*> edge_vector(out_edges.begin(), out_edges.end());
  std::sort(edge_vector.begin(), edge_vector.end(), EdgePtrCompare());
  for (const Edge* edge : edge_vector) {
    if (edge->IsControlEdge()) {
      continue;
    }
    output_used[edge->src_output()] = true;

    Node* dst = edge->dst();
    int dst_input = edge->dst_input();
    int src_output = edge->src_output();
    string identity_name =
        Uniquify(absl::StrCat(node->name(), "_", dst->name()), node_names);
    Node* identity_node;
    TF_RETURN_IF_ERROR(add_identity(src_output, identity_name, &identity_node));
    VLOG(6) << "Adding identity into " << node->name() << ":" << src_output
            << " -> " << dst->name() << ":" << dst_input << " \n"
            << identity_node->DebugString();

    // Make original dst node consume the new identity's output instead of
    // `node`'s output.
    TF_RETURN_IF_ERROR(graph->UpdateEdge(identity_node, 0, dst, dst_input));
  }

  for (int output_idx = 0; output_idx < node->num_outputs(); ++output_idx) {
    if (output_used[output_idx]) {
      continue;
    }
    // The output is unused in the graph. Just add an identity
    // consuming it.
    string identity_name = Uniquify(node->name(), node_names);
    Node* identity_node;
    TF_RETURN_IF_ERROR(add_identity(output_idx, identity_name, &identity_node));
    VLOG(6) << "Added identity into " << node->name() << ":" << output_idx
            << " -> <no consumer>: \n"
            << identity_node->DebugString();
  }
  return Status::OK();
}

Status IsolateNode(Node* node, Graph* graph) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_13(mht_13_v, 480, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "IsolateNode");

  // We use `node_names` to make sure we pick unique names.
  // We don't use graph->NewName() because it produces verbose names and
  // does not actually ensure that they are unique (it assumes all names
  // are generated using it, which is not true today).
  std::unordered_set<string> node_names(graph->num_nodes());
  for (Node* n : graph->nodes()) {
    node_names.insert(n->name());
  }

  for (int i = 0; i < node->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(AddInputIdentity(node, i, graph, &node_names));
  }
  TF_RETURN_IF_ERROR(AddOutputIdentities(node, graph, &node_names));
  return Status::OK();
}

}  // namespace

Status IsolatePlacerInspectionRequiredOps(
    const FunctionLibraryDefinition& flib_def, Graph* graph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacer_inspection_required_ops_utilsDTcc mht_14(mht_14_v, 503, "", "./tensorflow/core/common_runtime/placer_inspection_required_ops_utils.cc", "IsolatePlacerInspectionRequiredOps");

  PlacerInspectionRequiredOpChecker checker(graph, &flib_def);
  // It is OK to add nodes to the graph during iteration.
  // New nodes will get ids above current ids. The loop
  // will loop over current nodes only because the op_nodes()
  // iterator uses node ids to iterate.
  // Because the new nodes will be higher ids, the caching in
  // the checker will also work fine as new nodes are added.
  for (Node* node : graph->op_nodes()) {
    bool should_be_isolated = false;
    TF_RETURN_IF_ERROR(
        checker.IsPlacerInspectionRequired(*node, &should_be_isolated));
    if (!should_be_isolated) {
      continue;
    }
    TF_RETURN_IF_ERROR(IsolateNode(node, graph));
  }

  return Status::OK();
}

}  // namespace tensorflow
