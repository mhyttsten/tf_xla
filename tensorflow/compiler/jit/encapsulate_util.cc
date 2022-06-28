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
class MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc() {
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

#include "tensorflow/compiler/jit/encapsulate_util.h"

#include <algorithm>
#include <iterator>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using stream_executor::port::StatusOr;

namespace tensorflow {

namespace {

// Returns string attribute value for the node if the attribute is present,
// otherwise returns empty optional value.
absl::optional<string> GetStringAttr(const Node& n, const string& attr_name) {
  auto attr = n.attrs().Find(attr_name);
  if (!attr) {
    return absl::nullopt;
  } else {
    return attr->s();
  }
}

// Adds a value to the node's list attribute.
template <typename T>
Status AppendToListAttr(Node* n, const string& attr_name, const string& value) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_0_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "AppendToListAttr");

  std::vector<T> attr_value;
  Status s = GetNodeAttr(n->attrs(), attr_name, &attr_value);
  if (!s.ok() && s.code() != error::NOT_FOUND) {
    return s;
  }

  n->ClearAttr(attr_name);
  attr_value.push_back(value);
  n->AddAttr(attr_name, attr_value);
  return Status::OK();
}

// Replaces attribute value.
template <typename T>
void ReplaceAttr(Node* n, const string& attr_name, const T& value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "ReplaceAttr");

  n->ClearAttr(attr_name);
  n->AddAttr(attr_name, value);
}

// Step 1 for `PreprocessEdgesBetweenOutsideCompilations`. See comments of
// `PreprocessEdgesBetweenOutsideCompilations` for details.
Status PreprocessControlEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PreprocessControlEdgesBetweenOutsideCompilations");

  // Gather edges to remove. We should not remove the edge while iterating.
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* e : g->edges()) {
    if (!e->IsControlEdge()) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (src_outside_compilation && dst_outside_compilation) {
      if (*src_outside_compilation != *dst_outside_compilation) {
        // Case 1a: outside compilation to outside compilation control edge.
        edges_to_remove.push_back(e);

        TF_RETURN_IF_ERROR(AppendToListAttr<string>(
            e->dst(), kXlaControlDependenciesWithinXlaClusterAttrName,
            e->src()->name()));
      }
    } else if (src_outside_compilation && !dst_outside_compilation) {
      // Case 1b: outside compilation to its XLA computation control edge.
      ReplaceAttr(e->src(), kXlaConnectedToXlaComputationAttrName, true);
    } else if (!src_outside_compilation && dst_outside_compilation) {
      // Case 1b: XLA computation to outside compilation in it control edge.
      ReplaceAttr(e->dst(), kXlaConnectedFromXlaComputationAttrName, true);
    }
  }

  for (auto e : edges_to_remove) {
    g->RemoveEdge(e);
  }
  return Status::OK();
}

// Step 2 for `PreprocessEdgesBetweenOutsideCompilations`. See comments of
// `PreprocessEdgesBetweenOutsideCompilations` for details.
Status PreprocessDataEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_3(mht_3_v, 297, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PreprocessDataEdgesBetweenOutsideCompilations");

  // Gather edges between outside compilation and host computation. Notice that
  // we do not store `Edge*` directly because we remove some nodes while adding
  // Identity nodes, and those Edge pointers might be invalidated.
  struct EdgeInfo {
    int dst_input, dst_node_id;
  };
  std::vector<EdgeInfo> edges;
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (src_outside_compilation && dst_outside_compilation &&
        *src_outside_compilation != *dst_outside_compilation) {
      edges.push_back(EdgeInfo{e->dst_input(), e->dst()->id()});
      VLOG(4) << "Oc -> oc edge: " << e->DebugString();
    }
  }

  // Remove the edge from host to outside compilation. Add a placeholder as
  // outside compilation node input.
  std::map<std::pair<string, int>, Node*> placeholders;
  for (int i = 0, end = edges.size(); i < end; i++) {
    Node* dst = g->FindNodeId(edges[i].dst_node_id);
    const Edge* e;
    TF_RETURN_IF_ERROR(dst->input_edge(edges[i].dst_input, &e));
    Node* src = e->src();
    int src_output = e->src_output(), dst_input = e->dst_input();
    g->RemoveEdge(e);

    // Find or create placeholder node.
    string new_name =
        absl::StrCat(src->name(), "_oc_to_oc_placeholder_", src_output);
    auto placeholder_index = std::make_pair(src->name(), src_output);
    auto iter = placeholders.find(placeholder_index);
    Node* placeholder_node;
    if (iter == placeholders.end()) {
      NodeDefBuilder placeholder_builder(new_name, "Placeholder");
      placeholder_builder.Attr("dtype", src->output_type(src_output));
      string outside_compilation_attr;
      TF_RETURN_IF_ERROR(GetNodeAttr(dst->attrs(),
                                     outside_compilation_attr_name,
                                     &outside_compilation_attr));
      placeholder_builder.Attr(outside_compilation_attr_name,
                               outside_compilation_attr);
      placeholder_builder.Attr(kOutsideCompilationOriginalNodeAttrName,
                               src->name());
      placeholder_builder.Attr(kOutsideCompilationSrcOutputAttrName,
                               src_output);
      NodeDef placeholder_def;
      TF_RETURN_IF_ERROR(placeholder_builder.Finalize(&placeholder_def));
      TF_ASSIGN_OR_RETURN(placeholder_node, g->AddNode(placeholder_def));
      placeholders[placeholder_index] = placeholder_node;
    } else {
      placeholder_node = iter->second;
    }
    g->AddEdge(placeholder_node, 0, dst, dst_input);

    // Replace `e->dst()` because its input node changed.
    NodeDef new_def = dst->def();
    *new_def.mutable_input(dst_input) = placeholder_node->name();
    TF_ASSIGN_OR_RETURN(Node * dst_replace_node, ReplaceNode(g, dst, new_def));

    // Other edge in `edges` might have `e->dst()` as src or dst
    // node. Before removing `e->dst()`, replace those edges with
    // corresponding edges for `dst_replace_node`.
    for (int j = i + 1, end = edges.size(); j < end; j++) {
      if (edges[j].dst_node_id == edges[i].dst_node_id) {
        edges[j].dst_node_id = dst_replace_node->id();
      }
    }
  }
  return Status::OK();
}

// Step 1 for `PostprocessEdgesBetweenOutsideCompilations`. See comments of
// `PostprocessEdgesBetweenOutsideCompilations` for details.
Status PostprocessDataEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_4(mht_4_v, 385, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PostprocessDataEdgesBetweenOutsideCompilations");

  // Gather all outside compilation to outside compilation nodes.
  std::vector<Node*> placeholder_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "Placeholder" &&
        HasNodeAttr(n->def(), kOutsideCompilationOriginalNodeAttrName)) {
      placeholder_nodes.push_back(n);
    }
  }

  // Remove the placeholder nodes, and reconnect original edge.
  auto node_name_index = g->BuildNodeNameIndex();
  for (auto n : placeholder_nodes) {
    string node_name;
    int node_src_output;
    TF_RETURN_IF_ERROR(GetNodeAttr(
        n->attrs(), kOutsideCompilationOriginalNodeAttrName, &node_name));
    TF_RETURN_IF_ERROR(GetNodeAttr(
        n->attrs(), kOutsideCompilationSrcOutputAttrName, &node_src_output));
    auto iter = node_name_index.find(node_name);
    if (iter == node_name_index.end()) {
      return errors::Internal(
          "Cannot find original node for oc -> host placeholder node ",
          node_name);
    }

    // Change all usage node to use the original node instead.
    Node* original_node = iter->second;
    std::vector<const Edge*> control_edges;
    std::vector<OutEdgeInfo> data_edges;
    for (auto e : n->out_edges()) {
      if (e->IsControlEdge()) {
        control_edges.push_back(e);
      } else {
        data_edges.push_back({e->dst(), e->src_output(), e->dst_input()});
      }
    }
    for (const Edge* e : control_edges) {
      g->AddControlEdge(original_node, e->dst());
      g->RemoveEdge(e);
    }
    for (int i = 0, end = data_edges.size(); i < end; i++) {
      Node* dst = data_edges[i].dst;
      NodeDef new_def = dst->def();
      int dst_input = data_edges[i].dst_input;
      *new_def.mutable_input(dst_input) =
          absl::StrCat(original_node->name(), ":", node_src_output);
      TF_ASSIGN_OR_RETURN(Node * replace_node, ReplaceNode(g, dst, new_def));

      const Edge* edge_to_replace = nullptr;
      TF_RETURN_IF_ERROR(replace_node->input_edge(dst_input, &edge_to_replace));
      g->RemoveEdge(edge_to_replace);
      g->AddEdge(original_node, node_src_output, replace_node, dst_input);

      // Other edges might have `dst` as dst node. Update those edges with
      // `replace_node`.
      for (int j = i + 1, end = data_edges.size(); j < end; j++) {
        if (data_edges[j].dst == dst) {
          data_edges[j].dst = replace_node;
        }
      }

      // Other placeholder node might have `dst` as original node. Update
      // `node_name_index` with `replace_node`.
      node_name_index[replace_node->name()] = replace_node;
    }

    // Remove placeholder node.
    g->RemoveNode(n);
  }
  return Status::OK();
}

// Step 2 for `PostprocessEdgesBetweenOutsideCompilations`. See comments of
// `PostprocessEdgesBetweenOutsideCompilations` for details.
Status PostprocessControlEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_5(mht_5_v, 465, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PostprocessControlEdgesBetweenOutsideCompilations");

  auto node_name_index = g->BuildNodeNameIndex();

  // Reconnect outside compilation to outside compilation control edge.
  for (Node* n : g->nodes()) {
    std::vector<string> control_deps;
    Status s =
        GetNodeAttr(n->attrs(), kXlaControlDependenciesWithinXlaClusterAttrName,
                    &control_deps);
    if (!s.ok()) {
      if (s.code() != error::NOT_FOUND) {
        return s;
      } else {
        continue;
      }
    } else {
      n->ClearAttr(kXlaControlDependenciesWithinXlaClusterAttrName);
      for (const string& control_input : control_deps) {
        auto iter = node_name_index.find(control_input);
        if (iter == node_name_index.end()) {
          return errors::Internal("Cannot find original node for ",
                                  control_input);
        }
        g->AddControlEdge(iter->second, n);
      }
    }
  }
  return Status::OK();
}
}  // namespace

const char kXlaInferredShapesAttrName[] = "_xla_inferred_shapes";

const char kXlaConnectedToXlaComputationAttrName[] =
    "_xla_connected_to_xla_computation";
const char kXlaConnectedFromXlaComputationAttrName[] =
    "_xla_connected_from_xla_computation";
const char kOutsideCompilationOriginalNodeAttrName[] =
    "_xla_oc_to_oc_node_name";
const char kOutsideCompilationSrcOutputAttrName[] = "_xla_oc_to_oc_src_output";
const char kXlaControlDependenciesWithinXlaClusterAttrName[] =
    "_xla_control_dependencies_within_xla_cluster";
const char kXlaIsLiftedArgAttrName[] = "_xla_is_lifted_arg";
const char kXlaLiftedArgOutsideCompilationAttrName[] = "_xla_lifted_arg_oc";
const char kXlaOutsideCompilationInputsAttrName[] = "_xla_oc_inputs";
const char kXlaIsPlaceholderForArg[] = "_xla_is_placeholder_for_arg";

Status PerformStaticShapeInferenceBeforeEncapsulation(Graph* g) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_6(mht_6_v, 515, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PerformStaticShapeInferenceBeforeEncapsulation");

  // Perform shape inference.
  std::map<int, InferredShape> arg_shapes;
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(
      InferShapes(g, arg_shapes, /*fnlib_def=*/nullptr, &shape_info));

  // Add attribute for output shapes.
  auto node_name_index = g->BuildNodeNameIndex();
  for (auto iter : shape_info) {
    std::vector<PartialTensorShape> output_shapes;
    std::transform(iter.second.begin(), iter.second.end(),
                   std::back_inserter(output_shapes),
                   [](const InferredShape& inferred_shape) {
                     return inferred_shape.shape;
                   });
    Node* n = node_name_index[iter.first];
    n->AddAttr(kXlaInferredShapesAttrName, output_shapes);
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<absl::flat_hash_map<string, std::vector<string>>>>
OutsideCompilationClusterDependencies(
    const Graph* g, const string& outside_compilation_attr_name) {
  auto cluster_deps = absl::make_unique<
      absl::flat_hash_map<string, absl::flat_hash_set<string>>>();

  for (const Edge* e : g->edges()) {
    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (src_outside_compilation && dst_outside_compilation &&
        *src_outside_compilation != *dst_outside_compilation) {
      auto dst_deps_it = cluster_deps->find(*dst_outside_compilation);
      if (dst_deps_it == cluster_deps->end()) {
        cluster_deps->insert(std::make_pair(
            *dst_outside_compilation,
            absl::flat_hash_set<string>({*src_outside_compilation})));
      } else {
        dst_deps_it->second.insert(*src_outside_compilation);
      }
    }
  }

  auto cluster_deps_ordered =
      absl::make_unique<absl::flat_hash_map<string, std::vector<string>>>();

  for (auto it = cluster_deps->begin(); it != cluster_deps->end(); it++) {
    std::vector<string> ordered_deps(it->second.begin(), it->second.end());
    std::sort(ordered_deps.begin(), ordered_deps.end());
    cluster_deps_ordered->insert(std::make_pair(it->first, ordered_deps));
  }

  return std::move(cluster_deps_ordered);
}

Status PreprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_7(mht_7_v, 580, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PreprocessEdgesBetweenOutsideCompilations");

  // Remove edges from source node to outside compilation nodes, and edges
  // from outside compilation nodes to sink node.
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* e : g->source_node()->out_edges()) {
    if (HasNodeAttr(e->dst()->def(), outside_compilation_attr_name)) {
      edges_to_remove.push_back(e);
    }
  }
  for (const Edge* e : g->sink_node()->in_edges()) {
    if (HasNodeAttr(e->src()->def(), outside_compilation_attr_name)) {
      edges_to_remove.push_back(e);
    }
  }
  for (auto e : edges_to_remove) {
    g->RemoveEdge(e);
  }

  TF_RETURN_IF_ERROR(PreprocessControlEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(PreprocessDataEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  return Status::OK();
}

Status PostprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("outside_compilation_attr_name: \"" + outside_compilation_attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_utilDTcc mht_8(mht_8_v, 610, "", "./tensorflow/compiler/jit/encapsulate_util.cc", "PostprocessEdgesBetweenOutsideCompilations");

  TF_RETURN_IF_ERROR(PostprocessDataEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(PostprocessControlEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  return Status::OK();
}

}  // namespace tensorflow
