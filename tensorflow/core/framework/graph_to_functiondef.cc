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
class MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/graph_to_functiondef.h"

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

// Class that maintains a one-to-one original node name -> new node name
// mapping. We normalize the names used as input and output arguments to match
// regexp "[a-z][a-z0-9_]*" specified in definition of ArgDef.name.
// Once we rename them, we risk creating a name collision with the other
// node names, so if necessary we add a suffix to make
// names unique. If we have an input named "A" and a node in the function
// body named "a", they will be renamed to "a" and "a_0".
class NodeNameMapping {
 public:
  NodeNameMapping() = default;

  // Normalize the input name and make it unique. This is the same as the
  // function for output, expect that it adds a name mapping for the name.
  string GetInputName(const string& name);

  // Normalize the output name and make it unique.
  string GetOutputName(const string& name);

  // Make the node name unique.
  string Uniquify(const string& name);

  // Records name as a used name. If this name is already used,
  // returns an error status.
  Status UseOutputName(const string& name);

  // Look up how a node name was previously normalized/uniquified.
  // Returns empty if name was never seen.
  string Lookup(const string& name) const;

 private:
  string UniquifyHelper(const string& name);
  static string Normalize(string name);

  // The normalized/uniquified names already used as
  // input names (in signature), output names (in signature), and node names
  // (in node_def).
  // This is a superset of values in name_mapping_.
  std::unordered_map<string, uint64> used_names_;
  // Mapping from original node name from the graph to the normalized
  // and uniquified version of it.
  std::unordered_map<string, string> name_mapping_;
};

string NodeNameMapping::Normalize(string name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_0(mht_0_v, 252, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::Normalize");

  // Convert letters to lowercase and non-alphanumeric characters to '_'.
  if (name.empty()) return "unknown";
  const int n = name.size();
  for (int i = 0; i < n; ++i) {
    char c = name[i];
    if (isalnum(c)) {
      if (isupper(c)) {
        name[i] = tolower(c);
      }
    } else {
      name[i] = '_';
    }
  }

  // Find the first letter and start with it.
  int i = 0;
  for (; i < n; ++i) {
    if (isalpha(name[i])) break;
  }

  // Return "unknown" if none of the name's chars were letters.
  return i == n ? "unknown" : name.substr(i);
}

string NodeNameMapping::UniquifyHelper(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_1(mht_1_v, 281, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::UniquifyHelper");

  auto it = used_names_.emplace(name, 0);
  // If the name hasn't been used yet, use it as-is.
  if (it.second) return name;

  // Add a suffix to name to make it unique.
  while (true) {
    const string candidate = strings::StrCat(name, "_", it.first->second);
    it.first->second++;
    if (used_names_.emplace(candidate, 0).second) return candidate;
  }
}

string NodeNameMapping::GetInputName(const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_2(mht_2_v, 298, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::GetInputName");

  const string& input_name = UniquifyHelper(Normalize(name));
  name_mapping_[name] = input_name;
  return input_name;
}

string NodeNameMapping::GetOutputName(const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_3(mht_3_v, 308, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::GetOutputName");

  const string& input_name = UniquifyHelper(Normalize(name));
  // Don't add it to name_mapping_ since this name is not for a node.
  return input_name;
}

string NodeNameMapping::Uniquify(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_4(mht_4_v, 318, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::Uniquify");

  const string uniqued = UniquifyHelper(name);
  name_mapping_[name] = uniqued;
  return uniqued;
}

Status NodeNameMapping::UseOutputName(const string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_5(mht_5_v, 328, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::UseOutputName");

  const auto& iter = used_names_.find(name);
  if (iter != used_names_.end()) {
    return errors::InvalidArgument(
        "Cannot have duplicate output names. Name '", name,
        "' appears more than once in 'output_names' array.");
  }
  used_names_.emplace(name, 0);
  return Status::OK();
}

string NodeNameMapping::Lookup(const string& name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_6(mht_6_v, 343, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "NodeNameMapping::Lookup");

  const auto iter = name_mapping_.find(name);
  if (iter == name_mapping_.end()) return string();
  return iter->second;
}

Status FillFunctionBody(
    const string& fn_name, const NodeNameMapping& node_names,
    const std::vector<const Node*>& body_nodes,
    const std::unordered_map<string, string>& tensor_renaming,
    bool set_stateful_from_nodes, bool copy_placeholder_attrs_from_nodes,
    FunctionDef* fdef) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("fn_name: \"" + fn_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_7(mht_7_v, 358, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "FillFunctionBody");

  std::unordered_set<string> func_attr_names;
  for (const auto& func_attr : fdef->signature().attr()) {
    func_attr_names.insert(func_attr.name());
  }

  std::vector<const Edge*> in_edges;
  std::vector<const Edge*> control_edges;
  for (const Node* node : body_nodes) {
    NodeDef* node_def = fdef->add_node_def();
    // First, copy the node_def as is. We will patch it next.
    *node_def = node->def();
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }
    node_def->set_name(node_names.Lookup(node->name()));
    MergeDebugInfo(NodeDebugInfo(node->def()), node_def);

    // Input names must be set based on nested names in tensor_renaming.
    // Clear the flat input names we got from the original node_def
    // from the graph.
    node_def->clear_input();

    // Collect regular and control inputs. Regular inputs are indexed
    // by the index at which they come into the `node`. Control inputs
    // don't follow any order, and we sort control inputs to make sure generated
    // NodeDef is deterministic.
    in_edges.clear();
    in_edges.resize(node->num_inputs(), nullptr);
    control_edges.clear();
    for (const Edge* edge : node->in_edges()) {
      if (edge->src()->IsSource()) continue;
      if (edge->IsControlEdge()) {
        control_edges.push_back(edge);
      } else {
        in_edges[edge->dst_input()] = edge;
      }
    }
    std::sort(control_edges.begin(), control_edges.end(),
              [](const Edge* a, const Edge* b) {
                return a->src()->name() < b->src()->name();
              });

    // Add regular inputs.
    for (size_t i = 0; i < in_edges.size(); ++i) {
      const Edge* edge = in_edges[i];
      string original_input_name;
      if (edge == nullptr) {
        // A backedge might not appear as a regular Edge, but be only present
        // in the node_def. Such edges are referred to as requested_inputs().
        if (i >= node->requested_inputs().size()) {
          return errors::InvalidArgument(
              "Graph to be converted to function appears to be malformed. ",
              "Node ", node->name(), " is missing input edge ", i);
        }
        original_input_name =
            ParseTensorName(node->requested_inputs()[i]).ToString();
      } else {
        original_input_name =
            strings::StrCat(edge->src()->name(), ":", edge->src_output());
      }

      const auto iter = tensor_renaming.find(original_input_name);
      if (iter == tensor_renaming.end()) {
        return errors::InvalidArgument(
            "Input ", i, ", '", original_input_name, "', of node '",
            node->name(), "' in function '", fn_name,
            "' is not available. You might need to include it in inputs "
            "or include its source node in the body");
      }
      node_def->add_input(iter->second);
    }

    // Add control inputs.
    for (const Edge* edge : control_edges) {
      // Add this control input only if the src node is in the body or a part of
      // the inputs.
      const string normalized = node_names.Lookup(edge->src()->name());
      // If we did not find a name for the source of control edge, this
      // source must be outside of the body, and not an input. Raise an error.
      if (normalized.empty()) {
        return errors::InvalidArgument(
            "The source of control edge ", edge->DebugString(),
            " is not in the body. Encountered while creating function '",
            fn_name, "'");
      }
      node_def->add_input(strings::StrCat("^", normalized));
    }

    // A function is stateful if any of its nodes are stateful.
    if (set_stateful_from_nodes && node->op_def().is_stateful()) {
      fdef->mutable_signature()->set_is_stateful(true);
    }

    // If this node has any attributes with placeholder value, add the
    // attribute to FunctionDef signature.
    if (!copy_placeholder_attrs_from_nodes) {
      continue;
    }
    for (const auto& iter : node->attrs()) {
      if (iter.second.placeholder().empty()) {
        continue;
      }

      // If we already added the attribute, skip it.
      string func_attr_name = iter.second.placeholder();
      if (func_attr_names.find(func_attr_name) != func_attr_names.end()) {
        continue;
      }

      // This node's attribute is a placeholder value, so it does not have type
      // information. We check node's OpDef for attribute type.
      string node_attr_name = iter.first;
      const OpDef::AttrDef* node_attr_def = nullptr;
      for (const auto& node_attr : node->op_def().attr()) {
        if (node_attr.name() == node_attr_name) {
          node_attr_def = &node_attr;
        }
      }
      if (!node_attr_def) {
        return errors::Unimplemented(
            "Placeholder value is not supported for attributes not in OpDef. "
            "Attribute: ",
            node_attr_name, ", OpDef: ", node->op_def().DebugString());
      }
      OpDef::AttrDef* attr_def = fdef->mutable_signature()->add_attr();
      attr_def->set_name(func_attr_name);
      attr_def->set_type(node_attr_def->type());

      func_attr_names.insert(func_attr_name);
    }
  }
  return Status::OK();
}

Status GraphToFunctionDefHelper(
    const Graph& graph, const string& name,
    const std::function<absl::optional<string>(const Node*)>& control_ret,
    const std::vector<string>& output_names, FunctionDef* fdef) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_8(mht_8_v, 500, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "GraphToFunctionDefHelper");

  auto add_arg_or_retval = [](Node* node,
                              std::vector<OutputTensor>* args_or_retvals) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_9(mht_9_v, 505, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "lambda");

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
    if (index >= args_or_retvals->size()) {
      args_or_retvals->resize(index + 1);
    }
    if ((*args_or_retvals)[index].node == nullptr) {
      (*args_or_retvals)[index].node = node;
    } else {
      return errors::InvalidArgument("Multiple '", node->type_string(),
                                     "' nodes found with index ", index);
    }
    return Status::OK();
  };

  std::vector<const Node*> body_nodes;
  std::vector<OutputTensor> inputs;
  std::vector<OutputTensor> outputs;
  std::vector<const Node*> control_outputs;
  std::vector<string> control_output_names;
  for (Node* node : graph.op_nodes()) {
    if (node->IsArg()) {
      TF_RETURN_IF_ERROR(add_arg_or_retval(node, &inputs));
      continue;
    }

    if (node->IsRetval()) {
      TF_RETURN_IF_ERROR(add_arg_or_retval(node, &outputs));
      continue;
    }

    if (control_ret) {
      auto control_ret_name = control_ret(node);
      if (control_ret_name.has_value()) {
        control_outputs.push_back(node);
        control_output_names.push_back(control_ret_name.value());
      }
    }

    body_nodes.push_back(node);
  }

  auto validate_args_retvals =
      [](const std::vector<OutputTensor>& args_or_retvals,
         const string& op_type) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_10(mht_10_v, 553, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "lambda");

        for (int i = 0, e = args_or_retvals.size(); i < e; ++i) {
          if (args_or_retvals[i].node == nullptr) {
            return errors::InvalidArgument("Missing '", op_type,
                                           "' node at index ", i);
          }
        }
        return Status::OK();
      };

  TF_RETURN_IF_ERROR(validate_args_retvals(inputs, "_Arg"));
  TF_RETURN_IF_ERROR(validate_args_retvals(outputs, "_Retval"));

  return GraphToFunctionDef(graph, name, /*append_hash_to_fn_name=*/false,
                            /*set_stateful_from_nodes=*/false,
                            /*copy_placeholder_attrs_from_nodes=*/false,
                            body_nodes, inputs, outputs, output_names,
                            control_outputs, control_output_names,
                            /*description=*/nullptr, fdef);
}

}  // anonymous namespace

Status GraphToFunctionDef(const Graph& fn_body, const string& fn_name,
                          bool append_hash_to_fn_name,
                          bool set_stateful_from_nodes,
                          bool copy_placeholder_attrs_from_nodes,
                          const std::vector<const Node*>& body_nodes,
                          const std::vector<OutputTensor>& inputs,
                          const std::vector<OutputTensor>& outputs,
                          const std::vector<string>& output_names,
                          const std::vector<const Node*>& control_outputs,
                          const std::vector<string>& control_output_names,
                          const char* description, FunctionDef* fdef) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("fn_name: \"" + fn_name + "\"");
   mht_11_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_11(mht_11_v, 591, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "GraphToFunctionDef");

  if (!output_names.empty()) {
    DCHECK_EQ(output_names.size(), outputs.size());
  }

  if (description != nullptr) {
    fdef->mutable_signature()->set_description(description);
  }

  // Keep track of names we used and how we normalized them.
  NodeNameMapping node_names;

  // Mapping from original names of tensors (i.e. "<node_name>:<idx>") to the
  // name we used in the function:
  //  - For input tensors:
  //    {flat_tensor_name -> normalized_name_of_src_node}
  //    e.g. {In:3 -> in}
  //  - For tensors produced by nodes in function's body:
  //    {flat_tensor_name -> nested_tensor_name}
  //    e.g. {Add:3 -> add_0:z:1}
  std::unordered_map<string, string> tensor_renaming;

  // Fill outputs in function's signature.
  // We fill the outputs first to prevent output_names from colliding
  // with the input names we pick below. With this order, no names are used in
  // node_names yet, and output_names won't collide with anything (except
  // potentially with themselves).
  for (size_t i = 0; i < outputs.size(); ++i) {
    const Node* node = outputs[i].node;
    int idx = outputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_output_arg();
    if (node->IsRetval()) {
      argdef->set_type(node->input_type(idx));
    } else {
      argdef->set_type(node->output_type(idx));
    }
    if (!output_names.empty()) {
      TF_RETURN_IF_ERROR(node_names.UseOutputName(output_names[i]));
      argdef->set_name(output_names[i]);
    } else {
      argdef->set_name(node_names.GetOutputName(node->name()));
    }
  }

  // Fill inputs in function's signature.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Node* node = inputs[i].node;
    int idx = inputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_input_arg();
    argdef->set_type(node->output_type(idx));
    const string& input_name = node_names.GetInputName(node->name());
    argdef->set_name(input_name);
    FunctionDef::ArgAttrs arg_attrs;
    int64_t resource_arg_unique_id = -1;
    for (const auto& attr : node->attrs()) {
      // Only copy internal attributes. These attributes will be applied to
      // _Arg/Placeholder nodes when this FunctionDef is converted to graph,
      // and normal attributes for nodes cannot be applied to those
      // _Arg/Placeholder nodes.
      if (absl::StartsWith(attr.first, "_")) {
        arg_attrs.mutable_attr()->insert(attr);
      } else if (attr.first == "shape" && argdef->type() != DT_RESOURCE) {
        // Preserve known shapes by moving them to the _output_shapes list.
        // The _Arg shape function knows how to extract them from there.
        // Don't preserve the shape of a resource arg node, which is a scalar
        // resource handle.
        AttrValue value;
        *(value.mutable_list()->add_shape()) = attr.second.shape();
        arg_attrs.mutable_attr()->insert({"_output_shapes", value});
      } else if (attr.first == "value" && node->type_string() == "Const") {
        // Small eager tensors are captured as const ops rather than
        // Placeholders. Add a _output_shapes arg_attr with the shape of the
        // const tensor.
        AttrValue value;
        *(value.mutable_list()->add_shape()) =
            attr.second.tensor().tensor_shape();
        arg_attrs.mutable_attr()->insert({"_output_shapes", value});
      }
      if (attr.first == "_resource_arg_unique_id") {
        resource_arg_unique_id = attr.second.i();
      }
    }
    if (arg_attrs.attr_size() > 0) {
      (*fdef->mutable_arg_attr())[i] = std::move(arg_attrs);
    }
    if (resource_arg_unique_id >= 0) {
      (*fdef->mutable_resource_arg_unique_id())[idx] = resource_arg_unique_id;
    }
    tensor_renaming[strings::StrCat(node->name(), ":", idx)] = input_name;
  }

  // Populate tensor_renaming and node_names.
  // Generate the new output names for every node in the function.
  // The NodeDefs in FunctionDefs use a different naming scheme for
  // their inputs than the NodeDefs in a graph (see the comment for
  // FunctionDef.node_def in function.proto). We do the
  // graph tensor name -> function tensor name conversion for every
  // possible input (i.e. every node's outputs) and store the result
  // in tensor_renaming.
  for (const Node* node : body_nodes) {
    // Make sure node_name does not collide with an input or output name.
    const string& node_name = node_names.Uniquify(node->name());
    // For each output_arg in the op_def, the output_ranges
    // map will have [start, end] range of indices that this arg produces
    // among all the output tensors of this op.
    NameRangeMap output_ranges;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(*node, node->op_def(), nullptr, &output_ranges));
    for (const auto& output : output_ranges) {
      const StringPiece& output_name = output.first;
      int index_start = output.second.first;
      int index_end = output.second.second;
      for (int i = index_start; i < index_end; ++i) {
        const string& original_name = strings::StrCat(node->name(), ":", i);
        const string& new_name =
            strings::StrCat(node_name, ":", output_name, ":", i - index_start);
        // Record the mapping if this tensor is not already mapped.
        // Tensor can be already mapped if it is used as an input.
        if (tensor_renaming.find(original_name) == tensor_renaming.end()) {
          tensor_renaming[original_name] = new_name;
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(FillFunctionBody(fn_name, node_names, body_nodes,
                                      tensor_renaming, set_stateful_from_nodes,
                                      copy_placeholder_attrs_from_nodes, fdef));

  // Remap return values.
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string& ret_name = fdef->signature().output_arg(r).name();
    // We convert this flat tensor name to the nested value
    // (e.g. `add:z:1`) that we stored in tensor_renaming.
    string return_value;
    if (outputs[r].node->IsRetval()) {
      Edge const* edge;
      TF_RETURN_IF_ERROR(outputs[r].node->input_edge(0, &edge));
      return_value =
          strings::StrCat(edge->src()->name(), ":", edge->src_output());
    } else {
      return_value =
          strings::StrCat(outputs[r].node->name(), ":", outputs[r].index);
    }
    const auto iter = tensor_renaming.find(return_value);
    if (iter == tensor_renaming.end()) {
      return errors::InvalidArgument(
          "TF_Output ", return_value, " is neither in the function body ",
          "nor among function inputs. Encountered while creating function '",
          fn_name, "'");
    }
    (*fdef->mutable_ret())[ret_name] = iter->second;
  }

  if (append_hash_to_fn_name) {
    const uint64 hash = FunctionDefHash(*fdef);
    string encoded;
    TF_RETURN_IF_ERROR(Base64Encode(
        StringPiece(reinterpret_cast<const char*>(&hash), sizeof(hash)),
        &encoded));
    // Besides letters and digits our Base64 encoding uses '_' and '-'.
    // Dash is invalid in operation names and multiple underscores in random
    // places look strange. Since we never need to decode the hash back,
    // replace these chars with 'a' and 'A'. Replacing with different letters
    // keeps more entropy.
    std::replace(encoded.begin(), encoded.end(), '-', 'a');
    std::replace(encoded.begin(), encoded.end(), '_', 'A');
    fdef->mutable_signature()->set_name(strings::StrCat(fn_name, "_", encoded));
  } else {
    fdef->mutable_signature()->set_name(fn_name);
  }

  if (!control_output_names.empty() &&
      (control_outputs.size() != control_output_names.size())) {
    return errors::InvalidArgument(
        "Expected number of control outputs (", control_outputs.size(),
        ") and the number of control output names (",
        control_output_names.size(), ") to match but they do not.");
  }
  std::set<string> control_output_names_set;
  for (int i = 0; i < control_outputs.size(); ++i) {
    string signature_name;
    if (!control_output_names.empty()) {
      signature_name = control_output_names[i];
    } else {
      signature_name = control_outputs[i]->name();
    }
    if (signature_name.empty()) {
      return errors::InvalidArgument("Control output name must be not empty");
    }
    if (!control_output_names_set.insert(signature_name).second) {
      return errors::InvalidArgument("Repeated control output name: ",
                                     signature_name);
    }
    const string control_output_node =
        node_names.Lookup(control_outputs[i]->name());
    if (control_output_node.empty()) {
      return errors::InvalidArgument(
          "Control output node name must be not empty");
    }
    (*fdef->mutable_control_ret())[signature_name] = control_output_node;
  }
  for (const string& control_output : control_output_names_set) {
    fdef->mutable_signature()->add_control_output(control_output);
  }

  return Status::OK();
}

Status GraphToFunctionDef(
    const Graph& graph, const string& name,
    const std::function<absl::optional<string>(const Node*)>& control_ret,
    FunctionDef* fdef) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_12(mht_12_v, 807, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "GraphToFunctionDef");

  return GraphToFunctionDefHelper(graph, name, control_ret,
                                  /*output_names=*/{}, fdef);
}

Status GraphToFunctionDef(const Graph& graph, const string& name,
                          FunctionDef* fdef) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_13(mht_13_v, 817, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "GraphToFunctionDef");

  return GraphToFunctionDef(graph, name, /*control_ret=*/nullptr, fdef);
}

Status GraphToFunctionDef(const Graph& graph, const string& name,
                          const std::vector<std::string>& output_names,
                          FunctionDef* fdef) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondefDTcc mht_14(mht_14_v, 827, "", "./tensorflow/core/framework/graph_to_functiondef.cc", "GraphToFunctionDef");

  return GraphToFunctionDefHelper(graph, name, /*control_ret=*/nullptr,
                                  output_names, fdef);
}

}  // namespace tensorflow
