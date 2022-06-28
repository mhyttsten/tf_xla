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
class MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc {
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
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc() {
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

#include "tensorflow/cc/tools/freeze_saved_model.h"

#include <iostream>
#include <queue>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {

namespace {

// Gets tensor names from tensor_info and inserts them into the set of tensor
// names.
void GetTensorNamesFromTensorInfo(const TensorInfo& tensor_info,
                                  std::unordered_set<string>* tensor_names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_0(mht_0_v, 207, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetTensorNamesFromTensorInfo");

  if (tensor_info.has_coo_sparse()) {
    // If the tensor is sparse we have to add all three tensors of the sparse
    // representations.
    const TensorInfo_CooSparse& coo_sparse = tensor_info.coo_sparse();
    tensor_names->insert(coo_sparse.values_tensor_name());
    tensor_names->insert(coo_sparse.indices_tensor_name());
    tensor_names->insert(coo_sparse.dense_shape_tensor_name());
  } else if (tensor_info.has_composite_tensor()) {
    for (const auto& component : tensor_info.composite_tensor().components()) {
      tensor_names->insert(component.name());
    }
  } else {
    tensor_names->insert(tensor_info.name());
  }
}

// Gets the union of all inputs and outputs of all SignatureDefs in the bundle
void GetSignatureDefsInputsAndOutputs(
    const SavedModelBundle& saved_model_bundle,
    std::unordered_set<string>* inputs, std::unordered_set<string>* outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_1(mht_1_v, 230, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetSignatureDefsInputsAndOutputs");

  for (auto& sigdef_elem : saved_model_bundle.meta_graph_def.signature_def()) {
    const SignatureDef& signature_def = sigdef_elem.second;
    for (auto& input_elem : signature_def.inputs()) {
      GetTensorNamesFromTensorInfo(input_elem.second, inputs);
    }
    for (auto& output_elem : signature_def.outputs()) {
      GetTensorNamesFromTensorInfo(output_elem.second, outputs);
    }
  }
}

// Gets a map from string node name to NodeDef.
void GetNodeNameToNodeDefMap(
    GraphDef* graph_def,
    std::unordered_map<string, NodeDef*>* name_to_node_map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_2(mht_2_v, 248, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetNodeNameToNodeDefMap");

  for (size_t i = 0; i < graph_def->node_size(); i++) {
    NodeDef* node = graph_def->mutable_node(i);
    (*name_to_node_map)[node->name()] = node;
  }
}

// Strips off the tensor part of the tensor_name to get the node_name.
const string GetNodeNameFromTensorName(string tensor_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_3(mht_3_v, 260, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetNodeNameFromTensorName");

  if (tensor_name[0] == '^') {
    tensor_name.erase(0, 1);
  }
  std::vector<string> tensor_name_parts = str_util::Split(tensor_name, ':');
  return tensor_name_parts[0];
}

// Gets the set of node names needed by `outputs` and the corresponding set of
// variable nodes to convert.
void GetReachableNodesAndVariables(
    GraphDef* graph_def, const std::unordered_set<string>& outputs,
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    std::unordered_set<string>* reachable_node_names,
    std::unordered_set<string>* variable_node_names) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_4(mht_4_v, 277, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetReachableNodesAndVariables");

  // TODO(suharshs): Add support for ResourceVariables.
  static const std::unordered_set<string>* kVariableTypes =
      new std::unordered_set<string>({"Variable", "VariableV2", "VarHandleOp"});

  std::queue<string> nodes_to_visit;
  for (const string& output_tensor_name : outputs) {
    nodes_to_visit.push(GetNodeNameFromTensorName(output_tensor_name));
  }
  // We do a traversal backwards from the outputs specified in the MetaGraphDef.
  while (!nodes_to_visit.empty()) {
    const string node_name = nodes_to_visit.front();
    nodes_to_visit.pop();
    if (reachable_node_names->find(node_name) != reachable_node_names->end()) {
      continue;
    }
    reachable_node_names->insert(node_name);
    NodeDef* node = name_to_node_map.at(node_name);
    if (kVariableTypes->find(node->op()) != kVariableTypes->end()) {
      variable_node_names->insert(node->name());
    }
    for (const string& input_tensor_name : node->input()) {
      nodes_to_visit.push(GetNodeNameFromTensorName(input_tensor_name));
    }
  }
}

// Gets a map from variable name to variable value.
Status GetVariableNameToTensorMap(
    Session* session,
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    std::unordered_set<string> variable_names_set,
    std::unordered_map<string, Tensor>* variable_name_to_value_map) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_5(mht_5_v, 312, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "GetVariableNameToTensorMap");

  if (variable_names_set.empty()) {
    return Status::OK();
  }
  std::vector<string> variable_names;
  variable_names.reserve(variable_names_set.size());
  std::vector<string> tensor_names;
  tensor_names.reserve(variable_names_set.size());
  for (const string& node_name : variable_names_set) {
    variable_names.push_back(node_name);
    NodeDef* node_def = name_to_node_map.at(node_name);
    if (node_def->op() == "VarHandleOp") {
      // If this is a resource variable, we have to run the corresponding
      // ReadVariableOp.
      tensor_names.push_back(node_name + "/Read/ReadVariableOp:0");
    } else {
      tensor_names.push_back(node_name + ":0");
    }
  }
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(
      session->Run(/* inputs */ {}, tensor_names, /* targets */ {}, &outputs));
  for (size_t i = 0; i < variable_names.size(); i++) {
    (*variable_name_to_value_map)[variable_names[i]] = outputs[i];
  }
  return Status::OK();
}

// Converts a Variable NodeDef into a Constant NodeDef.
void ConvertVariableToConstant(const NodeDef& variable_node,
                               const Tensor& variable_value,
                               NodeDef* const_node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_6(mht_6_v, 346, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "ConvertVariableToConstant");

  const_node->set_name(variable_node.name());
  const_node->set_op("Const");
  (*const_node->mutable_attr())["dtype"] = variable_node.attr().at("dtype");
  variable_value.AsProtoTensorContent(
      (*const_node->mutable_attr())["value"].mutable_tensor());
}

// Converts a ReadVariableOp NodeDef to an Identity NodeDef.
void ConvertReadVariableOpToIdentity(const NodeDef& node,
                                     NodeDef* identity_node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_7(mht_7_v, 359, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "ConvertReadVariableOpToIdentity");

  identity_node->set_name(node.name());
  identity_node->set_op("Identity");
  (*identity_node->mutable_attr())["T"] = node.attr().at("dtype");
  identity_node->add_input(node.input(0));
}

// Returns the name of the VarHandleOp that provides input (possibly indirectly)
// to node with node_name. A typical indirect chain of nodes (that can occur due
// to graph inlining) is the following: VarHandleOp -> Identity -> Identity ->
// ReadVariableOp. Calling the function on any of these nodes would return the
// name of the VarHandleOp.
StatusOr<string> GetVarHandleName(
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    string node_name) {
  const NodeDef* node = name_to_node_map.at(node_name);
  while (node->input_size() > 0) {
    auto parent = name_to_node_map.find(node->input(0));
    if (parent == name_to_node_map.end()) break;
    node = parent->second;
    if (node->op() != "Identity") {
      VLOG(2) << "Stopping at non-identity node " << node->op();
      break;
    }
  }
  if (node->op() == "VarHandleOp") {
    return node->name();
  }
  return errors::NotFound("No VarHandleOp ancestor found");
}

// Looks up the variable handle that provides input to node with node_name,
// and returns the handle name if the handle corresponds to a variable that we
// want to freeze (i.e. its name is contained in variable_node_names). If there
// is no such handle in the graph (or we do not want to save that variable)
// then NotFound error is returned.
StatusOr<string> GetHandleNameIfNeedsToFreeze(
    const std::unordered_map<string, NodeDef*>& name_to_node_map,
    string node_name, const std::unordered_set<string>& variable_node_names) {
  StatusOr<string> var_handle_name =
      GetVarHandleName(name_to_node_map, node_name);
  if (var_handle_name.ok() && variable_node_names.count(*var_handle_name)) {
    return var_handle_name;
  }
  return errors::NotFound("No VarHandleOp ancestor found");
}

// Freezes the subgraph of all nodes needed by `outputs`.
Status FreezeGraphDef(const SavedModelBundle& saved_model_bundle,
                      const std::unordered_set<string>& outputs,
                      GraphDef* frozen_graph_def) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_8(mht_8_v, 412, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "FreezeGraphDef");

  GraphDef graph_def = saved_model_bundle.meta_graph_def.graph_def();
  // Copy versions and library as-is from original graph.
  *frozen_graph_def->mutable_versions() = graph_def.versions();
  *frozen_graph_def->mutable_library() = graph_def.library();
  // If the graph is empty there is nothing left to do.
  if (graph_def.node_size() == 0) {
    return Status::OK();
  }
  // name_to_node_map is needed to get the inputs from the NodeDef corresponding
  // the a string node name. These inputs are used when doing our backwards
  // traversal.
  std::unordered_map<string, NodeDef*> name_to_node_map;
  GetNodeNameToNodeDefMap(&graph_def, &name_to_node_map);
  std::unordered_set<string> reachable_node_names;
  std::unordered_set<string> variable_node_names;
  GetReachableNodesAndVariables(&graph_def, outputs, name_to_node_map,
                                &reachable_node_names, &variable_node_names);
  std::unordered_map<string, Tensor> variable_to_value_map;
  TF_RETURN_IF_ERROR(GetVariableNameToTensorMap(
      saved_model_bundle.session.get(), name_to_node_map, variable_node_names,
      &variable_to_value_map));
  // We copy the nodes in the same order they were in the original graph_def.
  for (const NodeDef& node : graph_def.node()) {
    if (reachable_node_names.find(node.name()) == reachable_node_names.end()) {
      continue;
    }
    if (variable_node_names.find(node.name()) != variable_node_names.end()) {
      ConvertVariableToConstant(node, variable_to_value_map[node.name()],
                                frozen_graph_def->add_node());
      continue;
    } else if (node.op() == "ReadVariableOp" &&
               GetHandleNameIfNeedsToFreeze(name_to_node_map, node.name(),
                                            variable_node_names)
                   .ok()) {
      // If the node is a ReadVariableOp, its input VarHandleOp will be
      // converted to a Constant, so we will need to convert it to an Identity.
      ConvertReadVariableOpToIdentity(node, frozen_graph_def->add_node());
      continue;
    } else if (node.op() == "Identity") {
      StatusOr<string> handle_name = GetHandleNameIfNeedsToFreeze(
          name_to_node_map, node.name(), variable_node_names);
      if (handle_name.ok()) {
        // Identity node that is forwarding the value of a frozen
        // VarhandleOp. We ensure that the dtype matches of the variable dtype.
        NodeDef* new_node = frozen_graph_def->add_node();
        *new_node = node;
        (*new_node->mutable_attr())["T"] =
            name_to_node_map.at(*handle_name)->attr().at("dtype");
        continue;
      }
    }
    // If the node isn't a variable, just copy the node as-is.
    *frozen_graph_def->add_node() = node;
  }
  return Status::OK();
}

}  // namespace

Status FreezeSavedModel(const SavedModelBundle& saved_model_bundle,
                        GraphDef* frozen_graph_def,
                        std::unordered_set<string>* inputs,
                        std::unordered_set<string>* outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPStoolsPSfreeze_saved_modelDTcc mht_9(mht_9_v, 478, "", "./tensorflow/cc/tools/freeze_saved_model.cc", "FreezeSavedModel");

  GetSignatureDefsInputsAndOutputs(saved_model_bundle, inputs, outputs);
  TF_RETURN_IF_ERROR(
      FreezeGraphDef(saved_model_bundle, *outputs, frozen_graph_def));
  return Status::OK();
}

}  // namespace tensorflow
