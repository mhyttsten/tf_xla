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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/side_effect_util.h"

#include "absl/strings/numbers.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

const char kXlaTokenInputNodesAttrName[] = "_xla_token_input_nodes";

const char kXlaTokenArgNodeName[] = "_xla_token_arg_node";

const char kXlaHasHostTransferAttrName[] = "_xla_has_host_transfer";

const char kXlaReplicaIdAttrName[] = "_xla_replica_id";

const char kXlaIsPlaceholderForTailOcAttrName[] =
    "_xla_is_placeholder_for_tail_oc";

const char kXlaOriginalOutsideCompilationNodeName[] =
    "_xla_original_oc_node_name";

Status SetDeviceOrdinalAttributeForNode(Node* node, int device_ordinal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2xla/side_effect_util.cc", "SetDeviceOrdinalAttributeForNode");

  if (!HasNodeAttr(node->def(), kXlaHasHostTransferAttrName)) {
    return errors::InvalidArgument("Node ", node->DebugString(),
                                   " does not have attribute ",
                                   kXlaHasHostTransferAttrName);
  }

  if (node->type_string() == "_XlaRecvAtHost" ||
      node->type_string() == "_XlaSendFromHost") {
    node->ClearAttr("device_ordinal");
    node->AddAttr("device_ordinal", device_ordinal);
  } else if (node->IsIfNode()) {
    AttrValue device_ordinal_value;
    device_ordinal_value.set_i(device_ordinal);
    for (const string& attr_name :
         std::vector<string>{"then_branch", "else_branch"}) {
      NameAttrList branch_func;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr_name, &branch_func));
      (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
      node->ClearAttr(attr_name);
      node->AddAttr(attr_name, branch_func);
    }
  } else if (node->IsWhileNode()) {
    AttrValue device_ordinal_value;
    device_ordinal_value.set_i(device_ordinal);
    for (const string& attr_name : std::vector<string>{"cond", "body"}) {
      NameAttrList branch_func;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr_name, &branch_func));
      (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
      node->ClearAttr(attr_name);
      node->AddAttr(attr_name, branch_func);
    }
  } else if (HasNodeAttr(node->def(), "_device_ordinal")) {
    // Function call node containing outside compilation.
    node->ClearAttr("_device_ordinal");
    node->AddAttr("_device_ordinal", device_ordinal);
  } else {
    return errors::Internal("Unknown node type to set 'device_ordinal': ",
                            node->DebugString());
  }
  return Status::OK();
}

std::set<std::string> CalculateTokenInputsForOutputToken(const Graph& g) {
  std::set<std::string> results;
  Node* first_side_effecting_node_on_path = nullptr;
  ReverseDFS(g,
             [&](Node* n) {
               std::vector<string> token_input_nodes;
               if (!GetNodeAttr(n->attrs(), kXlaTokenInputNodesAttrName,
                                &token_input_nodes)
                        .ok() ||
                   token_input_nodes.empty()) {
                 return;
               }

               if (first_side_effecting_node_on_path != nullptr) {
                 return;
               }

               first_side_effecting_node_on_path = n;
               string original_node_name;
               TF_CHECK_OK(GetNodeAttr(n->def(),
                                       kXlaOriginalOutsideCompilationNodeName,
                                       &original_node_name));
               results.insert(original_node_name);
             },
             [&](Node* n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc mht_1(mht_1_v, 276, "", "./tensorflow/compiler/tf2xla/side_effect_util.cc", "lambda");

               if (first_side_effecting_node_on_path == n) {
                 first_side_effecting_node_on_path = nullptr;
               }
             },
             NodeComparatorName());
  return results;
}

bool HasSideEffectingNodes(const Graph& g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/tf2xla/side_effect_util.cc", "HasSideEffectingNodes");

  for (Node* n : g.nodes()) {
    std::vector<string> token_input_nodes;
    if (GetNodeAttr(n->attrs(), kXlaTokenInputNodesAttrName, &token_input_nodes)
            .ok() &&
        !token_input_nodes.empty()) {
      return true;
    }
  }
  return false;
}

Status ParseHostComputeCoreList(absl::Span<const string> list_from_attr,
                                std::map<string, int>* host_compute_core) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSside_effect_utilDTcc mht_3(mht_3_v, 304, "", "./tensorflow/compiler/tf2xla/side_effect_util.cc", "ParseHostComputeCoreList");

  for (const auto& hc_core : list_from_attr) {
    std::vector<string> parts = str_util::Split(hc_core, ":");
    if (parts.size() != 2) {
      return errors::InvalidArgument(
          "Malformed host_compute_core entry ", hc_core,
          " should be <cluster_name>:<core_number>.");
    }
    int core;
    if (!absl::numbers_internal::safe_strto32_base(parts[1], &core, 10)) {
      return errors::InvalidArgument("Malformed host_compute_core entry ",
                                     hc_core,
                                     " part after ':' should be an integer.");
    }
    if (host_compute_core->find(parts[0]) != host_compute_core->end()) {
      return errors::InvalidArgument(
          "Duplicate host_compute_core entry for cluster ", parts[0]);
    }
    (*host_compute_core)[parts[0]] = core;
  }
  return Status::OK();
}

}  // namespace tensorflow
