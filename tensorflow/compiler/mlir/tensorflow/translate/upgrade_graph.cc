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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"

#include "llvm/ADT/StringSet.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace {

constexpr char kTpuReplicateAttr[] = "_tpu_replicate";

// Returns the ops that should use node name if shared_name is empty.
const llvm::StringSet<>& GetOpsUsingNodeName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "GetOpsUsingNodeName");

  static auto* const ops =
      new llvm::StringSet<>({"VariableV2", "Variable", "BatchFunction"});
  return *ops;
}

// Returns the set of ops that we want to generate shared_names for them if
// empty.
const llvm::StringSet<>& GetSharedNameGenerationCompatibleOps() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "GetSharedNameGenerationCompatibleOps");

  return GetOpsUsingNodeName();
}

}  // namespace

Status GenerateResourceSharedNameIfEmpty(
    GraphDef& gdef, const OpRegistryInterface* default_registry) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "GenerateResourceSharedNameIfEmpty");

  auto is_resource_op_with_empty_shared_name = [](const NodeDef& node_def,
                                                  const OpDef& op_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "lambda");

    if (!GetSharedNameGenerationCompatibleOps().contains(op_def.name())) {
      // If this op is not in the allowlist, then it is likely a custom op.
      // Currently for these ops, we are relying on its "use_node_name_sharing"
      // to decide whether it is valid to generate shared_names. If the OpDef
      // has "use_node_name_sharing" field, then it is valid to use node names
      // as shared names.
      if (!std::any_of(op_def.attr().begin(), op_def.attr().end(),
                       [](const auto& attr_def) {
                         return attr_def.name() == "use_node_name_sharing" &&
                                attr_def.type() == "bool";
                       }))
        return false;
    }

    if (!std::any_of(op_def.attr().begin(), op_def.attr().end(),
                     [](const auto& attr_def) {
                       return attr_def.name() == "shared_name" &&
                              attr_def.type() == "string";
                     }))
      return false;

    auto iter = node_def.attr().find("shared_name");
    if (iter == node_def.attr().end()) return true;
    return iter->second.s().empty();
  };

  FunctionDefLibrary* library = gdef.mutable_library();
  auto flib_def = library ? std::make_unique<FunctionLibraryDefinition>(
                                default_registry, *library)
                          : std::make_unique<FunctionLibraryDefinition>(
                                default_registry, FunctionDefLibrary());

  if (library) {
    // Upgrade nodes in the functions.
    for (FunctionDef& fdef : *library->mutable_function()) {
      auto func_name = fdef.signature().name();
      for (auto& node_def : *fdef.mutable_node_def()) {
        const OpDef* op_def = nullptr;
        // With lazy loading, some functions might not be executed, thus we skip
        // the node if the op is not registered.
        if (flib_def->LookUpOpDef(node_def.op(), &op_def).ok() &&
            is_resource_op_with_empty_shared_name(node_def, *op_def)) {
          // TODO(b/197144710): improve the shared_name attr, each op may use
          // the shared_name differently.
          if (GetOpsUsingNodeName().contains(op_def->name())) {
            // Use the node name for such ops as the shared_name according to
            // the document of variable ops.
            (*node_def.mutable_attr())["shared_name"].set_s(node_def.name());
          } else {
            // Use the concat of function name and node name for such ops in a
            // function as the shared_name. "@" is used as the separator because
            // it is not allowed in the function name or the node name.
            (*node_def.mutable_attr())["shared_name"].set_s(
                absl::StrCat(node_def.name(), "@", func_name));
          }
        }
      }
    }
  }

  // Upgrade nodes in the GraphDef.
  for (auto& node_def : *gdef.mutable_node()) {
    const OpDef* op_def = nullptr;
    TF_RETURN_IF_ERROR(flib_def->LookUpOpDef(node_def.op(), &op_def));
    // TODO(b/197144710): improve the shared_name attr, each op may use the
    // shared_name differently.
    if (is_resource_op_with_empty_shared_name(node_def, *op_def)) {
      (*node_def.mutable_attr())["shared_name"].set_s(node_def.name());
    }
  }

  return tensorflow::Status::OK();
}

Status UpgradeLegacyGraph(Graph* graph, FunctionLibraryDefinition* flib_def,
                          bool restrict_functionalization_to_tpu_nodes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_4(mht_4_v, 309, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "UpgradeLegacyGraph");

  // If `restrict_functionalization_to_tpu_nodes` is true let filter function
  // return true for `_tpu_replicate` nodes, otherwise don't set filter.
  NodeFilter node_filter =
      restrict_functionalization_to_tpu_nodes
          ? [](const Node* n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSupgrade_graphDTcc mht_5(mht_5_v, 317, "", "./tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.cc", "lambda");
 return n->attrs().Find(kTpuReplicateAttr); }
          : NodeFilter{};
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      FunctionalizeControlFlow(graph, flib_def, node_filter,
                               /*include_functions=*/true),
      "Failed to functionalize Control Flow V1 ops. Consider using Control "
      "Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/"
      "compat/v1/enable_control_flow_v2.");
  return Status::OK();
}

}  // namespace tensorflow
