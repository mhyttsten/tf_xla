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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc() {
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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/functionalize_cond.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/functionalize_while.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

// Helper functions for functionalizing control flow in functions.

// Maps function name to
// - new function name, if the function body was functionalized
// - absl::nullopt, if not
using FuncMap = std::map<string, absl::optional<string>>;
using FuncMapIter = std::map<string, absl::optional<string>>::const_iterator;

// Returns whether function has been processed before.
bool FunctionHasBeenProcessed(FuncMapIter func_iter, const FuncMap* func_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionHasBeenProcessed");

  return func_iter != func_map->end();
}

// Returns whether function has been modified (i.e., functionalized) before.
bool FunctionHasBeenModified(FuncMapIter func_iter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionHasBeenModified");

  return func_iter->second.has_value();
}

// Returns a name for the new functionalized version of a function.
string GetNewFunctionName(
    const string& func_name, Node* n,
    AssociatedFunctionInfo::AssociatedFunctionType func_type,
    FunctionLibraryDefinition* fld) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "GetNewFunctionName");

  // For SymbolicGradient, `func_name` is always "SymbolicGradient" which
  // is not very informative. Use node name instead.
  return (
      func_type ==
              AssociatedFunctionInfo::AssociatedFunctionType::kSymbolicGradient
          ? fld->UniqueFunctionName(absl::StrCat(n->name(), "_f15n_"))
          : fld->UniqueFunctionName(absl::StrCat(func_name, "_f15n_")));
}

// Returns name to which a modified function has been mapped.
const string& GetMappedFunctionName(FuncMapIter func_iter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "GetMappedFunctionName");

  DCHECK(func_iter->second.has_value());
  return func_iter->second.value();
}

// Updates `func_map` with function given by `canonicalized_name`.
void UpdateFunctionMap(FuncMap* func_map, const string& canonicalized_name,
                       const string& new_func_name, bool function_modified) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("canonicalized_name: \"" + canonicalized_name + "\"");
   mht_4_v.push_back("new_func_name: \"" + new_func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_4(mht_4_v, 273, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "UpdateFunctionMap");

  // If function was modified store its new name, otherwise add empty entry to
  // record that function has been processed and does not need to be rewritten.
  (*func_map)[canonicalized_name] =
      function_modified ? absl::make_optional(new_func_name) : absl::nullopt;
}

// Adds new function def to graph's function library if necessary.
Status AddFunctionDefToGraphLibrary(
    const string& func_name, const AssociatedFunctionInfo& associated_function,
    Graph* graph, FunctionLibraryDefinition* fld) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_5(mht_5_v, 287, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "AddFunctionDefToGraphLibrary");

  const OpRegistrationData* op_reg_data;
  // We have to be careful with adding the function def since there are three
  // different `OpRegistryInterface`s involved here:
  // `fld`, `graph->flib_def()` and `graph->flib_def().default_registry()`.
  // We have already added the function def to `fld` before calling this
  // function but for the subsequent `RewriteAssociatedFunction` call we need
  // the function def to be in one of the other two registries, otherwise
  // `RewriteAssociatedFunction` will fail for the `kFunctionCallNode` case
  // because it cannot find the associated function def.
  // On the other hand, we should not add the function def if it is already
  // contained in one of the last two registries, this would lead to errors when
  // the function def is already in one registry and we try to add it to the
  // other one (if we try to add it to the same it's fine). This can happen in
  // cases where one of the last two registries is identical to `fld` (which we
  // already updated).
  // Therefore, before adding the function def we have to check if it's already
  // contained in either `graph->flib_def()` or
  // `graph->flib_def().default_registry()` which is done in the following line
  // (we have to use `LookUp` instead of `Contains` or `Find` because the latter
  // both don't check the default registry).
  if (graph->flib_def().LookUp(func_name, &op_reg_data).ok())
    return Status::OK();

  const FunctionDef* new_fdef = fld->Find(func_name);
  DCHECK(new_fdef != nullptr);
  FunctionDefLibrary fdef_lib;
  *(fdef_lib.add_function()) = *new_fdef;
  return graph->AddFunctionLibrary(fdef_lib);
}

// Functionalizes function given by `func_name`. Update `func_map` accordingly.
Status FunctionalizeControlFlowForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    FuncMap* func_map, bool* function_modified,
    const NodeFilter& node_filter = {});

// Functionalizes all functions that are (directly or indirectly) associated to
// any node in `graph`. Adds processed functions to `func_map`.
Status FunctionalizeControlFlowForNodeAssociatedFunctions(
    FuncMap* func_map, Graph* graph, FunctionLibraryDefinition* fld,
    FunctionLibraryRuntime* flr, bool* any_function_modified,
    const NodeFilter& node_filter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_6(mht_6_v, 334, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionalizeControlFlowForNodeAssociatedFunctions");

  std::vector<std::pair<Node*, std::vector<AssociatedFunctionInfo>>>
      nodes_to_associated_functions;
  for (auto* n : graph->nodes()) {
    auto associated_functions = GetAssociatedFunctions(*n, fld);
    if (!associated_functions.empty()) {
      nodes_to_associated_functions.push_back({n, associated_functions});
    }
  }
  for (const auto& pair : nodes_to_associated_functions) {
    Node* n = pair.first;
    auto associated_functions = pair.second;
    for (auto& associated_function : associated_functions) {
      // Note that if `n` is a function call node, then potential calls of
      // `RewriteAssociatedFunction` below might delete `n` and create a new
      // node instead, making `n` an invalid pointer. That's fine because in
      // that case `n` only has one associated function, so this loop has only
      // one iteration and we don't use `n` again after the rewrite.
      // The invariant is guaranteed by `GetAssociatedFunctions` and confirmed
      // below.
      DCHECK(associated_function.type() !=
                 AssociatedFunctionInfo::kFunctionCallNode ||
             associated_functions.size() == 1);

      // Process one node-function-pair.
      string func_name = associated_function.func_name();
      string canonicalized_name =
          Canonicalize(func_name, AttrSlice(&associated_function.attrs()));
      auto func_iter = func_map->find(canonicalized_name);
      string new_func_name;
      if (FunctionHasBeenProcessed(func_iter, func_map)) {
        if (FunctionHasBeenModified(func_iter)) {
          *any_function_modified = true;
          new_func_name = GetMappedFunctionName(func_iter);
          TF_RETURN_IF_ERROR(RewriteAssociatedFunction(
              graph, n, fld, associated_function, new_func_name));
        }
        continue;
      }
      // Function is processed for the first time.
      bool function_modified = false;
      new_func_name =
          GetNewFunctionName(func_name, n, associated_function.type(), fld);
      // Perform functionalization for current function.
      TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
          func_name, new_func_name, associated_function.attrs(), fld, flr,
          func_map, &function_modified, node_filter));
      UpdateFunctionMap(func_map, canonicalized_name, new_func_name,
                        function_modified);
      if (function_modified) {
        *any_function_modified = true;
        TF_RETURN_IF_ERROR(AddFunctionDefToGraphLibrary(
            new_func_name, associated_function, graph, fld));
        TF_RETURN_IF_ERROR(RewriteAssociatedFunction(
            graph, n, fld, associated_function, new_func_name));
      }
    }
  }
  return Status::OK();
}

Status FunctionalizeControlFlowForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    FuncMap* func_map, bool* function_modified, const NodeFilter& node_filter) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("func_name: \"" + func_name + "\"");
   mht_7_v.push_back("new_func_name: \"" + new_func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_7(mht_7_v, 404, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionalizeControlFlowForFunction");

  *function_modified = false;

  // Convert the function to a graph.
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flr->Instantiate(func_name, AttrSlice(&attrs), &handle));
  Status ret_status = Status::OK();
  auto cleanup_handle = gtl::MakeCleanup([&]() {
    auto s = flr->ReleaseHandle(handle);
    if (!s.ok()) {
      ret_status.Update(s);
    }
  });
  const FunctionBody* body = flr->GetFunctionBody(handle);
  Graph* g = body->graph;

  // Check if the graph has Switch or Merge node.
  bool has_switch_or_merge = false;
  for (Node* n : body->graph->nodes()) {
    // Skip nodes that are filtered out.
    if (node_filter && !node_filter(n)) continue;
    if (n->type_string() == "Switch" || n->type_string() == "Merge") {
      has_switch_or_merge = true;
      break;
    }
  }
  // Before functionalizing control flow in `g` we functionalize control flow
  // in functions (directly or indirectly) associated with nodes in `g`.
  TF_RETURN_IF_ERROR(FunctionalizeControlFlowForNodeAssociatedFunctions(
      func_map, g, fld, flr, function_modified, node_filter));

  if (has_switch_or_merge) {
    *function_modified = true;

    // Functionalize the function body.
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile(
          absl::StrCat("functionalize_control_flow_before_fdef_", func_name),
          *g, fld);
    }
    TF_RETURN_IF_ERROR(FunctionalizeControlFlow(g, fld, node_filter));
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile(
          absl::StrCat("functionalize_control_flow_after_fdef_", func_name), *g,
          fld);
    }
  }
  if (*function_modified) {
    // Add rewritten FunctionDef into library.
    FunctionDef functionalized_fdef;
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*g, new_func_name, &functionalized_fdef));
    if (func_name == new_func_name) {
      VLOG(2) << "Replacing function " << func_name;
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(new_func_name, functionalized_fdef));
    } else {
      VLOG(2) << "Adding function " << new_func_name;
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(functionalized_fdef));
    }
  }

  return ret_status;
}

Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library,
                                const NodeFilter& node_filter,
                                bool include_functions) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_8(mht_8_v, 475, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionalizeControlFlow");

  VLOG(2) << "FunctionalizeControlFlow (initial): "
          << DumpGraphToFile("functionalize_initial", *graph, library);

  if (include_functions) {
    // Functionalize control flow in functions that are (directly or indirectly)
    // associated with a node in `graph`.
    auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
        /*device_mgr=*/nullptr, tensorflow::Env::Default(),
        /*config=*/nullptr, TF_GRAPH_DEF_VERSION, library,
        tensorflow::OptimizerOptions());
    // `pflr` has only one `FunctionLibraryRuntime`, for `kDefaultFLRDevice`
    // (because we constructed it with `device_mgr = nullptr`).
    FunctionLibraryRuntime* flr =
        pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

    FuncMap func_map;
    bool modified = false;
    TF_RETURN_IF_ERROR(FunctionalizeControlFlowForNodeAssociatedFunctions(
        &func_map, graph, library, flr, &modified, node_filter));
  }
  // Functionalize and remove while loops from graph.
  TF_RETURN_IF_ERROR(FunctionalizeWhileLoop(graph, library, node_filter));

  // FunctionalizeControlFlow is invoked for every function, so the loops's
  // bodies and conditionals that were extracted into functions will be handled
  // in successive invocations.
  TF_RETURN_IF_ERROR(FunctionalizeCond(graph, library, node_filter));

  VLOG(2) << "FunctionalizeControlFlow (final): "
          << DumpGraphToFile("functionalize_final", *graph, library);

  return Status::OK();
}

Status FunctionalizeControlFlowForGraphDef(GraphDef* graph_def,
                                           FunctionLibraryDefinition* library,
                                           const NodeFilter& node_filter,
                                           bool include_functions) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_9(mht_9_v, 516, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionalizeControlFlowForGraphDef");

  FunctionDefLibrary function_lib = graph_def->library();
  Graph graph(OpRegistry::Global());

  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph({}, *graph_def, &graph));
  TF_RETURN_IF_ERROR(FunctionalizeControlFlow(&graph, library, node_filter,
                                              include_functions));
  graph.ToGraphDef(graph_def);
  std::swap(*graph_def->mutable_library(), function_lib);
  return Status::OK();
}

Status FunctionalizeControlFlowForXlaPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flowDTcc mht_10(mht_10_v, 532, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow.cc", "FunctionalizeControlFlowForXlaPass::Run");

  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("functionalize_control_flow_before", *graph,
                    options.flib_def);
  }
  const auto* config = &options.session_options->config;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          /*device_mgr=*/nullptr, options.session_options->env, config,
          TF_GRAPH_DEF_VERSION, options.flib_def,
          config->graph_options().optimizer_options()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  // Find XLA compile ops and its corresponding FunctionDef.
  // TPUCompile op is not in the map because graph rewriting might happen
  // multiple times, and we want to avoid functionalize it again.
  static std::map<string, string>* kNodeTypeToFunctionAttrMapping =
      new std::map<string, string>{
          // _TPUReplicate ops are generated by EncapsulateTPUComputationsPass.
          {"_TPUReplicate", "computation"},
          // XlaLaunch ops are generated by EncapsulateXlaComputationsPass.
          {"XlaLaunch", "function"},
      };
  FuncMap func_map;
  bool fld_modified = false;
  for (Node* n : graph->nodes()) {
    auto it = kNodeTypeToFunctionAttrMapping->find(n->type_string());
    if (it == kNodeTypeToFunctionAttrMapping->end()) {
      continue;
    }
    const string func_attr = it->second;
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), func_attr, &func));
    VLOG(2) << "Graph has node " << n->type_string()
            << ". Corresponding function: " << func.name();
    string new_func_name = options.flib_def->UniqueFunctionName(
        absl::StrCat(func.name(), "_f15n_"));
    bool modified;
    TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
        func.name(), new_func_name, func.attr(), options.flib_def, flr,
        &func_map, &modified));
    if (modified) {
      n->ClearAttr(func_attr);
      func.set_name(new_func_name);
      n->AddAttr(func_attr, func);
      fld_modified = true;
    }
  }

  // TODO(ylc, endlessroad): Change this to "if (fld_modified")"
  if (false) {
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile("functionalize_control_flow_before_prune", *graph,
                      options.flib_def);
    }
    TF_RETURN_IF_ERROR(
        PruneUnreachableFunctionsFromGraph(*graph, options.flib_def));
  }

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("functionalize_control_flow_after", *graph,
                    options.flib_def);
  }
  return Status::OK();
}

}  // namespace tensorflow
