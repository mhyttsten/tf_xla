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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc() {
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

#include "tensorflow/compiler/tf2xla/const_analysis.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

Status GetFunctionBody(FunctionLibraryRuntime* flib_runtime,
                       const NodeDef& node, StringPiece func_attr_name,
                       const FunctionBody** fbody) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "GetFunctionBody");

  NameAttrList name_attr_list;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, func_attr_name, &name_attr_list));
  FunctionLibraryRuntime::Handle func_handle;
  TF_RETURN_IF_ERROR(flib_runtime->Instantiate(
      name_attr_list.name(), AttrSlice(&name_attr_list.attr()), &func_handle));
  *fbody = flib_runtime->GetFunctionBody(func_handle);
  return Status::OK();
}

Status GetFunctionBodies(FunctionLibraryRuntime* flib_runtime,
                         const NodeDef& node, StringPiece func_list_attr_name,
                         std::vector<const FunctionBody*>* fbodies) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "GetFunctionBodies");

  std::vector<NameAttrList> name_attr_lists;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, func_list_attr_name, &name_attr_lists));
  for (const NameAttrList& name_attr_list : name_attr_lists) {
    FunctionLibraryRuntime::Handle func_handle;
    TF_RETURN_IF_ERROR(flib_runtime->Instantiate(
        name_attr_list.name(), AttrSlice(&name_attr_list.attr()),
        &func_handle));
    fbodies->push_back(flib_runtime->GetFunctionBody(func_handle));
  }
  return Status::OK();
}

Status CondConstInputIndices(
    absl::Span<const FunctionBody* const> branch_bodies,
    std::vector<int>* const_input_idxs, FunctionLibraryRuntime* flib_runtime) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_2(mht_2_v, 240, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "CondConstInputIndices");

  TF_RET_CHECK(!branch_bodies.empty());
  TF_RET_CHECK(branch_bodies[0] != nullptr);
  int num_inputs = branch_bodies[0]->fdef.signature().input_arg_size();
  // Stores indices of the "branch function" inputs that are expected to be
  // compile time constants.
  std::vector<bool> compile_time_const_arg_indices(num_inputs);
  for (auto fbody : branch_bodies) {
    TF_RET_CHECK(fbody != nullptr);
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
        *(fbody->graph), &compile_time_const_arg_indices,
        /*compile_time_const_nodes=*/nullptr, flib_runtime));
  }
  for (int i = 0, end = compile_time_const_arg_indices.size(); i < end; i++) {
    if (compile_time_const_arg_indices[i]) {
      // The 0th input is the pred or branch index, which is not passed to the
      // branches. So the i'th input of a branch function corresponds to the
      // i + 1'th input of the If/Case op.
      const_input_idxs->push_back(i + 1);
    }
  }
  return Status::OK();
}

Status GetCompileTimeConstInputs(const NodeDef& node, const OpKernel* op_kernel,
                                 const OpDef* op_def,
                                 std::vector<int>* const_input_idxs,
                                 FunctionLibraryRuntime* flib_runtime) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "GetCompileTimeConstInputs");

  DCHECK(op_def != nullptr || op_kernel != nullptr);
  if (node.op() == "While" || node.op() == "StatelessWhile") {
    // For While nodes, recurse into the body and cond graphs.
    const FunctionBody* fcond = nullptr;
    const FunctionBody* fbody = nullptr;
    TF_RETURN_IF_ERROR(GetFunctionBody(flib_runtime, node, "cond", &fcond));
    TF_RETURN_IF_ERROR(GetFunctionBody(flib_runtime, node, "body", &fbody));
    TF_RET_CHECK(fcond);
    TF_RET_CHECK(fbody);
    int num_inputs = fbody->fdef.signature().input_arg_size();

    // Stores which of the loop inputs are expected to be compile time
    // constants.
    std::vector<bool> compile_time_const_arg_indices(num_inputs);
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
        *(fcond->graph), &compile_time_const_arg_indices,
        /*compile_time_const_nodes=*/nullptr, flib_runtime));
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
        *(fbody->graph), &compile_time_const_arg_indices,
        /*compile_time_const_nodes=*/nullptr, flib_runtime));
    for (int i = 0; i < num_inputs; i++) {
      if (compile_time_const_arg_indices[i]) {
        // Check that this input is actually a loop invariant.
        TF_ASSIGN_OR_RETURN(
            bool is_loop_invariant,
            IsLoopInvariant(fbody, i,
                            flib_runtime->GetFunctionLibraryDefinition()));
        if (is_loop_invariant) {
          const_input_idxs->push_back(i);
        } else {
          // TODO(b/178546817): Verify that it's OK and raise an error if we are
          // using this branch from jit_compile=True.
          Node* arg_i = fbody->arg_nodes[i];
          Node* ret_i = fbody->ret_nodes[i];
          VLOG(1) << "Argument " << i << " to while-loop " << node.name()
                  << " has to be constant, but it's not a loop invariant, "
                     "cluster compilation likely to fail at compile time: "
                  << arg_i->DebugString() << " vs. " << ret_i->DebugString();
          VLOG(1) << node.ShortDebugString();
        }
      }
    }
    return Status::OK();
  } else if (node.op() == "If" || node.op() == "StatelessIf") {
    const FunctionBody* fthen = nullptr;
    const FunctionBody* felse = nullptr;
    TF_RETURN_IF_ERROR(
        GetFunctionBody(flib_runtime, node, "then_branch", &fthen));
    TF_RETURN_IF_ERROR(
        GetFunctionBody(flib_runtime, node, "else_branch", &felse));
    return CondConstInputIndices({fthen, felse}, const_input_idxs,
                                 flib_runtime);
  } else if (node.op() == "Case" || node.op() == "StatelessCase") {
    std::vector<const FunctionBody*> branch_bodies;
    TF_RETURN_IF_ERROR(
        GetFunctionBodies(flib_runtime, node, "branches", &branch_bodies));
    return CondConstInputIndices(branch_bodies, const_input_idxs, flib_runtime);
  } else if (node.op() == "PartitionedCall" ||
             node.op() == "StatefulPartitionedCall") {
    const FunctionBody* fbody;
    TF_RETURN_IF_ERROR(GetFunctionBody(flib_runtime, node, "f", &fbody));
    int num_inputs = fbody->fdef.signature().input_arg_size();
    std::vector<bool> compile_time_const_arg_indices(num_inputs);
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
        *(fbody->graph), &compile_time_const_arg_indices,
        /*compile_time_const_nodes=*/nullptr, flib_runtime));
    for (int i = 0; i < num_inputs; i++) {
      if (compile_time_const_arg_indices[i]) {
        const_input_idxs->push_back(i);
      }
    }
    return Status::OK();
  } else if (op_def != nullptr) {
    return XlaOpRegistry::CompileTimeConstantInputs(node, *op_def,
                                                    const_input_idxs);
  } else {
    return XlaOpRegistry::CompileTimeConstantInputs(*op_kernel,
                                                    const_input_idxs);
  }
}

Status GetCompileTimeConstInputs(const Node* node,
                                 std::vector<int>* const_input_idxs,
                                 FunctionLibraryRuntime* flib_runtime) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_4(mht_4_v, 357, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "GetCompileTimeConstInputs");

  return GetCompileTimeConstInputs(node->def(), /*op_kernel=*/nullptr,
                                   &node->op_def(), const_input_idxs,
                                   flib_runtime);
}

}  // namespace

// Backwards dataflow analysis that finds arguments to a graph that must be
// compile-time constants.
Status BackwardsConstAnalysis(
    const Graph& g, std::vector<bool>* compile_time_const_arg_indices,
    std::vector<bool>* compile_time_const_nodes,
    FunctionLibraryRuntime* flib_runtime,
    std::function<bool(const Edge&)> edge_filter_input) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_5(mht_5_v, 374, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "BackwardsConstAnalysis");

  if (!compile_time_const_nodes && g.GetConstArgIndicesCache().has_value() &&
      !edge_filter_input) {
    VLOG(5) << "Using cached argument indices on graph " << &g;
    *compile_time_const_arg_indices = g.GetConstArgIndicesCache().value();
    return Status::OK();
  }
  auto edge_filter = [&](const Edge& e) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_6(mht_6_v, 384, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "lambda");

    return edge_filter_input ? edge_filter_input(e) : true;
  };

  std::vector<bool> compile_time_const_nodes_impl;
  if (compile_time_const_nodes) {
    CHECK_EQ(compile_time_const_nodes->size(), g.num_node_ids());
  } else {
    compile_time_const_nodes_impl.resize(g.num_node_ids());
    compile_time_const_nodes = &compile_time_const_nodes_impl;
  }

  Status status;
  auto visit = [&](Node* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_7(mht_7_v, 400, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "lambda");

    if (!status.ok()) return;

    // If this is a metadata-only op, don't propagate the const requirement.
    if (XlaOpRegistry::IsMetadataOp(node->type_string())) {
      VLOG(3) << "must-be-const node is metadata op: " << node->name();
      return;
    }

    // If this node must be const, and it isn't a metadata op, then all of its
    // parents must be const.
    if ((*compile_time_const_nodes)[node->id()]) {
      VLOG(3) << "marking consts for must-be-const node " << node->name();
      if (node->type_string() == "_Arg") {
        int index;
        status = GetNodeAttr(node->attrs(), "index", &index);
        if (!status.ok()) return;
        if (compile_time_const_arg_indices) {
          (*compile_time_const_arg_indices)[index] = true;
        }
        VLOG(3) << "  const _Arg " << index << ": " << node->name();
        return;
      }
      for (const Edge* pred : node->in_edges()) {
        if (!pred->IsControlEdge() && edge_filter(*pred)) {
          // If the src node of the `pred` is an IdentityN/While do not mark it
          // as a compile-time const. Only mark the corresponding input to the
          // IdentityN/While node as a const. XLA IdentityN op simply forwards
          // its inputs so this is safe; loop-invariance is checked elsewhere.
          while (edge_filter(*pred) && IsConstTraversableOpType(pred->src())) {
            status = pred->src()->input_edge(pred->src_output(), &pred);
            if (!status.ok()) return;
          }
          if (edge_filter(*pred)) {
            VLOG(4) << "  " << pred->src()->name() << " must be const (is "
                    << pred->src()->type_string() << ")";
            (*compile_time_const_nodes)[pred->src()->id()] = true;
          }
        }
      }
      return;
    }

    // Mark any compile-time constant operator arguments as const.
    std::vector<int> const_input_idxs;
    status = GetCompileTimeConstInputs(node, &const_input_idxs, flib_runtime);

    if (!status.ok() || const_input_idxs.empty()) {
      return;
    }

    VLOG(3) << "marking consts for must-be-const inputs of " << node->name();
    for (Edge const* edge : node->in_edges()) {
      if (!edge->IsControlEdge() &&
          absl::c_binary_search(const_input_idxs, edge->dst_input()) &&
          edge_filter(*edge)) {
        // Do not mark IdentityN / While nodes as compile-time const.
        // If the src node of the `pred` is an IdentityN do not mark it as a
        // compile-time const. Only mark the corresponding input to the
        // IdentityN/While node as a const. XLA IdentityN op simply forwards its
        // inputs so this is safe; loop invariance is checked elsewhere.
        while (edge_filter(*edge) && IsConstTraversableOpType(edge->src())) {
          status = edge->src()->input_edge(edge->src_output(), &edge);
          if (!status.ok()) return;
        }
        if (edge_filter(*edge)) {
          VLOG(4) << "  input " << edge->dst_input() << ": "
                  << edge->src()->name() << " must be const (is "
                  << edge->src()->type_string() << ")";
          (*compile_time_const_nodes)[edge->src()->id()] = true;
        }
      }
    }
  };

  // Post-order traversal visits nodes in reverse topological order for an
  // acyclic graph.
  DFS(g, /*enter=*/{}, /*leave=*/visit, NodeComparatorName{},
      [](const Edge& edge) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_8(mht_8_v, 481, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "lambda");
 return !edge.src()->IsNextIteration(); });
  if (compile_time_const_arg_indices && !edge_filter_input) {
    VLOG(5) << "Setting the cache on the graph: " << &g;
    g.GetConstArgIndicesCache() = *compile_time_const_arg_indices;
  }
  return status;
}

Status GetCompileTimeConstInputs(const OpKernel* op_kernel,
                                 std::vector<int>* const_input_idxs,
                                 FunctionLibraryRuntime* flib_runtime) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysisDTcc mht_9(mht_9_v, 494, "", "./tensorflow/compiler/tf2xla/const_analysis.cc", "GetCompileTimeConstInputs");

  return GetCompileTimeConstInputs(op_kernel->def(), op_kernel,
                                   /*op_def=*/nullptr, const_input_idxs,
                                   flib_runtime);
}

}  // namespace tensorflow
