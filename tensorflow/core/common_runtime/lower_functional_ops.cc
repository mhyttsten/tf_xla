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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc() {
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

#include "tensorflow/core/common_runtime/lower_functional_ops.h"

#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_case_op.h"
#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr;
constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

constexpr const char* const kTpuReplicateAttr = "_tpu_replicate";
constexpr const char* const kXlaClusterAttr = "_xla_compile_id";

// Checks if boolean attribute is defined and it's value is 'true'.
bool CheckBoolAttr(const Node* n, absl::string_view attr_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "CheckBoolAttr");

  bool match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && match;
}

// Checks if string attribute is defined and it's not empty.
bool CheckStringAttr(const Node* n, absl::string_view attr_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "CheckStringAttr");

  string match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && !match.empty();
}

bool LowerUsingSwitchMergeIsOn(const Node* n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "LowerUsingSwitchMergeIsOn");

  return CheckBoolAttr(n, kLowerUsingSwitchMergeAttr);
}

bool LowerAsMultiDeviceFunctionIsOn(const Node* n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "LowerAsMultiDeviceFunctionIsOn");

  return CheckBoolAttr(n, kLowerAsMultiDeviceFunctionAttr);
}

bool MarkedForTpuCompilation(const Node* n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_4(mht_4_v, 246, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "MarkedForTpuCompilation");

  return CheckStringAttr(n, kTpuReplicateAttr);
}

bool MarkedForXlaCompilation(const Node* n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "MarkedForXlaCompilation");

  return CheckStringAttr(n, kXlaClusterAttr);
}

bool HasArgsOrRetvals(const Graph& g) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_6(mht_6_v, 260, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "HasArgsOrRetvals");

  for (const Node* n : g.op_nodes()) {
    if (n->IsArg() || n->IsRetval()) return true;
  }
  return false;
}

}  // namespace

Status LowerFunctionalOpsPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_functional_opsDTcc mht_7(mht_7_v, 273, "", "./tensorflow/core/common_runtime/lower_functional_ops.cc", "LowerFunctionalOpsPass::Run");

  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Lowering If/While ops should happen before partitioning.");
  }
  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* g = options.graph->get();
  if (g == nullptr) {
    return errors::Internal(
        "Lowering While op requires a graph to be available.");
  }

  FunctionLibraryDefinition* flib_def = options.flib_def;
  if (flib_def == nullptr) {
    return errors::Internal(
        "Lowering If op requires a FunctionLibraryDefinition to be available.");
  }

  // Lower function calls only if it's explicitly enabled in session options.
  const bool lower_function_calls =
      options.session_options && options.session_options->config.graph_options()
                                     .optimizer_options()
                                     .do_function_inlining();

  // If graph is a function instantiation, it will have `_Arg` and `_Retval`
  // nodes for input and output tensors. Otherwise it's unsafe to remove any of
  // the nodes, because they might be later used as fetches.
  //
  // When we do not keep lowered nodes fetchable, we still add a NoOp node to
  // the graph with the same name as lowered node, because it might be used as a
  // control output source, and it's currently not expressed in a graph.
  bool keep_lowered_nodes_fetchable = !HasArgsOrRetvals(*g);

  // We disable lowering control flow to switch/merge variants when requested,
  // and for the single-threaded executor and TFRT runtime, which does not
  // support it.
  const bool functional_control_flow =
      options.session_options &&
      (options.session_options->config.experimental().executor_type() ==
           "SINGLE_THREADED_EXECUTOR" ||
       options.session_options->config.experimental().use_tfrt() ||
       options.session_options->config.experimental()
           .disable_functional_ops_lowering());

  // Returns true if `node` will be used for XLA compilation.
  const auto used_by_xla = [](Node* node) -> bool {
    return MarkedForTpuCompilation(node) || MarkedForXlaCompilation(node);
  };

  // Returns true if control flow `node` should be lowered to Switch/Merge.
  const auto lower_control_flow = [&](Node* node) -> bool {
    return LowerUsingSwitchMergeIsOn(node) && !used_by_xla(node);
  };

  // Lower all If, Case, While ops that have the `kLowerUsingSwitchMergeAttr`
  // attr set and inline all function calls into the graph.
  // We start at `i` = 2 to skip the source and sink nodes.
  // Note that `g->num_node_ids()` may change in the for body if a matching If,
  // Case, While node is lowered. Since new graph nodes are always added to the
  // end of the list of nodes it is ensured that nested If/Case/While nodes will
  // be lowered as well.
  for (int i = 2; i < g->num_node_ids(); ++i) {
    Node* n = g->FindNodeId(i);
    if (n == nullptr) continue;  // deleted node

    // Always lower function calls produced by lowering If/While nodes.
    if (IsFunctionCall(*flib_def, *n) && !used_by_xla(n) &&
        (lower_function_calls || LowerAsMultiDeviceFunctionIsOn(n))) {
      TF_RETURN_IF_ERROR(RewriteFunctionCallNode(n, g, *flib_def,
                                                 keep_lowered_nodes_fetchable));
      continue;
    }

    // If we are allowed to used function control flow, we do not need to check
    // for If/While/Case nodes in the graph.
    if (functional_control_flow) continue;

    if (n->IsIfNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(RewriteIfNode(n, g, keep_lowered_nodes_fetchable));

    } else if (n->IsCaseNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(RewriteCaseNode(n, g, keep_lowered_nodes_fetchable));

    } else if (n->IsWhileNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(
          RewriteWhileNode(n, g, flib_def, keep_lowered_nodes_fetchable));

    } else {
      DCHECK(!lower_control_flow(n))
          << "Node " << FormatNodeForError(*n) << " of type "
          << n->type_string() << " has '"
          << LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr
          << "' attr set but it does not support lowering.\n";
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 10,
                      LowerFunctionalOpsPass);

}  // namespace tensorflow
