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
class MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc() {
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

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
namespace {

class ClusterScopingPassImpl {
 public:
  ClusterScopingPassImpl(Graph* graph,
                         OptimizerOptions::GlobalJitLevel global_jit_level)
      : graph_(graph),
        global_jit_level_(global_jit_level),
        unique_scope_id_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPassImpl");
}

  Status Run();

 private:
  Status ScopingForPipelineStages();

  size_t GetUniqueScopeId() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "GetUniqueScopeId");
 return unique_scope_id_++; }

  void AddScopeToAllTransitivePredecessors(Node* start);

  void AddScopeToAllTransitiveSuccessors(Node* start);

 private:
  Graph* graph_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  size_t unique_scope_id_;
};

absl::optional<string> GetXlaInternalScope(Node* node) {
  string scope;
  if (GetNodeAttr(node->attrs(), kXlaInternalScopeAttr, &scope).ok()) {
    return scope;
  }

  return absl::nullopt;
}

void SetXlaInternalScope(Node* node, StringPiece scope) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "SetXlaInternalScope");

  node->AddAttr(kXlaInternalScopeAttr, scope);
}

// NB! We append a new scope as suffix to the _XlaInternalScope attribute
// instead of overriding the old value.  In other words, appending scope B to
// scope A creates the conjunction of the scopes A and B (i.e, A & B) and,
// in effect, the node gets both the old and new scopes.  As a unique scope
// disallows a node being merged with nodes in other scopes, the scope
// conjunction preserves the semantic of the old scope (i.e., the node still
// cannot be merged with the previously incompatible nodes.)
//
// For example, the below case should be rare in practice but can serve for the
// purpose of discussion.  After adding scopes for both Stage and Unstage,
// Node_Y will receive both scopes "unstage" and "stage", while Node_X receives
// only scope "stage".  The semantic of scope "unstage" is preserved although
// scope "stage" is later appended.  As a result, Node_X and Node_Y will be put
// into different clusters.
//
//                Unstage -> Node_Y (scope "unstage & stage")
//                              |
//                              V
//  Node_X (scope "stage") -> Stage
//
void AddOrAppendXlaInternalScope(Node* node, absl::string_view suffix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "AddOrAppendXlaInternalScope");

  string updated_scope;
  absl::optional<string> cur_scope = GetXlaInternalScope(node);
  if (cur_scope == absl::nullopt) {
    updated_scope = std::string(suffix);
  } else {
    updated_scope = absl::StrCat(cur_scope.value(), "&", suffix);
  }
  SetXlaInternalScope(node, updated_scope);
}

void ClusterScopingPassImpl::AddScopeToAllTransitivePredecessors(Node* start) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_4(mht_4_v, 280, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPassImpl::AddScopeToAllTransitivePredecessors");

  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_5(mht_5_v, 288, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "lambda");
 AddOrAppendXlaInternalScope(n, unique_suffix); };
  ReverseDFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
                 /*stable_comparator=*/NodeComparatorName());
}

void ClusterScopingPassImpl::AddScopeToAllTransitiveSuccessors(Node* start) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_6(mht_6_v, 296, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPassImpl::AddScopeToAllTransitiveSuccessors");

  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_7(mht_7_v, 304, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "lambda");
 AddOrAppendXlaInternalScope(n, unique_suffix); };
  DFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
          /*stable_comparator=*/NodeComparatorName(),
          // Do not filter any edges to better capture the semantics of
          // transitive closure of successors.  We may revisit this when
          // we see more cases needing cluster scoping in the future.
          /*edge_filter=*/nullptr);
}

// This preserves the parallelism between pipeline stages.  For example, below
// is a typical pattern of input pipelining in Tensorflow and this heuristic
// ensures Node_X and Node_Y are put into different clusters.  Without the
// heuristic, they may be put into the same cluster and it can introduce
// artificial dependencies and incur great performance loss.  In this example,
// Node_Y becomes dependent on IteratorGetNext and the latencies add up if
// Node_X and Node_Y are in the same cluster.
//
// IteratorGetNext -> Node_X -> Stage
//
// Unstage -> Node_Y
//
Status ClusterScopingPassImpl::ScopingForPipelineStages() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_8(mht_8_v, 328, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPassImpl::ScopingForPipelineStages");

  for (Node* n : graph_->nodes()) {
    DCHECK(n);
    if (n->type_string() == "Unstage") {
      AddScopeToAllTransitiveSuccessors(n);
    }
    if (n->type_string() == "Stage") {
      AddScopeToAllTransitivePredecessors(n);
    }
  }

  return Status::OK();
}

Status ClusterScopingPassImpl::Run() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_9(mht_9_v, 345, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPassImpl::Run");

  if (global_jit_level_ == OptimizerOptions::OFF) {
    return Status::OK();
  }

  return ScopingForPipelineStages();
}
}  // namespace

Status ClusterScopingPass::Run(const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_passDTcc mht_10(mht_10_v, 357, "", "./tensorflow/compiler/jit/cluster_scoping_pass.cc", "ClusterScopingPass::Run");

  Graph* graph = options.graph->get();

  return ClusterScopingPassImpl{graph, GetGlobalJitLevelForGraph(options)}
      .Run();
}
}  // namespace tensorflow
