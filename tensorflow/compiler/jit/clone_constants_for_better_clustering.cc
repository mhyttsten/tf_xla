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
class MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc() {
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

#include "tensorflow/compiler/jit/clone_constants_for_better_clustering.h"

#include <string>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

using se::port::StatusOr;

class CloneConstantsForBetterClusteringPassImpl {
 public:
  explicit CloneConstantsForBetterClusteringPassImpl(Graph* graph)
      : graph_(graph), unique_name_counter_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPassImpl");
}
  Status Run();

 private:
  Status CloneSmallHostConstantInputs(
      const absl::flat_hash_set<string>& name_set, Node* n);
  string GenerateUniqueName(const absl::flat_hash_set<string>& name_set,
                            absl::string_view prefix);
  se::port::StatusOr<Node*> CloneNode(
      const absl::flat_hash_set<string>& name_set, Node* n);

  Graph* graph_;
  int unique_name_counter_;
};

string CloneConstantsForBetterClusteringPassImpl::GenerateUniqueName(
    const absl::flat_hash_set<string>& name_set, absl::string_view prefix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPassImpl::GenerateUniqueName");

  string candidate;
  do {
    candidate = absl::StrCat(prefix, "/clone_", unique_name_counter_++);
  } while (name_set.contains(candidate));
  return candidate;
}

StatusOr<Node*> CloneConstantsForBetterClusteringPassImpl::CloneNode(
    const absl::flat_hash_set<string>& name_set, Node* n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPassImpl::CloneNode");

  NodeDef new_in_def = n->def();
  new_in_def.clear_input();
  new_in_def.set_name(GenerateUniqueName(name_set, new_in_def.name()));
  TF_ASSIGN_OR_RETURN(Node * new_in, graph_->AddNode(new_in_def));

  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(e->src(), new_in);
    } else {
      graph_->AddEdge(e->src(), e->src_output(), new_in, e->dst_input());
    }
  }

  new_in->set_assigned_device_name(n->assigned_device_name());
  return new_in;
}

namespace {
StatusOr<bool> IsConstantOnHost(Node* n) {
  if (n->output_type(0) == DT_INT32) {
    // TensorFlow always puts int32 tensors on the host.
    return true;
  }

  DeviceNameUtils::ParsedName parsed;
  TF_RET_CHECK(
      DeviceNameUtils::ParseFullName(n->assigned_device_name(), &parsed));
  return parsed.type == DEVICE_CPU;
}

StatusOr<bool> IsConstantSmall(Node* n) {
  const TensorProto* proto = nullptr;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "value", &proto));

  // TODO(sanjoy): It may make sense to combine this threshold with XLA's "large
  // constant" threshold, if there is one.
  const int kSmallTensorThreshold = 16;
  int64_t total_elements = 1;
  for (const auto& dim : proto->tensor_shape().dim()) {
    if (dim.size() < 0) {
      return errors::Internal("Unknown dimension size in constant tensor ",
                              n->name());
    }
    total_elements *= dim.size();
  }
  return total_elements < kSmallTensorThreshold;
}

// We only clone host constants for now since we want to avoid increasing memory
// pressure on GPUs.
StatusOr<bool> IsSmallHostConstant(Node* n) {
  if (!n->IsConstant()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool is_constant_on_host, IsConstantOnHost(n));
  if (!is_constant_on_host) {
    return false;
  }

  return IsConstantSmall(n);
}

bool IsInPlaceOp(absl::string_view op_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_3(mht_3_v, 303, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "IsInPlaceOp");

  return op_name == "InplaceUpdate" || op_name == "InplaceAdd" ||
         op_name == "InplaceSub";
}
}  // namespace

Status CloneConstantsForBetterClusteringPassImpl::CloneSmallHostConstantInputs(
    const absl::flat_hash_set<string>& name_set, Node* n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_4(mht_4_v, 313, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPassImpl::CloneSmallHostConstantInputs");

  std::vector<const Edge*> in_edges;
  // Get the edges and sort them so we clone in a deterministic order.
  absl::c_copy(n->in_edges(), std::back_inserter(in_edges));
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    return e1->id() < e2->id();
  });
  for (const Edge* e : in_edges) {
    Node* input = e->src();
    TF_ASSIGN_OR_RETURN(bool is_small_host_constant,
                        IsSmallHostConstant(input));
    if (is_small_host_constant && input->out_edges().size() != 1) {
      VLOG(2) << "Cloning small host constant " << input->name();
      TF_ASSIGN_OR_RETURN(Node* const input_cloned, CloneNode(name_set, input));
      if (e->IsControlEdge()) {
        graph_->AddControlEdge(input_cloned, e->dst());
      } else {
        int dst_input = e->dst_input();
        TF_RET_CHECK(e->src_output() == 0)
            << "expected constant to have exactly one non-control output, but "
               "found output index = "
            << e->src_output();
        graph_->RemoveEdge(e);
        graph_->AddEdge(input_cloned, 0, n, dst_input);
      }
    }
  }
  return Status::OK();
}

Status CloneConstantsForBetterClusteringPassImpl::Run() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_5(mht_5_v, 346, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPassImpl::Run");

  absl::flat_hash_set<string> name_set;
  absl::c_transform(graph_->nodes(), std::inserter(name_set, name_set.begin()),
                    [](Node* n) { return n->name(); });
  std::vector<Node*> nodes;
  for (Node* n : graph_->nodes()) {
    // We rely on the immutability of Tensors to safely clone Const operations.
    // However, "in place" ops do not respect the immutability of Tensors so we
    // avoid this transformation when such ops are present in the graph.
    //
    // In-place operations are problematic because they break the semantic
    // illusion that tensorflow::Tensor instances are immutable.  For instance
    // if we have the following graph:
    //
    // digraph {
    //   SRC -> Const
    //   SRC -> I
    //   SRC -> V
    //   Const -> Identity
    //   Const -> InplaceAdd [label="x"]
    //   I -> InplaceAdd [label="i"]
    //   V -> InplaceAdd [label="v"]
    //   InplaceAdd -> Identity [style=dotted]
    // }
    //
    // then the value produced by `Identity` is Const+I*V since InplaceAdd
    // modifies the tensor in place.  However, if we clone `Const` and turn the
    // graph into:
    //
    // digraph {
    //   SRC -> "Const/clone_1"
    //   SRC -> "Const/clone_2"
    //   SRC -> I
    //   SRC -> V
    //   "Const/clone_1" -> Identity
    //   "Const/clone_2" -> InplaceAdd [label="x"]
    //   I -> InplaceAdd [label="i"]
    //   V -> InplaceAdd [label="v"]
    //   InplaceAdd -> Identity [style=dotted]
    // }
    //
    // then `Identity` no longer produces Const+I*V because the InplaceAdd
    // operation only modifies Const/clone_2 in place.

    if (IsInPlaceOp(n->type_string())) {
      return Status::OK();
    }
    nodes.push_back(n);
  }

  // Iterate over a copy of the nodes to avoid iterating over g->nodes() while
  // creating more nodes.
  for (Node* n : nodes) {
    TF_RETURN_IF_ERROR(CloneSmallHostConstantInputs(name_set, n));
  }
  return Status::OK();
}

Status CloneConstantsForBetterClusteringPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clusteringDTcc mht_6(mht_6_v, 408, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering.cc", "CloneConstantsForBetterClusteringPass::Run");

  if (GetGlobalJitLevelForGraph(options) == OptimizerOptions::OFF) {
    return Status::OK();
  }

  Graph* g = options.graph->get();

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("before_clone_constants_for_better_clustering", *g);
  }

  TF_RETURN_IF_ERROR(CloneConstantsForBetterClusteringPassImpl{g}.Run());

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_clone_constants_for_better_clustering", *g);
  }

  return Status::OK();
}

}  // namespace tensorflow
