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
class MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/validate.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace graph {

Status ValidateGraphDef(const GraphDef& graph_def,
                        const OpRegistryInterface& op_registry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/graph/validate.cc", "ValidateGraphDef");

  Status s;
  const int version = graph_def.versions().producer();
  for (const NodeDef& node_def : graph_def.node()) {
    // Look up the OpDef for the node_def's op name.
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(op_registry.LookUpOpDef(node_def.op(), &op_def));
    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
    TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, version));
  }

  return s;
}

Status ValidateGraphDefAgainstOpRegistry(
    const GraphDef& graph_def, const OpRegistryInterface& op_registry) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/graph/validate.cc", "ValidateGraphDefAgainstOpRegistry");

  GraphDef copy(graph_def);
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&copy, op_registry, 0));
  return ValidateGraphDef(copy, op_registry);
}

Status ValidateGraphDefAgainstOpList(const GraphDef& graph_def,
                                     const OpList& op_list) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/graph/validate.cc", "ValidateGraphDefAgainstOpList");

  OpListOpRegistry registry(&op_list);
  return ValidateGraphDefAgainstOpRegistry(graph_def, registry);
}

void GetOpListForValidation(OpList* op_list, const OpRegistry& op_registry) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/graph/validate.cc", "GetOpListForValidation");

  op_registry.Export(false, op_list);
  RemoveDescriptionsFromOpList(op_list);
}

Status ValidateGraphHasNoCycle(const Graph& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/graph/validate.cc", "ValidateGraphHasNoCycle");

  // A node is ready when all of its inputs have been visited.
  std::vector<const Node*> ready;
  std::vector<int> pending_count(graph.num_node_ids(), 0);

  for (int i = 0; i < graph.num_node_ids(); ++i) {
    const Node* n = graph.FindNodeId(i);
    if (n == nullptr) continue;
    pending_count[i] = n->in_edges().size();
    if (n->IsMerge()) {
      // While-loop cycles are legal cycles so we manually adjust the
      // pending_count to make sure that the loop is visited.
      for (const Edge* e : n->in_edges()) {
        if (!e->IsControlEdge() && e->src()->IsNextIteration()) {
          pending_count[i]--;
        }
      }
    }
    if (pending_count[i] == 0) {
      ready.push_back(n);
    }
  }

  int processed = 0;
  while (!ready.empty()) {
    const Node* node = ready.back();
    ready.pop_back();
    ++processed;

    for (const Edge* out : node->out_edges()) {
      const int output_id = out->dst()->id();
      pending_count[output_id]--;
      if (pending_count[output_id] == 0) {
        ready.push_back(out->dst());
      }
    }
  }

  if (processed < graph.num_nodes()) {
    std::vector<string> nodes_in_cycle;
    for (int i = 0; i < pending_count.size() && nodes_in_cycle.size() < 3;
         ++i) {
      if (pending_count[i] != 0) {
        nodes_in_cycle.push_back(graph.FindNodeId(i)->name());
      }
    }
    return errors::InvalidArgument(
        "Graph is invalid, contains a cycle with ",
        graph.num_nodes() - processed,
        " nodes, including: ", absl::StrJoin(nodes_in_cycle, ", "));
  }
  return Status::OK();
}

Status VerifyNoDuplicateNodeNames(const GraphDef& graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSvalidateDTcc mht_5(mht_5_v, 302, "", "./tensorflow/core/graph/validate.cc", "VerifyNoDuplicateNodeNames");

  absl::flat_hash_set<absl::string_view> nodes;
  for (const auto& node : graph.node()) {
    if (nodes.contains(node.name())) {
      return errors::AlreadyExists("Node already exists: ", node.name());
    }
    nodes.insert(node.name());
  }
  return Status::OK();
}

}  // namespace graph
}  // namespace tensorflow
