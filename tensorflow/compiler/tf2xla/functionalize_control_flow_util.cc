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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flow_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flow_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flow_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"

namespace tensorflow {

bool NodeCmpByNameResourcesLast::operator()(const Node* lhs,
                                            const Node* rhs) const {
  bool lhs_is_resource =
      lhs->num_inputs() > 0 ? (lhs->input_type(0) == DT_RESOURCE) : false;
  bool rhs_is_resource =
      rhs->num_inputs() > 0 ? (rhs->input_type(0) == DT_RESOURCE) : false;
  return std::tie(lhs_is_resource, lhs->name()) <
         std::tie(rhs_is_resource, rhs->name());
}

StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index) {
  const char* const kRetValOp = "_Retval";
  NodeDef ret_def;
  ret_def.set_op(kRetValOp);
  ret_def.set_name(absl::StrCat(kRetValOp, index));
  AddNodeAttr("T", type, &ret_def);
  AddNodeAttr("index", index, &ret_def);
  return graph->AddNode(ret_def);
}

Status ExtractWhileLoopFrames(
    const std::vector<ControlFlowInfo>& cf_info, const Graph* graph,
    std::unordered_map<string, WhileLoopFrame>* frames,
    const NodeFilter& node_filter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flow_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow_util.cc", "ExtractWhileLoopFrames");

  for (Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];

    VLOG(2) << "node: " << node->name() << " (" << node->id()
            << ") frame_name: " << cf.frame_name
            << " frame: " << (cf.frame ? cf.frame->name() : "---")
            << " parent_frame: "
            << (cf.parent_frame ? cf.parent_frame->name() : "---");
    TF_RET_CHECK(cf.frame != nullptr && cf.parent_frame != nullptr);

    WhileLoopFrame& frame = (*frames)[cf.frame_name];
    WhileLoopFrame* parent =
        &(*frames)[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
      ++parent->num_children;
    }

    if (IsEnter(node)) {
      WhileLoopArg arg;
      arg.enter = node;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg.enter->attrs(), "is_constant",
                                     &arg.is_loop_invariant));
      frame.args.push_back(arg);
    } else if (IsLoopCond(node)) {
      frame.loop_cond = node;
    }
    frame.nodes.insert(node);
    if (node->IsControlFlow() && node_filter && !node_filter(node)) {
      frame.should_be_functionalized = false;
    }
  }

  return Status::OK();
}

// Check that the graph has no cycle containing the given node.
Status CheckNodeNotInCycle(const Node* node, const int num_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_control_flow_utilDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/tf2xla/functionalize_control_flow_util.cc", "CheckNodeNotInCycle");

  std::vector<const Node*> ready;
  ready.push_back(node);
  std::vector<bool> visited(num_nodes);
  while (!ready.empty()) {
    const Node* current_node = ready.back();
    ready.pop_back();
    visited[current_node->id()] = true;
    for (const Edge* out : current_node->out_edges()) {
      if (out->dst() == node) {
        return errors::Internal("Detected a cycle: ", FormatNodeForError(*node),
                                " (", node->def().op(), ") feeds into itself.");
      } else if (!visited[out->dst()->id()]) {
        ready.push_back(out->dst());
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
