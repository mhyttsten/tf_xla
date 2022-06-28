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
class MHTracer_DTPStensorflowPScorePSgraphPScontrol_flowDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPScontrol_flowDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPScontrol_flowDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/control_flow.h"

#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {
// Information about a loop frame structure.
struct Frame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  Frame* parent = nullptr;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  const Node* loop_cond = nullptr;
};

// Verify that the ControlFlowInfo of the graph has valid loop structure.
Status ValidateControlFlowInfo(const Graph* graph,
                               const std::vector<ControlFlowInfo>& cf_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPScontrol_flowDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/graph/control_flow.cc", "ValidateControlFlowInfo");

  std::unordered_map<string, Frame> frames;
  for (const Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];
    if (!cf.frame || !cf.parent_frame) {
      // Skip nodes unreachable from the source node. They might be pruned
      // later.
      continue;
    }

    Frame& frame = frames[cf.frame_name];
    Frame* parent = &frames[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
    } else if (frame.parent != parent) {
      return errors::Internal(
          "Invalid loop structure: Mismatched parent frames for \"",
          cf.frame_name, "\": \"", parent->name, "\" vs \"", frame.parent->name,
          "\". The node giving this error: ", FormatNodeForError(*node),
          ". This is an internal bug, please file a bug report with "
          "instructions on how to reproduce the error.");
    }
    if (IsLoopCond(node)) {
      // ForwardLoopCounter runs in the same frame as the forward loop and
      // BackPropLoopCounter runs in the same frame as the backprop loop. They
      // are the only cases that multiple loops share the same frame.
      if (frame.loop_cond &&
          !absl::StrContains(frame.loop_cond->name(), "LoopCounter") &&
          !absl::StrContains(node->name(), "LoopCounter")) {
        return errors::InvalidArgument(
            "Invalid loop structure: Loop \"", cf.frame_name,
            "\" has more than one LoopCond node: ", FormatNodeForError(*node),
            " and ", FormatNodeForError(*frame.loop_cond),
            ". This is an internal bug, please file a bug report with "
            "instructions on how to reproduce the error.");
      }
      frame.loop_cond = node;
    }
  }
  return Status::OK();
}
}  // namespace

Status BuildControlFlowInfo(const Graph* g, std::vector<ControlFlowInfo>* info,
                            std::vector<string>* unreachable_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPScontrol_flowDTcc mht_1(mht_1_v, 259, "", "./tensorflow/core/graph/control_flow.cc", "BuildControlFlowInfo");

  info->clear();
  info->resize(g->num_node_ids());

  std::vector<const Node*> parent_nodes;
  parent_nodes.resize(g->num_node_ids());

  const Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;

  string frame_name;
  std::deque<const Node*> ready;
  ready.push_back(src_node);
  while (!ready.empty()) {
    const Node* curr_node = ready.front();
    ready.pop_front();
    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;

    if (IsExit(curr_node)) {
      // Exit to the parent frame.
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      const Node* out = out_edge->dst();
      int out_id = out->id();
      ControlFlowInfo* out_info = &(*info)[out_id];
      const Node* out_parent = out_info->parent_frame;
      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        parent_nodes[out->id()] = curr_node;
        ready.push_back(out);
      }

      // Process the node 'out'.
      if (IsEnter(out)) {
        if (is_visited) {
          const string& parent_frame = (*info)[out_parent->id()].frame_name;
          if (parent_frame != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", parent_frame, "'.");
          }
        } else {
          out_info->frame = out;
          out_info->parent_frame = frame;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(out->attrs(), "frame_name", &out_info->frame_name));
          if (out_info->frame_name.empty()) {
            return errors::InvalidArgument("The Enter ",
                                           FormatNodeForError(*out),
                                           " must have a frame name.");
          }
        }
      } else {
        if (is_visited) {
          if (out_info->frame_name != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", out_info->frame_name, "'.");
          }
        } else {
          out_info->frame = frame;
          out_info->parent_frame = parent;
          out_info->frame_name = frame_name;
        }
      }
    }
  }
  if (unreachable_nodes) {
    for (const Node* node : g->op_nodes()) {
      if (!parent_nodes[node->id()]) {
        unreachable_nodes->push_back(node->name());
      }
    }
  }
  TF_RETURN_IF_ERROR(ValidateControlFlowInfo(g, *info));
  return Status::OK();
}

}  // namespace tensorflow
