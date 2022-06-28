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
class MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc() {
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
#include "tensorflow/core/graph/graph_node_util.h"

#include <vector>

#include "absl/container/btree_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

string SummarizeNode(const Node& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/graph/graph_node_util.cc", "SummarizeNode");
 return SummarizeNodeDef(node.def()); }

string FormatNodeForError(const Node& node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/graph/graph_node_util.cc", "FormatNodeForError");

  return FormatNodeDefForError(node.def());
}

Status NameRangesForNode(const Node& node, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_2(mht_2_v, 210, "", "./tensorflow/core/graph/graph_node_util.cc", "NameRangesForNode");

  return NameRangesForNode(node.def(), op_def, inputs, outputs);
}

Status AttachDef(const Status& status, const Node& node,
                 bool allow_multiple_formatted_node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_3(mht_3_v, 218, "", "./tensorflow/core/graph/graph_node_util.cc", "AttachDef");

  return AttachDef(status, node.def(), allow_multiple_formatted_node);
}

absl::btree_set<string> GetMergedNames(const std::vector<string>& from_names,
                                       const std::vector<string>& to_names) {
  absl::btree_set<string> merged_names;
  merged_names.insert(from_names.begin(), from_names.end());
  merged_names.insert(to_names.begin(), to_names.end());
  return merged_names;
}

void MergeDebugInfo(const NodeDebugInfo& from, Node* to_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/graph/graph_node_util.cc", "MergeDebugInfo");

  NodeDebugInfo to = NodeDebugInfo(*to_node);
  if (!from.original_node_names.empty()) {
    auto node_names =
        GetMergedNames(from.original_node_names, to.original_node_names);
    to_node->set_original_node_names({node_names.begin(), node_names.end()});
  }
  if (!from.original_func_names.empty()) {
    auto func_names =
        GetMergedNames(from.original_func_names, to.original_func_names);
    to_node->set_original_func_names({func_names.begin(), func_names.end()});
  }
}

void MergeDebugInfo(const NodeDebugInfo& from, NodeDef* to_node_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/graph/graph_node_util.cc", "MergeDebugInfo");

  NodeDebugInfo to = NodeDebugInfo(*to_node_def);
  if (!from.original_node_names.empty()) {
    auto node_names =
        GetMergedNames(from.original_node_names, to.original_node_names);
    to_node_def->mutable_experimental_debug_info()->clear_original_node_names();
    *to_node_def->mutable_experimental_debug_info()
         ->mutable_original_node_names() = {node_names.begin(),
                                            node_names.end()};
  }
  if (!from.original_func_names.empty()) {
    auto func_names =
        GetMergedNames(from.original_func_names, to.original_func_names);
    to_node_def->mutable_experimental_debug_info()->clear_original_func_names();
    *to_node_def->mutable_experimental_debug_info()
         ->mutable_original_func_names() = {func_names.begin(),
                                            func_names.end()};
  }
}

void MergeDebugInfo(const NodeDef& from_node_def, NodeDef* to_node_def) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_node_utilDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/graph/graph_node_util.cc", "MergeDebugInfo");

  MergeDebugInfo(NodeDebugInfo(from_node_def), to_node_def);
}
}  // namespace tensorflow
