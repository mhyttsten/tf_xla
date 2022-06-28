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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc() {
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

#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {
namespace grappler {

const NodeScopeAndName ParseNodeScopeAndName(const string& node_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "ParseNodeScopeAndName");

  auto pos = node_name.find_last_of('/');
  if (pos == string::npos) {
    return {"", node_name};
  } else {
    return {node_name.substr(0, pos), node_name.substr(pos + 1)};
  }
};

Status GetInputNode(const GraphOptimizerContext& ctx, const string& input,
                    NodeDef** node) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "GetInputNode");

  string node_name = NodeName(input);
  NodeDef* node_by_name = ctx.node_map->GetNode(node_name);
  if (node_by_name == nullptr) {
    return errors::FailedPrecondition("Node ", node_name,
                                      " doesn't exists in a node map");
  }
  *node = node_by_name;
  return Status::OK();
}

Status GetTensorProperties(const GraphOptimizerContext& ctx,
                           const string& tensor,
                           const OpInfo::TensorProperties** properties) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tensor: \"" + tensor + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "GetTensorProperties");

  if (ctx.graph_properties == nullptr) {
    return errors::InvalidArgument("Graph properties are unknown.");
  }

  // TODO(ezhulenev): Make it TensorId when graph properties will support
  // absl::string_view lookup.
  SafeTensorId tensor_id = ParseTensorName(tensor);

  if (tensor_id.index() < 0) {
    return errors::InvalidArgument(
        "Can't get tensor properties of control dependency ", tensor);
  }

  const auto& output_properties =
      ctx.graph_properties->GetOutputProperties(tensor_id.node());
  int num_outputs = output_properties.size();

  if (num_outputs == 0 || tensor_id.index() > num_outputs - 1) {
    return errors::InvalidArgument(
        "Node ", tensor_id.node(),
        " is missing output properties at position :", tensor_id.index(),
        " (num_outputs=", num_outputs, ")");
  }

  *properties = &output_properties[tensor_id.index()];
  return Status::OK();
}

NodeDef* AddCopyNode(const GraphOptimizerContext& ctx, const string& name,
                     const NodeDef* node_to_copy) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "AddCopyNode");

  CHECK(node_to_copy != nullptr);
  CHECK(!ctx.node_map->NodeExists(name))
      << "Node " << name << " already exists in a graph";
  NodeDef* new_node = ctx.optimized_graph->add_node();
  *new_node = *node_to_copy;
  new_node->set_name(name);
  ctx.node_map->AddNode(name, new_node);
  return new_node;
}

NodeDef* AddEmptyNode(const GraphOptimizerContext& ctx, const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "AddEmptyNode");

  std::string new_name = name;
  for (int count = 0; ctx.node_map->NodeExists(new_name); ++count) {
    LOG(WARNING) << name << " already exists in the graph.";
    new_name = absl::StrCat(name, "_", count);
  }
  NodeDef* new_node = ctx.optimized_graph->add_node();
  new_node->set_name(new_name);
  ctx.node_map->AddNode(new_name, new_node);
  return new_node;
}

const string MakeOptimizedNodeName(const NodeScopeAndName& node,
                                   const string& sub_scope,
                                   const string& prefix) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("sub_scope: \"" + sub_scope + "\"");
   mht_5_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "MakeOptimizedNodeName");

  CHECK(!sub_scope.empty() || !prefix.empty())
      << "Either optimized node name prefix or sub-scope must be non-empty";
  string optimized_node_name;
  if (!node.scope.empty()) {
    strings::StrAppend(&optimized_node_name, node.scope, "/");
  }
  if (!sub_scope.empty()) {
    strings::StrAppend(&optimized_node_name, sub_scope, "/");
  }
  if (!prefix.empty()) {
    strings::StrAppend(&optimized_node_name, prefix, "_");
  }
  strings::StrAppend(&optimized_node_name, node.name);
  return optimized_node_name;
}

const string MakeOptimizedNodeName(const NodeScopeAndName& root,
                                   const std::vector<string> node_names,
                                   const string& sub_scope,
                                   const string& prefix) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("sub_scope: \"" + sub_scope + "\"");
   mht_6_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stageDTcc mht_6(mht_6_v, 316, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage.cc", "MakeOptimizedNodeName");

  string optimized_node_name = MakeOptimizedNodeName(root, sub_scope, prefix);
  for (const string& node_name : node_names) {
    auto name_and_scope = ParseNodeScopeAndName(node_name);
    strings::StrAppend(&optimized_node_name, "_", name_and_scope.name);
  }
  return optimized_node_name;
}

}  // end namespace grappler
}  // end namespace tensorflow
