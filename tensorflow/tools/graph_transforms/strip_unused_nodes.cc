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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc() {
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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

namespace {

Status TypeForPlaceholder(const TransformFuncContext& context,
                          const string& node_name, DataType* result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc mht_0(mht_0_v, 201, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes.cc", "TypeForPlaceholder");

  // If we don't find anything else, return float.
  *result = DT_FLOAT;

  // Check to see if we have been given a default for all placeholders.
  if (context.params.count("type")) {
    if (context.params.at("type").size() != 1) {
      return errors::InvalidArgument(
          "You must pass no more than one default 'type' to "
          "strip_unused_nodes");
    }
    const string& type_string = context.params.at("type")[0];
    if (!DataTypeFromString(type_string, result)) {
      return errors::InvalidArgument("Couldn't understand type argument '",
                                     type_string, "'");
    }
  }

  // See if there's a particular type specified for this placeholder.
  if (context.params.count("name") || context.params.count("type_for_name")) {
    if (!context.params.count("name") ||
        !context.params.count("type_for_name") ||
        (context.params.at("type_for_name").size() !=
         context.params.at("name").size())) {
      return errors::InvalidArgument(
          "You must pass a 'type_for_name' arg for every 'name', e.g. "
          "strip_unused_nodes(name=foo, type_for_name=float, name=bar, "
          "type_for_name=quint8");
    }
    const int name_count = context.params.at("name").size();
    for (int i = 0; i < name_count; ++i) {
      if (context.params.at("name")[i] == node_name) {
        const string& type_string = context.params.at("type_for_name")[i];
        if (!DataTypeFromString(type_string, result)) {
          return errors::InvalidArgument("Couldn't understand type argument '",
                                         type_string, "'");
        }
      }
    }
  }

  return Status::OK();
}

Status ShapeForPlaceholder(const TransformFuncContext& context,
                           const string& node_name, TensorShape* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc mht_1(mht_1_v, 250, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes.cc", "ShapeForPlaceholder");

  // If we don't find anything else, return scalar.
  *result = {};

  // Check to see if we have been given a default for all placeholders.
  if (context.params.count("shape")) {
    if (context.params.at("shape").size() != 1) {
      return errors::InvalidArgument(
          "You must pass no more than one default 'shape' to "
          "strip_unused_nodes");
    }
    const string& shape_string = context.params.at("shape")[0];
    TF_RETURN_IF_ERROR(TensorShapeFromString(shape_string, result));
  }

  // See if there's a particular type specified for this placeholder.
  if (context.params.count("name") || context.params.count("shape_for_name")) {
    if (!context.params.count("name") ||
        !context.params.count("shape_for_name") ||
        (context.params.at("shape_for_name").size() !=
         context.params.at("name").size())) {
      return errors::InvalidArgument(
          "You must pass a 'shape_for_name' arg for every 'name', e.g. "
          "strip_unused_nodes(name=foo, shape_for_name=\"2,2,1\", name=bar, "
          "shape_for_name=\"1\"");
    }
    const int name_count = context.params.at("name").size();
    for (int i = 0; i < name_count; ++i) {
      if (context.params.at("name")[i] == node_name) {
        const string& shape_string = context.params.at("shape_for_name")[i];
        TF_RETURN_IF_ERROR(TensorShapeFromString(shape_string, result));
      }
    }
  }

  return Status::OK();
}
}  // namespace

// Delete any nodes that don't contribute to the inference result.
Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const TransformFuncContext& context,
                        GraphDef* output_graph_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodesDTcc mht_2(mht_2_v, 295, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes.cc", "StripUnusedNodes");

  std::set<string> required_nodes;
  std::set<string> input_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(NodeNameFromInput(input));
    input_nodes.insert(NodeNameFromInput(input));
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(output);
  }

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(input_graph_def, &node_lookup);

  std::vector<string> current_inputs;
  for (const string& output_name : context.output_names) {
    current_inputs.push_back(NodeNameFromInput(output_name));
  }

  while (!current_inputs.empty()) {
    std::set<string> next_inputs;
    for (const string& current_input : current_inputs) {
      required_nodes.insert(current_input);
      if (input_nodes.count(current_input)) {
        continue;
      }
      if (!node_lookup.count(current_input)) {
        return errors::InvalidArgument("Input node ", current_input,
                                       " not found in graph");
      }
      const NodeDef* current_node = node_lookup[current_input];
      for (const string& input_name : current_node->input()) {
        string input_node_name = NodeNameFromInput(input_name);
        if (!required_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs =
        std::vector<string>(next_inputs.begin(), next_inputs.end());
  }

  GraphDef filtered_graph_def;
  FilterGraphDef(input_graph_def,
                 [&](const NodeDef& node) {
                   return required_nodes.count(node.name()) > 0;
                 },
                 &filtered_graph_def);

  output_graph_def->Clear();
  for (const NodeDef& node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      NodeDef placeholder_node;
      if (node.op() == "Placeholder") {
        placeholder_node = node;
      } else {
        placeholder_node.set_op("Placeholder");
        placeholder_node.set_name(node.name());
        DataType type;
        TF_RETURN_IF_ERROR(TypeForPlaceholder(context, node.name(), &type));
        TensorShape shape;
        TF_RETURN_IF_ERROR(ShapeForPlaceholder(context, node.name(), &shape));
        SetNodeAttr("dtype", type, &placeholder_node);
        SetNodeAttr("shape", shape, &placeholder_node);
      }
      *(output_graph_def->mutable_node()->Add()) = placeholder_node;
    } else {
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("strip_unused_nodes", StripUnusedNodes);

}  // namespace graph_transforms
}  // namespace tensorflow
