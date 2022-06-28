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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_loggingDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_loggingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_loggingDTcc() {
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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Clears the device field of all ops in the graph.
Status InsertLogging(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_loggingDTcc mht_0(mht_0_v, 201, "", "./tensorflow/tools/graph_transforms/insert_logging.cc", "InsertLogging");

  std::unordered_set<string> ops;
  bool has_ops;
  if (context.params.count("op")) {
    has_ops = true;
    for (const string& op : context.params.at("op")) {
      ops.insert(op);
    }
  } else {
    has_ops = false;
  }

  std::unordered_set<string> prefixes;
  bool has_prefixes;
  if (context.params.count("prefix")) {
    has_prefixes = true;
    for (const string& prefix : context.params.at("prefix")) {
      prefixes.insert(prefix);
    }
  } else {
    has_prefixes = false;
  }

  string message;
  TF_RETURN_IF_ERROR(context.GetOneStringParameter("message", "", &message));

  bool show_name;
  TF_RETURN_IF_ERROR(
      context.GetOneBoolParameter("show_name", false, &show_name));

  bool show_op;
  TF_RETURN_IF_ERROR(context.GetOneBoolParameter("show_op", false, &show_op));

  int32_t first_n;
  TF_RETURN_IF_ERROR(context.GetOneInt32Parameter("first_n", -1, &first_n));

  int32_t summarize;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("summarize", 1024, &summarize));

  std::unordered_map<string, std::set<int>> node_outputs;
  for (const NodeDef& node : input_graph_def.node()) {
    for (const string& input : node.input()) {
      const string canonical_input = CanonicalInputName(input);
      string prefix;
      string name;
      string suffix;
      NodeNamePartsFromInput(canonical_input, &prefix, &name, &suffix);
      const string output_index_string = suffix.substr(1, suffix.size() - 1);
      int32_t output_index;
      if (!strings::safe_strto32(output_index_string, &output_index)) {
        return errors::InvalidArgument("Couldn't understand output number in ",
                                       input);
      }
      node_outputs[name].insert(output_index);
    }
  }

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> ignore_when_renaming;
  GraphDef logged_graph_def;
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = logged_graph_def.mutable_node()->Add();
    *new_node = node;
    if (node_outputs[node.name()].empty()) {
      // There were no outputs found to this node, so skip it.
      continue;
    }
    const bool op_matches = (ops.count(node.op()) > 0);
    bool prefix_matches = false;
    for (const string& prefix : prefixes) {
      if (absl::StartsWith(node.name(), prefix)) {
        prefix_matches = true;
      }
    }
    // If we're not looking for ops, or we found the right op, and if we're not
    // looking for prefixes or we found the right prefix, then add logging here.
    if ((!has_ops || op_matches) && (!has_prefixes || prefix_matches)) {
      const string name_suffix = "__print__";
      DataTypeVector input_types;
      DataTypeVector output_types;
      TF_RETURN_IF_ERROR(GetInOutTypes(node, &input_types, &output_types));
      NodeDef* print_node = logged_graph_def.mutable_node()->Add();
      print_node->set_op("Print");
      print_node->set_name(strings::StrCat(node.name(), name_suffix));
      string node_message;
      if (show_op) {
        node_message += ";" + node.op() + ";";
      }
      if (show_name) {
        node_message += ";" + print_node->name() + ";";
      }
      node_message += message;
      SetNodeAttr("message", node_message, print_node);
      SetNodeAttr("first_n", first_n, print_node);
      SetNodeAttr("summarize", summarize, print_node);
      print_node->add_input(node.name() + ":0");
      SetNodeAttr("T", output_types[0], print_node);
      for (int output_index : node_outputs[node.name()]) {
        print_node->add_input(strings::StrCat(node.name(), ":", output_index));
      }
      SetNodeAttr("U", output_types, print_node);
      ignore_when_renaming.insert(print_node->name());
      // Rewrite the graph so all references to the first input of the original
      // op now pull from the print op instead, so it's executed.
      inputs_to_rename[node.name() + ":0"] =
          strings::StrCat(node.name(), name_suffix, ":0");
    }
  }

  output_graph_def->Clear();
  return RenameNodeInputs(logged_graph_def, inputs_to_rename,
                          ignore_when_renaming, output_graph_def);
}

REGISTER_GRAPH_TRANSFORM("insert_logging", InsertLogging);

}  // namespace graph_transforms
}  // namespace tensorflow
