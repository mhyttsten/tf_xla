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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinline_partitionedcallDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinline_partitionedcallDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinline_partitionedcallDTcc() {
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

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

constexpr char kPartitionedCallOpName[] = "PartitionedCall";
constexpr char kFunctionAttrName[] = "f";

namespace {
absl::optional<FunctionDef> GetFunctionByNameFromLibrary(
    const GraphDef& graph, absl::string_view function_name) {
  for (const auto& fct : graph.library().function()) {
    if (fct.signature().name() == function_name) {
      return fct;
    }
  }
  return {};
}

std::string NormalizeNodeDefInput(const std::string& input_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinline_partitionedcallDTcc mht_0(mht_0_v, 216, "", "./tensorflow/tools/graph_transforms/inline_partitionedcall.cc", "NormalizeNodeDefInput");

  std::vector<std::string> name_parts =
      absl::StrSplit(input_name, absl::ByChar(':'));
  if (name_parts.size() > 2) {
    return absl::StrCat(name_parts[0], ":", name_parts.back());
  }
  return input_name;
}

}  // namespace

Status InlinePartitionedCall(const GraphDef& input_graph_def,
                             const TransformFuncContext& context,
                             GraphDef* output_graph_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinline_partitionedcallDTcc mht_1(mht_1_v, 232, "", "./tensorflow/tools/graph_transforms/inline_partitionedcall.cc", "InlinePartitionedCall");

  output_graph_def->Clear();
  absl::flat_hash_map<std::string, std::string> remap_input;

  for (const NodeDef& node : input_graph_def.node()) {
    if (node.op() == kPartitionedCallOpName) {
      if (node.attr().count(kFunctionAttrName) == 0) {
        return Status(
            error::Code::NOT_FOUND,
            "Node " + node.name() + " has no attribute: " + kFunctionAttrName);
      }

      if (!node.attr().at(kFunctionAttrName).has_func()) {
        return Status(error::Code::NOT_FOUND,
                      "Cannot figure out function name");
      }
      const std::string function_name =
          node.attr().at(kFunctionAttrName).func().name();
      absl::optional<FunctionDef> function =
          GetFunctionByNameFromLibrary(input_graph_def, function_name);
      if (!function.has_value()) {
        return Status(error::Code::NOT_FOUND,
                      "function " + function_name + " Not found");
      }

      const std::string prefix = node.name();

      const int kOutputArgumentCount =
          function->signature().output_arg().size();
      for (int k = 0; k < kOutputArgumentCount; ++k) {
        const std::string function_arg_output_name =
            function->ret().at(function->signature().output_arg()[k].name());
        remap_input.insert_or_assign(
            CanonicalInputName(absl::StrCat(node.name(), ":", k)),
            absl::StrCat(prefix, "/",
                         NormalizeNodeDefInput(function_arg_output_name)));
      }

      const int kInputArgumentCount = function->signature().input_arg().size();
      if (node.input().size() != kInputArgumentCount) {
        return Status(error::Code::INVALID_ARGUMENT,
                      "Called function  " + function_name +
                          " has invalid input signature.");
      }
      absl::flat_hash_map<std::string, std::string> input_argument_map;
      for (int k = 0; k < kInputArgumentCount; ++k) {
        const std::string canonical_name =
            CanonicalInputName(function->signature().input_arg()[k].name());
        input_argument_map.insert_or_assign(canonical_name, node.input()[k]);
      }

      for (const NodeDef& function_node : function->node_def()) {
        NodeDef* new_node = output_graph_def->mutable_node()->Add();
        *new_node = function_node;
        new_node->set_name(absl::StrCat(prefix, "/", function_node.name()));
        absl::c_transform(
            *new_node->mutable_input(), new_node->mutable_input()->begin(),
            [prefix, input_argument_map](const std::string& input_name) {
              const std::string canonical_input_name =
                  CanonicalInputName(input_name);
              if (input_argument_map.find(canonical_input_name) !=
                  input_argument_map.end()) {
                return input_argument_map.at(canonical_input_name);
              }
              return absl::StrCat(prefix, "/",
                                  NormalizeNodeDefInput(input_name));
            });
      }
    } else {
      NodeDef* new_node = output_graph_def->mutable_node()->Add();
      *new_node = node;
    }
  }

  // Remap PartitionCall outputs to correct nodes.
  for (NodeDef& node : *output_graph_def->mutable_node()) {
    absl::c_transform(
        *node.mutable_input(), node.mutable_input()->begin(),
        [remap_input](const std::string& input_name) {
          const std::string canonical_input_name =
              CanonicalInputName(input_name);
          if (remap_input.find(canonical_input_name) != remap_input.end()) {
            return remap_input.at(canonical_input_name);
          }
          return input_name;
        });
  }
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("inline_partitionedcall", InlinePartitionedCall);
}  // namespace graph_transforms
}  // namespace tensorflow
