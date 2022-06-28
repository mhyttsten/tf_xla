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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/function_utils.h"

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace function_utils {

FunctionDefTensorDesc::FunctionDefTensorDesc(const string& node_name,
                                             const string& output, int position)
    : node_name(node_name), node_output(output), position(position) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   mht_0_v.push_back("output: \"" + output + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FunctionDefTensorDesc::FunctionDefTensorDesc");

  full_str = strings::StrCat(node_name, ":", node_output, ":", position);
}

FunctionDefTensorDesc::FunctionDefTensorDesc(const string& input) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FunctionDefTensorDesc::FunctionDefTensorDesc");

  // Parses node_name:node_output:position string into its components.
  full_str = input;
  StringPiece capture;
  StringPiece remaining;

  // Parse "node_name"
  if (strings::Scanner(input)
          .One(strings::Scanner::LETTER_DIGIT_DOT_UNDERSCORE)
          .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_name = string(capture.data(), capture.size());
  }

  // Parse "node_output" if it exists
  if (strings::Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .One(strings::Scanner::LETTER)
          .Any(strings::Scanner::LETTER_DIGIT_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_output = string(capture.data(), capture.size());
  }

  // Parse "position" if it exists
  if (strings::Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .Many(strings::Scanner::DIGIT)
          .GetResult(nullptr, &capture)) {
    CHECK(strings::safe_strto32(capture, &position));
  }
}

// TODO(rachelim): Create a utility class similar to MutableGraphView for
// FunctionDefs, and use that to manipulate functions. It'll be more
// performant if we kept mappings of nodes->inputs/outputs, so that we don't
// have to search over all nodes each time.
// Note that we're not using GrapplerFunctionItem because it doesn't cover
// some of our desired uses (eg changing the outputs of a function), and the
// FunctionDef -> GraphDef conversion isn't really necessary in this case.
void ReplaceReferences(const string& from, const string& to,
                       FunctionDef* func) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("from: \"" + from + "\"");
   mht_2_v.push_back("to: \"" + to + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "ReplaceReferences");

  for (NodeDef& n : *func->mutable_node_def()) {
    std::replace(n.mutable_input()->begin(), n.mutable_input()->end(), from,
                 to);
  }

  for (auto& p : *func->mutable_ret()) {
    if (p.second == from) {
      p.second = to;
    }
  }
}

void AddFunctionOutputWithUniqueName(StringPiece prefix,
                                     StringPiece output_tensor_name,
                                     FunctionDef* fdef, DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_3(mht_3_v, 274, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "AddFunctionOutputWithUniqueName");

  string name = string(prefix);
  int id = fdef->signature().output_arg_size();
  while (ContainsFunctionOutputWithName(name, *fdef)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  auto* output = fdef->mutable_signature()->mutable_output_arg()->Add();
  output->set_name(name);
  output->set_type(dtype);

  (*fdef->mutable_ret())[name] = string(output_tensor_name);
}

OpDef_ArgDef* AddFunctionInput(const string& name, FunctionDef* fdef,
                               DataType dtype) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "AddFunctionInput");

  auto* input_arg = fdef->mutable_signature()->mutable_input_arg()->Add();
  input_arg->set_type(dtype);
  input_arg->set_name(name);

  return input_arg;
}

NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 FunctionDef* fd) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "AddNode");

  NodeDef* node = fd->add_node_def();
  if (!name.empty()) {
    node->set_name(string(name));
  } else {
    SetUniqueFunctionNodeName(op, fd, node);
  }
  node->set_op(string(op));
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (const auto& attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  return node;
}

bool ContainsFunctionNodeWithName(StringPiece name,
                                  const FunctionDef& function) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_6(mht_6_v, 328, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "ContainsFunctionNodeWithName");

  return FindFunctionNodeWithName(name, function) != -1;
}

bool ContainsFunctionNodeWithOp(StringPiece op, const FunctionDef& function) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_7(mht_7_v, 335, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "ContainsFunctionNodeWithOp");

  return FindFunctionNodeWithOp(op, function) != -1;
}

bool ContainsFunctionOutputWithName(StringPiece name,
                                    const FunctionDef& function) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_8(mht_8_v, 343, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "ContainsFunctionOutputWithName");

  return FindFunctionOutputWithName(name, function) != -1;
}

int FindFunctionInputWithName(StringPiece name, const FunctionDef& function) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_9(mht_9_v, 350, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FindFunctionInputWithName");

  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const OpDef_ArgDef& arg) { return arg.name() == name; },
      function.signature().input_arg());
}

int FindFunctionOutputWithName(StringPiece name, const FunctionDef& function) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_10(mht_10_v, 359, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FindFunctionOutputWithName");

  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const OpDef_ArgDef& arg) { return arg.name() == name; },
      function.signature().output_arg());
}

int FindFunctionNodeWithName(StringPiece name, const FunctionDef& function) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_11(mht_11_v, 368, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FindFunctionNodeWithName");

  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      function.node_def());
}

int FindFunctionNodeWithOp(StringPiece op, const FunctionDef& function) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_12(mht_12_v, 377, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "FindFunctionNodeWithOp");

  return graph_utils::GetFirstElementIndexWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; },
      function.node_def());
}

void SetUniqueFunctionNodeName(StringPiece prefix, FunctionDef* function,
                               NodeDef* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_13(mht_13_v, 387, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "SetUniqueFunctionNodeName");

  string name = string(prefix);
  int id = function->node_def_size();
  while (ContainsFunctionNodeWithName(name, *function)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  node->set_name(std::move(name));
}

bool IsFunctionStateful(const FunctionLibraryDefinition& library,
                        const FunctionDef& function_def, bool skip_assert) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_14(mht_14_v, 401, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "IsFunctionStateful");

  if (!function_def.signature().is_stateful()) return false;

  for (const NodeDef& node_def : function_def.node_def()) {
    if (IsNodeStateful(library, node_def, skip_assert)) return true;
  }
  return false;
}

bool IsNodeStateful(const FunctionLibraryDefinition& library,
                    const NodeDef& node, bool skip_assert) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfunction_utilsDTcc mht_15(mht_15_v, 414, "", "./tensorflow/core/grappler/optimizers/data/function_utils.cc", "IsNodeStateful");

  const OpDef* op_def;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);

  if (!s.ok()) return true;

  if (!op_def->is_stateful()) return false;

  if (skip_assert && op_def->name() == "Assert") {
    return false;
  }

  if (op_def->name() == "If") {
    const FunctionDef* then_func =
        library.Find(node.attr().at("then_branch").func().name());
    const FunctionDef* else_func =
        library.Find(node.attr().at("else_branch").func().name());
    if ((then_func != nullptr &&
         !IsFunctionStateful(library, *then_func, skip_assert)) &&
        (else_func != nullptr &&
         !IsFunctionStateful(library, *else_func, skip_assert))) {
      return false;
    }
  }

  if (op_def->name() == "While") {
    const FunctionDef* cond_func =
        library.Find(node.attr().at("cond").func().name());
    const FunctionDef* body_func =
        library.Find(node.attr().at("body").func().name());
    if ((cond_func != nullptr &&
         !IsFunctionStateful(library, *cond_func, skip_assert)) &&
        (body_func != nullptr &&
         !IsFunctionStateful(library, *body_func, skip_assert))) {
      return false;
    }
  }
  return true;
}

}  // namespace function_utils
}  // namespace grappler
}  // namespace tensorflow
