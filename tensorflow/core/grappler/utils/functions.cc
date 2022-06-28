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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc() {
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
#include "tensorflow/core/grappler/utils/functions.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {

GrapplerFunctionItem::GrapplerFunctionItem(
    string func_name, string description, AttrSlice func_attr,
    std::vector<const FunctionDef::ArgAttrs*> arg_attr,
    std::vector<InputArgInstantiation> input_args,
    std::vector<OutputArgInstantiation> output_args,
    std::vector<ControlOutput> control_outputs, const int graph_def_version,
    const bool is_stateful, GraphDef&& function_body)
    : description_(std::move(description)),
      func_attr_(func_attr),
      arg_attr_(std::move(arg_attr)),
      input_args_(std::move(input_args)),
      output_args_(std::move(output_args)),
      control_outputs_(std::move(control_outputs)),
      is_stateful_(is_stateful) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("func_name: \"" + func_name + "\"");
   mht_0_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::GrapplerFunctionItem");

  id = std::move(func_name);
  graph = std::move(function_body);
  graph.mutable_versions()->set_producer(graph_def_version);

  // Fill the feed nodes with function input arguments.
  for (const InputArgInstantiation& input_arg : input_args_) {
    feed.push_back({input_arg.node_name, Tensor()});
  }
  // Fill the fetch nodes with outputs.
  for (const OutputArgInstantiation& output_arg : output_args_) {
    fetch.push_back(output_arg.node_name);
  }
  // We must keep all control output nodes.
  for (const ControlOutput& control_output : control_outputs_) {
    keep_ops.push_back(control_output.node_name);
  }

  // Tensorflow functions execution semantics is different from the main graph,
  // and we need to preserve it when we do graph optimizations.
  optimization_options().allow_pruning_stateful_and_dataset_ops = false;
}

const string& GrapplerFunctionItem::description() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::description");
 return description_; }

const std::vector<InputArgInstantiation>& GrapplerFunctionItem::inputs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::inputs");

  return input_args_;
}

const InputArgInstantiation& GrapplerFunctionItem::input(int i) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::input");

  return input_args_[i];
}

const std::size_t GrapplerFunctionItem::input_size() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::input_size");

  return input_args_.size();
}

const std::vector<OutputArgInstantiation>& GrapplerFunctionItem::outputs()
    const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_5(mht_5_v, 276, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::outputs");

  return output_args_;
}

const OutputArgInstantiation& GrapplerFunctionItem::output(int i) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::output");

  return output_args_[i];
}

const std::size_t GrapplerFunctionItem::output_size() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_7(mht_7_v, 290, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::output_size");

  return output_args_.size();
}

const std::vector<ControlOutput>& GrapplerFunctionItem::control_outputs()
    const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_8(mht_8_v, 298, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::control_outputs");

  return control_outputs_;
}

const std::size_t GrapplerFunctionItem::control_output_size() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_9(mht_9_v, 305, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::control_output_size");

  return control_outputs_.size();
}

const AttrSlice& GrapplerFunctionItem::func_attr() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_10(mht_10_v, 312, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::func_attr");
 return func_attr_; }

const std::vector<const FunctionDef::ArgAttrs*>&
GrapplerFunctionItem::arg_attr() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_11(mht_11_v, 318, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::arg_attr");

  return arg_attr_;
}

const GraphDef& GrapplerFunctionItem::function_body() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_12(mht_12_v, 325, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::function_body");
 return graph; }

GraphDef& GrapplerFunctionItem::mutable_function_body() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_13(mht_13_v, 330, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::mutable_function_body");
 return graph; }

bool GrapplerFunctionItem::is_stateful() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_14(mht_14_v, 335, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::is_stateful");
 return is_stateful_; }

GrapplerFunctionItem& GrapplerFunctionItem::SwapFunctionBody(GraphDef&& other) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_15(mht_15_v, 340, "", "./tensorflow/core/grappler/utils/functions.cc", "GrapplerFunctionItem::SwapFunctionBody");

  graph = std::move(other);
  return *this;
}

bool HasParametrizedType(const FunctionDef& func) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_16(mht_16_v, 348, "", "./tensorflow/core/grappler/utils/functions.cc", "HasParametrizedType");

  const auto is_type_parametrized = [](const OpDef::ArgDef& arg) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_17(mht_17_v, 352, "", "./tensorflow/core/grappler/utils/functions.cc", "lambda");

    return !arg.type_attr().empty() || !arg.number_attr().empty() ||
           !arg.type_list_attr().empty();
  };

  const auto& input = func.signature().input_arg();
  const auto& output = func.signature().output_arg();
  return std::any_of(input.begin(), input.end(), is_type_parametrized) ||
         std::any_of(output.begin(), output.end(), is_type_parametrized);
}

bool HasParametrizedBody(const FunctionDef& func) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_18(mht_18_v, 366, "", "./tensorflow/core/grappler/utils/functions.cc", "HasParametrizedBody");

  const auto is_parametrized = [&](const NodeDef& node) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_19(mht_19_v, 370, "", "./tensorflow/core/grappler/utils/functions.cc", "lambda");

    for (const auto& attr : node.attr()) {
      if (!attr.second.placeholder().empty()) return true;
    }
    return false;
  };
  return std::any_of(func.node_def().begin(), func.node_def().end(),
                     is_parametrized);
}

bool IsParametrized(const FunctionDef& func) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_20(mht_20_v, 383, "", "./tensorflow/core/grappler/utils/functions.cc", "IsParametrized");

  return HasParametrizedType(func) || HasParametrizedBody(func);
}

Status InstantiationTypeParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, DataType>* type_parameters) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_21(mht_21_v, 392, "", "./tensorflow/core/grappler/utils/functions.cc", "InstantiationTypeParameters");

  if (!type_parameters->empty()) {
    return errors::InvalidArgument("Type parameters output map must be empty");
  }

  const auto resolve_type_attr = [&](const OpDef::ArgDef& arg) -> Status {
    if (!arg.type_attr().empty()) {
      DataType dtype;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(func_instantiation_attr, arg.type_attr(), &dtype));
      type_parameters->emplace(arg.type_attr(), dtype);

    } else if (!arg.type_list_attr().empty()) {
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(func_instantiation_attr, arg.type_list_attr(), &dtypes));
      int index = 0;
      for (const DataType& dtype : dtypes) {
        type_parameters->emplace(absl::StrCat(arg.type_list_attr(), ":", index),
                                 dtype);
        ++index;
      }
    }
    return Status::OK();
  };

  for (const auto& input : func.signature().input_arg())
    TF_RETURN_IF_ERROR(resolve_type_attr(input));
  for (const auto& output : func.signature().output_arg())
    TF_RETURN_IF_ERROR(resolve_type_attr(output));

  return Status::OK();
}

Status InstantiationBodyParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, AttrValue>* body_parameters) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_22(mht_22_v, 431, "", "./tensorflow/core/grappler/utils/functions.cc", "InstantiationBodyParameters");

  if (!body_parameters->empty()) {
    return errors::InvalidArgument("Body parameters output map must be empty");
  }

  for (const NodeDef& func_body_node : func.node_def()) {
    for (auto& attr : func_body_node.attr()) {
      const string& placeholder = attr.second.placeholder();

      if (placeholder.empty() || body_parameters->contains(placeholder)) {
        continue;
      }

      const AttrValue* placeholder_value =
          func_instantiation_attr.Find(placeholder);
      if (placeholder_value) {
        body_parameters->insert({placeholder, *placeholder_value});
      } else {
        return errors::InvalidArgument("Can't resolve placeholder: ",
                                       placeholder);
      }
    }
  }

  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrSlice& func_instantiation_attr,
                                const FunctionLibraryDefinition& flib,
                                const int graph_def_version,
                                GrapplerFunctionItem* item) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_23(mht_23_v, 465, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeGrapplerFunctionItem");

  const OpDef& signature = func.signature();

  if (signature.name().empty()) {
    return errors::InvalidArgument("Function name must be specified");
  }

  // Function types will be resolved from function instantiation attributes. All
  // other attributes will be lost during conversion to FunctionDef.
  for (const OpDef::AttrDef& attr : signature.attr()) {
    if (attr.type() != "type") {
      return errors::InvalidArgument(
          "Function signature must have only type attributes");
    }
  }

  // Instantiate function into a statically defined FunctionBody Graph.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(func, func_instantiation_attr, &flib, &fbody));

  GraphDef function_body;
  fbody->graph->ToGraphDef(&function_body);

  // Function body shares the library with the graph that instantiated it. We do
  // not need a full copy of the function library, just the reachable subset.
  *function_body.mutable_library() = flib.ReachableDefinitions(func).ToProto();

  VLOG(3) << absl::Substitute(
      "Deleted $0 unreachable functions from the Grappler function item "
      "instantiation of $1 (library size = $2)",
      flib.num_functions() - function_body.library().function_size(),
      signature.name(), function_body.library().function_size());

  const int num_instantiated_inputs = fbody->arg_types.size();
  const int num_instantiated_outputs = fbody->ret_types.size();

  std::vector<InputArgInstantiation> inputs;
  inputs.reserve(num_instantiated_inputs);

  for (int in_id = 0; in_id < num_instantiated_inputs; ++in_id) {
    const Node* node = fbody->arg_nodes[in_id];
    const DataType& dtype = fbody->arg_types[in_id];
    inputs.emplace_back(node->name(), dtype);
  }

  std::vector<OutputArgInstantiation> outputs;
  outputs.reserve(num_instantiated_outputs);

  for (int out_id = 0; out_id < num_instantiated_outputs; ++out_id) {
    const Node* node = fbody->ret_nodes[out_id];
    const DataType& dtype = fbody->ret_types[out_id];
    outputs.emplace_back(node->name(), dtype);
  }

  // Control outputs ensure that all side-effectful nodes in the function body
  // will execute, even if they are not required to compute regular output args.
  std::vector<ControlOutput> control_outputs;
  control_outputs.reserve(func.control_ret_size());
  for (const auto& control_ret : func.control_ret()) {
    control_outputs.push_back({control_ret.first, control_ret.second});
  }
  // Sort control outputs to keep FunctionDef output stable. The sort order of
  // map entries in func.control_ret() are not stable.
  // See b/174715578 for context on why stability is desired.
  std::sort(control_outputs.begin(), control_outputs.end());

  std::vector<const FunctionDef::ArgAttrs*> arg_attr(inputs.size(), nullptr);
  for (const auto& attr : func.arg_attr()) {
    arg_attr.at(attr.first) = &attr.second;
  }

  *item = GrapplerFunctionItem(
      /*func_name=*/signature.name(),
      /*description=*/signature.description(),
      /*func_attr=*/AttrSlice(&func.attr()), std::move(arg_attr),
      std::move(inputs), std::move(outputs), std::move(control_outputs),
      graph_def_version, signature.is_stateful(), std::move(function_body));
  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const FunctionLibraryDefinition& flib,
                                const int graph_def_version,
                                GrapplerFunctionItem* item) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_24(mht_24_v, 552, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeGrapplerFunctionItem");

  return MakeGrapplerFunctionItem(func, AttrSlice(), flib, graph_def_version,
                                  item);
}

Status ReplaceInputWithConst(const NodeDef& input_const, int input_index,
                             GrapplerFunctionItem* item) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_25(mht_25_v, 561, "", "./tensorflow/core/grappler/utils/functions.cc", "ReplaceInputWithConst");

  if (!IsConstant(input_const)) {
    return errors::InvalidArgument("Input node is not a constant: ",
                                   SummarizeNodeDef(input_const));
  }
  const int item_input_size = item->input_size();
  if (input_index < 0 || input_index >= item_input_size) {
    return errors::InvalidArgument(
        "Function input index is out of bound: index=", input_index,
        " input_size=", item->input_size());
  }

  const InputArgInstantiation& input_arg = item->input(input_index);

  for (NodeDef& node : *item->graph.mutable_node()) {
    // Replace '_Arg' node in the function body with a 'Const' node.
    if (node.name() == input_arg.node_name) {
      node = input_const;
      node.set_name(input_arg.node_name);
      node.clear_input();
      node.clear_device();  // device placement is defined by instantiating node
    }

    // Update index in all inputs after the removed const input.
    if (IsArg(node)) {
      auto attrs = AttrSlice(node);
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "index", &index));
      if (index >= input_index) {
        (*node.mutable_attr())["index"].set_i(index - 1);
      }
    }
  }

  item->input_args_.erase(item->input_args_.begin() + input_index);
  item->arg_attr_.erase(item->arg_attr_.begin() + input_index);

  return Status::OK();
}

Status RemoveFunctionOutputs(const absl::flat_hash_set<int>& remove_outputs,
                             GrapplerFunctionItem* item,
                             std::vector<std::pair<int, int>>* output_mapping) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_26(mht_26_v, 606, "", "./tensorflow/core/grappler/utils/functions.cc", "RemoveFunctionOutputs");

  DCHECK(output_mapping->empty());

  // Do some sanity checking of the removed outputs positions.
  for (int remove_output : remove_outputs) {
    const int item_output_size = item->output_size();
    if (remove_output < 0 || remove_output >= item_output_size) {
      return errors::InvalidArgument(
          "Function output index is out of bound: index=", remove_output,
          " output_size=", item->output_size());
    }
  }

  absl::flat_hash_set<const OutputArgInstantiation*> remove_output_args;
  const auto is_remove_output_arg = [&](const OutputArgInstantiation& output) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_27(mht_27_v, 623, "", "./tensorflow/core/grappler/utils/functions.cc", "lambda");

    return remove_output_args.find(&output) != remove_output_args.end();
  };

  for (int i = 0, end = item->output_size(); i < end; ++i) {
    const OutputArgInstantiation& output = item->output(i);
    if (remove_outputs.contains(i)) {
      VLOG(3) << "Remove functions output: name=" << output.node_name
              << "(index = " << i << ")";
      remove_output_args.insert(&output);
    } else if (!remove_output_args.empty()) {
      // Add output mapping only if output position changed.
      output_mapping->push_back({i, i - remove_output_args.size()});
    }
  }

  // Update 'index' attribute in all '_Retval' nodes that are in output mapping.
  for (NodeDef& node : *item->graph.mutable_node()) {
    if (IsRetval(node)) {
      auto attrs = AttrSlice(node);
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "index", &index));

      for (const auto& mapping : *output_mapping) {
        const int from = mapping.first;
        const int to = mapping.second;
        if (index == from) {
          (*node.mutable_attr())["index"].set_i(to);
        }
      }
    }
  }

  auto& o = item->output_args_;
  o.erase(std::remove_if(o.begin(), o.end(), is_remove_output_arg), o.end());

  return Status::OK();
}

namespace {

// FunctionDef uses different connectivity encoding for the function body nodes,
// than a GraphDef (see function.proto for details). This is a helper class that
// converts inputs in GraphDef format (node[:position]) to the FunctionDef
// format (node:output[:position]).
class MakeFunctionDefHelper {
 public:
  MakeFunctionDefHelper() = default;

  Status Initialize(const GrapplerFunctionItem& item,
                    const FunctionLibraryDefinition& flib);

  // Converts input name from GraphDef format (name[:position]) to the
  // FunctionDef input format (name[:output][:position]) using registered input
  // arg instantiations and function body outputs.
  Status AsFunctionDefInput(const string& graph_def_input,
                            string* func_def_input) const;

  // Updates Node inputs from GraphDef to FunctionDef format.
  Status AsFunctionDefNode(NodeDef* function_body_node) const;

  bool IsInputNode(const NodeDef& node) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_28(mht_28_v, 687, "", "./tensorflow/core/grappler/utils/functions.cc", "IsInputNode");

    return input_nodes_.contains(node.name());
  }

  bool IsOutputNode(const NodeDef& node) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_29(mht_29_v, 694, "", "./tensorflow/core/grappler/utils/functions.cc", "IsOutputNode");

    return output_nodes_.contains(node.name());
  }

 private:
  absl::flat_hash_set<absl::string_view> input_nodes_;
  absl::flat_hash_set<absl::string_view> output_nodes_;
  // Mapping from function body node name to output names range map.
  absl::flat_hash_map<string, tensorflow::NameRangeMap> function_body_outputs_;
};

Status MakeFunctionDefHelper::Initialize(
    const GrapplerFunctionItem& item, const FunctionLibraryDefinition& flib) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_30(mht_30_v, 709, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeFunctionDefHelper::Initialize");

  for (const InputArgInstantiation& input_arg : item.inputs()) {
    input_nodes_.insert(input_arg.node_name);
  }
  for (const OutputArgInstantiation& output_arg : item.outputs()) {
    output_nodes_.insert(output_arg.node_name);
  }

  for (const NodeDef& node : item.function_body().node()) {
    const OpRegistrationData* registration;
    TF_RETURN_IF_ERROR(flib.LookUp(node.op(), &registration));

    tensorflow::NameRangeMap outputs_range_map;
    TF_RETURN_IF_ERROR(tensorflow::NameRangesForNode(
        node, registration->op_def, nullptr, &outputs_range_map));

    function_body_outputs_.emplace(node.name(), std::move(outputs_range_map));
  }

  return Status::OK();
}

Status MakeFunctionDefHelper::AsFunctionDefInput(const string& graph_def_input,
                                                 string* func_def_input) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("graph_def_input: \"" + graph_def_input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_31(mht_31_v, 736, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeFunctionDefHelper::AsFunctionDefInput");

  if (IsControlInput(graph_def_input)) {
    *func_def_input = graph_def_input;
    return Status::OK();
  }

  const SafeTensorId tensor = ParseTensorName(graph_def_input);
  DCHECK_GE(tensor.index(), 0);

  // Graph def input corresponds to one of the function inputs.
  const auto is_input = input_nodes_.find(tensor.node());
  if (is_input != input_nodes_.end()) {
    DCHECK_EQ(tensor.index(), 0);
    *func_def_input = tensor.node();
    return Status::OK();
  }

  // Or it must be output from one of the function body nodes
  const auto is_body_output = function_body_outputs_.find(tensor.node());
  if (is_body_output != function_body_outputs_.end()) {
    const tensorflow::NameRangeMap& outputs_range_map = is_body_output->second;

    for (const auto& el : outputs_range_map) {
      const auto& output_name = el.first;
      const auto& output_range = el.second;
      if (tensor.index() >= output_range.first &&
          tensor.index() < output_range.second) {
        *func_def_input = absl::StrCat(tensor.node(), ":", output_name, ":",
                                       tensor.index() - output_range.first);
        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Unknown graph def input: ", graph_def_input);
}

Status MakeFunctionDefHelper::AsFunctionDefNode(
    NodeDef* function_body_node) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_32(mht_32_v, 777, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeFunctionDefHelper::AsFunctionDefNode");

  string func_def_input;

  for (int i = 0; i < function_body_node->input_size(); ++i) {
    TF_RETURN_IF_ERROR(
        AsFunctionDefInput(function_body_node->input(i), &func_def_input));
    function_body_node->set_input(i, func_def_input);
  }

  return Status::OK();
}

}  // namespace

Status MakeFunctionDef(const GrapplerFunctionItem& item,
                       const FunctionLibraryDefinition& flib,
                       FunctionDef* func) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSfunctionsDTcc mht_33(mht_33_v, 796, "", "./tensorflow/core/grappler/utils/functions.cc", "MakeFunctionDef");

  func->mutable_signature()->set_name(item.id);
  func->mutable_signature()->set_description(item.description());
  func->mutable_signature()->set_is_stateful(item.is_stateful());

  MakeFunctionDefHelper helper;
  TF_RETURN_IF_ERROR(helper.Initialize(item, flib));

  // Mapping from the '_Retval' node name to the output tensor.
  absl::flat_hash_map<absl::string_view, string> output_tensors;
  for (const NodeDef& func_body_node : item.function_body().node()) {
    if (!helper.IsOutputNode(func_body_node)) continue;
    if (func_body_node.input_size() != 1) {
      return errors::Internal("_Retval node must have single input: ",
                              SummarizeNodeDef(func_body_node));
    }
    output_tensors.emplace(func_body_node.name(), func_body_node.input(0));
  }

  for (const InputArgInstantiation& input_arg : item.inputs()) {
    OpDef::ArgDef arg_def;
    arg_def.set_name(input_arg.node_name);
    arg_def.set_type(input_arg.data_type);
    arg_def.set_is_ref(IsRefType(input_arg.data_type));
    *func->mutable_signature()->add_input_arg() = arg_def;
  }

  // Add function output arguments.
  for (const OutputArgInstantiation& output_arg : item.outputs()) {
    const string output_name =
        absl::StrReplaceAll(output_arg.node_name, {{"_RetVal", ""}});

    OpDef::ArgDef arg_def;
    arg_def.set_name(output_name);
    arg_def.set_type(output_arg.data_type);
    arg_def.set_is_ref(IsRefType(output_arg.data_type));
    *func->mutable_signature()->add_output_arg() = arg_def;

    auto it = output_tensors.find(output_arg.node_name);
    if (it == output_tensors.end()) {
      return errors::Internal(
          "Can't find an output tensor for the output node: ",
          output_arg.node_name);
    }

    TF_RETURN_IF_ERROR(helper.AsFunctionDefInput(
        it->second, &(*func->mutable_ret())[output_name]));
  }

  // Add function control outputs.
  for (const ControlOutput& control_out : item.control_outputs()) {
    func->mutable_control_ret()->insert(
        {control_out.output_name, control_out.node_name});
    *func->mutable_signature()->add_control_output() = control_out.output_name;
  }

  // Copy function definition specific attributes.
  for (const auto& attr : item.func_attr()) {
    const auto& attr_name = attr.first;
    const auto& attr_value = attr.second;
    (*func->mutable_attr())[attr_name] = attr_value;
  }

  // Copy function arg attributes.
  for (int i = 0, end = item.arg_attr().size(); i < end; ++i) {
    const auto* attr = item.arg_attr().at(i);
    if (attr != nullptr) {
      (*func->mutable_arg_attr())[i] = *attr;
    }
  }

  // Copy function body nodes to the FunctionDef and update input format
  for (const NodeDef& func_node : item.function_body().node()) {
    // Skip original `_Arg` and `_Retval` nodes. If node was converted to some
    // other type (e.g. inputs converted to placeholders), we need to check that
    // it's not registered as function input or output node.
    if (IsArg(func_node) || IsRetval(func_node) ||
        helper.IsInputNode(func_node) || helper.IsOutputNode(func_node))
      continue;

    NodeDef* func_def_node = func->add_node_def();
    *func_def_node = func_node;
    TF_RETURN_IF_ERROR(helper.AsFunctionDefNode(func_def_node));
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
