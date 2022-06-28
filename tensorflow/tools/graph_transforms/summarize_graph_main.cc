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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc() {
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

// This program prints out a summary of a GraphDef file's contents, listing
// things that are useful for debugging and reusing the model it contains. For
// example it looks at the graph structure and op types to figure out likely
// input and output nodes, and shows which ops are used by the graph. To use it,
// run something like this:
//
// bazel build tensorflow/tools/graph_transforms:summarize_graph
// bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
// --in_graph=my_graph.pb

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
namespace {

void PrintNodeInfo(const NodeDef* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_0(mht_0_v, 212, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "PrintNodeInfo");

  string shape_description = "None";
  if (node->attr().count("shape")) {
    TensorShapeProto shape_proto = node->attr().at("shape").shape();
    Status shape_status = PartialTensorShape::IsValidShape(shape_proto);
    if (shape_status.ok()) {
      shape_description = PartialTensorShape(shape_proto).DebugString();
    } else {
      shape_description = shape_status.error_message();
    }
  }
  DataType dtype = DT_INVALID;
  if (node->attr().count("dtype")) {
    dtype = node->attr().at("dtype").type();
  }
  std::cout << "(name=" << node->name();
  std::cout << ", type=" << DataTypeString(dtype) << "(" << dtype << ")";
  std::cout << ", shape=" << shape_description << ") ";
}

void PrintBenchmarkUsage(const std::vector<const NodeDef*>& placeholders,
                         const std::vector<const NodeDef*>& variables,
                         const std::vector<const NodeDef*> outputs,
                         const string& graph_path) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("graph_path: \"" + graph_path + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_1(mht_1_v, 239, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "PrintBenchmarkUsage");

  std::vector<const NodeDef*> all_inputs(placeholders);
  all_inputs.insert(all_inputs.end(), variables.begin(), variables.end());

  std::vector<string> input_layers;
  std::vector<string> input_layer_types;
  std::vector<string> input_layer_shapes;
  for (const NodeDef* node : all_inputs) {
    input_layers.push_back(node->name());
    DataType dtype = DT_INVALID;
    if (node->attr().count("dtype")) {
      dtype = node->attr().at("dtype").type();
    }
    input_layer_types.push_back(DataTypeString(dtype));
    std::vector<int64_t> sizes;
    PartialTensorShape shape;
    if (node->attr().count("shape")) {
      TensorShapeProto shape_proto = node->attr().at("shape").shape();
      if (PartialTensorShape::IsValid(shape_proto)) {
        shape = PartialTensorShape(shape_proto);
      }
    }
    string sizes_string;
    if (shape.dims() == -1) {
      // Unknown shapes can have -1 for dims, so leave these blank.
      sizes_string = "";
    } else {
      sizes.reserve(shape.dims());
      for (int i = 0; i < shape.dims(); ++i) {
        sizes.push_back(shape.dim_size(i));
      }
      sizes_string = absl::StrJoin(sizes, ",");
    }
    input_layer_shapes.push_back(sizes_string);
  }
  std::vector<string> output_layers;
  output_layers.reserve(outputs.size());
  for (const NodeDef* node : outputs) {
    output_layers.push_back(node->name());
  }
  string input_layer_value = absl::StrJoin(input_layers, ",");
  string input_layer_type_value = absl::StrJoin(input_layer_types, ",");
  string input_layer_shape_value = absl::StrJoin(input_layer_shapes, ":");
  string output_layer_value = absl::StrJoin(output_layers, ",");

  std::cout << "To use with tensorflow/tools/benchmark:benchmark_model try "
               "these arguments:"
            << std::endl;
  std::cout << "bazel run tensorflow/tools/benchmark:benchmark_model --";
  std::cout << " --graph=" << graph_path;
  std::cout << " --show_flops";
  std::cout << " --input_layer=" << input_layer_value;
  std::cout << " --input_layer_type=" << input_layer_type_value;
  std::cout << " --input_layer_shape=" << input_layer_shape_value;
  std::cout << " --output_layer=" << output_layer_value;
  std::cout << std::endl;
}

Status PrintStructure(const GraphDef& graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_2(mht_2_v, 300, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "PrintStructure");

  GraphDef sorted_graph;
  TF_RETURN_IF_ERROR(SortByExecutionOrder(graph, &sorted_graph));
  for (const NodeDef& node : sorted_graph.node()) {
    std::cout << node.name() << " (" << node.op() << "): ["
              << absl::StrJoin(node.input(), ", ") << "]";
    if (node.op() == "Const") {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        std::cout << ", value=" << tensor.DebugString();
      } else {
        LOG(WARNING) << "Decoding Tensor failed for node" << node.name();
      }
    }
    std::cout << std::endl;
  }
  return Status::OK();
}

Status SummarizeGraph(const GraphDef& graph, const string& graph_path,
                      bool print_structure) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("graph_path: \"" + graph_path + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_3(mht_3_v, 325, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "SummarizeGraph");

  std::vector<const NodeDef*> placeholders;
  std::vector<const NodeDef*> variables;
  for (const NodeDef& node : graph.node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
    if (node.op() == "Variable" || node.op() == "VariableV2") {
      variables.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    std::cout << "No inputs spotted." << std::endl;
  } else {
    std::cout << "Found " << placeholders.size() << " possible inputs: ";
    for (const NodeDef* node : placeholders) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }

  if (variables.empty()) {
    std::cout << "No variables spotted." << std::endl;
  } else {
    std::cout << "Found " << variables.size() << " variables: ";
    for (const NodeDef* node : variables) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }

  std::map<string, std::vector<const NodeDef*>> output_map;
  MapNodesToOutputs(graph, &output_map);
  std::vector<const NodeDef*> outputs;
  std::unordered_set<string> unlikely_output_types = {"Const", "Assign", "NoOp",
                                                      "Placeholder"};
  for (const NodeDef& node : graph.node()) {
    if ((output_map.count(node.name()) == 0) &&
        (unlikely_output_types.count(node.op()) == 0)) {
      outputs.push_back(&node);
    }
  }

  if (outputs.empty()) {
    std::cout << "No outputs spotted." << std::endl;
  } else {
    std::cout << "Found " << outputs.size() << " possible outputs: ";
    for (const NodeDef* node : outputs) {
      std::cout << "(name=" << node->name();
      std::cout << ", op=" << node->op() << ") ";
    }
    std::cout << std::endl;
  }

  int64_t const_parameter_count = 0;
  int64_t variable_parameter_count = 0;
  int control_edge_count = 0;
  std::map<string, int> device_counts;
  for (const NodeDef& node : graph.node()) {
    for (const string& input : node.input()) {
      if (input.substr(0, 1) == "^") {
        ++control_edge_count;
      }
    }
    if (!node.device().empty()) {
      ++device_counts[node.device()];
    }
    if ((node.op() == "Const") || (node.op() == "Variable") ||
        (node.op() == "VariableV2")) {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        const size_t num_elements = tensor.NumElements();
        if (node.op() == "Const") {
          const_parameter_count += num_elements;
        } else {
          variable_parameter_count += num_elements;
        }
      } else {
        LOG(WARNING) << "Decoding Tensor failed for node" << node.name();
      }
    }
  }

  std::cout << "Found " << const_parameter_count << " ("
            << strings::HumanReadableNum(const_parameter_count)
            << ") const parameters, " << variable_parameter_count << " ("
            << strings::HumanReadableNum(variable_parameter_count)
            << ") variable parameters, and " << control_edge_count
            << " control_edges" << std::endl;
  if (!device_counts.empty()) {
    for (const auto& device_info : device_counts) {
      std::cout << device_info.second << " nodes assigned to device '"
                << device_info.first << "'";
    }
  }

  std::vector<std::pair<string, string>> invalid_inputs;
  FindInvalidInputs(graph, &invalid_inputs);
  if (!invalid_inputs.empty()) {
    for (const std::pair<string, string>& invalid_input : invalid_inputs) {
      std::cout << "Invalid input " << invalid_input.second << " for node "
                << invalid_input.first << std::endl;
    }
    return errors::Internal(
        "Invalid graph with inputs referring to nonexistent nodes");
  }

  std::map<string, int> op_counts;
  for (const NodeDef& node : graph.node()) {
    ++op_counts[node.op()];
  }
  for (const FunctionDef& function : graph.library().function()) {
    for (const NodeDef& node : function.node_def()) {
      ++op_counts[node.op()];
    }
  }
  std::vector<std::pair<string, int>> op_counts_vec(op_counts.begin(),
                                                    op_counts.end());
  std::sort(op_counts_vec.begin(), op_counts_vec.end(),
            [](std::pair<string, int> a, std::pair<string, int> b) {
              return (a.second > b.second);
            });
  std::cout << "Op types used: ";
  bool is_first = true;
  for (const std::pair<string, int>& op_count : op_counts_vec) {
    if (!is_first) {
      std::cout << ", ";
    } else {
      is_first = false;
    }
    std::cout << op_count.second << " " << op_count.first;
  }
  std::cout << std::endl;

  PrintBenchmarkUsage(placeholders, variables, outputs, graph_path);

  if (print_structure) {
    TF_RETURN_IF_ERROR(PrintStructure(graph));
  }

  return Status::OK();
}

int ParseFlagsAndSummarizeGraph(int argc, char* argv[]) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_4(mht_4_v, 473, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "ParseFlagsAndSummarizeGraph");

  string in_graph = "";
  bool print_structure = false;
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph, "input graph file name"),
      Flag("print_structure", &print_structure,
           "whether to print the network connections of the graph"),
  };
  string usage = Flags::Usage(argv[0], flag_list);

  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << ".\n" << usage;
    return -1;
  }
  if (in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }

  GraphDef graph_def;
  Status load_status = LoadTextOrBinaryGraphFile(in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph << "' failed with "
               << load_status.error_message();
    LOG(ERROR) << usage;
    return -1;
  }

  Status summarize_result =
      SummarizeGraph(graph_def, in_graph, print_structure);
  if (!summarize_result.ok()) {
    LOG(ERROR) << summarize_result.error_message() << "\n" << usage;
    return -1;
  }

  return 0;
}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsummarize_graph_mainDTcc mht_5(mht_5_v, 526, "", "./tensorflow/tools/graph_transforms/summarize_graph_main.cc", "main");

  return tensorflow::graph_transforms::ParseFlagsAndSummarizeGraph(argc, argv);
}
