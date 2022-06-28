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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include <cstddef>

#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

constexpr char kConstOpName[] = "Const";
constexpr char kRetValOp[] = "_Retval";

constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
constexpr char kToutputTypes[] = "Toutput_types";

template <typename Predicate, typename Collection>
std::vector<int> GetElementIndicesWithPredicate(const Predicate& predicate,
                                                const Collection& collection) {
  std::vector<int> indices = {};
  unsigned idx = 0;
  for (auto&& element : collection) {
    if (predicate(element)) {
      indices.push_back(idx);
    }
    idx++;
  }
  return indices;
}

std::vector<int> CreateNameIndex(const GraphDef& graph) {
  std::map<string, int> names;
  for (int i = 0; i < graph.node_size(); ++i) {
    names[graph.node(i).name()] = i;
  }
  std::vector<int> index(graph.node_size());
  int i = 0;
  for (const auto& pair : names) {
    index[i++] = pair.second;
  }
  return index;
}

std::vector<int> CreateInputIndex(const NodeDef& node) {
  std::map<string, int> inputs;
  for (int i = 0; i < node.input_size(); ++i) {
    inputs[node.input(i)] = i;
  }
  std::vector<int> index(node.input_size());
  int i = 0;
  for (const auto& pair : inputs) {
    index[i++] = pair.second;
  }
  return index;
}

NodeDef* AddScalarConstNodeHelper(
    DataType dtype, const std::function<void(TensorProto*)>& add_value,
    MutableGraphView* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_0(mht_0_v, 251, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNodeHelper");

  NodeDef node;
  node.set_op(kConstOpName);
  SetUniqueGraphNodeName(kConstOpName, graph->graph(), &node);

  (*node.mutable_attr())["dtype"].set_type(dtype);
  std::unique_ptr<tensorflow::TensorProto> tensor =
      tensorflow::MakeUnique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      tensorflow::MakeUnique<tensorflow::TensorShapeProto>();
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  tensor->set_dtype(dtype);
  add_value(tensor.get());
  (*node.mutable_attr())["value"].set_allocated_tensor(tensor.release());

  return graph->AddNode(std::move(node));
}

}  // namespace

NodeDef* AddScalarPlaceholder(DataType dtype, MutableGraphView* graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_1(mht_1_v, 274, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarPlaceholder");

  NodeDef node;
  node.set_op("Placeholder");
  SetUniqueGraphNodeName(node.op(), graph->graph(), &node);
  (*node.mutable_attr())["dtype"].set_type(dtype);
  TensorShapeProto* shape = (*node.mutable_attr())["shape"].mutable_shape();
  shape->set_unknown_rank(false);
  return graph->AddNode(std::move(node));
}

NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 MutableGraphView* graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddNode");

  NodeDef node;
  if (!name.empty()) {
    node.set_name(string(name));
  } else {
    SetUniqueGraphNodeName(op, graph->graph(), &node);
  }
  node.set_op(string(op));
  for (const string& input : inputs) {
    node.add_input(input);
  }
  for (const auto& attr : attributes) {
    (*node.mutable_attr())[attr.first] = attr.second;
  }
  return graph->AddNode(std::move(node));
}

template <>
NodeDef* AddScalarConstNode(bool v, MutableGraphView* graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_3(mht_3_v, 311, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_BOOL, [v](TensorProto* proto) { proto->add_bool_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(double v, MutableGraphView* graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_4(mht_4_v, 320, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_DOUBLE, [v](TensorProto* proto) { proto->add_double_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(float v, MutableGraphView* graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_5(mht_5_v, 329, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_FLOAT, [v](TensorProto* proto) { proto->add_float_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(int v, MutableGraphView* graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_INT32, [v](TensorProto* proto) { proto->add_int_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(int64_t v, MutableGraphView* graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_7(mht_7_v, 347, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_INT64, [v](TensorProto* proto) { proto->add_int64_val(v); }, graph);
}

template <>
NodeDef* AddScalarConstNode(StringPiece v, MutableGraphView* graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_8(mht_8_v, 356, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "AddScalarConstNode");

  return AddScalarConstNodeHelper(
      DT_STRING,
      [v](TensorProto* proto) { proto->add_string_val(v.data(), v.size()); },
      graph);
}

Status GetScalarConstNodeValueHelper(
    const NodeDef& node, DataType dtype,
    const std::function<void(const Tensor&)>& get_value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_9(mht_9_v, 368, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetScalarConstNodeValueHelper");

  if (node.op() != kConstOpName)
    return errors::InvalidArgument("Node ", node.name(),
                                   " is not a Const node. Op: ", node.op());

  Tensor tensor;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &tensor));
  if (!TensorShapeUtils::IsScalar(tensor.shape())) {
    return errors::InvalidArgument(
        "Node ", node.name(),
        " should be a scalar but has shape: ", tensor.shape());
  }

  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument(
        "Node ", node.name(), " should have type ", DataTypeString(dtype),
        " but has type: ", DataTypeString(tensor.dtype()));
  }

  get_value(tensor);

  return Status::OK();
}

template <>
Status GetScalarConstNodeValue(const NodeDef& node, int64_t* value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_10(mht_10_v, 396, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetScalarConstNodeValue");

  return GetScalarConstNodeValueHelper(
      node, DT_INT64,
      [value](const Tensor& tensor) { *value = tensor.scalar<int64_t>()(); });
}

template <>
Status GetScalarConstNodeValue(const NodeDef& node, bool* value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_11(mht_11_v, 406, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetScalarConstNodeValue");

  return GetScalarConstNodeValueHelper(
      node, DT_BOOL,
      [value](const Tensor& tensor) { *value = tensor.scalar<bool>()(); });
}

bool Compare(const GraphDef& g1, const GraphDef& g2) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_12(mht_12_v, 415, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "Compare");

  if (g1.node_size() != g2.node_size()) {
    return false;
  }
  std::vector<int> name_index1 = CreateNameIndex(g1);
  std::vector<int> name_index2 = CreateNameIndex(g2);
  for (int i = 0; i < g1.node_size(); ++i) {
    int idx1 = name_index1[i];
    int idx2 = name_index2[i];
    if (g1.node(idx1).op() != g2.node(idx2).op()) {
      return false;
    }
    if (g1.node(idx1).name() != g2.node(idx2).name()) {
      return false;
    }
    if (g1.node(idx1).input_size() != g2.node(idx2).input_size()) {
      return false;
    }
    std::vector<int> input_index1 = CreateInputIndex(g1.node(idx1));
    std::vector<int> input_index2 = CreateInputIndex(g2.node(idx2));
    for (int j = 0; j < g1.node(idx1).input_size(); ++j) {
      if (!IsSameInput(g1.node(idx1).input(input_index1[j]),
                       g2.node(idx2).input(input_index2[j]))) {
        return false;
      }
    }
  }
  return true;
}

bool ContainsGraphFunctionWithName(StringPiece name,
                                   const FunctionDefLibrary& library) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_13(mht_13_v, 449, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "ContainsGraphFunctionWithName");

  return FindGraphFunctionWithName(name, library) != -1;
}

bool ContainsGraphNodeWithName(StringPiece name, const GraphDef& graph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_14(mht_14_v, 456, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "ContainsGraphNodeWithName");

  return FindGraphNodeWithName(name, graph) != -1;
}

bool ContainsNodeWithOp(StringPiece op, const GraphDef& graph) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_15(mht_15_v, 463, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "ContainsNodeWithOp");

  return FindGraphNodeWithOp(op, graph) != -1;
}

int FindGraphFunctionWithName(StringPiece name,
                              const FunctionDefLibrary& library) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_16(mht_16_v, 471, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "FindGraphFunctionWithName");

  return GetFirstElementIndexWithPredicate(
      [&name](const FunctionDef& function) {
        return function.signature().name() == name;
      },
      library.function());
}

int FindGraphNodeWithName(StringPiece name, const GraphDef& graph) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_17(mht_17_v, 482, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "FindGraphNodeWithName");

  return GetFirstElementIndexWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      graph.node());
}

int FindGraphNodeWithOp(StringPiece op, const GraphDef& graph) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_18(mht_18_v, 491, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "FindGraphNodeWithOp");

  return GetFirstElementIndexWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; }, graph.node());
}

std::vector<int> FindAllGraphNodesWithOp(const string& op,
                                         const GraphDef& graph) {
  return GetElementIndicesWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; }, graph.node());
}

NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_19(mht_19_v, 505, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetInputNode");

  if (node.input_size() == 0) return nullptr;
  MutableGraphView::InputPort input_port = graph.GetInputPort(node.name(), 0);
  return graph.GetRegularFanin(input_port).node;
}

NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph,
                      int64_t i) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_20(mht_20_v, 515, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetInputNode");

  if (node.input_size() <= i) return nullptr;
  MutableGraphView::InputPort input_port = graph.GetInputPort(node.name(), i);
  return graph.GetRegularFanin(input_port).node;
}

Status GetDatasetOutputTypesAttr(const NodeDef& node,
                                 DataTypeVector* output_types) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_21(mht_21_v, 525, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetDatasetOutputTypesAttr");

  // We don't name the output_types attr consistently, so should check for both.
  for (const string& attr_name : {"output_types", "Toutput_types"}) {
    if (node.attr().contains(attr_name)) {
      return GetNodeAttr(node, attr_name, output_types);
    }
  }
  return errors::InvalidArgument("Could not find output_types attr for node: ",
                                 node.name(), " with op: ", node.op());
}

void SetUniqueGraphNodeName(StringPiece prefix, GraphDef* graph,
                            NodeDef* node) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_22(mht_22_v, 540, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "SetUniqueGraphNodeName");

  string name = string(prefix);
  int id = graph->node_size();
  while (ContainsGraphNodeWithName(name, *graph)) {
    if (name.rfind("_generated") != string::npos &&
        (name.rfind("_generated") == (name.size() - strlen("_generated")))) {
      name.insert(name.rfind("_generated"), strings::StrCat("/_", id));
    } else {
      name = strings::StrCat(prefix, "/_", id);
    }
    ++id;
  }
  node->set_name(std::move(name));
}

void SetUniqueGraphFunctionName(StringPiece prefix,
                                const FunctionDefLibrary* library,
                                FunctionDef* function) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_23(mht_23_v, 560, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "SetUniqueGraphFunctionName");

  string name = string(prefix);
  int id = library->function_size();
  while (ContainsGraphFunctionWithName(name, *library)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  function->mutable_signature()->set_name(std::move(name));
}

void CopyAttribute(const string& attribute_name, const NodeDef& from,
                   NodeDef* to_node) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("attribute_name: \"" + attribute_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_24(mht_24_v, 575, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "CopyAttribute");

  (*to_node->mutable_attr())[attribute_name] = from.attr().at(attribute_name);
}

void ConcatAttributeList(const string& attribute_name, const NodeDef& first,
                         const NodeDef& second, NodeDef* to_node) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("attribute_name: \"" + attribute_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_25(mht_25_v, 584, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "ConcatAttributeList");

  CopyAttribute(attribute_name, first, to_node);
  (*to_node->mutable_attr())
      .at(attribute_name)
      .mutable_list()
      ->MergeFrom(second.attr().at(attribute_name).list());
}

Status EnsureNodeNamesUnique(Graph* g) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_26(mht_26_v, 595, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "EnsureNodeNamesUnique");

  // Modeled after Scope::Impl::GetUniqueName
  std::unordered_map<string, int> name_map;

  for (auto node : g->op_nodes()) {
    const string& prefix = node->name();
    if (auto entry = gtl::FindOrNull(name_map, prefix)) {
      string unique_name;
      do {
        unique_name = strings::StrCat(prefix, "_", ++(*entry));
      } while (name_map.find(unique_name) != name_map.end());
      name_map.insert({unique_name, 0});
      node->set_name(std::move(unique_name));
    } else {
      name_map.insert({node->name(), 0});
    }
  }

  return Status::OK();
}

Status GetFetchNode(const MutableGraphView& graph, const GrapplerItem& item,
                    NodeDef** fetch_node) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_27(mht_27_v, 620, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "GetFetchNode");

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  *fetch_node = graph.GetNode(item.fetch.at(0));

  return Status::OK();
}

bool IsItemDerivedFromFunctionDef(const GrapplerItem& item,
                                  const MutableGraphView& graph_view) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_28(mht_28_v, 636, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "IsItemDerivedFromFunctionDef");

  for (const auto& fetch_name : item.fetch) {
    auto fetch = graph_view.GetNode(fetch_name);
    if (fetch != nullptr && fetch->op() != kRetValOp) {
      // We found a fetch node which is not a `Retval` op.
      return false;
    }
  }
  // All fetch nodes are `Retval` ops (or we don't have any fetch nodes).
  return true;
}

void MaybeSetFusedMetadata(const NodeDef& node1, const NodeDef& node2,
                           NodeDef* fused_node) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_29(mht_29_v, 652, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "MaybeSetFusedMetadata");

  data::Metadata metadata1;
  if (node1.attr().contains("metadata")) {
    metadata1.ParseFromString(node1.attr().at("metadata").s());
  }
  data::Metadata metadata2;
  if (node2.attr().contains("metadata")) {
    metadata2.ParseFromString(node2.attr().at("metadata").s());
  }
  data::Metadata fused_metadata;
  auto normalize_name = [](const string& name) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_30(mht_30_v, 666, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "lambda");

    return name.empty() ? "?" : name;
  };
  *fused_metadata.mutable_name() =
      strings::StrCat("fused(", normalize_name(metadata1.name()), ",",
                      normalize_name(metadata2.name()), ")");
  fused_metadata.SerializeToString(
      (*fused_node->mutable_attr())["metadata"].mutable_s());
}

bool CopyShapesAndTypesAttrs(const NodeDef& from, NodeDef* to_node) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_31(mht_31_v, 679, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "CopyShapesAndTypesAttrs");

  auto* attr = gtl::FindOrNull(from.attr(), kOutputTypes);
  attr = (attr == nullptr ? gtl::FindOrNull(from.attr(), kToutputTypes) : attr);

  if (attr == nullptr) return false;
  (*to_node->mutable_attr())[kOutputTypes] = *attr;

  attr = gtl::FindOrNull(from.attr(), kOutputShapes);
  if (attr == nullptr) return false;
  (*to_node->mutable_attr())[kOutputShapes] = *attr;
  return true;
}

namespace {
const auto* kSloppyAttrOps = new absl::flat_hash_set<string>{
    "ParallelInterleaveDatasetV2",
    "ParallelMapDataset",
    "ParseExampleDataset",
};

const auto* kDeterministicAttrOps = new absl::flat_hash_set<string>{
    "LegacyParallelInterleaveDatasetV2",
    "ParallelInterleaveDatasetV3",
    "ParallelInterleaveDatasetV4",
    "ParallelMapDatasetV2",
    "ParallelBatchDataset",
};
}  // anonymous namespace

bool HasSloppyAttr(const string& op) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_32(mht_32_v, 712, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "HasSloppyAttr");
 return kSloppyAttrOps->contains(op); }

bool HasDeterministicAttr(const string& op) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_33(mht_33_v, 718, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "HasDeterministicAttr");

  return kDeterministicAttrOps->contains(op);
}

Status SetMetadataName(const std::string& name, NodeDef* node) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTcc mht_34(mht_34_v, 726, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.cc", "SetMetadataName");

  data::Metadata metadata;
  if (node->attr().contains("metadata")) {
    metadata.ParseFromString(node->attr().at("metadata").s());
  }
  if (!metadata.name().empty()) {
    return errors::InvalidArgument("Node ", node->name(),
                                   " already has a metadata name \"",
                                   metadata.name(), "\".");
  }
  *metadata.mutable_name() = name;
  metadata.SerializeToString((*node->mutable_attr())["metadata"].mutable_s());
  return Status::OK();
}

}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
