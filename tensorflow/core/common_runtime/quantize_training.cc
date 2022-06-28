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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc() {
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

#include "tensorflow/core/common_runtime/quantize_training.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

// TODO(suharshs): If desired, make these values configurable.
const uint32 kAllowedInputs = 2;
const float kEMADecay = 0.999;

// Node types to rewrite. Insert quantize_and_dequantize op for their inputs.
const auto* nodes_to_rewrite =
    new std::unordered_set<string, StringPieceHasher>{"MatMul", "Conv2D"};

// Contains necessary parameters to convert an edge.
struct EdgeToConvert {
  // edge is not owned here.
  const Edge* edge;
  int32 num_bits;
  bool signed_input;
  bool range_given;
  float input_min;
  float input_max;

  EdgeToConvert(const Edge* e, int32_t bits, bool sign, bool range, float min,
                float max)
      : edge(e),
        num_bits(bits),
        signed_input(sign),
        range_given(range),
        input_min(min),
        input_max(max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/common_runtime/quantize_training.cc", "EdgeToConvert");
}
};

// Decide if a node is in backward pass by checking if its name is led by
// "gradients".
// TODO(jmchen): Make this check more robust as it is not guaranteed that the
// forward node will not be named with a leading "gradients".
inline bool IsGradientNode(const Graph* graph, const Node* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/common_runtime/quantize_training.cc", "IsGradientNode");

  static const string tag = "gradients";
  return (node->name().compare(0, tag.size(), tag) == 0);
}

// Find the type of the input to set the parameters for the
// quantize_and_dequantize op.
// Returns true if the root tensor op type is known, false otherwise.
bool FindType(const Graph* graph, const Node* node, bool* signed_input,
              bool* range_given, float* input_min, float* input_max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/common_runtime/quantize_training.cc", "FindType");

  const string& src_op = node->type_string();
  if (src_op == "Const" || src_op == "Variable" || src_op == "VariableV2") {
    *signed_input = true;
    *range_given = false;
  } else if (src_op == "Relu") {
    // Range is not given for Relu.
    *signed_input = false;
    *range_given = false;
  } else if (src_op == "Relu6") {
    // TODO(suharshs): Also the theoretical min and max is 0 and 6, if the
    // actual activations are somewhere in within this range, we can quantize
    // this even further. This is true for other activations like Sigmoid6 too.
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 6;
  } else if (src_op == "Sigmoid") {
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 1;
  } else if (src_op == "Tanh") {
    *signed_input = true;
    *range_given = true;
    *input_min = -1;
    *input_max = 1;
  } else if (src_op == "Reshape" || src_op == "ConcatV2") {
    // Reshape has 2 inputs and the first one is the tensor.
    // ConcatV2 has many inputs but they should all have the same activation
    // function (i.e. Inception). So we just recurse on the first input.
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot && edge->dst_input() == 0) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else if (src_op == "Identity" || src_op == "MaxPool" ||
             src_op == "AvgPool" || src_op == "MaxPool3D" ||
             src_op == "AvgPool3D") {
    // All these Ops only have 1 data input.
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else {
    // Unknown type, could be the model input examples.
    // TODO(jmchen): Set the params for input with user's hint.
    *signed_input = true;
    *range_given = false;
    return false;
  }

  return true;
}

// Find the Save op and inputs.
Status FindSaveOp(const Graph* graph, Node** save_op,
                  std::vector<const Edge*>* in_edges, bool* found) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_3(mht_3_v, 316, "", "./tensorflow/core/common_runtime/quantize_training.cc", "FindSaveOp");

  *found = false;
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == "SaveV2") {
      // We found multiple save ops.
      if (*found) {
        return errors::InvalidArgument("Input graph has multiple SaveV2 ops.");
      }
      *save_op = node;
      *found = true;
      TF_RETURN_IF_ERROR(node->input_edges(in_edges));
    }
  }
  return Status::OK();
}

Node* FindRestoreAllOp(const Graph* graph, StringPiece save_prefix) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_4(mht_4_v, 335, "", "./tensorflow/core/common_runtime/quantize_training.cc", "FindRestoreAllOp");

  for (Node* node : graph->op_nodes()) {
    // The restore_all op should have the same prefix of the save_op.
    if (node->name() == strings::StrCat(save_prefix, "/restore_all")) {
      return node;
    }
  }
  return nullptr;
}

// Strips the last "/suffix" from a name.
// We use this to construct the name of restore ops in the same way they are
// constructed by the Saver.
StringPiece GetNodeNamePrefix(const Node* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_5(mht_5_v, 351, "", "./tensorflow/core/common_runtime/quantize_training.cc", "GetNodeNamePrefix");

  StringPiece name = node->name();
  return name.substr(0, name.rfind('/'));
}

void FillStringTensor(Tensor* dst, const Tensor& src) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_6(mht_6_v, 359, "", "./tensorflow/core/common_runtime/quantize_training.cc", "FillStringTensor");

  auto dst_flat = dst->flat<tstring>();
  auto src_flat = src.flat<tstring>();
  for (int i = 0; i < src.NumElements(); i++) {
    dst_flat(i) = src_flat(i);
  }
}

// Add the added_variables as an inputs to the Save op.
// We change the inputs of the SaveV2 op to include the names of the added
// variables. We also add the variables as inputs to the save op.
Status ConnectVariablesToSaveOp(Graph* graph, Node* save_op,
                                const std::vector<const Edge*>& in_edges,
                                const std::vector<Node*>& added_variables) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_7(mht_7_v, 375, "", "./tensorflow/core/common_runtime/quantize_training.cc", "ConnectVariablesToSaveOp");

  Node* tensor_names_op = in_edges[1]->src();
  Node* shape_and_slices_op = in_edges[2]->src();

  // Get the tensor_names and shape_and_slices tensors from the const op.
  Tensor tensor_names;
  Tensor shape_and_slices;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_names_op->attrs(), "value", &tensor_names));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slices_op->attrs(), "value", &shape_and_slices));

  int tn_size = tensor_names.NumElements();
  int var_size = added_variables.size();

  // Create a new save_op that has inputs to all the new variables.
  NodeBuilder save_op_builder =
      NodeBuilder(save_op->name(), save_op->type_string());
  // The first three inputs are prefix, tensor_names, and shapes_and_slices.
  for (int i = 0; i < 3; i++) {
    save_op_builder = save_op_builder.Input(in_edges[i]->src());
  }
  std::vector<NodeBuilder::NodeOut> var_nodeouts;
  var_nodeouts.reserve(tn_size + var_size);
  // The rest of the inputs need to be used the construct the tensor list arg.
  for (int i = 3; i < in_edges.size(); i++) {
    var_nodeouts.emplace_back(in_edges[i]->src());
  }

  // Add the new values to the tensors and the op input.
  Tensor new_tensor_names(DT_STRING, TensorShape({tn_size + var_size}));
  Tensor new_shape_and_slices(DT_STRING, TensorShape({tn_size + var_size}));
  FillStringTensor(&new_tensor_names, tensor_names);
  FillStringTensor(&new_shape_and_slices, shape_and_slices);
  for (int i = 0; i < var_size; i++) {
    Node* var = added_variables[i];
    new_tensor_names.flat<tstring>()(tn_size + i) = var->name();
    new_shape_and_slices.flat<tstring>()(tn_size + i) = "";
    var_nodeouts.emplace_back(var);
  }
  save_op_builder = save_op_builder.Input(var_nodeouts);

  // Update the attrs.
  tensor_names_op->AddAttr("value", new_tensor_names);
  shape_and_slices_op->AddAttr("value", new_shape_and_slices);

  // Remove the old save_op and add the new one.
  Node* new_save_op;
  TF_RETURN_IF_ERROR(save_op_builder.Finalize(graph, &new_save_op));
  // Add outputs to the new_save_op, all outputs are control edges.
  for (const Edge* edge : save_op->out_edges()) {
    graph->AddControlEdge(new_save_op, edge->dst());
  }
  graph->RemoveNode(save_op);

  return Status::OK();
}

// Add a restore subgraph for each variable and connect to the restore_all op.
// For each variable we add the following subgraph:
//           Assign----restore_all
//          |      |
//   RestoreV2    Variable
Status AddRestoreVariableSubgraphs(Graph* graph, Node* save_op,
                                   const std::vector<const Edge*>& in_edges,
                                   const std::vector<Node*>& variables) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_8(mht_8_v, 443, "", "./tensorflow/core/common_runtime/quantize_training.cc", "AddRestoreVariableSubgraphs");

  Node* prefix_op = in_edges[0]->src();
  StringPiece name_prefix = GetNodeNamePrefix(save_op);
  Node* restore_all = FindRestoreAllOp(graph, name_prefix);
  if (restore_all == nullptr) {
    return errors::InvalidArgument("graph has SaveOp, but no restore_all NoOp");
  }
  const string restore_op_name = strings::StrCat(name_prefix, "/RestoreV2");
  const string assign_op_name = strings::StrCat(name_prefix, "/Assign");
  for (Node* var : variables) {
    // Add an extra prefix after calling graph->NewName because the "unique"
    // name may conflict with names generated for Send nodes.
    // TODO(b/77547936): fix this more generally and get rid of the extra prefix
    // here.
    string new_restore_op_name =
        strings::StrCat(graph->NewName(restore_op_name), "_qt");
    string new_assign_op_name =
        strings::StrCat(graph->NewName(assign_op_name), "_qt");
    string tensor_names_op_name =
        strings::StrCat(new_restore_op_name, "/tensor_names");
    string shape_and_slices_op_name =
        strings::StrCat(new_restore_op_name, "/shape_and_slices");

    // Construct the tensor_names input with the variable name.
    Node* tensor_names;
    Tensor tensor_names_val(DT_STRING, TensorShape({1}));
    tensor_names_val.flat<tstring>()(0) = var->name();
    TF_RETURN_IF_ERROR(NodeBuilder(tensor_names_op_name, "Const")
                           .Attr("dtype", DT_STRING)
                           .Attr("value", tensor_names_val)
                           .Finalize(graph, &tensor_names));

    // Construct the shape_and_slices input with empty string.
    Node* shape_and_slices;
    Tensor shape_and_slices_val(DT_STRING, TensorShape({1}));
    shape_and_slices_val.flat<tstring>()(0) = "";
    TF_RETURN_IF_ERROR(NodeBuilder(shape_and_slices_op_name, "Const")
                           .Attr("dtype", DT_STRING)
                           .Attr("value", shape_and_slices_val)
                           .Finalize(graph, &shape_and_slices));

    // Build the new Restore op for this variable.
    Node* restore_op;
    TF_RETURN_IF_ERROR(NodeBuilder(new_restore_op_name, "RestoreV2")
                           .Input(prefix_op)
                           .Input(tensor_names)
                           .Input(shape_and_slices)
                           .Attr("dtypes", {DT_FLOAT})
                           .Finalize(graph, &restore_op));

    // Create Assign op, attaching the variable and Restore op to it.
    Node* assign_op;
    TF_RETURN_IF_ERROR(NodeBuilder(new_assign_op_name, "Assign")
                           .Input(var)
                           .Input(restore_op)
                           .Finalize(graph, &assign_op));

    // Add a control edge from the assign op to restore_all op.
    graph->AddControlEdge(assign_op, restore_all);
  }
  return Status::OK();
}

// Adds new variables to save and restore ops matching the Save and Restore
// graphs created in tensorflow/python/training/saver.py.
Status AddSaveAndRestore(Graph* graph, const std::vector<Node*>& variables) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_9(mht_9_v, 511, "", "./tensorflow/core/common_runtime/quantize_training.cc", "AddSaveAndRestore");

  Node* save_op = nullptr;
  std::vector<const Edge*> in_edges;
  bool found = false;
  TF_RETURN_IF_ERROR(FindSaveOp(graph, &save_op, &in_edges, &found));
  if (found) {
    TF_RETURN_IF_ERROR(
        AddRestoreVariableSubgraphs(graph, save_op, in_edges, variables));
    TF_RETURN_IF_ERROR(
        ConnectVariablesToSaveOp(graph, save_op, in_edges, variables));
  }
  return Status::OK();
}

// Sets output to the Node that computes reduction axes corresponding to all
// dimensions of input and return.
Status MakeReductionAxes(Graph* graph, string name_prefix, Node* input,
                         Node** output) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_10(mht_10_v, 532, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeReductionAxes");

  name_prefix = strings::StrCat(name_prefix, "/ReductionAxes");
  Node* start;
  Tensor zero_tensor(DT_INT32, TensorShape());
  zero_tensor.flat<int32>()(0) = 0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeStart"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", zero_tensor)
          .Finalize(graph, &start));
  Node* delta;
  Tensor one_tensor(DT_INT32, TensorShape());
  one_tensor.flat<int32>()(0) = 1;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeDelta"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", one_tensor)
          .Finalize(graph, &delta));
  Node* rank;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/InputRank"), "Rank")
          .Input(input)
          .Finalize(graph, &rank));
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ReductionAxes"), "Range")
          .Input(start)
          .Input(rank)
          .Input(delta)
          .Finalize(graph, output));
  return Status::OK();
}

// Computes the exponential moving average of input, updated in update_variable.
Status MakeExponentialMovingAverage(Graph* graph, string name_prefix,
                                    const NodeBuilder::NodeOut& input,
                                    Node* decay, Node* update_variable,
                                    Node** assign_value) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_11(mht_11_v, 572, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeExponentialMovingAverage");

  // variable_t+1 = variable_t - [(variable_t - value) * (1 - decay)]
  name_prefix = strings::StrCat(name_prefix, "/EMA");
  Node* one;
  Tensor one_tensor(DT_FLOAT, TensorShape());
  one_tensor.flat<float>()(0) = 1.0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/OneConst"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", one_tensor)
          .Finalize(graph, &one));
  Node* decay_complement;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/DecayComplement"), "Sub")
          .Input(one)
          .Input(decay)
          .Finalize(graph, &decay_complement));

  Node* value_diff;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ValueDiff"), "Sub")
          .Input(update_variable)
          .Input(input)
          .Finalize(graph, &value_diff));
  Node* update_value;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/UpdateValue"), "Mul")
          .Input(value_diff)
          .Input(decay_complement)
          .Finalize(graph, &update_value));

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/EMAValue"), "Sub")
          .Input(update_variable)
          .Input(update_value)
          .Finalize(graph, assign_value));
  return Status::OK();
}

// Creates an automatically initialized exponential moving average variable.
// This uses a switch op to assign a value to the variable on the first run,
// and update with the moving average for all other runs:
//                   init_val
//                      |
//      var--is_init--switch
//       |      true /      \ false
//       |          |        |
//       |         EMA    init_val
//       |           \      /
//       +----------- assign
Status MakeInitializedEMAVariable(Graph* graph, const string& name, Node* decay,
                                  Node* init_val,
                                  std::vector<Node*>* added_variables,
                                  Node** var) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_12(mht_12_v, 629, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeInitializedEMAVariable");

  // TODO(suharshs): Update this to use ResourceVariables when they are ready.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/Variable"), "VariableV2")
          .Attr("shape", TensorShape())
          .Attr("dtype", DT_FLOAT)
          .Finalize(graph, var));
  added_variables->push_back(*var);

  Node* is_initialized;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/IsInitialized"),
                                 "IsVariableInitialized")
                         .Input(*var)
                         .Finalize(graph, &is_initialized));
  Node* switch_node;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Switch"), "Switch")
                         .Input(init_val)
                         .Input(is_initialized)
                         .Finalize(graph, &switch_node));
  NodeBuilder::NodeOut output_false = NodeBuilder::NodeOut(switch_node, 0);
  NodeBuilder::NodeOut output_true = NodeBuilder::NodeOut(switch_node, 1);

  Node* ema_value;
  TF_RETURN_IF_ERROR(MakeExponentialMovingAverage(graph, name, output_true,
                                                  decay, *var, &ema_value));

  Node* assign_value;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Merge"), "Merge")
                         .Input({output_false, ema_value})
                         .Finalize(graph, &assign_value));

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/AssignValue"), "Assign")
          .Input(*var)
          .Input(assign_value)
          .Finalize(graph, var));
  return Status::OK();
}

// Computes the min and max EMA of input and stores them in min_var and max_var.
Status MakeEMAMinMaxVars(Graph* graph, const string& name_prefix, Node* input,
                         std::vector<Node*>* added_variables, Node** min_var,
                         Node** max_var) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_13(mht_13_v, 675, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeEMAMinMaxVars");

  // TODO(suharshs): The decay will be constant, so we could make only one for
  // all quantize_and_dequantize ops to share, this would have to live outside
  // this function.
  Tensor decay_tensor(DT_FLOAT, TensorShape());
  decay_tensor.flat<float>()(0) = kEMADecay;
  Node* decay;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/Decay"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", decay_tensor)
          .Finalize(graph, &decay));

  Node* reduction_axes;
  TF_RETURN_IF_ERROR(
      MakeReductionAxes(graph, name_prefix, input, &reduction_axes));
  Node* min;
  string min_name = strings::StrCat(name_prefix, "/Min");
  TF_RETURN_IF_ERROR(NodeBuilder(min_name, "Min")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &min));
  Node* max;
  string max_name = strings::StrCat(name_prefix, "/Max");
  TF_RETURN_IF_ERROR(NodeBuilder(max_name, "Max")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &max));
  TF_RETURN_IF_ERROR(MakeInitializedEMAVariable(graph, min_name, decay, min,
                                                added_variables, min_var));
  TF_RETURN_IF_ERROR(MakeInitializedEMAVariable(graph, max_name, decay, max,
                                                added_variables, max_var));
  return Status::OK();
}

// Makes an input min and max constant if the range is given. Otherwise, makes
// min and max variables that are updated by an EMA.
Status MakeInputMinMax(Graph* graph, const string& name_prefix,
                       const EdgeToConvert& edge,
                       std::vector<Node*>* added_variables, Node** input_min,
                       Node** input_max) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_14(mht_14_v, 719, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeInputMinMax");

  if (edge.range_given) {
    // Make constant nodes for the input_min and input_max if the range is
    // provided.
    Tensor input_min_tensor(DT_FLOAT, TensorShape());
    input_min_tensor.flat<float>()(0) = edge.input_min;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMin"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_min_tensor)
            .Finalize(graph, input_min));
    Tensor input_max_tensor(DT_FLOAT, TensorShape());
    input_max_tensor.flat<float>()(0) = edge.input_max;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMax"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_max_tensor)
            .Finalize(graph, input_max));
  } else {
    // If the range is not given, estimate the range with EMA variables.
    TF_RETURN_IF_ERROR(MakeEMAMinMaxVars(graph, name_prefix, edge.edge->src(),
                                         added_variables, input_min,
                                         input_max));
  }

  return Status::OK();
}

// Adds a QuantizeAndDequantizeV2 or FakeQuantizeWithMinMaxVars op
// (and required input nodes) based on edge.
// The result is stored in convert_node.
Status MakeQuantizeOp(Graph* graph, const string& name_prefix,
                      const string& quant_op_type, const EdgeToConvert& edge,
                      std::vector<Node*>* added_variables,
                      Node** convert_node) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name_prefix: \"" + name_prefix + "\"");
   mht_15_v.push_back("quant_op_type: \"" + quant_op_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_15(mht_15_v, 758, "", "./tensorflow/core/common_runtime/quantize_training.cc", "MakeQuantizeOp");

  Node* input_min;
  Node* input_max;
  TF_RETURN_IF_ERROR(MakeInputMinMax(graph, name_prefix, edge, added_variables,
                                     &input_min, &input_max));
  string quant_name = strings::StrCat(name_prefix, "/", quant_op_type);
  if (quant_op_type == "QuantizeAndDequantizeV2") {
    TF_RETURN_IF_ERROR(NodeBuilder(quant_name, quant_op_type)
                           .Input(edge.edge->src())
                           .Input(input_min)
                           .Input(input_max)
                           .Attr("signed_input", edge.signed_input)
                           .Attr("num_bits", edge.num_bits)
                           .Attr("range_given", true)
                           .Finalize(graph, convert_node));
  } else if (quant_op_type == "FakeQuantWithMinMaxVars") {
    TF_RETURN_IF_ERROR(NodeBuilder(quant_name, quant_op_type)
                           .Input(edge.edge->src())
                           .Input(input_min)
                           .Input(input_max)
                           .Attr("num_bits", edge.num_bits)
                           .Finalize(graph, convert_node));
  } else {
    return errors::InvalidArgument("Unknown quant op type: ", quant_op_type);
  }
  return Status::OK();
}

// Insert conversion op, connect it to the graph and remove the old edge.
Status ProcessTargetEdges(Graph* graph, const string& quant_op_type,
                          const std::vector<EdgeToConvert>& target_edges) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("quant_op_type: \"" + quant_op_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_16(mht_16_v, 792, "", "./tensorflow/core/common_runtime/quantize_training.cc", "ProcessTargetEdges");

  // Remember previously converted ops to avoid duplicated conversion on the
  // same input.
  std::unordered_map<string, Node*, StringPieceHasher> name_index;
  std::vector<Node*> added_variables;
  for (const EdgeToConvert edge : target_edges) {
    Node* convert_node;
    string name_prefix = edge.edge->src()->name();

    auto iter = name_index.find(name_prefix);
    if (iter == name_index.end()) {
      TF_RETURN_IF_ERROR(MakeQuantizeOp(graph, name_prefix, quant_op_type, edge,
                                        &added_variables, &convert_node));
      name_index[name_prefix] = convert_node;
    } else {
      convert_node = iter->second;
    }

    graph->AddEdge(convert_node, 0, edge.edge->dst(), edge.edge->dst_input());
    graph->RemoveEdge(edge.edge);
  }

  TF_RETURN_IF_ERROR(AddSaveAndRestore(graph, added_variables));

  return Status::OK();
}

}  // namespace

Status DoQuantizeTraining(int32_t num_bits, const string& quant_op_type,
                          Graph* graph) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("quant_op_type: \"" + quant_op_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_17(mht_17_v, 826, "", "./tensorflow/core/common_runtime/quantize_training.cc", "DoQuantizeTraining");

  if (graph == nullptr) {
    return errors::InvalidArgument("Cannot accept empty graph pointer.");
  }

  if (num_bits < 1 || num_bits > 63) {
    return errors::OutOfRange("num_bits should be in range [1, 63] but is: ",
                              num_bits);
  }
  int potential_input = 0;
  std::vector<EdgeToConvert> target_edges;
  for (Node* node : graph->nodes()) {
    if (nodes_to_rewrite->find(node->type_string()) !=
            nodes_to_rewrite->end() &&
        !IsGradientNode(graph, node)) {
      // Find out which types are the inputs and convert them accordingly.
      // 1. Const/Variable OP: This is quantized as signed tensors with no given
      // range.
      // 2. Activation OP: Set the range accordingly for different types of
      // activations. Currently we handle {Relu, Relu6, Sigmoid, Tanh}
      // 3. Identity OP: The quantization parameters depend on its input.
      // 4. Pooling OPs: various pooling ops. Also depends on its input.
      // 5. Reshape OP: Also depends on the first input to this op.
      // 6. Not-Listed-Above OP: If there is only 1 such op, consider it as the
      // model input. However, if there are >1 unknown ops, then returns an
      // error for now to avoid unexpected behavior.
      // Note: The list above might not be a complete list. Please let us
      // know if you see the error so we can handle your case.
      for (const Edge* edge : node->in_edges()) {
        if (edge->src_output() == Graph::kControlSlot) {
          // Skip the control dependency input.
          continue;
        } else {
          bool signed_input = false;
          bool range_given = false;
          float input_min = 0;
          float input_max = 0;
          bool known_op = FindType(graph, edge->src(), &signed_input,
                                   &range_given, &input_min, &input_max);
          if (!known_op) {
            // Unknown op is considered as input.
            potential_input++;
            if (potential_input > kAllowedInputs) {
              return errors::Unimplemented(
                  "Found an unknown op: ", edge->src()->name(),
                  " with type: ", edge->src()->type_string(),
                  "; Unknown ops are considered as model input for now and "
                  "only ",
                  kAllowedInputs, " inputs are supported currently.");
            }
          }

          target_edges.emplace_back(EdgeToConvert(
              edge, num_bits, signed_input, range_given, input_min, input_max));
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(ProcessTargetEdges(graph, quant_op_type, target_edges));

  return Status::OK();
}

Status DoQuantizeTrainingOnGraphDef(const GraphDef& input_graphdef,
                                    int32_t num_bits,
                                    const string& quant_op_type,
                                    GraphDef* result_graphdef) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("quant_op_type: \"" + quant_op_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_18(mht_18_v, 897, "", "./tensorflow/core/common_runtime/quantize_training.cc", "DoQuantizeTrainingOnGraphDef");

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, input_graphdef, &graph));

  // Call the rewriter on the graph.
  TF_RETURN_IF_ERROR(DoQuantizeTraining(num_bits, quant_op_type, &graph));

  // Convert the result graph back to a GraphDef.
  graph.ToGraphDef(result_graphdef);
  return Status::OK();
}

Status DoQuantizeTrainingOnSerializedGraphDef(const string& input_graph_string,
                                              int32_t num_bits,
                                              const string& quant_op_type,
                                              string* result_graph_string) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("input_graph_string: \"" + input_graph_string + "\"");
   mht_19_v.push_back("quant_op_type: \"" + quant_op_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_trainingDTcc mht_19(mht_19_v, 918, "", "./tensorflow/core/common_runtime/quantize_training.cc", "DoQuantizeTrainingOnSerializedGraphDef");

  // First create the graph from the GraphDef.
  GraphDef input_graphdef;
  if (!ParseProtoUnlimited(&input_graphdef, input_graph_string)) {
    return errors::InvalidArgument(
        "input_graph_string is not a serialized GraphDef protocol buffer");
  }
  GraphDef output_graphdef;
  TF_RETURN_IF_ERROR(DoQuantizeTrainingOnGraphDef(
      input_graphdef, num_bits, quant_op_type, &output_graphdef));

  if (!output_graphdef.SerializeToString(result_graph_string)) {
    return errors::Internal(
        "quantize training transformation resulted in invalid GraphDef");
  }
  return Status::OK();
}

}  // namespace tensorflow
