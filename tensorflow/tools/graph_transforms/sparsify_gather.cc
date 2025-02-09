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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc() {
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

#include <cmath>
#include <memory>
#include <unordered_map>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
using str_util::Split;
using str_util::StringReplace;
using strings::StrCat;

namespace graph_transforms {

// Sparsify Tensor of shape [N, 1]. Return the indices and values vectors for
// non-zero tensor content.
Status SparsifyWeights(const Tensor& tensor, Tensor* indices_tensor,
                       Tensor* values_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_0(mht_0_v, 210, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "SparsifyWeights");

  if (tensor.dims() != 2 || tensor.dim_size(1) != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Transform only applicable to subgraph with 'Const' with "
        "tensor of shape [N, 1]. But instead get shape ",
        tensor.shape().DebugString(), ".");
  }

  auto flat = tensor.flat<float>();
  std::vector<int64_t> indices;
  std::vector<float> values;

  for (int64_t i = 0; i < flat.size(); i++) {
    float val = flat(i);
    if (std::abs(val) >= 1.0e-5) {
      indices.push_back(i);
      values.push_back(val);
    }
  }

  // During model initialization, InitializeTableOp makes use of
  // KeyValueTensorIterator, which does not accept empty keys or values.
  // Consequently, adding a dummy pair of indices and values as a walkaround.
  if (indices.empty() || values.empty()) {
    indices.push_back(0);
    values.push_back(0);
  }
  *indices_tensor = Tensor(DataTypeToEnum<int64_t>::value,
                           {static_cast<int64_t>(indices.size())});
  std::copy_n(indices.begin(), indices.size(),
              indices_tensor->flat<int64_t>().data());

  *values_tensor = Tensor(DataTypeToEnum<float>::value,
                          {static_cast<int64_t>(values.size())});
  std::copy_n(values.begin(), values.size(),
              values_tensor->flat<float>().data());

  return Status::OK();
}

void CreateConstNode(const Tensor& tensor, const string& name,
                     NodeDef* node_def) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_1(mht_1_v, 255, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "CreateConstNode");

  node_def->set_op("Const");
  node_def->set_name(name);
  SetNodeTensorAttr<float>("value", tensor, node_def);
}

string GetMonolithicTensorKey(const string& tensor_slice_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tensor_slice_name: \"" + tensor_slice_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_2(mht_2_v, 265, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "GetMonolithicTensorKey");

  std::vector<string> names = Split(tensor_slice_name, "/");
  if (absl::StartsWith(names[names.size() - 1], "part_")) {
    CHECK_GE(names.size(), 2);
    names.pop_back();
  }
  return absl::StrJoin(names, "/");
}

Status ObtainTensorSlice(const GraphDef& input_graph_def,
                         const string& target_name,
                         string* shape_slice_string) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target_name: \"" + target_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_3(mht_3_v, 280, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "ObtainTensorSlice");

  string restore_node_name;
  for (const auto& node : input_graph_def.node()) {
    std::vector<string> node_name_parts = Split(node.name(), "/");
    if (node_name_parts.size() == 2 &&
        absl::StartsWith(node_name_parts[0], "save") &&
        absl::StartsWith(node_name_parts[1], "Assign") &&
        node.input(0) == target_name) {
      restore_node_name = node.input(1);
      break;
    }
  }

  std::vector<string> restore_node_parts = Split(restore_node_name, ":");
  CHECK_LE(restore_node_parts.size(), 2);
  string tensor_names_node;
  string shape_and_slices_node;
  for (const auto& node : input_graph_def.node()) {
    if ((node.name() == restore_node_parts[0]) && (node.op() == "RestoreV2")) {
      tensor_names_node = node.input(1);
      shape_and_slices_node = node.input(2);
      break;
    }
  }

  int offset = -1;
  for (const auto& node : input_graph_def.node()) {
    if (node.name() == tensor_names_node) {
      Tensor tensor_names_tensor;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &tensor_names_tensor));
      const auto& tensor_names_value = tensor_names_tensor.flat<tstring>();
      for (int i = 0; i < tensor_names_value.size(); i++) {
        if (tensor_names_value(i) == GetMonolithicTensorKey(target_name)) {
          offset = i;
          break;
        }
      }
    }
  }
  if (offset == -1) {
    return errors::Internal("Unable to find RestoreV2 entry for variable: ",
                            target_name);
  }
  for (const auto& node : input_graph_def.node()) {
    if (node.name() == shape_and_slices_node) {
      Tensor shape_and_slices_tensor;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &shape_and_slices_tensor));
      const auto& shape_and_slices_value =
          shape_and_slices_tensor.flat<tstring>();
      *shape_slice_string = shape_and_slices_value(offset);
      return Status::OK();
    }
  }
  return errors::Internal("Unable to find slice for variable: ", target_name);
}

Status ReadTensorFromCheckpoint(
    const string& tensor_name, const std::unique_ptr<BundleReader>& ckpt_reader,
    const string& shape_and_slice, Tensor* tensor) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tensor_name: \"" + tensor_name + "\"");
   mht_4_v.push_back("shape_and_slice: \"" + shape_and_slice + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_4(mht_4_v, 343, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "ReadTensorFromCheckpoint");

  if (ckpt_reader) {
    TensorShape parsed_full_shape;
    TensorSlice parsed_slice;
    TensorShape parsed_slice_shape;

    bool get_slice = false;
    if (!shape_and_slice.empty()) {
      TF_RETURN_IF_ERROR(
          checkpoint::ParseShapeAndSlice(shape_and_slice, &parsed_full_shape,
                                         &parsed_slice, &parsed_slice_shape));
      get_slice = (parsed_full_shape != parsed_slice_shape);
    }
    if (get_slice) {
      TF_RETURN_IF_ERROR(ckpt_reader->LookupSlice(
          GetMonolithicTensorKey(tensor_name), parsed_slice, tensor));
    } else {
      TF_RETURN_IF_ERROR(
          ckpt_reader->Lookup(GetMonolithicTensorKey(tensor_name), tensor));
    }
    return Status::OK();
  }
  return errors::Internal("Checkpoint reader was not initialized. ");
}

Status InitializeCheckpointReader(const TransformFuncContext& context,
                                  std::unique_ptr<BundleReader>* ckpt_reader) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_5(mht_5_v, 372, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "InitializeCheckpointReader");

  if (context.params.count("input_checkpoint")) {
    const string input_checkpoint = context.params.at("input_checkpoint")[0];
    ckpt_reader->reset(new BundleReader(Env::Default(), input_checkpoint));
    TF_RETURN_IF_ERROR((*ckpt_reader)->status());
  }
  return Status::OK();
}

Status ObtainVariableInfo(
    const GraphDef& input_graph_def,
    std::unique_ptr<std::unordered_map<string, string> >* shapes_and_slices) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_6(mht_6_v, 386, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "ObtainVariableInfo");

  shapes_and_slices->reset(new std::unordered_map<string, string>());
  for (const auto& node : input_graph_def.node()) {
    if ((node.op() == "Variable") || (node.op() == "VariableV2")) {
      string s;
      TF_RETURN_IF_ERROR(ObtainTensorSlice(input_graph_def, node.name(), &s));
      (**shapes_and_slices)[node.name()] = s;
    }
  }
  return Status::OK();
}

Status RemoveInputAtIndex(NodeDef* n, int index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_7(mht_7_v, 401, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "RemoveInputAtIndex");

  for (int i = index; i < n->input_size() - 1; i++) {
    n->mutable_input()->SwapElements(i, i + 1);
  }
  n->mutable_input()->RemoveLast();
  return Status::OK();
}

Status RemoveNodeAtIndex(GraphDef* g, int index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_8(mht_8_v, 412, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "RemoveNodeAtIndex");

  for (int i = index; i < g->node_size() - 1; i++) {
    g->mutable_node()->SwapElements(i, i + 1);
  }
  g->mutable_node()->RemoveLast();
  return Status::OK();
}

Status SparsifyGatherInternal(
    const GraphDef& input_graph_def,
    const std::unique_ptr<std::unordered_map<string, string> >&
        shapes_and_slices,
    const TransformFuncContext& context, const OpTypePattern& pattern,
    const std::unique_ptr<BundleReader>& ckpt_reader,
    GraphDef* output_graph_def) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_9(mht_9_v, 429, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "SparsifyGatherInternal");

  string group_init_node = "group_deps";
  if (context.params.count("group_init_node")) {
    group_init_node = context.params.at("group_init_node")[0];
  }
  GraphDef current_graph_def = input_graph_def;
  bool any_match_found = false;

  // Populate references.
  std::unordered_map<string, int> refs;
  for (const auto& node : current_graph_def.node()) {
    for (const auto& input : node.input()) {
      auto parsed_input = StringReplace(input, "^", "", true);
      refs[parsed_input] += 1;
    }
  }

  // The subgraphs may have overlapping components, therefore GraphMatcher
  // doesn't return all subgraphs in one round -- this has to be multi-round
  // update.
  do {
    any_match_found = false;
    GraphDef replaced_graph_def = current_graph_def;
    std::vector<string> init_table_node_names;
    std::vector<string> removed_node_names;

    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, pattern,
        [&ckpt_reader, &any_match_found, &init_table_node_names,
         &shapes_and_slices, &removed_node_names,
         &refs](const NodeMatch& match, const std::set<string>& input_nodes,
                const std::set<string>& output_nodes,
                std::vector<NodeDef>* new_nodes) {
          any_match_found = true;

          // The captured subgraph should be of the following pattern:
          // Const --> Identity --> Gather --> ...
          //                          ^
          //                          |
          //                        (ids)
          //
          // After transform, it becomes:
          //                   --> NoOp(group_deps)
          //                   |
          // Const --> InitializeTable --> HashTable
          //                   ^              |
          //                   |              |
          // Const -------------              |
          //                                  v
          //               (ids) ---> LookupTableFind <--- Const(default)
          //                                  |
          //                                  v
          //                                 ...

          // clang-format off
          // For each subgraph, do the following
          // 1. Sparsify the `Const`, creating two `Const`, for hashtable
          // key/val.
          // 2. Create a `InitializeTable` op connecting to the above 2 `Const`.
          // 3. Create a `HashTable` op connecting to `InitializeTable` op.
          // 4. Replace the `Gather` with a `LookupTableFind` op.
          // 5. Connect the `LookupTableFind` with
          //    a. `HashTable`
          //    b. `Gather`'s ids input
          //    c. a `default_val` arg, valued at 0
          // clang-format on
          const NodeDef& gather_node = match.node;

          // GatherV2 adds an "axis" parameter. sparsify_gather only supports
          // axis 0 gathers.
          if (gather_node.op() == "GatherV2") {
            // Per the OpTypePattern, the 3rd input to Gather must be a Const.
            const NodeDef& axis_node = match.inputs[2].node;

            Tensor axis_t;
            TF_RETURN_IF_ERROR(GetNodeAttr(axis_node, "value", &axis_t));
            int64_t axis = 0;
            if (axis_t.dtype() == DT_INT32) {
              axis = axis_t.scalar<int32>()();
            } else if (axis_t.dtype() == DT_INT64) {
              axis = axis_t.scalar<int64_t>()();
            } else {
              return tensorflow::errors::FailedPrecondition(
                  "Gather axis was not int32 or int64.");
            }

            if (axis != 0) {
              return tensorflow::errors::FailedPrecondition(
                  "Transform only applicable to subgraph with GatherV2 over "
                  "axis 0. Found axis ",
                  axis, ".");
            }
          }

          const NodeDef& weights_node = match.inputs[0].inputs[0].node;

          DataType data_type;
          TF_RETURN_IF_ERROR(GetNodeAttr(weights_node, "dtype", &data_type));
          if (data_type != DT_FLOAT) {
            return tensorflow::errors::FailedPrecondition(
                "Transform only applicable to subgraph with 'Const',"
                "'Variable', or 'VariableV2' of dtype "
                "'DT_FLOAT'. Found '" +
                    weights_node.op() + "' with name '",
                weights_node.name(), "' and dtype '", data_type, "'.");
          }

          Tensor weight;
          if (weights_node.op() == "Const") {
            weight = GetNodeTensorAttr(weights_node, "value");
          } else {
            TF_RETURN_IF_ERROR(ReadTensorFromCheckpoint(
                weights_node.name(), ckpt_reader,
                (*shapes_and_slices)[weights_node.name()], &weight));
          }
          // Add both weight and identity node names.
          removed_node_names.push_back(weights_node.name());
          removed_node_names.push_back(match.inputs[0].node.name());
          for (auto input_node : match.inputs[0].node.input()) {
            auto parsed_input = StringReplace(input_node, "^", "", true);
            refs[parsed_input]--;
          }
          Tensor indices_tensor;
          Tensor values_tensor;
          TF_RETURN_IF_ERROR(
              SparsifyWeights(weight, &indices_tensor, &values_tensor));

          // indices and values of sparsified `Const`
          DataType key_dtype = DT_INT64;
          NodeDef indices_node;
          CreateConstNode(indices_tensor,
                          StrCat(weights_node.name(), "/indices"),
                          &indices_node);
          SetNodeAttr("dtype", key_dtype, &indices_node);

          NodeDef values_node;
          CreateConstNode(values_tensor, StrCat(weights_node.name(), "/values"),
                          &values_node);
          SetNodeAttr("dtype", data_type, &values_node);

          // HashTable node
          NodeDef hashtable_node;
          hashtable_node.set_op("HashTable");
          hashtable_node.set_name(StrCat(weights_node.name(), "/HashTable"));
          SetNodeAttr("key_dtype", key_dtype, &hashtable_node);
          SetNodeAttr("value_dtype", data_type, &hashtable_node);

          // InitializeTable node
          NodeDef init_table_node;
          init_table_node.set_op("InitializeTable");
          init_table_node.set_name(
              StrCat(weights_node.name(), "/InitializeTable"));
          SetNodeAttr("Tkey", key_dtype, &init_table_node);
          SetNodeAttr("Tval", data_type, &init_table_node);
          init_table_node_names.push_back(init_table_node.name());

          // LookupTableFind node
          NodeDef lookup_node;
          lookup_node.set_op("LookupTableFind");
          lookup_node.set_name(StrCat(gather_node.name(), "/LookupTableFind"));
          SetNodeAttr("Tin", key_dtype, &lookup_node);
          SetNodeAttr("Tout", data_type, &lookup_node);

          // Default return value of hashtable lookup
          Tensor zero_tensor(data_type, TensorShape({}));
          zero_tensor.flat<float>()(0) = 0.0;
          NodeDef default_value_node;
          CreateConstNode(zero_tensor, StrCat(gather_node.name(), "/Const"),
                          &default_value_node);
          SetNodeAttr("dtype", data_type, &default_value_node);

          // ExpandDims argument
          Tensor dim_idx(DT_INT32, TensorShape({}));
          dim_idx.flat<int32>()(0) = -1;
          NodeDef dim_idx_node;
          dim_idx_node.set_op("Const");
          dim_idx_node.set_name(
              StrCat(gather_node.name(), "/ExpandDims/Const"));
          SetNodeAttr("value", dim_idx, &dim_idx_node);
          SetNodeAttr("dtype", DT_INT32, &dim_idx_node);

          // ExpandDims node
          NodeDef expand_dims_node;
          expand_dims_node.set_op("ExpandDims");
          // Reuse gather_node's name so not to change dependent's inputs
          expand_dims_node.set_name(gather_node.name());
          SetNodeAttr("T", data_type, &expand_dims_node);

          // Connect nodes
          AddNodeInput(hashtable_node.name(), &init_table_node);
          refs[hashtable_node.name()]++;
          AddNodeInput(indices_node.name(), &init_table_node);
          refs[indices_node.name()]++;
          AddNodeInput(values_node.name(), &init_table_node);
          refs[values_node.name()]++;

          AddNodeInput(hashtable_node.name(), &lookup_node);
          refs[hashtable_node.name()]++;
          AddNodeInput(gather_node.input(1), &lookup_node);
          refs[gather_node.input(1)]++;
          AddNodeInput(default_value_node.name(), &lookup_node);
          refs[default_value_node.name()]++;

          AddNodeInput(lookup_node.name(), &expand_dims_node);
          refs[lookup_node.name()]++;
          AddNodeInput(dim_idx_node.name(), &expand_dims_node);
          refs[dim_idx_node.name()]++;

          // Copy 'ids' input of original 'Gather'
          new_nodes->push_back(match.inputs[1].node);
          new_nodes->push_back(indices_node);
          new_nodes->push_back(values_node);
          new_nodes->push_back(hashtable_node);
          new_nodes->push_back(init_table_node);
          new_nodes->push_back(lookup_node);
          new_nodes->push_back(default_value_node);
          new_nodes->push_back(dim_idx_node);
          new_nodes->push_back(expand_dims_node);

          return Status::OK();
        },
        {true}, &replaced_graph_def));

    NodeDef* init_op = nullptr;
    for (int i = 0; i < replaced_graph_def.node_size(); i++) {
      if (replaced_graph_def.node(i).name() == group_init_node &&
          replaced_graph_def.node(i).op() == "NoOp") {
        init_op = replaced_graph_def.mutable_node(i);
        break;
      }
    }
    if (!init_op) {
      // Init node
      init_op = replaced_graph_def.mutable_node()->Add();
      init_op->set_op("NoOp");
      init_op->set_name(group_init_node);
    }
    for (const string& name : init_table_node_names) {
      // Add control dependence from init_table_node to group_deps_node
      AddNodeInput(StrCat("^", name), init_op);
      refs[name]++;
    }

    // Erase inputs and outputs as they are not considered for deletion.
    for (const auto& output : context.output_names) {
      refs.erase(output);
    }

    for (const auto& input : context.input_names) {
      refs.erase(input);
    }

    // Add nodes with a reference count of 0 for deletion.
    for (const auto& entry : refs) {
      if (entry.second == 0) {
        removed_node_names.push_back(entry.first);
      }
    }

    while (!removed_node_names.empty()) {
      auto name = removed_node_names.back();
      removed_node_names.pop_back();

      int i = 0;
      while (i < replaced_graph_def.node_size()) {
        // Revisit this to see if we can safely remove RestoreV2 nodes.
        if ((replaced_graph_def.node(i).name() == name) &&
            (replaced_graph_def.node(i).op() != "RestoreV2")) {
          for (const auto& input : replaced_graph_def.node(i).input()) {
            auto parsed_input = StringReplace(input, "^", "", true);
            refs[parsed_input] -= 1;
            if (refs[parsed_input] == 0) {
              removed_node_names.push_back(parsed_input);
            }
          }
          TF_RETURN_IF_ERROR(RemoveNodeAtIndex(&replaced_graph_def, i));
          continue;
        }
        int j = 0;
        bool deleted_inputs = false;
        while (j < replaced_graph_def.node(i).input_size()) {
          if (replaced_graph_def.node(i).input(j) == name ||
              replaced_graph_def.node(i).input(j) == ("^" + name)) {
            TF_RETURN_IF_ERROR(
                RemoveInputAtIndex(replaced_graph_def.mutable_node(i), j));
            deleted_inputs = true;
            continue;
          }
          j++;
        }
        if (deleted_inputs) {
          if (replaced_graph_def.node(i).op() == "ConcatV2") {
            if (replaced_graph_def.node(i).input_size() > 2) {
              SetNodeAttr("N", replaced_graph_def.node(i).input_size() - 1,
                          replaced_graph_def.mutable_node(i));
            } else if (replaced_graph_def.node(i).input_size() == 2) {
              if (refs[replaced_graph_def.node(i).input(1)] != 1) {
                return errors::Internal(
                    "Expect axis tensor of ConcatV2 node to only be referenced "
                    "once.");
              }
              refs[replaced_graph_def.node(i).input(1)] -= 1;
              removed_node_names.push_back(replaced_graph_def.node(i).input(1));
              replaced_graph_def.mutable_node(i)->mutable_input()->RemoveLast();
              replaced_graph_def.mutable_node(i)->mutable_attr()->erase("N");
              replaced_graph_def.mutable_node(i)->set_op("Identity");
            } else {
              return errors::Internal(
                  "ConcatV2 should have at least two elements");
            }
          }
          if ((replaced_graph_def.node(i).op() == "Assign" ||
               replaced_graph_def.node(i).op() == "Reshape" ||
               replaced_graph_def.node(i).op() == "Equal" ||
               replaced_graph_def.node(i).op() == "Mean" ||
               replaced_graph_def.node(i).op() == "ScalarSummary") &&
              replaced_graph_def.node(i).input_size() == 1) {
            removed_node_names.push_back(replaced_graph_def.node(i).name());
          }
          if (!replaced_graph_def.node(i).input_size()) {
            removed_node_names.push_back(replaced_graph_def.node(i).name());
          }
        }
        i++;
      }
    }
    current_graph_def = replaced_graph_def;
  } while (any_match_found);
  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gatherDTcc mht_10(mht_10_v, 766, "", "./tensorflow/tools/graph_transforms/sparsify_gather.cc", "SparsifyGather");

  // clang-format off
  const OpTypePattern gather_pattern =
    {"Gather",
     {
       {"Identity",
        {
          {"Const|Variable|VariableV2"}
        }
       },
       {"*"},
     }
    };
  const OpTypePattern gather_v2_pattern =
    {"GatherV2",
      {
        {"Identity",
          {
            {"Const|Variable|VariableV2"}
          }
        },
        {"*"},
        // GatherV2's axis must be constant.
        {"Const"},
      }
    };
  // clang-format on

  GraphDef cleaned_input_graph_def;
  RemoveAttributes(input_graph_def, {"_output_shapes"},
                   &cleaned_input_graph_def);

  GraphDef temp_output;

  std::unique_ptr<BundleReader> ckpt_reader;
  TF_RETURN_IF_ERROR(InitializeCheckpointReader(context, &ckpt_reader));

  std::unique_ptr<std::unordered_map<string, string> > shapes_and_slices;
  TF_RETURN_IF_ERROR(
      ObtainVariableInfo(cleaned_input_graph_def, &shapes_and_slices));

  TF_RETURN_IF_ERROR(SparsifyGatherInternal(
      cleaned_input_graph_def, shapes_and_slices, context, gather_pattern,
      ckpt_reader, &temp_output));

  TF_RETURN_IF_ERROR(SparsifyGatherInternal(temp_output, shapes_and_slices,
                                            context, gather_v2_pattern,
                                            ckpt_reader, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("sparsify_gather", SparsifyGather);

}  // namespace graph_transforms
}  // namespace tensorflow
