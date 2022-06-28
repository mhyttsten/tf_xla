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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc() {
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

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
namespace {
using StringPieceSet = std::unordered_set<StringPiece, StringPieceHasher>;
template <typename T>
using StringPieceMap = std::unordered_map<StringPiece, T, StringPieceHasher>;
}  // namespace

Status ReplaceSendRecvs(const GraphDef& original_graph_def,
                        const GraphDef& rewritten_graph_def,
                        const std::vector<string>& inputs,
                        const std::vector<string>& outputs,
                        GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_0(mht_0_v, 219, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "ReplaceSendRecvs");

  // recv_node_names serves as a string storage for recv node names.
  std::vector<string> recv_node_names(inputs.size());
  StringPieceMap<TensorId> recv_node_map;
  StringPieceSet input_nodes;
  for (int i = 0; i < inputs.size(); ++i) {
    // RewriteGraphForExecution adds a recv node for each input edge. We assume
    // here that adding such recv node did not fail. For example, the original
    // graph did not already have a node with the name for the new added recv
    // node.
    TensorId id = ParseTensorName(inputs[i]);
    input_nodes.insert(id.first);
    string& recv_node_name = recv_node_names[i];
    recv_node_name = strings::StrCat("_recv_", id.first, "_", id.second);
    recv_node_map.emplace(recv_node_name, id);
  }

  StringPieceMap<const NodeDef*> original_map;
  for (const NodeDef& node : original_graph_def.node()) {
    original_map.emplace(node.name(), &node);
  }

  for (const NodeDef& node : rewritten_graph_def.node()) {
    if ((node.op() == "_Send") || (node.op() == "_Recv")) {
      // If the op is a Send or Recv that wasn't in the original, skip it.
      if (original_map.count(node.name()) == 0) {
        continue;
      }
    }

    NodeDef* new_node = output_graph_def->add_node();
    new_node->MergeFrom(node);
    for (int i = 0; i < new_node->input_size(); ++i) {
      string& input = *new_node->mutable_input(i);
      TensorId id = ParseTensorName(input);
      const auto iter = recv_node_map.find(id.first);
      if (iter != recv_node_map.end()) {
        // The node being substituted is a Recv node, and it has only one
        // output. If this input is not a control input, then replace the input
        // with the mapped value. Otherwise, replace the node name only.
        if (id.second != Graph::kControlSlot) {
          CHECK_EQ(id.second, 0);
          input = iter->second.ToString();
        } else {
          id.first = iter->second.first;
          input = id.ToString();
        }
      }
    }

    // RewriteGraphForExecution() did not remove this input node. Remove this
    // node name from input_nodes so that a duplicate does not get added to the
    // output_graph_def.
    auto iter = input_nodes.find(new_node->name());
    if (iter != input_nodes.end()) {
      input_nodes.erase(iter);
    }
  }

  // Some input nodes are removed in rewrite_graph_def. Add those nodes to
  // output_graph_def.
  for (StringPiece name : input_nodes) {
    const NodeDef& removed_node = *CHECK_NOTNULL(original_map[name]);
    output_graph_def->add_node()->MergeFrom(removed_node);
  }

  return Status::OK();
}

Status RewriteInputsAsPlaceholders(const TransformFuncContext& context,
                                   GraphDef* graph_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_1(mht_1_v, 292, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "RewriteInputsAsPlaceholders");

  std::unordered_set<string> input_names;
  for (const string& input_name : context.input_names) {
    input_names.emplace(ParseTensorName(input_name).first);
  }

  for (NodeDef& node : *graph_def->mutable_node()) {
    if (input_names.find(node.name()) == input_names.end()) {
      continue;
    }
    if (node.op() == "PlaceholderWithDefault") {
      node.set_op("Placeholder");
      node.clear_input();
    } else if (node.op() != "Placeholder") {
      return errors::InvalidArgument(
          "Input '", node.name(),
          "' was expected to be a Placeholder or PlaceholderWithDefault op, "
          "but was ",
          node.op());
    }
  }
  return Status::OK();
}

Status RemoveUnusedNodes(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_2(mht_2_v, 321, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "RemoveUnusedNodes");

  StringPieceMap<const NodeDef*> node_map;
  for (const NodeDef& node : input_graph_def.node()) {
    node_map.emplace(node.name(), &node);
  }

  std::unordered_set<TensorId, TensorId::Hasher> input_names;
  for (const string& input : context.input_names) {
    input_names.insert(ParseTensorName(input));
  }
  StringPieceSet used_nodes;
  StringPieceSet current_nodes;
  for (const string& name : context.output_names) {
    TensorId id = ParseTensorName(name);
    used_nodes.insert(id.first);
    current_nodes.insert(id.first);
  }
  while (!current_nodes.empty()) {
    StringPieceSet next_nodes;
    for (StringPiece node_name : current_nodes) {
      if (node_map.count(node_name) == 0) {
        LOG(ERROR) << "Bad graph structure, no node named '" << node_name
                   << "' found for input lookup";
        return errors::InvalidArgument("Bad graph structure, no node named '",
                                       node_name, "' found for input lookup");
      }
      const NodeDef& node = *(node_map[node_name]);
      for (const string& input : node.input()) {
        TensorId id = ParseTensorName(input);
        if (input_names.count(id) > 0) {
          continue;
        }
        if (used_nodes.insert(id.first).second) {
          next_nodes.insert(id.first);
        }
      }
    }
    current_nodes.swap(next_nodes);
  }
  for (const TensorId& id : input_names) {
    used_nodes.insert(id.first);
  }
  FilterGraphDef(
      input_graph_def,
      [&](const NodeDef& node) { return used_nodes.count(node.name()) > 0; },
      output_graph_def);
  TF_RETURN_IF_ERROR(RewriteInputsAsPlaceholders(context, output_graph_def));

  return Status::OK();
}

// Converts a shape inference handle to a PartialTensorShape.
Status ShapeHandleToTensorShape(const shape_inference::ShapeHandle& handle,
                                shape_inference::InferenceContext* context,
                                PartialTensorShape* shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_3(mht_3_v, 378, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "ShapeHandleToTensorShape");

  // The default is already unknown.
  if (!context->RankKnown(handle)) return Status::OK();

  std::vector<int64_t> dims(context->Rank(handle));
  for (int32_t i = 0; i < dims.size(); ++i) {
    dims[i] = context->Value(context->Dim(handle, i));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), dims.size(), shape);
}

// Converts any sub-graphs that can be resolved into constant expressions into
// single Const ops.
Status FoldConstants(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_4(mht_4_v, 396, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "FoldConstants");

  Graph input_graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(input_graph.AddFunctionLibrary(input_graph_def.library()));

  ShapeRefiner shape_refiner(input_graph.versions(), input_graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  shape_refiner.set_disable_constant_propagation(false);
  shape_refiner.set_function_library_for_shape_inference(
      &input_graph.flib_def());

  bool clear_output_shapes;
  TF_RETURN_IF_ERROR(context.GetOneBoolParameter("clear_output_shapes", true,
                                                 &clear_output_shapes));
  if (clear_output_shapes) {
    // Some older GraphDefs have saved _output_shapes attributes which are out
    // of date and cause import errors, so clean them up first.
    GraphDef cleaned_graph_def;
    RemoveAttributes(input_graph_def, {"_output_shapes"}, &cleaned_graph_def);

    TF_RETURN_IF_ERROR(
        ImportGraphDef({}, cleaned_graph_def, &input_graph, &shape_refiner));
  } else {
    TF_RETURN_IF_ERROR(
        ImportGraphDef({}, input_graph_def, &input_graph, &shape_refiner));
  }

  // Sorted array of input names as lookup table.
  std::vector<TensorId> input_names;
  input_names.reserve(context.input_names.size());
  std::transform(context.input_names.begin(), context.input_names.end(),
                 std::back_inserter(input_names),
                 [](const string& name) { return ParseTensorName(name); });

  const auto compare = [](TensorId lhs, TensorId rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_5(mht_5_v, 432, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "lambda");

    return lhs.first < rhs.first;
  };

  std::sort(input_names.begin(), input_names.end(), compare);

  // Set statically inferred shapes.
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  for (const Node* const node : input_graph.nodes()) {
    auto ctx = shape_refiner.GetContext(node);
    if (ctx == nullptr) {
      continue;
    }

    std::vector<PartialTensorShape>& partial_shapes = shape_map[node->name()];
    if (ctx->num_outputs() <= 0) continue;
    partial_shapes.resize(ctx->num_outputs());

    // Check all outputs.
    for (const Edge* out_edge : node->out_edges()) {
      if (out_edge->IsControlEdge()) continue;

      const int output_idx = out_edge->src_output();
      TF_RETURN_IF_ERROR(ShapeHandleToTensorShape(ctx->output(output_idx), ctx,
                                                  &partial_shapes[output_idx]));
    }

    // RewriteGraphForExecution() will add a Recv node for each input. Shape
    // refiner does not include shape information of these Recv nodes. Therefore
    // we add entries for Recv nodes here.
    const auto pair = std::equal_range(input_names.begin(), input_names.end(),
                                       TensorId{node->name(), 0}, compare);
    for (auto it = pair.first; it != pair.second; ++it) {
      const string recv_name =
          strings::StrCat("_recv_", it->first, "_", it->second);
      auto& recv_partial_shapes = shape_map[recv_name];
      // For whatever reason (for example, name collision) if the map entry was
      // already there, then do nothing.
      if (recv_partial_shapes.empty()) {
        recv_partial_shapes.push_back(partial_shapes[it->second]);
      }
    }
  }

  subgraph::RewriteGraphMetadata unused_metadata;
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      &input_graph, context.input_names, context.output_names, {}, {},
      false /* use_function_convention */, &unused_metadata));

  ConstantFoldingOptions cf_opts;
  cf_opts.shape_map = &shape_map;

  // Exclude specified nodes from constant folding.
  std::set<string> excluded_ops, excluded_nodes;
  if (context.params.count("exclude_op") > 0) {
    const auto& ops = context.params.at("exclude_op");
    excluded_ops = std::set<string>(ops.begin(), ops.end());
  }
  if (context.params.count("exclude_node") > 0) {
    const auto& nodes = context.params.at("exclude_node");
    excluded_nodes = std::set<string>(nodes.begin(), nodes.end());
  }
  if (!excluded_ops.empty() || !excluded_nodes.empty()) {
    cf_opts.consider = [excluded_ops, excluded_nodes](const Node* n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_libDTcc mht_6(mht_6_v, 498, "", "./tensorflow/tools/graph_transforms/fold_constants_lib.cc", "lambda");

      return excluded_ops.find(n->op_def().name()) == excluded_ops.end() &&
             excluded_nodes.find(n->name()) == excluded_nodes.end();
    };
  }

  TF_RETURN_IF_ERROR(context.GetOneInt64Parameter(
      "max_constant_size_in_bytes", cf_opts.max_constant_size_in_bytes,
      &cf_opts.max_constant_size_in_bytes));

  // Constant folding.
  bool was_mutated;
  TF_RETURN_IF_ERROR(ConstantFold(cf_opts, nullptr, Env::Default(), nullptr,
                                  &input_graph, &was_mutated));
  GraphDef folded_graph_def;
  input_graph.ToGraphDef(&folded_graph_def);
  GraphDef send_recvs_replaced;
  TF_RETURN_IF_ERROR(ReplaceSendRecvs(input_graph_def, folded_graph_def,
                                      context.input_names, context.output_names,
                                      &send_recvs_replaced));
  TF_RETURN_IF_ERROR(
      RemoveUnusedNodes(send_recvs_replaced, context, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_constants", FoldConstants);

}  // namespace graph_transforms
}  // namespace tensorflow
