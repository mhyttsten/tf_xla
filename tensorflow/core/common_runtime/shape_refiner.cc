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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc() {
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
#include "tensorflow/core/common_runtime/shape_refiner.h"

#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

ShapeRefiner::ShapeRefiner(int graph_def_version,
                           const OpRegistryInterface* ops)
    : graph_def_version_(graph_def_version),
      ops_registry_(ops),
      graph_runner_(Env::Default()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::ShapeRefiner");
}

ShapeRefiner::ShapeRefiner(const VersionDef& versions,
                           const OpRegistryInterface* ops)
    : ShapeRefiner(versions.producer(), ops) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::ShapeRefiner");
}

ShapeRefiner::~ShapeRefiner() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::~ShapeRefiner");

  // The lifetime of the tensors are bound to the GraphRunner, so the tensors
  // should be deleted before it.
  const_tensor_map_.clear();
}

namespace {

constexpr char kArgOp[] = "_Arg";
constexpr char kRetvalOp[] = "_Retval";

}  // namespace

// Runs shape inference for the given node using the given ShapeRefiner.
// The node must be a sub-node of a function node and the outer_context is
// the inference context of that function node in the outer graph.
Status ShapeRefiner::InferShapesForFunctionSubNode(
    const Node* node, InferenceContext* outer_context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::InferShapesForFunctionSubNode");

  TF_RETURN_IF_ERROR(AddNodeInternal(node, outer_context));
  InferenceContext* node_context = CHECK_NOTNULL(GetContext(node));

  if (StringPiece(node->type_string()) == kArgOp) {
    // Handle special node: function input.
    // Shapes for these nodes are provided in the outer inference
    // context.

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));

    if (index < 0 || outer_context->num_inputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid input index: ", index,
          " not in [0, ", outer_context->num_inputs(), ").");
    }

    // TODO(b/134547156): TEMPORARY WORKAROUND. If input shape handle is not set
    // in outer context, set _Arg node output shape to unknown.
    if (outer_context->input(index).SameHandle(ShapeHandle())) {
      VLOG(1) << "Function instantiation has undefined input shape at "
              << "index: " << index << " in the outer inference context.";
      node_context->set_output(0, node_context->UnknownShape());
    } else {
      node_context->set_output(0, outer_context->input(index));
    }

    auto* resource = outer_context->input_handle_shapes_and_types(index);
    if (resource) {
      node_context->set_output_handle_shapes_and_types(0, *resource);
    }
  } else if (StringPiece(node->type_string()) == kRetvalOp) {
    // Handle special node: function output.
    // Shapes inferred for these nodes go into the outer inference
    // context.

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));

    if (index < 0 || outer_context->num_outputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid output index: ", index,
          " not in [0, ", outer_context->num_outputs(), ").");
    }

    // outer_context outlives node_context, therefore we need to create
    // a new shape handle owned by outer_context instead.
    ShapeHandle handle;
    TensorShapeProto proto;
    node_context->ShapeHandleToProto(node_context->input(0), &proto);
    TF_RETURN_IF_ERROR(outer_context->MakeShapeFromShapeProto(proto, &handle));
    outer_context->set_output(index, handle);

    const std::vector<ShapeAndType>* resource =
        node_context->input_handle_shapes_and_types(0);
    if (resource) {
      // `ShapesAndType`s contain `ShapeHandle`s.  These `ShapeHandle`s point
      // to `Shape`s that are owned by a different inference context too.  We
      // need to copy them to the outer context to prevent them from being
      // destroyed before they are used.
      std::vector<ShapeAndType> copied_shapes_and_types;
      for (auto& shape_and_type : *resource) {
        ShapeHandle handle;
        TensorShapeProto proto;
        node_context->ShapeHandleToProto(shape_and_type.shape, &proto);
        TF_RETURN_IF_ERROR(
            outer_context->MakeShapeFromShapeProto(proto, &handle));
        copied_shapes_and_types.push_back(
            ShapeAndType(handle, shape_and_type.dtype, shape_and_type.type));
      }

      outer_context->set_output_handle_shapes_and_types(
          index, copied_shapes_and_types);
    }
  }

  return Status::OK();
}

// TODO(cwhipkey): When an inference context inside function has
// requested_input_tensor(i) or requested_input_tensor_as_partial_shape(i)
// set when input(i) is an _Arg op, then this request should propagate to
// context, and vice versa.
//
// NOTE: Recursive user-defined functions are not supported.
// Maybe we won't support recursive functions at all in TF, because of
// other maintainability issues.
Status ShapeRefiner::InferShapesForFunction(
    const FunctionDef* function_def, AttrSlice attributes,
    ExtendedInferenceContext* outer_context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_4(mht_4_v, 340, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::InferShapesForFunction");

  const Graph* graph;
  auto it = functions_.find(function_def);
  if (it != functions_.end()) {
    graph = it->second.get();
  } else {
    InstantiationResult result;
    TF_RETURN_IF_ERROR(InstantiateFunction(
        *function_def, attributes,
        [this](const string& op, const OpDef** sig) {
          return this->function_library_->LookUpOpDef(op, sig);
        },
        &result));

    Graph* new_graph = new Graph(function_library_);
    GraphConstructorOptions options;
    options.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(
        ConvertNodeDefsToGraph(options, result.nodes, new_graph));
    functions_[function_def].reset(new_graph);
    graph = new_graph;
  }

  std::unordered_set<const Node*> function_nodes;
  Status inference_status = Status::OK();
  {
    auto node_shape_inference_lambda = [this, &outer_context, &function_nodes,
                                        &inference_status](const Node* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_5(mht_5_v, 370, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "lambda");

      if (!inference_status.ok()) return;
      inference_status =
          InferShapesForFunctionSubNode(node, outer_context->get_context());
      function_nodes.insert(node);
    };

    // Calls inference lambda for each node after visiting all predecessors.
    // Ensures that we are adding nodes to ShapeRefiner in the topological
    // order.
    ReverseDFS(*graph, {}, node_shape_inference_lambda);
  }

  // Delete the contexts created for the functions nodes to save memory.
  for (const Node* node : function_nodes) {
    node_to_context_.erase(node);
  }

  return inference_status;
}

Status ShapeRefiner::AddNode(const Node* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_6(mht_6_v, 394, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::AddNode");

  return AddNodeInternal(node, /*outer_context=*/nullptr);
}

Status ShapeRefiner::AddNodeInternal(
    const Node* node, shape_inference::InferenceContext* outer_context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_7(mht_7_v, 402, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::AddNodeInternal");

  // Create the inference context for this node with the existing input shapes.
  std::unique_ptr<InferenceContext> ic(new InferenceContext(
      graph_def_version_, node->def(), node->op_def(),
      std::vector<ShapeHandle>(node->num_inputs()), {}, {}, {}));
  TF_RETURN_IF_ERROR(ic->construction_status());

  // For each 'input' of this node, fetch the corresponding shape
  // from 'input's InferenceContext, and store into this node's
  // InferenceContext.
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    if (e->dst_input() < 0) {
      return tensorflow::errors::Internal(
          "Index ", e->dst_input(), " is negative but not a control edge.");
    }

    const Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      // v1 control flow adds loops to the graph; we have to break them
      // somewhere, so we'll ignore this input and leave its shape undefined.
      ic->SetInput(e->dst_input(), ic->UnknownShape());
      continue;
    }

    InferenceContext* input_ic = it->second->get_context();
    ic->SetInput(e->dst_input(), input_ic->output(e->src_output()));

    const auto* in_v =
        input_ic->output_handle_shapes_and_types(e->src_output());
    if (in_v != nullptr) {
      DataType input_type = e->src()->output_type(e->src_output());
      DCHECK(input_type == DT_RESOURCE || input_type == DT_VARIANT);
      ic->set_input_handle_shapes_and_types(e->dst_input(),
                                            std::vector<ShapeAndType>(*in_v));
    }
  }

  // Get the shape function for this node
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  std::unique_ptr<ExtendedInferenceContext> ec(
      new ExtendedInferenceContext(std::move(ic), node));

  // Run the shape inference function, and return if there was an error.
  TF_RETURN_IF_ERROR(RunShapeFn(node, op_reg_data, ec.get(), outer_context));

  // Store the resulting context object in the map.
  node_to_context_[node].swap(ec);

  return Status::OK();
}

Status ShapeRefiner::SetShape(const Node* node, int output_port,
                              ShapeHandle shape) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_8(mht_8_v, 468, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::SetShape");

  auto c = GetContext(node);
  if (c == nullptr) {
    return errors::Internal("Could not find context for ", node->name());
  }

  if (output_port < 0 || output_port >= node->num_outputs()) {
    return errors::InvalidArgument(
        "output_port '", output_port, "' is out of range, ", "node '",
        node->name(), "' has ", node->num_outputs(), " outputs");
  }
  // Note: it's possible, if the node's been updated, that the shape inference
  // context doesn't have the right number of outputs.
  if (node->num_outputs() > c->num_outputs()) {
    TF_RETURN_IF_ERROR(c->ExpandOutputs(node->num_outputs()));
  }

  // Check compatibility, and merge the shapes.
  ShapeHandle existing_shape = c->output(output_port);
  TF_RETURN_IF_ERROR(c->Merge(existing_shape, shape, &shape));
  c->set_output(output_port, shape);

  // TODO(vrv): Do we need to propagate the new shape through all
  // consumers that change their outputs?  At the moment, python
  // does not do this, but this seems like a nice feature.

  // TODO(vrv): We might need to keep track of the fact that the
  // existing shape is invalidated, in case we need to propagate
  // this information to remote workers.
  return Status::OK();
}

Status ShapeRefiner::UpdateNode(const Node* node, bool relax, bool* refined) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_9(mht_9_v, 503, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::UpdateNode");

  auto it = node_to_context_.find(node);
  if (it == node_to_context_.end()) {
    *refined = true;
    return AddNode(node);
  }
  ExtendedInferenceContext* node_ext_context = it->second.get();
  InferenceContext* node_context = node_ext_context->get_context();

  // Give up if the context wasn't successfully built by the AddNode() method.
  TF_RETURN_IF_ERROR(node_context->construction_status());

  // Check if the shapes of the nodes in the fan-in of this node have changed,
  // and if they have update the node input shapes.
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    int dst_input = e->dst_input();
    int src_output = e->src_output();

    Node* input = e->src();
    auto iter = node_to_context_.find(input);
    if (iter == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", dst_input, " ('", input->name(), "') for '", node->name(),
          "' was not previously added to ShapeRefiner.");
    }

    InferenceContext* c = iter->second->get_context();
    DCHECK_GE(dst_input, 0);
    ShapeHandle existing_input = node_context->input(dst_input);
    if (!relax) {
      if (node_context->MergeInput(dst_input, c->output(src_output))) {
        if (!SameDefinedShape(node_context, node_context->input(dst_input),
                              existing_input)) {
          *refined = true;
        }
      }
    } else {
      if (node_context->RelaxInput(dst_input, c->output(src_output))) {
        if (!SameDefinedShape(node_context, node_context->input(dst_input),
                              existing_input)) {
          *refined = true;
        }
      }
    }
    if (node_context->requested_input_tensor_as_partial_shape(dst_input)) {
      // The input value may have changed. Since we have no way to know if
      // that's indeed the case, err on the safe side.
      *refined = true;
    }

    // Also propagate handle shape and dtype of edges which are carrying
    // resource handles.
    if (e->src()->output_type(src_output) == DT_RESOURCE) {
      auto* outputs = c->output_handle_shapes_and_types(src_output);
      if (!outputs) continue;

      if (!relax &&
          node_context->MergeInputHandleShapesAndTypes(dst_input, *outputs)) {
        *refined = true;
      } else if (relax) {
        std::vector<ShapeAndType> existing_inputs;
        const std::vector<ShapeAndType>* inputs =
            node_context->input_handle_shapes_and_types(dst_input);
        if (inputs) {
          existing_inputs = *inputs;
        }
        if (node_context->RelaxInputHandleShapesAndMergeTypes(dst_input,
                                                              *outputs)) {
          if (IsUpdatedShapesOrTypes(
                  node_context, existing_inputs,
                  *node_context->input_handle_shapes_and_types(dst_input))) {
            *refined = true;
          }
        }
      }
    }
  }

  if (!*refined) {
    // No input shape has changed, we're done
    return Status::OK();
  }

  // Get and run the shape function for this node to update the shapes of the
  // outputs.
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  if (!op_reg_data->shape_inference_fn) {
    // There is nothing more we can infer
    return Status::OK();
  }

  return RunShapeFn(node, op_reg_data, node_ext_context);
}

Status ShapeRefiner::EvaluateConstantTensorForEdge(
    const Node* node, int dst_idx, bool* evaluated, Tensor* result,
    InferenceContext* outer_context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_10(mht_10_v, 612, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::EvaluateConstantTensorForEdge");

  *evaluated = false;
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));
  OutputTensor tensor(input_edge->src(), input_edge->src_output());
  return EvaluateConstantTensor(
      tensor, *this, *ops_registry_, graph_def_version_, evaluated, result,
      &graph_runner_, &const_tensor_map_, kMaxTensorSize,
      disable_constant_propagation_, outer_context);
}

Status ShapeRefiner::EvaluateConstantIntScalarEdge(
    const Node* node, int dst_idx, bool* evaluated, int64_t* result,
    shape_inference::InferenceContext* outer_context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_11(mht_11_v, 628, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::EvaluateConstantIntScalarEdge");

  Tensor scalar;
  TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, evaluated,
                                                   &scalar, outer_context));
  if (*evaluated) {
    if (scalar.NumElements() != 1) {
      return errors::InvalidArgument(
          "EvaluateConstantIntScalarEdge called on non-scalar edge: ",
          scalar.NumElements());
    }
    if (scalar.dtype() == DT_INT32) {
      *result = scalar.scalar<int32>()();
    } else {
      if (scalar.dtype() != DT_INT64) {
        return errors::InvalidArgument(
            "EvaluateConstantIntScalarEdge called on non-integer edge: ",
            scalar.dtype());
      }
      *result = scalar.scalar<int64_t>()();
    }
  }
  return Status::OK();
}

Status ShapeRefiner::ConstantPartialShape(
    InferenceContext* target_context, const Node* node, int dst_idx,
    ShapeHandle* result, shape_inference::InferenceContext* outer_context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_12(mht_12_v, 657, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::ConstantPartialShape");

  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));

  InferenceContext* src_context = GetContext(input_edge->src());
  if (src_context == nullptr) return errors::Internal("Missing src context");
  ShapeHandle src_shape = src_context->output(input_edge->src_output());

  // All shapes are expected to be 1D integer tensors with the exception of the
  // sentinel that represents an unknown shape (scalar/rank 0 tensor with -1 as
  // value). Handle the special case first before considering the more general
  // rank 1 case.

  if (src_context->Value(src_context->Rank(src_shape)) == 0) {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, &evaluated,
                                                     &t, outer_context));
    if (!evaluated) {
      return errors::InvalidArgument(
          "Received a shape scalar with unknown static value.  A static value "
          "of '-1' is required to represent an unknown shape.");
    }
    if (t.dims() == 0) {
      if (t.dtype() == DT_INT32 && t.scalar<int32>()() == -1) {
        *result = target_context->UnknownShape();
        return Status::OK();
      } else if (t.dtype() == DT_INT64 && t.scalar<int64_t>()() == -1) {
        *result = target_context->UnknownShape();
        return Status::OK();
      }
    }
    return errors::InvalidArgument(
        "Received an invalid shape scalar with a static value that is not "
        "'-1': ",
        t.DebugString());
  }

  TF_RETURN_IF_ERROR(src_context->WithRank(src_shape, 1, &src_shape));

  const string& src_op = input_edge->src()->type_string();
  if (src_context->Value(src_context->Dim(src_shape, 0)) == 0) {
    // Source tensor is a vector of length 0, so the shape it
    // represents is as scalar.
    *result = target_context->Scalar();
  } else if (src_op == "Cast") {
    // First try to evaluate the current tensor, as it might be a valid cast of
    // a float.
    Tensor t;
    bool evaluated = false;
    if (EvaluateConstantTensorForEdge(node, dst_idx, &evaluated, &t,
                                      outer_context)
            .ok()) {
      if (evaluated &&
          target_context->MakeShapeFromTensor(&t, src_shape, result).ok()) {
        return Status::OK();
      }
    }

    // Then try to infer partial shape from the input to the cast tensor.
    ShapeHandle pre_cast_shape;
    if (!ConstantPartialShape(target_context, input_edge->src(), 0,
                              &pre_cast_shape, outer_context)
             .ok()) {
      TF_RETURN_IF_ERROR(
          target_context->MakeShapeFromTensor(nullptr, src_shape, result));
    }
    if (!target_context->RankKnown(pre_cast_shape)) {
      // Failed to evaluate. Treat the output as completely unknown.
      *result = target_context->UnknownShape();
      return Status::OK();
    }
    auto* dest_type = input_edge->src()->attrs().Find("DstT");
    if (dest_type == nullptr || dest_type->value_case() != AttrValue::kType ||
        (dest_type->type() != DT_INT32 && dest_type->type() != DT_INT64)) {
      // Casting to a weird type. Do not attempt to infer across it.
      *result = target_context->MakeShape(std::vector<DimensionHandle>(
          target_context->Rank(pre_cast_shape), target_context->UnknownDim()));
      return Status::OK();
    }
    *result = pre_cast_shape;
  } else if (src_op == "Shape") {
    *result = src_context->input(0);
  } else if (src_op == "ShapeN") {
    *result = src_context->input(input_edge->src_output());
  } else if (src_op == "Pack") {
    std::vector<DimensionHandle> dims;
    // Pack is concatenating its input scalars to form the shape tensor vector.
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      int64_t size;
      bool evaluated;
      TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(
          input_edge->src(), i, &evaluated, &size, outer_context));
      if (evaluated) {
        dims.push_back(size < 0 ? target_context->UnknownDim()
                                : target_context->MakeDim(size));
      } else {
        dims.push_back(target_context->UnknownDim());
      }
    }
    *result = target_context->MakeShape(dims);
  } else if (src_op == "Concat" || src_op == "ConcatV2") {
    *result = target_context->Scalar();
    // For Concat, input 0 is concat dim; for V2 it is the last input.
    const int concat_dim =
        src_op == "Concat" ? 0 : src_context->num_inputs() - 1;
    // Concat is concatenating its input shape vectors.
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      // Concat dim is ignored (and will always be a scalar).
      if (i == concat_dim) continue;
      ShapeHandle sub_result;
      TF_RETURN_IF_ERROR(ConstantPartialShape(target_context, input_edge->src(),
                                              i, &sub_result, outer_context));
      if (!target_context->RankKnown(sub_result)) {
        // Failed to evaluate. Treat the output as completely unknown.
        // TODO(cwhipkey): we could rely on all inputs being the same rank, so
        // figure that rank out and append the right number of unknown dims.
        *result = target_context->UnknownShape();
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(
          target_context->Concatenate(*result, sub_result, result));
    }
  } else if (src_op == "StridedSlice") {
    TF_RETURN_IF_ERROR(PartialStridedSliceShape(input_edge->src(), src_context,
                                                result, outer_context));
  } else if (src_op == "VariableShape") {
    auto* handle_data = src_context->input_handle_shapes_and_types(0);
    if (handle_data != nullptr && !handle_data->empty()) {
      *result = handle_data->at(0).shape;
    } else {
      *result = target_context->UnknownShape();
    }
  } else {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, &evaluated,
                                                     &t, outer_context));
    TF_RETURN_IF_ERROR(target_context->MakeShapeFromTensor(
        evaluated ? &t : nullptr, src_shape, result));
  }
  return Status::OK();
}

Status ShapeRefiner::PartialStridedSliceShape(
    Node* slice_node, InferenceContext* ctx, ShapeHandle* result,
    shape_inference::InferenceContext* outer_context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_13(mht_13_v, 806, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::PartialStridedSliceShape");

  // Only attempt to evaluate if begin/end/strides all are scalars.
  for (int i = 1; i <= 3; ++i) {
    ShapeHandle input_shape = ctx->input(i);
    if (ctx->Value(ctx->Dim(input_shape, 0)) != 1) {
      *result = ctx->UnknownShape();
      return Status::OK();
    }
  }

  int begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(slice_node->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "ellipsis_mask", &ellipsis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "shrink_axis_mask", &shrink_axis_mask));

  // Only attempt to evaluate if there are no special masks set (note that we
  // can handle begin/end_mask == 1).
  if (!(begin_mask == 0 || begin_mask == 1) ||
      !(end_mask == 0 || end_mask == 1) || ellipsis_mask != 0 ||
      new_axis_mask != 0 || shrink_axis_mask != 0) {
    *result = ctx->UnknownShape();
    return Status::OK();
  }

  bool evaluated;
  int64_t begin;
  if (begin_mask == 1) {
    begin = 0;
  } else {
    TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 1, &evaluated,
                                                     &begin, outer_context));
    if (!evaluated) {
      *result = ctx->UnknownShape();
      return Status::OK();
    }
  }

  int64_t end;
  if (end_mask == 1) {
    end = std::numeric_limits<int64_t>::max();
  } else {
    TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 2, &evaluated,
                                                     &end, outer_context));
    if (!evaluated) {
      *result = ctx->UnknownShape();
      return Status::OK();
    }
  }

  int64_t stride;
  TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 3, &evaluated,
                                                   &stride, outer_context));
  if (!evaluated) {
    *result = ctx->UnknownShape();
    return Status::OK();
  }

  // Apply stride to input interpreted as a partial shape.
  ShapeHandle input;
  TF_RETURN_IF_ERROR(
      ConstantPartialShape(ctx, slice_node, 0, &input, outer_context));
  TF_RETURN_IF_ERROR(ctx->Subshape(input, begin, end, stride, result));
  return Status::OK();
}

Status ShapeRefiner::RunShapeFn(const Node* node,
                                const OpRegistrationData* op_reg_data,
                                ExtendedInferenceContext* ec,
                                InferenceContext* outer_context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_14(mht_14_v, 883, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::RunShapeFn");

  // This will be filled in with real data in a second pass.
  std::vector<const Tensor*> input_tensors(node->num_inputs(), nullptr);
  std::vector<Tensor> real_tensors(node->num_inputs());
  std::vector<bool> attempted_materialization(node->num_inputs());
  std::vector<bool> attempted_tensor_as_shape_conversion(node->num_inputs());
  std::vector<ShapeHandle> input_tensors_as_shapes;

  auto* c = ec->get_context();

  c->set_input_tensors(input_tensors);
  c->set_input_tensors_as_shapes(input_tensors_as_shapes);

  // Run the shape inference function, and return if there was an error.
  // Capture as lambda, because we might need to re-run inference later on.
  auto run_inference_lambda = [&]() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_15(mht_15_v, 901, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "lambda");

    if (function_library_ && IsFunctionCall(*function_library_, *node)) {
      bool disable_shape_inference;
      if (!GetNodeAttr(AttrSlice(node->def()), "_disable_call_shape_inference",
                       &disable_shape_inference)
               .ok() ||
          !disable_shape_inference) {
        // Special inference logic for user-defined functions.
        NameAttrList function;
        TF_RETURN_IF_ERROR(
            NameAndAttrsFromFunctionCall(node->def(), &function));
        const FunctionDef* function_def =
            function_library_->Find(function.name());
        if (function_def != nullptr) {
          // The constant Tensor map we have for the outside context is not
          // valid inside the function. We need to push a new clean map while
          // performing inference on the function body.
          auto const_tensor_map_copy = const_tensor_map_;
          const_tensor_map_.clear();
          Status function_inference_status = InferShapesForFunction(
              function_def, AttrSlice(&function.attr()), ec);
          const_tensor_map_ = const_tensor_map_copy;
          return function_inference_status;
        }
      }
    }

    if (op_reg_data->shape_inference_fn) {
      TF_RETURN_IF_ERROR(c->Run(op_reg_data->shape_inference_fn));
    } else {
      TF_RETURN_IF_ERROR(c->Run(shape_inference::UnknownShape));
    }
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(run_inference_lambda());

  // We must run the shape function repeatedly, in case users write
  // shape functions where they only conditionally call input_tensor()
  // based on the values of another input tensor.
  bool rerun_shape_fn;
  do {
    // If the result of running shape inference would have benefitted
    // from knowing the values of input tensors, try to materialize
    // the results of those tensors, and then run the shape inference
    // function again using those known tensors.
    rerun_shape_fn = false;

    // NOTE: It is possible to batch the extraction and
    // materialization of inputs, instead of materializing one input
    // at a time like we do below.  If input-at-a-time computation
    // becomes a bottleneck, we could separate ExtractConstantSubgraph
    // into two functions: one that returns true if an input is
    // derivable from constants, and another function that extracts
    // the subgraph for multiple target nodes and executes the whole
    // subgraph once.

    for (int i = 0; i < c->num_inputs(); ++i) {
      if (!c->requested_input_tensor(i)) {
        continue;
      }
      // Check if we have not already filled in the requested input,
      // and if not, try to materialize the tensors.
      if (!attempted_materialization[i]) {
        attempted_materialization[i] = true;

        Tensor result;
        bool evaluated = false;
        TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(
            node, i, &evaluated, &result, outer_context));
        if (evaluated) {
          real_tensors[i] = result;
          input_tensors[i] = &real_tensors[i];
          // We have more concrete information about a shape,
          // so re-run shape inference.
          rerun_shape_fn = true;
        }
      }
      if (c->requested_input_tensor_as_partial_shape(i) &&
          !attempted_tensor_as_shape_conversion[i]) {
        attempted_tensor_as_shape_conversion[i] = true;
        if (i >= input_tensors_as_shapes.size()) {
          input_tensors_as_shapes.resize(i + 1);
        }
        ShapeHandle s;
        TF_RETURN_IF_ERROR(ConstantPartialShape(c, node, i, &s, outer_context));
        input_tensors_as_shapes[i] = s;
        rerun_shape_fn = true;
      }
    }

    if (rerun_shape_fn) {
      // We have more information about the shapes on this pass,
      // so re-run shape inference.
      c->set_input_tensors(input_tensors);
      c->set_input_tensors_as_shapes(input_tensors_as_shapes);
      TF_RETURN_IF_ERROR(run_inference_lambda());
    }
  } while (rerun_shape_fn);

  return Status::OK();
}

bool ShapeRefiner::SameDefinedShape(InferenceContext* c, ShapeHandle s0,
                                    ShapeHandle s1) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_16(mht_16_v, 1007, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::SameDefinedShape");

  if (s0.SameHandle(s1)) {
    return true;
  }
  if (c->Rank(s0) != c->Rank(s1)) {
    return false;
  }
  if (!c->RankKnown(s0) && !c->RankKnown(s1)) {
    return false;
  }
  for (int i = 0; i < c->Rank(s0); ++i) {
    if (!c->Dim(s0, i).SameHandle(c->Dim(s1, i))) {
      int64_t val0 = c->Value(c->Dim(s0, i));
      int64_t val1 = c->Value(c->Dim(s1, i));
      if (val0 < 0 || val1 < 0 || val0 != val1) {
        return false;
      }
    }
  }

  return true;
}

bool ShapeRefiner::IsUpdatedShapesOrTypes(
    InferenceContext* c, const std::vector<ShapeAndType>& existing,
    const std::vector<ShapeAndType>& updated) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refinerDTcc mht_17(mht_17_v, 1035, "", "./tensorflow/core/common_runtime/shape_refiner.cc", "ShapeRefiner::IsUpdatedShapesOrTypes");

  if (existing.size() != updated.size()) {
    return true;
  }
  for (int i = 0; i < existing.size(); i++) {
    if (!SameDefinedShape(c, existing[i].shape, updated[i].shape) ||
        existing[i].dtype != updated[i].dtype) {
      return true;
    }
  }
  return false;
}

}  // namespace tensorflow
