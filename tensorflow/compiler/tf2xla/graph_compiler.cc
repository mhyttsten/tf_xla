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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/graph_compiler.h"

#include <deque>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {
Status PrepareArguments(XlaOpKernelContext* ctx, Graph* graph,
                        const std::vector<const XlaExpression*>& expressions,
                        const NameAttrList& func,
                        std::vector<XlaCompiler::Argument>* args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/tf2xla/graph_compiler.cc", "PrepareArguments");

  auto client = ctx->compiler()->client();
  std::vector<bool> arg_must_be_compile_time_constant(expressions.size());

  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
      *graph, &arg_must_be_compile_time_constant,
      /*compile_time_const_nodes=*/nullptr, ctx->function_library()));

  args->resize(expressions.size());
  for (int i = 0, end = args->size(); i < end; ++i) {
    XlaCompiler::Argument& arg = (*args)[i];
    arg.type = ctx->input_type(i);
    arg.shape = ctx->InputShape(i);

    switch (expressions[i]->kind()) {
      case XlaExpression::Kind::kConstant:
        arg.kind = XlaCompiler::Argument::kConstant;
        arg.constant_value = *expressions[i]->constant_value();
        break;
      case XlaExpression::Kind::kXlaOp:
        if (arg_must_be_compile_time_constant[i]) {
          TF_ASSIGN_OR_RETURN(absl::optional<Tensor> value,
                              expressions[i]->ResolveConstant(client));
          if (value.has_value()) {
            arg.kind = XlaCompiler::Argument::kConstant;
            arg.constant_value = *value;
          } else {
            arg.kind = XlaCompiler::Argument::kParameter;
          }

        } else {
          arg.kind = XlaCompiler::Argument::kParameter;
        }
        break;
      case XlaExpression::Kind::kResource: {
        XlaResource* resource = expressions[i]->resource();
        XlaCompiler::PopulateArgumentFromResource(*resource, &arg);
        break;
      }
      case XlaExpression::Kind::kTensorList: {
        arg.kind = XlaCompiler::Argument::kTensorList;
        const xla::XlaOp& tensor_list = expressions[i]->handle();
        arg.shape = tensor_list.builder()->GetShape(tensor_list).ValueOrDie();
        break;
      }
      case XlaExpression::Kind::kInvalid:
        return errors::InvalidArgument("Invalid function argument");
    }
  }
  return Status::OK();
}
}  // namespace
Status GraphCompiler::Compile() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc mht_1(mht_1_v, 283, "", "./tensorflow/compiler/tf2xla/graph_compiler.cc", "GraphCompiler::Compile");

  // Check that the graph has no illegal cycles.
  TF_RETURN_IF_ERROR(graph::ValidateGraphHasNoCycle(*graph_));
  // Maintain a mapping from node id to node outputs.
  using NodeOutputs = std::vector<TensorValue>;
  std::vector<NodeOutputs> output_registry(graph_->num_node_ids());
  auto output_registry_cleanup = gtl::MakeCleanup([&output_registry] {
    for (const NodeOutputs& outputs : output_registry) {
      for (const TensorValue& value : outputs) {
        CHECK(!value.is_ref());
        delete value.tensor;
      }
    }
  });

  // XLA requires determinism, generate a stable ordering from DFS.
  std::vector<Node*> topo_sorted_nodes;
  GetReversePostOrder(*graph_, &topo_sorted_nodes,
                      /*stable_comparator=*/NodeComparatorName());

  OpKernelContext::Params params;
  PartiallySetupParams(&params);

  for (Node* n : topo_sorted_nodes) {
    OpKernel* op_kernel_raw = nullptr;
    // The kernel is not actually run for functional ops, we just need it
    // for metadata.
    Status s = flib_->CreateKernel(n->properties(), &op_kernel_raw);
    // Transfer ownership of the kernel to a local smart pointer.
    std::unique_ptr<OpKernel> op_kernel(op_kernel_raw);

    if (!s.ok()) {
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }

    TF_RET_CHECK(!n->IsRecv() && !n->IsSend() && !n->IsSwitch())
        << "Not supported node: " << n->DebugString();
    params.op_kernel = op_kernel.get();
    absl::InlinedVector<AllocatorAttributes, 4> output_attr(n->num_outputs());
    params.output_attr_array = output_attr.data();

    // tensor_inputs_ is a buffer reused across graph traversal. We clean up and
    // reinitialize the buffer before we visit a new node.
    tensor_inputs_.clear();
    tensor_inputs_.resize(n->num_inputs());

    // Set up inputs from outputs of previous nodes.
    for (auto* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      const Node* src = e->src();
      const int output_registry_size = output_registry.size();
      TF_RET_CHECK(src->id() < output_registry_size);
      const NodeOutputs& src_outputs = output_registry[src->id()];

      tensor_inputs_.at(e->dst_input()) = src_outputs.at(e->src_output());
    }

    OpKernelContext op_context(&params, n->num_outputs());
    VLOG(3) << "Translating " << params.op_kernel->name();
    if (IsFunctionCall(*flib_->GetFunctionLibraryDefinition(), *n)) {
      TF_RETURN_IF_ERROR(CompileFunctionalNode(n, &op_context));
    } else {
      device_->Compute(CHECK_NOTNULL(params.op_kernel), &op_context);
      Status s = op_context.status();
      if (!s.ok()) {
        return AttachDef(s, n->def());
      }
    }

    // Set up outputs. Also check if outputs from the previous computation is
    // valid.
    NodeOutputs& outputs = output_registry[n->id()];
    outputs.resize(n->num_outputs());
    for (int o = 0; o < n->num_outputs(); ++o) {
      outputs[o] = op_context.release_output(o);
      if (outputs[o].tensor == nullptr) {
        return errors::Internal("Missing xla_context ", o, "-th output from ",
                                FormatNodeForError(*n));
      }
    }
  }
  return Status::OK();
}

namespace {

Status GetFunctionNameAndAttr(const FunctionLibraryRuntime& flib,
                              const Node& node, NameAttrList* func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc mht_2(mht_2_v, 375, "", "./tensorflow/compiler/tf2xla/graph_compiler.cc", "GetFunctionNameAndAttr");

  if (node.IsPartitionedCall()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        node.attrs().Find(FunctionLibraryDefinition::kFuncAttr, &attr_value));
    if (!attr_value->has_func()) {
      return errors::InvalidArgument(
          "The attribute value for attribute 'f' in node ", node.DebugString(),
          " does not have 'func' field set");
    }
    *func = attr_value->func();
    return Status::OK();
  }

  if (flib.GetFunctionLibraryDefinition()->Find(node.def().op())) {
    func->set_name(node.type_string());
  } else {
    func->set_name(FunctionLibraryDefinition::kGradientOp);
  }
  *func->mutable_attr() = node.def().attr();
  return Status::OK();
}

}  // namespace

Status GraphCompiler::CompileFunctionalNode(Node* n,
                                            OpKernelContext* op_context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc mht_3(mht_3_v, 404, "", "./tensorflow/compiler/tf2xla/graph_compiler.cc", "GraphCompiler::CompileFunctionalNode");

  TF_RET_CHECK(IsFunctionCall(*flib_->GetFunctionLibraryDefinition(), *n));
  // For functional nodes, compile them using compiler from the context and call
  // into the functions.
  XlaOpKernelContext xla_op_context(op_context);

  XlaContext& context = XlaContext::Get(op_context);
  auto* b = context.builder();

  XlaCompiler* compiler = xla_op_context.compiler();

  NameAttrList func;
  TF_RETURN_IF_ERROR(GetFunctionNameAndAttr(*flib_, *n, &func));

  std::vector<const XlaExpression*> expressions;

  for (auto tensor : tensor_inputs_) {
    auto expression =
        reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
    expressions.push_back(expression);
  }

  // Prepare the arguments and compile the function.
  std::vector<XlaCompiler::Argument> arguments;
  const FunctionBody* fbody;
  TF_RETURN_IF_ERROR(compiler->FindFunctionBody(func, &fbody));

  auto graph = compiler->GetGraph(fbody);

  TF_RETURN_IF_ERROR(PrepareArguments(&xla_op_context, graph.get(), expressions,
                                      func, &arguments));

  bool add_token_input_output =
      func.attr().find(kXlaTokenInputNodesAttrName) != func.attr().end();

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = false;
  compile_options.add_token_input_output = add_token_input_output;
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(
      compiler->CompileFunction(compile_options, func, arguments, &result));

  TF_RET_CHECK(arguments.size() == expressions.size());

  std::vector<xla::XlaOp> handles;
  for (int64_t i = 0, end = expressions.size(); i < end; ++i) {
    if (arguments[i].kind == XlaCompiler::Argument::kConstant) {
      continue;
    }
    if (arguments[i].kind == XlaCompiler::Argument::kResource) {
      handles.push_back(expressions[i]->resource()->value());
    } else {
      handles.push_back(expressions[i]->handle());
    }
  }
  if (add_token_input_output) {
    std::vector<string> token_input_nodes;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(&func.attr()),
                                   kXlaTokenInputNodesAttrName,
                                   &token_input_nodes));
    std::vector<xla::XlaOp> token_inputs;
    for (const string& node_name : token_input_nodes) {
      auto token_or = compiler->GetNodeToken(node_name);
      TF_RETURN_IF_ERROR(token_or.status());
      token_inputs.push_back(token_or.ConsumeValueOrDie());
    }
    xla::XlaOp token_input = xla::AfterAll(b, token_inputs);
    handles.push_back(token_input);
  }

  auto output_handle = xla::Call(b, *result.computation, handles);
  // The output handle of `Call` computation is a tuple type. Unzip it so
  // that it can fit into future computations.
  int computation_output = 0;
  for (int64_t i = 0; i < n->num_outputs(); ++i) {
    if (result.outputs[i].is_constant) {
      xla_op_context.SetConstantOutput(i, result.outputs[i].constant_value);
    } else {
      if (result.outputs[i].is_tensor_list) {
        xla_op_context.SetTensorListOutput(
            i, xla::GetTupleElement(output_handle, computation_output));
      } else {
        xla_op_context.SetOutput(
            i, xla::GetTupleElement(output_handle, computation_output));
      }
      ++computation_output;
    }
  }

  for (int64_t i = 0, end = result.resource_updates.size(); i < end; i++) {
    if (result.resource_updates[i].modified) {
      XlaResource* resource =
          expressions[result.resource_updates[i].input_index]->resource();
      xla::XlaOp updated_value =
          xla::GetTupleElement(output_handle, i + n->num_outputs());
      TF_RETURN_IF_ERROR(resource->SetValue(updated_value));
    }
  }

  if (add_token_input_output) {
    std::string node_name;
    if (!GetNodeAttr(n->attrs(), kXlaOriginalOutsideCompilationNodeName,
                     &node_name)
             .ok())
      node_name = n->name();
    TF_RETURN_IF_ERROR(compiler->SetNodeToken(
        node_name, xla::GetTupleElement(output_handle, computation_output)));
  }
  return b->first_error();
}

void GraphCompiler::PartiallySetupParams(OpKernelContext::Params* params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSgraph_compilerDTcc mht_4(mht_4_v, 518, "", "./tensorflow/compiler/tf2xla/graph_compiler.cc", "GraphCompiler::PartiallySetupParams");

  params->device = device_;
  params->inputs = &tensor_inputs_;
  params->step_container = step_container_;
  params->resource_manager = device_->resource_manager();
  params->function_library = flib_;
}

}  // namespace tensorflow
