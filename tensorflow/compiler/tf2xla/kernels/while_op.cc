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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/kernels/while_op.h"

#include "absl/strings/str_split.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/kernels/if_while_utils.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace {

// Verify that input resources are grouped in the end.
Status VerifyResourceArgsGroupedAtEnd(XlaOpKernelContext* ctx,
                                      const NameAttrList& body_name_attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "VerifyResourceArgsGroupedAtEnd");

  const FunctionBody* body;
  TF_RETURN_IF_ERROR(ctx->compiler()->FindFunctionBody(body_name_attr, &body));
  bool has_seen_resource = false;
  for (int i = 0; i < body->arg_types.size(); i++) {
    DataType arg_type = body->arg_types[i];
    if (has_seen_resource) {
      if (arg_type != DT_RESOURCE) {
        return errors::InvalidArgument(
            "Expect input resources are grouped in the end of while body ",
            body_name_attr.name(), ", but the ", i, "-th argument ",
            body->arg_nodes[i]->name(), " is not a resource.");
      }
    } else {
      if (arg_type == DT_RESOURCE) {
        has_seen_resource = true;
      }
    }
  }
  return Status::OK();
}

// Builds XlaCompiler argument descriptions `args` from `ctx`.
Status MakeXlaCompilerArgumentsFromInputs(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>* args,
    bool* has_uninitialized_vars, bool* has_tensor_arrays,
    bool* has_uninitialized_tensor_lists) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "MakeXlaCompilerArgumentsFromInputs");

  VLOG(2) << "Num inputs " << ctx->num_inputs();
  args->resize(ctx->num_inputs());
  *has_uninitialized_vars = false;
  *has_tensor_arrays = false;
  *has_uninitialized_tensor_lists = false;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    VLOG(2) << " Input " << i << " type: " << DataTypeString(ctx->input_type(i))
            << " shape: " << ctx->InputShape(i).DebugString();
    XlaCompiler::Argument& arg = (*args)[i];
    DataType type = ctx->input_type(i);
    // When reading a resource input, use the type and shape of the resource's
    // current value.
    if (type == DT_RESOURCE) {
      XlaResource* resource;
      TF_RETURN_IF_ERROR(ctx->GetResourceInput(i, &resource));
      XlaCompiler::PopulateArgumentFromResource(*resource, &arg);
      if (arg.resource_kind == XlaResource::kTensorArray) {
        *has_tensor_arrays = true;
      }
      if (!arg.initialized) {
        *has_uninitialized_vars = true;
      }
      VLOG(2) << "    resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.ShapeHumanString()
              << " initialized: " << arg.initialized;
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = type;
      TF_ASSIGN_OR_RETURN(arg.shape, ctx->builder()->GetShape(ctx->Input(i)));
      if (IsTensorListInput(ctx, i)) {
        // arg.initialized == false means that the element_shape of the list
        // was not available at the time of building the list so an empty list
        // was created instead. If so, the body function of While is run once
        // to infer the shape of the list before actually building the While op.
        TF_RETURN_IF_ERROR(
            IsTensorListInitialized(ctx->Input(i), &arg.initialized));
        if (!arg.initialized) {
          *has_uninitialized_tensor_lists = true;
        }
      }
    }
  }
  return Status::OK();
}

// Populates loop invariant indices to true in `loop_invariants`.
void GetLoopInvariants(XlaOpKernelContext* ctx,
                       const NameAttrList& body_name_attr,
                       std::vector<bool>* const loop_invariants) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_2(mht_2_v, 295, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "GetLoopInvariants");

  const FunctionBody* body;
  OP_REQUIRES_OK(ctx, ctx->compiler()->FindFunctionBody(body_name_attr, &body));
  const tensorflow::FunctionLibraryDefinition* fld =
      ctx->compiler()->flib_runtime()->GetFunctionLibraryDefinition();
  for (int i = 0; i < body->ret_nodes.size(); i++) {
    StatusOr<bool> is_loop_invariant = IsLoopInvariant(body, i, fld);
    OP_REQUIRES_OK(ctx, is_loop_invariant.status());
    (*loop_invariants)[i] = *is_loop_invariant;
    VLOG(2) << "Arg " << i << " of " << body_name_attr.name() << " is "
            << ((*loop_invariants)[i] ? "" : "not ") << "loop invariant";
  }
}

// Converts entries in `args` which are loop invariants and have compile time
// constant inputs and need to be constants in order to be compilable to
// constants so that they can be propagated in the loop body.
Status ConvertLoopInvariantsToConst(
    XlaOpKernelContext* ctx, const NameAttrList& body_name_attr,
    const NameAttrList& cond_name_attr,
    std::vector<XlaCompiler::Argument>* args,
    std::vector<bool>* compile_time_const_arg_indices,
    int* num_compile_time_const_args, xla::Client* client) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_3(mht_3_v, 320, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "ConvertLoopInvariantsToConst");

  std::vector<bool> loop_invariants(ctx->num_inputs());
  GetLoopInvariants(ctx, body_name_attr, &loop_invariants);

  std::vector<bool> body_must_be_const_nodes;
  const FunctionBody* body;
  std::vector<bool> cond_must_be_const_nodes;
  const FunctionBody* cond;
  TF_RETURN_IF_ERROR(FindMustBeConstNodes(ctx, body_name_attr,
                                          &body_must_be_const_nodes, &body));
  TF_RETURN_IF_ERROR(FindMustBeConstNodes(ctx, cond_name_attr,
                                          &cond_must_be_const_nodes, &cond));

  auto should_convert_to_const = [&](int arg_idx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_4(mht_4_v, 336, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "lambda");

    XlaCompiler::Argument& arg = (*args)[arg_idx];
    return arg.kind != XlaCompiler::Argument::kResource &&
           loop_invariants[arg_idx] &&
           (body_must_be_const_nodes[body->arg_nodes[arg_idx]->id()] ||
            cond_must_be_const_nodes[cond->arg_nodes[arg_idx]->id()]);
  };
  absl::InlinedVector<int, 5> converted_constants =
      ConvertCompileTimeConstArgumentsToConst(ctx, args,
                                              /*xla_expression_offset=*/0,
                                              should_convert_to_const);
  VLOG(2) << "Converted args to constants: {"
          << absl::StrJoin(converted_constants, ",") << "}";
  for (int arg_idx : converted_constants) {
    compile_time_const_arg_indices->at(arg_idx) = true;
    (*num_compile_time_const_args)++;
  }
  return Status::OK();
}

Status VerifyBodyInputAndOutputShapeMatch(
    XlaOpKernelContext* ctx,
    const std::vector<bool>& compile_time_const_arg_indices,
    const XlaCompiler::CompilationResult& body, bool has_token_input_output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_5(mht_5_v, 362, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "VerifyBodyInputAndOutputShapeMatch");

  xla::Shape body_input_shape = body.xla_input_shapes[0];
  xla::Shape body_output_shape;
  body_output_shape.set_element_type(xla::TUPLE);
  for (int i = 0; i < ctx->num_outputs(); i++) {
    if (!compile_time_const_arg_indices[i]) {
      *(body_output_shape.add_tuple_shapes()) =
          body.xla_output_shape.tuple_shapes(i);
    }
  }
  // If `body` has a token output, append its shape to `body_output_shape`.
  if (has_token_input_output) {
    *(body_output_shape.add_tuple_shapes()) =
        body.xla_output_shape.tuple_shapes(ctx->num_inputs());
  }
  if (!xla::ShapeUtil::Compatible(body_input_shape, body_output_shape)) {
    return errors::InvalidArgument(
        "Input and output shapes of loop body do not match: ",
        xla::ShapeUtil::HumanString(body_input_shape), " vs. ",
        xla::ShapeUtil::HumanString(body_output_shape));
  }
  return Status::OK();
}

StatusOr<xla::XlaComputation> BuildWrappedCond(
    XlaOpKernelContext* ctx, const XlaCompiler::CompilationResult& cond) {
  xla::Shape cond_input_shape = cond.xla_input_shapes[0];
  std::unique_ptr<xla::XlaBuilder> cb =
      ctx->builder()->CreateSubBuilder("cond_wrapper");
  auto inputs = xla::Parameter(cb.get(), 0, cond_input_shape, "inputs");
  auto outputs = xla::Call(cb.get(), *cond.computation, {inputs});
  xla::GetTupleElement(outputs, 0);
  return cb->Build();
}

StatusOr<xla::XlaComputation> BuildWrappedBody(
    XlaOpKernelContext* ctx, const XlaCompiler::CompilationResult& body,
    const std::vector<bool>& compile_time_const_arg_indices,
    int num_compile_time_const_args, bool has_token_input_output) {
  if (num_compile_time_const_args <= 0) {
    return xla::XlaComputation(body.computation->proto());
  }
  xla::XlaComputation body_wrapper;
  std::unique_ptr<xla::XlaBuilder> cb =
      ctx->builder()->CreateSubBuilder("body_wrapper");
  xla::Shape body_input_shape = body.xla_input_shapes[0];
  auto inputs = xla::Parameter(cb.get(), 0, body_input_shape, "inputs");
  // Call the original body function which has mismatched inputs and outputs
  // and strip the compile time consts from the list of outputs. While requires
  // the inputs and outputs of its body function to match.
  auto outputs = xla::Call(cb.get(), *body.computation, {inputs});
  std::vector<xla::XlaOp> non_compile_time_const_outputs;
  for (int i = 0; i < compile_time_const_arg_indices.size(); i++) {
    if (!compile_time_const_arg_indices[i]) {
      non_compile_time_const_outputs.push_back(
          xla::GetTupleElement(outputs, i));
    }
  }
  // If `body` has a token output, append it to
  // `non_compile_time_const_outputs`.
  if (has_token_input_output) {
    non_compile_time_const_outputs.push_back(
        xla::GetTupleElement(outputs, ctx->num_outputs()));
  }
  xla::Tuple(cb.get(), non_compile_time_const_outputs);
  return cb->Build();
}

xla::XlaOp BuildWhile(XlaOpKernelContext* ctx,
                      const xla::XlaComputation& wrapped_cond,
                      const xla::XlaComputation& wrapped_body,
                      const xla::XlaOp& initial_values,
                      const std::vector<int>& input_mapping,
                      const std::vector<bool>& compile_time_const_arg_indices,
                      int num_compile_time_const_args,
                      bool has_token_input_output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_6(mht_6_v, 440, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "BuildWhile");

  xla::XlaOp while_result =
      xla::While(wrapped_cond, wrapped_body, initial_values);
  std::vector<xla::XlaOp> padded_while_outputs(ctx->num_outputs());
  int while_result_index = 0;
  for (int i = 0; i < ctx->num_inputs(); i++) {
    if (!compile_time_const_arg_indices[i]) {
      padded_while_outputs[input_mapping[while_result_index]] =
          xla::GetTupleElement(while_result, while_result_index);
      while_result_index++;
    } else {
      padded_while_outputs[i] = ctx->Input(i);
    }
  }
  // If `body` has a token output, append it to `padded_while_outputs`.
  if (has_token_input_output) {
    padded_while_outputs.push_back(xla::GetTupleElement(
        while_result, ctx->num_inputs() - num_compile_time_const_args));
  }
  return xla::Tuple(ctx->builder(), padded_while_outputs);
}

}  // anonymous namespace

XlaWhileOp::XlaWhileOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_7(mht_7_v, 467, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "XlaWhileOp::XlaWhileOp");

  const NameAttrList* name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("cond", &name_attr));
  cond_name_attr_ = *name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("body", &name_attr));
  body_name_attr_ = *name_attr;
  if (!ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_).ok()) {
    has_token_input_output_ = false;
  } else {
    has_token_input_output_ = !token_input_nodes_.empty();
  }
  if (ctx->HasAttr(kPropagateCompileTimeConsts)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kPropagateCompileTimeConsts,
                                     &propagate_compile_time_consts_));
  }
  if (!ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                    &original_node_name_)
           .ok())
    original_node_name_ = name();
}

void XlaWhileOp::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhile_opDTcc mht_8(mht_8_v, 491, "", "./tensorflow/compiler/tf2xla/kernels/while_op.cc", "XlaWhileOp::Compile");

  VLOG(1) << "WhileOp::Compile";

  // Input resources need to be grouped in the end of the body function
  // according to the convention of the XLA bridge.
  OP_REQUIRES_OK(ctx, VerifyResourceArgsGroupedAtEnd(ctx, body_name_attr_));

  std::vector<XlaCompiler::Argument> arguments;
  bool has_uninitialized_vars;
  bool has_tensor_arrays;
  bool has_uninitialized_tensor_lists;
  OP_REQUIRES_OK(ctx, MakeXlaCompilerArgumentsFromInputs(
                          ctx, &arguments, &has_uninitialized_vars,
                          &has_tensor_arrays, &has_uninitialized_tensor_lists));

  xla::XlaBuilder* builder = ctx->builder();
  XlaCompiler* compiler = ctx->compiler();

  // Indices of loop vars which satisfy the following conditions:
  // 1. They are loop invariants.
  // 2. The op inputs at these indices are compile time constants.
  //
  // These compile time consts do not appear as _Args in the cond/body functions
  // and are replaced by kConstant nodes instead. As a result, the compiled
  // body function does not have matching input and output shape. We fix this
  // by rewriting the body computation (see body_wrapper below) to output
  // just the non compile-time-const values and later pad up the while output
  // with the const args.
  std::vector<bool> compile_time_const_arg_indices(ctx->num_inputs());
  int num_compile_time_const_args = 0;
  if (propagate_compile_time_consts_) {
    OP_REQUIRES_OK(ctx, ConvertLoopInvariantsToConst(
                            ctx, body_name_attr_, cond_name_attr_, &arguments,
                            &compile_time_const_arg_indices,
                            &num_compile_time_const_args, compiler->client()));
  }

  VLOG(1) << "Compiling body";

  // All resource that are inputs to the loop's body must also be
  // present as loop body outputs; the signature of the loop's input and
  // output must match. We ensure this by asking the compiler to include the
  // current values of all resources, even if they haven't been updated by the
  // computation. We must also ask the compiler to keep compile-time constant
  // outputs as part of the generated computation, for the same reason.
  // TODO(phawkins): consider adding loop-invariant inputs to XLA's While()
  // operator.
  XlaCompiler::CompileOptions body_options;
  body_options.use_tuple_arg = true;
  body_options.return_updated_values_for_all_resources = true;
  body_options.is_entry_computation = false;
  body_options.add_token_input_output = has_token_input_output_;
  XlaCompiler::CompilationResult body;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(body_options, body_name_attr_,
                                                arguments, &body));
  OP_REQUIRES_OK(
      ctx, ctx->xla_context()->RecordCollectiveInfoFromNestedCompilationResult(
               body));

  // We must use a static shape for parameters to an XLA compilation. However,
  // we may not know the shape of a resource if it is first
  // written inside the loop. Furthermore, we do not know ahead of time which
  // gradient TensorArrays will be created by the TensorArrayGradV3 operator.
  //
  // Ideally we would change TensorFlow to provide static shape always, but
  // but this is not easy to do. So if uninitialized resources or TensorArrays
  // are used by the loop body, we compile the body function twice:
  // 1) once with uninitialized resource inputs and no TensorArray gradient
  //    inputs. We then discard the computation but we assume resource shapes
  //    and the set of gradients read or written will reach a fixpoint after one
  //    iteration.
  //    Hence we can use the output shapes and TensorArray gradients of each
  //    resource as the "true" shapes.
  // 2) again with the "correct" resource information determined by (1).
  if (has_uninitialized_vars || has_tensor_arrays ||
      has_uninitialized_tensor_lists) {
    VLOG(2) << "Recompiling loop body: has_uninitialized_vars: "
            << has_uninitialized_vars
            << " has_tensor_arrays: " << has_tensor_arrays
            << " has_uninitialized_tensor_lists: "
            << has_uninitialized_tensor_lists;
    // Initializes any uninitialized resource with zero values of the
    // shape determined by the first compilation.
    for (int i = 0; i < body.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = body.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));

      XlaCompiler::Argument& arg = arguments[update.input_index];
      if (!arg.initialized) {
        VLOG(2) << "Update shape for argument " << update.input_index << " "
                << update.shape.DebugString();
        arg.initialized = true;

        arg.shape = update.shape;
        OP_REQUIRES_OK(ctx,
                       resource->SetTypeAndShape(update.type, update.shape));

        OP_REQUIRES_OK(ctx, resource->SetZeroValue(builder));
      }

      // Add any TensorArray gradients touched by the body to the enclosing
      // graph.
      for (const string& grad_source : update.tensor_array_gradients_accessed) {
        VLOG(4) << "TensorArray " << resource->name() << " accessed gradient "
                << grad_source;
        XlaResource* gradient;
        OP_REQUIRES_OK(ctx, resource->GetOrCreateTensorArrayGradient(
                                grad_source, builder, &gradient));
      }

      // Add all of the TensorArray gradients to the argument. For simplicity,
      // we always pass all known gradients.
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
    }

    // Set the shape of any uninitialized TensorLists to the shape determined by
    // the first compilation. Note that, unlike resources, we do not initialize
    // the input list with zeros here, that is done later.
    xla::Shape body_output_shape = body.xla_output_shape;
    OP_REQUIRES(ctx, body_output_shape.IsTuple(),
                errors::FailedPrecondition(
                    "xla_output_shape of while body must be a tuple."));
    for (int i = 0; i < arguments.size(); i++) {
      XlaCompiler::Argument& arg = arguments[i];
      if (arg.initialized || !IsTensorListInput(ctx, i)) {
        continue;
      }
      arg.shape = body_output_shape.tuple_shapes(i);
      arg.initialized = true;
    }

    // Recompile the body with the "correct" resource shapes.
    VLOG(1) << "Recompiling body with corrected resource shapes";
    body = {};
    OP_REQUIRES_OK(ctx, compiler->CompileFunction(body_options, body_name_attr_,
                                                  arguments, &body));
  }

  VLOG(1) << "Compiling condition";

  XlaCompiler::CompileOptions cond_options;
  cond_options.use_tuple_arg = true;
  cond_options.is_entry_computation = false;
  cond_options.add_token_input_output = has_token_input_output_;
  XlaCompiler::CompilationResult cond;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(cond_options, cond_name_attr_,
                                                arguments, &cond));

  OP_REQUIRES(ctx, body.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape body_input_shape = body.xla_input_shapes[0];
  OP_REQUIRES(ctx, body_input_shape.IsTuple(),
              errors::FailedPrecondition("Expected tuple shape"));
  OP_REQUIRES(ctx, cond.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape cond_input_shape = cond.xla_input_shapes[0];
  OP_REQUIRES(ctx, cond_input_shape.IsTuple(),
              errors::FailedPrecondition("Expected tuple shape"));

  VLOG(2) << "Body shape: " << xla::ShapeUtil::HumanString(body_input_shape)
          << " -> " << xla::ShapeUtil::HumanString(body.xla_output_shape);
  VLOG(2) << "Cond shape: " << xla::ShapeUtil::HumanString(cond_input_shape)
          << " -> " << xla::ShapeUtil::HumanString(cond.xla_output_shape);

  OP_REQUIRES(ctx,
              xla::ShapeUtil::Compatible(body_input_shape, cond_input_shape),
              errors::InvalidArgument(
                  "Input shapes of loop body and condition do not match: ",
                  xla::ShapeUtil::HumanString(body_input_shape), " vs. ",
                  xla::ShapeUtil::HumanString(cond_input_shape)));

  // Check that the shape of the body outputs excluding the compile time const
  // args (which are pruned from the body outputs in body_wapper) matches the
  // shape of the inputs.
  OP_REQUIRES_OK(ctx, VerifyBodyInputAndOutputShapeMatch(
                          ctx, compile_time_const_arg_indices, body,
                          has_token_input_output_));

  xla::Shape expected_cond_output_shape_without_side_effect =
      xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::PRED, {})});
  xla::Shape expected_cond_output_shape_with_side_effect =
      xla::ShapeUtil::MakeTupleShape({xla::ShapeUtil::MakeShape(xla::PRED, {}),
                                      xla::ShapeUtil::MakeTokenShape()});
  OP_REQUIRES(ctx,
              xla::ShapeUtil::Compatible(
                  cond.xla_output_shape,
                  expected_cond_output_shape_without_side_effect) ||
                  xla::ShapeUtil::Compatible(
                      cond.xla_output_shape,
                      expected_cond_output_shape_with_side_effect),
              errors::InvalidArgument(
                  "Output shape of loop condition should be (pred[]) or "
                  "(pred[], token[]), got: ",
                  xla::ShapeUtil::HumanString(cond.xla_output_shape)));

  int num_inputs = body.input_mapping.size();
  std::vector<xla::XlaOp> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    int input_num = body.input_mapping[i];
    if (has_token_input_output_ && i == num_inputs - 1) {
      // Set token input for this "while" op.
      std::vector<xla::XlaOp> token_inputs;
      token_inputs.reserve(token_input_nodes_.size());
      for (const string& node_name : token_input_nodes_) {
        auto token_or = compiler->GetNodeToken(node_name);
        OP_REQUIRES_OK(ctx, token_or.status());
        token_inputs.push_back(token_or.ValueOrDie());
      }
      inputs[i] = xla::AfterAll(builder, token_inputs);
    } else if (ctx->input_type(input_num) == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(input_num, &resource));
      OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], builder));
    } else if (IsTensorListInput(ctx, input_num)) {
      xla::XlaOp input = ctx->Input(input_num);
      auto input_shape_or = ctx->builder()->GetShape(input);
      OP_REQUIRES_OK(ctx, input_shape_or.status());
      xla::Shape input_shape = input_shape_or.ValueOrDie();
      const xla::Shape& list_shape = body_input_shape.tuple_shapes(i);
      // Shape/datatype of the input list may differ from shape/datatype of the
      // body/cond input if the list's shape/datatype was inferred after the
      // first compilation and the body/cond was recompiled with the updated
      // shape/datatype of the list.
      if (input_shape != list_shape) {
        // Prepare dynamic dimensions for element shapes.
        std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
        for (int i = 0; i < list_shape.tuple_shapes_size() - 1; ++i) {
          std::vector<xla::XlaOp> dynamic_dims;

          const xla::Shape& shape = list_shape.tuple_shapes(i);

          // We already have the dynamic size of leading dimension outside of
          // the while loop without initializing the TensorList inside the while
          // loop.
          if (shape.is_dynamic_dimension(0)) {
            xla::XlaOp leading_dim_size = xla::GetDimensionSize(input, 0);
            dynamic_dims.push_back(leading_dim_size);
          } else {
            int32_t dim_size = shape.dimensions(0);
            dynamic_dims.push_back(
                xla::ConstantR0<int32>(ctx->builder(), dim_size));
          }

          // Set dynamic dimension size to 0 for element value. Inside the while
          // loop, TensorlistSetItem will properly set the element shape's
          // dynamic dimension.
          for (int64_t dim = 1; dim < shape.dimensions_size(); ++dim) {
            int32_t dim_size = shape.dimensions(dim);
            if (shape.is_dynamic_dimension(dim)) {
              dim_size = 0;
            }
            dynamic_dims.push_back(
                xla::ConstantR0<int32_t>(ctx->builder(), dim_size));
          }
          list_dynamic_dims.push_back(dynamic_dims);
        }
        OP_REQUIRES_OK(
            ctx, CreateZerosTensorListWithShape(ctx->builder(), list_shape,
                                                list_dynamic_dims, &inputs[i]));
      } else {
        inputs[i] = ctx->Input(input_num);
      }
    } else {
      inputs[i] = ctx->Input(input_num);
    }
  }

  xla::XlaOp init = xla::Tuple(builder, inputs);

  VLOG(1) << "Building while loop";

  // Wraps the condition in a computation that unpacks the output tuple.
  StatusOr<xla::XlaComputation> cond_result = BuildWrappedCond(ctx, cond);
  OP_REQUIRES_OK(ctx, cond_result.status());
  xla::XlaComputation wrapped_cond = std::move(cond_result.ValueOrDie());

  // Remove compile time const args from the list of body outputs.
  StatusOr<xla::XlaComputation> body_result =
      BuildWrappedBody(ctx, body, compile_time_const_arg_indices,
                       num_compile_time_const_args, has_token_input_output_);
  OP_REQUIRES_OK(ctx, body_result.status());
  xla::XlaComputation wrapped_body = std::move(body_result.ValueOrDie());

  // Builds the While op and pads its output with the compile time const args.
  xla::XlaOp while_result =
      BuildWhile(ctx, wrapped_cond, wrapped_body, init, body.input_mapping,
                 compile_time_const_arg_indices, num_compile_time_const_args,
                 has_token_input_output_);

  // Sets non-variable outputs and determine when resource variables start.
  int resource_index = 0;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    if (ctx->input_type(i) != DT_RESOURCE) {
      if (IsTensorListInput(ctx, i)) {
        ctx->SetTensorListOutput(i, xla::GetTupleElement(while_result, i));
      } else {
        ctx->SetOutput(i, xla::GetTupleElement(while_result, i));
      }
      ++resource_index;
    } else {
      break;
    }
  }
  if (has_token_input_output_) {
    // Set token output for this "while" op.
    xla::XlaOp token_output =
        xla::GetTupleElement(while_result, ctx->num_outputs());
    auto shape_or = builder->GetShape(token_output);
    OP_REQUIRES_OK(ctx, shape_or.status());
    OP_REQUIRES(ctx, shape_or.ValueOrDie().IsToken(),
                errors::FailedPrecondition(
                    "Token output is not token type: ",
                    xla::ShapeUtil::HumanString(shape_or.ValueOrDie())));
    OP_REQUIRES_OK(ctx,
                   compiler->SetNodeToken(original_node_name_, token_output));
  }

  // Updates the values of any resource variables modified by the loop.
  for (int i = 0; i < body.resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& update = body.resource_updates[i];
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
    if (update.modified) {
      int pos = resource_index + i;
      OP_REQUIRES_OK(ctx,
                     resource->SetFromPack(
                         arguments[update.input_index].tensor_array_gradients,
                         xla::GetTupleElement(while_result, pos), builder));
    }
    VLOG(2) << "Loop-carried variable: pos: " << update.input_index
            << " name: " << resource->name() << " modified: " << update.modified
            << " type: " << DataTypeString(update.type)
            << " shape: " << update.shape.DebugString();
    // Copies the identity of the resource variable from input to output
    // unchanged, even if the variable was not modified.
    ctx->op_kernel_context()->set_output(
        update.input_index,
        ctx->op_kernel_context()->input(update.input_index));
  }

  VLOG(1) << "Done building while loop";
}

REGISTER_XLA_OP(Name("While").AllowResourceTypes().AllowVariantTypes(),
                XlaWhileOp);
REGISTER_XLA_OP(Name("StatelessWhile").AllowResourceTypes().AllowVariantTypes(),
                XlaWhileOp);
REGISTER_XLA_OP(Name("XlaWhile").AllowResourceTypes().AllowVariantTypes(),
                XlaWhileOp);

}  // namespace tensorflow
