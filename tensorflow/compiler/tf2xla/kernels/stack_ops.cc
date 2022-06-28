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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc() {
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

// XLA Stack operators.

#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

Status GetStackShape(xla::XlaBuilder* builder, XlaResource* resource,
                     TensorShape* stack_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "GetStackShape");

  auto shape_or_status = builder->GetShape(resource->value());
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  xla::Shape shape = shape_or_status.ValueOrDie();
  TF_RET_CHECK(shape.IsTuple());
  return XLAShapeToTensorShape(xla::ShapeUtil::GetTupleElementShape(shape, 0),
                               stack_shape);
}

// Since the element shape is not provided to the Stack operator,
// we lazily initialize the Stack at the time of the first write.
//
// If a Stack `resource` has not been initialized, constructs storage for the
// Stack with elements of `elem_shape`. For both initialized and
// uninitialized Stacks, checks that the tensor has a type compatible with
// 'dtype' and shape compatible with 'elem_shape'.
//
// TODO(phawkins): consider changing the API of the stack operators to
// allow an optional element shape at stack construction time.
Status MaybeInitializeStack(xla::XlaBuilder* builder, XlaResource* resource,
                            DataType dtype, const TensorShape& elem_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "MaybeInitializeStack");

  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "Stack dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  TensorShape stack_shape;
  stack_shape.AddDim(resource->max_array_size());
  stack_shape.AppendShape(elem_shape);

  if (!resource->initialized()) {
    // Stack has not been initialized.
    TF_RETURN_IF_ERROR(resource->SetTypeAndShape(dtype, elem_shape));
    TF_RETURN_IF_ERROR(resource->SetZeroValue(builder));
  } else {
    // Checks the expected shape matches the actual shape.
    TensorShape actual_shape;
    TF_RETURN_IF_ERROR(GetStackShape(builder, resource, &actual_shape));
    if (stack_shape != actual_shape) {
      return errors::InvalidArgument(
          "Mismatched Stack shapes: ", stack_shape.DebugString(), " vs ",
          actual_shape.DebugString());
    }
  }
  return Status::OK();
}

class StackOp : public XlaOpKernel {
 public:
  explicit StackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "StackOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("elem_type", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stack_name", &stack_name_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_3(mht_3_v, 276, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "Compile");

    int64_t max_size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &max_size));
    OP_REQUIRES(
        ctx, max_size >= 0,
        errors::InvalidArgument(
            "XLA compilation requires a fixed stack size upper bound. If "
            "you are using tf.while_loop, set the maximum_iterations parameter "
            "to fix this issue."));

    // We defer initializing the Stack resource until we see the first push.
    // Otherwise we do not know the shape of the stack elements.
    XlaResource* resource =
        ctx->xla_context()->AddResource(XlaResource::CreateStack(
            /*name=*/absl::StrCat("Stack: ", stack_name_), dtype_, max_size));
    ctx->SetResourceOutput(0, resource);
  }

 private:
  DataType dtype_;
  string stack_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(StackOp);
};

REGISTER_XLA_OP(
    Name("StackV2").CompileTimeConstantInput("max_size").CompilationOnly(),
    StackOp);

class StackPushOp : public XlaOpKernel {
 public:
  explicit StackPushOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_4(mht_4_v, 310, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "StackPushOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_5(mht_5_v, 317, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    TensorShape elem_shape = ctx->InputShape(1);

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    // Initializes the Stack, if the element shape was not already known.
    OP_REQUIRES_OK(ctx, MaybeInitializeStack(b, resource, dtype_, elem_shape));

    xla::XlaOp ta = xla::GetTupleElement(resource->value(), 0);
    xla::XlaOp index = xla::GetTupleElement(resource->value(), 1);
    xla::XlaOp value = ctx->Input(1);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    // TODO(phawkins): We don't check the index is in bounds --- there is no
    // error mechanism in XLA.
    OP_REQUIRES_OK(ctx,
                   resource->SetValue(xla::Tuple(
                       b, {xla::DynamicUpdateSlice(ta, update, start_indices),
                           xla::Add(index, xla::ConstantR0<int32>(b, 1))})));

    ctx->SetOutput(0, value);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StackPushOp);
};

REGISTER_XLA_OP(Name("StackPushV2").CompilationOnly(), StackPushOp);

class StackPopOp : public XlaOpKernel {
 public:
  explicit StackPopOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_6(mht_6_v, 363, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "StackPopOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("elem_type", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_7(mht_7_v, 370, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    // There is a somewhat subtle issue here: here "uninitialized" means we have
    // not yet seen a pop in the order that we compile operators, not the order
    // that we run them. However, in practice the two orders should be the same
    // for the sole user of the stack operators (loop gradients).
    OP_REQUIRES(ctx, resource->initialized(),
                errors::InvalidArgument("Stack pop on uninitialized stack"));

    TensorShape stack_shape;
    OP_REQUIRES_OK(ctx, GetStackShape(b, resource, &stack_shape));

    xla::XlaOp state = resource->value();
    xla::XlaOp ta = xla::GetTupleElement(state, 0);
    xla::XlaOp index = xla::GetTupleElement(state, 1);

    index = Sub(index, xla::ConstantR0<int32>(b, 1));
    OP_REQUIRES_OK(ctx, resource->SetValue(xla::Tuple(b, {ta, index})));

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(stack_shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    auto slice_shape = stack_shape.dim_sizes();
    slice_shape[0] = 1LL;

    // TODO(phawkins): We don't check the index is in bounds --- there is no
    // error mechanism in XLA.
    xla::XlaOp read = xla::DynamicSlice(ta, start_indices, slice_shape);

    // Remove the leading '1' dimension.
    std::vector<int64_t> value_shape(slice_shape.begin() + 1,
                                     slice_shape.end());
    ctx->SetOutput(0, xla::Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StackPopOp);
};

REGISTER_XLA_OP(Name("StackPopV2").CompilationOnly(), StackPopOp);

class StackCloseOp : public XlaOpKernel {
 public:
  explicit StackCloseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_8(mht_8_v, 424, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "StackCloseOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstack_opsDTcc mht_9(mht_9_v, 429, "", "./tensorflow/compiler/tf2xla/kernels/stack_ops.cc", "Compile");

    // Do nothing.
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StackCloseOp);
};

REGISTER_XLA_OP(Name("StackCloseV2").CompilationOnly(), StackCloseOp);

}  // anonymous namespace
}  // namespace tensorflow
