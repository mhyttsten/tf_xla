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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc() {
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

// XLA TensorArray operators.

#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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

// Since the element shape is not always provided to the TensorArrayV3 operator,
// we must support lazily initialization of the TensorArray at the time of the
// first write.
// If a TensorArray `resource` has not been initialized, constructs storage for
// the TensorArray with elements of `elem_shape`. For both initialized and
// uninitialized TensorArrays, checks that the tensor has a type compatible with
// 'dtype' and shape compatible with 'elem_shape'.
Status MaybeInitializeTensorArray(xla::XlaBuilder* builder,
                                  XlaResource* resource, DataType dtype,
                                  const TensorShape& elem_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "MaybeInitializeTensorArray");

  if (resource->kind() != XlaResource::kTensorArray) {
    return errors::InvalidArgument("Unexpected non-TensorArray resource");
  }

  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  TF_RET_CHECK(resource->max_array_size() >= 0)
      << resource->name() << " size " << resource->max_array_size();

  if (!resource->initialized()) {
    TF_RETURN_IF_ERROR(resource->SetTypeAndShape(dtype, elem_shape));
    TF_RETURN_IF_ERROR(resource->SetZeroValue(builder));
  } else {
    // Checks the elem_shape matches the TensorArray shape.
    auto shape_or_status = builder->GetShape(resource->value());
    if (!shape_or_status.ok()) {
      return shape_or_status.status();
    }
    TensorShape shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(shape_or_status.ValueOrDie(), &shape));

    TensorShape ta_shape;
    ta_shape.AddDim(resource->max_array_size());
    ta_shape.AppendShape(elem_shape);
    if (ta_shape != shape) {
      return errors::InvalidArgument(
          "Mismatched TensorArray sizes: ", ta_shape.DebugString(), " vs ",
          shape.DebugString());
    }
  }
  return Status::OK();
}

// Checks that the TensorArray 'resource' has been initialized, and has type
// 'dtype'. Sets 'shape' to the shape
Status CheckTensorArrayIsInitialized(const string& op_name,
                                     const XlaResource* resource,
                                     DataType dtype) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_1(mht_1_v, 268, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "CheckTensorArrayIsInitialized");

  if (resource->kind() != XlaResource::kTensorArray) {
    return errors::InvalidArgument(
        "Unexpected non-TensorArray resource passed to ", op_name);
  }
  if (!resource->initialized()) {
    return errors::InvalidArgument("Uninitialized TensorArray passed to ",
                                   op_name);
  }
  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  return Status::OK();
}

Status GetTensorArrayShape(const XlaResource* resource,
                           xla::XlaBuilder* builder, TensorShape* shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_2(mht_2_v, 290, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "GetTensorArrayShape");

  *shape = resource->shape();
  shape->InsertDim(0, resource->max_array_size());
  return Status::OK();
}

// Like XlaBuilder::DynamicUpdateSlice, but adds 'update' to the
// relevant slice of 'operand'.
xla::XlaOp DynamicAddSlice(xla::XlaBuilder* builder, const xla::XlaOp& operand,
                           const xla::XlaOp& update,
                           absl::Span<const int64_t> update_dims,
                           absl::Span<const xla::XlaOp> start_indices,
                           DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_3(mht_3_v, 305, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "DynamicAddSlice");

  xla::XlaOp current = xla::DynamicSlice(operand, start_indices, update_dims);
  xla::XlaOp sum =
      dtype == DT_BOOL ? xla::Or(current, update) : xla::Add(current, update);
  return xla::DynamicUpdateSlice(operand, sum, start_indices);
}

class TensorArrayOp : public XlaOpKernel {
 public:
  explicit TensorArrayOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_4(mht_4_v, 317, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_shape", &element_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    bool dynamic_size;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_size", &dynamic_size));
    OP_REQUIRES(
        ctx, !dynamic_size,
        errors::Unimplemented(
            "TensorArrays with dynamic size are not supported by XLA."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_array_name", &tensor_array_name_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_5(mht_5_v, 333, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    int64_t size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &size));
    OP_REQUIRES(ctx, size >= 0,
                errors::InvalidArgument("TensorArray size must be >= 0"));

    xla::XlaBuilder* b = ctx->builder();

    // Initializes the TensorArray value if we know the element shape.
    // Otherwise, defer initialization to the first write.
    xla::XlaOp value;
    TensorShape shape;
    if (element_shape_.IsFullyDefined()) {
      CHECK(element_shape_.AsTensorShape(&shape));
      TensorShape ta_shape;
      ta_shape.AddDim(size);
      ta_shape.AppendShape(shape);
      xla::XlaOp zero = XlaHelpers::Zero(b, dtype_);
      value = xla::Broadcast(zero, ta_shape.dim_sizes());
    }

    XlaResource* var =
        ctx->xla_context()->AddResource(XlaResource::CreateTensorArray(
            /*name=*/absl::StrCat("TensorArray: ", tensor_array_name_), dtype_,
            shape, /*initial_value=*/value, /*max_array_size=*/size));
    ctx->SetResourceOutput(0, var);

    Tensor flow(DT_FLOAT, TensorShape({}));
    flow.scalar<float>()() = 0.0f;
    ctx->SetConstantOutput(1, flow);
  }

 private:
  PartialTensorShape element_shape_;
  DataType dtype_;
  string tensor_array_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayOp);
};

REGISTER_XLA_OP(Name("TensorArrayV3").CompileTimeConstantInput("size"),
                TensorArrayOp);

class TensorArrayWriteOp : public XlaOpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayWriteOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_7(mht_7_v, 388, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    TensorShape elem_shape = ctx->InputShape(2);

    // Initializes the TensorArray, if the element shape was not known at
    // construction time.
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));

    xla::XlaOp ta = resource->value();
    xla::XlaOp index = ctx->Input(1);
    xla::XlaOp value = ctx->Input(2);
    xla::XlaOp flow = ctx->Input(3);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    xla::XlaOp written;
    if (resource->tensor_array_multiple_writes_aggregate()) {
      written = DynamicAddSlice(b, ta, update, slice_shape.dim_sizes(),
                                start_indices, dtype_);
    } else {
      // TODO(b/117569591): Ideally we would report an error in the case that we
      // see multiple writes to the same offset. Unfortunately there is no way
      // to report errors at the moment, so we silently overwrite.
      written = xla::DynamicUpdateSlice(ta, update, start_indices);
    }
    OP_REQUIRES_OK(ctx, resource->SetValue(written));
    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayWriteOp);
};

REGISTER_XLA_OP(Name("TensorArrayWriteV3"), TensorArrayWriteOp);

class TensorArrayReadOp : public XlaOpKernel {
 public:
  explicit TensorArrayReadOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_8(mht_8_v, 441, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayReadOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_9(mht_9_v, 448, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::XlaOp ta = resource->value();
    xla::XlaOp index = ctx->Input(1);

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(ta_shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    auto slice_shape = ta_shape.dim_sizes();
    slice_shape[0] = 1LL;

    xla::XlaOp read = xla::DynamicSlice(ta, start_indices, slice_shape);

    // Remove the leading '1' dimension.
    std::vector<int64_t> value_shape(slice_shape.begin() + 1,
                                     slice_shape.end());
    ctx->SetOutput(0, xla::Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayReadOp);
};

REGISTER_XLA_OP(Name("TensorArrayReadV3"), TensorArrayReadOp);

class TensorArrayGatherOp : public XlaOpKernel {
 public:
  explicit TensorArrayGatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_10(mht_10_v, 491, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayGatherOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_11(mht_11_v, 498, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, indices_shape.dims() == 1,
                errors::InvalidArgument("indices must be rank 1"));
    auto indices = ctx->Input(1);
    DataType index_type = ctx->input_type(1);

    xla::XlaOp ta = resource->value();

    // Look for the case where the gather takes a simple slice from the
    // tensor array (0, 1, 2, 3, 4, ..., N)
    std::vector<int64_t> const_indices;
    Status status = ctx->ConstantInputAsIntVector(1, &const_indices);
    if (status.ok()) {
      bool gather_is_dense_slice = true;
      for (auto i = 0; i < const_indices.size(); i++) {
        if (const_indices[i] != i) {
          gather_is_dense_slice = false;
          break;
        }
      }

      if (gather_is_dense_slice) {
        std::vector<int64_t> begin(ta_shape.dims(), 0);
        std::vector<int64_t> strides(ta_shape.dims(), 1);
        std::vector<int64_t> end(ta_shape.dims(), 1);
        end[0] = const_indices.size();
        for (auto i = 1; i < ta_shape.dims(); i++) {
          end[i] = ta_shape.dim_size(i);
        }
        ctx->SetOutput(0, xla::Slice(ta, begin, end, strides));
        return;
      }
    }

    xla::XlaOp gather;
    OP_REQUIRES_OK(
        ctx,
        XlaGather(ta, ta_shape, indices, indices_shape, /*axis=*/0,
                  /*indices_are_nd=*/false, dtype_, index_type, b, &gather));
    ctx->SetOutput(0, gather);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGatherOp);
};

REGISTER_XLA_OP(Name("TensorArrayGatherV3"), TensorArrayGatherOp);

class TensorArrayScatterOp : public XlaOpKernel {
 public:
  explicit TensorArrayScatterOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_12(mht_12_v, 564, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayScatterOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_13(mht_13_v, 571, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    const TensorShape value_shape = ctx->InputShape(2);

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    TensorShape elem_shape = value_shape;
    elem_shape.RemoveDim(0);
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));

    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, indices_shape.dims() >= 1,
                errors::InvalidArgument("indices must be rank 1"));
    const int num_indices = indices_shape.dim_size(0);
    const xla::XlaOp indices = ctx->Input(1);

    xla::XlaOp ta = resource->value();
    const xla::XlaOp value = ctx->Input(2);
    const xla::XlaOp flow = ctx->Input(3);

    // Look for the case where the scatter is for each sub-tensor in order. The
    // tensor array implementation allows for this to be a straight addition.
    bool scatter_all_elements_in_order = false;
    std::vector<int64_t> const_indices;
    Status status = ctx->ConstantInputAsIntVector(1, &const_indices);
    if (status.ok() && num_indices == value_shape.dim_size(0)) {
      scatter_all_elements_in_order = true;
      for (auto i = 0; i < num_indices; i++) {
        if (const_indices[i] != i) {
          scatter_all_elements_in_order = false;
          break;
        }
      }
    }

    if (scatter_all_elements_in_order) {
      if (dtype_ == DT_BOOL) {
        ta = xla::Or(ta, value);
      } else {
        ta = xla::Add(ta, value);
      }
    } else {
      auto slice_dims = value_shape.dim_sizes();
      slice_dims[0] = 1LL;

      std::vector<int64_t> value_starts(value_shape.dims(), 0);
      auto value_ends = value_shape.dim_sizes();

      std::vector<int64_t> value_strides(value_shape.dims(), 1);

      // For every (index, value) pair, update the corresponding TensorArray
      // storage.
      for (int i = 0; i < num_indices; ++i) {
        // Slice out part of the value.
        value_starts[0] = i;
        value_ends[0] = i + 1;
        auto slice = xla::Slice(value, value_starts, value_ends, value_strides);

        // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
        auto index = xla::Reshape(xla::Slice(indices, {i}, {i + 1}, {1}), {});
        std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                              xla::ConstantR0<int32>(b, 0));
        start_indices[0] = index;
        ta = DynamicAddSlice(b, ta, slice, slice_dims, start_indices, dtype_);
      }
    }

    OP_REQUIRES_OK(ctx, resource->SetValue(ta));
    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayScatterOp);
};

REGISTER_XLA_OP(Name("TensorArrayScatterV3"), TensorArrayScatterOp);

class TensorArrayConcatOp : public XlaOpKernel {
 public:
  explicit TensorArrayConcatOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_14(mht_14_v, 657, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayConcatOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_15(mht_15_v, 664, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::XlaOp ta = resource->value();

    auto ta_dims = ta_shape.dim_sizes();
    std::vector<int64_t> shape(ta_dims.begin() + 1, ta_dims.end());
    shape[0] *= ta_shape.dim_size(0);
    ctx->SetOutput(0, xla::Reshape(ta, shape));

    Tensor lengths(DT_INT64, {ta_dims[0]});
    auto lengths_vec = lengths.vec<int64_t>();
    for (int i = 0; i < ta_dims[0]; ++i) {
      lengths_vec(i) = ta_dims[1];
    }
    ctx->SetConstantOutput(1, lengths);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayConcatOp);
};

REGISTER_XLA_OP(Name("TensorArrayConcatV3"), TensorArrayConcatOp);

class TensorArraySplitOp : public XlaOpKernel {
 public:
  explicit TensorArraySplitOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_16(mht_16_v, 703, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArraySplitOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_17(mht_17_v, 710, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    std::vector<int64_t> lengths;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &lengths));

    int64_t length = 0;
    if (!lengths.empty()) {
      length = lengths[0];
      for (int i = 1; i < lengths.size(); ++i) {
        OP_REQUIRES(ctx, lengths[i] == length,
                    errors::InvalidArgument("lengths must be equal: ", length,
                                            " vs. ", lengths[i]));
      }
    }

    TensorShape value_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, value_shape.dims() >= 1,
                errors::InvalidArgument("value must have rank >= 1, got ",
                                        value_shape.DebugString()));
    TensorShape elem_shape = value_shape;
    elem_shape.set_dim(0, length);

    xla::XlaBuilder* b = ctx->builder();
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));
    xla::XlaOp ta = resource->value();

    TensorShape ta_shape;
    ta_shape.AddDim(resource->max_array_size());
    ta_shape.AppendShape(elem_shape);

    OP_REQUIRES(ctx, lengths.size() == resource->max_array_size(),
                errors::InvalidArgument(
                    "TensorArray's size is not equal to the size of lengths (",
                    lengths.size(), " vs. ", resource->max_array_size(), ")"));

    const xla::XlaOp value = ctx->Input(1);
    const xla::XlaOp flow = ctx->Input(3);

    OP_REQUIRES(ctx, value_shape.num_elements() == ta_shape.num_elements(),
                errors::InvalidArgument("mismatched element count ",
                                        value_shape.DebugString(), " vs. ",
                                        ta_shape.DebugString()));

    const xla::XlaOp reshape = xla::Reshape(value, ta_shape.dim_sizes());
    if (dtype_ == DT_BOOL) {
      ta = xla::Or(ta, reshape);
    } else {
      ta = xla::Add(ta, reshape);
    }
    OP_REQUIRES_OK(ctx, resource->SetValue(ta));

    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArraySplitOp);
};

REGISTER_XLA_OP(Name("TensorArraySplitV3").CompileTimeConstantInput("lengths"),
                TensorArraySplitOp);

class TensorArraySizeOp : public XlaOpKernel {
 public:
  explicit TensorArraySizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_18(mht_18_v, 780, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArraySizeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_19(mht_19_v, 785, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    XlaResource* var;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &var));
    Tensor size_tensor(DT_INT32, {});
    size_tensor.scalar<int32>()() = static_cast<int32>(var->max_array_size());
    ctx->SetConstantOutput(0, size_tensor);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorArraySizeOp);
};

REGISTER_XLA_OP(Name("TensorArraySizeV3"), TensorArraySizeOp);

class TensorArrayGradOp : public XlaOpKernel {
 public:
  explicit TensorArrayGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_20(mht_20_v, 804, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayGradOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("source", &source_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_21(mht_21_v, 811, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(
        ctx, CheckTensorArrayIsInitialized(name(), resource, resource->type()));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    // Finds or looks up the corresponding gradient TensorArray, which stores
    // gradients computed during backpropagation.
    XlaResource* gradient;
    OP_REQUIRES_OK(
        ctx, resource->GetOrCreateTensorArrayGradient(source_, b, &gradient));

    ctx->SetResourceOutput(0, gradient);
    ctx->SetConstantOutput(1, Tensor(DT_FLOAT));
  }

 private:
  string source_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGradOp);
};

REGISTER_XLA_OP(Name("TensorArrayGradV3"), TensorArrayGradOp);

class TensorArrayCloseOp : public XlaOpKernel {
 public:
  explicit TensorArrayCloseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_22(mht_22_v, 845, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "TensorArrayCloseOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStensor_array_opsDTcc mht_23(mht_23_v, 850, "", "./tensorflow/compiler/tf2xla/kernels/tensor_array_ops.cc", "Compile");

    // Do nothing; XLA handles resource management.
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayCloseOp);
};

REGISTER_XLA_OP(Name("TensorArrayCloseV3"), TensorArrayCloseOp);

}  // anonymous namespace
}  // namespace tensorflow
