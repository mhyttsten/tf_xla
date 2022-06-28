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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/data_flow_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>
#include <vector>
// TODO(b/31496047): Fix non-standard include order.
#include <numeric>  // clang-format off

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/kernels/tensor_array.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/ptr_util.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// clang-format on

namespace tensorflow {

Status GetHandle(OpKernelContext* ctx, string* container, string* ta_handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "GetHandle");

  {
    Tensor tensor;
    // Assuming that handle is the input at index 0.
    if (IsRefType(ctx->input_dtype(0))) {
      tensor = ctx->mutable_input(0, false);
    } else {
      tensor = ctx->input(0);
    }
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Tensor array handle must be 2-element vector, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<tstring>();
    *container = h(0);
    *ta_handle = h(1);
  }
  return Status::OK();
}

Status GetTensorArray(OpKernelContext* ctx, TensorArray** tensor_array) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "GetTensorArray");

  string container;
  string ta_handle;
  if (ctx->input_dtype(0) != DT_RESOURCE) {
    TF_RETURN_IF_ERROR(GetHandle(ctx, &container, &ta_handle));
    ResourceMgr* rm = ctx->resource_manager();
    if (rm == nullptr) return errors::Internal("No resource manager.");
    TF_RETURN_IF_ERROR(
        ctx->step_container()->Lookup(rm, container + ta_handle, tensor_array));
    return Status::OK();
  } else {
    return LookupResource(ctx, HandleFromInput(ctx, 0), tensor_array);
  }
}

Status SetupFlowControlInputs(OpKernelContext* ctx, bool set_output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "SetupFlowControlInputs");

  const Tensor* flow_control;
  TF_RETURN_IF_ERROR(ctx->input("flow_in", &flow_control));
  if (set_output) {
    TF_RETURN_IF_ERROR(ctx->set_output("flow_out", *flow_control));
  }
  return Status::OK();
}

// CREATION *******************************************************************

// Virtual class for shared behavior between TensorArrayOp and
// TensorArrayGradOp.
class TensorArrayCreationOp : public OpKernel {
 public:
  explicit TensorArrayCreationOp(OpKernelConstruction* context)
      : OpKernel(context), device_type_(context->device_type()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_3(mht_3_v, 285, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayCreationOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    Tensor tensor_array_output_handle;

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            tensorflow::DT_STRING, tensorflow::TensorShape({2}),
                            &tensor_array_output_handle, alloc_attr));
    // Store the handle in a per-step container of the RM.
    ResourceMgr* rm = ctx->resource_manager();
    OP_REQUIRES(ctx, rm != nullptr, errors::Internal("No resource manager."));

    TensorArray* output_tensor_array;
    OP_REQUIRES_OK(ctx, CreateTensorArray(ctx, rm, &tensor_array_output_handle,
                                          &output_tensor_array));
    if (IsRefType(ctx->expected_output_dtype(0))) {
      ctx->set_output_ref(0, output_tensor_array->mu(),
                          output_tensor_array->handle());
    } else if (ctx->expected_output_dtype(0) == DT_STRING) {
      ctx->set_output(0, *output_tensor_array->handle());
    } else {
      Tensor* handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
      handle->flat<ResourceHandle>()(0) =
          output_tensor_array->resource_handle(ctx);
    }
    if (ctx->num_outputs() == 2) {
      // Create the flow output.
      Tensor* flow;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &flow));
      if (device_type_ == DEVICE_CPU) {
        // Value doesn't matter, but this makes msan not complaint about
        // copying an uninitialized value. To do this on GPU would require
        // a kernel launch or a host->device memcpy, so we avoid that.
        flow->flat<float>()(0) = 0;
      }
    }
  }

 protected:
  virtual Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                                   Tensor* tensor_array_output_handle,
                                   TensorArray** output_tensor_array) = 0;

 private:
  const DeviceType device_type_;
};

// A per-run local tensor array. The tensor array uses a "per-step" resource
// manager which ensures that correct garbage collection on error or
// successful completion.
class TensorArrayOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_5(mht_5_v, 347, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayOp");

    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("element_shape", &element_shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dynamic_size", &dynamic_size_));
    // The HasAttr check is for backwards compatibility with older op
    // versions which do not have this attribute.
    if (context->HasAttr("identical_element_shapes")) {
      OP_REQUIRES_OK(context, context->GetAttr("identical_element_shapes",
                                               &identical_element_shapes_));
    } else {
      identical_element_shapes_ = false;
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("clear_after_read", &clear_after_read_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_array_name", &tensor_array_name_));
    if (tensor_array_name_.empty()) tensor_array_name_ = name();
  }

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_6(mht_6_v, 371, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "CreateTensorArray");

    const Tensor* tensor_size;
    TF_RETURN_IF_ERROR(ctx->input("size", &tensor_size));

    if (!TensorShapeUtils::IsScalar(tensor_size->shape())) {
      return errors::InvalidArgument(
          "TensorArray size must be scalar, but had shape: ",
          tensor_size->shape().DebugString());
    }
    const int32_t size = tensor_size->scalar<int32>()();
    if (size < 0) {
      return errors::InvalidArgument("Size should be >= 0.");
    }

    auto handle = tensor_array_output_handle->flat<tstring>();
    string unique_tensor_array_name =
        strings::StrCat(tensor_array_name_, "_",
                        TensorArray::tensor_array_counter.fetch_add(1));
    handle(0) = "_tensor_arrays";
    handle(1) = unique_tensor_array_name;

    auto key = strings::StrCat(handle(0), unique_tensor_array_name);

    TensorArray* tensor_array = new TensorArray(
        key, dtype_, *tensor_array_output_handle, size, element_shape_,
        identical_element_shapes_, dynamic_size_,
        false /* multiple_writes_aggregate */, false /* is_grad */,
        -1 /* marked_size */, clear_after_read_);

    TF_RETURN_IF_ERROR(ctx->step_container()->Create(rm, key, tensor_array));

    *output_tensor_array = tensor_array;

    return Status::OK();
  }

 private:
  DataType dtype_;
  PartialTensorShape element_shape_;
  bool identical_element_shapes_;
  bool dynamic_size_;
  bool clear_after_read_;
  string tensor_array_name_;  // The name used to create the TensorArray.

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArray").Device(DEVICE_CPU), TensorArrayOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayV2").Device(DEVICE_CPU),
                        TensorArrayOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayV3").Device(DEVICE_CPU),
                        TensorArrayOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArray")                \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOp);                    \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayV2")              \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOp);                    \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayV3")              \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOp);

TF_CALL_int64(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// GRADIENT *******************************************************************
// Note that this op may have an optional third input. If present, it represents
// a shape value. It indicates that element shape of this gradient array is that
// shape value concatenated with the element shape of the original tensor array.
// See TensorArrayGradWithShape.
class TensorArrayGradOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayGradOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_7(mht_7_v, 465, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayGradOp");

    OP_REQUIRES_OK(context, context->GetAttr("source", &source_));
  }

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_8(mht_8_v, 474, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "CreateTensorArray");

    string container;
    string tensor_array_name;
    if (ctx->input_dtype(0) != DT_RESOURCE) {
      TF_RETURN_IF_ERROR(GetHandle(ctx, &container, &tensor_array_name));
      if (container != "_tensor_arrays") {
        return errors::InvalidArgument(
            "Input container should be '_tensor_arrays',  but received '",
            container, "'");
      }
    } else {
      container = "_tensor_arrays";
      const auto& resource = ctx->input(0).flat<ResourceHandle>()(0);
      if (StringPiece(resource.name()).substr(0, container.size()) !=
          container) {
        return errors::InvalidArgument("Wrong input container. ",
                                       resource.name());
      }
      tensor_array_name =
          string(StringPiece(resource.name()).substr(container.size()));
    }

    auto output_handle = tensor_array_output_handle->flat<tstring>();
    output_handle(0) = "_tensor_array_grads";
    output_handle(1) = strings::StrCat(tensor_array_name, "@", source_);

    TensorArray* tensor_array;
    TF_RETURN_IF_ERROR(ctx->step_container()->Lookup(
        rm, strings::StrCat(container, tensor_array_name), &tensor_array));
    core::ScopedUnref unref(tensor_array);

    // Once gradients are being calculated, the forward TensorArray
    // may no longer be resized by new Writes.
    tensor_array->DisableDynamicSize();

    int32_t array_size = 0;
    int32_t marked_size = 0;
    TF_RETURN_IF_ERROR(tensor_array->Size(&array_size));
    TF_RETURN_IF_ERROR(tensor_array->MarkedSize(&marked_size));

    if (array_size < 0) {
      return errors::InvalidArgument("ArraySize should be >= 0.");
    }
    if (!tensor_array->GradientsAllowed()) {
      return errors::InvalidArgument(
          "Unable to create a gradients TensorArray for ", tensor_array_name,
          ".  Perhaps you used the multiple_writes_aggregate flag on a "
          "previous write?  Gradient calculation is impossible when multiple "
          "writes are performed to the same index.");
    }
    TensorShape shape_to_prepend;
    auto element_shape = PartialTensorShape();
    if (ctx->num_inputs() > 2) {
      TF_RETURN_IF_ERROR(tensor::MakeShape(ctx->input(2), &shape_to_prepend));
      auto ta_element_shape = tensor_array->ElemShape();
      if (!ta_element_shape.unknown_rank()) {
        std::vector<int64_t> dims;
        for (auto dim : shape_to_prepend) {
          dims.push_back(dim.size);
        }
        for (auto dim : ta_element_shape) {
          dims.push_back(dim.size);
        }
        TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
            gtl::ArraySlice<int64_t>(dims), &element_shape));
      }
    } else {
      element_shape = tensor_array->ElemShape();
    }

    const auto key = strings::StrCat(output_handle(0), output_handle(1));
    auto creator = [key, tensor_array, array_size, marked_size, element_shape,
                    shape_to_prepend,
                    tensor_array_output_handle](TensorArray** ret) -> Status {
      *ret = new TensorArray(
          key, tensor_array->ElemType(), *tensor_array_output_handle,
          array_size, element_shape, tensor_array->HasIdenticalElementShapes(),
          false /* dynamic_size */, true /* multiple_writes_aggregate */,
          true /* is_grad */, marked_size /* marked_size */,
          true /* close_after_read */);
      return (*ret)->CopyShapesFrom(tensor_array, &shape_to_prepend);
    };

    Status s = ctx->step_container()->LookupOrCreate<TensorArray>(
        rm, key, output_tensor_array, creator);
    (*output_tensor_array)->Unref();

    return s;
  }

 private:
  // The gradient source for creating the given
  // gradient TensorArray.  This should be unique to each gradients
  // call.  Typical values look like "gradients", "gradients_1", ...
  string source_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGradOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad").Device(DEVICE_CPU),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradV2").Device(DEVICE_CPU),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradV3").Device(DEVICE_CPU),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradWithShape").Device(DEVICE_CPU),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradV3")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayGradWithShape")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("shape_to_prepend")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);

// WRITE **********************************************************************

template <typename Device, typename T>
class TensorArrayWriteOp : public OpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_9(mht_9_v, 612, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayWriteOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_10(mht_10_v, 617, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    const Tensor* tensor_index;
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const int32_t index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    Status s =
        tensor_array->WriteOrAggregate<Device, T>(ctx, index, tensor_value);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_WRITE(type)                                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayWrite").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      TensorArrayWriteOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayWriteV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayWriteOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayWriteV3").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayWriteOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_WRITE);

#undef REGISTER_WRITE

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWrite")              \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("handle")             \
                              .HostMemory("index"),             \
                          TensorArrayWriteOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWriteV2")            \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("handle")             \
                              .HostMemory("index"),             \
                          TensorArrayWriteOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWriteV3")            \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("handle")             \
                              .HostMemory("index"),             \
                          TensorArrayWriteOp<GPUDevice, type>);

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// READ ***********************************************************************

template <typename Device, typename T>
class TensorArrayReadOp : public OpKernel {
 public:
  explicit TensorArrayReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_11(mht_11_v, 699, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayReadOp");

    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_12(mht_12_v, 706, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    const Tensor* tensor_index;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);

    const int32_t index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));
    Tensor value;
    Status s = tensor_array->Read<Device, T>(ctx, index, &value);
    OP_REQUIRES_OK(ctx, s);
    ctx->set_output(0, value);
  }

 private:
  DataType dtype_;
};

#define REGISTER_READ(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayRead")              \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<type>("dtype"),  \
                          TensorArrayReadOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayReadV2")            \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<type>("dtype"),  \
                          TensorArrayReadOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayReadV3")            \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<type>("dtype"),  \
                          TensorArrayReadOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_READ)

#undef REGISTER_READ

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayRead")              \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("dtype")   \
                              .HostMemory("handle")            \
                              .HostMemory("index"),            \
                          TensorArrayReadOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayReadV2")            \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("dtype")   \
                              .HostMemory("handle")            \
                              .HostMemory("index"),            \
                          TensorArrayReadOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayReadV3")            \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("dtype")   \
                              .HostMemory("handle")            \
                              .HostMemory("index"),            \
                          TensorArrayReadOp<GPUDevice, type>);

TF_CALL_int64(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// PACK and GATHER ************************************************************

// Concatenate the elements in a TensorArray.  All elements must be
// defined and have the same shape.
template <typename Device, typename T, bool LEGACY_PACK>
class TensorArrayPackOrGatherOp : public OpKernel {
 public:
  typedef typename TTypes<T, 2>::ConstMatrix ConstMatrix;
  typedef std::vector<std::unique_ptr<ConstMatrix> > ConstMatrixVector;

  explicit TensorArrayPackOrGatherOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_13(mht_13_v, 799, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayPackOrGatherOp");

    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("element_shape", &element_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_14(mht_14_v, 807, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));

    core::ScopedUnref unref(tensor_array);
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));

    // Ensure new element shape is compatible with the one stored in the
    // TensorArray.
    OP_REQUIRES_OK(ctx, tensor_array->SetElemShape(element_shape_));

    int32_t num_indices;
    std::vector<Tensor> values;
    std::vector<int32> indices;
    if (LEGACY_PACK) {
      OP_REQUIRES_OK(ctx, tensor_array->PackOrConcatSize(&num_indices));
      indices.resize(num_indices);
      std::iota(indices.begin(), indices.end(), 0);
    } else {
      const Tensor* tensor_indices;
      OP_REQUIRES_OK(ctx, ctx->input("indices", &tensor_indices));
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(tensor_indices->shape()),
                  errors::InvalidArgument(
                      "Expected indices to be a vector, but received shape: ",
                      tensor_indices->shape().DebugString()));
      const auto indices_t = tensor_indices->vec<int32>();
      num_indices = tensor_indices->NumElements();
      indices.resize(num_indices);
      std::copy(indices_t.data(), indices_t.data() + num_indices,
                indices.begin());
    }

    // If there are no elements to return, return a zero-element Tensor with
    // shape [0] + element_shape_
    if (num_indices == 0) {
      OP_REQUIRES(ctx, element_shape_.IsFullyDefined(),
                  errors::Unimplemented(
                      "TensorArray has size zero, but element shape ",
                      element_shape_.DebugString(),
                      " is not fully defined. "
                      "Currently only static shapes are supported when packing "
                      "zero-size TensorArrays."));
      TensorShape empty_shape;
      element_shape_.AsTensorShape(&empty_shape);
      empty_shape.InsertDim(0, 0);
      Tensor* empty_unused;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, empty_shape, &empty_unused));
      return;
    }

    // Read all the Tensors into a vector to keep track of their memory.
    Status s = tensor_array->ReadMany<Device, T>(ctx, indices, &values);
    OP_REQUIRES_OK(ctx, s);

    const Tensor* value_0_t = &values[0];

    OP_REQUIRES(
        ctx, element_shape_.IsCompatibleWith(value_0_t->shape()),
        errors::InvalidArgument("TensorArray was passed element_shape ",
                                element_shape_.DebugString(),
                                " which does not match the Tensor at index 0: ",
                                value_0_t->shape().DebugString()));

    TensorShape output_shape(value_0_t->shape());
    output_shape.InsertDim(0, num_indices);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    // If output_tensor is empty, there is nothing to concatenate so return it.
    if (output_shape.num_elements() == 0) {
      return;
    }

    ConstMatrixVector input_tensors_flat;
    input_tensors_flat.reserve(num_indices);
    auto output_flat =
        output_tensor->shaped<T, 2>({1, output_shape.num_elements()});

    // Insert the first value
    input_tensors_flat.push_back(MakeUnique<ConstMatrix>(
        value_0_t->shaped<T, 2>({1, value_0_t->NumElements()})));

    for (int i = 1; i < num_indices; ++i) {
      const Tensor* value_t = &values[i];
      OP_REQUIRES(
          ctx, value_0_t->shape() == value_t->shape(),
          errors::InvalidArgument(
              "TensorArray has inconsistent shapes.  Index 0 has shape: ",
              value_0_t->shape().DebugString(), " but index ", i,
              " has shape: ", value_t->shape().DebugString()));
      input_tensors_flat.push_back(MakeUnique<ConstMatrix>(
          value_t->shaped<T, 2>({1, value_t->NumElements()})));
    }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (std::is_same<Device, GPUDevice>::value) {
      ConcatGPU<T>(ctx, input_tensors_flat, output_tensor, &output_flat);
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    ConcatCPU<T>(ctx->device(), input_tensors_flat, &output_flat);
  }

 private:
  DataType dtype_;
  PartialTensorShape element_shape_;
};

#define REGISTER_GATHER_AND_PACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayPack")                                               \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("dtype"),                                   \
      TensorArrayPackOrGatherOp<CPUDevice, type, true /* LEGACY_PACK */>);  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGather")                                             \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("dtype"),                                   \
      TensorArrayPackOrGatherOp<CPUDevice, type, false /* LEGACY_PACK */>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGatherV2")                                           \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("dtype"),                                   \
      TensorArrayPackOrGatherOp<CPUDevice, type, false /* LEGACY_PACK */>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGatherV3")                                           \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("dtype"),                                   \
      TensorArrayPackOrGatherOp<CPUDevice, type, false /* LEGACY_PACK */>);

TF_CALL_POD_STRING_TYPES(REGISTER_GATHER_AND_PACK);
TF_CALL_variant(REGISTER_GATHER_AND_PACK);
REGISTER_GATHER_AND_PACK(quint8);
REGISTER_GATHER_AND_PACK(qint8);
REGISTER_GATHER_AND_PACK(qint32);

#undef REGISTER_GATHER_AND_PACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                                  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayPack")                                               \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("dtype")                                    \
          .HostMemory("handle"),                                            \
      TensorArrayPackOrGatherOp<GPUDevice, type, true /* LEGACY_PACK */>);  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGather")                                             \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("dtype")                                    \
          .HostMemory("indices")                                            \
          .HostMemory("handle"),                                            \
      TensorArrayPackOrGatherOp<GPUDevice, type, false /* LEGACY_PACK */>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGatherV2")                                           \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("dtype")                                    \
          .HostMemory("indices")                                            \
          .HostMemory("handle"),                                            \
      TensorArrayPackOrGatherOp<GPUDevice, type, false /* LEGACY_PACK */>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorArrayGatherV3")                                           \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("dtype")                                    \
          .HostMemory("indices")                                            \
          .HostMemory("handle"),                                            \
      TensorArrayPackOrGatherOp<GPUDevice, type, false /* LEGACY_PACK */>);

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("TensorArrayGather")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("dtype")
        .HostMemory("indices")
        .HostMemory("handle"),
    TensorArrayPackOrGatherOp<CPUDevice, int32, false /* LEGACY_PACK */>);
REGISTER_KERNEL_BUILDER(
    Name("TensorArrayGatherV2")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("dtype")
        .HostMemory("indices")
        .HostMemory("handle"),
    TensorArrayPackOrGatherOp<CPUDevice, int32, false /* LEGACY_PACK */>);
REGISTER_KERNEL_BUILDER(
    Name("TensorArrayGatherV3")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("dtype")
        .HostMemory("indices")
        .HostMemory("handle"),
    TensorArrayPackOrGatherOp<CPUDevice, int32, false /* LEGACY_PACK */>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// CONCAT *********************************************************************

// Concatenate the elements in a TensorArray.  All elements must be
// defined and (excepting the first dimension) have the same shape.
template <typename Device, typename T>
class TensorArrayConcatOp : public OpKernel {
 public:
  typedef typename TTypes<T, 2>::ConstMatrix ConstMatrix;
  typedef std::vector<std::unique_ptr<ConstMatrix> > ConstMatrixVector;

  explicit TensorArrayConcatOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_15(mht_15_v, 1029, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayConcatOp");

    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("element_shape_except0",
                                             &element_shape_except0_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_16(mht_16_v, 1038, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));

    int32_t array_size;
    OP_REQUIRES_OK(ctx, tensor_array->PackOrConcatSize(&array_size));

    // If there are no elements, return a zero-element Tensor with
    // shape [0] + element_shape_except0_
    if (array_size == 0) {
      OP_REQUIRES(
          ctx, element_shape_except0_.IsFullyDefined(),
          errors::Unimplemented(
              "TensorArray has size zero, but element_shape_except0 ",
              element_shape_except0_.DebugString(),
              " is not fully defined. "
              "Currently only static shapes are supported when concatenating "
              "zero-size TensorArrays."));
      TensorShape empty_shape;
      element_shape_except0_.AsTensorShape(&empty_shape);
      empty_shape.InsertDim(0, 0);
      Tensor* empty_unused;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, empty_shape, &empty_unused));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {0}, &empty_unused));
      return;
    }

    // Read all the Tensors into a vector to keep track of their memory.
    std::vector<Tensor> values;
    std::vector<int32> indices(array_size);
    std::iota(indices.begin(), indices.end(), 0);
    Status s = tensor_array->ReadMany<Device, T>(ctx, indices, &values);
    OP_REQUIRES_OK(ctx, s);

    Tensor* lengths_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       1, TensorShape({static_cast<int64_t>(values.size())}),
                       &lengths_tensor));
    auto lengths_tensor_t = lengths_tensor->vec<int64_t>();

    TensorShape output_shape;
    TensorShape output_shape_except0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      TensorShape value_shape_t = values[i].shape();

      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVectorOrHigher(value_shape_t),
          errors::InvalidArgument(
              "Concat saw a scalar shape at index ", i,
              " but requires at least vectors.  Did you mean to call pack?"));

      lengths_tensor_t(i) = value_shape_t.dim_size(0);

      TensorShape value_shape_t_except0 = value_shape_t;
      value_shape_t_except0.RemoveDim(0);
      if (i == 0) {
        output_shape = value_shape_t;
        output_shape_except0 = value_shape_t_except0;
        OP_REQUIRES(
            ctx, element_shape_except0_.IsCompatibleWith(output_shape_except0),
            errors::InvalidArgument(
                "TensorArray was passed element_shape_except0 ",
                element_shape_except0_.DebugString(),
                " but index 0 has (excepting dimension 0) shape: ",
                value_shape_t_except0.DebugString(), " which does not match."));
      } else {
        OP_REQUIRES(ctx, output_shape_except0 == value_shape_t_except0,
                    errors::InvalidArgument(
                        "TensorArray has inconsistent shapes.  Index 0 has "
                        "(excepting dimension 0) shape: ",
                        output_shape_except0.DebugString(), " but index ", i,
                        " has (excepting dimension 0) shape: ",
                        value_shape_t_except0.DebugString()));
        // Store the previous maximum length as the offset for this tensor.
        output_shape.set_dim(
            0, output_shape.dim_size(0) + value_shape_t.dim_size(0));
      }
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    ConstMatrixVector input_tensors_flat;
    input_tensors_flat.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      const Tensor* value_t = &values[i];
      if (value_t->NumElements() > 0) {
        input_tensors_flat.push_back(MakeUnique<ConstMatrix>(
            value_t->shaped<T, 2>({1, value_t->NumElements()})));
      }
    }

    if (output_shape.num_elements() > 0) {
      auto output_flat =
          output_tensor->shaped<T, 2>({1, output_shape.num_elements()});
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(ctx, input_tensors_flat, output_tensor, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      ConcatCPU<T>(ctx->device(), input_tensors_flat, &output_flat);
    }
  }

 private:
  DataType dtype_;
  PartialTensorShape element_shape_except0_;
};

#define REGISTER_CONCAT(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")              \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV2")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<CPUDevice, type>)  \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV3")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_CONCAT);
REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(qint32);

#undef REGISTER_CONCAT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV2")            \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<GPUDevice, type>)  \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV3")            \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("dtype")     \
                              .HostMemory("lengths")             \
                              .HostMemory("handle"),             \
                          TensorArrayConcatOp<GPUDevice, type>)

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("lengths")
                            .HostMemory("handle"),
                        TensorArrayConcatOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("lengths")
                            .HostMemory("handle"),
                        TensorArrayConcatOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("TensorArrayConcatV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("lengths")
                            .HostMemory("handle"),
                        TensorArrayConcatOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// UNPACK and SCATTER *********************************************************

template <typename Device, typename T, bool LEGACY_UNPACK>
class TensorArrayUnpackOrScatterOp : public OpKernel {
 public:
  explicit TensorArrayUnpackOrScatterOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_17(mht_17_v, 1243, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayUnpackOrScatterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_18(mht_18_v, 1248, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));
    TensorShape element_shape(tensor_value->shape());

    OP_REQUIRES(ctx,
                FastBoundsCheck(element_shape.dim_size(0),
                                std::numeric_limits<int32>::max()),
                errors::InvalidArgument("tensor dim0 too large to unpack"));

    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    OP_REQUIRES(ctx, element_shape.dims() > 0,
                errors::InvalidArgument("Input value for unpack must be at "
                                        "least a vector but received shape: ",
                                        element_shape.DebugString()));
    int32_t array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));

    int32_t max_index;
    int32_t num_values;
    std::vector<int32> write_indices;
    if (LEGACY_UNPACK) {
      num_values = element_shape.dim_size(0);
      max_index = num_values - 1;
      write_indices.resize(num_values);
      std::iota(write_indices.begin(), write_indices.end(), 0);
    } else {
      const Tensor* tensor_indices;
      OP_REQUIRES_OK(ctx, ctx->input("indices", &tensor_indices));
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(tensor_indices->shape()),
                  errors::InvalidArgument(
                      "Expected indices to be a vector, but received shape: ",
                      tensor_indices->shape().DebugString()));
      OP_REQUIRES(ctx,
                  tensor_indices->NumElements() == element_shape.dim_size(0),
                  errors::InvalidArgument(
                      "Expected len(indices) == values.shape[0], but saw: ",
                      tensor_indices->NumElements(), " vs. ",
                      element_shape.dim_size(0)));
      const auto indices_t = tensor_indices->vec<int32>();
      num_values = tensor_indices->NumElements();
      max_index = (num_values == 0)
                      ? -1
                      : *std::max_element(indices_t.data(),
                                          indices_t.data() + num_values);
      write_indices.resize(num_values);
      // Copy into write_indices.
      std::copy(indices_t.data(), indices_t.data() + num_values,
                write_indices.begin());
    }

    bool dynamic_size = tensor_array->HasDynamicSize();

    // If dynamic size, we may have to resize the TensorArray to fit.
    if (dynamic_size && array_size < max_index + 1) {
      array_size = static_cast<int32>(max_index + 1);
    }

    if (LEGACY_UNPACK) {
      OP_REQUIRES(
          ctx, element_shape.dim_size(0) == array_size,
          errors::InvalidArgument(
              "Input value must have first dimension equal to the array size (",
              element_shape.dim_size(0), " vs. ", array_size, ")"));
    } else {
      OP_REQUIRES(
          ctx, max_index < array_size,
          errors::InvalidArgument("Max scatter index must be < array size (",
                                  max_index, " vs. ", array_size, ")"));
    }
    element_shape.RemoveDim(0);

    auto tensor_value_t = tensor_value->shaped<T, 3>(
        {1, num_values, element_shape.num_elements()});

    Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, 0, 0};
    Eigen::DSizes<Eigen::DenseIndex, 3> sizes{
        1, 1, static_cast<Eigen::DenseIndex>(element_shape.num_elements())};

    std::vector<Tensor> write_values;
    write_values.reserve(num_values);

    for (int i = 0; i < num_values; ++i) {
      Tensor tensor_value_i;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensor_array->ElemType(),
                                             element_shape, &tensor_value_i));
      auto tensor_value_i_t =
          tensor_value_i.shaped<T, 3>({1, 1, element_shape.num_elements()});
      indices[1] = i;

      if (element_shape.num_elements() > 0) {
        functor::Split<Device, T, 3>()(ctx->eigen_device<Device>(),
                                       tensor_value_i_t, tensor_value_t,
                                       indices, sizes);
      }

      write_values.push_back(tensor_value_i);
    }

    // Record the pack size of the TensorArray.
    if (LEGACY_UNPACK) {
      OP_REQUIRES_OK(ctx, tensor_array->SetMarkedSize(array_size));
    }

    Status s = tensor_array->WriteOrAggregateMany<Device, T>(ctx, write_indices,
                                                             &write_values);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_SCATTER_AND_UNPACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayUnpack").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      TensorArrayUnpackOrScatterOp<CPUDevice, type,                            \
                                   true /* LEGACY_UNPACK */>);                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayScatter").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayUnpackOrScatterOp<CPUDevice, type,                            \
                                   false /* LEGACY_UNPACK */>);                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayScatterV2")                                             \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T"),                                          \
      TensorArrayUnpackOrScatterOp<CPUDevice, type,                            \
                                   false /* LEGACY_UNPACK */>);                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArrayScatterV3")                                             \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T"),                                          \
      TensorArrayUnpackOrScatterOp<CPUDevice, type,                            \
                                   false /* LEGACY_UNPACK */>);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_AND_UNPACK);
#undef REGISTER_SCATTER_AND_UNPACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                      \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("TensorArrayUnpack")                                 \
          .Device(DEVICE_GPU)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("handle"),                                \
      TensorArrayUnpackOrScatterOp<GPUDevice, type,             \
                                   true /* LEGACY_UNPACK */>);  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("TensorArrayScatter")                                \
          .Device(DEVICE_GPU)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("indices")                                \
          .HostMemory("handle"),                                \
      TensorArrayUnpackOrScatterOp<GPUDevice, type,             \
                                   false /* LEGACY_UNPACK */>); \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("TensorArrayScatterV2")                              \
          .Device(DEVICE_GPU)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("indices")                                \
          .HostMemory("handle"),                                \
      TensorArrayUnpackOrScatterOp<GPUDevice, type,             \
                                   false /* LEGACY_UNPACK */>); \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("TensorArrayScatterV3")                              \
          .Device(DEVICE_GPU)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("indices")                                \
          .HostMemory("handle"),                                \
      TensorArrayUnpackOrScatterOp<GPUDevice, type,             \
                                   false /* LEGACY_UNPACK */>);

TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// SPLIT *********************************************************************

template <typename Device, typename T>
class TensorArraySplitOp : public OpKernel {
 public:
  explicit TensorArraySplitOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_19(mht_19_v, 1444, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArraySplitOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_20(mht_20_v, 1449, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));
    const Tensor* tensor_lengths;
    OP_REQUIRES_OK(ctx, ctx->input("lengths", &tensor_lengths));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(tensor_lengths->shape()),
                errors::InvalidArgument(
                    "Expected lengths to be a vector, received shape: ",
                    tensor_lengths->shape().DebugString()));
    OP_REQUIRES(ctx,
                FastBoundsCheck(tensor_lengths->NumElements(),
                                std::numeric_limits<int32>::max()),
                errors::InvalidArgument(
                    "Expected lengths to have < max int32 entries"));

    int32_t num_tensors = static_cast<int32>(tensor_lengths->NumElements());
    auto tensor_lengths_t = tensor_lengths->vec<int64_t>();
    std::vector<int64_t> cumulative_lengths;
    cumulative_lengths.reserve(num_tensors);
    int64_t total_length = 0;
    for (int i = 0; i < num_tensors; ++i) {
      total_length += tensor_lengths_t(i);
      cumulative_lengths.push_back(total_length);
    }

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(tensor_value->shape()),
        errors::InvalidArgument(
            "Expected value to be at least a vector, but received shape: ",
            tensor_value->shape().DebugString()));

    OP_REQUIRES(
        ctx, total_length == tensor_value->shape().dim_size(0),
        errors::InvalidArgument("Expected sum of lengths to be equal to "
                                "values.shape[0], but sum of lengths is ",
                                total_length, " and value's shape is: ",
                                tensor_value->shape().DebugString()));
    int64_t elements_per_row =
        (total_length == 0) ? 0 : (tensor_value->NumElements() / total_length);

    int32_t array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));
    bool dynamic_size = tensor_array->HasDynamicSize();

    std::vector<TensorShape> element_shapes(num_tensors, tensor_value->shape());
    for (int32_t i = 0; i < num_tensors; ++i) {
      element_shapes[i].set_dim(0, tensor_lengths_t(i));
    }

    // If dynamic size, we may have to resize the TensorArray to fit.
    if (dynamic_size && array_size < num_tensors) {
      array_size = num_tensors;
    }

    OP_REQUIRES(
        ctx, array_size == num_tensors,
        errors::InvalidArgument(
            "TensorArray's size is not equal to the size of lengths (",
            array_size, " vs. ", num_tensors, "), and the TensorArray is not ",
            "marked as dynamically resizeable"));

    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));

    auto tensor_value_t =
        tensor_value->shaped<T, 3>({1, total_length, elements_per_row});

    std::vector<Tensor> write_values;
    write_values.reserve(array_size);

    for (int i = 0; i < array_size; ++i) {
      Tensor tensor_value_i;

      int64_t previous_length = (i == 0) ? 0 : cumulative_lengths[i - 1];
      Eigen::DSizes<Eigen::DenseIndex, 3> indices{
          0, static_cast<Eigen::DenseIndex>(previous_length), 0};
      Eigen::DSizes<Eigen::DenseIndex, 3> sizes{
          1, static_cast<Eigen::DenseIndex>(tensor_lengths_t(i)),
          static_cast<Eigen::DenseIndex>(elements_per_row)};

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(tensor_array->ElemType(), element_shapes[i],
                                  &tensor_value_i));

      if (tensor_lengths_t(i) > 0) {
        auto tensor_value_i_t = tensor_value_i.shaped<T, 3>(
            {1, tensor_lengths_t(i), elements_per_row});

        functor::Split<Device, T, 3>()(ctx->eigen_device<Device>(),
                                       tensor_value_i_t, tensor_value_t,
                                       indices, sizes);
      }

      write_values.push_back(tensor_value_i);
    }

    // Record the concat size of the TensorArray.
    OP_REQUIRES_OK(ctx, tensor_array->SetMarkedSize(array_size));

    std::vector<int32> indices(array_size);
    std::iota(indices.begin(), indices.end(), 0);

    Status s = tensor_array->WriteOrAggregateMany<Device, T>(ctx, indices,
                                                             &write_values);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_SPLIT(type)                                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArraySplit").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      TensorArraySplitOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArraySplitV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArraySplitOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TensorArraySplitV3").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArraySplitOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_SPLIT);
#undef REGISTER_SPLIT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("TensorArraySplit")              \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("lengths")            \
                              .HostMemory("handle"),            \
                          TensorArraySplitOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArraySplitV2")            \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("lengths")            \
                              .HostMemory("handle"),            \
                          TensorArraySplitOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("TensorArraySplitV3")            \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("lengths")            \
                              .HostMemory("handle"),            \
                          TensorArraySplitOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// SIZE ***********************************************************************

// Get the size of the TensorArray
class TensorArraySizeOp : public OpKernel {
 public:
  explicit TensorArraySizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_21(mht_21_v, 1618, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArraySizeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_22(mht_22_v, 1623, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    TensorArray* tensor_array;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, tensor_array->Size(&(output->scalar<int32>()())));
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorArraySize").Device(DEVICE_CPU),
                        TensorArraySizeOp);
REGISTER_KERNEL_BUILDER(Name("TensorArraySizeV2").Device(DEVICE_CPU),
                        TensorArraySizeOp);
REGISTER_KERNEL_BUILDER(Name("TensorArraySizeV3").Device(DEVICE_CPU),
                        TensorArraySizeOp);

REGISTER_KERNEL_BUILDER(Name("TensorArraySize")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("size"),
                        TensorArraySizeOp);
REGISTER_KERNEL_BUILDER(Name("TensorArraySizeV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("size"),
                        TensorArraySizeOp);
REGISTER_KERNEL_BUILDER(Name("TensorArraySizeV3")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("size"),
                        TensorArraySizeOp);

// CLOSE
// **********************************************************************

// Delete the TensorArray from its resource container.  This enables
// the user to close and release the resource in the middle of a step/run.
// TODO(ebrevdo): decide whether closing the grad op should happen
// here or on the python side.
class TensorArrayCloseOp : public OpKernel {
 public:
  explicit TensorArrayCloseOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_23(mht_23_v, 1669, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "TensorArrayCloseOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_array_opsDTcc mht_24(mht_24_v, 1674, "", "./tensorflow/core/kernels/tensor_array_ops.cc", "Compute");

    TensorArray* tensor_array;
    OP_REQUIRES_OK(ctx, GetTensorArray(ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    // Instead of deleting this TA from the ResourceManager, we just
    // clear it away and mark it as closed.  The remaining memory
    // consumed store its mutex and handle Tensor.  This will be
    // cleared out at the end of the step anyway, so it's fine to keep
    // it around until the end of the step.  Further calls to the
    // TensorArray will fail because TensorArray checks internally to
    // see if it is closed or not.
    tensor_array->ClearAndMarkClosed();
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayClose").Device(DEVICE_CPU),
                        TensorArrayCloseOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayCloseV2").Device(DEVICE_CPU),
                        TensorArrayCloseOp);
REGISTER_KERNEL_BUILDER(Name("TensorArrayCloseV3").Device(DEVICE_CPU),
                        TensorArrayCloseOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorArrayClose").Device(DEVICE_GPU).HostMemory("handle"),
    TensorArrayCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("TensorArrayCloseV2").Device(DEVICE_GPU).HostMemory("handle"),
    TensorArrayCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("TensorArrayCloseV3").Device(DEVICE_GPU).HostMemory("handle"),
    TensorArrayCloseOp);

}  // namespace tensorflow
