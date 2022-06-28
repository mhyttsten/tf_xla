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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc() {
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

// See docs in ../ops/state_ops.cc.
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/kernels/scatter_nd_util.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64_t num_inputs, int64_t num_indices,
                           int64_t num_updates) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "ValidEmptyOutputShape");

  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

template <typename Device, typename T, typename Index>
class ScatterNdOp : public OpKernel {
 public:
  explicit ScatterNdOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_1(mht_1_v, 236, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "ScatterNdOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({index_t, dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "Compute");

    const Tensor& indices = c->input(0);
    const Tensor& updates = c->input(1);
    const Tensor& shape_input = c->input(2);

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    auto vec = shape_input.flat<Index>();
    TensorShape shape;
    OP_REQUIRES_OK(c,
                   TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape));

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape_input.NumElements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64_t outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(
          c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
          errors::InvalidArgument(
              "Dimensions [0,", outer_dims,
              ") of indices[shape=", indices.shape().DebugString(),
              "] must match dimensions [0,", outer_dims,
              ") of updates[shape=", updates.shape().DebugString(), "]"));
    }

    const int64_t ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(c, updates.shape().dims() - outer_dims == shape.dims() - ix,
                errors::InvalidArgument(
                    "Dimensions [", ix, ",", shape.dims(), ") of input[shape=",
                    shape.DebugString(), "] must match dimensions [",
                    outer_dims, ",", updates.shape().dims(),
                    ") of updates[shape=", updates.shape().DebugString(), "]"));

    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument("Dimensions [", ix, ",", shape.dims(),
                                  ") of input[shape=", shape.DebugString(),
                                  "] must match dimensions [", outer_dims, ",",
                                  updates.shape().dims(), ") of updates[shape=",
                                  updates.shape().DebugString(), "]"));
    }
    OP_REQUIRES(c, shape_input.dims() == 1,
                errors::InvalidArgument("Shape must be a vector"));

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, scatter_nd_op::UpdateOp::ADD>(
               c, indices, updates, shape, &out, true /*allocate*/));
    c->set_output(0, out);
  }
};

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op>
class TensorScatterOp : public OpKernel {
 public:
  explicit TensorScatterOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_3(mht_3_v, 318, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "TensorScatterOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t, dt}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_4(mht_4_v, 327, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "Compute");

    const Tensor& input = c->input(0);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    TensorShape shape = input.shape();

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape.num_elements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64_t outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
                  errors::InvalidArgument(
                      "Outer dimensions of indices and update must match. "
                      "Indices shape: ",
                      indices.shape().DebugString(),
                      ", updates shape:", updates.shape().DebugString()));
    }

    const int64_t ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(
        c, updates.shape().dims() - outer_dims == shape.dims() - ix,
        errors::InvalidArgument("Inner dimensions of output shape must match "
                                "inner dimensions of updates shape. Output: ",
                                shape.DebugString(),
                                " updates: ", updates.shape().DebugString()));
    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument(
              "The inner ", shape.dims() - ix,
              " dimensions of output.shape=", shape.DebugString(),
              " must match the inner ", updates.shape().dims() - outer_dims,
              " dimensions of updates.shape=", updates.shape().DebugString()));
    }

    AllocatorAttributes alloc_attr;
    MemoryType memory_type = DEVICE_MEMORY;
    if (std::is_same<Device, CPUDevice>::value) {
      alloc_attr.set_on_host(true);
      memory_type = HOST_MEMORY;
    } else {
      memory_type = DEVICE_MEMORY;
    }
    std::unique_ptr<Tensor> forwarded_input =
        c->forward_input(0, 0, input.dtype(), shape, memory_type, alloc_attr);

    if (forwarded_input == nullptr) {
      // We were not able to forward the input, so we deep copy the tensor and
      // set the output.
      Tensor* out;
      OP_REQUIRES_OK(c, c->allocate_output(0, input.shape(), &out));

      OP_REQUIRES_OK(c, tensorflow::functor::DoCopy(c->eigen_device<Device>(),
                                                    input, out));
      OP_REQUIRES_OK(c,
                     functor::DoScatterNd<Device, T, Index, op>(
                         c, indices, updates, shape, out, false /*allocate*/));
    } else {
      // Output forwarded, so simply perform the scatter.
      OP_REQUIRES_OK(c, functor::DoScatterNd<Device, T, Index, op>(
                            c, indices, updates, shape, forwarded_input.get(),
                            false /*allocate*/));

      c->set_output(0, *forwarded_input);
    }
  }
};

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op>
class ScatterNdUpdateOp : public OpKernel {
 public:
  explicit ScatterNdUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_5(mht_5_v, 418, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "ScatterNdUpdateOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType dt_ref = DataTypeToEnum<T>::ref();
    const DataType index_t = DataTypeToEnum<Index>::v();
    dtype_ = c->input_type(0);
    // If we are updating a resource, we always use the exclusive lock.
    // For ref types, we lock based on the use_locking parameter
    // Otherwise, we don't mutate the input tensor (we copy-on-write if needed).
    if (c->input_type(0) == DT_RESOURCE) {
      // TODO(apassos): what to validate here?
    } else if (IsRefType(c->input_type(0))) {
      OP_REQUIRES_OK(c, c->MatchSignature({dt_ref, index_t, dt}, {dt_ref}));
      OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
    } else {
      OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t, dt}, {dt}));
      use_exclusive_lock_ = false;
    }
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_6(mht_6_v, 440, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "Compute");

    if (dtype_ == DT_RESOURCE) {
      core::RefCountPtr<Var> v;
      OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
      OP_REQUIRES_OK(c, EnsureSparseVariableAccess<Device, T>(c, v.get()));
      mutex_lock m(*v->mu());
      DoCompute(c);
    } else if (use_exclusive_lock_) {
      // If we're here, it means the input type is a ref.
      DCHECK(IsRefType(c->input_dtype(0)));
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  DataType dtype_;
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_7(mht_7_v, 465, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "DoCompute");

    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    Tensor params;
    TensorShape params_shape;

    if (dtype_ == DT_RESOURCE) {
      core::RefCountPtr<Var> v;
      OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
      Tensor* t = v->tensor();
      params = *t;
      params_shape = params.shape();
    } else if (IsRefType(c->input_dtype(0))) {
      params = c->mutable_input(0, use_exclusive_lock_);
      params_shape = params.shape();
      c->forward_ref_input_to_ref_output(0, 0);
      OP_REQUIRES(c, params.IsInitialized(),
                  errors::FailedPrecondition("Null ref for params"));
    } else {
      Tensor* params_ptr;
      params_shape = c->input(0).shape();
      if (!c->forward_input_to_output_with_shape(0, 0, params_shape,
                                                 &params_ptr)) {
        // We weren't able to forward the input to output, so just
        // allocate a new output tensor and copy the values over.
        OP_REQUIRES_OK(c, c->allocate_output(0, params_shape, &params_ptr));
        params = *params_ptr;
        functor::DenseUpdate<Device, T, ASSIGN> copy;
        const Tensor& input_copy = c->input(0);
        copy(c->eigen_device<Device>(), params.flat<T>(), input_copy.flat<T>());
      } else {
        params = *params_ptr;
      }
    }

    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, op>(
               c, indices, updates, params_shape, &params, false /*allocate*/));
  }
};

#define REGISTER_SCATTER_ND_KERNEL_INDEX(type, index_type, dev, name) \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_##dev)                   \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("shape"),                   \
                          ScatterNdOp<dev##Device, type, index_type>)

#define REGISTER_SCATTER_ND_KERNEL_INDEX_INT32_GPU(index_type, name)  \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_DEFAULT)                 \
                              .TypeConstraint<int32>("T")             \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("indices")                  \
                              .HostMemory("updates")                  \
                              .HostMemory("shape")                    \
                              .HostMemory("output"),                  \
                          ScatterNdOp<CPUDevice, int32, index_type>)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, dev, name, \
                                                op)                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_##dev)                                              \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      ScatterNdUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(index_type, name, \
                                                          op)               \
  REGISTER_KERNEL_BUILDER(Name(name)                                        \
                              .Device(DEVICE_DEFAULT)                       \
                              .TypeConstraint<int32>("T")                   \
                              .TypeConstraint<index_type>("Tindices")       \
                              .HostMemory("ref")                            \
                              .HostMemory("indices")                        \
                              .HostMemory("updates")                        \
                              .HostMemory("output_ref"),                    \
                          ScatterNdUpdateOp<CPUDevice, int32, index_type, op>)

#define REGISTER_SCATTER_ND_NON_ALIASING_UPDATE_KERNEL_INDEX_INT32_GPU( \
    index_type, name, op)                                               \
  REGISTER_KERNEL_BUILDER(Name(name)                                    \
                              .Device(DEVICE_DEFAULT)                   \
                              .TypeConstraint<int32>("T")               \
                              .TypeConstraint<index_type>("Tindices")   \
                              .HostMemory("input")                      \
                              .HostMemory("indices")                    \
                              .HostMemory("updates")                    \
                              .HostMemory("output"),                    \
                          ScatterNdUpdateOp<CPUDevice, int32, index_type, op>)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, \
                                                         dev, name, op)    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(name)                                                           \
          .Device(DEVICE_##dev)                                            \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<index_type>("Tindices")                          \
          .HostMemory("ref"),                                              \
      ScatterNdUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(index_type, \
                                                                   name, op)   \
  REGISTER_KERNEL_BUILDER(Name(name)                                           \
                              .Device(DEVICE_DEFAULT)                          \
                              .TypeConstraint<int32>("T")                      \
                              .TypeConstraint<index_type>("Tindices")          \
                              .HostMemory("ref")                               \
                              .HostMemory("indices")                           \
                              .HostMemory("updates"),                          \
                          ScatterNdUpdateOp<CPUDevice, int32, index_type, op>)

#define REGISTER_SCATTER_ND_KERNEL(type, dev, name)         \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int32, dev, name); \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int64_t, dev, name)

#define REGISTER_SCATTER_ND_KERNEL_INT32_GPU(name)         \
  REGISTER_SCATTER_ND_KERNEL_INDEX_INT32_GPU(int32, name); \
  REGISTER_SCATTER_ND_KERNEL_INDEX_INT32_GPU(int64_t, name)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64_t, dev, name, op)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(name, op)         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(int32, name, op); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(int64_t, name, op)

#define REGISTER_SCATTER_ND_NON_ALIASING_UPDATE_KERNEL_INT32_GPU(name, op)    \
  REGISTER_SCATTER_ND_NON_ALIASING_UPDATE_KERNEL_INDEX_INT32_GPU(int32, name, \
                                                                 op);         \
  REGISTER_SCATTER_ND_NON_ALIASING_UPDATE_KERNEL_INDEX_INT32_GPU(int64_t,     \
                                                                 name, op)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, \
                                                   op);                    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64_t, dev, name, op)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(name, op)         \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(int32, name, op); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU(int64_t, name, op)

#define REGISTER_SCATTER_ND_ADD_SUB(type, dev)                            \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdAdd",            \
                                    scatter_nd_op::UpdateOp::ADD);        \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdNonAliasingAdd", \
                                    scatter_nd_op::UpdateOp::ADD);        \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdSub",            \
                                    scatter_nd_op::UpdateOp::SUB);        \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                             \
      type, dev, "ResourceScatterNdAdd", scatter_nd_op::UpdateOp::ADD);   \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                             \
      type, dev, "ResourceScatterNdSub", scatter_nd_op::UpdateOp::SUB);

#define REGISTER_SCATTER_ND_ADD_SUB_INT32_GPU()                              \
  REGISTER_SCATTER_ND_NON_ALIASING_UPDATE_KERNEL_INT32_GPU(                  \
      "ScatterNdNonAliasingAdd", scatter_nd_op::UpdateOp::ADD);              \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU("ScatterNdAdd",                \
                                              scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU("ScatterNdSub",                \
                                              scatter_nd_op::UpdateOp::SUB); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(                      \
      "ResourceScatterNdAdd", scatter_nd_op::UpdateOp::ADD);                 \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(                      \
      "ResourceScatterNdSub", scatter_nd_op::UpdateOp::SUB);

#define REGISTER_SCATTER_ND(type, dev) \
  REGISTER_SCATTER_ND_KERNEL(type, dev, "ScatterNd");

#define REGISTER_SCATTER_ND_INT32_GPU() \
  REGISTER_SCATTER_ND_KERNEL_INT32_GPU("ScatterNd");

#define REGISTER_SCATTER_ND_UPDATE(type, dev)                         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdUpdate",     \
                                    scatter_nd_op::UpdateOp::ASSIGN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                         \
      type, dev, "ResourceScatterNdUpdate", scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_UPDATE_INT32_GPU()             \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(             \
      "ScatterNdUpdate", scatter_nd_op::UpdateOp::ASSIGN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(    \
      "ResourceScatterNdUpdate", scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MIN_MAX(type, dev)                          \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMax",          \
                                    scatter_nd_op::UpdateOp::MAX);      \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMin",          \
                                    scatter_nd_op::UpdateOp::MIN);      \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdMin", scatter_nd_op::UpdateOp::MIN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdMax", scatter_nd_op::UpdateOp::MAX);

#define REGISTER_SCATTER_ND_MIN_MAX_INT32_GPU()                              \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU("ScatterNdMax",                \
                                              scatter_nd_op::UpdateOp::MAX); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU("ScatterNdMin",                \
                                              scatter_nd_op::UpdateOp::MIN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(                      \
      "ResourceScatterNdMin", scatter_nd_op::UpdateOp::MIN);                 \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU(                      \
      "ResourceScatterNdMax", scatter_nd_op::UpdateOp::MAX);

// Registers CPU kernels.
#define REGISTER_SCATTER_ND_ADD_SUB_CPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, CPU);

#define REGISTER_SCATTER_ND_UPDATE_CPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, CPU);

#define REGISTER_SCATTER_ND_MIN_MAX_CPU(type) \
  REGISTER_SCATTER_ND_MIN_MAX(type, CPU);

#define REGISTER_SCATTER_ND_CPU(type) REGISTER_SCATTER_ND(type, CPU);
#define REGISTER_SCATTER_ND_GPU(type) REGISTER_SCATTER_ND(type, GPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_ADD_SUB_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_CPU);
TF_CALL_tstring(REGISTER_SCATTER_ND_CPU);
TF_CALL_tstring(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_ADD_SUB_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_UPDATE_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_CPU);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_ND_MIN_MAX_CPU);

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, index_type, \
                                                          dev)              \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterUpdate")                       \
                              .Device(DEVICE_##dev)                         \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<index_type>("Tindices"),      \
                          TensorScatterOp<dev##Device, type, index_type,    \
                                          scatter_nd_op::UpdateOp::ASSIGN>)

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_INT32_GPU_INDEX_TYPE(index_type) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterUpdate")                      \
                              .Device(DEVICE_DEFAULT)                      \
                              .TypeConstraint<int32>("T")                  \
                              .TypeConstraint<index_type>("Tindices")      \
                              .HostMemory("tensor")                        \
                              .HostMemory("indices")                       \
                              .HostMemory("updates")                       \
                              .HostMemory("output"),                       \
                          TensorScatterOp<CPUDevice, int32, index_type,    \
                                          scatter_nd_op::UpdateOp::ASSIGN>)

#define REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterAdd")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::ADD>)

#define REGISTER_SCATTER_ND_TENSOR_ADD_INT32_GPU_INDEX_TYPE(index_type) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterAdd")                      \
                              .Device(DEVICE_DEFAULT)                   \
                              .TypeConstraint<int32>("T")               \
                              .TypeConstraint<index_type>("Tindices")   \
                              .HostMemory("tensor")                     \
                              .HostMemory("indices")                    \
                              .HostMemory("updates")                    \
                              .HostMemory("output"),                    \
                          TensorScatterOp<CPUDevice, int32, index_type, \
                                          scatter_nd_op::UpdateOp::ADD>)

#define REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterSub")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::SUB>)

#define REGISTER_SCATTER_ND_TENSOR_SUB_INT32_GPU_INDEX_TYPE(index_type) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterSub")                      \
                              .Device(DEVICE_DEFAULT)                   \
                              .TypeConstraint<int32>("T")               \
                              .TypeConstraint<index_type>("Tindices")   \
                              .HostMemory("tensor")                     \
                              .HostMemory("indices")                    \
                              .HostMemory("updates")                    \
                              .HostMemory("output"),                    \
                          TensorScatterOp<CPUDevice, int32, index_type, \
                                          scatter_nd_op::UpdateOp::SUB>)

#define REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMin")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::MIN>)

#define REGISTER_SCATTER_ND_TENSOR_MIN_INT32_GPU_INDEX_TYPE(index_type) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMin")                      \
                              .Device(DEVICE_DEFAULT)                   \
                              .TypeConstraint<int32>("T")               \
                              .TypeConstraint<index_type>("Tindices")   \
                              .HostMemory("tensor")                     \
                              .HostMemory("indices")                    \
                              .HostMemory("updates")                    \
                              .HostMemory("output"),                    \
                          TensorScatterOp<CPUDevice, int32, index_type, \
                                          scatter_nd_op::UpdateOp::MIN>)

#define REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMax")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::MAX>)

#define REGISTER_SCATTER_ND_TENSOR_MAX_INT32_GPU_INDEX_TYPE(index_type) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMax")                      \
                              .Device(DEVICE_DEFAULT)                   \
                              .TypeConstraint<int32>("T")               \
                              .TypeConstraint<index_type>("Tindices")   \
                              .HostMemory("tensor")                     \
                              .HostMemory("indices")                    \
                              .HostMemory("updates")                    \
                              .HostMemory("output"),                    \
                          TensorScatterOp<CPUDevice, int32, index_type, \
                                          scatter_nd_op::UpdateOp::MAX>)

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int64_t, CPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int64_t, CPU);

#define REGISTER_SCATTER_ND_TENSOR_SUB_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int64_t, CPU);

#define REGISTER_SCATTER_ND_TENSOR_MIN_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int64_t, CPU);

#define REGISTER_SCATTER_ND_TENSOR_MAX_CPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int32, CPU); \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int64_t, CPU);

#define REGISTER_SCATTER_ND_TENSOR_CPU(type)   \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU(type); \
  REGISTER_SCATTER_ND_TENSOR_ADD_CPU(type);    \
  REGISTER_SCATTER_ND_TENSOR_SUB_CPU(type);

// Register TensorScatterUpdate/Add/Sub for all number types.
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_CPU);
// Register min/max operations only for Real number types
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_MIN_CPU);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_MAX_CPU);
// Register only TensorScatterUpdate for string/bool types as well.
TF_CALL_tstring(REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU);
TF_CALL_bool(REGISTER_SCATTER_ND_TENSOR_UPDATE_CPU);

#undef REGISTER_SCATTER_ND_TENSOR_CPU

// Registers GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_SCATTER_ND_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_ND_UPDATE_GPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, GPU);

#define REGISTER_SCATTER_ND_MIN_MAX_GPU(type) \
  REGISTER_SCATTER_ND_MIN_MAX(type, GPU);

#define REGISTER_SCATTER_ND_ALL_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB_GPU(type);  \
  REGISTER_SCATTER_ND_UPDATE_GPU(type);   \
  REGISTER_SCATTER_ND_GPU(type);

#define REGISTER_SCATTER_ND_ALL_INT32_GPU() \
  REGISTER_SCATTER_ND_ADD_SUB_INT32_GPU();  \
  REGISTER_SCATTER_ND_UPDATE_INT32_GPU();   \
  REGISTER_SCATTER_ND_INT32_GPU();

REGISTER_SCATTER_ND_ALL_INT32_GPU();
REGISTER_SCATTER_ND_MIN_MAX_INT32_GPU();

TF_CALL_int64(REGISTER_SCATTER_ND_ALL_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_ALL_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_SCATTER_ND_ALL_GPU);

#undef REGISTER_SCATTER_ND_ALL_GPU

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int64_t, GPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int64_t, GPU);

#define REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int64_t, GPU);

#define REGISTER_SCATTER_ND_TENSOR_MIN_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int64_t, GPU);

#define REGISTER_SCATTER_ND_TENSOR_MAX_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int64_t, GPU);

#define REGISTER_SCATTER_ND_TENSOR_GPU(type)   \
  REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type);    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU(type); \
  REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type);

#define REGISTER_SCATTER_ND_TENSOR_INT32_GPU()                   \
  REGISTER_SCATTER_ND_TENSOR_ADD_INT32_GPU_INDEX_TYPE(int32);    \
  REGISTER_SCATTER_ND_TENSOR_ADD_INT32_GPU_INDEX_TYPE(int64_t);  \
  REGISTER_SCATTER_ND_TENSOR_SUB_INT32_GPU_INDEX_TYPE(int32);    \
  REGISTER_SCATTER_ND_TENSOR_SUB_INT32_GPU_INDEX_TYPE(int64_t);  \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_INT32_GPU_INDEX_TYPE(int32); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_INT32_GPU_INDEX_TYPE(int64_t);

#define REGISTER_SCATTER_ND_TENSOR_GPU_MIN_MAX(type) \
  REGISTER_SCATTER_ND_TENSOR_MIN_GPU(type);          \
  REGISTER_SCATTER_ND_TENSOR_MAX_GPU(type);

#define REGISTER_SCATTER_ND_TENSOR_MIN_MAX_INT32_GPU()          \
  REGISTER_SCATTER_ND_TENSOR_MIN_INT32_GPU_INDEX_TYPE(int32);   \
  REGISTER_SCATTER_ND_TENSOR_MIN_INT32_GPU_INDEX_TYPE(int64_t); \
  REGISTER_SCATTER_ND_TENSOR_MAX_INT32_GPU_INDEX_TYPE(int32);   \
  REGISTER_SCATTER_ND_TENSOR_MAX_INT32_GPU_INDEX_TYPE(int64_t);

REGISTER_SCATTER_ND_TENSOR_INT32_GPU();
REGISTER_SCATTER_ND_TENSOR_MIN_MAX_INT32_GPU();

TF_CALL_int64(REGISTER_SCATTER_ND_TENSOR_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_TENSOR_GPU_MIN_MAX);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_TENSOR_GPU);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ND_TENSOR_GPU_MIN_MAX);
TF_CALL_COMPLEX_TYPES(REGISTER_SCATTER_ND_TENSOR_GPU);

#undef REGISTER_SCATTER_ND_ADD
#undef REGISTER_SCATTER_ND_ADD_SUB
#undef REGISTER_SCATTER_ND_ADD_SUB_CPU
#undef REGISTER_SCATTER_ND_ADD_SUB_GPU
#undef REGISTER_SCATTER_ND_MIN_MAX
#undef REGISTER_SCATTER_ND_MIN_MAX_CPU
#undef REGISTER_SCATTER_ND_MIN_MAX_GPU
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_UPDATE_CPU
#undef REGISTER_SCATTER_ND_UPDATE_GPU
#undef REGISTER_SCATTER_ND_KERNEL
#undef REGISTER_SCATTER_ND_KERNEL_INDEX
#undef REGISTER_SCATTER_ND_TENSOR_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_CPU
#undef REGISTER_SCATTER_ND_TENSOR_GPU
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_ADD_INT32_GPU_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_SUB_INT32_GPU_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MIN_INT32_GPU_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MAX_INT32_GPU_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_INT32_GPU_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_ADD_GPU
#undef REGISTER_SCATTER_ND_TENSOR_SUB_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MIN_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MAX_GPU
#undef REGISTER_SCATTER_ND_TENSOR_GPU
#undef REGISTER_SCATTER_ND_TENSOR_INT32_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MIN_MAX_INT32_GPU
#undef REGISTER_SCATTER_ND_ADD_SUB_INT32_GPU
#undef REGISTER_SCATTER_ND_ALL_INT32_GPU
#undef REGISTER_SCATTER_ND_MIN_MAX_INT32_GPU
#undef REGISTER_SCATTER_ND_INT32_GPU
#undef REGISTER_SCATTER_ND_UPDATE_INT32_GPU
#undef REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INT32_GPU
#undef REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU
#undef REGISTER_SCATTER_ND_UPDATE_KERNEL_INT32_GPU
#undef REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX_INT32_GPU
#undef REGISTER_SCATTER_ND_KERNEL_INT32_GPU
#undef REGISTER_SCATTER_ND_KERNEL_INDEX_INT32_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename Index>
Status PrepareAndValidateInputs(const TensorShape& params_shape,
                                const Tensor& indices, const Tensor& updates,
                                int64_t* slice_dim, Index* num_updates,
                                Index* slice_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_8(mht_8_v, 973, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "PrepareAndValidateInputs");

  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  if (!TensorShapeUtils::IsVectorOrHigher(params_shape)) {
    return errors::InvalidArgument("Output must be at least 1-D, ",
                                   "got shape: ", params_shape.DebugString());
  }

  if (!ValidEmptyOutputShape(params_shape.num_elements(),
                             indices_shape.num_elements(),
                             updates_shape.num_elements())) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output.  indices shape: ",
        indices.shape().DebugString());
  }

  if (updates.dim_size(0) != indices.dim_size(0)) {
    return errors::InvalidArgument(
        "Dimensions [0,1) of indices[shape=", indices_shape.DebugString(),
        "] = ", indices.dim_size(0), " must match dimensions [0,1) of updates[",
        "shape=", updates_shape.DebugString(), "] = ", updates.dim_size(0));
  }
  TF_RETURN_IF_ERROR(ValidateScatterNdUpdateShape(params_shape, indices.shape(),
                                                  updates.shape()));

  // Check that we have enough index space
  const int64_t N_big = indices.NumElements();
  if (N_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("indices has too many elements for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", N_big, " > ",
                                   std::numeric_limits<Index>::max());
  }
  if (params_shape.dim_size(0) > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params_shape[0] too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params_shape.dim_size(0),
                                   " > ", std::numeric_limits<Index>::max());
  }

  // Calculate the number of dimensions in indices
  *slice_dim = (indices_shape.dims() > 1)
                   ? indices_shape.dim_size(indices_shape.dims() - 1)
                   : 1;

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  Index total_nd = params_shape.dims();

  int64_t slice_size_big = 1;
  for (int64_t i = *slice_dim; i < total_nd; ++i) {
    slice_size_big *= params_shape.dim_size(i);
  }

  if (slice_size_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument(
        "slice size is too large for indexing: ", slice_size_big, " > ",
        std::numeric_limits<Index>::max());
  }

  *slice_size = static_cast<Index>(slice_size_big);

  const int64_t safe_slice_dim = (*slice_dim < 1) ? 1 : *slice_dim;
  *num_updates = indices_shape.num_elements() / safe_slice_dim;

  return Status::OK();
}

template <typename Device, typename Index>
class IndexFlattener {
 public:
  inline typename TTypes<Index, 2>::ConstTensor operator()(
      OpKernelContext*, const Tensor& indices) {
    return indices.flat_inner_dims<Index>();
  }
};

namespace {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Copies inputs to the CPU, runs DoScatterNd on the CPU, then copies output
// back to GPU. This is useful because the CPU implementation is deterministic
// and the GPU implementation is not. Tensor inputs to this function must be on
// the GPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp Op>
Status DoScatterNdOnCpu(OpKernelContext* c, const Tensor& indices,
                        const Tensor& updates, const TensorShape& shape,
                        Tensor* out, bool allocate) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_9(mht_9_v, 1066, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "DoScatterNdOnCpu");

  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  auto stream = c->op_device_context()->stream();

  // Copy 'indices' to host.
  Tensor host_indices;
  TF_RETURN_IF_ERROR(c->allocate_temp(indices.dtype(), indices.shape(),
                                      &host_indices, alloc_attr));
  se::DeviceMemoryBase indices_ptr(
      const_cast<Tensor&>(indices).flat<Index>().data(),
      indices.flat<Index>().size() * sizeof(Index));
  stream->ThenMemcpy(host_indices.flat<Index>().data(), indices_ptr,
                     indices.NumElements() * sizeof(Index));
  if (!stream) {
    return errors::Internal("Failed to copy indices to host");
  }

  // Copy 'updates' to host.
  Tensor host_updates;
  TF_RETURN_IF_ERROR(c->allocate_temp(updates.dtype(), updates.shape(),
                                      &host_updates, alloc_attr));
  se::DeviceMemoryBase updates_ptr(
      const_cast<Tensor&>(updates).flat<T>().data(),
      updates.flat<T>().size() * sizeof(T));
  stream->ThenMemcpy(host_updates.flat<T>().data(), updates_ptr,
                     updates.NumElements() * sizeof(T));
  if (!stream) {
    return errors::Internal("Failed to copy updates to host");
  }

  // Create 'out' on host, copying from device if 'allocate' is false.
  Tensor host_out;
  TF_RETURN_IF_ERROR(
      c->allocate_temp(updates.dtype(), shape, &host_out, alloc_attr));
  if (allocate) {
    TF_RETURN_IF_ERROR(c->allocate_temp(DataTypeToEnum<T>::value, shape, out));
    functor::SetZeroFunctor<CPUDevice, T> fill;
    fill(c->eigen_device<CPUDevice>(), host_out.flat<T>());
  } else {
    CHECK_NOTNULL(out);  // Crash OK
    se::DeviceMemoryBase out_ptr(out->flat<T>().data(),
                                 out->flat<T>().size() * sizeof(T));
    stream->ThenMemcpy(host_out.flat<T>().data(), out_ptr,
                       host_out.NumElements() * sizeof(T));
    if (!stream) {
      return errors::Internal("Failed to copy output to host");
    }
  }

  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  TF_RETURN_IF_ERROR(DoScatterNd<CPUDevice, T, Index, Op>(
      c, host_indices, host_updates, shape, &host_out, /*allocate=*/false));

  // Copy 'host_out' to device.
  se::DeviceMemoryBase out_ptr(out->flat<T>().data(),
                               out->flat<T>().size() * sizeof(T));
  stream->ThenMemcpy(&out_ptr, host_out.flat<T>().data(),
                     host_out.NumElements() * sizeof(T));
  if (!stream) {
    return errors::Internal("Failed to copy output to device");
  }
  // Block host, since 'host_out' cannot be destructed until the copy is done.
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return Status::OK();
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp Op>
Status DoScatterNd(OpKernelContext* c, const Tensor& indices,
                   const Tensor& updates, const TensorShape& shape, Tensor* out,
                   bool allocate) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_opDTcc mht_10(mht_10_v, 1145, "", "./tensorflow/core/kernels/scatter_nd_op.cc", "DoScatterNd");

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (std::is_same<Device, GPUDevice>::value &&
      tensorflow::OpDeterminismRequired()) {
    return DoScatterNdOnCpu<T, Index, Op>(c, indices, updates, shape, out,
                                          allocate);
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  int64_t slice_dim;
  Index num_updates;
  Index slice_size;
  TF_RETURN_IF_ERROR(PrepareAndValidateInputs<Index>(
      shape, indices, updates, &slice_dim, &num_updates, &slice_size));

  IndexFlattener<Device, Index> index_flattener;
  auto indices_flat = index_flattener(c, indices);
  auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

  if (allocate) {
    AllocatorAttributes alloc_attr;
    if (std::is_same<Device, CPUDevice>::value) {
      alloc_attr.set_on_host(true);
    }
    TF_RETURN_IF_ERROR(
        c->allocate_temp(DataTypeToEnum<T>::value, shape, out, alloc_attr));
  } else {
    CHECK_NOTNULL(out);
  }

  if (shape.num_elements() == 0) {
    return Status::OK();
  }

  if (allocate) {
    // Brand new tensor, zero it out.
    functor::SetZeroFunctor<Device, T> fill;
    fill(c->eigen_device<Device>(), out->flat<T>());
  }
  auto output_matrix =
      out->shaped<T, 2>({shape.num_elements() / slice_size, slice_size});

  Index bad_i = -1;

  if (shape.num_elements() > 0) {
    switch (slice_dim) {
#define PARAMS_CASE(IXDIM)                                                  \
  case IXDIM: {                                                             \
    typename Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;    \
    for (int i = 0; i < IXDIM; ++i) {                                       \
      output_shape_prefix[i] = shape.dim_size(i);                           \
    }                                                                       \
    functor::ScatterNdFunctor<Device, T, Index, Op, IXDIM> functor;         \
    bad_i =                                                                 \
        functor(c->eigen_device<Device>(), slice_size, output_shape_prefix, \
                output_matrix, indices_flat, updates_flat, output_matrix);  \
  } break
      // TODO(simister): Re-enable this once binary size is under control.
      //      PARAMS_CASE(0);
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
      PARAMS_CASE(6);
      PARAMS_CASE(7);
#undef PARAMS_CASE
      default:
        return errors::InvalidArgument(
            "Only indices.shape[-1] values between 1 and 5 "
            "are currently supported.  Requested rank: ",
            slice_dim);
    }
  }
  if (bad_i >= 0) {
    auto slice_shape = indices.shape();
    slice_shape.RemoveLastDims(1);
    return errors::InvalidArgument(
        "indices", SliceDebugString(slice_shape, bad_i), " = [",
        absl::StrJoin(
            gtl::ArraySlice<Index>(&indices_flat(bad_i, 0), slice_dim), ", "),
        "] does not index into shape ", shape.DebugString());
  }
  return Status::OK();
}
}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, IXDIM)           \
  template <>                                                           \
  Index ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>::operator()(   \
      const GPUDevice& d, const Index slice_size,                       \
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix, \
      typename TTypes<T, 2>::Tensor Tparams,                            \
      typename TTypes<Index, 2>::ConstTensor Tindices,                  \
      typename TTypes<T, 2>::ConstTensor Tupdates,                      \
      typename TTypes<T, 2>::Tensor Toutput);                           \
  extern template struct ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>;

#define DECLARE_GPU_SPECS_INDEX_OP(T, Index, op)     \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 1); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 2); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 3); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 4); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 5); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 6); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 7);

#define DECLARE_GPU_SPECS_INDEX(T, Index)                                \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::SUB)

#define DECLARE_GPU_SPECS_INDEX_MIN_MAX(T, Index)                     \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::MIN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::MAX)

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64_t)

#define DECLARE_GPU_SPECS_MIN_MAX(T)         \
  DECLARE_GPU_SPECS_INDEX_MIN_MAX(T, int32); \
  DECLARE_GPU_SPECS_INDEX_MIN_MAX(T, int64_t)

TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int32(DECLARE_GPU_SPECS_MIN_MAX);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS_MIN_MAX);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS_MIN_MAX
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX_MIN_MAX
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_INDEX_OP

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
