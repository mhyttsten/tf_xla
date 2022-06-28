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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/gather_functor_batched.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::DenseIndex IndexType;

template <typename Device, typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/gather_op.cc", "GatherOp");

    // Set batch_dims_ to 0 if the attribute does not exist.
    if (c->HasAttr("batch_dims")) {
      OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
    } else {
      batch_dims_ = 0;
    }
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_opDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/gather_op.cc", "Compute");

    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    int64_t axis = 0;
    bool axis_is_set = false;  // Indicates whether the axis argument was set.
    if (c->num_inputs() == 3) {
      axis_is_set = true;
      const Tensor& axis_tensor = c->input(2);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = axis_tensor.scalar<int32>()();
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64_t>()();
      } else {
        OP_REQUIRES(c, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    }

    int64_t min_params_dim = axis < 0 ? -axis : axis + 1;
    OP_REQUIRES(
        c, params.dims() >= min_params_dim,
        errors::InvalidArgument("Shape must be at least rank ", min_params_dim,
                                " but is rank ", params.dims()));

    if (axis < 0) {
      axis = params.dims() + axis;
    }

    // Modify only a local copy of batch_dims_.
    int32_t batch_dims = batch_dims_;
    if (batch_dims != 0) {
      OP_REQUIRES(c,
                  batch_dims >= -indices.dims() && batch_dims <= indices.dims(),
                  errors::InvalidArgument("Expected batch_dims in the range [",
                                          -indices.dims(), ", ", indices.dims(),
                                          "], but got ", batch_dims));

      if (batch_dims < 0) {
        batch_dims = indices.dims() + batch_dims;
      }

      if (!axis_is_set) axis = batch_dims;

      OP_REQUIRES(c, batch_dims < params.dims(),
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than rank(params) (",
                                          params.dims(), ")."));

      OP_REQUIRES(c, axis >= batch_dims,
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than or equal to ",
                                          "axis (", axis, ")."));
      for (int i = 0; i < batch_dims; ++i) {
        OP_REQUIRES(c, params.dim_size(i) == indices.dim_size(i),
                    errors::InvalidArgument(
                        "params.shape[", i, "]: ", params.dim_size(i),
                        " should be equal to indices.shape[", i,
                        "]: ", indices.dim_size(i)));
      }
    }

    // Check that we have enough index space
    int64_t gather_dim_size = params.dim_size(axis);
    const int64_t N = indices.NumElements();
    OP_REQUIRES(
        c, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64_t batch_size = 1;
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    for (int i = 0; i < batch_dims; ++i) {
      result_shape.AddDim(params.dim_size(i));
      batch_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < axis; ++i) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < indices.dims(); ++i) {
      result_shape.AddDim(indices.dim_size(i));
    }
    for (int i = axis + 1; i < params.dims(); ++i) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N == 0) return;
    if (inner_size == 0) return;

    int64_t bad_i = -1;
    auto indices_flat = indices.flat<Index>();
    if (batch_dims > 0) {
      auto params_flat = params.shaped<T, 4>(
          {batch_size, outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 4>(
          {batch_size, outer_size, N / batch_size, inner_size});

      functor::GatherFunctorBatched<Device, T, Index> functor;
      bad_i = functor(c, params_flat, indices_flat, out_flat);
    } else {
      auto params_flat =
          params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

      functor::GatherFunctor<Device, T, Index> functor;
      bad_i = functor(c, params_flat, indices_flat, out_flat);
    }
    OP_REQUIRES(
        c, bad_i < 0,
        errors::InvalidArgument(
            "indices", SliceDebugString(indices.shape(), bad_i), " = ",
            indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
  }

 private:
  // The number of batch dimensions, as passed in the batch_dims attribute.
  // It must be less than or equal to rank(indices).
  int32 batch_dims_ = 0;
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<dev##Device, type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("axis"),                     \
                          GatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64_t)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);
TF_CALL_quint16(REGISTER_GATHER_CPU);
TF_CALL_qint16(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

}  // namespace tensorflow
