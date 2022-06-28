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
class MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc() {
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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/argmax_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tout, typename ArgFunctor>
class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/argmax_op.cc", "ArgOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/argmax_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));

    const int32_t dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();

    int axis = dim < 0 ? dim + input_dims : dim;

    OP_REQUIRES(context, FastBoundsCheck(axis, input_dims),
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();
    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() == 0) {
      return;
    }

#define HANDLE_DIM(NDIM)                                        \
  case NDIM:                                                    \
    ArgFunctor::Reduce##NDIM(context->eigen_device<Device>(),   \
                             input.tensor<T, NDIM>(), axis,     \
                             output->tensor<Tout, NDIM - 1>()); \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Argmax and Argmin only support up "
                                            "to 7 input dimensions, but got ",
                                            input_dims, ". Inputs shape: ",
                                            input.shape().DebugString()));
    }
  }
#undef HANDLE_DIM

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

template <typename Device, typename T, typename Tout>
class ArgMaxOp
    : public ArgOp<Device, T, Tout, functor::ArgMax<Device, T, Tout> > {
 public:
  explicit ArgMaxOp(OpKernelConstruction* context)
      : ArgOp<Device, T, Tout, functor::ArgMax<Device, T, Tout> >(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc mht_2(mht_2_v, 294, "", "./tensorflow/core/kernels/argmax_op.cc", "ArgMaxOp");
}
};

template <typename Device, typename T, typename Tout>
class ArgMinOp
    : public ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> > {
 public:
  explicit ArgMinOp(OpKernelConstruction* context)
      : ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> >(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSargmax_opDTcc mht_3(mht_3_v, 305, "", "./tensorflow/core/kernels/argmax_op.cc", "ArgMinOp");
}
};

#define REGISTER_ARGMAX(type)                                         \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<CPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .HostMemory("dimension"),               \
                          ArgMinOp<CPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<CPUDevice, type, int32>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .HostMemory("dimension"),               \
                          ArgMinOp<CPUDevice, type, int32>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int16>("output_type")   \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<CPUDevice, type, int16>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<uint16>("output_type")  \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<CPUDevice, type, uint16>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ARGMAX);
TF_CALL_bool(REGISTER_ARGMAX);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T, Tout, Dims)                                       \
  template <>                                                                 \
  void ArgMax<GPUDevice, T, Tout>::Reduce##Dims(                              \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,        \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output); \
  template <>                                                                 \
  void ArgMin<GPUDevice, T, Tout>::Reduce##Dims(                              \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,        \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output);

#define DECLARE_GPU_SPECS(T)       \
  DECLARE_GPU_SPEC(T, int64_t, 1); \
  DECLARE_GPU_SPEC(T, int64_t, 2); \
  DECLARE_GPU_SPEC(T, int64_t, 3); \
  DECLARE_GPU_SPEC(T, int64_t, 4); \
  DECLARE_GPU_SPEC(T, int64_t, 5); \
  DECLARE_GPU_SPEC(T, int64_t, 6); \
  DECLARE_GPU_SPEC(T, int64_t, 7); \
  DECLARE_GPU_SPEC(T, int32, 1);   \
  DECLARE_GPU_SPEC(T, int32, 2);   \
  DECLARE_GPU_SPEC(T, int32, 3);   \
  DECLARE_GPU_SPEC(T, int32, 4);   \
  DECLARE_GPU_SPEC(T, int32, 5);   \
  DECLARE_GPU_SPEC(T, int32, 6);   \
  DECLARE_GPU_SPEC(T, int32, 7);

#define DECLARE_GPU_CLASS(T)                            \
  extern template struct ArgMax<GPUDevice, T, int64_t>; \
  extern template struct ArgMin<GPUDevice, T, int64_t>; \
  extern template struct ArgMax<GPUDevice, T, int32>;   \
  extern template struct ArgMin<GPUDevice, T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_bool(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_CLASS);
TF_CALL_bool(DECLARE_GPU_CLASS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_CLASS

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_ARGMAX_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<GPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int64_t>("output_type") \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMinOp<GPUDevice, type, int64>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMaxOp<GPUDevice, type, int32>);          \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                              \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<int32>("output_type")   \
                              .TypeConstraint<int32>("Tidx")          \
                              .HostMemory("dimension"),               \
                          ArgMinOp<GPUDevice, type, int32>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMAX_GPU);
TF_CALL_bool(REGISTER_ARGMAX_GPU);

#undef REGISTER_ARGMAX_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
