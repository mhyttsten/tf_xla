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
class MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTcc() {
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

// See docs in ../ops/array_ops.cc

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/one_hot_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename TI>
class OneHotOp : public OpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/one_hot_op.cc", "OneHotOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/one_hot_op.cc", "Compute");

    const Tensor& indices = ctx->input(0);
    const Tensor& depth = ctx->input(1);
    const Tensor& on_value = ctx->input(2);
    const Tensor& off_value = ctx->input(3);
    const TensorShape& indices_shape = indices.shape();

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    const int axis = (axis_ == -1) ? indices_dims : axis_;

    // The one-hot dimension.
    const int32_t depth_v = depth.scalar<int32>()();
    OP_REQUIRES(
        ctx, depth_v >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
    OP_REQUIRES(
        ctx,
        MultiplyWithoutOverflow(indices_shape.num_elements(), depth_v) >= 0,
        errors::InvalidArgument("OneHot result would have shape ",
                                indices_shape.DebugString(), " + [", depth_v,
                                "], which exceeds 2**63 - 1 elements"));

    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis, depth_v);

    auto on_value_t = on_value.scalar<T>();
    auto off_value_t = off_value.scalar<T>();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() > 0) {
      // prefix_dim_size == # of elements before the axis
      // depth_v == # of elements per axis
      // suffix_dim_size == # of elements after the axis
      int64_t prefix_dim_size = 1;
      for (int i = 0; i < axis; ++i) {
        prefix_dim_size *= indices_shape.dim_size(i);
      }
      int64_t suffix_dim_size = indices_shape.num_elements() / prefix_dim_size;

      // Split indices into matrix of size prefix_dim_size x suffix_dim_size
      auto indices_t =
          indices.shaped<TI, 2>({prefix_dim_size, suffix_dim_size});
      // Split output into 3-Tensor of size:
      //   prefix_dim_size x depth x suffix_dim_size.
      auto output_t =
          output->shaped<T, 3>({prefix_dim_size, depth_v, suffix_dim_size});

      functor::OneHot<Device, T, TI>::Compute(ctx->eigen_device<Device>(),
                                              indices_t, on_value_t,
                                              off_value_t, &output_t);
    }
  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotOp);
};

#define REGISTER_ONE_HOT_INDEX(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("OneHot")                        \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotOp<CPUDevice, type, index_type>);

#define REGISTER_ONE_HOT(type)         \
  REGISTER_ONE_HOT_INDEX(type, uint8); \
  REGISTER_ONE_HOT_INDEX(type, int32); \
  REGISTER_ONE_HOT_INDEX(type, int64_t)

TF_CALL_ALL_TYPES(REGISTER_ONE_HOT);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC_INDEX(T, TI)                                      \
  template <>                                                              \
  void OneHot<GPUDevice, T, TI>::Compute(                                  \
      const GPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices, \
      const typename TTypes<T>::ConstScalar& on_value,                     \
      const typename TTypes<T>::ConstScalar& off_value,                    \
      typename TTypes<T, 3>::Tensor* output);                              \
  extern template struct OneHot<GPUDevice, T, TI>;

#define DECLARE_GPU_SPEC(T)         \
  DECLARE_GPU_SPEC_INDEX(T, uint8); \
  DECLARE_GPU_SPEC_INDEX(T, int32); \
  DECLARE_GPU_SPEC_INDEX(T, int64_t);

TF_CALL_int32(DECLARE_GPU_SPEC);
TF_CALL_int64(DECLARE_GPU_SPEC);
TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC_INDEX
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_ONE_HOT_GPU_INDEX(type, index_type)            \
  REGISTER_KERNEL_BUILDER(Name("OneHot")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotOp<GPUDevice, type, index_type>);

#define REGISTER_ONE_HOT_GPU(type)         \
  REGISTER_ONE_HOT_GPU_INDEX(type, uint8); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int32); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int64_t);

TF_CALL_int32(REGISTER_ONE_HOT_GPU);
TF_CALL_int64(REGISTER_ONE_HOT_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_ONE_HOT_GPU);

#undef REGISTER_ONE_HOT_GPU_INDEX
#undef REGISTER_ONE_HOT_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
