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
class MHTracer_DTPStensorflowPScorePSkernelsPSbetainc_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbetainc_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbetainc_opDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS
// TODO(b/31098934): Figure out why this is necessary here but not in
// any other place, e.g., the cwise lgamma ops.
#define EIGEN_HAS_C99_MATH 1

#include "tensorflow/core/kernels/betainc_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BetaincOp : public OpKernel {
 public:
  explicit BetaincOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbetainc_opDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/betainc_op.cc", "BetaincOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbetainc_opDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/betainc_op.cc", "Compute");

    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& x = ctx->input(2);

    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& x_shape = x.shape();
    if (a_shape.dims() > 0 && b_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == b_shape,
                  errors::InvalidArgument(
                      "Shapes of a and b are inconsistent: ",
                      a_shape.DebugString(), " vs. ", b_shape.DebugString()));
    }
    if (a_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of a and x are inconsistent: ",
                      a_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }
    if (b_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, b_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of b and x are inconsistent: ",
                      b_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }

    TensorShape merged_shape(a_shape);
    if (b_shape.dims() > 0) merged_shape = b_shape;
    if (x_shape.dims() > 0) merged_shape = x_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, merged_shape, &output));

    if (a_shape == b_shape && a_shape == x_shape) {
      functor::Betainc<Device, T, 1> functor;
      functor(ctx->eigen_device<Device>(), a.flat<T>(), b.flat<T>(),
              x.flat<T>(), output->flat<T>());
      return;
    }

    auto merged_shape_vec = BCast::FromShape(merged_shape);
    BCast a_shaper(BCast::FromShape(a_shape), merged_shape_vec);
    BCast b_shaper(BCast::FromShape(b_shape), merged_shape_vec);
    BCast x_shaper(BCast::FromShape(x_shape), merged_shape_vec);

    int ndims = static_cast<int>(a_shaper.x_reshape().size());

    switch (ndims) {
#define CASE(NDIM)                                                        \
  case NDIM: {                                                            \
    functor::Betainc<Device, T, NDIM> functor;                            \
    auto a_value = a.shaped<T, NDIM>(a_shaper.x_reshape());               \
    auto b_value = b.shaped<T, NDIM>(b_shaper.x_reshape());               \
    auto x_value = x.shaped<T, NDIM>(x_shaper.x_reshape());               \
    functor.BCast(ctx->eigen_device<Device>(), a_value,                   \
                  BCast::ToIndexArray<NDIM>(a_shaper.x_bcast()), b_value, \
                  BCast::ToIndexArray<NDIM>(b_shaper.x_bcast()), x_value, \
                  BCast::ToIndexArray<NDIM>(x_shaper.x_bcast()),          \
                  output->shaped<T, NDIM>(a_shaper.y_reshape()));         \
    return;                                                               \
  }

      CASE(1);
      CASE(2);
      default: {
        ctx->SetStatus(errors::InvalidArgument(
            "Broadcasting rank not supported: ", ndims));
        return;
      }
    }
  }
};

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Betainc").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BetaincOp<CPUDevice, type>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(double);
#undef REGISTER_KERNELS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC_NDIM(T, NDIM)                               \
  template <>                                                        \
  void Betainc<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::ConstTensor a,   \
      typename TTypes<T, NDIM>::ConstTensor b,                       \
      typename TTypes<T, NDIM>::ConstTensor x,                       \
      typename TTypes<T, NDIM>::Tensor output);                      \
  template <>                                                        \
  void Betainc<GPUDevice, T, NDIM>::BCast(                           \
      const GPUDevice& d, typename TTypes<T, NDIM>::ConstTensor a,   \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_a, \
      typename TTypes<T, NDIM>::ConstTensor b,                       \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_b, \
      typename TTypes<T, NDIM>::ConstTensor x,                       \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_x, \
      typename TTypes<T, NDIM>::Tensor output);                      \
  extern template struct Betainc<GPUDevice, T, NDIM>;

#define DECLARE_GPU_SPEC(T)   \
  DECLARE_GPU_SPEC_NDIM(T, 1) \
  DECLARE_GPU_SPEC_NDIM(T, 2)

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_NDIM
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Betainc").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BetaincOp<GPUDevice, type>);

REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
