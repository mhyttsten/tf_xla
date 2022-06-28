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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/batch_norm_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BatchNormOp : public OpKernel {
 public:
  explicit BatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/batch_norm_op.cc", "BatchNormOp");

    float variance_epsilon;
    OP_REQUIRES_OK(context,
                   context->GetAttr("variance_epsilon", &variance_epsilon));
    variance_epsilon_ = T(variance_epsilon);
    OP_REQUIRES_OK(context, context->GetAttr("scale_after_normalization",
                                             &scale_after_normalization_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/batch_norm_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& mean = context->input(1);
    const Tensor& var = context->input(2);
    const Tensor& beta = context->input(3);
    const Tensor& gamma = context->input(4);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(context, var.dims() == 1,
                errors::InvalidArgument("var must be 1-dimensional",
                                        var.shape().DebugString()));
    OP_REQUIRES(context, beta.dims() == 1,
                errors::InvalidArgument("beta must be 1-dimensional",
                                        beta.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    functor::BatchNorm<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(), mean.vec<T>(),
        var.vec<T>(), beta.vec<T>(), gamma.vec<T>(), variance_epsilon_,
        scale_after_normalization_, output->tensor<T, 4>());
  }

 private:
  T variance_epsilon_;
  bool scale_after_normalization_;
};

template <typename Device, typename T>
class BatchNormGradOp : public OpKernel {
 public:
  explicit BatchNormGradOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/kernels/batch_norm_op.cc", "BatchNormGradOp");

    float variance_epsilon;
    OP_REQUIRES_OK(context,
                   context->GetAttr("variance_epsilon", &variance_epsilon));
    variance_epsilon_ = T(variance_epsilon);
    OP_REQUIRES_OK(context, context->GetAttr("scale_after_normalization",
                                             &scale_after_normalization_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatch_norm_opDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/kernels/batch_norm_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& mean = context->input(1);
    const Tensor& var = context->input(2);
    const Tensor& gamma = context->input(3);
    const Tensor& out_backprop = context->input(4);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(context, var.dims() == 1,
                errors::InvalidArgument("var must be 1-dimensional",
                                        var.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional",
                                        out_backprop.shape().DebugString()));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0, 4}, 0, input.shape(), &dx));
    Tensor* dm = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 1, mean.shape(), &dm));
    Tensor* dv = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {2}, 2, var.shape(), &dv));
    Tensor* db = nullptr;
    if (scale_after_normalization_) {
      OP_REQUIRES_OK(context, context->allocate_output(3, mean.shape(), &db));
    } else {
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {3}, 3, mean.shape(), &db));
    }
    Tensor* dg = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(4, gamma.shape(), &dg));

    // Scratch buffer of [depth] dimension, aka the 4th dimension of input,
    // which is dim_size(3), for calculating various combinations of
    // (var + epsilon).
    Tensor scratch1;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({input.dim_size(3)}), &scratch1));

    // Scratch buffer of [depth] dimension for saving intermediate calculation
    // values.
    Tensor scratch2;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({input.dim_size(3)}), &scratch2));

    functor::BatchNormGrad<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(), mean.vec<T>(),
        var.vec<T>(), gamma.vec<T>(), out_backprop.tensor<T, 4>(),
        variance_epsilon_, scale_after_normalization_, dx->tensor<T, 4>(),
        dm->vec<T>(), dv->vec<T>(), db->vec<T>(), dg->vec<T>(),
        scratch1.vec<T>(), scratch2.vec<T>());
  }

 private:
  T variance_epsilon_;
  bool scale_after_normalization_;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("BatchNormWithGlobalNormalization") \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          BatchNormOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void BatchNorm<GPUDevice, T>::operator()(                                  \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,          \
      typename TTypes<T>::ConstVec mean, typename TTypes<T>::ConstVec var,   \
      typename TTypes<T>::ConstVec beta, typename TTypes<T>::ConstVec gamma, \
      T variance_epsilon, bool scale_after_normalization,                    \
      typename TTypes<T, 4>::Tensor output);                                 \
  extern template struct BatchNorm<GPUDevice, T>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);

TF_CALL_half(DECLARE_GPU_SPECS);
TF_CALL_float(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("BatchNormWithGlobalNormalization") \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          BatchNormOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("BatchNormWithGlobalNormalizationGrad") \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          BatchNormGradOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                \
  template <>                                                              \
  void BatchNormGrad<GPUDevice, T>::operator()(                            \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,        \
      typename TTypes<T>::ConstVec mean, typename TTypes<T>::ConstVec var, \
      typename TTypes<T>::ConstVec gamma,                                  \
      typename TTypes<T, 4>::ConstTensor out_backprop, T variance_epsilon, \
      bool scale_after_normalization, typename TTypes<T, 4>::Tensor dx,    \
      typename TTypes<T>::Vec dm, typename TTypes<T>::Vec dv,              \
      typename TTypes<T>::Vec db, typename TTypes<T>::Vec dg,              \
      typename TTypes<T>::Vec scratch1, typename TTypes<T>::Vec scratch2); \
  extern template struct BatchNormGrad<GPUDevice, T>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);

TF_CALL_half(DECLARE_GPU_SPECS);
TF_CALL_float(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("BatchNormWithGlobalNormalizationGrad") \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          BatchNormGradOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
