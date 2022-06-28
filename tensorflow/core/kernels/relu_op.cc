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
class MHTracer_DTPStensorflowPScorePSkernelsPSrelu_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrelu_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrelu_opDTcc() {
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

#include "tensorflow/core/kernels/relu_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_RELU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      ReluOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      Relu6Op<CPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<CPUDevice, type>)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

// Register LeakyRelu here for all types except bfloat16
// bfloat16 is in cwise_op_leakyrelu_bf16.cc
#define REGISTER_LEAKYRELU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LeakyRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluOp<CPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_LEAKYRELU_KERNELS)
TF_CALL_half(REGISTER_LEAKYRELU_KERNELS)
    TF_CALL_float(REGISTER_LEAKYRELU_KERNELS)
        TF_CALL_double(REGISTER_LEAKYRELU_KERNELS)
#undef REGISTER_LEAKYRELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Elu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      EluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("EluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      EluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Selu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SeluGradOp<CPUDevice, type>)

    // Elu and Selu only make sense with float or double.
    TF_CALL_FLOAT_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

namespace functor {
#define DECLARE_GPU_NO_MLIR_SPEC(T)                                            \
  template <>                                                                  \
  void Relu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void Elu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct Elu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void Selu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Selu<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_NO_MLIR_SPEC);
}  // namespace functor

#define REGISTER_GPU_NO_MLIR_KERNELS(type)                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ReluOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      EluOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SeluOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_NO_MLIR_KERNELS);
#undef REGISTER_RELU_KERNEL
#endif

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ReluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct ReluGrad<GPUDevice, T>;                               \
                                                                               \
  template <>                                                                  \
  void Relu6<GPUDevice, T>::operator()(                                        \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu6<GPUDevice, T>;                                  \
                                                                               \
  template <>                                                                  \
  void Relu6Grad<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct Relu6Grad<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyRelu<GPUDevice, T>::operator()(LeakyReluArgs args);                \
  extern template struct LeakyRelu<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyReluGrad<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features, T alpha,                       \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct LeakyReluGrad<GPUDevice, T>;                          \
                                                                               \
  template <>                                                                  \
  void EluGrad<GPUDevice, T>::operator()(                                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct EluGrad<GPUDevice, T>;                                \
                                                                               \
  template <>                                                                  \
  void SeluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SeluGrad<GPUDevice, T>;

template <>
void Relu<GPUDevice, qint8>::operator()(
    const GPUDevice& d, typename TTypes<qint8>::ConstTensor features,
    typename TTypes<qint8>::Tensor activations);
extern template struct Relu<GPUDevice, qint8>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      Relu6Op<GPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluGradOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SeluGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

template <typename Device>
class ReluOp<Device, qint8>
    : public UnaryElementWiseOp<qint8, ReluOp<Device, qint8>> {
 public:
  using UnaryElementWiseOp<qint8, ReluOp<Device, qint8>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrelu_opDTcc mht_0(mht_0_v, 386, "", "./tensorflow/core/kernels/relu_op.cc", "Operate");

    auto flat_input = input.flat<qint8>();
    OP_REQUIRES(context, (flat_input.size() % 4) == 0,
                errors::InvalidArgument(
                    "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
                    flat_input.size()));
    functor::Relu<Device, qint8> func;
    func(context->eigen_device<Device>(), flat_input, output->flat<qint8>());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    ReluOp<GPUDevice, qint8>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
