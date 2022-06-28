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
class MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc() {
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

#include "tensorflow/core/kernels/softplus_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftplusOp : public UnaryElementWiseOp<T, SoftplusOp<Device, T>> {
 public:
  explicit SoftplusOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, SoftplusOp<Device, T>>(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/kernels/softplus_op.cc", "SoftplusOp");
}

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/softplus_op.cc", "Operate");

    functor::Softplus<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class SoftplusGradOp
    : public BinaryElementWiseOp<T, SoftplusGradOp<Device, T>> {
 public:
  explicit SoftplusGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, SoftplusGradOp<Device, T>>(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/softplus_op.cc", "SoftplusGradOp");
}

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to SoftplusOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftplus_opDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/kernels/softplus_op.cc", "Operate");

    OperateNoTemplate(context, g, a, output);
  }
};
template <typename Device, typename T>
void SoftplusGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                                  const Tensor& g,
                                                  const Tensor& a,
                                                  Tensor* output) {
  OP_REQUIRES(context, a.IsSameSize(g),
              errors::InvalidArgument("g and a must be the same size"));
  functor::SoftplusGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Softplus").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SoftplusOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftplusGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SoftplusGradOp<CPUDevice, type>);

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_SOFTPLUS_GPU_SPEC(T)                                \
  template <>                                                       \
  void Softplus<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, \
      typename TTypes<T>::Tensor activations);                      \
  extern template struct Softplus<GPUDevice, T>;

#define DECLARE_SOFTPLUS_GRAD_GPU_SPEC(T)                            \
  template <>                                                        \
  void SoftplusGrad<GPUDevice, T>::operator()(                       \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct SoftplusGrad<GPUDevice, T>;

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTPLUS_GPU_SPEC);
#endif

TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTPLUS_GRAD_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_SOFTPLUS_GPU_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Softplus").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftplusOp<GPUDevice, type>);

#define REGISTER_SOFTPLUS_GRAD_GPU_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftplusGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftplusGradOp<GPUDevice, type>);

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTPLUS_GPU_KERNELS);
#endif

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTPLUS_GRAD_GPU_KERNELS);

#undef REGISTER_SOFTPLUS_GPU_KERNELS
#undef REGISTER_SOFTPLUS_GRAD_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
