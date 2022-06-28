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
class MHTracer_DTPStensorflowPScorePSkernelsPScross_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScross_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScross_opDTcc() {
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

#include <algorithm>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/cross_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Type>
class CrossOp : public OpKernel {
 public:
  explicit CrossOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScross_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/cross_op.cc", "CrossOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScross_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/cross_op.cc", "Compute");

    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    OP_REQUIRES(context, in0.shape() == in1.shape(),
                errors::InvalidArgument("Both inputs must be of same shape: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    OP_REQUIRES(context, in0.dims() >= 1,
                errors::InvalidArgument("Input must be at least 1D",
                                        in0.shape().DebugString()));

    // Cross-products only really make sense for three and
    // seven dimensions, and the latter is very obscure. If there is
    // demand, we could perhaps allow 2D vectors where the last
    // element is taken to be zero, but for now, we simply require
    // that all are 3D.
    auto inner_dim = in0.dim_size(in0.dims() - 1);
    OP_REQUIRES(context, inner_dim == 3,
                errors::FailedPrecondition(
                    "Cross-products are only defined for 3-element vectors."));

    // Create the output Tensor with the same dimensions as the input Tensors.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, in0.shape(), &output));

    // Make a canonical tensor, maintaining the last (3-vector) dimension,
    // while flattening all others do give the functor easy to work with data.
    typename TTypes<Type, 2>::ConstTensor in0_data =
        in0.flat_inner_dims<Type>();
    typename TTypes<Type, 2>::ConstTensor in1_data =
        in1.flat_inner_dims<Type>();
    typename TTypes<Type, 2>::Tensor output_data =
        output->flat_inner_dims<Type>();

    functor::Cross<Device, Type>()(context->eigen_device<Device>(), in0_data,
                                   in1_data, output_data);
  }
};

#define REGISTER_CPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cross").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CrossOp<CPUDevice, type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_KERNEL(type)                                 \
  template <>                                                    \
  void Cross<GPUDevice, type>::operator()(                       \
      const GPUDevice& d, TTypes<type, 2>::ConstTensor in0_data, \
      TTypes<type, 2>::ConstTensor in1_data,                     \
      TTypes<type, 2>::Tensor output_data);                      \
  extern template struct Cross<GPUDevice, type>;
TF_CALL_REAL_NUMBER_TYPES(DECLARE_GPU_KERNEL);
#undef DECLARE_GPU_KERNEL
}  // namespace functor
#define REGISTER_GPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cross").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      CrossOp<GPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif

}  // namespace tensorflow
