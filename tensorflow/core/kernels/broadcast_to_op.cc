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
class MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_opDTcc() {
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

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/broadcast_to_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BroadcastToOp : public OpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/broadcast_to_op.cc", "BroadcastToOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_opDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/broadcast_to_op.cc", "Compute");

    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();

    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &output_shape));

    // Handle copy.
    if (output_shape == input_shape) {
      ctx->set_output(0, input_tensor);
      return;
    }

    OP_REQUIRES(ctx, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Rank of input (", input_shape.dims(),
                    ") must be no greater than rank of output shape (",
                    output_shape.dims(), ")."));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    // Handle broadcast from Scalar.
    const Device& device = ctx->eigen_device<Device>();
    if (input_shape.dims() == 0) {
      functor::FillFunctor<Device, T>()(device, output_tensor->flat<T>(),
                                        input_tensor.scalar<T>());
      return;
    }

    // Check whether the broadcast is valid.
    BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_shape.DebugString(), " vs. ",
                    output_shape.DebugString()));
    OP_REQUIRES(ctx, BCast::ToShape(bcast.output_shape()) == output_shape,
                errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                        input_shape, " to tensor of shape ",
                                        output_shape));

    // Handle empty case.
    if (output_shape.num_elements() == 0) {
      return;
    }

    functor::BroadcastTo<Device, T>()(device, ctx, *output_tensor, output_shape,
                                      input_tensor, input_shape, bcast);
  }
};

// As tensor::MakeShape is able to handle both DT_INT32 and DT_INT64,
// no need to have TypeConstraint for `Tidx`
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BroadcastTo").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BroadcastToOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

namespace functor {
#define DECLARE_GPU_TEMPLATE(Type)                               \
  template <>                                                    \
  void BroadcastTo<GPUDevice, Type>::operator()(                 \
      const GPUDevice& d, OpKernelContext* ctx, Tensor& output,  \
      const TensorShape& output_shape, const Tensor& input,      \
      const TensorShape& input_shape, const BCast& bcast) const; \
  extern template struct BroadcastTo<GPUDevice, Type>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_TEMPLATE);
TF_CALL_int64(DECLARE_GPU_TEMPLATE);
#undef DECLARE_GPU_KERNEL
}  // namespace functor

#define REGISTER_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("BroadcastTo")            \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("shape"),      \
                          BroadcastToOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("BroadcastTo")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("shape")
                            .HostMemory("output"),
                        BroadcastToOp<CPUDevice, int32>);
#endif

}  // namespace tensorflow
