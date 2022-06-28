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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc() {
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

#include "tensorflow/core/kernels/cwise_op_clip.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Basic coefficient-wise tenary operations.
// This is the case for example of the clip_by_value.
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined above. E.g., functor::clip.
template <typename Device, typename T>
class ClipOp : public OpKernel {
 public:
  explicit ClipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/cwise_op_clip.cc", "ClipOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/kernels/cwise_op_clip.cc", "Compute");

    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, (in0.shape() == in1.shape() ||
                      TensorShapeUtils::IsScalar(in1.shape())) &&
                     (in0.shape() == in2.shape() ||
                      TensorShapeUtils::IsScalar(in2.shape())),
                errors::InvalidArgument(
                    "clip_value_min and clip_value_max must be either of "
                    "the same shape as input, or a scalar. ",
                    "input shape: ", in0.shape().DebugString(),
                    "clip_value_min shape: ", in1.shape().DebugString(),
                    "clip_value_max shape: ", in2.shape().DebugString()));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, in0.shape(), &out));
    if (out->NumElements() == 0) return;  // Nothing to do for empty output

    auto in0_flat = in0.flat<T>();
    auto in1_flat = in1.flat<T>();
    auto in2_flat = in2.flat<T>();
    auto out_flat = out->flat<T>();
    const Device& d = ctx->eigen_device<Device>();

    if (in1.shape() == in2.shape()) {
      if (in0.shape() == in1.shape()) {
        functor::TernaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                            out_flat);
      } else {
        functor::UnaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                          out_flat);
      }
    } else {
      if (in0.shape() == in1.shape()) {
        functor::BinaryLeftClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                               out_flat);
      } else {
        functor::BinaryRightClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                                out_flat);
      }
    }
  }
};

namespace functor {
// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename T>
struct UnaryClipFunc {
  UnaryClipFunc(const T& value_min, const T& value_max)
      : value_min(value_min), value_max(value_max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/kernels/cwise_op_clip.cc", "UnaryClipFunc");
}
  const T operator()(const T& value) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_min;
  T value_max;
};
template <typename T>
struct UnaryClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat = in0_flat.unaryExpr(UnaryClipFunc<T>(in1_flat(0), in2_flat(0)));
  }
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename T>
struct BinaryRightClipFunc {
  explicit BinaryRightClipFunc(const T& value_min) : value_min(value_min) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/kernels/cwise_op_clip.cc", "BinaryRightClipFunc");
}
  const T operator()(const T& value, const T& value_max) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_min;
};
template <typename T>
struct BinaryRightClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat =
        in0_flat.binaryExpr(in2_flat, BinaryRightClipFunc<T>(in1_flat(0)));
  }
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename T>
struct BinaryLeftClipFunc {
  explicit BinaryLeftClipFunc(const T& value_max) : value_max(value_max) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clipDTcc mht_4(mht_4_v, 304, "", "./tensorflow/core/kernels/cwise_op_clip.cc", "BinaryLeftClipFunc");
}
  const T operator()(const T& value, const T& value_min) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_max;
};
template <typename T>
struct BinaryLeftClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat =
        in0_flat.binaryExpr(in1_flat, BinaryLeftClipFunc<T>(in2_flat(0)));
  }
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename T>
struct TernaryClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat.device(d) = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
  }
};

#define INSTANTIATE_CPU(T)                         \
  template struct UnaryClipOp<CPUDevice, T>;       \
  template struct BinaryRightClipOp<CPUDevice, T>; \
  template struct BinaryLeftClipOp<CPUDevice, T>;  \
  template struct TernaryClipOp<CPUDevice, T>;
INSTANTIATE_CPU(Eigen::half);
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(double);
INSTANTIATE_CPU(bfloat16);
INSTANTIATE_CPU(int8);
INSTANTIATE_CPU(int16);
INSTANTIATE_CPU(int32);
INSTANTIATE_CPU(int64_t);
INSTANTIATE_CPU(uint8);
INSTANTIATE_CPU(uint16);
#undef INSTANTIATE_CPU
}  // namespace functor

#define REGISTER_CPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ClipOp<CPUDevice, type>);

REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(bfloat16);
REGISTER_CPU_KERNEL(int8);
REGISTER_CPU_KERNEL(int16);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64_t);
REGISTER_CPU_KERNEL(uint8);
REGISTER_CPU_KERNEL(uint16);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ClipOp<GPUDevice, type>);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(int8);
REGISTER_GPU_KERNEL(int16);
REGISTER_GPU_KERNEL(int64_t);
REGISTER_GPU_KERNEL(uint8);
REGISTER_GPU_KERNEL(uint16);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ClipByValue")
                            .Device(DEVICE_GPU)
                            .HostMemory("t")
                            .HostMemory("clip_value_min")
                            .HostMemory("clip_value_max")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ClipOp<CPUDevice, int32>);

#undef REGISTER_GPU_KERNEL
#endif

}  // namespace tensorflow
