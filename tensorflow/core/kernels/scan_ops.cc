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
class MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTcc() {
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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/scan_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T, typename Reducer, typename Tidx>
class ScanOp : public OpKernel {
 public:
  explicit ScanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/scan_ops.cc", "ScanOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_opsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/scan_ops.cc", "Compute");

    const Tensor& input = ctx->input(0);
    const Tensor& tensor_axis = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis.shape()),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis.shape().DebugString()));

    const Tidx axis_arg =
        internal::SubtleMustCopy(tensor_axis.scalar<Tidx>()());
    const Tidx axis = (axis_arg < 0) ? input.dims() + axis_arg : axis_arg;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, input.dims()),
                errors::InvalidArgument(
                    "ScanOp: Expected scan axis in the range [", -input.dims(),
                    ", ", input.dims(), "), but got ", axis));

    const TensorShape& output_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Exit early if there's nothing to compute
    if (output_shape.num_elements() == 0) return;

    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

    // Dim reduction.
    int64_t reduced_shape[3] = {1, 1, 1};
    for (Tidx i = 0; i < axis; ++i) {
      reduced_shape[0] *= input.dim_size(i);
    }
    reduced_shape[1] = input.dim_size(axis);
    for (Tidx i = axis + 1; i < input.dims(); ++i) {
      reduced_shape[2] *= input.dim_size(i);
    }

    functor::Scan<Device, Reducer, T>()(d, input.shaped<T, 3>(reduced_shape),
                                        output->shaped<T, 3>(reduced_shape),
                                        reducer, reverse_, exclusive_);
  }

 private:
  bool reverse_;
  bool exclusive_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {

// Forward declarations of GPU functors
#define DECLARE(REDUCER, T)                                                 \
  template <>                                                               \
  void Scan<GPUDevice, REDUCER, T>::operator()(                             \
      const GPUDevice& d, TTypes<T, 3>::ConstTensor in,                     \
      TTypes<T, 3>::Tensor out, const REDUCER& reducer, const bool reverse, \
      const bool exclusive);                                                \
  extern template struct Scan<GPUDevice, REDUCER, T>;

#define DECLARE_FOR_ALL_REDUCERS(T)           \
  DECLARE(Eigen::internal::SumReducer<T>, T); \
  DECLARE(Eigen::internal::ProdReducer<T>, T);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_ALL_REDUCERS);
DECLARE_FOR_ALL_REDUCERS(int32);
DECLARE_FOR_ALL_REDUCERS(int64_t);
#undef DECLARE_FOR_ALL_REDUCERS

#define DECLARE_FOR_LOGSUMEXP_REDUCER(T) DECLARE(LogSumExpReducer<T>, T);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_LOGSUMEXP_REDUCER)
#undef DECLARE_FOR_LOGSUMEXP_REDUCER

#undef DECLARE

}  // namespace functor
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Register Cumsum kernels
#define REGISTER_CPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx"),                                \
      ScanOp<CPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64_t>("Tidx"),                              \
      ScanOp<CPUDevice, type, Eigen::internal::SumReducer<type>, int64>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64_t>("Tidx")                               \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int64>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS)
REGISTER_GPU_KERNELS(int32);
REGISTER_GPU_KERNELS(int64_t);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Register Cumprod kernels
#define REGISTER_CPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx"),                                 \
      ScanOp<CPUDevice, type, Eigen::internal::ProdReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int64_t>("Tidx"),                               \
      ScanOp<CPUDevice, type, Eigen::internal::ProdReducer<type>, int64>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx")                                  \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int64_t>("Tidx")                                \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int64>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS)
REGISTER_GPU_KERNELS(int32);
REGISTER_GPU_KERNELS(int64_t);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_CUMLOGSUMEXP_KERNEL(device, device_type, type, type_idx) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CumulativeLogsumexp")                                         \
          .Device(device)                                                 \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<type_idx>("Tidx")                               \
          .HostMemory("axis"),                                            \
      ScanOp<device_type, type, functor::LogSumExpReducer<type>, type_idx>)

#define REGISTER_CPU_KERNELS(type)                                 \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_CPU, CPUDevice, type, int32) \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_CPU, CPUDevice, type, int64_t)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                 \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_GPU, GPUDevice, type, int32) \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_GPU, GPUDevice, type, int64_t)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_CUMLOGSUMEXP_KERNEL

}  // namespace tensorflow
