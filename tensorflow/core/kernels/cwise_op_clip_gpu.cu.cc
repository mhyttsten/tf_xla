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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/cwise_op_clip.h"
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

template <typename T>
__global__ void UnaryClipCustomKernel(const int32 size_in,
                                      const T *__restrict__ in0,
                                      const T *__restrict__ in1,
                                      const T *__restrict__ in2,
                                      T *__restrict__ out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/cwise_op_clip_gpu.cu.cc", "UnaryClipCustomKernel");

  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[0] < in0[i] ? in2[0] : in0[i];
    out[i] = value < in1[0] ? in1[0] : value;
  }
}

template <typename T>
__global__ void BinaryRightClipCustomKernel(const int32 size_in,
                                            const T *__restrict__ in0,
                                            const T *__restrict__ in1,
                                            const T *__restrict__ in2,
                                            T *__restrict__ out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/cwise_op_clip_gpu.cu.cc", "BinaryRightClipCustomKernel");

  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[i] < in0[i] ? in2[i] : in0[i];
    out[i] = value < in1[0] ? in1[0] : value;
  }
}

template <typename T>
__global__ void BinaryLeftClipCustomKernel(const int32 size_in,
                                           const T *__restrict__ in0,
                                           const T *__restrict__ in1,
                                           const T *__restrict__ in2,
                                           T *__restrict__ out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_op_clip_gpuDTcuDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/kernels/cwise_op_clip_gpu.cu.cc", "BinaryLeftClipCustomKernel");

  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[0] < in0[i] ? in2[0] : in0[i];
    out[i] = value < in1[i] ? in1[i] : value;
  }
}

namespace functor {

// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename T>
struct UnaryClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        UnaryClipCustomKernel<T>, config.block_count, config.thread_per_block,
        0, d.stream(), in0_flat.size(), in0_flat.data(), in1_flat.data(),
        in2_flat.data(), out_flat.data()));
  }
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename T>
struct BinaryRightClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        BinaryRightClipCustomKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), in0_flat.size(),
        in0_flat.data(), in1_flat.data(), in2_flat.data(), out_flat.data()));
  }
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename T>
struct BinaryLeftClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        BinaryLeftClipCustomKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), in0_flat.size(),
        in0_flat.data(), in1_flat.data(), in2_flat.data(), out_flat.data()));
  }
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename T>
struct TernaryClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    out_flat.device(d) = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
  }
};

#define INSTANTIATE_GPU(T)                         \
  template struct UnaryClipOp<GPUDevice, T>;       \
  template struct BinaryRightClipOp<GPUDevice, T>; \
  template struct BinaryLeftClipOp<GPUDevice, T>;  \
  template struct TernaryClipOp<GPUDevice, T>;
INSTANTIATE_GPU(Eigen::half);
INSTANTIATE_GPU(float);
INSTANTIATE_GPU(double);
INSTANTIATE_GPU(int8);
INSTANTIATE_GPU(int16);
INSTANTIATE_GPU(int32);
INSTANTIATE_GPU(int64_t);
INSTANTIATE_GPU(uint8);
INSTANTIATE_GPU(uint16);
#undef INSTANTIATE_GPU

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
