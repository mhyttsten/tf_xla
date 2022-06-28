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
class MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc() {
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

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// There are no native fp16 atomics (we simulate them using 32-bit atomics),
// so fp16 sums are done in fp32 internally. (We don't have a lot of shared
// memory traffic; BiasGradNCHW_SharedAtomics in particular works almost
// entirely on a local variable.)
template <class T>
struct AccumulatorType {
  typedef T type;
};

template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

// Definition of the GPU implementations declared in bias_op.cc.

template <typename T>
__global__ void BiasNHWCKernel(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasNHWCKernel");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

template <typename T>
__global__ void BiasNCHWKernel(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size,
                               int32 image_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasNCHWKernel");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

// Add "bias" to "input", broadcasting it on all dimensions but the bias
// dimension.
template <typename T>
void BiasGPU<T>::compute(const GPUDevice& d, const T* input, const T* bias,
                         T* output, int32 batch, int32 height, int32 width,
                         int depth, int32 channel, TensorFormat data_format) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGPU<T>::compute");

  const int32 bias_size = channel;
  const int32 image_size = height * width * depth;
  const int32 total_count = batch * bias_size * image_size;
  if (total_count == 0) {
    return;
  }
  if (data_format == FORMAT_NHWC) {
    GpuLaunchConfig config =
        GetGpuLaunchConfig(total_count, d, BiasNHWCKernel<T>, 0, 0);
    TF_CHECK_OK(GpuLaunchKernel(BiasNHWCKernel<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                config.virtual_thread_count, input, bias,
                                output, bias_size));
  } else {
    GpuLaunchConfig config =
        GetGpuLaunchConfig(total_count, d, BiasNCHWKernel<T>, 0, 0);
    TF_CHECK_OK(GpuLaunchKernel(BiasNCHWKernel<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                config.virtual_thread_count, input, bias,
                                output, bias_size, image_size));
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNHWC_Naive(int32 nthreads,
                                   const T* __restrict__ output_backprop,
                                   T* __restrict__ bias_backprop,
                                   int32 bias_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_3(mht_3_v, 287, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradNHWC_Naive");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    GpuAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNCHW_Naive(int32 nthreads,
                                   const T* __restrict__ output_backprop,
                                   T* __restrict__ bias_backprop,
                                   int32 bias_size, int32 image_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_4(mht_4_v, 302, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradNCHW_Naive");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    GpuAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

template <typename T>
__global__ void BiasGradNHWC_SharedAtomics(
    int32 nthreads, const T* __restrict__ output_backprop,
    T* __restrict__ bias_backprop, int32 bias_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_5(mht_5_v, 316, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradNHWC_SharedAtomics");

  typedef typename AccumulatorType<T>::type AccT;
  GPU_DYNAMIC_SHARED_MEM_DECL(8, char, s_buf);
  AccT* s_data = reinterpret_cast<AccT*>(s_buf);
  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    s_data[index] = AccT(0);
  }
  __syncthreads();

  for (int32 index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int32 bias_offset = index % bias_size;
    GpuAtomicAdd(s_data + bias_offset, AccT(ldg(output_backprop + index)));
  }
  __syncthreads();

  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    GpuAtomicAdd(bias_backprop + index, T(s_data[index]));
  }
}

template <typename T>
__global__ void BiasGradNCHW_SharedAtomics(
    const T* __restrict__ output_backprop, T* __restrict__ bias_backprop,
    int32 batch, int32 bias_size, int32 image_size, int group_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_6(mht_6_v, 343, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradNCHW_SharedAtomics");

  // Initialize the shared memory.
  typedef typename AccumulatorType<T>::type AccT;
  const int32 kSDataSize = 32;
  __shared__ AccT s_data[kSDataSize];
  for (int32 index = threadIdx.x; index < kSDataSize; index += blockDim.x) {
    s_data[index] = AccT(0);
  }
  __syncthreads();

  // Accumulate all the values within this thread. They all have the same bias
  // index.
  int32 bias_index = blockIdx.x % bias_size;
  int32 group_index = blockIdx.x / bias_size;
  int32 total_count = batch * image_size;
  AccT sum(0);
  for (int32 index = group_index * blockDim.x + threadIdx.x;
       index < total_count; index += blockDim.x * group_size) {
    int32 image_offset = index % image_size;
    int32 batch = index / image_size;
    T val = ldg(output_backprop +
                (batch * bias_size + bias_index) * image_size + image_offset);
    sum += AccT(val);
  }

  // Write the accumulated sum in this thread to the shared memory. Each thread
  // shifts their write location to avoid bank conflict.
  int bias_offset = threadIdx.x % 32;
  GpuAtomicAdd(s_data + bias_offset, sum);
  __syncthreads();

  // Accumulate the results in the shared memory into the first element.
  // No syncthreads is needed since this is only in the same warp.
  int32 thread_index = threadIdx.x;
#if GOOGLE_CUDA
  if (thread_index < 32) {
    AccT data = s_data[thread_index];
    for (int32 delta = warpSize / 2; delta > 0; delta /= 2) {
      data += GpuShuffleXorSync(kCudaWarpAll, data, delta);
    }
    if (thread_index == 0) {
      GpuAtomicAdd(bias_backprop + bias_index, T(data));
    }
  }
#elif TENSORFLOW_USE_ROCM
  if (thread_index < 16) s_data[thread_index] += s_data[thread_index + 16];
  if (thread_index < 8) s_data[thread_index] += s_data[thread_index + 8];
  if (thread_index < 4) s_data[thread_index] += s_data[thread_index + 4];
  if (thread_index < 2) s_data[thread_index] += s_data[thread_index + 2];
  if (thread_index < 1) s_data[thread_index] += s_data[thread_index + 1];

  // The first thread writes out the accumulated result to the global location.
  if (thread_index == 0) {
    GpuAtomicAdd(bias_backprop + bias_index, T(s_data[0]));
  }
#endif
}

template <typename T>
void BiasGradGPU<T>::compute(const GPUDevice& d, const T* output_backprop,
                             T* bias_backprop, int32 batch, int32 height,
                             int32 width, int32 depth, int32 channel,
                             TensorFormat data_format) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_7(mht_7_v, 408, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradGPU<T>::compute");

  const int32 bias_size = channel;
  const int32 image_size = height * width * depth;
  const int32 total_count = batch * bias_size * image_size;
  if (total_count == 0) {
    return;
  }
  static constexpr int32 kWarpSize = 32;
  GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);

  const int max_shared_memory_size = d.sharedMemPerBlock() / 2;
  int32 shared_memory_size = 0;
  if (data_format == FORMAT_NHWC) {
    shared_memory_size = bias_size * sizeof(typename AccumulatorType<T>::type);
  }
  // Check if we have enough shared memory.
  if (shared_memory_size <= max_shared_memory_size) {
    if (data_format == FORMAT_NHWC) {
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNHWC_SharedAtomics<T>,
                                  config.block_count, config.thread_per_block,
                                  shared_memory_size, d.stream(), total_count,
                                  output_backprop, bias_backprop, bias_size));
    } else {
      // Round up the block count to multiple of bias_size.
      int group_size = (config.block_count + bias_size - 1) / bias_size;
      config.block_count = group_size * bias_size;
      if (config.thread_per_block < kWarpSize) {
        config.thread_per_block = kWarpSize;
      }
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNCHW_SharedAtomics<T>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), output_backprop, bias_backprop,
                                  batch, bias_size, image_size, group_size));
    }
  } else {
    // Note that even if we don't have enough shared memory to fit the entire
    // output block, it is possible to process one group of elements at a time.
    // But for now, we simply fall back to the naive implementation.
    if (data_format == FORMAT_NHWC) {
      TF_CHECK_OK(GpuLaunchKernel(
          BiasGradNHWC_Naive<T>, config.block_count, config.thread_per_block, 0,
          d.stream(), total_count, output_backprop, bias_backprop, bias_size));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNCHW_Naive<T>, config.block_count,
                                  config.thread_per_block, 0, d.stream(),
                                  total_count, output_backprop, bias_backprop,
                                  bias_size, image_size));
    }
  }
}

template <typename T>
void BiasGradGPU<T>::DoRowReduction(OpKernelContext* context, T* output,
                                    const T* input, int rows, int cols) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_8(mht_8_v, 464, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradGPU<T>::DoRowReduction");

  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;
  gpuprim::Sum op;
  functor::ReduceImpl<T, gpuprim::Sum, T*, const T*, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, op);
}

template <typename T>
void BiasGradGPU<T>::DoColReduction(OpKernelContext* context, T* output,
                                    const T* input, int rows, int cols) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_op_gpuDTcuDTcc mht_9(mht_9_v, 477, "", "./tensorflow/core/kernels/bias_op_gpu.cu.cc", "BiasGradGPU<T>::DoColReduction");

  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;
  gpuprim::Sum op;
  functor::ReduceImpl<T, gpuprim::Sum, T*, const T*, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kZero, op);
}

#define DEFINE_GPU_SPECS(T)   \
  template struct BiasGPU<T>; \
  template struct BiasGradGPU<T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

// No BiasGrad kernel for int32.
template struct BiasGPU<int32>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
