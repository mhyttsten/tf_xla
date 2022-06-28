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
class MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc() {
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

#include <stdio.h>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/kernels/split_lib_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDims>
void Split<Device, T, NDims>::operator()(
    const Device& d, typename TTypes<T, NDims>::Tensor output,
    typename TTypes<T, NDims>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

template <typename Device, typename T>
void SplitCustom<Device, T>::operator()(
    const Device& d, typename TTypes<T, 2>::Tensor output,
    typename TTypes<T, 2>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

#define DEFINE_GPU_KERNELS(T)                    \
  template struct Split<Eigen::GpuDevice, T, 2>; \
  template struct Split<Eigen::GpuDevice, T, 3>;

TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_bfloat16(DEFINE_GPU_KERNELS);
TF_CALL_uint8(DEFINE_GPU_KERNELS);
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_KERNELS);

#undef DEFINE_GPU_KERNELS
#define DEFINE_GPU_KERNELS(T) template struct SplitCustom<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_COMPLEX_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_bfloat16(DEFINE_GPU_KERNELS);

#undef DEFINE_GPU_KERNELS

}  // namespace functor

namespace {

template <typename T>
__global__ void SplitOpKernel(const T* __restrict__ input,
                              int32 prefix_dim_size, int32 split_dim_size,
                              int32 suffix_dim_size,
                              GpuDeviceArrayStruct<T*> output_ptr_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc mht_0(mht_0_v, 245, "", "./tensorflow/core/kernels/split_lib_gpu.cu.cc", "SplitOpKernel");

  const int32 num_split = output_ptr_data.size;
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);

  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(split_dim_size % num_split == 0);

  int32 size = prefix_dim_size * split_dim_size * suffix_dim_size;
  int32 piece_size = split_dim_size / num_split;

  GPU_1D_KERNEL_LOOP(offset, size) {
    // Calculate the index into input from offset.
    int32 i = offset / (split_dim_size * suffix_dim_size);
    int32 j = (offset % (split_dim_size * suffix_dim_size)) / suffix_dim_size;
    int32 k = offset % suffix_dim_size;

    // Find the output buffer that should be written to.
    T* output_ptr = output_ptrs[j / piece_size];
    // output_ptr is pointing to an array of size
    //  [prefix_dim_size][piece_size][suffix_dim_size].
    //
    // output_ptr[i][j % piece_size][k] = input[offset];
    // Linearize (i, j % piece_size, k) into an offset.
    int32 output_offset = i * piece_size * suffix_dim_size +
                          (j % piece_size) * suffix_dim_size + k;
    *(output_ptr + output_offset) = ldg(input + offset);
  }
}

}  // namespace

// cannot be in anonymous namespace due to extern shared memory
// very similar to the concat kernel except the input/output logic
// is reversed
template <typename T, typename IntType, bool useSmem>
__global__ void split_v_kernel(const T* __restrict__ input_ptr,
                               GpuDeviceArrayStruct<IntType> output_scan,
                               IntType total_rows, IntType total_cols,
                               GpuDeviceArrayStruct<T*> output_ptr_data) {
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);
  IntType* col_scan = GetGpuDeviceArrayOnDevice(&output_scan);

  // do upper_bound on col to find which pointer we should be using
  IntType gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_outputs = output_ptr_data.size;

  // verbose declaration needed due to template
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char, smem);
  IntType* smem_col_scan = reinterpret_cast<IntType*>(smem);

  if (useSmem) {
    IntType lidx = threadIdx.y * blockDim.x + threadIdx.x;
    IntType blockSize = blockDim.x * blockDim.y;

    for (IntType i = lidx; i < output_scan.size; i += blockSize) {
      smem_col_scan[i] = col_scan[i];
    }

    __syncthreads();

    col_scan = smem_col_scan;
  }

  // do an initial binary search and then scan linearly from there
  // works well when there are many small segments and when the
  // segments are much longer
  IntType segment =
      gpu_helper::upper_bound<IntType>(col_scan, num_outputs, gidx) - 1;

  IntType curr_offset = col_scan[segment];
  IntType curr_segment = segment;
  for (; gidx < total_cols; gidx += blockDim.x * gridDim.x) {
    IntType curr_col_offset;
    while ((curr_col_offset = col_scan[curr_segment + 1]) <= gidx) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    IntType local_col = gidx - curr_offset;
    IntType segment_width = curr_col_offset - curr_offset;
    T* output_ptr = output_ptrs[curr_segment];

    IntType gidy = blockIdx.y * blockDim.y + threadIdx.y;
    for (; gidy < total_rows; gidy += blockDim.y * gridDim.y)
      output_ptr[gidy * segment_width + local_col] =
          input_ptr[gidy * total_cols + gidx];
  }
}

// different from the original split implementation due to 2D vs 3D
// dimensions.  This version is likely faster due to less integer math.
template <typename T>
__global__ void SplitVOpKernel_fixed(const T* __restrict__ input,
                                     int32 prefix_dim_size,
                                     int32 suffix_dim_size,
                                     GpuDeviceArrayStruct<T*> output_ptr_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc mht_1(mht_1_v, 344, "", "./tensorflow/core/kernels/split_lib_gpu.cu.cc", "SplitVOpKernel_fixed");

  const int32 num_split = output_ptr_data.size;
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);

  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);

  int32 size = prefix_dim_size * suffix_dim_size;
  int32 piece_size = suffix_dim_size / num_split;

  GPU_1D_KERNEL_LOOP(offset, size) {
    // Calculate the index into input from offset.
    int32 i = offset / suffix_dim_size;
    int32 j = offset % suffix_dim_size;

    // Find the output buffer that should be written to.
    T* output_ptr = output_ptrs[j / piece_size];
    int32 output_offset = i * piece_size + (j % piece_size);
    output_ptr[output_offset] = input[offset];
  }
}

template <typename T>
void SplitOpGPULaunch<T>::Run(const Eigen::GpuDevice& d, const T* input,
                              int32 prefix_dim_size, int32 split_dim_size,
                              int32 suffix_dim_size,
                              const GpuDeviceArrayStruct<T*>& output_ptr_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_lib_gpuDTcuDTcc mht_2(mht_2_v, 373, "", "./tensorflow/core/kernels/split_lib_gpu.cu.cc", "SplitOpGPULaunch<T>::Run");

  GpuLaunchConfig config =
      GetGpuLaunchConfig(prefix_dim_size * split_dim_size * suffix_dim_size, d);

  TF_CHECK_OK(GpuLaunchKernel(SplitOpKernel<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(), input,
                              prefix_dim_size, split_dim_size, suffix_dim_size,
                              output_ptr_data));
}

template <typename T, typename IntType>
void SplitVOpGPULaunch<T, IntType>::Run(
    const Eigen::GpuDevice& gpu_device, bool fixed_size, const T* input_ptr,
    int total_rows, int total_cols,
    const GpuDeviceArrayStruct<IntType>& output_scan,
    const GpuDeviceArrayStruct<T*>& output_ptr_data) {
  if (fixed_size) {
    GpuLaunchConfig config =
        GetGpuLaunchConfig(total_rows * total_cols, gpu_device);

    TF_CHECK_OK(GpuLaunchKernel(SplitVOpKernel_fixed<T>, config.block_count,
                                config.thread_per_block, 0, gpu_device.stream(),
                                input_ptr, total_rows, total_cols,
                                output_ptr_data));
  } else {
    auto config = GetGpu2DLaunchConfig(total_cols, total_rows, gpu_device);
    IntType smem_max = gpu_device.sharedMemPerBlock();
    IntType smem_usage = output_scan.size * sizeof(IntType);
    // performance crossover is less than using maximum available shared
    // memory on most processors possibly due to decreasing occupancy
    // 4096 inputs is a lot, most code will take the smem path
    const int32 kMaxSmemBytesPerformance = 16384;
    if (smem_usage < smem_max && smem_usage < kMaxSmemBytesPerformance) {
      TF_CHECK_OK(GpuLaunchKernel(
          split_v_kernel<T, IntType, true>, config.block_count,
          config.thread_per_block, smem_usage, gpu_device.stream(), input_ptr,
          output_scan, total_rows, total_cols, output_ptr_data));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          split_v_kernel<T, IntType, false>, config.block_count,
          config.thread_per_block, 0, gpu_device.stream(), input_ptr,
          output_scan, total_rows, total_cols, output_ptr_data));
    }
  }
}

#define REGISTER_GPU_KERNEL(T) template struct SplitOpGPULaunch<T>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_bfloat16(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#define REGISTER_GPU_KERNEL(T)                 \
  template struct SplitVOpGPULaunch<T, int32>; \
  template struct SplitVOpGPULaunch<T, int64>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_bfloat16(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
