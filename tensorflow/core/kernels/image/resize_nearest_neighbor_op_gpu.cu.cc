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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_gpuDTcuDTcc() {
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
#include "tensorflow/core/kernels/image/resize_nearest_neighbor_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void ResizeNearestNeighborNHWC(
    const int nthreads, const T* __restrict__ bottom_data, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* top_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_gpuDTcuDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_gpu.cu.cc", "ResizeNearestNeighborNHWC");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const T* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_y =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_y) + 0.5f) * height_scale)),
                in_height - 1),
            0);
    const int in_x =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                in_width - 1),
            0);
    const int idx = (in_y * in_width + in_x) * channels + c;
    top_data[index] = ldg(bottom_data_n + idx);
  }
}

template <typename T, bool align_corners>
__global__ void LegacyResizeNearestNeighborNHWC(
    const int nthreads, const T* __restrict__ bottom_data, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* __restrict__ top_data) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const T* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_y =
        min((align_corners) ? static_cast<int>(roundf(out_y * height_scale))
                            : static_cast<int>(floorf(out_y * height_scale)),
            in_height - 1);
    const int in_x =
        min((align_corners) ? static_cast<int>(roundf(out_x * width_scale))
                            : static_cast<int>(floorf(out_x * width_scale)),
            in_width - 1);
    const int idx = (in_y * in_width + in_x) * channels + c;
    top_data[index] = ldg(bottom_data_n + idx);
  }
}

template <typename T>
__global__ void ResizeNearestNeighborBackwardNHWC(
    const int nthreads, const T* __restrict__ top_diff, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* __restrict__ bottom_diff) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_gpuDTcuDTcc mht_1(mht_1_v, 271, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_gpu.cu.cc", "ResizeNearestNeighborBackwardNHWC");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int in_x = n % in_width;
    n /= in_width;
    int in_y = n % in_height;
    n /= in_height;

    T* bottom_diff_n = bottom_diff + n * channels * out_height * out_width;
    const int out_y =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(in_y) + 0.5f) * height_scale)),
                out_height - 1),
            0);
    const int out_x =
        max(min(static_cast<int>(
                    floorf((static_cast<float>(in_x) + 0.5f) * width_scale)),
                out_width - 1),
            0);
    const int idx = (out_y * out_width + out_x) * channels + c;
    GpuAtomicAdd(bottom_diff_n + idx, ldg(top_diff + index));
  }
}

template <typename T, bool align_corners>
__global__ void LegacyResizeNearestNeighborBackwardNHWC(
    const int nthreads, const T* __restrict__ top_diff, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* __restrict__ bottom_diff) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int in_x = n % in_width;
    n /= in_width;
    int in_y = n % in_height;
    n /= in_height;

    T* bottom_diff_n = bottom_diff + n * channels * out_height * out_width;
    const int out_y =
        min((align_corners) ? static_cast<int>(roundf(in_y * height_scale))
                            : static_cast<int>(floorf(in_y * height_scale)),
            out_height - 1);
    const int out_x =
        min((align_corners) ? static_cast<int>(roundf(in_x * width_scale))
                            : static_cast<int>(floorf(in_x * width_scale)),
            out_width - 1);
    const int idx = (out_y * out_width + out_x) * channels + c;
    GpuAtomicAdd(bottom_diff_n + idx, ldg(top_diff + index));
  }
}

}  // namespace

namespace functor {

// Partial specialization of ResizeNearestNeighbor functor for a GPUDevice.
template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighbor<GPUDevice, T, half_pixel_centers, align_corners> {
  bool operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int in_height = input.dimension(1);
    const int in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);

    const int output_size = batch_size * out_height * out_width * channels;
    if (output_size == 0) return true;

    GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d);
    void (*kernel)(const int nthreads, const T* __restrict__ bottom_data,
                   const int in_height, const int in_width, const int channels,
                   const int out_height, const int out_width,
                   const float height_scale, const float width_scale,
                   T* top_data) =
        half_pixel_centers ? ResizeNearestNeighborNHWC<T>
                           : LegacyResizeNearestNeighborNHWC<T, align_corners>;
    TF_CHECK_OK(
        GpuLaunchKernel(kernel, config.block_count, config.thread_per_block, 0,
                        d.stream(), config.virtual_thread_count, input.data(),
                        in_height, in_width, channels, out_height, out_width,
                        height_scale, width_scale, output.data()));
    return d.ok();
  }
};

#define DECLARE_GPU_SPEC(T)                                          \
  template struct ResizeNearestNeighbor<GPUDevice, T, false, false>; \
  template struct ResizeNearestNeighbor<GPUDevice, T, false, true>;  \
  template struct ResizeNearestNeighbor<GPUDevice, T, true, false>;  \
  template struct ResizeNearestNeighbor<GPUDevice, T, true, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

// Partial specialization of ResizeNearestNeighborGrad functor for a GPUDevice.
template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad<GPUDevice, T, half_pixel_centers,
                                 align_corners> {
  bool operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int in_height = input.dimension(1);
    const int in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);

    const int output_size = batch_size * channels * out_height * out_width;

    GpuLaunchConfig output_config = GetGpuLaunchConfig(output_size, d);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, output_config.block_count,
                                output_config.thread_per_block, 0, d.stream(),
                                output_size, output.data()));
    if (!d.ok()) return false;

    const int input_size = batch_size * channels * in_height * in_width;
    if (input_size == 0) return true;

    GpuLaunchConfig input_config = GetGpuLaunchConfig(input_size, d);
    void (*kernel)(const int nthreads, const T* __restrict__ top_diff,
                   const int in_height, const int in_width, const int channels,
                   const int out_height, const int out_width,
                   const float height_scale, const float width_scale,
                   T* __restrict__ bottom_diff) =
        half_pixel_centers
            ? ResizeNearestNeighborBackwardNHWC<T>
            : LegacyResizeNearestNeighborBackwardNHWC<T, align_corners>;
    TF_CHECK_OK(GpuLaunchKernel(
        kernel, input_config.block_count, input_config.thread_per_block, 0,
        d.stream(), input_config.virtual_thread_count, input.data(), in_height,
        in_width, channels, out_height, out_width, height_scale, width_scale,
        output.data()));
    return d.ok();
  }
};

#define DECLARE_GPU_SPEC(T)                                              \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, false, false>; \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, false, true>;  \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, true, false>;  \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, true, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
