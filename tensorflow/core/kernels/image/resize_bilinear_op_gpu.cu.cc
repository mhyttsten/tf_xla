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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc() {
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

/* Copyright 2016-2020 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/image_ops.cc.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/image/resize_bilinear_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void ResizeBilinearKernel_faster(
    const int num_channel_threads, const T* __restrict__ images,
    float height_scale, float width_scale, int batch, int in_height,
    int in_width, int channels, int out_height, int out_width,
    float* __restrict__ output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "ResizeBilinearKernel_faster");

  constexpr int kChannelsPerThread = 16 / sizeof(T);
  for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
       out_idx < out_width * out_height * num_channel_threads;
       out_idx += blockDim.x * gridDim.x) {
    int idx = out_idx;
    const int c_start = idx % num_channel_threads;
    idx /= num_channel_threads;
    const int x = idx % out_width;
    idx /= out_width;
    const int y = idx % out_height;

    const float in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    const int top_y_index = in_y > 0.0 ? floorf(in_y) : 0;
    const int bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    const float in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    const int left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    float top_left_reg[kChannelsPerThread];
    float top_right_reg[kChannelsPerThread];
    float bottom_left_reg[kChannelsPerThread];
    float bottom_right_reg[kChannelsPerThread];
    float out_reg[kChannelsPerThread];
    for (int b = 0; b < batch; b++) {
      for (int c = c_start * kChannelsPerThread; c < channels;
           c += kChannelsPerThread * num_channel_threads) {
        // 16 byte read from global memory and cache them in registers.
        ((float4*)top_left_reg)[0] =
            ((float4*)images)[(((b * in_height + top_y_index) * in_width +
                                left_x_index) *
                                   channels +
                               c) /
                              4];
        ((float4*)top_right_reg)[0] =
            ((float4*)images)[(((b * in_height + top_y_index) * in_width +
                                right_x_index) *
                                   channels +
                               c) /
                              4];
        ((float4*)bottom_left_reg)[0] =
            ((float4*)images)[(((b * in_height + bottom_y_index) * in_width +
                                left_x_index) *
                                   channels +
                               c) /
                              4];
        ((float4*)bottom_right_reg)[0] =
            ((float4*)images)[(((b * in_height + bottom_y_index) * in_width +
                                right_x_index) *
                                   channels +
                               c) /
                              4];
#pragma unroll
        for (int unroll = 0; unroll < kChannelsPerThread; ++unroll) {
          const float top =
              top_left_reg[unroll] +
              (top_right_reg[unroll] - top_left_reg[unroll]) * x_lerp;
          const float bottom =
              bottom_left_reg[unroll] +
              (bottom_right_reg[unroll] - bottom_left_reg[unroll]) * x_lerp;
          out_reg[unroll] = top + (bottom - top) * y_lerp;
        }
        ((float4*)
             output)[(((b * out_height + y) * out_width + x) * channels + c) /
                     4] = ((float4*)out_reg)[0];
      }
    }
  }
}

template <typename T>
__global__ void ResizeBilinearKernel(
    const int32 nthreads, const T* __restrict__ images, float height_scale,
    float width_scale, int batch, int in_height, int in_width, int channels,
    int out_height, int out_width, float* __restrict__ output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_1(mht_1_v, 291, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "ResizeBilinearKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = c + channels * (x + out_width * (y + out_height * b))
    int idx = out_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % out_width;
    idx /= out_width;
    const int y = idx % out_height;
    const int b = idx / out_height;

    const float in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    const int top_y_index = in_y > 0.0 ? floorf(in_y) : 0;
    const int bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    const float in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    const int left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const float top_left(
        images[((b * in_height + top_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float top_right(
        images[((b * in_height + top_y_index) * in_width + right_x_index) *
                   channels +
               c]);
    const float bottom_left(
        images[((b * in_height + bottom_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float bottom_right(
        images[((b * in_height + bottom_y_index) * in_width + right_x_index) *
                   channels +
               c]);

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    output[out_idx] = top + (bottom - top) * y_lerp;
  }
}

template <typename T>
__global__ void ResizeBilinearGradKernel(const int32 nthreads,
                                         const float* __restrict__ input_grad,
                                         float height_scale, float width_scale,
                                         int batch, int original_height,
                                         int original_width, int channels,
                                         int resized_height, int resized_width,
                                         T* __restrict__ output_grad) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_2(mht_2_v, 347, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "ResizeBilinearGradKernel");

  GPU_1D_KERNEL_LOOP(in_idx, nthreads) {
    // in_idx = c + channels * (x + resized_width * (y + resized_height * b))
    int idx = in_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % resized_width;
    idx /= resized_width;
    const int y = idx % resized_height;
    const int b = idx / resized_height;

    const float original_y =
        (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    const int top_y_index = original_y > 0.0 ? floorf(original_y) : 0;
    const int bottom_y_index = (original_y < original_height - 1)
                                   ? ceilf(original_y)
                                   : original_height - 1;
    const float y_lerp = original_y - floorf(original_y);

    const float original_x =
        (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;

    const int left_x_index = original_x > 0.0 ? floorf(original_x) : 0;
    const int right_x_index = (original_x < original_width - 1)
                                  ? ceilf(original_x)
                                  : original_width - 1;
    const float x_lerp = original_x - floorf(original_x);

    const float dtop = (1 - y_lerp) * input_grad[in_idx];
    GpuAtomicAdd(output_grad +
                     ((b * original_height + top_y_index) * original_width +
                      left_x_index) *
                         channels +
                     c,
                 static_cast<T>((1 - x_lerp) * dtop));
    GpuAtomicAdd(output_grad +
                     ((b * original_height + top_y_index) * original_width +
                      right_x_index) *
                         channels +
                     c,
                 static_cast<T>(x_lerp * dtop));

    const float dbottom = y_lerp * input_grad[in_idx];
    GpuAtomicAdd(output_grad +
                     ((b * original_height + bottom_y_index) * original_width +
                      left_x_index) *
                         channels +
                     c,
                 static_cast<T>((1 - x_lerp) * dbottom));
    GpuAtomicAdd(output_grad +
                     ((b * original_height + bottom_y_index) * original_width +
                      right_x_index) *
                         channels +
                     c,
                 static_cast<T>(x_lerp * dbottom));
  }
}

template <typename T>
__global__ void ResizeBilinearDeterministicGradKernel(
    const int32 nthreads, const float* __restrict__ input_grad,
    float height_scale, float inverse_height_scale, float width_scale,
    float inverse_width_scale, int batch, int original_height,
    int original_width, int channels, int resized_height, int resized_width,
    float offset, T* __restrict__ output_grad) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_3(mht_3_v, 414, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "ResizeBilinearDeterministicGradKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = c + channels * (x + original_width * (y + original_height * b))
    int idx = out_idx;
    const int c = idx % channels;
    idx /= channels;
    const int out_x_center = idx % original_width;
    idx /= original_width;
    const int out_y_center = idx % original_height;
    const int b = idx / original_height;

    int in_y_start = max(
        0, __float2int_ru((out_y_center - 1 + offset) * inverse_height_scale -
                          offset));
    const float out_y_start = (in_y_start + offset) * height_scale - offset;
    int in_x_start =
        max(0, __float2int_ru(
                   (out_x_center - 1 + offset) * inverse_width_scale - offset));
    const float out_x_start = (in_x_start + offset) * width_scale - offset;
    T acc = 0;
    // For clarity, prior to C++17, while loops are preferable to for loops here
    float out_y = out_y_start;
    int in_y = in_y_start;
    while (out_y < out_y_center + 1 && in_y < resized_height) {
      float out_x = out_x_start;
      int in_x = in_x_start;
      while (out_x < out_x_center + 1 && in_x < resized_width) {
        int in_idx =
            ((b * resized_height + in_y) * resized_width + in_x) * channels + c;
        // Clamping to zero is necessary because out_x and out_y can be negative
        // due to half-pixel adjustments to out_y_start and out_x_start.
        // Clamping to height/width is necessary when upscaling.
        float out_y_clamped = fmaxf(0, fminf(out_y, original_height - 1));
        float out_x_clamped = fmaxf(0, fminf(out_x, original_width - 1));
        float y_lerp = (1 - fabsf(out_y_clamped - out_y_center));
        float x_lerp = (1 - fabsf(out_x_clamped - out_x_center));
        acc += static_cast<T>(input_grad[in_idx] * y_lerp * x_lerp);
        out_x += width_scale;
        in_x++;
      }
      out_y += height_scale;
      in_y++;
    }
    output_grad[out_idx] = acc;
  }
}

template <typename T>
__global__ void LegacyResizeBilinearKernel(
    const int32 nthreads, const T* __restrict__ images, float height_scale,
    float width_scale, int batch, int in_height, int in_width, int channels,
    int out_height, int out_width, float* __restrict__ output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_4(mht_4_v, 468, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "LegacyResizeBilinearKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = c + channels * (x + out_width * (y + out_height * b))
    int idx = out_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % out_width;
    idx /= out_width;
    const int y = idx % out_height;
    const int b = idx / out_height;

    const float in_y = y * height_scale;
    const int top_y_index = floorf(in_y);
    const int bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - top_y_index;

    const float in_x = x * width_scale;
    const int left_x_index = floorf(in_x);
    const int right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - left_x_index;

    const float top_left(
        images[((b * in_height + top_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float top_right(
        images[((b * in_height + top_y_index) * in_width + right_x_index) *
                   channels +
               c]);
    const float bottom_left(
        images[((b * in_height + bottom_y_index) * in_width + left_x_index) *
                   channels +
               c]);
    const float bottom_right(
        images[((b * in_height + bottom_y_index) * in_width + right_x_index) *
                   channels +
               c]);

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    output[out_idx] = top + (bottom - top) * y_lerp;
  }
}

template <typename T>
__global__ void LegacyResizeBilinearGradKernel(
    const int32 nthreads, const float* __restrict__ input_grad,
    float height_scale, float width_scale, int batch, int original_height,
    int original_width, int channels, int resized_height, int resized_width,
    T* __restrict__ output_grad) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_gpuDTcuDTcc mht_5(mht_5_v, 522, "", "./tensorflow/core/kernels/image/resize_bilinear_op_gpu.cu.cc", "LegacyResizeBilinearGradKernel");

  GPU_1D_KERNEL_LOOP(in_idx, nthreads) {
    // in_idx = c + channels * (x + resized_width * (y + resized_height * b))
    int idx = in_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % resized_width;
    idx /= resized_width;
    const int y = idx % resized_height;
    const int b = idx / resized_height;

    const float original_y = y * height_scale;
    const int top_y_index = floorf(original_y);
    const int bottom_y_index = (original_y < original_height - 1)
                                   ? ceilf(original_y)
                                   : original_height - 1;
    const float y_lerp = original_y - top_y_index;

    const float original_x = x * width_scale;
    const int left_x_index = floorf(original_x);
    const int right_x_index = (original_x < original_width - 1)
                                  ? ceilf(original_x)
                                  : original_width - 1;
    const float x_lerp = original_x - left_x_index;

    const float dtop = (1 - y_lerp) * input_grad[in_idx];
    GpuAtomicAdd(output_grad +
                     ((b * original_height + top_y_index) * original_width +
                      left_x_index) *
                         channels +
                     c,
                 static_cast<T>((1 - x_lerp) * dtop));
    GpuAtomicAdd(output_grad +
                     ((b * original_height + top_y_index) * original_width +
                      right_x_index) *
                         channels +
                     c,
                 static_cast<T>(x_lerp * dtop));

    const float dbottom = y_lerp * input_grad[in_idx];
    GpuAtomicAdd(output_grad +
                     ((b * original_height + bottom_y_index) * original_width +
                      left_x_index) *
                         channels +
                     c,
                 static_cast<T>((1 - x_lerp) * dbottom));
    GpuAtomicAdd(output_grad +
                     ((b * original_height + bottom_y_index) * original_width +
                      right_x_index) *
                         channels +
                     c,
                 static_cast<T>(x_lerp * dbottom));
  }
}

}  // namespace

namespace functor {

// Partial specialization of ResizeBilinear functor for a GPUDevice.
template <typename T>
struct ResizeBilinear<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<float, 4>::Tensor output) {
    const int batch = images.dimension(0);
    const int in_height = images.dimension(1);
    const int in_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);

    const int total_count = batch * out_height * out_width * channels;
    if (total_count == 0) return;

    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
    void (*kernel)(const int num_threads, const T* __restrict__ images,
                   float height_scale, float width_scale, int batch,
                   int in_height, int in_width, int channels, int out_height,
                   int out_width, float* __restrict__ output) =
        LegacyResizeBilinearKernel<T>;

    if (half_pixel_centers) {
      // If centers are not at half-pixel, use the legacy kernel instead.
      kernel = ResizeBilinearKernel<T>;

      // 16 bytes per thread and 8 threads for coalesced 128 bytes global memory
      // access.
      constexpr int max_num_threads_per_pixel = 8;
      constexpr int channels_per_thread = 16 / sizeof(T);
      if (channels % channels_per_thread == 0 &&
          std::is_same<float, T>::value) {
        int num_threads_per_pixel =
            std::min(max_num_threads_per_pixel, channels / channels_per_thread);
        config = GetGpuLaunchConfig(
            out_height * out_width * num_threads_per_pixel, d);
        config.virtual_thread_count = num_threads_per_pixel;
        kernel = ResizeBilinearKernel_faster<T>;
      }
    }

    TF_CHECK_OK(
        GpuLaunchKernel(kernel, config.block_count, config.thread_per_block, 0,
                        d.stream(), config.virtual_thread_count, images.data(),
                        height_scale, width_scale, batch, in_height, in_width,
                        channels, out_height, out_width, output.data()));
  }
};

// Partial specialization of ResizeBilinearGrad functor for a GPUDevice.
template <typename T>
struct ResizeBilinearGrad<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<T, 4>::Tensor output_grad) {
    const int batch = output_grad.dimension(0);
    const int original_height = output_grad.dimension(1);
    const int original_width = output_grad.dimension(2);
    const int channels = output_grad.dimension(3);

    const int resized_height = input_grad.dimension(1);
    const int resized_width = input_grad.dimension(2);

    int total_count;
    GpuLaunchConfig config;

    total_count = batch * original_height * original_width * channels;
    if (total_count == 0) return;
    config = GetGpuLaunchConfig(total_count, d);

    if (OpDeterminismRequired()) {
      // The scale values below should never be zero, enforced by
      // ImageResizerGradientState
      float inverse_height_scale = 1 / height_scale;
      float inverse_width_scale = 1 / width_scale;
      float offset = half_pixel_centers ? 0.5 : 0;
      TF_CHECK_OK(GpuLaunchKernel(
          ResizeBilinearDeterministicGradKernel<T>, config.block_count,
          config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
          input_grad.data(), height_scale, inverse_height_scale, width_scale,
          inverse_width_scale, batch, original_height, original_width, channels,
          resized_height, resized_width, offset, output_grad.data()));
    } else {
      // Initialize output_grad with all zeros.
      TF_CHECK_OK(GpuLaunchKernel(
          SetZero<T>, config.block_count, config.thread_per_block, 0,
          d.stream(), config.virtual_thread_count, output_grad.data()));
      // Accumulate.
      total_count = batch * resized_height * resized_width * channels;
      config = GetGpuLaunchConfig(total_count, d);
      if (half_pixel_centers) {
        TF_CHECK_OK(GpuLaunchKernel(
            ResizeBilinearGradKernel<T>, config.block_count,
            config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
            input_grad.data(), height_scale, width_scale, batch,
            original_height, original_width, channels, resized_height,
            resized_width, output_grad.data()));
      } else {
        TF_CHECK_OK(GpuLaunchKernel(
            LegacyResizeBilinearGradKernel<T>, config.block_count,
            config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
            input_grad.data(), height_scale, width_scale, batch,
            original_height, original_width, channels, resized_height,
            resized_width, output_grad.data()));
      }
    }
  }
};

#define DEFINE_GPU_SPEC(T) template struct ResizeBilinear<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);

#define DEFINE_GRAD_GPU_SPEC(T) \
  template struct ResizeBilinearGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DEFINE_GRAD_GPU_SPEC);

#undef DEFINE_GPU_SPEC
#undef DEFINE_GRAD_GPU_SPEC

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
