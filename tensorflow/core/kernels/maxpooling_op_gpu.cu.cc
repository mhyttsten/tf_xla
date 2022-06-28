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
class MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc() {
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

#include <cfloat>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/maxpooling_op.h"
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {
template <bool propagate_nans, typename dtype>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool IsGreaterThan(dtype a, dtype b) {
  if (propagate_nans) {
    return !(a <= b);
  } else {
    return a > b;
  }
}

// This is Yangqing's custom kernel for the maxpooling operation. There are
// three functions: MaxPoolForwardNCHW and MaxPoolForwardNHWC are the two
// forward functions, dealing with the forward case. MaxPoolBackward is the
// backward function that deals with the backward case for both storage orders.
// The parameters to the kernels in the forward function is as follows:
//     nthreads: the number of threads, which is equal to the output size.
//     bottom_data: the bottom data of N*H*W*C (or N*C*H*W) items.
//     height, width, pooled_height, pooled_width: the input and output sizes.
//     kernel_h, kernel_w: the kernel sizes.
//     stride_h, stride_w: the strides.
//     pad_t, pad_l: the padding values on the top and left side.
//     top_data: the maxpool output.
//     mask: the output mask of the same size as top_data. It is stored in
//         int form, keeping track of the flattened index of the input item that
//         produces the max output. If a nullptr is passed in for mask, no mask
//         will be produced.
//     include_batch_in_index: whether to include batch dimension in flattened
//         index of `argmax`.
//
// To call the forward and backward functions, use e.g.:
// const int kThreadsPerBlock = 1024
// const int output_size = batch * channels * pooled_height * pooled_width;
// MaxPoolForwardNCHW<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
//                      kThreadsPerBlock, 0, cuda_stream>>>(...);
template <bool propagate_nans, typename dtype>
__global__ void MaxPoolForwardNCHW(
    const int nthreads, const dtype* __restrict__ bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, dtype* __restrict__ top_data, int64* __restrict__ mask,
    const bool include_batch_in_index) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dtype maxval = Eigen::NumTraits<dtype>::lowest();
    int maxidx = -1;
    const int offset = n * channels * height * width;
    const dtype* bottom_data_n = bottom_data + offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        if (IsGreaterThan<propagate_nans>(bottom_data_n[idx], maxval)) {
          maxidx = include_batch_in_index ? idx + offset : idx;
          maxval = bottom_data_n[idx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask != nullptr) {
      mask[index] = maxidx;
    }
  }
}

#if GOOGLE_CUDA
// The parameters for MaxPoolForwardNoMaskKernel_NCHW_VECT_C are the same as for
// MaxPoolForwardNCHW above, except that mask is not supported, and each
// element of the input and output contains 4 adjacent channel values for
// the same X, y coordinate.
// (so channels = outer_channels, output_size = real output size / 4).
__global__ void MaxPoolForwardNoMaskKernel_NCHW_VECT_C(
    const int nthreads, const int32* __restrict__ bottom_data, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    int32* __restrict__ top_data) {
  // TODO(pauldonnelly): Implement a better optimized version of this kernel.
  const int32 kMinINT8X4 = 0x80808080;
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int32 maxval = kMinINT8X4;
    const int32* bottom_data_n = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (c * height + h) * width + w;
        maxval = __vmaxs4(maxval, bottom_data_n[idx]);
      }
    }
    top_data[index] = maxval;
  }
}
#endif  // GOOGLE_CUDA

template <bool propagate_nans, typename dtype>
__global__ void MaxPoolForwardNHWC(
    const int nthreads, const dtype* __restrict__ bottom_data, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    dtype* __restrict__ top_data, int64* __restrict__ mask,
    const bool include_batch_in_index) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dtype maxval = Eigen::NumTraits<dtype>::lowest();
    int maxidx = -1;
    const int offset = n * height * width * channels;
    const dtype* bottom_data_n = bottom_data + offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (IsGreaterThan<propagate_nans>(bottom_data_n[idx], maxval)) {
          maxidx = include_batch_in_index ? idx + offset : idx;
          maxval = bottom_data_n[idx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask != nullptr) {
      mask[index] = maxidx;
    }
  }
}

// The parameters to the kernels in the backward function is as follows:
//     nthreads: the number of threads, which is equal to the output size.
//     top_diff: the gradient of the output data, of size N*Hout*Wout*C (or
//        N*C*Hout*Wout). As we have stored the flattened index of the input
//        entries, the backward function is agnostic of the input storage order.
//     mask: the output mask of the same size as top_data. It is stored in
//         int form, keeping track of the flattened index of the input item that
//         produces the max output.
//     top_offset: the pre-computed per-image offset of the maxpool output. This
//         is equal to Hout*Wout*C. We choose to pre-compute this so we do not
//         need to compute it every time inside the kernel.
//     bottom_offset: the pre-computed per-image offset of the maxpool input.
//         This is equal to H*W*C.
//     bottom_diff: the gradient with respect to the input.
//     include_batch_in_index: whether to include batch dimension in flattened
//         index of `argmax`.
// This function relies on GpuAtomicAdd to avoid race conditions. Also, before
// the kernel is run, you will need to make sure that bottom_diff is filled with
// zero first.
template <typename dtype>
__global__ void MaxPoolBackward(const int nthreads,
                                const dtype* __restrict__ top_diff,
                                const int64* __restrict__ mask,
                                const int top_offset, const int bottom_offset,
                                dtype* __restrict__ bottom_diff,
                                const bool include_batch_in_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc mht_0(mht_0_v, 377, "", "./tensorflow/core/kernels/maxpooling_op_gpu.cu.cc", "MaxPoolBackward");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    const int offset =
        include_batch_in_index ? 0 : (index / top_offset) * bottom_offset;
    GpuAtomicAdd(bottom_diff + offset + mask[index], top_diff[index]);
  }
}

// The parameters to the kernels in the gradient function is as follows:
//     nthreads: the number of threads, which is equal to the output size. The
//         gradient of the MaxPooling gradient w.r.t. the output data has a
//         dimensions of N*C*Hout*Wout
//     bottom_data: the bottom data of N*H*W*C (or N*C*H*W) items.
//     output_data: the output data of N*Hout*Wout*C (or N*C*Hout*Wout) items.
//     height, width, pooled_height, pooled_width: the input and output sizes.
//     kernel_h, kernel_w: the kernel sizes.
//     stride_h, stride_w: the strides.
//     pad_t, pad_l: the padding values on the top and left side.
//     top_diff: the gradient of the gradient of the output data w.r.t. the
//         input data, of size N*H*W*C (or N*C*H*W).
//     bottom_diff: the gradient of the gradient w.r.t. output.
template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNCHW(
    const int nthreads, const dtype* __restrict__ bottom_data,
    const dtype* __restrict__ output_data, const int pooled_height,
    const int pooled_width, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l,
    const dtype* __restrict__ top_diff, dtype* __restrict__ bottom_diff) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc mht_1(mht_1_v, 408, "", "./tensorflow/core/kernels/maxpooling_op_gpu.cu.cc", "MaxPoolGradBackwardNoMaskNCHW");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n = bottom_data + n * channels * height * width;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int h = hstart; h < hend && !should_stop; ++h) {
      for (int w = wstart; w < wend && !should_stop; ++w) {
        int idx = c * height * width + h * width + w;
        if (output_data[index] == bottom_data_n[idx]) {
          maxidx = idx;
          should_stop = true;
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] = top_diff[n * channels * height * width + maxidx];
    }
  }
}

template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNHWC(
    const int nthreads, const dtype* __restrict__ bottom_data,
    const dtype* __restrict__ output_data, const int pooled_height,
    const int pooled_width, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l,
    const dtype* __restrict__ top_diff, dtype* __restrict__ bottom_diff) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc mht_2(mht_2_v, 452, "", "./tensorflow/core/kernels/maxpooling_op_gpu.cu.cc", "MaxPoolGradBackwardNoMaskNHWC");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n = bottom_data + n * height * width * channels;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int h = hstart; h < hend && !should_stop; ++h) {
      for (int w = wstart; w < wend && !should_stop; ++w) {
        int idx = (h * width + w) * channels + c;
        if (output_data[index] == bottom_data_n[idx]) {
          maxidx = idx;
          should_stop = true;
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] = top_diff[n * height * width * channels + maxidx];
    }
  }
}

// The parameters to the kernels in the gradient function is as follows:
//     nthreads: the number of threads, which is equal to the output size. The
//         gradient of the MaxPooling gradient w.r.t. the output data has a
//         dimensions of N*C*Hout*Wout
//     top_diff: the gradient of the gradient of the output data w.r.t. the
//         input data, of size N*H*W*C (or N*C*H*W). As we have stored the
//         flattened index of the input entries, the backward function is
//         agnostic of the input storage order.
//     mask: the output mask of the same size as top_data. It is stored in
//         int form, keeping track of the flattened index of the input item that
//         produces the max output.
//     top_offset: the pre-computed per-image offset of the maxpool input
//         gradient. This is equal to H*W*C. We choose to pre-compute this so we
//         do not  need to compute it every time inside the kernel.
//     bottom_offset: the pre-computed per-image offset of the maxpool output.
//         This is equal to Hout*Wout*C.
//     bottom_diff: the gradient of the gradient w.r.t. output.
//     include_batch_in_index: whether to include batch dimension in flattened
//         index of `argmax`.
template <typename dtype>
__global__ void MaxPoolGradBackward(const int nthreads,
                                    const dtype* __restrict__ top_diff,
                                    const int64* __restrict__ mask,
                                    const int top_offset,
                                    const int bottom_offset,
                                    dtype* __restrict__ bottom_diff,
                                    const bool include_batch_in_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmaxpooling_op_gpuDTcuDTcc mht_3(mht_3_v, 516, "", "./tensorflow/core/kernels/maxpooling_op_gpu.cu.cc", "MaxPoolGradBackward");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    const int offset =
        include_batch_in_index ? 0 : (index / bottom_offset) * top_offset;
    bottom_diff[index] = top_diff[offset + mask[index]];
  }
}

#undef GPU_1D_KERNEL_LOOP
}  // namespace

namespace functor {

#if GOOGLE_CUDA
// Note: channels is the outer channels (dim 1) which has already been
// divided by 4.
bool MaxPoolForwardNoMask_NCHW_VECT_C::operator()(
    const int32* bottom_data, const int batch, const int height,
    const int width, int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    int32* top_data, const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * pooled_height * pooled_width;
  if (output_size == 0) return true;
  TF_CHECK_OK(GpuLaunchKernel(
      MaxPoolForwardNoMaskKernel_NCHW_VECT_C,
      (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock,
      0, d.stream(), output_size, bottom_data, height, width, channels,
      pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w,
      pad_t, pad_l, top_data));
  return d.ok();
}
#endif  // GOOGLE_CUDA

template <typename T>
bool MaxPoolForwardWithOptionalArgmax<T>::operator()(
    const T* bottom_data, const int batch, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data,
    int64* mask, const Eigen::GpuDevice& d, bool propagate_nans,
    const bool include_batch_in_index) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * pooled_height * pooled_width;
  if (output_size == 0) return true;
  if (propagate_nans) {
    TF_CHECK_OK(
        GpuLaunchKernel(MaxPoolForwardNHWC<true, T>,
                        (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        kThreadsPerBlock, 0, d.stream(), output_size,
                        bottom_data, height, width, channels, pooled_height,
                        pooled_width, kernel_h, kernel_w, stride_h, stride_w,
                        pad_t, pad_l, top_data, mask, include_batch_in_index));
  } else {
    TF_CHECK_OK(
        GpuLaunchKernel(MaxPoolForwardNHWC<false, T>,
                        (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        kThreadsPerBlock, 0, d.stream(), output_size,
                        bottom_data, height, width, channels, pooled_height,
                        pooled_width, kernel_h, kernel_w, stride_h, stride_w,
                        pad_t, pad_l, top_data, mask, include_batch_in_index));
  }
  return d.ok();
}

template <typename T>
bool MaxPoolBackwardWithArgmax<T>::operator()(
    const int output_size, const int input_size, const T* top_diff,
    const int64* mask, const int top_offset, const int bottom_offset,
    T* bottom_diff, const Eigen::GpuDevice& d,
    const bool include_batch_in_index) {
  const int kThreadsPerBlock = 1024;
  if (input_size == 0) return true;
  TF_CHECK_OK(GpuLaunchKernel(
      SetZero<T>, (input_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
      kThreadsPerBlock, 0, d.stream(), input_size, bottom_diff));
  TF_CHECK_OK(GpuLaunchKernel(
      MaxPoolBackward<T>,
      (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock,
      0, d.stream(), output_size, top_diff, mask, top_offset, bottom_offset,
      bottom_diff, include_batch_in_index));
  return d.ok();
}

template <typename T>
bool MaxPoolGradBackwardNoMask<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const T* output_data,
    const int batch, const int pooled_height, const int pooled_width,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, const T* top_diff, T* bottom_diff,
    const Eigen::GpuDevice& d) {
  const int num_kernels = batch * channels * pooled_height * pooled_width;
  if (num_kernels == 0) return true;
  GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, d);

  if (data_format == FORMAT_NHWC) {
    TF_CHECK_OK(
        GpuLaunchKernel(MaxPoolGradBackwardNoMaskNHWC<T>, config.block_count,
                        config.thread_per_block, 0, d.stream(), num_kernels,
                        bottom_data, output_data, pooled_height, pooled_width,
                        channels, height, width, kernel_h, kernel_w, stride_h,
                        stride_w, pad_t, pad_l, top_diff, bottom_diff));
  } else {
    TF_CHECK_OK(
        GpuLaunchKernel(MaxPoolGradBackwardNoMaskNCHW<T>, config.block_count,
                        config.thread_per_block, 0, d.stream(), num_kernels,
                        bottom_data, output_data, pooled_height, pooled_width,
                        channels, height, width, kernel_h, kernel_w, stride_h,
                        stride_w, pad_t, pad_l, top_diff, bottom_diff));
  }
  return d.ok();
}

template <typename T>
bool MaxPoolGradBackwardWithArgmax<T>::operator()(
    const int output_size, const int input_size, const T* top_diff,
    const int64* mask, const int top_offset, const int bottom_offset,
    T* bottom_diff, const Eigen::GpuDevice& d,
    const bool include_batch_in_index) {
  if (input_size == 0) return true;
  GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d);
  TF_CHECK_OK(GpuLaunchKernel(
      MaxPoolGradBackward<T>, config.block_count, config.thread_per_block, 0,
      d.stream(), output_size, top_diff, mask, top_offset, bottom_offset,
      bottom_diff, include_batch_in_index));
  return d.ok();
}

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                          \
  template struct SpatialMaxPooling<GPUDevice, T>;     \
  template struct MaxPoolForwardWithOptionalArgmax<T>; \
  template struct MaxPoolBackwardWithArgmax<T>;        \
  template struct MaxPoolGradBackwardWithArgmax<T>;    \
  template struct MaxPoolGradBackwardNoMask<T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

#undef DEFINE_GPU_KERNELS

}  // namespace functor

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
