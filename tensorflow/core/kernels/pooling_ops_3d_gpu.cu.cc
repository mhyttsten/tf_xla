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
class MHTracer_DTPStensorflowPScorePSkernelsPSpooling_ops_3d_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSpooling_ops_3d_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSpooling_ops_3d_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/pooling_ops_3d_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNCDHW(
    const int nthreads, const dtype* __restrict__ bottom_data,
    const dtype* __restrict__ output_data, const int pooled_plane,
    const int pooled_height, const int pooled_width, const int channels,
    const int plane, const int height, const int width, const int kernel_p,
    const int kernel_h, const int kernel_w, const int stride_p,
    const int stride_h, const int stride_w, const int pad_p, const int pad_t,
    const int pad_l, const dtype* __restrict__ top_diff,
    dtype* __restrict__ bottom_diff) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpooling_ops_3d_gpuDTcuDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/pooling_ops_3d_gpu.cu.cc", "MaxPoolGradBackwardNoMaskNCDHW");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pp = (index / pooled_width / pooled_height) % pooled_plane;
    int c = (index / pooled_width / pooled_height / pooled_plane) % channels;
    int n = (index / pooled_width / pooled_height / pooled_plane / channels);
    int pstart = pp * stride_p - pad_p;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    const int pend = min(pstart + kernel_p, plane);
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    pstart = max(pstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n =
        bottom_data + n * channels * plane * height * width;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx = c * plane * height * width + (p * height + h) * width + w;
          if (output_data[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] =
          top_diff[n * channels * plane * height * width + maxidx];
    }
  }
}

template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNDHWC(
    const int nthreads, const dtype* __restrict__ bottom_data,
    const dtype* __restrict__ output_data, const int pooled_plane,
    const int pooled_height, const int pooled_width, const int channels,
    const int plane, const int height, const int width, const int kernel_p,
    const int kernel_h, const int kernel_w, const int stride_p,
    const int stride_h, const int stride_w, const int pad_p, const int pad_t,
    const int pad_l, const dtype* __restrict__ top_diff,
    dtype* __restrict__ bottom_diff) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpooling_ops_3d_gpuDTcuDTcc mht_1(mht_1_v, 261, "", "./tensorflow/core/kernels/pooling_ops_3d_gpu.cu.cc", "MaxPoolGradBackwardNoMaskNDHWC");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    int wend = min(wstart + kernel_w, width);
    wstart = max(wstart, 0);
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    n /= pooled_height;
    int pstart = (n % pooled_plane) * stride_p - pad_p;
    int pend = min(pstart + kernel_p, plane);
    pstart = max(pstart, 0);
    n /= pooled_plane;
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n =
        bottom_data + n * plane * height * width * channels;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx = ((p * height + h) * width + w) * channels + c;
          if (output_data[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] =
          top_diff[n * plane * height * width * channels + maxidx];
    }
  }
}

}  // namespace

namespace functor {

template <typename T>
bool MaxPool3dGradBackward<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const T* output_data,
    const int batch, const int pooled_plane, const int pooled_height,
    const int pooled_width, const int channels, const int plane,
    const int height, const int width, const int kernel_p, const int kernel_h,
    const int kernel_w, const int stride_p, const int stride_h,
    const int stride_w, const int pad_p, const int pad_t, const int pad_l,
    const T* top_diff, T* bottom_diff, const Eigen::GpuDevice& d) {
  int num_kernels =
      batch * channels * pooled_plane * pooled_height * pooled_width;
  GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, d);
  if (data_format == FORMAT_NHWC) {
    TF_CHECK_OK(GpuLaunchKernel(
        MaxPoolGradBackwardNoMaskNDHWC<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), num_kernels, bottom_data,
        output_data, pooled_plane, pooled_height, pooled_width, channels, plane,
        height, width, kernel_p, kernel_h, kernel_w, stride_p, stride_h,
        stride_w, pad_p, pad_t, pad_l, top_diff, bottom_diff));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        MaxPoolGradBackwardNoMaskNCDHW<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), num_kernels, bottom_data,
        output_data, pooled_plane, pooled_height, pooled_width, channels, plane,
        height, width, kernel_p, kernel_h, kernel_w, stride_p, stride_h,
        stride_w, pad_p, pad_t, pad_l, top_diff, bottom_diff));
  }
  return d.ok();
}

}  // namespace functor

#define DEFINE_GPU_SPECS(T) template struct functor::MaxPool3dGradBackward<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
#undef DEFINE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
