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
class MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <cfloat>
#include <vector>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dilation_ops.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void DilationKernel(
    const int32 nthreads, const T* __restrict__ input_ptr,
    const T* __restrict__ filter_ptr, int batch, int input_rows, int input_cols,
    int depth, int filter_rows, int filter_cols, int output_rows,
    int output_cols, int stride_rows, int stride_cols, int rate_rows,
    int rate_cols, int pad_top, int pad_left, T* __restrict__ output_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/dilation_ops_gpu.cu.cc", "DilationKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
            }
          }
        }
      }
    }
    output_ptr[out_idx] = cur_val;
  }
}

template <typename T>
__global__ void DilationBackpropInputKernel(
    const int32 nthreads, const T* __restrict__ input_ptr,
    const T* __restrict__ filter_ptr, const T* __restrict__ out_backprop_ptr,
    int batch, int input_rows, int input_cols, int depth, int filter_rows,
    int filter_cols, int output_rows, int output_cols, int stride_rows,
    int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left,
    T* __restrict__ in_backprop_ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc mht_1(mht_1_v, 255, "", "./tensorflow/core/kernels/dilation_ops_gpu.cu.cc", "DilationBackpropInputKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    int h_in_max = (h_beg < 0) ? 0 : h_beg;
    int w_in_max = (w_beg < 0) ? 0 : w_beg;
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
              h_in_max = h_in;
              w_in_max = w_in;
            }
          }
        }
      }
    }
    GpuAtomicAdd(
        in_backprop_ptr + d +
            depth * (w_in_max + input_cols * (h_in_max + input_rows * b)),
        out_backprop_ptr[out_idx]);
  }
}

template <typename T>
__global__ void DilationBackpropFilterKernel(
    const int32 nthreads, const T* __restrict__ input_ptr,
    const T* __restrict__ filter_ptr, const T* __restrict__ out_backprop_ptr,
    int batch, int input_rows, int input_cols, int depth, int filter_rows,
    int filter_cols, int output_rows, int output_cols, int stride_rows,
    int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left,
    T* __restrict__ filter_backprop_ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdilation_ops_gpuDTcuDTcc mht_2(mht_2_v, 308, "", "./tensorflow/core/kernels/dilation_ops_gpu.cu.cc", "DilationBackpropFilterKernel");

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    int h_max = 0;
    int w_max = 0;
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
              h_max = h;
              w_max = w;
            }
          }
        }
      }
    }
    GpuAtomicAdd(
        filter_backprop_ptr + d + depth * (w_max + filter_cols * h_max),
        out_backprop_ptr[out_idx]);
  }
}

}  // namespace

namespace functor {

template <typename T>
struct Dilation<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, typename TTypes<T, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = output.dimension(1);
    const int output_cols = output.dimension(2);

    const int total_count = batch * output_rows * output_cols * depth;
    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);

    TF_CHECK_OK(GpuLaunchKernel(
        DilationKernel<T>, config.block_count, config.thread_per_block, 0,
        d.stream(), config.virtual_thread_count, input.data(), filter.data(),
        batch, input_rows, input_cols, depth, filter_rows, filter_cols,
        output_rows, output_cols, stride_rows, stride_cols, rate_rows,
        rate_cols, pad_top, pad_left, output.data()));
  }
};

template <typename T>
struct DilationBackpropInput<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 4>::Tensor in_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    int total_count;
    GpuLaunchConfig config;

    // Initialize in_backprop with all zeros.
    total_count = batch * input_rows * input_cols * depth;
    config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                total_count, in_backprop.data()));

    // Accumulate.
    total_count = batch * output_rows * output_cols * depth;
    config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        DilationBackpropInputKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
        input.data(), filter.data(), out_backprop.data(), batch, input_rows,
        input_cols, depth, filter_rows, filter_cols, output_rows, output_cols,
        stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        in_backprop.data()));
  }
};

template <typename T>
struct DilationBackpropFilter<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 3>::Tensor filter_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    int total_count;
    GpuLaunchConfig config;

    // Initialize filter_backprop with all zeros.
    total_count = filter_rows * filter_cols * depth;
    config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                total_count, filter_backprop.data()));

    // Accumulate.
    total_count = batch * output_rows * output_cols * depth;
    config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        DilationBackpropFilterKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
        input.data(), filter.data(), out_backprop.data(), batch, input_rows,
        input_cols, depth, filter_rows, filter_cols, output_rows, output_cols,
        stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        filter_backprop.data()));
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS(T)                                     \
  template struct functor::Dilation<GPUDevice, T>;              \
  template struct functor::DilationBackpropInput<GPUDevice, T>; \
  template struct functor::DilationBackpropFilter<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
