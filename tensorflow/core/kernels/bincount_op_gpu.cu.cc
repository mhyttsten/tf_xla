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
class MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bincount_op.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, false> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/bincount_op_gpu.cu.cc", "Compute");

    if (weights.size() != 0) {
      return errors::InvalidArgument(
          "Weights should not be passed as it should be "
          "handled by unsorted_segment_sum");
    }
    if (output.size() == 0) {
      return Status::OK();
    }
    if (tensorflow::OpDeterminismRequired()) {
      // TODO(reedwm): Is this really nondeterministic? There is no
      // documentation in DeviceHistogram::HistogramEven on whether it is
      // deterministic or not.
      return errors::Unimplemented(
          "Determinism is not yet supported in GPU implementation of "
          "Bincount.");
    }
    // In case weight.size() == 0, use CUB
    size_t temp_storage_bytes = 0;
    const Tidx* d_samples = arr.data();
    T* d_histogram = output.data();
    int num_levels = output.size() + 1;
    Tidx lower_level = Tidx(0);
    Tidx upper_level = num_bins;
    int num_samples = arr.size();
    const gpuStream_t& stream = GetGpuStream(context);

    // The first HistogramEven is to obtain the temp storage size required
    // with d_temp_storage = NULL passed to the call.
    auto err = gpuprim::DeviceHistogram::HistogramEven(
        /* d_temp_storage */ NULL,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* lower_level */ lower_level,
        /* upper_level */ upper_level,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramEven to get temp storage: ",
          GpuGetErrorString(err), ".");
    }
    Tensor temp_storage;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<int8>::value,
        TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    void* d_temp_storage = temp_storage.flat<int8>().data();
    // The second HistogramEven is to actual run with d_temp_storage
    // allocated with temp_storage_bytes.
    err = gpuprim::DeviceHistogram::HistogramEven(
        /* d_temp_storage */ d_temp_storage,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_samples */ d_samples,
        /* d_histogram */ d_histogram,
        /* num_levels */ num_levels,
        /* lower_level */ lower_level,
        /* upper_level */ upper_level,
        /* num_samples */ num_samples,
        /* stream */ stream);
    if (err != gpuSuccess) {
      return errors::Internal(
          "Could not launch HistogramEven: ", GpuGetErrorString(err), ".");
    }
    return Status::OK();
  }
};

template <typename Tidx, typename T>
__global__ void BincountReduceKernel(const Tidx* in, T* out, const int nthreads,
                                     const Tidx num_bins) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      out[bin] = T(1);
    }
  }
}

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, true> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc mht_1(mht_1_v, 303, "", "./tensorflow/core/kernels/bincount_op_gpu.cu.cc", "Compute");

    const int nthreads = arr.dimension(0);

    auto d = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(nthreads, d);
    return GpuLaunchKernel(BincountReduceKernel<Tidx, T>, config.block_count,
                           config.thread_per_block, 0, d.stream(), arr.data(),
                           output.data(), nthreads, num_bins);
  }
};

template <typename Tidx, typename T, bool binary_count>
__global__ void BincountColReduceKernel(const Tidx* in, const T* weights,
                                        const int weights_size, T* out,
                                        const int num_rows, const int num_cols,
                                        const Tidx num_bins) {
  const int nthreads = num_rows * num_cols;
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        out[offset] = T(1);
      } else {
        T value = (weights_size == 0) ? T(1) : ldg(weights + index);
        GpuAtomicAdd(out + offset, value);
      }
    }
  }
}

template <typename Tidx, typename T, bool binary_count>
__global__ void BincountColReduceSharedKernel(const Tidx* in, const T* weights,
                                              const int weights_size, T* out,
                                              const int num_rows,
                                              const int num_cols,
                                              const Tidx num_bins) {
  const int out_size = num_rows * num_bins;
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char, shared_col_mem);
  T* shared_col_bins = reinterpret_cast<T*>(shared_col_mem);
  for (unsigned int binIdx = threadIdx.x; binIdx < out_size;
       binIdx += blockDim.x) {
    shared_col_bins[binIdx] = T(0);
  }
  __syncthreads();
  const int nthreads = num_rows * num_cols;
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    Tidx bin = ldg(in + index);
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        shared_col_bins[offset] = T(1);
      } else {
        T value = (weights_size == 0) ? T(1) : ldg(weights + index);
        GpuAtomicAdd(shared_col_bins + offset, value);
      }
    }
  }
  __syncthreads();
  for (unsigned int binIdx = threadIdx.x; binIdx < out_size;
       binIdx += blockDim.x) {
    if (binary_count) {
      // out[binIdx] = out[binIdx] & shared_col_bins[binIdx];
      if (shared_col_bins[binIdx]) {
        out[binIdx] = shared_col_bins[binIdx];
      }
    } else {
      GpuAtomicAdd(out + binIdx, shared_col_bins[binIdx]);
    }
  }
}

template <typename Tidx, typename T, bool binary_count>
struct BincountReduceFunctor<GPUDevice, Tidx, T, binary_count> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 2>::ConstTensor& in,
                        const typename TTypes<T, 2>::ConstTensor& weights,
                        typename TTypes<T, 2>::Tensor& out,
                        const Tidx num_bins) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbincount_op_gpuDTcuDTcc mht_2(mht_2_v, 386, "", "./tensorflow/core/kernels/bincount_op_gpu.cu.cc", "Compute");

    const int num_rows = in.dimension(0);
    const int num_cols = in.dimension(1);

    auto d = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_rows * num_cols, d);

    // Use half of maximum shared memory, approximately 6 * 1024 inputs.
    int smem_max = d.sharedMemPerBlock() / 2;
    int smem_usage = out.size() * sizeof(T);
    if (smem_usage < smem_max) {
      return GpuLaunchKernel(
          BincountColReduceSharedKernel<Tidx, T, binary_count>,
          config.block_count, config.thread_per_block, smem_usage, d.stream(),
          in.data(), weights.data(), weights.size(), out.data(), num_rows,
          num_cols, num_bins);
    }
    return GpuLaunchKernel(
        BincountColReduceKernel<Tidx, T, binary_count>, config.block_count,
        config.thread_per_block, 0, d.stream(), in.data(), weights.data(),
        weights.size(), out.data(), num_rows, num_cols, num_bins);
  }
};

}  // end namespace functor

#define REGISTER_GPU_SPEC(T)                                                  \
  template struct functor::BincountFunctor<GPUDevice, int32, T, true>;        \
  template struct functor::BincountFunctor<GPUDevice, int64, T, true>;        \
  template struct functor::BincountFunctor<GPUDevice, int32, T, false>;       \
  template struct functor::BincountFunctor<GPUDevice, int64, T, false>;       \
  template struct functor::BincountReduceFunctor<GPUDevice, int32, T, true>;  \
  template struct functor::BincountReduceFunctor<GPUDevice, int64, T, true>;  \
  template struct functor::BincountReduceFunctor<GPUDevice, int32, T, false>; \
  template struct functor::BincountReduceFunctor<GPUDevice, int64, T, false>;

TF_CALL_int32(REGISTER_GPU_SPEC);
TF_CALL_float(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
