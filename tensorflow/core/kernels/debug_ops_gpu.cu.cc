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
class MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc() {
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

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A CUDA kernel that fills the second element of a vector according
// to whether any of the input data contains infinity or NaN.
template <typename Tin, typename Tout>
__global__ void CurtHealthKernel(const Tin* __restrict__ data, int size,
                                 Tout output[1]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;
  while (offset < size) {
    if (Eigen::numext::isinf(data[offset]) ||
        Eigen::numext::isnan(data[offset])) {
      output[0] = 1.0;
    }
    offset += total_thread_count;
  }
}

// A CUDA kernel that fills the three elements of an output
// vector with the number of NaNs, -infs, and infs in the input respectively.
template <typename Tin, typename Tout>
__global__ void ConciseHealthKernel(const Tin* __restrict__ data, int size,
                                    Tout output[3]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;
  Tout accum[3] = {0.0, 0.0, 0.0};

  while (offset < size) {
    if (Eigen::numext::isinf(data[offset])) {
      if (data[offset] < static_cast<Tin>(0.f)) {
        ++accum[0];
      } else {
        ++accum[1];
      }
    }
    if (Eigen::numext::isnan(data[offset])) {
      ++accum[2];
    }
    offset += total_thread_count;
  }

  GpuAtomicAdd(output, accum[0]);
  GpuAtomicAdd(output + 1, accum[1]);
  GpuAtomicAdd(output + 2, accum[2]);
}

// A CUDA kernel that fills the six elements of an output vector with the
// number of -infs, infs, nans, negatives, zeros, and positives in the input
// respectively.
template <typename Tin, typename Tout>
__global__ void FullHealthKernel(const Tin* __restrict__ data, int size,
                                 Tout output[6]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;
  Tout accum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  while (offset < size) {
    if (Eigen::numext::isinf(data[offset])) {
      if (data[offset] < static_cast<Tin>(0.f)) {
        ++accum[0];
      } else {
        ++accum[1];
      }
    } else if (Eigen::numext::isnan(data[offset])) {
      ++accum[2];
    } else {
      if (data[offset] < static_cast<Tin>(0.f)) {
        ++accum[3];
      } else if (data[offset] == static_cast<Tin>(0.f)) {
        ++accum[4];
      } else {
        ++accum[5];
      }
    }
    offset += total_thread_count;
  }

  GpuAtomicAdd(output, accum[0]);
  GpuAtomicAdd(output + 1, accum[1]);
  GpuAtomicAdd(output + 2, accum[2]);
  GpuAtomicAdd(output + 3, accum[3]);
  GpuAtomicAdd(output + 4, accum[4]);
  GpuAtomicAdd(output + 5, accum[5]);
}

// A CUDA kernel that fills a length-3 vector according to whether any of the
// input data contains negative infinity, positive infinity, or NaN. The first
// element is filled with -infinity if any of the elements is -infinity.
// The second element is filled with +infinity if any of the elements is
// +infinity. The last is filled with NaN if any of the elements is NaN.
template <typename Tin, typename Tout>
__global__ void ReduceInfNanThreeSlotsKernel(const Tin* __restrict__ data,
                                             int size, Tout output[3]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;

  while (offset < size) {
    if (Eigen::numext::isinf(data[offset])) {
      if (data[offset] < static_cast<Tin>(0.f)) {
        output[0] = -std::numeric_limits<Tout>::infinity();
      } else {
        output[1] = std::numeric_limits<Tout>::infinity();
      }
    }
    if (Eigen::numext::isnan(data[offset])) {
      output[2] = std::numeric_limits<Tout>::quiet_NaN();
    }
    offset += total_thread_count;
  }
}

}  // namespace

template <typename Tin, typename Tout>
struct CurtHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[1]) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc mht_0(mht_0_v, 326, "", "./tensorflow/core/kernels/debug_ops_gpu.cu.cc", "Run");

    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(GpuLaunchKernel(CurtHealthKernel<Tin, Tout>, num_blocks,
                                block_size, 0, d.stream(), data, size, output));
  }
};

template struct CurtHealthLaunch<Eigen::half, float>;
template struct CurtHealthLaunch<float, float>;
template struct CurtHealthLaunch<double, float>;
template struct CurtHealthLaunch<int16, float>;
template struct CurtHealthLaunch<int32, float>;
template struct CurtHealthLaunch<Eigen::half, double>;
template struct CurtHealthLaunch<float, double>;
template struct CurtHealthLaunch<double, double>;
template struct CurtHealthLaunch<int16, double>;
template struct CurtHealthLaunch<int32, double>;

template <typename Tin, typename Tout>
struct ConciseHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[3]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc mht_1(mht_1_v, 353, "", "./tensorflow/core/kernels/debug_ops_gpu.cu.cc", "Run");

    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(GpuLaunchKernel(ConciseHealthKernel<Tin, Tout>, num_blocks,
                                block_size, 0, d.stream(), data, size, output));
  }
};

template struct ConciseHealthLaunch<Eigen::half, float>;
template struct ConciseHealthLaunch<float, float>;
template struct ConciseHealthLaunch<double, float>;
template struct ConciseHealthLaunch<int16, float>;
template struct ConciseHealthLaunch<int32, float>;
template struct ConciseHealthLaunch<Eigen::half, double>;
template struct ConciseHealthLaunch<float, double>;
template struct ConciseHealthLaunch<double, double>;
template struct ConciseHealthLaunch<int16, double>;
template struct ConciseHealthLaunch<int32, double>;

template <typename Tin, typename Tout>
struct FullHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[6]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc mht_2(mht_2_v, 380, "", "./tensorflow/core/kernels/debug_ops_gpu.cu.cc", "Run");

    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(GpuLaunchKernel(FullHealthKernel<Tin, Tout>, num_blocks,
                                block_size, 0, d.stream(), data, size, output));
  }
};

template struct FullHealthLaunch<Eigen::half, float>;
template struct FullHealthLaunch<float, float>;
template struct FullHealthLaunch<double, float>;
template struct FullHealthLaunch<int16, float>;
template struct FullHealthLaunch<int32, float>;
template struct FullHealthLaunch<Eigen::half, double>;
template struct FullHealthLaunch<float, double>;
template struct FullHealthLaunch<double, double>;
template struct FullHealthLaunch<int16, double>;
template struct FullHealthLaunch<int32, double>;

template <typename Tin, typename Tout>
struct ReduceInfNanThreeSlotsLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[3]) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdebug_ops_gpuDTcuDTcc mht_3(mht_3_v, 407, "", "./tensorflow/core/kernels/debug_ops_gpu.cu.cc", "Run");

    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(GpuLaunchKernel(ReduceInfNanThreeSlotsKernel<Tin, Tout>,
                                num_blocks, block_size, 0, d.stream(), data,
                                size, output));
  }
};

template struct ReduceInfNanThreeSlotsLaunch<Eigen::half, float>;
template struct ReduceInfNanThreeSlotsLaunch<float, float>;
template struct ReduceInfNanThreeSlotsLaunch<double, float>;
template struct ReduceInfNanThreeSlotsLaunch<int16, float>;
template struct ReduceInfNanThreeSlotsLaunch<int32, float>;
template struct ReduceInfNanThreeSlotsLaunch<Eigen::half, double>;
template struct ReduceInfNanThreeSlotsLaunch<float, double>;
template struct ReduceInfNanThreeSlotsLaunch<double, double>;
template struct ReduceInfNanThreeSlotsLaunch<int16, double>;
template struct ReduceInfNanThreeSlotsLaunch<int32, double>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
