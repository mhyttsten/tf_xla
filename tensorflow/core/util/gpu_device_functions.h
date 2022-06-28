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

#ifndef TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
#define TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh() {
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


/**
 * Wrappers and helpers for CUDA device code.
 *
 * Wraps the warp-cooperative intrinsics introduced in CUDA 9 to provide
 * backwards compatibility, see go/volta-porting for details.
 * Provides atomic operations on types that aren't natively supported.
 * Defines a number of macros and types providing a shared interface
 * to either CUDA or ROCm APIs, depending on the build.
 */

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>
#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#else
#include "rocm/include/hip/hip_complex.h"
#endif

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_cuda_alias.h"

#if GOOGLE_CUDA
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventDestroy cudaEventDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuFree cudaFree
#elif TENSORFLOW_USE_ROCM
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using cudaError = int;
using cudaError_t = int;
#define cudaSuccess 0
#define cudaGetLastError hipGetLastError
#define gpuEventRecord hipEventRecord
#define gpuEventDestroy hipEventDestroy
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuFree hipFree
static std::string cudaGetErrorString(int err) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_0(mht_0_v, 239, "", "./tensorflow/core/util/gpu_device_functions.h", "cudaGetErrorString");
 return std::to_string(err); }
#endif

#define TF_RETURN_IF_CUDA_ERROR(result)                   \
  do {                                                    \
    cudaError_t error(result);                            \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {         \
      return errors::Internal("Cuda call failed with ",   \
                              cudaGetErrorString(error)); \
    }                                                     \
  } while (0)

#define TF_OP_REQUIRES_CUDA_SUCCESS(context, result)                   \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      return;                                                          \
    }                                                                  \
  } while (0)

namespace tensorflow {
// According to HIP developer guide at
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md#assert
// assert is not supported by HIP. While we are waiting for assert support in
// hip kernels, the assert call should be macroed to NOP so that it does not
// block us from creating a debug build
#if TENSORFLOW_USE_ROCM
#undef assert
#define assert(x) \
  {}
#endif

namespace detail {

// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRange?() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_1(mht_1_v, 283, "", "./tensorflow/core/util/gpu_device_functions.h", "Iterator");
}
    __device__ T operator*() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_2(mht_2_v, 287, "", "./tensorflow/core/util/gpu_device_functions.h", "*");
 return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_3(mht_3_v, 316, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuGridRange");
}

  __device__ Iterator begin() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_4(mht_4_v, 321, "", "./tensorflow/core/util/gpu_device_functions.h", "begin");
 return Iterator{begin_, delta_}; }
  __device__ Iterator end() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_5(mht_5_v, 325, "", "./tensorflow/core/util/gpu_device_functions.h", "end");
 return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

#ifndef TENSORFLOW_USE_ROCM
template <typename... T>
using CudaGridRange = GpuGridRange<T...>;
#endif
}  // namespace detail

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
  return detail::GpuGridRange<T>(
      /*begin=*/blockIdx.x * blockDim.x + threadIdx.x,
      /*delta=*/gridDim.x * blockDim.x, /*end=*/count);
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuGridRangeX, CudaGridRangeX);

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : GpuGridRangeY(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeY(T count) {
  return detail::GpuGridRange<T>(
      /*begin=*/blockIdx.y * blockDim.y + threadIdx.y,
      /*delta=*/gridDim.y * blockDim.y, /*end=*/count);
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuGridRangeY, CudaGridRangeY);

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : GpuGridRangeZ(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeZ(T count) {
  return detail::GpuGridRange<T>(
      /*begin=*/blockIdx.z * blockDim.z + threadIdx.z,
      /*delta=*/gridDim.z * blockDim.z, /*end=*/count);
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuGridRangeZ, CudaGridRangeZ);

// Mask for all 32 threads in a warp.
__device__ const unsigned kCudaWarpAll = 0xffffffff;
// ROCM TODO add ROCM implementation
// Mask for all 64 threads in a wavefront.
__device__ const unsigned kGpuWarpAll = 0xffffffff;

// Returns the warp lane ID of the calling thread
__device__ inline unsigned GpuLaneId() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_6(mht_6_v, 381, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuLaneId");

  unsigned int lane_id;
#if GOOGLE_CUDA
#if __clang__
  return __nvvm_read_ptx_sreg_laneid();
#else   // __clang__
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#endif  // __clang__
#elif TENSORFLOW_USE_ROCM
  lane_id = __lane_id();
#endif
  return lane_id;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuLaneId, CudaLaneId);

namespace detail {
// Returns true if mask is a valid parameter for __shfl*sync to return a well
// defined value, assuming the calling lane will read from src_lane as part of
// the shuffle operation.
//
// Specifically, returns true iff mask has the calling lane bit and the src_lane
// bit set, and the src_lane calls this function with the same mask value
// (required for the two threads to wait for each other).
//
// On Volta, for some invalid masks, this function hangs or returns false
// positives, because the implementation shuffles with the same mask that
// we are validating. Run on Pascal if you suspect that the mask is incorrect.
__device__ inline bool GpuValidateShuffleSyncMask(unsigned mask,
                                                  unsigned src_lane) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_7(mht_7_v, 412, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuValidateShuffleSyncMask");

  unsigned src_dst_mask = 1u << GpuLaneId() | 1u << src_lane;
#if CUDA_VERSION >= 9000
  unsigned src_lane_mask = __shfl_sync(mask, mask, src_lane);
#else
#if GOOGLE_CUDA
  unsigned src_lane_mask = __shfl(mask, src_lane);
#elif TENSORFLOW_USE_ROCM
  unsigned src_lane_mask =
      __shfl(static_cast<int>(mask), static_cast<int>(src_lane));
#endif
#endif
  return (src_dst_mask & ~mask) == 0 && src_lane_mask == mask;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuValidateShuffleSyncMask,
                                  CudaValidateShuffleSyncMask);

// Returns the actual source lane for shuffle.
__device__ inline unsigned GpuShuffleGetSrcLane(int src_lane, int width) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_8(mht_8_v, 433, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleGetSrcLane");

  int lane_id = GpuLaneId();
  int lane_base = lane_id & ~width + 1;
  int lane_offset = src_lane & width - 1;
  return lane_base + lane_offset;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleGetSrcLane, CudaShuffleGetSrcLane);

// Returns the source lane for shuffle up.
__device__ inline unsigned GpuShuffleUpGetSrcLane(unsigned delta, int width) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_9(mht_9_v, 445, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleUpGetSrcLane");

  unsigned lane_id = GpuLaneId();
  if ((lane_id & width - 1) < delta) {
    return lane_id;
  }
  return lane_id - delta;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleUpGetSrcLane,
                                  CudaShuffleUpGetSrcLane);

// Returns the source lane for shuffle down.
__device__ inline unsigned GpuShuffleDownGetSrcLane(unsigned delta, int width) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_10(mht_10_v, 459, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleDownGetSrcLane");

  unsigned lane_id = GpuLaneId();
  if ((lane_id & width - 1) + delta >= width) {
    return lane_id;
  }
  return lane_id + delta;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleDownGetSrcLane,
                                  CudaShuffleDownGetSrcLane);

// Returns the source lane for shuffle xor.
__device__ inline unsigned GpuShuffleXorGetSrcLane(int lane_mask, int width) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_11(mht_11_v, 473, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleXorGetSrcLane");

  int lane_id = GpuLaneId();
  int src_lane = lane_id ^ lane_mask;
  if (src_lane > (lane_id | width - 1)) {
    return lane_id;
  }
  return src_lane;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleXorGetSrcLane,
                                  CudaShuffleXorGetSrcLane);
}  // namespace detail

// For all *_sync wrappers below, it is illegal to synchronize threads from
// different program locations, because that is not supported before sm_70.
// In other words, all threads in 'mask' must call the functions in convergence.
// Code that requires sm_70 (and CUDA 9) may use the intrinsic directly.
//
// It is also illegal to shuffle with a mask that produces an undefined result
// for any of the threads. Specifically, all source threads of the shuffle
// must have their corresponding bit in 'mask' set.

// Wrapper for __syncwarp. No-op for CUDA 8 and earlier.
__device__ inline void GpuSyncWarp(unsigned mask = kCudaWarpAll) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_12(mht_12_v, 498, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuSyncWarp");

  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  __syncwarp(mask);
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuSyncWarp, CudaSyncWarp);

// Wrapper for __ballot_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline unsigned GpuBallotSync(unsigned mask, int pred) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_13(mht_13_v, 511, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuBallotSync");

  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __ballot_sync(mask, pred);
#else
  return __ballot(pred) & mask;  // Apply mask to match __ballot_sync's spec.
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuBallotSync, CudaBallotSync);

// Wrapper for __any_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int GpuAnySync(unsigned mask, int pred) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_14(mht_14_v, 526, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAnySync");

  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __any_sync(mask, pred);
#else
  return __any(pred);
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAnySync, CudaAnySync);

// Wrapper for __all_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int GpuAllSync(unsigned mask, int pred) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_15(mht_15_v, 541, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAllSync");

  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __all_sync(mask, pred);
#else
  return __all(pred);
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAllSync, CudaAllSync);

// Wrapper for __shfl_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T GpuShuffleSync(unsigned mask, T value, int src_lane,
                            int width = warpSize) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_16(mht_16_v, 558, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleSync");

  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleGetSrcLane(src_lane, width)));
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleSync(unsigned mask, double value,
                                        int src_lane, int width = warpSize) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_17(mht_17_v, 576, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleSync");

#if GOOGLE_CUDA
  auto tmp = __double_as_longlong(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = GpuShuffleSync(mask, hi, src_lane, width);
  lo = GpuShuffleSync(mask, lo, src_lane, width);
  return __longlong_as_double(static_cast<uint64_t>(hi) << 32 | lo);
#elif TENSORFLOW_USE_ROCM
  auto tmp = static_cast<uint64_t>(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl(static_cast<int>(hi), src_lane, width);
  lo = __shfl(static_cast<int>(lo), src_lane, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                             static_cast<uint64_t>(lo));
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleSync, CudaShuffleSync);

// Wrapper for __shfl_up_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ inline T GpuShuffleUpSync(unsigned mask, T value, unsigned delta,
                                     int width = warpSize) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_18(mht_18_v, 603, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleUpSync");

  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleUpGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleUpSync(unsigned mask, double value,
                                          unsigned delta,
                                          int width = warpSize) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_19(mht_19_v, 622, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleUpSync");

#if GOOGLE_CUDA
  auto tmp = __double_as_longlong(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = GpuShuffleUpSync(mask, hi, delta, width);
  lo = GpuShuffleUpSync(mask, lo, delta, width);
  return __longlong_as_double(static_cast<uint64_t>(hi) << 32 | lo);
#elif TENSORFLOW_USE_ROCM
  auto tmp = static_cast<uint64_t>(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_up(static_cast<int>(hi), delta, width);
  lo = __shfl_up(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                             static_cast<uint64_t>(lo));
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleUpSync, CudaShuffleUpSync);

// Wrapper for __shfl_down_sync. All threads in 'mask' must call this function
// in convergence, see comment above for details.
template <typename T>
__device__ inline T GpuShuffleDownSync(unsigned mask, T value, unsigned delta,
                                       int width = warpSize) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_20(mht_20_v, 649, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleDownSync");

  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleDownGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleDownSync(unsigned mask, double value,
                                            unsigned delta,
                                            int width = warpSize) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_21(mht_21_v, 668, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleDownSync");

#if GOOGLE_CUDA
  auto tmp = __double_as_longlong(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = GpuShuffleDownSync(mask, hi, delta, width);
  lo = GpuShuffleDownSync(mask, lo, delta, width);
  return __longlong_as_double(static_cast<uint64_t>(hi) << 32 | lo);
#elif TENSORFLOW_USE_ROCM
  auto tmp = static_cast<uint64_t>(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_down(static_cast<int>(hi), delta, width);
  lo = __shfl_down(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                             static_cast<uint64_t>(lo));
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleDownSync, CudaShuffleDownSync);

// Wrapper for __shfl_xor_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T GpuShuffleXorSync(unsigned mask, T value, int lane_mask,
                               int width = warpSize) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_22(mht_22_v, 695, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleXorSync");

  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleXorGetSrcLane(lane_mask, width)));
#if GOOGLE_CUDA
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, lane_mask, width);
#else
  return __shfl_xor(value, lane_mask, width);
#endif
#elif TENSORFLOW_USE_ROCM
  // ROCM TODO: check if HIP should be changed to cope with more types
  return __shfl_xor(static_cast<int>(value), lane_mask, width);
#endif
}

#if TENSORFLOW_USE_ROCM
__device__ inline Eigen::half GpuShuffleXorSync(unsigned mask,
                                                Eigen::half value,
                                                int lane_mask,
                                                int width = warpSize) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_23(mht_23_v, 718, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuShuffleXorSync");

  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleXorGetSrcLane(lane_mask, width)));
  // TODO(rocm): This doesn't preserve NaN payload and flushes denorms to zero,
  // maybe this should be implemented differently?
  return static_cast<Eigen::half>(
      __shfl_xor(static_cast<float>(value), lane_mask, width));
}
#endif

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleXorSync(unsigned mask, double value,
                                           int lane_mask,
                                           int width = warpSize) {
#if GOOGLE_CUDA
  auto tmp = __double_as_longlong(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = GpuShuffleXorSync(mask, hi, lane_mask, width);
  lo = GpuShuffleXorSync(mask, lo, lane_mask, width);
  return __longlong_as_double(static_cast<uint64_t>(hi) << 32 | lo);
#elif TENSORFLOW_USE_ROCM
  auto tmp = static_cast<uint64_t>(value);
  auto lo = static_cast<unsigned>(tmp);
  auto hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_xor(static_cast<int>(hi), lane_mask, width);
  lo = __shfl_xor(static_cast<int>(lo), lane_mask, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                             static_cast<uint64_t>(lo));
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuShuffleXorSync, CudaShuffleXorSync);

// Wrapper for __ldg.
template <typename T>
__host__ __device__ T GpuLdg(const T* address) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_24(mht_24_v, 759, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuLdg");

#if __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

__host__ __device__ inline bool GpuLdg(const bool* address) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_25(mht_25_v, 770, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuLdg");

  return GpuLdg(reinterpret_cast<const char*>(address)) != 0;
}

__host__ __device__ inline std::complex<float> GpuLdg(
    const std::complex<float>* address) {
#if __CUDA_ARCH__ >= 350
  float2 mem = __ldg(reinterpret_cast<const float2*>(address));
  return std::complex<float>(mem.x, mem.y);
#else
  return *address;
#endif
}

__host__ __device__ inline std::complex<double> GpuLdg(
    const std::complex<double>* address) {
#if __CUDA_ARCH__ >= 350
  double2 mem = __ldg(reinterpret_cast<const double2*>(address));
  return std::complex<double>(mem.x, mem.y);
#else
  return *address;
#endif
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuLdg, CudaLdg);

// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T>
__global__ void SetZero(const int count, T* __restrict__ ptr) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_26(mht_26_v, 802, "", "./tensorflow/core/util/gpu_device_functions.h", "SetZero");

  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = T(0);
  }
}

// Helper to set all tensor entries to a specific value.
template <typename T, typename Tvalue = T>
__global__ void SetToValue(const int count, T* __restrict__ ptr, Tvalue value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = static_cast<T>(value);
  }
}

namespace detail {
// Helper function for atomic accumulation implemented as CAS.
template <typename T, typename F>
__device__ T GpuAtomicCasHelper(T* ptr, F accumulate) {
  T old = *ptr;
  T assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, accumulate(assumed));
  } while (assumed != old);
  return old;
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicCasHelper, CudaAtomicCasHelper);

// Overload for floating point (using integer comparison to handle NaN
// correctly).
template <typename F>
__device__ float GpuAtomicCasHelper(float* ptr, F accumulate) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_27(mht_27_v, 844, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicCasHelper");

  return __int_as_float(
      GpuAtomicCasHelper(reinterpret_cast<int32*>(ptr), [accumulate](int32 a) {
        return __float_as_int(accumulate(__int_as_float(a)));
      }));
}
template <typename F>
__device__ double GpuAtomicCasHelper(double* ptr, F accumulate) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_28(mht_28_v, 854, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicCasHelper");

#if TENSORFLOW_USE_ROCM
  // FIXME: remove the workaround below once bug is fixed.
  // HIP has a bug in the implementation of __longlong_as_double
  // So workaround it by using reinterpret_cast<double*>.
  uint64_t result =
      GpuAtomicCasHelper(reinterpret_cast<unsigned long long*>(ptr),
                         [accumulate](tensorflow::uint64 a) {
                           return __double_as_longlong(
                               accumulate(*(reinterpret_cast<double*>(&a))));
                         });
  return *(reinterpret_cast<double*>(&result));
#else
  return __longlong_as_double(GpuAtomicCasHelper(
      reinterpret_cast<unsigned long long*>(ptr),
      [accumulate](tensorflow::uint64 a) {
        return __double_as_longlong(accumulate(__longlong_as_double(a)));
      }));
#endif
}

// Overload of above function for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// This version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
template <typename F>
__device__ Eigen::half GpuAtomicCasHelper(Eigen::half* ptr, F accumulate) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_29(mht_29_v, 891, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicCasHelper");

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr - 2);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      Eigen::half acc = accumulate(Eigen::numext::bit_cast<Eigen::half>(high));
      return (static_cast<uint32>(Eigen::numext::bit_cast<uint16>(acc)) << 16) |
             (arg & 0xffff);
    });
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16>(result >> 16));
  } else {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short low = static_cast<unsigned short>(arg & 0xffff);
      Eigen::half acc = accumulate(Eigen::numext::bit_cast<Eigen::half>(low));
      return (arg & 0xffff0000) |
             static_cast<uint32>(Eigen::numext::bit_cast<uint16>(acc));
    });
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16>(result & 0xffff));
  }
}

template <typename F>
__device__ long long GpuAtomicCasHelper(long long* ptr, F accumulate) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_30(mht_30_v, 926, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicCasHelper");

  return static_cast<long long>(
      GpuAtomicCasHelper(reinterpret_cast<unsigned long long*>(ptr),
                         [accumulate](unsigned long long a) {
                           return static_cast<unsigned long long>(
                               accumulate(static_cast<long long>(a)));
                         }));
}

template <typename From, typename To>
using ToTypeIfConvertible =
    typename std::enable_if<std::is_convertible<From, To>::value, To>::type;

template <typename T>
struct CudaSupportedTypeImpl {
  using type = T;
};

template <>
struct CudaSupportedTypeImpl<long long> {
  using type = unsigned long long;
};

template <>
struct CudaSupportedTypeImpl<unsigned long> {
  using type =
      typename std::conditional<sizeof(unsigned long) == sizeof(unsigned int),
                                unsigned int, unsigned long long>::type;
};

template <>
struct CudaSupportedTypeImpl<long> {
  // This cast should be safe since module-2 addition should work fine. However,
  // signed overflow is not handled correctly since it's undefined behavior.
  using type = typename CudaSupportedTypeImpl<unsigned long>::type;
};

template <typename T>
using CudaSupportedType = typename CudaSupportedTypeImpl<T>::type;

template <typename T>
__device__ CudaSupportedType<T>* ToCudaSupportedPtr(T* ptr) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_31(mht_31_v, 970, "", "./tensorflow/core/util/gpu_device_functions.h", "ToCudaSupportedPtr");

  return reinterpret_cast<CudaSupportedType<T>*>(ptr);
}

}  // namespace detail

// CUDA provides atomic ops, but not for all types.  We provide wrappers
// for some ops and provide implementation for all reasonable types.

template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicAdd(T* ptr, U value) {
  return atomicAdd(detail::ToCudaSupportedPtr(ptr), value);
}

__device__ inline Eigen::half GpuAtomicAdd(Eigen::half* ptr,
                                           Eigen::half value) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_32(mht_32_v, 988, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicAdd");

  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a + value; });
}

#if (__CUDA_ARCH__ < 600) || TENSORFLOW_USE_ROCM
__device__ inline double GpuAtomicAdd(double* ptr, double value) {
  return detail::GpuAtomicCasHelper(ptr,
                                    [value](double a) { return a + value; });
}
#endif

// GpuAtomicAdd
// Specializations of GpuAtomicAdd for complex types, which GpuAtomicAdd does
// not support. We treat a std::complex<T>* as a T* (the C++ standard section
// 26.4.4 allows this explicitly) and atomic add the real and imaginary
// components individually. The operation as a whole is not atomic, but we can
// safely treat the components independently for the purpose of accumulating.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
__device__ inline std::complex<float> GpuAtomicAdd(std::complex<float>* ptr,
                                                   std::complex<float> value) {
  auto ptr_scalar = reinterpret_cast<float*>(ptr);
  return std::complex<float>(GpuAtomicAdd(ptr_scalar, value.real()),
                             GpuAtomicAdd(ptr_scalar + 1, value.imag()));
}

__device__ inline std::complex<double> GpuAtomicAdd(
    std::complex<double>* ptr, std::complex<double> value) {
  auto ptr_scalar = reinterpret_cast<double*>(ptr);
  return std::complex<double>(GpuAtomicAdd(ptr_scalar, value.real()),
                              GpuAtomicAdd(ptr_scalar + 1, value.imag()));
}
#endif
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicAdd, CudaAtomicAdd);

// GpuAtomicSub
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicSub(T* ptr, U value) {
  return atomicSub(ptr, value);
}

// Specializations of substraction which add the negative value.
__device__ inline float GpuAtomicSub(float* ptr, float value) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_33(mht_33_v, 1034, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicSub");

  return GpuAtomicAdd(ptr, -value);
}

__device__ inline double GpuAtomicSub(double* ptr, double value) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_34(mht_34_v, 1041, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicSub");

  return GpuAtomicAdd(ptr, -value);
}

__device__ inline int64_t GpuAtomicSub(int64_t* ptr, int64_t value) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_35(mht_35_v, 1048, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicSub");

  return GpuAtomicAdd(ptr, -value);
}

__device__ inline tensorflow::uint64 GpuAtomicSub(tensorflow::uint64* ptr,
                                                  tensorflow::uint64 value) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_36(mht_36_v, 1056, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicSub");

  return GpuAtomicAdd(ptr, -static_cast<int64_t>(value));
}

__device__ inline Eigen::half GpuAtomicSub(Eigen::half* ptr,
                                           Eigen::half value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_37(mht_37_v, 1064, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicSub");

  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a - value; });
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicSub, CudaAtomicSub);

// GpuAtomicMax
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMax(T* ptr, U value) {
  return atomicMax(detail::ToCudaSupportedPtr(ptr), value);
}

#if TENSORFLOW_USE_ROCM

/*
 * CUDA runtime headers have the following defined
 *   __device__  int max(int, int)
 *   __device__  float max(float, float)
 *   __device__  double max(double, double)
 *
 * and many others, where as HIP runtime headers only have the "int" version
 *
 * Therefore need to special case ROCm version to call the correct underlying
 * routines for float and double types.
 *
 */

__device__ inline float GpuAtomicMax(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](float a) { return fmaxf(a, value); });
}

__device__ inline double GpuAtomicMax(double* ptr, double value) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_38(mht_38_v, 1099, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMax");

  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return fmax(a, value); });
}

#else

__device__ inline float GpuAtomicMax(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(ptr,
                                    [value](float a) { return max(a, value); });
}

__device__ inline double GpuAtomicMax(double* ptr, double value) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_39(mht_39_v, 1114, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMax");

  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return max(a, value); });
}

#endif

__device__ inline Eigen::half GpuAtomicMax(Eigen::half* ptr,
                                           Eigen::half value) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_40(mht_40_v, 1125, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMax");

  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return max(a, value); });
}

#if TENSORFLOW_USE_ROCM || (__CUDA_ARCH__ < 320)
__device__ inline tensorflow::uint64 GpuAtomicMax(tensorflow::uint64* ptr,
                                                  tensorflow::uint64 value) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_41(mht_41_v, 1135, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMax");

  return detail::GpuAtomicCasHelper(
      detail::ToCudaSupportedPtr(ptr),
      [value](tensorflow::uint64 a) { return max(a, value); });
}

__device__ inline int64_t GpuAtomicMax(int64_t* ptr, int64_t value) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_42(mht_42_v, 1144, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMax");

  return detail::GpuAtomicCasHelper(
      detail::ToCudaSupportedPtr(ptr),
      [value](int64_t a) { return max(a, value); });
}
#endif
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicMax, CudaAtomicMax);

// GpuAtomicMin
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMin(T* ptr, U value) {
  return atomicMin(detail::ToCudaSupportedPtr(ptr), value);
}

#if TENSORFLOW_USE_ROCM

/*
 * CUDA runtime headers have the following defined
 *   __device__  int min(int, int)
 *   __device__  float min(float, float)
 *   __device__  double min(double, double)
 *
 * and many others, where as HIP runtime headers only have the "int" version
 *
 * Therefore need to special case ROCm version to call the correct underlying
 * routines for float and double types.
 *
 */

__device__ inline float GpuAtomicMin(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](float a) { return fminf(a, value); });
}

__device__ inline double GpuAtomicMin(double* ptr, double value) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_43(mht_43_v, 1181, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMin");

  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return fmin(a, value); });
}

#else

__device__ inline float GpuAtomicMin(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(ptr,
                                    [value](float a) { return min(a, value); });
}

__device__ inline double GpuAtomicMin(double* ptr, double value) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_44(mht_44_v, 1196, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMin");

  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return min(a, value); });
}

#endif

__device__ inline Eigen::half GpuAtomicMin(Eigen::half* ptr,
                                           Eigen::half value) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_45(mht_45_v, 1207, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMin");

  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return min(a, value); });
}

#if TENSORFLOW_USE_ROCM || (__CUDA_ARCH__ < 320)
__device__ inline tensorflow::uint64 GpuAtomicMin(tensorflow::uint64* ptr,
                                                  tensorflow::uint64 value) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_46(mht_46_v, 1217, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMin");

  return detail::GpuAtomicCasHelper(
      detail::ToCudaSupportedPtr(ptr),
      [value](tensorflow::uint64 a) { return min(a, value); });
}

__device__ inline int64_t GpuAtomicMin(int64_t* ptr, int64_t value) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_device_functionsDTh mht_47(mht_47_v, 1226, "", "./tensorflow/core/util/gpu_device_functions.h", "GpuAtomicMin");

  return detail::GpuAtomicCasHelper(
      detail::ToCudaSupportedPtr(ptr),
      [value](int64_t a) { return min(a, value); });
}
#endif
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicMin, CudaAtomicMin);

// GpuAtomicMul
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMul(T* ptr, U value) {
  return detail::GpuAtomicCasHelper(ptr, [value](T a) { return a * value; });
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicMul, CudaAtomicMul);

// GpuAtomicDiv
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicDiv(T* ptr, U value) {
  return detail::GpuAtomicCasHelper(ptr, [value](T a) { return a / value; });
}
CREATE_CUDA_DEVICE_FUNCTION_ALIAS(GpuAtomicDiv, CudaAtomicDiv);

// Import all specialized std::complex device operators in namespace tensorflow.
#if GOOGLE_CUDA && defined(EIGEN_USING_STD_COMPLEX_OPERATORS)
EIGEN_USING_STD_COMPLEX_OPERATORS
#endif  // GOOGLE_CUDA

namespace functor {
// Import all specialized std::complex device operators in namespace functor.
#if GOOGLE_CUDA && defined(EIGEN_USING_STD_COMPLEX_OPERATORS)
EIGEN_USING_STD_COMPLEX_OPERATORS
#endif  // GOOGLE_CUDA

// ROCm hcc(clang) has severe difficulties dealing with std::complex directly
// due to a header issue. This template assists in casting std::complex into the
// corresponding internal ROCm types.
template <class T>
struct MapComplexToHipComplex {
  typedef T TM;
};

#if TENSORFLOW_USE_ROCM
template <>
struct MapComplexToHipComplex<std::complex<float> > {
  typedef hipFloatComplex TM;
};

template <>
struct MapComplexToHipComplex<std::complex<double> > {
  typedef hipDoubleComplex TM;
};
#endif
};  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
