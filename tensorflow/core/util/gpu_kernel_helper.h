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

#ifndef TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <type_traits>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#endif
#include "tensorflow/core/util/gpu_cuda_alias.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#if GOOGLE_CUDA
#define TF_RED_WARPSIZE 32
#elif TENSORFLOW_USE_ROCM
#define TF_RED_WARPSIZE 64
#endif

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))

// Deprecated, use 'for(int i : GpuGridRange?(n))' instead.
#define GPU_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
#endif

// macro wrapper to declare dynamic shared memory
#if GOOGLE_CUDA

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  extern __shared__ __align__(ALIGN) TYPE NAME[]

#elif TENSORFLOW_USE_ROCM

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  HIP_DYNAMIC_SHARED(TYPE, NAME)

#endif

namespace tensorflow {

#if GOOGLE_CUDA
// cudaGetErrorString is available to both host and device
__host__ __device__ inline const char* GpuGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}
#elif TENSORFLOW_USE_ROCM
// hipGetErrorString is available on host side only
inline const char* GpuGetErrorString(hipError_t error) {
  return hipGetErrorString(error);
}
#endif

// Returns a raw reference to the current cuda stream. Required by a
// number of kernel calls (for which StreamInterface* does not work),
// i.e. CUB and certain cublas primitives.
inline const gpuStream_t& GetGpuStream(OpKernelContext* context) {
  const gpuStream_t* ptr = CHECK_NOTNULL(
      reinterpret_cast<const gpuStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack()));
  return *ptr;
}

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
Status GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                       size_t shared_memory_size_bytes, gpuStream_t stream,
                       Args... arguments) {
  static_assert(detail::NoneIsReference<Ts...>(),
                "Kernels with reference arguments have undefined behaviour.");
#if GOOGLE_CUDA
  auto func_ptr = absl::bit_cast<const void*>(function);
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = detail::GetArrayOfElementPointers(&args_tuple);
  auto result = cudaLaunchKernel(func_ptr, grid_dim, block_dim, arg_ptrs.data(),
                                 shared_memory_size_bytes, stream);
  if (result != cudaSuccess) {
    return errors::Internal(cudaGetErrorString(result));
  }
#elif TENSORFLOW_USE_ROCM
  hipLaunchKernelGGL(function, grid_dim, block_dim, shared_memory_size_bytes,
                     stream, std::forward<Args>(arguments)...);
#endif
  return Status::OK();
}

// Perfect forwarding to make CudaLaunchKernel available to both ROCm and CUDA
// builds
template <typename... Args>
auto CudaLaunchKernel(Args&&... args)
    -> decltype(GpuLaunchKernel(std::forward<Args>(args)...)) {
  return GpuLaunchKernel(std::forward<Args>(args)...);
}

__host__ __device__ inline tensorflow::bfloat16 GpuLdg(
    const tensorflow::bfloat16* address) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_0(mht_0_v, 302, "", "./tensorflow/core/util/gpu_kernel_helper.h", "GpuLdg");

  return Eigen::numext::bit_cast<tensorflow::bfloat16>(
      GpuLdg(reinterpret_cast<const uint16_t*>(address)));
}
// Already aliased in gpu_device_functions.h

template <typename T>
__host__ __device__ inline T ldg(const T* ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_1(mht_1_v, 312, "", "./tensorflow/core/util/gpu_kernel_helper.h", "ldg");

  return GpuLdg(ptr);
}

template <typename T>
__host__ __device__ inline const T& tf_min(const T& x, const T& y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_2(mht_2_v, 320, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_min");

  return x < y ? x : y;
}

template <typename T>
__host__ __device__ inline const T& tf_max(const T& x, const T& y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_3(mht_3_v, 328, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_max");

  return x < y ? y : x;
}

// Overloads of the above functions for float and double.
__host__ __device__ inline float tf_min(float x, float y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_4(mht_4_v, 336, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_min");

  return fminf(x, y);
}
__host__ __device__ inline double tf_min(double x, double y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_5(mht_5_v, 342, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_min");

  return fmin(x, y);
}
__host__ __device__ inline float tf_max(float x, float y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_6(mht_6_v, 348, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_max");

  return fmaxf(x, y);
}
__host__ __device__ inline double tf_max(double x, double y) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_7(mht_7_v, 354, "", "./tensorflow/core/util/gpu_kernel_helper.h", "tf_max");

  return fmax(x, y);
}

// ROCM TODO re-enable them after adding fp16 support logic
#if GOOGLE_CUDA
__device__ inline Eigen::half GpuShuffleSync(unsigned mask, Eigen::half value,
                                             int src_lane,
                                             int width = warpSize) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_8(mht_8_v, 365, "", "./tensorflow/core/util/gpu_kernel_helper.h", "GpuShuffleSync");

  return Eigen::half(
      GpuShuffleSync(mask, static_cast<uint16>(value), src_lane, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleUpSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_9(mht_9_v, 375, "", "./tensorflow/core/util/gpu_kernel_helper.h", "GpuShuffleUpSync");

  return Eigen::half(
      GpuShuffleUpSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleDownSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_10(mht_10_v, 385, "", "./tensorflow/core/util/gpu_kernel_helper.h", "GpuShuffleDownSync");

  return Eigen::half(
      GpuShuffleDownSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleXorSync(
    unsigned mask, Eigen::half value, int lane_mask, int width = warpSize) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_11(mht_11_v, 395, "", "./tensorflow/core/util/gpu_kernel_helper.h", "GpuShuffleXorSync");

  return Eigen::half(
      GpuShuffleXorSync(mask, static_cast<uint16>(value), lane_mask, width));
}
// Aliased in gpu_device_functions.h
#endif

#ifdef __CUDA_ARCH__
#define UNROLL_ON_DEVICE _Pragma("unroll")
#else
#define UNROLL_ON_DEVICE
#endif

// Represents an aligned array of N elements of T. Data pointers can be
// reinterpreted as this type to generate vectorized loads/stores in a kernel.
template <typename T, int N>
class alignas(alignof(T) * N) AlignedVector {
 public:
  typedef T value_type;
  static constexpr const int kSize = N;

  AlignedVector() = default;

  // Uniform initialization.
  __host__ __device__ explicit AlignedVector(value_type uniform) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_12(mht_12_v, 422, "", "./tensorflow/core/util/gpu_kernel_helper.h", "AlignedVector");

    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = uniform; }
  }
  // Uniform initialization with explicit conversion.
  // Note: This is required for T=Eigen::half because it only supports explicit
  // conversions from other types and its template constructor is too relaxed
  // to be able to use std::is_constructible.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  __host__ __device__ explicit AlignedVector(U uniform_u) {
    value_type uniform(uniform_u);
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = uniform; }
  }
  // Implicit conversion.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value, int>::type = 0>
  __host__ __device__ AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = other[i]; }
  }
  // Explicit conversion.
  template <typename U,
            typename std::enable_if<!std::is_convertible<U, T>::value &&
                                        std::is_constructible<T, U>::value,
                                    int>::type = 0>
  __host__ __device__ explicit AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {
      values_[i] = T(other[i]);
    }
  }

  __host__ __device__ value_type& operator[](int i) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_13(mht_13_v, 455, "", "./tensorflow/core/util/gpu_kernel_helper.h", "lambda");
 return values_[i]; }
  __host__ __device__ const value_type& operator[](int i) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_14(mht_14_v, 459, "", "./tensorflow/core/util/gpu_kernel_helper.h", "lambda");

    return values_[i];
  }

#define DEFINE_BINARY_UPDATE_OPERATOR(op)                                      \
  __host__ __device__ AlignedVector& operator op(const AlignedVector& rhs) {   \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] op rhs[i]; } \
    return *this;                                                              \
  }
  DEFINE_BINARY_UPDATE_OPERATOR(+=)
  DEFINE_BINARY_UPDATE_OPERATOR(-=)
  DEFINE_BINARY_UPDATE_OPERATOR(*=)
  DEFINE_BINARY_UPDATE_OPERATOR(/=)
#undef DEFINE_BINARY_UPDATE_OPERATOR

#define DEFINE_BINARY_OPERATOR(op)                          \
  friend __host__ __device__ AlignedVector operator op(     \
      const AlignedVector& lhs, const AlignedVector& rhs) { \
    AlignedVector ret;                                      \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {      \
      ret[i] = lhs[i] op rhs[i];                            \
    }                                                       \
    return ret;                                             \
  }
  DEFINE_BINARY_OPERATOR(+)
  DEFINE_BINARY_OPERATOR(-)
  DEFINE_BINARY_OPERATOR(*)
  DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_FUNCTION(func)                                        \
  friend __host__ __device__ AlignedVector func(const AlignedVector& lhs,   \
                                                const AlignedVector& rhs) { \
    AlignedVector ret;                                                      \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {                      \
      ret[i] = func(lhs[i], rhs[i]);                                        \
    }                                                                       \
    return ret;                                                             \
  }
  DEFINE_BINARY_FUNCTION(min)
  DEFINE_BINARY_FUNCTION(max)
#undef DEFINE_BINARY_FUNCTION

 private:
  value_type values_[N];
};

#undef UNROLL_ON_DEVICE

// Returns the maximum power-of-two alignment (in units of elements, not bytes)
// of a stride or pointer value.
inline int64_t alignment_of(int64_t element_stride) {
  return element_stride & -element_stride;
}

template <typename T>
inline int64_t alignment_of(T* ptr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helperDTh mht_15(mht_15_v, 518, "", "./tensorflow/core/util/gpu_kernel_helper.h", "alignment_of");

  const intptr_t ptr_val = reinterpret_cast<std::uintptr_t>(ptr);
  // Pointers should always be aligned to sizeof(T) bytes.
  DCHECK_EQ(ptr_val % sizeof(T), 0);
  // Note that we want the alignment in elements, not bytes.
  return alignment_of(ptr_val / sizeof(T));
}

template <typename... Args>
int64_t MinAlignmentOf(Args... args) {
  return std::min({alignment_of(args)...});
}

// Calls Functor<vec_size>()(args...) with vec_size set to the optimal GPU
// vector instruction size for type T that is <= max_vec_size. The max_vec_size
// argument should be set to the minimum alignment of all relevant parameters.
template <typename T, template <int vec_size> class Functor, typename... Args>
Status DispatchToVectorized(int64_t max_vec_size, Args&&... args) {
  constexpr const int kOptimalVecSizeBytes = 16;
  // The optimal number of (aligned) elements of T to load/store in a
  // single instruction inside a kernel.
  constexpr const int optimal_vec_size =
      (kOptimalVecSizeBytes - 1) / sizeof(T) + 1;
  int64_t vec_size = std::min((int64_t)optimal_vec_size, max_vec_size);
  if (vec_size >= 16) {
    return Functor<16>()(std::forward<Args>(args)...);
  } else if (vec_size >= 8) {
    return Functor<8>()(std::forward<Args>(args)...);
  } else if (vec_size >= 4) {
    return Functor<4>()(std::forward<Args>(args)...);
  } else if (vec_size >= 2) {
    return Functor<2>()(std::forward<Args>(args)...);
  } else {
    return Functor<1>()(std::forward<Args>(args)...);
  }
}

namespace gpu_helper {
template <typename T, typename OutType = int32>
__device__ OutType upper_bound(const T* first, OutType count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  OutType step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

template <typename T, typename OutType = int32>
__device__ OutType lower_bound(const T* first, OutType count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  OutType step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (*it < val) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

}  // namespace gpu_helper

#ifndef TENSORFLOW_USE_ROCM
namespace cuda_helper = gpu_helper;
#endif

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
