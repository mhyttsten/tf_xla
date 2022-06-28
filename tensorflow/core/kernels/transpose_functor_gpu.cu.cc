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
class MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc() {
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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

// TODO(yangzihao): Remove the dependency of conv_2d.h once we move all
// GPU util functions and transpose kernels into separate files.
#include "tensorflow/core/kernels/conv_2d.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {
namespace internal {

template <typename T, bool conjugate>
__global__ void TransposeKernel(int nthreads, const T* __restrict__ src,
                                const int32* __restrict__ buf,
                                const int32 ndims, T* __restrict__ dst) {
  const int32* in_strides = buf;
  const int32* out_strides = buf + ndims;
  const int32* perm = buf + ndims * 2;
  GPU_1D_KERNEL_LOOP(o_idx, nthreads) {
    int32 i_idx = 0;
    int32 t = o_idx;
    for (int32 i = 0; i < ndims; ++i) {
      const int32 ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    if (conjugate) {
      dst[o_idx] = Eigen::numext::conj(ldg(src + i_idx));
    } else {
      dst[o_idx] = ldg(src + i_idx);
    }
  }
}

template <typename T, bool conjugate>
void TransposeSimple(const GPUDevice& d, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out) {
  // Ensures we can use 32-bit index.
  const int64 nelem = in.NumElements();
  CHECK_LT(nelem, kint32max) << "Tensor too large to transpose on GPU";
  // Pack strides and permutation into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 24> host_buf(ndims * 3);
  gtl::InlinedVector<int32, 8> in_strides = ComputeStride<int32>(in.shape());
  gtl::InlinedVector<int32, 8> out_strides = ComputeStride<int32>(out->shape());
  // Dimension permutation.
  for (int i = 0; i < ndims; ++i) {
    host_buf[i] = in_strides[i];
    host_buf[ndims + i] = out_strides[i];
    host_buf[ndims * 2 + i] = perm[i];
  }
  // Copies the input strides, output strides and permutation to the device.
  auto num_bytes = sizeof(int32) * host_buf.size();
  auto dev_buf = d.allocate(num_bytes);
  // NOTE: host_buf is not allocated by GpuHostAllocator, and
  // therefore we are doing a sync copy effectively.
  d.memcpyHostToDevice(dev_buf, host_buf.data(), num_bytes);
  // Launch kernel to q[...] = p[...].
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  TF_CHECK_OK(GpuLaunchKernel(
      TransposeKernel<T, conjugate>, cfg.block_count, cfg.thread_per_block, 0,
      d.stream(), cfg.virtual_thread_count, p,
      reinterpret_cast<const int32*>(dev_buf), ndims, q));
  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_buf);
}

// TransposeUsingTile tries to reduce the dimension of the input tensor to 3 and
// then call special kernels to swap either dimension 1 and dimension 2 or
// dimension 0 and dimension 2. It returns true if the operation is success,
// false otherwise.
template <typename T, bool conjugate = false>
struct TransposeUsingTile {
  static bool run(const Eigen::GpuDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_0(mht_0_v, 268, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "run");

    // First try to reduce the dimensions of the input tensor.
    TransposePermsVec new_perm;
    TransposeDimsVec new_dims;
    ReduceTransposeDimensions(in.shape(), perm, &new_perm, &new_dims);

    // Only use special GPU kernel when dimension is 2 or 3.
    int dims = new_dims.size();
    if (dims < 2 || dims > 3) return false;
    auto in_data = reinterpret_cast<const T*>(in.tensor_data().data());
    auto out_data =
        reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data()));
    switch (dims) {
      case 2:
        if (new_perm[0] == 1 && new_perm[1] == 0) {
          // Add the first dimension size as 1.
          new_dims.insert(new_dims.begin(), 1);
          tensorflow::functor::SwapDimension1And2InTensor3<GPUDevice, T,
                                                           conjugate>()(
              d, in_data, new_dims, out_data);
          return true;
        }
        break;
      case 3:
        if (new_perm == TransposePermsVec({0, 2, 1})) {
          tensorflow::functor::SwapDimension1And2InTensor3<GPUDevice, T,
                                                           conjugate>()(
              d, in_data, new_dims, out_data);
          return true;
        } else if (new_perm == TransposePermsVec({2, 1, 0})) {
          tensorflow::functor::SwapDimension0And2InTensor3<GPUDevice, T,
                                                           conjugate>()(
              d, in_data, new_dims, out_data);
          return true;
        } else {
          // do not handle other 3D permutations
          return false;
        }
        break;
      default:
        return false;
    }
    return false;
  }
};

template <bool conjugate>
struct TransposeUsingTile<complex64, conjugate> {
  static bool run(const Eigen::GpuDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_1(mht_1_v, 320, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "run");

    if (!conjugate) {
      return TransposeUsingTile<uint64>::run(d, in, perm, out);
    } else {
      return TransposeUsingTile<float2, true>::run(d, in, perm, out);
    }
  }
};

template <bool conjugate>
struct TransposeUsingTile<complex128, conjugate> {
  static bool run(const Eigen::GpuDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_2(mht_2_v, 335, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "run");

    if (!conjugate) {
      return TransposeUsingTile<float4>::run(d, in, perm, out);
    } else {
      return TransposeUsingTile<double2, true>::run(d, in, perm, out);
    }
  }
};

}  // namespace internal

// Transpose kernel specialized for GPU Device.
#define HANDLE_DIM(DIM)                                                      \
  case DIM:                                                                  \
    internal::TransposeUsingEigen<GPUDevice, T, DIM>(d, in, perm, conjugate, \
                                                     out);                   \
    break

template <typename T, bool conjugate>
struct Transpose<GPUDevice, T, conjugate> {
  static void run(const GPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_3(mht_3_v, 359, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "run");

    if (in.dims() < 2) return;
    if (internal::TransposeUsingTile<T, conjugate>::run(d, in, perm, out)) {
      return;
    }

    switch (in.dims()) {
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);
      default:
        internal::TransposeSimple<T, conjugate>(d, in, perm, out);
        break;
    }
  }
};

#undef HANDLE_DIM

template <bool conjugate>
struct Transpose<GPUDevice, tstring, conjugate> {
  static void run(const GPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_4(mht_4_v, 388, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "run");

    LOG(FATAL) << "Transpose of DT_STRING tensor not supported on GPU.";
  }
};

// Explicit instantiation.
template struct Transpose<GPUDevice, tstring, false>;

template <>
Status DoTranspose(const GPUDevice& device, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_5(mht_5_v, 401, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "DoTranspose");

  return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/false, out);
}
template <>
Status DoConjugateTranspose(const GPUDevice& device, const Tensor& in,
                            const gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_6(mht_6_v, 409, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "DoConjugateTranspose");

  return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/true, out);
}
template <>
Status DoMatrixTranspose(const GPUDevice& device, const Tensor& in,
                         Tensor* out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_7(mht_7_v, 417, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "DoMatrixTranspose");

  return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/false, out);
}
template <>
Status DoConjugateMatrixTranspose(const GPUDevice& device, const Tensor& in,
                                  Tensor* out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_functor_gpuDTcuDTcc mht_8(mht_8_v, 425, "", "./tensorflow/core/kernels/transpose_functor_gpu.cu.cc", "DoConjugateMatrixTranspose");

  return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/true, out);
}

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
