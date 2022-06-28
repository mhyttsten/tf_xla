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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_gpuDTcuDTcc() {
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

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct OutOfBoundsValue {
  __host__ __device__ static T value() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_gpuDTcuDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/sparse_tensor_dense_matmul_op_gpu.cu.cc", "value");

    return Eigen::NumTraits<T>::quiet_NaN();
  }
};

template <typename T>
struct OutOfBoundsValue<std::complex<T>> {
  __host__ __device__ static std::complex<T> value() {
    return std::complex<T>(OutOfBoundsValue<T>::value(),
                           OutOfBoundsValue<T>::value());
  }
};

template <typename T, typename Tsum, typename Tindices, bool ADJ_A, bool ADJ_B>
__global__ void SparseTensorDenseMatMulKernel(
    int nnz, int m, int b_rows, int b_cols, int p,
    const Tindices* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, Tsum* __restrict__ out) {
  // out_{ij} = sum_k {a_ik b_kj}
  // out = A * B', out_{ij} = sum_k {a_ik (b')_kj}; b'_{kj} = b_{jk}
  const int n = (ADJ_B) ? b_cols : b_rows;
  GPU_1D_KERNEL_LOOP(index, nnz * p) {
    const int a_ix = index / p;
    const int j = index % p;
    const int i = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 1 : 0));
    const int k = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 0 : 1));
    if (!FastBoundsCheck(i, m)) {
      continue;  // Nowhere to signal an error :(
    }
    // out[i, j]
    Tsum* out_location = out + i * p + j;
    if (!FastBoundsCheck(k, n)) {
      GpuAtomicAdd(out_location, OutOfBoundsValue<Tsum>::value());
      continue;
    }

    // a_value == (ADJ_A) ? conj(a[k, i]) : a[i, k]
    const T a_input = ldg(a_values + a_ix);
    const T a_value = ADJ_A ? Eigen::numext::conj(a_input) : a_input;

    // b_value == (ADJ_B) ? conj(b[j, k]) : b[k, j]
    const T b_input = ldg(b + ((ADJ_B) ? j * b_cols + k : k * b_cols + j));
    const T b_value = ADJ_B ? Eigen::numext::conj(b_input) : b_input;
    GpuAtomicAdd(out_location,
                 static_cast<Tsum>(a_value) * static_cast<Tsum>(b_value));
  }
}

namespace functor {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B> {
  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* ctx, typename TTypes<T>::Matrix out,
          typename TTypes<Tindices>::ConstMatrix a_indices,
          typename TTypes<T>::ConstVec a_values,
          typename TTypes<T>::ConstMatrix b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_gpuDTcuDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/kernels/sparse_tensor_dense_matmul_op_gpu.cu.cc", "Compute");

    int nnz = a_values.size();
    // out = A * B, A is [m x n] and B is [n x p], out is [m x p]
    int m = out.dimension(0);
    int p = out.dimension(1);
    int b_rows = b.dimension(0);
    int b_cols = b.dimension(1);

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    using Tsum = typename SumType<T>::type;
    Tsum* maybe_temp_out_data = nullptr;
    Tensor temp_out_t;
    bool sum_type_is_different = !std::is_same<T, Tsum>::value;
    if (sum_type_is_different) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DataTypeToEnum<Tsum>::value,
          TensorShape({out.dimension(0), out.dimension(1)}), &temp_out_t));
      auto temp_out = temp_out_t.matrix<Tsum>();
      maybe_temp_out_data = temp_out.data();
      temp_out.device(d) = temp_out.constant(Tsum(0));
    } else {
      // Note: The reinterpret cast is only required to avoid a compilation
      // error; it is only used if Tsum == T.
      maybe_temp_out_data = reinterpret_cast<Tsum*>(out.data());
      out.device(d) = out.constant(T(0));
    }

    // TODO(ebrevdo): Should this be alpha * nnz instead of
    // out.size()?  Perhaps p * nnz ?
    GpuLaunchConfig config = GetGpuLaunchConfig(p * nnz, d);

    if (OpDeterminismRequired()) {
      return errors::Unimplemented(
          "A deterministic GPU implementation of "
          "SparseTensorDenseMatmulOp is not currently available.");
    }

    TF_CHECK_OK(GpuLaunchKernel(
        SparseTensorDenseMatMulKernel<T, Tsum, Tindices, ADJ_A, ADJ_B>,
        config.block_count, config.thread_per_block, 0, d.stream(), nnz, m,
        b_rows, b_cols, p, a_indices.data(), a_values.data(), b.data(),
        maybe_temp_out_data));

    if (sum_type_is_different) {
      out.device(d) = temp_out_t.matrix<Tsum>().template cast<T>();
    }

    return Status::OK();
  }
};

}  // namespace functor

#define DEFINE(T, Tindices)                                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, false>;               \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, true>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, false>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, true>;

#define DEFINE_ALL_INDEX_TYPES(T) \
  DEFINE(T, int32);               \
  DEFINE(T, int64)

DEFINE_ALL_INDEX_TYPES(Eigen::half);
DEFINE_ALL_INDEX_TYPES(float);
DEFINE_ALL_INDEX_TYPES(double);
DEFINE_ALL_INDEX_TYPES(complex64);
DEFINE_ALL_INDEX_TYPES(complex128);

#undef DEFINE_ALL_INDEX_TYPES
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
