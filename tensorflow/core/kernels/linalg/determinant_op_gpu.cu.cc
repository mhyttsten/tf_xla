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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/linalg/determinant_op.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
namespace {
__device__ int PermutationOrder(int n, const int* __restrict__ pivots) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_op_gpuDTcuDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/linalg/determinant_op_gpu.cu.cc", "PermutationOrder");

  // Compute the order of the permutation from the number of transpositions
  // encoded in the pivot array, see:
  // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=340
  int order = 0;
  for (int i = 0; i < n - 1; ++i) {
    // Notice: Internally, the cuBlas code uses Fortran convention (1-based)
    // indexing so we expect pivots[i] == i + 1 for rows that were not moved.
    order += pivots[i] != (i + 1);
  }
  return order;
}
}  // namespace

// This kernel computes either determinant or log_abs_determinant, depending
// on the value of the template parameter. If compute_log_abs_det is false,
// the sign argument is ignored.
template <typename Scalar, bool compute_log_abs_det = true>
__global__ void DeterminantFromPivotedLUKernel(
    int nthreads, int n, const Scalar* __restrict__ lu_factor,
    const int* __restrict__ all_pivots, Scalar* __restrict__ sign,
    Scalar* __restrict__ log_abs_det) {
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  const int matrix_size = n * n;
  const int stride = n + 1;
  // We only parallelize over batches here. Performance is not critical,
  // since this cheap O(n) kernel always follows an O(n^3) LU factorization.
  // The main purpose is to avoid having to copy the LU decomposition to
  // host memory.
  GPU_1D_KERNEL_LOOP(o_idx, nthreads) {
    // Initialize sign to (-1)^order.
    const int order = PermutationOrder(n, all_pivots + o_idx * n);
    Scalar prod_sign = order % 2 ? Scalar(-1) : Scalar(1);
    RealScalar sum_log_abs_det = RealScalar(0);
    int i_idx = matrix_size * o_idx;
    for (int i = 0; i < n; ++i, i_idx += stride) {
      const RealScalar abs_i = Eigen::numext::abs(lu_factor[i_idx]);
      sum_log_abs_det += Eigen::numext::log(abs_i);
      prod_sign = prod_sign * (lu_factor[i_idx] / abs_i);
    }
    if (!Eigen::numext::isfinite(sum_log_abs_det)) {
      prod_sign = Scalar(0);
      sum_log_abs_det = sum_log_abs_det > 0 ? -Eigen::numext::log(RealScalar(0))
                                            : Eigen::numext::log(RealScalar(0));
    }
    if (compute_log_abs_det) {
      sign[o_idx] = prod_sign;
      log_abs_det[o_idx] = Scalar(sum_log_abs_det);
    } else {
      log_abs_det[o_idx] = prod_sign * Eigen::numext::exp(sum_log_abs_det);
    }
  }
}

template <typename Scalar>
struct DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  const int* pivots, typename TTypes<Scalar, 1>::Tensor output,
                  int* info) {
    const int64 num_matrices = output.size();
    const int64 n = lu_factor.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(num_matrices, device);

    TF_CHECK_OK(GpuLaunchKernel(
        DeterminantFromPivotedLUKernel<Scalar, /*compute_log_abs_det=*/false>,
        config.block_count, config.thread_per_block, 0, device.stream(),
        config.virtual_thread_count, n, lu_factor.data(), pivots, nullptr,
        output.data()));
  }
};

template struct DeterminantFromPivotedLUFunctor<GPUDevice, float>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, double>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex64>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex128>;

template <typename Scalar>
struct LogDeterminantFromPivotedLUFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  const int* pivots, typename TTypes<Scalar, 1>::Tensor sign,
                  typename TTypes<Scalar, 1>::Tensor log_abs_det) {
    const int64 num_matrices = sign.size();
    const int64 n = lu_factor.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(num_matrices, device);
    TF_CHECK_OK(GpuLaunchKernel(
        DeterminantFromPivotedLUKernel<Scalar, /*compute_log_abs_det=*/true>,
        config.block_count, config.thread_per_block, 0, device.stream(),
        config.virtual_thread_count, n, lu_factor.data(), pivots, sign.data(),
        log_abs_det.data()));
  }
};

template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, float>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, double>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, complex64>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
