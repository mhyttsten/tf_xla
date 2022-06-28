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
==============================================================================
*/

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_GPU_SOLVERS_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_GPU_SOLVERS_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh() {
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


// This header declares the class GpuSolver, which contains wrappers of linear
// algebra solvers in the cuBlas/cuSolverDN or rocmSolver libraries for use in
// TensorFlow kernels.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <functional>
#include <vector>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#else
#include "rocm/include/hip/hip_complex.h"
#include "rocm/include/rocblas.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/rocm/rocsolver_wrapper.h"
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

#if GOOGLE_CUDA
// Type traits to get CUDA complex types from std::complex<T>.
template <typename T>
struct CUDAComplexT {
  typedef T type;
};
template <>
struct CUDAComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct CUDAComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};
// Converts pointers of std::complex<> to pointers of
// cuComplex/cuDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename CUDAComplexT<T>::type* CUDAComplex(const T* p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_0(mht_0_v, 233, "", "./tensorflow/core/util/gpu_solvers.h", "CUDAComplex");

  return reinterpret_cast<const typename CUDAComplexT<T>::type*>(p);
}
template <typename T>
inline typename CUDAComplexT<T>::type* CUDAComplex(T* p) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/util/gpu_solvers.h", "CUDAComplex");

  return reinterpret_cast<typename CUDAComplexT<T>::type*>(p);
}

// Template to give the Cublas adjoint operation for real and complex types.
template <typename T>
cublasOperation_t CublasAdjointOp() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_2(mht_2_v, 249, "", "./tensorflow/core/util/gpu_solvers.h", "CublasAdjointOp");

  return Eigen::NumTraits<T>::IsComplex ? CUBLAS_OP_C : CUBLAS_OP_T;
}
#else  // TENSORFLOW_USE_ROCM
// Type traits to get ROCm complex types from std::complex<T>.
template <typename T>
struct ROCmComplexT {
  typedef T type;
};
template <>
struct ROCmComplexT<std::complex<float>> {
  typedef rocblas_float_complex type;
};
template <>
struct ROCmComplexT<std::complex<double>> {
  typedef rocblas_double_complex type;
};
// Converts pointers of std::complex<> to pointers of
// ROCmComplex/ROCmDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename ROCmComplexT<T>::type* ROCmComplex(const T* p) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_3(mht_3_v, 272, "", "./tensorflow/core/util/gpu_solvers.h", "ROCmComplex");

  return reinterpret_cast<const typename ROCmComplexT<T>::type*>(p);
}
template <typename T>
inline typename ROCmComplexT<T>::type* ROCmComplex(T* p) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_4(mht_4_v, 279, "", "./tensorflow/core/util/gpu_solvers.h", "ROCmComplex");

  return reinterpret_cast<typename ROCmComplexT<T>::type*>(p);
}

// Type traits to get HIP complex types from std::complex<>

template <typename T>
struct HipComplexT {
  typedef T type;
};

template <>
struct HipComplexT<std::complex<float>> {
  typedef hipFloatComplex type;
};

template <>
struct HipComplexT<std::complex<double>> {
  typedef hipDoubleComplex type;
};

// Convert pointers of std::complex<> to pointers of
// hipFloatComplex/hipDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename HipComplexT<T>::type* AsHipComplex(const T* p) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_5(mht_5_v, 306, "", "./tensorflow/core/util/gpu_solvers.h", "AsHipComplex");

  return reinterpret_cast<const typename HipComplexT<T>::type*>(p);
}

template <typename T>
inline typename HipComplexT<T>::type* AsHipComplex(T* p) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_6(mht_6_v, 314, "", "./tensorflow/core/util/gpu_solvers.h", "AsHipComplex");

  return reinterpret_cast<typename HipComplexT<T>::type*>(p);
}
// Template to give the Rocblas adjoint operation for real and complex types.
template <typename T>
rocblas_operation RocblasAdjointOp() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_7(mht_7_v, 322, "", "./tensorflow/core/util/gpu_solvers.h", "RocblasAdjointOp");

  return Eigen::NumTraits<T>::IsComplex ? rocblas_operation_conjugate_transpose
                                        : rocblas_operation_transpose;
}

#if TF_ROCM_VERSION >= 40500
using gpuSolverOp_t = hipsolverOperation_t;
using gpuSolverFill_t = hipsolverFillMode_t;
using gpuSolverSide_t = hipsolverSideMode_t;
#else
using gpuSolverOp_t = rocblas_operation;
using gpuSolverFill_t = rocblas_fill;
using gpuSolverSide_t = rocblas_side;
#endif
#endif

// Container of LAPACK info data (an array of int) generated on-device by
// a GpuSolver call. One or more such objects can be passed to
// GpuSolver::CopyLapackInfoToHostAsync() along with a callback to
// check the LAPACK info data after the corresponding kernels
// finish and LAPACK info has been copied from the device to the host.
class DeviceLapackInfo;

// Host-side copy of LAPACK info.
class HostLapackInfo;

// The GpuSolver class provides a simplified templated API for the dense linear
// solvers implemented in cuSolverDN (http://docs.nvidia.com/cuda/cusolver) and
// cuBlas (http://docs.nvidia.com/cuda/cublas/#blas-like-extension/).
// An object of this class wraps static cuSolver and cuBlas instances,
// and will launch Cuda kernels on the stream wrapped by the GPU device
// in the OpKernelContext provided to the constructor.
//
// Notice: All the computational member functions are asynchronous and simply
// launch one or more Cuda kernels on the Cuda stream wrapped by the GpuSolver
// object. To check the final status of the kernels run, call
// CopyLapackInfoToHostAsync() on the GpuSolver object to set a callback that
// will be invoked with the status of the kernels launched thus far as
// arguments.
//
// Example of an asynchronous TensorFlow kernel using GpuSolver:
//
// template <typename Scalar>
// class SymmetricPositiveDefiniteSolveOpGpu : public AsyncOpKernel {
//  public:
//   explicit SymmetricPositiveDefiniteSolveOpGpu(OpKernelConstruction* context)
//       : AsyncOpKernel(context) { }
//   void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
//     // 1. Set up input and output device ptrs. See, e.g.,
//     // matrix_inverse_op.cc for a full example.
//     ...
//
//     // 2. Initialize the solver object.
//     std::unique_ptr<GpuSolver> solver(new GpuSolver(context));
//
//     // 3. Launch the two compute kernels back to back on the stream without
//     // synchronizing.
//     std::vector<DeviceLapackInfo> dev_info;
//     const int batch_size = 1;
//     dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "potrf");
//     // Compute the Cholesky decomposition of the input matrix.
//     OP_REQUIRES_OK_ASYNC(context,
//                          solver->Potrf(uplo, n, dev_matrix_ptrs, n,
//                                        dev_info.back().mutable_data()),
//                          done);
//     dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "potrs");
//     // Use the Cholesky decomposition of the input matrix to solve A X = RHS.
//     OP_REQUIRES_OK_ASYNC(context,
//                          solver->Potrs(uplo, n, nrhs, dev_matrix_ptrs, n,
//                                        dev_output_ptrs, ldrhs,
//                                        dev_info.back().mutable_data()),
//                          done);
//
//     // 4. Check the status after the computation finishes and call done.
//     solver.CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
//                                                std::move(done));
//   }
// };

template <typename Scalar>
class ScratchSpace;

class GpuSolver {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit GpuSolver(OpKernelContext* context);
  virtual ~GpuSolver();

  // Launches a memcpy of solver status data specified by dev_lapack_info from
  // device to the host, and asynchronously invokes the given callback when the
  // copy is complete. The first Status argument to the callback will be
  // Status::OK if all lapack infos retrieved are zero, otherwise an error
  // status is given. The second argument contains a host-side copy of the
  // entire set of infos retrieved, and can be used for generating detailed
  // error messages.
  // `info_checker_callback` must call the DoneCallback of any asynchronous
  // OpKernel within which `solver` is used.
  static void CheckLapackInfoAndDeleteSolverAsync(
      std::unique_ptr<GpuSolver> solver,
      const std::vector<DeviceLapackInfo>& dev_lapack_info,
      std::function<void(const Status&, const std::vector<HostLapackInfo>&)>
          info_checker_callback);

  // Simpler version to use if no special error checking / messages are needed
  // apart from checking that the Status of all calls was Status::OK.
  // `done` may be nullptr.
  static void CheckLapackInfoAndDeleteSolverAsync(
      std::unique_ptr<GpuSolver> solver,
      const std::vector<DeviceLapackInfo>& dev_lapack_info,
      AsyncOpKernel::DoneCallback done);

  // Returns a ScratchSpace. The GpuSolver object maintains a TensorReference
  // to the underlying Tensor to prevent it from being deallocated prematurely.
  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(const TensorShape& shape,
                                       const std::string& debug_info,
                                       bool on_host);
  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(int64_t size,
                                       const std::string& debug_info,
                                       bool on_host);
  // Returns a DeviceLapackInfo that will live for the duration of the
  // GpuSolver object.
  inline DeviceLapackInfo GetDeviceLapackInfo(int64_t size,
                                              const std::string& debug_info);

  // Allocates a temporary tensor that will live for the duration of the
  // GpuSolver object.
  Status allocate_scoped_tensor(DataType type, const TensorShape& shape,
                                Tensor* scoped_tensor);
  Status forward_input_or_allocate_scoped_tensor(
      gtl::ArraySlice<int> candidate_input_indices, DataType type,
      const TensorShape& shape, Tensor* input_alias_or_new_scoped_tensor);

  OpKernelContext* context() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_8(mht_8_v, 459, "", "./tensorflow/core/util/gpu_solvers.h", "context");
 return context_; }

#if TENSORFLOW_USE_ROCM
  // ====================================================================
  // Wrappers for ROCSolver start here
  //
  // The method names below
  // map to those in ROCSolver/Hipsolver, which follow the naming
  // convention in LAPACK. See rocm_solvers.cc for a mapping of
  // GpuSolverMethod to library API

  // LU factorization.
  // Computes LU factorization with partial pivoting P * A = L * U.
  template <typename Scalar>
  Status Getrf(int m, int n, Scalar* dev_A, int lda, int* dev_pivots,
               int* info);

  // Uses LU factorization to solve A * X = B.
  template <typename Scalar>
  Status Getrs(const gpuSolverOp_t trans, int n, int nrhs, Scalar* A, int lda,
               const int* dev_pivots, Scalar* B, int ldb, int* dev_lapack_info);

  template <typename Scalar>
  Status GetrfBatched(int n, Scalar** dev_A, int lda, int* dev_pivots,
                      DeviceLapackInfo* info, const int batch_count);

  // No GetrsBatched for HipSolver yet.
  template <typename Scalar>
  Status GetrsBatched(const rocblas_operation trans, int n, int nrhs,
                      Scalar** A, int lda, int* dev_pivots, Scalar** B,
                      const int ldb, int* lapack_info, const int batch_count);

  // Computes the Cholesky factorization A = L * L^H for a single matrix.
  template <typename Scalar>
  Status Potrf(gpuSolverFill_t uplo, int n, Scalar* dev_A, int lda,
               int* dev_lapack_info);

  // Computes matrix inverses for a batch of small matrices. Uses the outputs
  // from GetrfBatched. No HipSolver implementation yet
  template <typename Scalar>
  Status GetriBatched(int n, const Scalar* const host_a_dev_ptrs[], int lda,
                      const int* dev_pivots,
                      const Scalar* const host_a_inverse_dev_ptrs[], int ldainv,
                      DeviceLapackInfo* dev_lapack_info, int batch_size);

  // Computes matrix inverses for a batch of small matrices with size n < 32.
  // Returns Status::OK() if the kernel was launched successfully. Uses
  // GetrfBatched and GetriBatched
  template <typename Scalar>
  Status MatInvBatched(int n, const Scalar* const host_a_dev_ptrs[], int lda,
                       const Scalar* const host_a_inverse_dev_ptrs[],
                       int ldainv, DeviceLapackInfo* dev_lapack_info,
                       int batch_size);

  // Cholesky factorization
  // Computes the Cholesky factorization A = L * L^H for a batch of small
  // matrices.
  template <typename Scalar>
  Status PotrfBatched(gpuSolverFill_t uplo, int n,
                      const Scalar* const host_a_dev_ptrs[], int lda,
                      DeviceLapackInfo* dev_lapack_info, int batch_size);

  template <typename Scalar>
  Status Trsm(rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
              rocblas_diagonal diag, int m, int n, const Scalar* alpha,
              const Scalar* A, int lda, Scalar* B, int ldb);

  // QR factorization.
  // Computes QR factorization A = Q * R.
  template <typename Scalar>
  Status Geqrf(int m, int n, Scalar* dev_A, int lda, Scalar* dev_tau,
               int* dev_lapack_info);

  // This function performs the matrix-matrix addition/transposition
  //   C = alpha * op(A) + beta * op(B).
  template <typename Scalar>
  Status Geam(rocblas_operation transa, rocblas_operation transb, int m, int n,
              const Scalar* alpha, /* host or device pointer */
              const Scalar* A, int lda,
              const Scalar* beta, /* host or device pointer */
              const Scalar* B, int ldb, Scalar* C, int ldc);

  // Overwrite matrix C by product of C and the unitary Householder matrix Q.
  // The Householder matrix Q is represented by the output from Geqrf in dev_a
  // and dev_tau.
  template <typename Scalar>
  Status Unmqr(gpuSolverSide_t side, gpuSolverOp_t trans, int m, int n, int k,
               const Scalar* dev_a, int lda, const Scalar* dev_tau,
               Scalar* dev_c, int ldc, int* dev_lapack_info);

  // Overwrites QR factorization produced by Geqrf by the unitary Householder
  // matrix Q. On input, the Householder matrix Q is represented by the output
  // from Geqrf in dev_a and dev_tau. On output, dev_a is overwritten with the
  // first n columns of Q. Requires m >= n >= 0.
  template <typename Scalar>
  Status Ungqr(int m, int n, int k, Scalar* dev_a, int lda,
               const Scalar* dev_tau, int* dev_lapack_info);

#if TF_ROCM_VERSION >= 40500
  // Hermitian (Symmetric) Eigen decomposition.
  template <typename Scalar>
  Status Heevd(gpuSolverOp_t jobz, gpuSolverFill_t uplo, int n, Scalar* dev_A,
               int lda, typename Eigen::NumTraits<Scalar>::Real* dev_W,
               int* dev_lapack_info);
#endif

#else  // GOOGLE_CUDA
  // ====================================================================
  // Wrappers for cuSolverDN and cuBlas solvers start here.
  //
  // Apart from capitalization of the first letter, the method names below
  // map to those in cuSolverDN and cuBlas, which follow the naming
  // convention in LAPACK see, e.g.,
  // http://docs.nvidia.com/cuda/cusolver/#naming-convention

  // This function performs the matrix-matrix addition/transposition
  //   C = alpha * op(A) + beta * op(B).
  // Returns Status::OK() if the kernel was launched successfully.  See:
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam
  // NOTE(ebrevdo): Does not support in-place transpose of non-square
  // matrices.

  template <typename Scalar>
  Status Geam(cublasOperation_t transa, cublasOperation_t transb, int m, int n,
              const Scalar* alpha, /* host or device pointer */
              const Scalar* A, int lda,
              const Scalar* beta, /* host or device pointer */
              const Scalar* B, int ldb, Scalar* C,
              int ldc) const TF_MUST_USE_RESULT;

  // Computes the Cholesky factorization A = L * L^H for a single matrix.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrf
  template <typename Scalar>
  Status Potrf(cublasFillMode_t uplo, int n, Scalar* dev_A, int lda,
               int* dev_lapack_info) TF_MUST_USE_RESULT;

#if CUDA_VERSION >= 9020
  // Computes the Cholesky factorization A = L * L^H for a batch of small
  // matrices.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrfBatched
  template <typename Scalar>
  Status PotrfBatched(cublasFillMode_t uplo, int n,
                      const Scalar* const host_a_dev_ptrs[], int lda,
                      DeviceLapackInfo* dev_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;
#endif  // CUDA_VERSION >= 9020
  // LU factorization.
  // Computes LU factorization with partial pivoting P * A = L * U.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-getrf
  template <typename Scalar>
  Status Getrf(int m, int n, Scalar* dev_A, int lda, int* dev_pivots,
               int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Uses LU factorization to solve A * X = B.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-getrs
  template <typename Scalar>
  Status Getrs(cublasOperation_t trans, int n, int nrhs, const Scalar* A,
               int lda, const int* pivots, Scalar* B, int ldb,
               int* dev_lapack_info) const TF_MUST_USE_RESULT;

  // Computes partially pivoted LU factorizations for a batch of small matrices.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
  template <typename Scalar>
  Status GetrfBatched(int n, const Scalar* const host_a_dev_ptrs[], int lda,
                      int* dev_pivots, DeviceLapackInfo* dev_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;

  // Batched linear solver using LU factorization from getrfBatched.
  // Notice that lapack_info is returned on the host, as opposed to
  // most of the other functions that return it on the device. See:
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrsbatched
  template <typename Scalar>
  Status GetrsBatched(cublasOperation_t trans, int n, int nrhs,
                      const Scalar* const dev_Aarray[], int lda,
                      const int* devIpiv, const Scalar* const dev_Barray[],
                      int ldb, int* host_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;

  // Computes matrix inverses for a batch of small matrices. Uses the outputs
  // from GetrfBatched. Returns Status::OK() if the kernel was launched
  // successfully. See:
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched
  template <typename Scalar>
  Status GetriBatched(int n, const Scalar* const host_a_dev_ptrs[], int lda,
                      const int* dev_pivots,
                      const Scalar* const host_a_inverse_dev_ptrs[], int ldainv,
                      DeviceLapackInfo* dev_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;

  // Computes matrix inverses for a batch of small matrices with size n < 32.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-matinvbatched
  template <typename Scalar>
  Status MatInvBatched(int n, const Scalar* const host_a_dev_ptrs[], int lda,
                       const Scalar* const host_a_inverse_dev_ptrs[],
                       int ldainv, DeviceLapackInfo* dev_lapack_info,
                       int batch_size) TF_MUST_USE_RESULT;

  // QR factorization.
  // Computes QR factorization A = Q * R.
  // Returns Status::OK() if the kernel was launched successfully.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-geqrf
  template <typename Scalar>
  Status Geqrf(int m, int n, Scalar* dev_A, int lda, Scalar* dev_tau,
               int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Overwrite matrix C by product of C and the unitary Householder matrix Q.
  // The Householder matrix Q is represented by the output from Geqrf in dev_a
  // and dev_tau.
  // Notice: If Scalar is real, only trans=CUBLAS_OP_N or trans=CUBLAS_OP_T is
  // supported. If Scalar is complex, trans=CUBLAS_OP_N or trans=CUBLAS_OP_C is
  // supported.
  // Returns Status::OK() if the kernel was launched successfully.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-ormqr
  template <typename Scalar>
  Status Unmqr(cublasSideMode_t side, cublasOperation_t trans, int m, int n,
               int k, const Scalar* dev_a, int lda, const Scalar* dev_tau,
               Scalar* dev_c, int ldc, int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Overwrites QR factorization produced by Geqrf by the unitary Householder
  // matrix Q. On input, the Householder matrix Q is represented by the output
  // from Geqrf in dev_a and dev_tau. On output, dev_a is overwritten with the
  // first n columns of Q. Requires m >= n >= 0.
  // Returns Status::OK() if the kernel was launched successfully.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-orgqr
  template <typename Scalar>
  Status Ungqr(int m, int n, int k, Scalar* dev_a, int lda,
               const Scalar* dev_tau, int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Hermitian (Symmetric) Eigen decomposition.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-syevd
  template <typename Scalar>
  Status Heevd(cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
               Scalar* dev_A, int lda,
               typename Eigen::NumTraits<Scalar>::Real* dev_W,
               int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Singular value decomposition.
  // Returns Status::OK() if the kernel was launched successfully.
  // TODO(rmlarsen, volunteers): Add support for complex types.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-gesvd
  template <typename Scalar>
  Status Gesvd(signed char jobu, signed char jobvt, int m, int n, Scalar* dev_A,
               int lda, Scalar* dev_S, Scalar* dev_U, int ldu, Scalar* dev_VT,
               int ldvt, int* dev_lapack_info) TF_MUST_USE_RESULT;
  template <typename Scalar>
  Status GesvdjBatched(cusolverEigMode_t jobz, int m, int n, Scalar* dev_A,
                       int lda, Scalar* dev_S, Scalar* dev_U, int ldu,
                       Scalar* dev_V, int ldv, int* dev_lapack_info,
                       int batch_size);

  // Triangular solve
  // Returns Status::OK() if the kernel was launched successfully.
  // See https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsm
  template <typename Scalar>
  Status Trsm(cublasSideMode_t side, cublasFillMode_t uplo,
              cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
              const Scalar* alpha, const Scalar* A, int lda, Scalar* B,
              int ldb);

  template <typename Scalar>
  Status Trsv(cublasFillMode_t uplo, cublasOperation_t trans,
              cublasDiagType_t diag, int n, const Scalar* A, int lda, Scalar* x,
              int intcx);

  // See
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsmbatched
  template <typename Scalar>
  Status TrsmBatched(cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag, int m,
                     int n, const Scalar* alpha,
                     const Scalar* const dev_Aarray[], int lda,
                     Scalar* dev_Barray[], int ldb, int batch_size);
#endif

 private:
  OpKernelContext* context_;  // not owned.
#if GOOGLE_CUDA
  cudaStream_t cuda_stream_;
  cusolverDnHandle_t cusolver_dn_handle_;
  cublasHandle_t cublas_handle_;
#else  // TENSORFLOW_USE_ROCM
  hipStream_t hip_stream_;
  rocblas_handle rocm_blas_handle_;
#endif

  std::vector<TensorReference> scratch_tensor_refs_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuSolver);
};

// Helper class to allocate scratch memory and keep track of debug info.
// Mostly a thin wrapper around Tensor & allocate_temp.
template <typename Scalar>
class ScratchSpace {
 public:
  ScratchSpace(OpKernelContext* context, int64_t size, bool on_host)
      : ScratchSpace(context, TensorShape({size}), "", on_host) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_9(mht_9_v, 762, "", "./tensorflow/core/util/gpu_solvers.h", "ScratchSpace");
}

  ScratchSpace(OpKernelContext* context, int64_t size,
               const std::string& debug_info, bool on_host)
      : ScratchSpace(context, TensorShape({size}), debug_info, on_host) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_10(mht_10_v, 770, "", "./tensorflow/core/util/gpu_solvers.h", "ScratchSpace");
}

  ScratchSpace(OpKernelContext* context, const TensorShape& shape,
               const std::string& debug_info, bool on_host)
      : context_(context), debug_info_(debug_info), on_host_(on_host) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_11(mht_11_v, 778, "", "./tensorflow/core/util/gpu_solvers.h", "ScratchSpace");

    AllocatorAttributes alloc_attr;
    if (on_host) {
      // Allocate pinned memory on the host to avoid unnecessary
      // synchronization.
      alloc_attr.set_on_host(true);
      alloc_attr.set_gpu_compatible(true);
    }
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<Scalar>::value, shape,
                                       &scratch_tensor_, alloc_attr));
  }

  virtual ~ScratchSpace() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_12(mht_12_v, 793, "", "./tensorflow/core/util/gpu_solvers.h", "~ScratchSpace");
}

  Scalar* mutable_data() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_13(mht_13_v, 798, "", "./tensorflow/core/util/gpu_solvers.h", "mutable_data");

    return scratch_tensor_.template flat<Scalar>().data();
  }
  const Scalar* data() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_14(mht_14_v, 804, "", "./tensorflow/core/util/gpu_solvers.h", "data");

    return scratch_tensor_.template flat<Scalar>().data();
  }
  Scalar& operator()(int64_t i) {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  const Scalar& operator()(int64_t i) const {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  int64_t bytes() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_15(mht_15_v, 816, "", "./tensorflow/core/util/gpu_solvers.h", "bytes");
 return scratch_tensor_.TotalBytes(); }
  int64_t size() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_16(mht_16_v, 820, "", "./tensorflow/core/util/gpu_solvers.h", "size");
 return scratch_tensor_.NumElements(); }
  const std::string& debug_info() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_17(mht_17_v, 824, "", "./tensorflow/core/util/gpu_solvers.h", "debug_info");
 return debug_info_; }

  Tensor& tensor() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_18(mht_18_v, 829, "", "./tensorflow/core/util/gpu_solvers.h", "tensor");
 return scratch_tensor_; }
  const Tensor& tensor() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_19(mht_19_v, 833, "", "./tensorflow/core/util/gpu_solvers.h", "tensor");
 return scratch_tensor_; }

  // Returns true if this ScratchSpace is in host memory.
  bool on_host() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_20(mht_20_v, 839, "", "./tensorflow/core/util/gpu_solvers.h", "on_host");
 return on_host_; }

 protected:
  OpKernelContext* context() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_21(mht_21_v, 845, "", "./tensorflow/core/util/gpu_solvers.h", "context");
 return context_; }

 private:
  OpKernelContext* context_;  // not owned
  const std::string debug_info_;
  const bool on_host_;
  Tensor scratch_tensor_;
};

class HostLapackInfo : public ScratchSpace<int> {
 public:
  HostLapackInfo(OpKernelContext* context, int64_t size,
                 const std::string& debug_info)
      : ScratchSpace<int>(context, size, debug_info, /* on_host */ true) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_22(mht_22_v, 862, "", "./tensorflow/core/util/gpu_solvers.h", "HostLapackInfo");
}
};

class DeviceLapackInfo : public ScratchSpace<int> {
 public:
  DeviceLapackInfo(OpKernelContext* context, int64_t size,
                   const std::string& debug_info)
      : ScratchSpace<int>(context, size, debug_info, /* on_host */ false) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_23(mht_23_v, 873, "", "./tensorflow/core/util/gpu_solvers.h", "DeviceLapackInfo");
}

  // Allocates a new scratch space on the host and launches a copy of the
  // contents of *this to the new scratch space. Sets success to true if
  // the copy kernel was launched successfully.
  HostLapackInfo CopyToHost(bool* success) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_24(mht_24_v, 881, "", "./tensorflow/core/util/gpu_solvers.h", "CopyToHost");

    CHECK(success != nullptr);
    HostLapackInfo copy(context(), size(), debug_info());
    auto stream = context()->op_device_context()->stream();
    se::DeviceMemoryBase wrapped_src(
        static_cast<void*>(const_cast<int*>(this->data())));
    *success =
        stream->ThenMemcpy(copy.mutable_data(), wrapped_src, this->bytes())
            .ok();
    return copy;
  }
};

template <typename Scalar>
ScratchSpace<Scalar> GpuSolver::GetScratchSpace(const TensorShape& shape,
                                                const std::string& debug_info,
                                                bool on_host) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_25(mht_25_v, 901, "", "./tensorflow/core/util/gpu_solvers.h", "GpuSolver::GetScratchSpace");

  ScratchSpace<Scalar> new_scratch_space(context_, shape, debug_info, on_host);
  scratch_tensor_refs_.emplace_back(new_scratch_space.tensor());
  return std::move(new_scratch_space);
}

template <typename Scalar>
ScratchSpace<Scalar> GpuSolver::GetScratchSpace(int64_t size,
                                                const std::string& debug_info,
                                                bool on_host) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_26(mht_26_v, 914, "", "./tensorflow/core/util/gpu_solvers.h", "GpuSolver::GetScratchSpace");

  return GetScratchSpace<Scalar>(TensorShape({size}), debug_info, on_host);
}

inline DeviceLapackInfo GpuSolver::GetDeviceLapackInfo(
    int64_t size, const std::string& debug_info) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_solversDTh mht_27(mht_27_v, 923, "", "./tensorflow/core/util/gpu_solvers.h", "GpuSolver::GetDeviceLapackInfo");

  DeviceLapackInfo new_dev_info(context_, size, debug_info);
  scratch_tensor_refs_.emplace_back(new_dev_info.tensor());
  return new_dev_info;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_GPU_SOLVERS_H_
