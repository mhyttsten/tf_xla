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
class MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <complex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

namespace {

// A set of initialized handles to the underlying ROCm libraries used by
// GpuSparse. We maintain one such set of handles per unique stream.
class HipSparseHandles {
 public:
  explicit HipSparseHandles(hipStream_t stream)
      : initialized_(false), stream_(stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/util/rocm_sparse.cc", "HipSparseHandles");
}

  HipSparseHandles(HipSparseHandles&& rhs)
      : initialized_(rhs.initialized_),
        stream_(std::move(rhs.stream_)),
        hipsparse_handle_(rhs.hipsparse_handle_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/util/rocm_sparse.cc", "HipSparseHandles");

    rhs.initialized_ = false;
  }

  HipSparseHandles& operator=(HipSparseHandles&& rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/util/rocm_sparse.cc", "=");

    if (this == &rhs) return *this;
    Release();
    stream_ = std::move(rhs.stream_);
    hipsparse_handle_ = std::move(rhs.hipsparse_handle_);
    initialized_ = rhs.initialized_;
    rhs.initialized_ = false;
    return *this;
  }

  ~HipSparseHandles() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/util/rocm_sparse.cc", "~HipSparseHandles");
 Release(); }

  Status Initialize() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_4(mht_4_v, 250, "", "./tensorflow/core/util/rocm_sparse.cc", "Initialize");

    if (initialized_) return Status::OK();
    TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseCreate(&hipsparse_handle_));
    TF_RETURN_IF_GPUSPARSE_ERROR(
        wrap::hipsparseSetStream(hipsparse_handle_, stream_));
    initialized_ = true;
    return Status::OK();
  }

  hipsparseHandle_t& handle() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_5(mht_5_v, 262, "", "./tensorflow/core/util/rocm_sparse.cc", "handle");

    DCHECK(initialized_);
    return hipsparse_handle_;
  }

  const hipsparseHandle_t& handle() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_6(mht_6_v, 270, "", "./tensorflow/core/util/rocm_sparse.cc", "handle");

    DCHECK(initialized_);
    return hipsparse_handle_;
  }

 private:
  void Release() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/util/rocm_sparse.cc", "Release");

    if (initialized_) {
      // This should never return anything other than success
      auto err = wrap::hipsparseDestroy(hipsparse_handle_);
      DCHECK(err == HIPSPARSE_STATUS_SUCCESS)
          << "Failed to destroy hipSPARSE instance.";
      initialized_ = false;
    }
  }
  bool initialized_;
  hipStream_t stream_;
  hipsparseHandle_t hipsparse_handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(HipSparseHandles);
};

// TODO(ebrevdo): Replace global mutex guarding CudaSparseHandles
// lookup with one of:
//    1. Adding the handle to the CudaStream structure; do the lookup there.
//    2. Add a thread-local cusparse, set it to the current stream
//       upon each call.
// #1 seems like the cleanest option but will need to wait until this
// is moved into TF core.
static mutex handle_map_mutex(LINKER_INITIALIZED);

using HandleMap = std::unordered_map<hipStream_t, HipSparseHandles>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_8(mht_8_v, 311, "", "./tensorflow/core/util/rocm_sparse.cc", "GetHandleMapSingleton");

  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

GpuSparse::GpuSparse(OpKernelContext* context)
    : initialized_(false), context_(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_9(mht_9_v, 322, "", "./tensorflow/core/util/rocm_sparse.cc", "GpuSparse::GpuSparse");

  auto hip_stream_ptr =
      reinterpret_cast<const hipStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack());
  DCHECK(hip_stream_ptr);
  gpu_stream_ = *hip_stream_ptr;
}

Status GpuSparse::Initialize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_10(mht_10_v, 335, "", "./tensorflow/core/util/rocm_sparse.cc", "GpuSparse::Initialize");

  HandleMap* handle_map = GetHandleMapSingleton();
  DCHECK(handle_map);
  mutex_lock lock(handle_map_mutex);
  auto it = handle_map->find(gpu_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating GpuSparse handles for stream " << gpu_stream_;
    // Previously unseen ROCm stream. Initialize a set of ROCm sparse library
    // handles for it.
    HipSparseHandles new_handles(gpu_stream_);
    TF_RETURN_IF_ERROR(new_handles.Initialize());
    it = handle_map->insert(std::make_pair(gpu_stream_, std::move(new_handles)))
             .first;
  }
  gpusparse_handle_ = &it->second.handle();
  initialized_ = true;
  return Status::OK();
}

#define TF_CALL_HIPSPARSE_DTYPES(m)          \
  m(float, ROCM_R_32F) m(double, ROCM_R_64F) \
      m(std::complex<float>, ROCM_C_32F) m(std::complex<double>, ROCM_C_64F)

// Macro that specializes a sparse method for all 4 standard
// numeric types.
#define TF_CALL_HIP_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct hipsparse method names.
#define SPARSE_FN(method, sparse_prefix) wrap::hipsparse##sparse_prefix##method
#define BUFSIZE_FN(method, sparse_prefix) \
  wrap::hipsparse##sparse_prefix##method##_bufferSizeExt

Status GpuSparse::Coo2csr(const int* cooRowInd, int nnz, int m,
                          int* csrRowPtr) const {
  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(
      wrap::hipsparseXcoo2csr(*gpusparse_handle_, cooRowInd, nnz, m, csrRowPtr,
                              HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

Status GpuSparse::Csr2coo(const int* csrRowPtr, int nnz, int m,
                          int* cooRowInd) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_11(mht_11_v, 381, "", "./tensorflow/core/util/rocm_sparse.cc", "GpuSparse::Csr2coo");

  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(
      wrap::hipsparseXcsr2coo(*gpusparse_handle_, csrRowPtr, nnz, m, cooRowInd,
                              HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

#if TF_ROCM_VERSION < 40200
template <typename Scalar, typename SparseFnT>
static inline Status CsrmmImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t hipsparse_handle,
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, int nnz, const Scalar* alpha_host, const hipsparseMatDescr_t descrA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* B, int ldb,
    const Scalar* beta_host, Scalar* C, int ldc) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      hipsparse_handle, transA, transB, m, n, k, nnz, AsHipComplex(alpha_host),
      descrA, AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(B), ldb, AsHipComplex(beta_host), AsHipComplex(C), ldc));
  return Status::OK();
}

#define CSRMM_INSTANCE(Scalar, sparse_prefix)                                 \
  template <>                                                                 \
  Status GpuSparse::Csrmm<Scalar>(                                            \
      hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, \
      int k, int nnz, const Scalar* alpha_host,                               \
      const hipsparseMatDescr_t descrA, const Scalar* csrSortedValA,          \
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,               \
      const Scalar* B, int ldb, const Scalar* beta_host, Scalar* C, int ldc)  \
      const {                                                                 \
    DCHECK(initialized_);                                                     \
    return CsrmmImpl(SPARSE_FN(csrmm2, sparse_prefix), context_,              \
                     *gpusparse_handle_, transA, transB, m, n, k, nnz,        \
                     alpha_host, descrA, csrSortedValA, csrSortedRowPtrA,     \
                     csrSortedColIndA, B, ldb, beta_host, C, ldc);            \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRMM_INSTANCE);

#else

#define SPMM_BUFFERSIZE_INSTANCE(Scalar, dtype)                               \
  template <>                                                                 \
  Status GpuSparse::SpMMBufferSize<Scalar>(                                   \
      hipsparseOperation_t transA, hipsparseOperation_t transB,               \
      const Scalar* alpha, const hipsparseSpMatDescr_t matA,                  \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                   \
      gpusparseDnMatDescr_t matC, hipsparseSpMMAlg_t alg, size_t* bufferSize) \
      const {                                                                 \
    DCHECK(initialized_);                                                     \
    TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseSpMM_bufferSize(              \
        *gpusparse_handle_, transA, transB, alpha, matA, matB, beta, matC,    \
        dtype, alg, bufferSize));                                             \
    return Status::OK();                                                      \
  }

TF_CALL_HIPSPARSE_DTYPES(SPMM_BUFFERSIZE_INSTANCE);

#define SPMM_INSTANCE(Scalar, dtype)                                         \
  template <>                                                                \
  Status GpuSparse::SpMM<Scalar>(                                            \
      hipsparseOperation_t transA, hipsparseOperation_t transB,              \
      const Scalar* alpha, const hipsparseSpMatDescr_t matA,                 \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                  \
      gpusparseDnMatDescr_t matC, hipsparseSpMMAlg_t alg, int8* buffer)      \
      const {                                                                \
    DCHECK(initialized_);                                                    \
    TF_RETURN_IF_GPUSPARSE_ERROR(                                            \
        wrap::hipsparseSpMM(*gpusparse_handle_, transA, transB, alpha, matA, \
                            matB, beta, matC, dtype, alg, buffer));          \
    return Status::OK();                                                     \
  }

TF_CALL_HIPSPARSE_DTYPES(SPMM_INSTANCE);

#endif

template <typename Scalar, typename SparseFnT>
static inline Status CsrmvImpl(SparseFnT op, OpKernelContext* context,
                               hipsparseHandle_t hipsparse_handle,
                               hipsparseOperation_t transA, int m, int n,
                               int nnz, const Scalar* alpha_host,
                               const hipsparseMatDescr_t descrA,
                               const Scalar* csrSortedValA,
                               const int* csrSortedRowPtrA,
                               const int* csrSortedColIndA, const Scalar* x,
                               const Scalar* beta_host, Scalar* y) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(hipsparse_handle, transA, m, n, nnz, AsHipComplex(alpha_host), descrA,
         AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         AsHipComplex(x), AsHipComplex(beta_host), AsHipComplex(y)));
  return Status::OK();
}

// TODO(ebrevdo,rmlarsen): Use csrmv_mp for all cases when available in CUDA 9.
#define CSRMV_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                \
  Status GpuSparse::Csrmv<Scalar>(                                           \
      hipsparseOperation_t transA, int m, int n, int nnz,                    \
      const Scalar* alpha_host, const hipsparseMatDescr_t descrA,            \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,              \
      const int* csrSortedColIndA, const Scalar* x, const Scalar* beta_host, \
      Scalar* y) const {                                                     \
    DCHECK(initialized_);                                                    \
    return CsrmvImpl(SPARSE_FN(csrmv, sparse_prefix), context_,              \
                     *gpusparse_handle_, transA, m, n, nnz, alpha_host,      \
                     descrA, csrSortedValA, csrSortedRowPtrA,                \
                     csrSortedColIndA, x, beta_host, y);                     \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRMV_INSTANCE);

Status GpuSparse::CsrgemmNnz(
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, const hipsparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const hipsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_12(mht_12_v, 505, "", "./tensorflow/core/util/rocm_sparse.cc", "GpuSparse::CsrgemmNnz");

  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
  TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseXcsrgemmNnz(
      *gpusparse_handle_, transA, transB, m, n, k, descrA, nnzA,
      csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
      csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status CsrgemmImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t hipsparse_handle,
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, const hipsparseMatDescr_t descrA, int nnzA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const hipsparseMatDescr_t descrB, int nnzB,
    const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      hipsparse_handle, transA, transB, m, n, k, descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB,
      nnzB, AsHipComplex(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB,
      descrC, AsHipComplex(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC));
  return Status::OK();
}

#define CSRGEMM_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                  \
  Status GpuSparse::Csrgemm<Scalar>(                                           \
      hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,  \
      int k, const hipsparseMatDescr_t descrA, int nnzA,                       \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,                \
      const int* csrSortedColIndA, const hipsparseMatDescr_t descrB, int nnzB, \
      const Scalar* csrSortedValB, const int* csrSortedRowPtrB,                \
      const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,           \
      Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) {   \
    DCHECK(initialized_);                                                      \
    return CsrgemmImpl(SPARSE_FN(csrgemm, sparse_prefix), context_,            \
                       *gpusparse_handle_, transA, transB, m, n, k, descrA,    \
                       nnzA, csrSortedValA, csrSortedRowPtrA,                  \
                       csrSortedColIndA, descrB, nnzB, csrSortedValB,          \
                       csrSortedRowPtrB, csrSortedColIndB, descrC,             \
                       csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);     \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEMM_INSTANCE);

#if TF_ROCM_VERSION >= 40200

template <typename Scalar, typename BufferSizeFnT, typename SparseFnT>
static inline Status Csru2csrImpl(SparseFnT op, BufferSizeFnT buffer_size_op,
                                  OpKernelContext* context,
                                  hipsparseHandle_t hipsparse_handle, int m,
                                  int n, int nnz,
                                  const hipsparseMatDescr_t descrA,
                                  Scalar* csrVal, const int* csrRowPtr,
                                  int* csrColInd) {
  GpuSparseCsrSortingConversionInfo info;
  TF_RETURN_IF_ERROR(info.Initialize());

  size_t pBufferSizeInBytes = 0;

  TF_RETURN_IF_GPUSPARSE_ERROR(
      buffer_size_op(hipsparse_handle, m, n, nnz, AsHipComplex(csrVal),
                     csrRowPtr, csrColInd, info.info(), &pBufferSizeInBytes));

  Tensor pBuffer_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(pBufferSizeInBytes)}),
      &pBuffer_t));
  auto pBuffer = pBuffer_t.flat<int8>();
  DCHECK(pBuffer.data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(op(hipsparse_handle, m, n, nnz, descrA,
                                  AsHipComplex(csrVal), csrRowPtr, csrColInd,
                                  info.info(), pBuffer.data()));

  return Status::OK();
}

#define CSRU2CSR_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                  \
  Status GpuSparse::Csru2csr<Scalar>(                                          \
      int m, int n, int nnz, const hipsparseMatDescr_t descrA, Scalar* csrVal, \
      const int* csrRowPtr, int* csrColInd) {                                  \
    DCHECK(initialized_);                                                      \
    return Csru2csrImpl(SPARSE_FN(csru2csr, sparse_prefix),                    \
                        BUFSIZE_FN(csru2csr, sparse_prefix), context_,         \
                        *gpusparse_handle_, m, n, nnz, descrA, csrVal,         \
                        csrRowPtr, csrColInd);                                 \
  }

TF_CALL_LAPACK_TYPES(CSRU2CSR_INSTANCE);

#endif

template <typename Scalar, typename SparseFnT>
static inline Status Csr2cscImpl(SparseFnT op, OpKernelContext* context,
                                 hipsparseHandle_t hipsparse_handle, int m,
                                 int n, int nnz, const Scalar* csrVal,
                                 const int* csrRowPtr, const int* csrColInd,
                                 Scalar* cscVal, int* cscRowInd, int* cscColPtr,
                                 const hipsparseAction_t copyValues) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(hipsparse_handle, m, n, nnz,
                                  AsHipComplex(csrVal), csrRowPtr, csrColInd,
                                  AsHipComplex(cscVal), cscRowInd, cscColPtr,
                                  copyValues, HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

#define CSR2CSC_INSTANCE(Scalar, sparse_prefix)                              \
  template <>                                                                \
  Status GpuSparse::Csr2csc<Scalar>(                                         \
      int m, int n, int nnz, const Scalar* csrVal, const int* csrRowPtr,     \
      const int* csrColInd, Scalar* cscVal, int* cscRowInd, int* cscColPtr,  \
      const hipsparseAction_t copyValues) {                                  \
    DCHECK(initialized_);                                                    \
    return Csr2cscImpl(SPARSE_FN(csr2csc, sparse_prefix), context_,          \
                       *gpusparse_handle_, m, n, nnz, csrVal, csrRowPtr,     \
                       csrColInd, cscVal, cscRowInd, cscColPtr, copyValues); \
  }

TF_CALL_HIP_LAPACK_TYPES(CSR2CSC_INSTANCE);

template <typename Scalar, typename SparseFnT>
static inline Status CsrgeamBufferSizeExtImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t sparse_handle,
    int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      sparse_handle, m, n, AsHipComplex(alpha), descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(beta), descrB, nnzB, AsHipComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsHipComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, bufferSize));
  return Status::OK();
}

#define CSRGEAM_BUFFERSIZE_INSTANCE(Scalar, sparse_prefix)                     \
  template <>                                                                  \
  Status GpuSparse::CsrgeamBufferSizeExt<Scalar>(                              \
      int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,      \
      const int* csrSortedColIndA, const Scalar* beta,                         \
      const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,                \
      const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {      \
    DCHECK(initialized_);                                                      \
    return CsrgeamBufferSizeExtImpl(                                           \
        SPARSE_FN(csrgeam2_bufferSizeExt, sparse_prefix), context_,            \
        *gpusparse_handle_, m, n, alpha, descrA, nnzA, csrSortedValA,          \
        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, \
        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,             \
        csrSortedRowPtrC, csrSortedColIndC, bufferSize);                       \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEAM_BUFFERSIZE_INSTANCE);

Status GpuSparse::CsrgeamNnz(
    int m, int n, const hipsparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const hipsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSrocm_sparseDTcc mht_13(mht_13_v, 679, "", "./tensorflow/core/util/rocm_sparse.cc", "GpuSparse::CsrgeamNnz");

  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
  TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseXcsrgeam2Nnz(
      *gpusparse_handle_, m, n, descrA, nnzA, csrSortedRowPtrA,
      csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
      descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status Csrgeam2Impl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t cusparse_handle,
    int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      cusparse_handle, m, n, AsHipComplex(alpha), descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(beta), descrB, nnzB, AsHipComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsHipComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, workspace));
  return Status::OK();
}

#define CSRGEAM_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                  \
  Status GpuSparse::Csrgeam<Scalar>(                                           \
      int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,      \
      const int* csrSortedColIndA, const Scalar* beta,                         \
      const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,                \
      const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {         \
    DCHECK(initialized_);                                                      \
    return Csrgeam2Impl(SPARSE_FN(csrgeam2, sparse_prefix), context_,          \
                        *gpusparse_handle_, m, n, alpha, descrA, nnzA,         \
                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,     \
                        beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,   \
                        csrSortedColIndB, descrC, csrSortedValC,               \
                        csrSortedRowPtrC, csrSortedColIndC, workspace);        \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEAM_INSTANCE);

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
