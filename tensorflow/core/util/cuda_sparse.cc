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
class MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc() {
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

#ifdef GOOGLE_CUDA

#include "tensorflow/core/util/cuda_sparse.h"

#include <complex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party/gpus/cuda/include/cusparse.h"
#include "third_party/gpus/cuda/include/library_types.h"
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
#include "tensorflow/core/util/gpu_solvers.h"

// TODO(rmlarsen,penporn): Investigate using newer kernels in CUDA 10.1+.

namespace tensorflow {
namespace {

// Type traits to get CUDA complex types from std::complex<>.
// TODO: reuse with cuda_solvers
template <typename T>
struct CudaComplexT {
  typedef T type;
};
template <>
struct CudaComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct CudaComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};
// Converts pointers of std::complex<> to pointers of
// cuComplex/cuDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename CudaComplexT<T>::type* AsCudaComplex(const T* p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/util/cuda_sparse.cc", "AsCudaComplex");

  return reinterpret_cast<const typename CudaComplexT<T>::type*>(p);
}
template <typename T>
inline typename CudaComplexT<T>::type* AsCudaComplex(T* p) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/util/cuda_sparse.cc", "AsCudaComplex");

  return reinterpret_cast<typename CudaComplexT<T>::type*>(p);
}

// A set of initialized handles to the underlying Cuda libraries used by
// GpuSparse. We maintain one such set of handles per unique stream.
class CudaSparseHandles {
 public:
  explicit CudaSparseHandles(cudaStream_t stream)
      : initialized_(false), stream_(stream) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/util/cuda_sparse.cc", "CudaSparseHandles");
}

  CudaSparseHandles(CudaSparseHandles&& rhs)
      : initialized_(rhs.initialized_),
        stream_(std::move(rhs.stream_)),
        cusparse_handle_(rhs.cusparse_handle_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/util/cuda_sparse.cc", "CudaSparseHandles");

    rhs.initialized_ = false;
  }

  CudaSparseHandles& operator=(CudaSparseHandles&& rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/util/cuda_sparse.cc", "=");

    if (this == &rhs) return *this;
    Release();
    stream_ = std::move(rhs.stream_);
    cusparse_handle_ = std::move(rhs.cusparse_handle_);
    initialized_ = rhs.initialized_;
    rhs.initialized_ = false;
    return *this;
  }

  ~CudaSparseHandles() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/util/cuda_sparse.cc", "~CudaSparseHandles");
 Release(); }

  Status Initialize() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/util/cuda_sparse.cc", "Initialize");

    if (initialized_) return Status::OK();
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreate(&cusparse_handle_));
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseSetStream(cusparse_handle_, stream_));
    initialized_ = true;
    return Status::OK();
  }

  cusparseHandle_t& handle() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/util/cuda_sparse.cc", "handle");

    DCHECK(initialized_);
    return cusparse_handle_;
  }

  const cusparseHandle_t& handle() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_8(mht_8_v, 304, "", "./tensorflow/core/util/cuda_sparse.cc", "handle");

    DCHECK(initialized_);
    return cusparse_handle_;
  }

 private:
  void Release() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_9(mht_9_v, 313, "", "./tensorflow/core/util/cuda_sparse.cc", "Release");

    if (initialized_) {
      // This should never return anything other than success
      auto err = cusparseDestroy(cusparse_handle_);
      DCHECK(err == CUSPARSE_STATUS_SUCCESS)
          << "Failed to destroy cuSparse instance.";
      initialized_ = false;
    }
  }
  bool initialized_;
  cudaStream_t stream_;
  cusparseHandle_t cusparse_handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparseHandles);
};

// TODO(ebrevdo): Replace global mutex guarding CudaSparseHandles
// lookup with one of:
//    1. Adding the handle to the CudaStream structure; do the lookup there.
//    2. Add a thread-local cusparse, set it to the current stream
//       upon each call.
// #1 seems like the cleanest option but will need to wait until this
// is moved into TF core.
static mutex handle_map_mutex(LINKER_INITIALIZED);

using HandleMap = std::unordered_map<cudaStream_t, CudaSparseHandles>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_10(mht_10_v, 345, "", "./tensorflow/core/util/cuda_sparse.cc", "GetHandleMapSingleton");

  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

GpuSparse::GpuSparse(OpKernelContext* context)
    : initialized_(false), context_(context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_11(mht_11_v, 356, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::GpuSparse");

  auto cuda_stream_ptr =
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack());
  DCHECK(cuda_stream_ptr);
  gpu_stream_ = *cuda_stream_ptr;
}

Status GpuSparse::Initialize() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_12(mht_12_v, 369, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::Initialize");

  HandleMap* handle_map = GetHandleMapSingleton();
  DCHECK(handle_map);
  mutex_lock lock(handle_map_mutex);
  auto it = handle_map->find(gpu_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating CudaSparse handles for stream " << gpu_stream_;
    // Previously unseen Cuda stream. Initialize a set of Cuda sparse library
    // handles for it.
    CudaSparseHandles new_handles(gpu_stream_);
    TF_RETURN_IF_ERROR(new_handles.Initialize());
    it = handle_map->insert(std::make_pair(gpu_stream_, std::move(new_handles)))
             .first;
  }
  gpusparse_handle_ = &it->second.handle();
  initialized_ = true;
  return Status::OK();
}

#define TF_CALL_CUSPARSE_DTYPES(m)           \
  m(float, CUDA_R_32F) m(double, CUDA_R_64F) \
      m(std::complex<float>, CUDA_C_32F) m(std::complex<double>, CUDA_C_64F)

// Macro that specializes a sparse method for all 4 standard
// numeric types.
// TODO: reuse with cuda_solvers
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct cusparse method names.
#define SPARSE_FN(method, sparse_prefix) cusparse##sparse_prefix##method
#define SPARSE_NAME(method, sparse_prefix) "cusparse" #sparse_prefix #method
#define BUFSIZE_FN(method, sparse_prefix) \
  cusparse##sparse_prefix##method##_bufferSizeExt

//=============================================================================
// Wrappers of cuSparse computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cusparse.h header file.
//=============================================================================

template <typename Scalar, typename SparseFn>
static inline Status Gtsv2Impl(SparseFn op, cusparseHandle_t cusparse_handle,
                               int m, int n, const Scalar* dl, const Scalar* d,
                               const Scalar* du, Scalar* B, int ldb,
                               void* pBuffer) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(cusparse_handle, m, n, AsCudaComplex(dl),
                                  AsCudaComplex(d), AsCudaComplex(du),
                                  AsCudaComplex(B), ldb, pBuffer));
  return Status::OK();
}

#define GTSV2_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                \
  Status GpuSparse::Gtsv2<Scalar>(int m, int n, const Scalar* dl,            \
                                  const Scalar* d, const Scalar* du,         \
                                  Scalar* B, int ldb, void* pBuffer) const { \
    DCHECK(initialized_);                                                    \
    return Gtsv2Impl(SPARSE_FN(gtsv2, sparse_prefix), *gpusparse_handle_, m, \
                     n, dl, d, du, B, ldb, pBuffer);                         \
  }

TF_CALL_LAPACK_TYPES(GTSV2_INSTANCE);

#define GTSV2_NO_PIVOT_INSTANCE(Scalar, sparse_prefix)                      \
  template <>                                                               \
  Status GpuSparse::Gtsv2NoPivot<Scalar>(                                   \
      int m, int n, const Scalar* dl, const Scalar* d, const Scalar* du,    \
      Scalar* B, int ldb, void* pBuffer) const {                            \
    DCHECK(initialized_);                                                   \
    return Gtsv2Impl(SPARSE_FN(gtsv2_nopivot, sparse_prefix),               \
                     *gpusparse_handle_, m, n, dl, d, du, B, ldb, pBuffer); \
  }

TF_CALL_LAPACK_TYPES(GTSV2_NO_PIVOT_INSTANCE);

template <typename Scalar, typename SparseFn>
static inline Status Gtsv2BufferSizeExtImpl(SparseFn op,
                                            cusparseHandle_t cusparse_handle,
                                            int m, int n, const Scalar* dl,
                                            const Scalar* d, const Scalar* du,
                                            const Scalar* B, int ldb,
                                            size_t* bufferSizeInBytes) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(cusparse_handle, m, n, AsCudaComplex(dl),
                                  AsCudaComplex(d), AsCudaComplex(du),
                                  AsCudaComplex(B), ldb, bufferSizeInBytes));
  return Status::OK();
}

#define GTSV2_BUFFER_SIZE_INSTANCE(Scalar, sparse_prefix)                     \
  template <>                                                                 \
  Status GpuSparse::Gtsv2BufferSizeExt<Scalar>(                               \
      int m, int n, const Scalar* dl, const Scalar* d, const Scalar* du,      \
      const Scalar* B, int ldb, size_t* bufferSizeInBytes) const {            \
    DCHECK(initialized_);                                                     \
    return Gtsv2BufferSizeExtImpl(                                            \
        SPARSE_FN(gtsv2_bufferSizeExt, sparse_prefix), *gpusparse_handle_, m, \
        n, dl, d, du, B, ldb, bufferSizeInBytes);                             \
  }

TF_CALL_LAPACK_TYPES(GTSV2_BUFFER_SIZE_INSTANCE);

#define GTSV2_NO_PIVOT_BUFFER_SIZE_INSTANCE(Scalar, sparse_prefix)       \
  template <>                                                            \
  Status GpuSparse::Gtsv2NoPivotBufferSizeExt<Scalar>(                   \
      int m, int n, const Scalar* dl, const Scalar* d, const Scalar* du, \
      const Scalar* B, int ldb, size_t* bufferSizeInBytes) const {       \
    DCHECK(initialized_);                                                \
    return Gtsv2BufferSizeExtImpl(                                       \
        SPARSE_FN(gtsv2_nopivot_bufferSizeExt, sparse_prefix),           \
        *gpusparse_handle_, m, n, dl, d, du, B, ldb, bufferSizeInBytes); \
  }

TF_CALL_LAPACK_TYPES(GTSV2_NO_PIVOT_BUFFER_SIZE_INSTANCE);

template <typename Scalar, typename SparseFn>
static inline Status Gtsv2StridedBatchImpl(SparseFn op,
                                           cusparseHandle_t cusparse_handle,
                                           int m, const Scalar* dl,
                                           const Scalar* d, const Scalar* du,
                                           Scalar* x, int batchCount,
                                           int batchStride, void* pBuffer) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      cusparse_handle, m, AsCudaComplex(dl), AsCudaComplex(d),
      AsCudaComplex(du), AsCudaComplex(x), batchCount, batchStride, pBuffer));
  return Status::OK();
}

#define GTSV2_STRIDED_BATCH_INSTANCE(Scalar, sparse_prefix)                   \
  template <>                                                                 \
  Status GpuSparse::Gtsv2StridedBatch<Scalar>(                                \
      int m, const Scalar* dl, const Scalar* d, const Scalar* du, Scalar* x,  \
      int batchCount, int batchStride, void* pBuffer) const {                 \
    DCHECK(initialized_);                                                     \
    return Gtsv2StridedBatchImpl(SPARSE_FN(gtsv2StridedBatch, sparse_prefix), \
                                 *gpusparse_handle_, m, dl, d, du, x,         \
                                 batchCount, batchStride, pBuffer);           \
  }

TF_CALL_LAPACK_TYPES(GTSV2_STRIDED_BATCH_INSTANCE);

template <typename Scalar, typename SparseFn>
static inline Status Gtsv2StridedBatchBufferSizeImpl(
    SparseFn op, cusparseHandle_t cusparse_handle, int m, const Scalar* dl,
    const Scalar* d, const Scalar* du, const Scalar* x, int batchCount,
    int batchStride, size_t* bufferSizeInBytes) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(cusparse_handle, m, AsCudaComplex(dl),
                                  AsCudaComplex(d), AsCudaComplex(du),
                                  AsCudaComplex(x), batchCount, batchStride,
                                  bufferSizeInBytes));
  return Status::OK();
}

#define GTSV2_STRIDED_BATCH_BUFFER_SIZE_INSTANCE(Scalar, sparse_prefix) \
  template <>                                                           \
  Status GpuSparse::Gtsv2StridedBatchBufferSizeExt<Scalar>(             \
      int m, const Scalar* dl, const Scalar* d, const Scalar* du,       \
      const Scalar* x, int batchCount, int batchStride,                 \
      size_t* bufferSizeInBytes) const {                                \
    DCHECK(initialized_);                                               \
    return Gtsv2StridedBatchBufferSizeImpl(                             \
        SPARSE_FN(gtsv2StridedBatch_bufferSizeExt, sparse_prefix),      \
        *gpusparse_handle_, m, dl, d, du, x, batchCount, batchStride,   \
        bufferSizeInBytes);                                             \
  }

TF_CALL_LAPACK_TYPES(GTSV2_STRIDED_BATCH_BUFFER_SIZE_INSTANCE);

Status GpuSparse::Coo2csr(const int* cooRowInd, int nnz, int m,
                          int* csrRowPtr) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_13(mht_13_v, 544, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::Coo2csr");

  // cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle,
  //                                               const int *cooRowInd,
  //                                               int nnz,
  //                                               int m,
  //                                               int *csrSortedRowPtr,
  //                                               cusparseIndexBase_t
  //                                               idxBase);
  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcoo2csr(*gpusparse_handle_, cooRowInd,
                                                nnz, m, csrRowPtr,
                                                CUSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

Status GpuSparse::Csr2coo(const int* csrRowPtr, int nnz, int m,
                          int* cooRowInd) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_14(mht_14_v, 563, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::Csr2coo");

  // cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle,
  //                                               const int *csrRowPtr,
  //                                               int nnz,
  //                                               int m,
  //                                               int *cooRowInd,
  //                                               cusparseIndexBase_t
  //                                               idxBase);
  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcsr2coo(*gpusparse_handle_, csrRowPtr,
                                                nnz, m, cooRowInd,
                                                CUSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

Status GpuSparse::CsrgeamNnz(
    int m, int n, const cusparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_15(mht_15_v, 586, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::CsrgeamNnz");

  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
#if CUDA_VERSION >= 10000
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcsrgeam2Nnz(
      *gpusparse_handle_, m, n, descrA, nnzA, csrSortedRowPtrA,
      csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
      descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace));
#else
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcsrgeamNnz(
      *gpusparse_handle_, m, n, descrA, nnzA, csrSortedRowPtrA,
      csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
      descrC, csrSortedRowPtrC, nnzTotalDevHostPtr));
#endif
  return Status::OK();
}

#if CUDA_VERSION < 10020

template <typename Scalar, typename SparseFnT>
static inline Status CsrmmImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k,
    int nnz, const Scalar* alpha_host, const cusparseMatDescr_t descrA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* B, int ldb,
    const Scalar* beta_host, Scalar* C, int ldc) {
  // cusparseStatus_t CUSPARSEAPI cusparseScsrmm2(
  //     cusparseHandle_t handle, cusparseOperation_t transA,
  //     cusparseOperation_t transB, int m, int n, int k, int nnz,
  //     const float* alpha, const cusparseMatDescr_t descrA,
  //     const float* csrSortedValA, const int* csrSortedRowPtrA,
  //     const int* csrSortedColIndA, const float* B, int ldb, const float*
  //     beta, float* C, int ldc);
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      cusparse_handle, transA, transB, m, n, k, nnz, AsCudaComplex(alpha_host),
      descrA, AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsCudaComplex(B), ldb, AsCudaComplex(beta_host), AsCudaComplex(C), ldc));
  return Status::OK();
}

#define CSRMM_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                \
  Status GpuSparse::Csrmm<Scalar>(                                           \
      cusparseOperation_t transA, cusparseOperation_t transB, int m, int n,  \
      int k, int nnz, const Scalar* alpha_host,                              \
      const cusparseMatDescr_t descrA, const Scalar* csrSortedValA,          \
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,              \
      const Scalar* B, int ldb, const Scalar* beta_host, Scalar* C, int ldc) \
      const {                                                                \
    DCHECK(initialized_);                                                    \
    return CsrmmImpl(SPARSE_FN(csrmm2, sparse_prefix), context_,             \
                     *gpusparse_handle_, transA, transB, m, n, k, nnz,       \
                     alpha_host, descrA, csrSortedValA, csrSortedRowPtrA,    \
                     csrSortedColIndA, B, ldb, beta_host, C, ldc);           \
  }

TF_CALL_LAPACK_TYPES(CSRMM_INSTANCE);

#else

#define SPMM_BUFFERSIZE_INSTANCE(Scalar, dtype)                              \
  template <>                                                                \
  Status GpuSparse::SpMMBufferSize<Scalar>(                                  \
      cusparseOperation_t transA, cusparseOperation_t transB,                \
      const Scalar* alpha, const cusparseSpMatDescr_t matA,                  \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                  \
      gpusparseDnMatDescr_t matC, cusparseSpMMAlg_t alg, size_t* bufferSize) \
      const {                                                                \
    DCHECK(initialized_);                                                    \
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseSpMM_bufferSize(                    \
        *gpusparse_handle_, transA, transB, alpha, matA, matB, beta, matC,   \
        dtype, alg, bufferSize));                                            \
    return Status::OK();                                                     \
  }

TF_CALL_CUSPARSE_DTYPES(SPMM_BUFFERSIZE_INSTANCE);

#define SPMM_INSTANCE(Scalar, dtype)                                           \
  template <>                                                                  \
  Status GpuSparse::SpMM<Scalar>(                                              \
      cusparseOperation_t transA, cusparseOperation_t transB,                  \
      const Scalar* alpha, const cusparseSpMatDescr_t matA,                    \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                    \
      gpusparseDnMatDescr_t matC, cusparseSpMMAlg_t alg, int8* buffer) const { \
    DCHECK(initialized_);                                                      \
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseSpMM(*gpusparse_handle_, transA,      \
                                              transB, alpha, matA, matB, beta, \
                                              matC, dtype, alg, buffer));      \
    return Status::OK();                                                       \
  }

TF_CALL_CUSPARSE_DTYPES(SPMM_INSTANCE);

#endif

#if CUDA_VERSION < 10020

template <typename Scalar, typename SparseFnT>
static inline Status CsrmvImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    cusparseOperation_t transA, int m, int n, int nnz, const Scalar* alpha_host,
    const cusparseMatDescr_t descrA, const Scalar* csrSortedValA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA, const Scalar* x,
    const Scalar* beta_host, Scalar* y) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(cusparse_handle, transA, m, n, nnz, AsCudaComplex(alpha_host), descrA,
         AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         AsCudaComplex(x), AsCudaComplex(beta_host), AsCudaComplex(y)));
  return Status::OK();
}

// TODO(ebrevdo,rmlarsen): Use csrmv_mp for all cases when available in CUDA 9.
#define CSRMV_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                \
  Status GpuSparse::Csrmv<Scalar>(                                           \
      cusparseOperation_t transA, int m, int n, int nnz,                     \
      const Scalar* alpha_host, const cusparseMatDescr_t descrA,             \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,              \
      const int* csrSortedColIndA, const Scalar* x, const Scalar* beta_host, \
      Scalar* y) const {                                                     \
    DCHECK(initialized_);                                                    \
    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) {                        \
      return CsrmvImpl(SPARSE_FN(csrmv_mp, sparse_prefix), context_,         \
                       *gpusparse_handle_, transA, m, n, nnz, alpha_host,    \
                       descrA, csrSortedValA, csrSortedRowPtrA,              \
                       csrSortedColIndA, x, beta_host, y);                   \
    } else {                                                                 \
      return CsrmvImpl(SPARSE_FN(csrmv, sparse_prefix), context_,            \
                       *gpusparse_handle_, transA, m, n, nnz, alpha_host,    \
                       descrA, csrSortedValA, csrSortedRowPtrA,              \
                       csrSortedColIndA, x, beta_host, y);                   \
    }                                                                        \
  }

TF_CALL_LAPACK_TYPES(CSRMV_INSTANCE);

#else

template <typename Scalar>
static inline Status CsrmvExImpl(cudaDataType_t dtype, OpKernelContext* context,
                                 cusparseHandle_t cusparse_handle,
                                 cusparseOperation_t transA, int m, int n,
                                 int nnz, const Scalar* alpha_host,
                                 const Scalar* csrSortedValA,
                                 const int* csrSortedRowPtrA,
                                 const int* csrSortedColIndA, const Scalar* x,
                                 const Scalar* beta_host, Scalar* y) {
  cusparseMatDescr_t descrA;
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
  TF_RETURN_IF_GPUSPARSE_ERROR(
      cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  TF_RETURN_IF_GPUSPARSE_ERROR(
      cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  // CUSPARSE_ALG_MERGE_PATH algo only supports non-transpose matrix.
  DCHECK(transA == CUSPARSE_OPERATION_NON_TRANSPOSE);

  size_t bufferSize;
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCsrmvEx_bufferSize(
      cusparse_handle, CUSPARSE_ALG_MERGE_PATH, transA, m, n, nnz, alpha_host,
      dtype, descrA, csrSortedValA, dtype, csrSortedRowPtrA, csrSortedColIndA,
      x, dtype, beta_host, dtype, y, dtype, dtype, &bufferSize));

  Tensor buffer;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(bufferSize)}), &buffer));
  auto pBuffer = buffer.flat<int8>();
  DCHECK(pBuffer.data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCsrmvEx(
      cusparse_handle, CUSPARSE_ALG_MERGE_PATH, transA, m, n, nnz, alpha_host,
      dtype, descrA, csrSortedValA, dtype, csrSortedRowPtrA, csrSortedColIndA,
      x, dtype, beta_host, dtype, y, dtype, dtype, pBuffer.data()));

  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseDestroyMatDescr(descrA));
  return Status::OK();
}

template <typename Scalar>
static inline Status SpMVImpl(cudaDataType_t dtype, OpKernelContext* context,
                              cusparseHandle_t cusparse_handle,
                              cusparseOperation_t transA, int m, int n, int nnz,
                              const Scalar* alpha_host,
                              const Scalar* csrSortedValA,
                              const int* csrSortedRowPtrA,
                              const int* csrSortedColIndA, const Scalar* x,
                              const Scalar* beta_host, Scalar* y) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_16(mht_16_v, 775, "", "./tensorflow/core/util/cuda_sparse.cc", "SpMVImpl");

  cusparseSpMatDescr_t matA;
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateCsr(
      &matA, m, n, nnz, const_cast<int*>(csrSortedRowPtrA),
      const_cast<int*>(csrSortedColIndA), const_cast<Scalar*>(csrSortedValA),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dtype));

  cusparseDnVecDescr_t vecX, vecY;
  int sizeX = (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
  int sizeY = (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;
  TF_RETURN_IF_GPUSPARSE_ERROR(
      cusparseCreateDnVec(&vecX, sizeX, const_cast<Scalar*>(x), dtype));
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateDnVec(&vecY, sizeY, y, dtype));

  size_t bufferSize;
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseSpMV_bufferSize(
      cusparse_handle, transA, alpha_host, matA, vecX, beta_host, vecY, dtype,
      CUSPARSE_CSRMV_ALG1, &bufferSize));

  Tensor buffer;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(bufferSize)}), &buffer));
  auto pBuffer = buffer.flat<int8>();
  DCHECK(pBuffer.data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(
      cusparseSpMV(cusparse_handle, transA, alpha_host, matA, vecX, beta_host,
                   vecY, dtype, CUSPARSE_CSRMV_ALG1, pBuffer.data()));

  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseDestroyDnVec(vecY));
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseDestroySpMat(matA));
  return Status::OK();
}

#define CSRMV_INSTANCE(Scalar, cudaDataType)                                   \
  template <>                                                                  \
  Status GpuSparse::Csrmv<Scalar>(                                             \
      cusparseOperation_t transA, int m, int n, int nnz,                       \
      const Scalar* alpha_host, const Scalar* csrSortedValA,                   \
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,                \
      const Scalar* x, const Scalar* beta_host, Scalar* y) const {             \
    DCHECK(initialized_);                                                      \
    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) {                          \
      return CsrmvExImpl(cudaDataType, context_, *gpusparse_handle_, transA,   \
                         m, n, nnz, alpha_host, csrSortedValA,                 \
                         csrSortedRowPtrA, csrSortedColIndA, x, beta_host, y); \
    } else {                                                                   \
      return SpMVImpl(cudaDataType, context_, *gpusparse_handle_, transA, m,   \
                      n, nnz, alpha_host, csrSortedValA, csrSortedRowPtrA,     \
                      csrSortedColIndA, x, beta_host, y);                      \
    }                                                                          \
  }

TF_CALL_CUSPARSE_DTYPES(CSRMV_INSTANCE);

#endif  // CUDA_VERSION < 10020

#if CUDA_VERSION < 10000

template <typename Scalar, typename SparseFnT>
static inline Status CsrgeamImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(cusparse_handle, m, n, AsCudaComplex(alpha), descrA, nnzA,
         AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         AsCudaComplex(beta), descrB, nnzB, AsCudaComplex(csrSortedValB),
         csrSortedRowPtrB, csrSortedColIndB, descrC,
         AsCudaComplex(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC));
  return Status::OK();
}

#define CSRGEAM_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                 \
  Status GpuSparse::Csrgeam<Scalar>(                                          \
      int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,     \
      const int* csrSortedColIndA, const Scalar* beta,                        \
      const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,               \
      const cusparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {        \
    DCHECK(initialized_);                                                     \
    return CsrgeamImpl(SPARSE_FN(csrgeam, sparse_prefix), context_,           \
                       *gpusparse_handle_, m, n, alpha, descrA, nnzA,         \
                       csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,     \
                       beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,   \
                       csrSortedColIndB, descrC, csrSortedValC,               \
                       csrSortedRowPtrC, csrSortedColIndC);                   \
  }

#else

template <typename Scalar, typename SparseFnT>
static inline Status Csrgeam2Impl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      cusparse_handle, m, n, AsCudaComplex(alpha), descrA, nnzA,
      AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsCudaComplex(beta), descrB, nnzB, AsCudaComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsCudaComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, workspace));
  return Status::OK();
}

#define CSRGEAM_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                 \
  Status GpuSparse::Csrgeam<Scalar>(                                          \
      int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,     \
      const int* csrSortedColIndA, const Scalar* beta,                        \
      const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,               \
      const cusparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {        \
    DCHECK(initialized_);                                                     \
    return Csrgeam2Impl(SPARSE_FN(csrgeam2, sparse_prefix), context_,         \
                        *gpusparse_handle_, m, n, alpha, descrA, nnzA,        \
                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,    \
                        beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,  \
                        csrSortedColIndB, descrC, csrSortedValC,              \
                        csrSortedRowPtrC, csrSortedColIndC, workspace);       \
  }

#endif

TF_CALL_LAPACK_TYPES(CSRGEAM_INSTANCE);

#if CUDA_VERSION < 10000

#define CSRGEAM_BUFFERSIZE_INSTANCE(Scalar, sparse_prefix)                    \
  template <>                                                                 \
  Status GpuSparse::CsrgeamBufferSizeExt<Scalar>(                             \
      int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,     \
      const int* csrSortedColIndA, const Scalar* beta,                        \
      const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,               \
      const cusparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {     \
    DCHECK(initialized_);                                                     \
    *bufferSize = 0;                                                          \
    return Status::OK();                                                      \
  }

#else

template <typename Scalar, typename SparseFnT>
static inline Status CsrgeamBufferSizeExtImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t sparse_handle,
    int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      sparse_handle, m, n, AsCudaComplex(alpha), descrA, nnzA,
      AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsCudaComplex(beta), descrB, nnzB, AsCudaComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsCudaComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, bufferSize));
  return Status::OK();
}

#define CSRGEAM_BUFFERSIZE_INSTANCE(Scalar, sparse_prefix)                     \
  template <>                                                                  \
  Status GpuSparse::CsrgeamBufferSizeExt<Scalar>(                              \
      int m, int n, const Scalar* alpha, const cusparseMatDescr_t descrA,      \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,      \
      const int* csrSortedColIndA, const Scalar* beta,                         \
      const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,  \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,                \
      const cusparseMatDescr_t descrC, Scalar* csrSortedValC,                  \
      int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {      \
    DCHECK(initialized_);                                                      \
    return CsrgeamBufferSizeExtImpl(                                           \
        SPARSE_FN(csrgeam2_bufferSizeExt, sparse_prefix), context_,            \
        *gpusparse_handle_, m, n, alpha, descrA, nnzA, csrSortedValA,          \
        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, \
        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,             \
        csrSortedRowPtrC, csrSortedColIndC, bufferSize);                       \
  }

#endif

TF_CALL_LAPACK_TYPES(CSRGEAM_BUFFERSIZE_INSTANCE);

#if CUDA_VERSION < 10000

Status GpuSparse::CsrgemmNnz(
    cusparseOperation_t transA, cusparseOperation_t transB, int m, int k, int n,
    const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, int* csrSortedRowPtrC,
    int* nnzTotalDevHostPtr) {
  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcsrgemmNnz(
      *gpusparse_handle_, transA, transB, m, k, n, descrA, nnzA,
      csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
      csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status CsrgemmImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    cusparseOperation_t transA, cusparseOperation_t transB, int m, int k, int n,
    const cusparseMatDescr_t descrA, int nnzA, const Scalar* csrSortedValA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(cusparse_handle, transA, transB, m, k, n, descrA, nnzA,
         AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         descrB, nnzB, AsCudaComplex(csrSortedValB), csrSortedRowPtrB,
         csrSortedColIndB, descrC, AsCudaComplex(csrSortedValC),
         csrSortedRowPtrC, csrSortedColIndC));
  return Status::OK();
}

#define CSRGEMM_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                 \
  Status GpuSparse::Csrgemm<Scalar>(                                          \
      cusparseOperation_t transA, cusparseOperation_t transB, int m, int k,   \
      int n, const cusparseMatDescr_t descrA, int nnzA,                       \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,               \
      const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, \
      const Scalar* csrSortedValB, const int* csrSortedRowPtrB,               \
      const int* csrSortedColIndB, const cusparseMatDescr_t descrC,           \
      Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) {  \
    DCHECK(initialized_);                                                     \
    return CsrgemmImpl(SPARSE_FN(csrgemm, sparse_prefix), context_,           \
                       *gpusparse_handle_, transA, transB, m, k, n, descrA,   \
                       nnzA, csrSortedValA, csrSortedRowPtrA,                 \
                       csrSortedColIndA, descrB, nnzB, csrSortedValB,         \
                       csrSortedRowPtrB, csrSortedColIndB, descrC,            \
                       csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);    \
  }

TF_CALL_LAPACK_TYPES(CSRGEMM_INSTANCE);

#else

template <typename T>
static const T* one_ptr() {
  static const T one = static_cast<T>(1);
  return &one;
}

template <typename T>
static const T* null_ptr() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_17(mht_17_v, 1048, "", "./tensorflow/core/util/cuda_sparse.cc", "null_ptr");

  return nullptr;
}

#define CSRGEMM_BUFFERSIZE_INSTANCE(Scalar, sparse_prefix)                     \
  template <>                                                                  \
  Status GpuSparse::CsrgemmBufferSize<Scalar>(                                 \
      int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,          \
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,                \
      const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,  \
      const int* csrSortedColIndB, csrgemm2Info_t info,                        \
      size_t* workspaceBytes) {                                                \
    DCHECK(initialized_);                                                      \
    TF_RETURN_IF_GPUSPARSE_ERROR(SPARSE_FN(csrgemm2_bufferSizeExt,             \
                                           sparse_prefix)(                     \
        *gpusparse_handle_, m, n, k, AsCudaComplex(one_ptr<Scalar>()), descrA, \
        nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,                \
        csrSortedRowPtrB, csrSortedColIndB, AsCudaComplex(null_ptr<Scalar>()), \
        descrA, 0, null_ptr<int>(), null_ptr<int>(), info, workspaceBytes));   \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES(CSRGEMM_BUFFERSIZE_INSTANCE);

Status GpuSparse::CsrgemmNnz(
    int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, csrgemm2Info_t info,
    void* workspace) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPScuda_sparseDTcc mht_18(mht_18_v, 1081, "", "./tensorflow/core/util/cuda_sparse.cc", "GpuSparse::CsrgemmNnz");

  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseXcsrgemm2Nnz(
      *gpusparse_handle_, m, n, k, descrA, nnzA, csrSortedRowPtrA,
      csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
      descrA, 0, null_ptr<int>(), null_ptr<int>(), descrC, csrSortedRowPtrC,
      nnzTotalDevHostPtr, info, workspace));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status CsrgemmImpl(
    SparseFnT op, OpKernelContext* context, cusparseHandle_t cusparse_handle,
    int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
    const csrgemm2Info_t info, void* workspace) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(cusparse_handle, m, n, k, AsCudaComplex(one_ptr<Scalar>()), descrA,
         nnzA, AsCudaComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         descrB, nnzB, AsCudaComplex(csrSortedValB), csrSortedRowPtrB,
         csrSortedColIndB, AsCudaComplex(null_ptr<Scalar>()), descrA, 0,
         AsCudaComplex(null_ptr<Scalar>()), null_ptr<int>(), null_ptr<int>(),
         descrC, AsCudaComplex(csrSortedValC), csrSortedRowPtrC,
         csrSortedColIndC, info, workspace));
  return Status::OK();
}

#define CSRGEMM_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                 \
  Status GpuSparse::Csrgemm<Scalar>(                                          \
      int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,         \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,               \
      const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, \
      const Scalar* csrSortedValB, const int* csrSortedRowPtrB,               \
      const int* csrSortedColIndB, const cusparseMatDescr_t descrC,           \
      Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,    \
      const csrgemm2Info_t info, void* workspace) {                           \
    DCHECK(initialized_);                                                     \
    return CsrgemmImpl(SPARSE_FN(csrgemm2, sparse_prefix), context_,          \
                       *gpusparse_handle_, m, n, k, descrA, nnzA,             \
                       csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,     \
                       descrB, nnzB, csrSortedValB, csrSortedRowPtrB,         \
                       csrSortedColIndB, descrC, csrSortedValC,               \
                       csrSortedRowPtrC, csrSortedColIndC, info, workspace);  \
  }

TF_CALL_LAPACK_TYPES(CSRGEMM_INSTANCE);

#endif  // CUDA_VERSION < 10000

template <typename Scalar, typename BufferSizeFnT, typename SparseFnT>
static inline Status Csru2csrImpl(SparseFnT op, BufferSizeFnT buffer_size_op,
                                  OpKernelContext* context,
                                  cusparseHandle_t cusparse_handle, int m,
                                  int n, int nnz,
                                  const cusparseMatDescr_t descrA,
                                  Scalar* csrVal, const int* csrRowPtr,
                                  int* csrColInd) {
  GpuSparseCsrSortingConversionInfo info;
  TF_RETURN_IF_ERROR(info.Initialize());

  size_t pBufferSizeInBytes = 0;

  TF_RETURN_IF_GPUSPARSE_ERROR(
      buffer_size_op(cusparse_handle, m, n, nnz, AsCudaComplex(csrVal),
                     csrRowPtr, csrColInd, info.info(), &pBufferSizeInBytes));

  Tensor pBuffer_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(pBufferSizeInBytes)}),
      &pBuffer_t));
  auto pBuffer = pBuffer_t.flat<int8>();
  DCHECK(pBuffer.data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(op(cusparse_handle, m, n, nnz, descrA,
                                  AsCudaComplex(csrVal), csrRowPtr, csrColInd,
                                  info.info(), pBuffer.data()));

  return Status::OK();
}

#define CSRU2CSR_INSTANCE(Scalar, sparse_prefix)                              \
  template <>                                                                 \
  Status GpuSparse::Csru2csr<Scalar>(                                         \
      int m, int n, int nnz, const cusparseMatDescr_t descrA, Scalar* csrVal, \
      const int* csrRowPtr, int* csrColInd) {                                 \
    DCHECK(initialized_);                                                     \
    return Csru2csrImpl(SPARSE_FN(csru2csr, sparse_prefix),                   \
                        BUFSIZE_FN(csru2csr, sparse_prefix), context_,        \
                        *gpusparse_handle_, m, n, nnz, descrA, csrVal,        \
                        csrRowPtr, csrColInd);                                \
  }

TF_CALL_LAPACK_TYPES(CSRU2CSR_INSTANCE);

#if CUDA_VERSION < 10010

template <typename Scalar, typename SparseFnT>
static inline Status Csr2cscImpl(SparseFnT op, OpKernelContext* context,
                                 cusparseHandle_t cusparse_handle, int m, int n,
                                 int nnz, const Scalar* csrVal,
                                 const int* csrRowPtr, const int* csrColInd,
                                 Scalar* cscVal, int* cscRowInd, int* cscColPtr,
                                 const cusparseAction_t copyValues) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(cusparse_handle, m, n, nnz,
                                  AsCudaComplex(csrVal), csrRowPtr, csrColInd,
                                  AsCudaComplex(cscVal), cscRowInd, cscColPtr,
                                  copyValues, CUSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

#define CSR2CSC_INSTANCE(Scalar, sparse_prefix)                              \
  template <>                                                                \
  Status GpuSparse::Csr2csc<Scalar>(                                         \
      int m, int n, int nnz, const Scalar* csrVal, const int* csrRowPtr,     \
      const int* csrColInd, Scalar* cscVal, int* cscRowInd, int* cscColPtr,  \
      const cusparseAction_t copyValues) {                                   \
    DCHECK(initialized_);                                                    \
    return Csr2cscImpl(SPARSE_FN(csr2csc, sparse_prefix), context_,          \
                       *gpusparse_handle_, m, n, nnz, csrVal, csrRowPtr,     \
                       csrColInd, cscVal, cscRowInd, cscColPtr, copyValues); \
  }

TF_CALL_LAPACK_TYPES(CSR2CSC_INSTANCE);

#else

template <typename Scalar>
static inline Status Csr2cscImpl(cudaDataType_t dtype, OpKernelContext* context,
                                 cusparseHandle_t cusparse_handle, int m, int n,
                                 int nnz, const Scalar* csrVal,
                                 const int* csrRowPtr, const int* csrColInd,
                                 Scalar* cscVal, int* cscRowInd, int* cscColPtr,
                                 const cusparseAction_t copyValues) {
  size_t bufferSize;
  TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(
      cusparse_handle, m, n, nnz, AsCudaComplex(csrVal), csrRowPtr, csrColInd,
      AsCudaComplex(cscVal), cscColPtr, cscRowInd, dtype, copyValues,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &bufferSize));

  Tensor buffer;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataTypeToEnum<Scalar>::value,
      TensorShape({static_cast<int64_t>(bufferSize)}), &buffer));

  DCHECK(buffer.flat<Scalar>().data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(
      cusparseCsr2cscEx2(cusparse_handle, m, n, nnz, AsCudaComplex(csrVal),
                         csrRowPtr, csrColInd, AsCudaComplex(cscVal), cscColPtr,
                         cscRowInd, dtype, copyValues, CUSPARSE_INDEX_BASE_ZERO,
                         CUSPARSE_CSR2CSC_ALG2, buffer.flat<Scalar>().data()));

  return Status::OK();
}

#define CSR2CSC_INSTANCE(Scalar, cudaDataType)                                \
  template <>                                                                 \
  Status GpuSparse::Csr2csc<Scalar>(                                          \
      int m, int n, int nnz, const Scalar* csrVal, const int* csrRowPtr,      \
      const int* csrColInd, Scalar* cscVal, int* cscRowInd, int* cscColPtr,   \
      const cusparseAction_t copyValues) {                                    \
    DCHECK(initialized_);                                                     \
    return Csr2cscImpl(cudaDataType, context_, *gpusparse_handle_, m, n, nnz, \
                       csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd,       \
                       cscColPtr, copyValues);                                \
  }

TF_CALL_CUSPARSE_DTYPES(CSR2CSC_INSTANCE);

#endif  // CUDA_VERSION < 10010

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
