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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/cusolver_context.h"

#include <algorithm>
#include <complex>
#include <utility>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

namespace {

// Type traits to get CUDA complex types from std::complex<T>.
template <typename T>
struct GpuComplexT {
  typedef T type;
};

// For ROCm, use hipsolver if the ROCm version >= 4.5 and
// rocblas/rocsolver if the ROCm version < 4.5.

#if !TENSORFLOW_USE_ROCM

#define GPU_SOLVER_CONTEXT_PREFIX cusolverDn
#define GPU_SOLVER_PREFIX cusolverDn

using gpuStream_t = cudaStream_t;

template <>
struct GpuComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef cuComplex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef cuDoubleComplex* type;
};

#else

using gpuStream_t = hipStream_t;

#if TF_ROCM_VERSION >= 40500
#define GPU_SOLVER_CONTEXT_PREFIX tensorflow::wrap::hipsolver
#define GPU_SOLVER_PREFIX tensorflow::wrap::hipsolver

template <>
struct GpuComplexT<std::complex<float>> {
  typedef hipFloatComplex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef hipDoubleComplex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef hipFloatComplex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef hipDoubleComplex* type;
};
#else
#define GPU_SOLVER_CONTEXT_PREFIX tensorflow::wrap::rocblas_
#define GPU_SOLVER_PREFIX tensorflow::wrap::rocsolver_

template <>
struct GpuComplexT<std::complex<float>> {
  typedef rocblas_float_complex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef rocblas_double_complex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef rocblas_float_complex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef rocblas_double_complex* type;
};
#endif  // TF_ROCM_VERSION >= 40500

#endif  // !TENSORFLOW_USE_ROCM

template <typename T>
inline typename GpuComplexT<T>::type* ToDevicePointer(se::DeviceMemory<T> p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_0(mht_0_v, 284, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "ToDevicePointer");

  return static_cast<typename GpuComplexT<T>::type*>(p.opaque());
}

#if !TENSORFLOW_USE_ROCM
cublasFillMode_t GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case se::blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

// Converts a cuSolver status to a Status.
Status ConvertStatus(cusolverStatus_t status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_1(mht_1_v, 304, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "ConvertStatus");

  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return Status::OK();
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return FailedPrecondition("cuSolver has not been initialized");
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return ResourceExhausted("cuSolver allocation failed");
    case CUSOLVER_STATUS_INVALID_VALUE:
      return InvalidArgument("cuSolver invalid value error");
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return FailedPrecondition("cuSolver architecture mismatch error");
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return Unknown("cuSolver mapping error");
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return Unknown("cuSolver execution failed");
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return Internal("cuSolver internal error");
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return Unimplemented("cuSolver matrix type not supported error");
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return Unimplemented("cuSolver not supported error");
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return InvalidArgument("cuSolver zero pivot error");
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return FailedPrecondition("cuSolver invalid license error");
    default:
      return Unknown("Unknown cuSolver error");
  }
}
#else
#if TF_ROCM_VERSION >= 40500
hipsolverFillMode_t GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return HIPSOLVER_FILL_MODE_UPPER;
    case se::blas::UpperLower::kLower:
      return HIPSOLVER_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

Status ConvertStatus(hipsolverStatus_t status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_2(mht_2_v, 350, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "ConvertStatus");

  switch (status) {
    case HIPSOLVER_STATUS_SUCCESS:
      return Status::OK();
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
      return FailedPrecondition("hipsolver has not been initialized");
    case HIPSOLVER_STATUS_ALLOC_FAILED:
      return ResourceExhausted("hipsolver allocation failed");
    case HIPSOLVER_STATUS_INVALID_VALUE:
      return InvalidArgument("hipsolver invalid value error");
    case HIPSOLVER_STATUS_MAPPING_ERROR:
      return Unknown("hipsolver mapping error");
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
      return Unknown("hipsolver execution failed");
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
      return Internal("hipsolver internal error");
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
      return Unimplemented("hipsolver not supported error");
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
      return FailedPrecondition("cuSolver architecture mismatch error");
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
      return InvalidArgument("hipsolver handle is nullptr error");
    case HIPSOLVER_STATUS_INVALID_ENUM:
      return InvalidArgument("hipsolver invalid enum error");
    case HIPSOLVER_STATUS_UNKNOWN:
      return Unknown("hipsolver status unknown");
    default:
      return Unknown("Unknown hipsolver error");
  }
}
#else
rocblas_fill GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case se::blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

Status ConvertStatus(rocblas_status status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_3(mht_3_v, 395, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "ConvertStatus");

  switch (status) {
    case rocblas_status_success:
      return Status::OK();
    case rocblas_status_invalid_handle:
      return FailedPrecondition("handle not initialized, invalid or null");
    case rocblas_status_not_implemented:
      return Internal("function is not implemented");
    case rocblas_status_invalid_pointer:
      return InvalidArgument("invalid pointer argument");
    case rocblas_status_invalid_size:
      return InvalidArgument("invalid size argument");
    case rocblas_status_memory_error:
      return Internal("failed internal memory allocation, copy or dealloc");
    case rocblas_status_internal_error:
      return Internal("other internal library failure");
    case rocblas_status_perf_degraded:
      return Internal("performance degraded due to low device memory");
    case rocblas_status_size_query_mismatch:
      return Unknown("unmatched start/stop size query");
    case rocblas_status_size_increased:
      return Unknown("queried device memory size increased");
    case rocblas_status_size_unchanged:
      return Unknown("queried device memory size unchanged");
    case rocblas_status_invalid_value:
      return InvalidArgument("passed argument not valid");
    case rocblas_status_continue:
      return Unknown("nothing preventing function to proceed");
    default:
      return Unknown("Unknown rocsolver error");
  }
}
#endif  // TF_ROCM_VERSION >= 40500
#endif  // TENSORFLOW_USE_ROCM

#define GPU_SOLVER_CAT_NX(A, B) A##B
#define GPU_SOLVER_CAT(A, B) GPU_SOLVER_CAT_NX(A, B)

#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
#define GpuSolverCreate GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, Create)
#define GpuSolverSetStream GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, SetStream)
#define GpuSolverDestroy GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, Destroy)
#else  // TENSORFLOW_USE_ROCSOLVER
#define GpuSolverCreate GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, create_handle)
#define GpuSolverSetStream GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, set_stream)
#define GpuSolverDestroy \
  GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, destroy_handle)
#endif
#define GpuSolverSpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Spotrf_bufferSize)
#define GpuSolverDpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Dpotrf_bufferSize)
#define GpuSolverCpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Cpotrf_bufferSize)
#define GpuSolverZpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Zpotrf_bufferSize)
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
#define GpuSolverSpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Spotrf)
#define GpuSolverDpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Dpotrf)
#define GpuSolverCpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Cpotrf)
#define GpuSolverZpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Zpotrf)
#define GpuSolverSpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, SpotrfBatched)
#define GpuSolverDpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, DpotrfBatched)
#define GpuSolverCpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, CpotrfBatched)
#define GpuSolverZpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, ZpotrfBatched)
#else  // TENSORFLOW_USE_ROCSOLVER
#define GpuSolverSpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, spotrf)
#define GpuSolverDpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, dpotrf)
#define GpuSolverCpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, cpotrf)
#define GpuSolverZpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, zpotrf)
#define GpuSolverSpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, spotrf_batched)
#define GpuSolverDpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, dpotrf_batched)
#define GpuSolverCpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, cpotrf_batched)
#define GpuSolverZpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, zpotrf_batched)
#endif

}  // namespace

StatusOr<GpuSolverContext> GpuSolverContext::Create(se::Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_4(mht_4_v, 476, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::Create");

  gpusolverHandle_t handle;
  TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverCreate(&handle)));
  GpuSolverContext context(stream, handle);

  if (stream) {
    // StreamExecutor really should just expose the Cuda stream to clients...
    const gpuStream_t* gpu_stream =
        CHECK_NOTNULL(reinterpret_cast<const gpuStream_t*>(
            stream->implementation()->GpuStreamMemberHack()));
    TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverSetStream(handle, *gpu_stream)));
  }

  return std::move(context);
}

GpuSolverContext::GpuSolverContext(se::Stream* stream, gpusolverHandle_t handle)
    : stream_(stream), handle_(handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_5(mht_5_v, 496, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::GpuSolverContext");
}

GpuSolverContext::GpuSolverContext(GpuSolverContext&& other) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_6(mht_6_v, 501, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::GpuSolverContext");

  handle_ = other.handle_;
  stream_ = other.stream_;
  other.handle_ = nullptr;
  other.stream_ = nullptr;
}

GpuSolverContext& GpuSolverContext::operator=(GpuSolverContext&& other) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_7(mht_7_v, 511, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "=");

  std::swap(handle_, other.handle_);
  std::swap(stream_, other.stream_);
  return *this;
}

GpuSolverContext::~GpuSolverContext() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_8(mht_8_v, 520, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::~GpuSolverContext");

  if (handle_) {
    Status status = ConvertStatus(GpuSolverDestroy(handle_));
    if (!status.ok()) {
      LOG(ERROR) << "GpuSolverDestroy failed: " << status;
    }
  }
}

// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
StatusOr<int64_t> GpuSolverContext::PotrfBufferSize(PrimitiveType type,
                                                    se::blas::UpperLower uplo,
                                                    int n, int lda,
                                                    int batch_size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_9(mht_9_v, 538, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::PotrfBufferSize");

#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
  int size = -1;
  switch (type) {
    case F32: {
      TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverSpotrf_bufferSize(
          handle(), GpuBlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case F64: {
      TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverDpotrf_bufferSize(
          handle(), GpuBlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case C64: {
      TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverCpotrf_bufferSize(
          handle(), GpuBlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case C128: {
      TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverZpotrf_bufferSize(
          handle(), GpuBlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    default:
      return InvalidArgument("Invalid type for cholesky decomposition: %s",
                             PrimitiveType_Name(type));
  }
  // CUDA's potrfBatched needs space for the `as` array, which contains
  // batch_size pointers.  Divide by sizeof(type) because this function returns
  // not bytes but a number of elements of `type`.
  int64_t potrf_batched_scratch = CeilOfRatio<int64_t>(
      batch_size * sizeof(void*), primitive_util::ByteWidth(type));

  return std::max<int64_t>(size, potrf_batched_scratch);
#else  // not supported in rocsolver
  return 0;
#endif
}

Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                               se::DeviceMemory<float> a, int lda,
                               se::DeviceMemory<int> lapack_info,
                               se::DeviceMemoryBase workspace) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_10(mht_10_v, 584, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::Potrf");

  return ConvertStatus(GpuSolverSpotrf(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(a), lda,
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
      ToDevicePointer(se::DeviceMemory<float>(workspace)),
      se::DeviceMemory<float>(workspace).ElementCount(),
#endif
      ToDevicePointer(lapack_info)));
}

Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                               se::DeviceMemory<double> a, int lda,
                               se::DeviceMemory<int> lapack_info,
                               se::DeviceMemoryBase workspace) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_11(mht_11_v, 600, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::Potrf");

  return ConvertStatus(GpuSolverDpotrf(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(a), lda,
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
      ToDevicePointer(se::DeviceMemory<double>(workspace)),
      se::DeviceMemory<double>(workspace).ElementCount(),
#endif
      ToDevicePointer(lapack_info)));
}

Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                               se::DeviceMemory<std::complex<float>> a, int lda,
                               se::DeviceMemory<int> lapack_info,
                               se::DeviceMemoryBase workspace) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_12(mht_12_v, 616, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::Potrf");

  return ConvertStatus(GpuSolverCpotrf(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(a), lda,
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
      ToDevicePointer(se::DeviceMemory<std::complex<float>>(workspace)),
      se::DeviceMemory<std::complex<float>>(workspace).ElementCount(),
#endif
      ToDevicePointer(lapack_info)));
}

Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                               se::DeviceMemory<std::complex<double>> a,
                               int lda, se::DeviceMemory<int> lapack_info,
                               se::DeviceMemoryBase workspace) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_13(mht_13_v, 632, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::Potrf");

  return ConvertStatus(GpuSolverZpotrf(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(a), lda,
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
      ToDevicePointer(se::DeviceMemory<std::complex<double>>(workspace)),
      se::DeviceMemory<std::complex<double>>(workspace).ElementCount(),
#endif
      ToDevicePointer(lapack_info)));
}

Status GpuSolverContext::PotrfBatched(se::blas::UpperLower uplo, int n,
                                      se::DeviceMemory<float*> as, int lda,
                                      se::DeviceMemory<int> lapack_info,
                                      int batch_size) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_14(mht_14_v, 648, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::PotrfBatched");

  return ConvertStatus(GpuSolverSpotrfBatched(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

Status GpuSolverContext::PotrfBatched(se::blas::UpperLower uplo, int n,
                                      se::DeviceMemory<double*> as, int lda,
                                      se::DeviceMemory<int> lapack_info,
                                      int batch_size) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_15(mht_15_v, 663, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::PotrfBatched");

  return ConvertStatus(GpuSolverDpotrfBatched(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

Status GpuSolverContext::PotrfBatched(se::blas::UpperLower uplo, int n,
                                      se::DeviceMemory<std::complex<float>*> as,
                                      int lda,
                                      se::DeviceMemory<int> lapack_info,
                                      int batch_size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_16(mht_16_v, 679, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::PotrfBatched");

  return ConvertStatus(GpuSolverCpotrfBatched(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

Status GpuSolverContext::PotrfBatched(
    se::blas::UpperLower uplo, int n,
    se::DeviceMemory<std::complex<double>*> as, int lda,
    se::DeviceMemory<int> lapack_info, int batch_size) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPScusolver_contextDTcc mht_17(mht_17_v, 694, "", "./tensorflow/compiler/xla/service/gpu/cusolver_context.cc", "GpuSolverContext::PotrfBatched");

  return ConvertStatus(GpuSolverZpotrfBatched(
      handle(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

}  // namespace gpu
}  // namespace xla
