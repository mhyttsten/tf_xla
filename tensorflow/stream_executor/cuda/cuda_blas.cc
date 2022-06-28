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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc() {
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

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"

#define SE_CUDA_DATA_HALF CUDA_R_16F

#include "tensorflow/stream_executor/cuda/cuda_blas.h"

// Both Eigen Half.h and CUDA cuda_fp16.h provide similar typedef for __half. As
// such, there are two ways to get the typedef for __half:
//
// (1) Includes cuda_fp16.h and defines EIGEN_HAS_CUDA_FP16.
// (2) Neither includes cuda_fp16.h nor defines EIGEN_HAS_CUDA_FP16.
//
// Due to issue b/73793421, when the first approach is used and NVCC is used to
// compile this file, NVCC will complain duplicated definition for
// EIGEN_HAS_CUDA_FP16. On the other hand, when the second approach is used and
// clang is used to compile this file, clang will not understand __half
// due to missing the definition and macro EIGEN_HAS_CUDA_FP16.
//
// Because this file may be compiled with CLANG but will never be compiled with
// NVCC, we choose the first approach for CUDA < 9.0. For CUDA >= 9.0, we have
// to use the second approach because the data member in the __half defined
// by CUDA > 9.0 is `__x` while Eigen expects it to be `x`.
//
// TODO(b/73793421): Remove the following code block to switch to the second
// approach when the issue is fixed.
#if CUDA_VERSION < 9000
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#define EIGEN_HAS_CUDA_FP16
#endif

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

static std::string ToString(cublasStatus_t status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_0(mht_0_v, 246, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToString");

  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 8000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return absl::StrCat("<invalid cublas status: ", status, ">");
  }
}

// cuBLAS has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with cublasSetPointerMode.
//
// This helper sets the cuBLAS pointer mode to a desired value for a cuBLAS call
// you are about to perform in a given scope.
//
// The prior cuBLAS pointer mode is retained and restored when this object goes
// out of scope.
class ScopedCublasPointerMode {
 public:
  // Note that, because the setting of the cublas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the pointer mode.
  explicit ScopedCublasPointerMode(cublasHandle_t handle)
      : handle_(handle), ok_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_1(mht_1_v, 296, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ScopedCublasPointerMode");
}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasPointerMode_t new_mode) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_2(mht_2_v, 305, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "Init");

    cublasStatus_t ret = cublasGetPointerMode(handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = cublasSetPointerMode(handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasPointerMode() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_3(mht_3_v, 326, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "~ScopedCublasPointerMode");

    if (ok_) {
      cublasStatus_t ret = cublasSetPointerMode(handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
  cublasPointerMode_t old_mode_;  // Prior cuBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};

#if CUDA_VERSION >= 9000
// cuBLAS has interfaces that permit computations to use the Volta hardware.
// This must be enabled via the cublasGet/SetMathMode APIs.
//
// This helper sets the cuBLAS math mode to a desired value for a cuBLAS call
// you are about to perform in a given scope.
//
// The prior cuBLAS math mode is retained and restored when this object goes
// out of scope.
class ScopedCublasMathMode {
 public:
  // Note that, because the setting of the cublas math mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the math mode.
  explicit ScopedCublasMathMode(cublasHandle_t handle)
      : handle_(handle), ok_(false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_4(mht_4_v, 363, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ScopedCublasMathMode");
}

  // Attempts the switch to the requested scoped math mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasMath_t new_mode) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_5(mht_5_v, 372, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "Init");

    cublasStatus_t ret = cublasGetMathMode(handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas math mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = cublasSetMathMode(handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas math mode: " << ToString(ret);
      return ok_ = false;
    }
    return ok_ = true;
  }

  // Switches back to the prior math mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasMathMode() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_6(mht_6_v, 392, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "~ScopedCublasMathMode");

    if (ok_) {
      cublasStatus_t ret = cublasSetMathMode(handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas math mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
  cublasMath_t old_mode_;  // Prior cuBLAS math mode, to be restored.
  bool ok_;                // Whether the change was successful.
};
#endif  // CUDA_VERSION >= 9000

static const char *const kCublasNotInitializedExplanation =
    "Failure to initialize cublas may be due to OOM (cublas needs some free "
    "memory when you initialize it, and your deep-learning framework may have "
    "preallocated more than its fair share), or may be because this binary was "
    "not built with support for the GPU in your machine.";

bool CUDABlas::Init() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_7(mht_7_v, 418, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::Init");

  gpu::ScopedActivateExecutorContext sac{parent_};
  cublasStatus_t ret = cublasCreate(&blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    if (ret == CUBLAS_STATUS_NOT_INITIALIZED) {
      LOG(ERROR) << kCublasNotInitializedExplanation;
    }
    return false;
  }

#if CUDA_VERSION >= 11000
  ret = cublasLtCreate(&blasLt_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublasLt handle: " << ToString(ret);
    if (ret == CUBLAS_STATUS_NOT_INITIALIZED) {
      LOG(ERROR) << kCublasNotInitializedExplanation;
    }
    return false;
  }
#endif  // CUDA_VERSION >= 11000

  return true;
}

CUDABlas::CUDABlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)),
      blas_(nullptr)
#if CUDA_VERSION >= 11000
      ,
      blasLt_(nullptr)
#endif
{
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_8(mht_8_v, 453, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::CUDABlas");

}

CUDABlas::~CUDABlas() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_9(mht_9_v, 459, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::~CUDABlas");

  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    cublasDestroy(blas_);
  }
#if CUDA_VERSION >= 11000
  if (blasLt_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    cublasLtDestroy(blasLt_);
  }
#endif
}

bool CUDABlas::SetStream(Stream *stream) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_10(mht_10_v, 475, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::SetStream");

  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  cublasStatus_t ret = cublasSetStream(blas_, AsGpuStreamValue(stream));
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

cudaStream_t CUDABlas::CUDAStream(Stream *stream) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_11(mht_11_v, 492, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::CUDAStream");

  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  return AsGpuStreamValue(stream);
}

namespace {

// Helper functions transforming blas arguments into cuBLAS arguments.

cublasOperation_t CUDABlasTranspose(blas::Transpose trans) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_12(mht_12_v, 506, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasTranspose");

  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

cublasFillMode_t CUDABlasUpperLower(blas::UpperLower uplo) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_13(mht_13_v, 522, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasUpperLower");

  switch (uplo) {
    case blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

cublasDiagType_t CUDABlasDiagonal(blas::Diagonal diag) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_14(mht_14_v, 536, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasDiagonal");

  switch (diag) {
    case blas::Diagonal::kUnit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

cublasSideMode_t CUDABlasSide(blas::Side side) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_15(mht_15_v, 550, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasSide");

  switch (side) {
    case blas::Side::kLeft:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return CUBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

// CUDADataType<T>::type translates from a C++ type (e.g. float) to a
// cudaDataType_t (e.g. CUDA_R_32F).  CUDAComputationType(ty) translates from a
// blas::ComputationType to a cudaDataType_t.
//
// These are used to build the argument type and computation type args to
// cublasGemmEx.
template <typename T>
struct CUDADataType;

template <>
struct CUDADataType<Eigen::half> {
  static constexpr cudaDataType_t type = SE_CUDA_DATA_HALF;
};

template <>
struct CUDADataType<std::complex<Eigen::half>> {
  static constexpr cudaDataType_t type = CUDA_C_16F;
};

template <>
struct CUDADataType<float> {
  static constexpr cudaDataType_t type = CUDA_R_32F;
};

template <>
struct CUDADataType<std::complex<float>> {
  static constexpr cudaDataType_t type = CUDA_C_32F;
};

template <>
struct CUDADataType<double> {
  static constexpr cudaDataType_t type = CUDA_R_64F;
};

template <>
struct CUDADataType<std::complex<double>> {
  static constexpr cudaDataType_t type = CUDA_C_64F;
};

template <>
struct CUDADataType<int> {
  static constexpr cudaDataType_t type = CUDA_R_32I;
};

template <>
struct CUDADataType<int8> {
  static constexpr cudaDataType_t type = CUDA_R_8I;
};

template <>
struct CUDADataType<std::complex<int8>> {
  static constexpr cudaDataType_t type = CUDA_C_8I;
};

template <>
struct CUDADataType<uint8> {
  static constexpr cudaDataType_t type = CUDA_R_8U;
};

template <>
struct CUDADataType<std::complex<uint8>> {
  static constexpr cudaDataType_t type = CUDA_C_8U;
};

cudaDataType_t CUDAComputationType(blas::ComputationType ty) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_16(mht_16_v, 628, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDAComputationType");

  switch (ty) {
    case blas::ComputationType::kF16:
      return CUDA_R_16F;
    case blas::ComputationType::kF32:
      return CUDA_R_32F;
    case blas::ComputationType::kF64:
      return CUDA_R_64F;
    case blas::ComputationType::kI32:
      return CUDA_R_32I;
    case blas::ComputationType::kComplexF32:
      return CUDA_C_32F;
    case blas::ComputationType::kComplexF64:
      return CUDA_C_64F;
    case blas::ComputationType::kTF32AsF32:  // fall-through
    case blas::ComputationType::kBF16AsF32:
      // These cases are currently only supported in the blasLt routines, which
      // use CUBLASComputationType() instead.
      LOG(FATAL) << "Invalid value of blas::ComputationType.";
  }
}

#if CUDA_VERSION >= 11000
cublasComputeType_t CUBLASComputationType(blas::ComputationType ty) {
  switch (ty) {
    case blas::ComputationType::kF16:
      return CUBLAS_COMPUTE_16F;
    case blas::ComputationType::kF32:  // fall-through
    case blas::ComputationType::kComplexF32:
      return CUBLAS_COMPUTE_32F;
    case blas::ComputationType::kF64:  // fall-through
    case blas::ComputationType::kComplexF64:
      return CUBLAS_COMPUTE_64F;
    case blas::ComputationType::kI32:
      return CUBLAS_COMPUTE_32I;
    case blas::ComputationType::kTF32AsF32:
      return CUBLAS_COMPUTE_32F_FAST_TF32;
    case blas::ComputationType::kBF16AsF32:
      return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
}
#endif  // CUDA_VERSION >= 11000

blas::DataType GetScaleType(blas::DataType data_type,
                            blas::ComputationType compute_type) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_17(mht_17_v, 675, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "GetScaleType");

  bool is_complex = data_type == blas::DataType::kComplexFloat ||
                    data_type == blas::DataType::kComplexDouble;
  switch (compute_type) {
    case blas::ComputationType::kF16:
      return blas::DataType::kHalf;
    case blas::ComputationType::kF32:         // fall-through
    case blas::ComputationType::kComplexF32:  // fall-through
    case blas::ComputationType::kTF32AsF32:   // fall-through
    case blas::ComputationType::kBF16AsF32:
      return is_complex ? blas::DataType::kComplexFloat
                        : blas::DataType::kFloat;
    case blas::ComputationType::kF64:  // fall-through
    case blas::ComputationType::kComplexF64:
      return is_complex ? blas::DataType::kComplexDouble
                        : blas::DataType::kDouble;
    case blas::ComputationType::kI32:
      return blas::DataType::kInt32;
  }
}

#if CUDA_VERSION >= 11000
cublasLtPointerMode_t CUBLASPointerMode(blas::PointerMode pointer_mode) {
  switch (pointer_mode) {
    case blas::PointerMode::kHost:
      return CUBLASLT_POINTER_MODE_HOST;
    case blas::PointerMode::kDevice:
      return CUBLASLT_POINTER_MODE_DEVICE;
  }
}
cublasLtEpilogue_t CUBLASEpilogue(blas::Epilogue epilogue) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_18(mht_18_v, 708, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUBLASEpilogue");

  switch (epilogue) {
    case blas::Epilogue::kDefault:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case blas::Epilogue::kReLU:
      return CUBLASLT_EPILOGUE_RELU;
    case blas::Epilogue::kBias:
      return CUBLASLT_EPILOGUE_BIAS;
    case blas::Epilogue::kBiasThenReLU:
      return CUBLASLT_EPILOGUE_RELU_BIAS;
  }
}
#endif  // CUDA_VERSION >= 11000

cudaDataType_t GetCUDADataType(blas::DataType ty) {
  switch (ty) {
    case blas::DataType::kHalf:
      return CUDA_R_16F;
#if CUDA_VERSION >= 11000
    case blas::DataType::kBF16:
      return CUDA_R_16BF;
#endif
    case blas::DataType::kFloat:
      return CUDA_R_32F;
    case blas::DataType::kDouble:
      return CUDA_R_64F;
    case blas::DataType::kInt8:
      return CUDA_R_8I;
    case blas::DataType::kInt32:
      return CUDA_R_32I;
    case blas::DataType::kComplexFloat:
      return CUDA_C_32F;
    case blas::DataType::kComplexDouble:
      return CUDA_C_64F;
    default:
      LOG(FATAL) << "Invalid value of blas::DataType in GetCUDADataType";
  }
}

int GetDataTypeSizeBytes(blas::DataType ty) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_19(mht_19_v, 750, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "GetDataTypeSizeBytes");

  switch (ty) {
    case blas::DataType::kHalf:
      return 2;
    case blas::DataType::kFloat:
      return 4;
    case blas::DataType::kDouble:
      return 8;
    case blas::DataType::kInt8:
      return 1;
    case blas::DataType::kInt32:
      return 4;
    case blas::DataType::kComplexFloat:
      return 8;
    case blas::DataType::kComplexDouble:
      return 16;
    default:
      LOG(FATAL) << "Invalid value of blas::DataType in GetDataTypeSizeBytes";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
port::Status CUDABlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                          bool pointer_mode_host,
                                          cublasMath_t math_type,
                                          Args... args) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_20(mht_20_v, 780, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasInternalImpl");

  absl::MutexLock lock(&mu_);

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return port::InternalError("Failed setting stream");
  }

#if CUDA_VERSION >= 9000
  ScopedCublasMathMode math_mode{blas_};
#if CUBLAS_VER_MAJOR >= 11
  if (math_type == CUBLAS_TF32_TENSOR_OP_MATH &&
      tensorflow::tensor_float_32_execution_enabled()) {
#else
  if (math_type == CUBLAS_TENSOR_OP_MATH) {
#endif
    if (!math_mode.Init(math_type)) {
      return port::InternalError("Failed initializing math mode");
    }
  }
#endif

  gpu::ScopedActivateExecutorContext sac{parent_};
  ScopedCublasPointerMode pointer_mode{blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return port::InternalError("Failed setting error mode");
  }
  cublasStatus_t ret = cublas_func(blas_, args...);
  if (ret == CUBLAS_STATUS_SUCCESS) {
    return port::Status::OK();
  }
  return port::InternalError(ToString(ret));
}

// cublas_func may be overloaded, so we need to figure out which one we really
// need to call based on the args. One way to do it is to wrap it in lambda.
#define AS_LAMBDA(func)                                                  \
  [](auto &&... args) -> decltype(                                       \
                          func(std::forward<decltype(args)>(args)...)) { \
    return func(std::forward<decltype(args)>(args)...);                  \
  }

bool CUDABlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_21(mht_21_v, 828, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAsum");

  return DoBlasInternal(cublasSasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_22(mht_22_v, 839, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAsum");

  return DoBlasInternal(cublasDasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_23(mht_23_v, 850, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAsum");

  return DoBlasInternal(cublasScasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_24(mht_24_v, 861, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAsum");

  return DoBlasInternal(cublasDzasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_25(mht_25_v, 872, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAxpy");

  return DoBlasInternal(cublasSaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_26(mht_26_v, 883, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAxpy");

  return DoBlasInternal(cublasDaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_27(mht_27_v, 895, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAxpy");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_28(mht_28_v, 909, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasAxpy");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_29(mht_29_v, 922, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasCopy");

  return DoBlasInternal(cublasScopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_30(mht_30_v, 933, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasCopy");

  return DoBlasInternal(cublasDcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_31(mht_31_v, 944, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasCopy");

  return DoBlasInternal(cublasCcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_32(mht_32_v, 955, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasCopy");

  return DoBlasInternal(cublasZcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64_t elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_33(mht_33_v, 967, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDot");

  return DoBlasInternal(cublasSdot, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64_t elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_34(mht_34_v, 979, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDot");

  return DoBlasInternal(cublasDdot, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_35(mht_35_v, 991, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDotc");

  return DoBlasInternal(cublasCdotc, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_36(mht_36_v, 1004, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDotc");

  return DoBlasInternal(cublasZdotc, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_37(mht_37_v, 1017, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDotu");

  return DoBlasInternal(cublasCdotu, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_38(mht_38_v, 1030, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasDotu");

  return DoBlasInternal(cublasZdotu, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_39(mht_39_v, 1042, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasNrm2");

  return DoBlasInternal(cublasSnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_40(mht_40_v, 1053, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasNrm2");

  return DoBlasInternal(cublasDnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_41(mht_41_v, 1064, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasNrm2");

  return DoBlasInternal(cublasScnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_42(mht_42_v, 1075, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasNrm2");

  return DoBlasInternal(cublasDznrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_43(mht_43_v, 1086, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRot");

  return DoBlasInternal(cublasSrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_44(mht_44_v, 1098, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRot");

  return DoBlasInternal(cublasDrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_45(mht_45_v, 1110, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRot");

  return DoBlasInternal(cublasCsrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_46(mht_46_v, 1122, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRot");

  return DoBlasInternal(cublasZdrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_47(mht_47_v, 1133, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotg");

  return DoBlasInternal(cublasSrotg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(a), GpuMemoryMutable(b),
                        GpuMemoryMutable(c), GpuMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_48(mht_48_v, 1144, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotg");

  return DoBlasInternal(cublasDrotg, stream, false /* = pointer_mode_host */,
                        GpuComplex(GpuMemoryMutable(a)), GpuMemoryMutable(b),
                        GpuMemoryMutable(c), GpuMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_49(mht_49_v, 1156, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotg");

  return DoBlasInternal(
      cublasCrotg, stream, false /* = pointer_mode_host */,
      GpuComplex(GpuMemoryMutable(a)), GpuComplex(GpuMemoryMutable(b)),
      GpuComplex(GpuMemoryMutable(c)), GpuComplex(GpuMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_50(mht_50_v, 1169, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotg");

  return DoBlasInternal(
      cublasZrotg, stream, false /* = pointer_mode_host */,
      GpuComplex(GpuMemoryMutable(a)), GpuComplex(GpuMemoryMutable(b)),
      GpuComplex(GpuMemoryMutable(c)), GpuComplex(GpuMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64_t elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_51(mht_51_v, 1182, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotm");

  return DoBlasInternal(cublasSrotm, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64_t elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_52(mht_52_v, 1194, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotm");

  return DoBlasInternal(cublasDrotm, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_53(mht_53_v, 1206, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotmg");

  return DoBlasInternal(cublasSrotmg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(d1), GpuMemoryMutable(d2),
                        GpuMemoryMutable(x1), GpuMemory(y1),
                        GpuMemoryMutable(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_54(mht_54_v, 1219, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasRotmg");

  return DoBlasInternal(cublasDrotmg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(d1), GpuMemoryMutable(d2),
                        GpuMemoryMutable(x1), GpuMemory(y1),
                        GpuMemoryMutable(param));
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_55(mht_55_v, 1230, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  return DoBlasInternal(cublasSscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_56(mht_56_v, 1239, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  return DoBlasInternal(cublasDscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_57(mht_57_v, 1248, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  return DoBlasInternal(cublasCsscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_58(mht_58_v, 1258, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  return DoBlasInternal(cublasZdscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_59(mht_59_v, 1269, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_60(mht_60_v, 1281, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasScal");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_61(mht_61_v, 1293, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSwap");

  return DoBlasInternal(cublasSswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_62(mht_62_v, 1304, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSwap");

  return DoBlasInternal(cublasDswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_63(mht_63_v, 1315, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSwap");

  return DoBlasInternal(cublasCswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_64(mht_64_v, 1326, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSwap");

  return DoBlasInternal(cublasZswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_65(mht_65_v, 1337, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamax");

  return DoBlasInternal(cublasIsamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_66(mht_66_v, 1348, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamax");

  return DoBlasInternal(cublasIdamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_67(mht_67_v, 1359, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamax");

  return DoBlasInternal(cublasIcamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_68(mht_68_v, 1370, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamax");

  return DoBlasInternal(cublasIzamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_69(mht_69_v, 1381, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamin");

  return DoBlasInternal(cublasIsamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_70(mht_70_v, 1392, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamin");

  return DoBlasInternal(cublasIdamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_71(mht_71_v, 1403, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamin");

  return DoBlasInternal(cublasIcamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_72(mht_72_v, 1414, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasIamin");

  return DoBlasInternal(cublasIzamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_73(mht_73_v, 1427, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGbmv");

  return DoBlasInternal(cublasSgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku, &alpha,
                        GpuMemory(a), lda, GpuMemory(x), incx, &beta,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_74(mht_74_v, 1441, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGbmv");

  return DoBlasInternal(cublasDgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku, &alpha,
                        GpuMemory(a), lda, GpuMemory(x), incx, &beta,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_75(mht_75_v, 1457, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGbmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(x)), incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_76(mht_76_v, 1476, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGbmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(x)), incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_77(mht_77_v, 1492, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemv");

  return DoBlasInternal(cublasSgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_78(mht_78_v, 1506, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemv");

  return DoBlasInternal(cublasDgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_79(mht_79_v, 1521, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_80(mht_80_v, 1539, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64_t m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_81(mht_81_v, 1555, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGer");

  return DoBlasInternal(cublasSger, stream, true /* = pointer_mode_host */, m,
                        n, &alpha, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64_t m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_82(mht_82_v, 1567, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGer");

  return DoBlasInternal(cublasDger, stream, true /* = pointer_mode_host */, m,
                        n, &alpha, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_83(mht_83_v, 1580, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGerc");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCgerc, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_84(mht_84_v, 1595, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGerc");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZgerc, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_85(mht_85_v, 1610, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGeru");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCgeru, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_86(mht_86_v, 1625, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGeru");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZgeru, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_87(mht_87_v, 1641, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHbmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_88(mht_88_v, 1659, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHbmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_89(mht_89_v, 1677, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHemv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChemv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_90(mht_90_v, 1695, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHemv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhemv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_91(mht_91_v, 1711, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer");

  return DoBlasInternal(cublasCher, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_92(mht_92_v, 1724, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer");

  return DoBlasInternal(cublasZher, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_93(mht_93_v, 1738, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer2");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCher2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_94(mht_94_v, 1754, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer2");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZher2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_95(mht_95_v, 1771, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(ap)), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_96(mht_96_v, 1789, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpmv");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(ap)), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_97(mht_97_v, 1805, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpr");

  return DoBlasInternal(cublasChpr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_98(mht_98_v, 1818, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpr");

  return DoBlasInternal(cublasZhpr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_99(mht_99_v, 1832, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpr2");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasChpr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_100(mht_100_v, 1848, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHpr2");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZhpr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_101(mht_101_v, 1863, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSbmv");

  return DoBlasInternal(cublasSsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_102(mht_102_v, 1877, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSbmv");

  return DoBlasInternal(cublasDsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_103(mht_103_v, 1890, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpmv");

  return DoBlasInternal(cublasSspmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_104(mht_104_v, 1902, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpmv");

  return DoBlasInternal(cublasDspmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_105(mht_105_v, 1913, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpr");

  return DoBlasInternal(cublasSspr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_106(mht_106_v, 1924, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpr");

  return DoBlasInternal(cublasDspr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_107(mht_107_v, 1936, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpr2");

  return DoBlasInternal(cublasSspr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_108(mht_108_v, 1948, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSpr2");

  return DoBlasInternal(cublasDspr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_109(mht_109_v, 1960, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymv");

  return DoBlasInternal(cublasSsymv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_110(mht_110_v, 1972, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymv");

  return DoBlasInternal(cublasDsymv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_111(mht_111_v, 1983, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr");

  return DoBlasInternal(cublasSsyr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_112(mht_112_v, 1994, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr");

  return DoBlasInternal(cublasDsyr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_113(mht_113_v, 2006, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2");

  return DoBlasInternal(cublasSsyr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_114(mht_114_v, 2018, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2");

  return DoBlasInternal(cublasDsyr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<float> &a,
                          int lda, DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_115(mht_115_v, 2030, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbmv");

  return DoBlasInternal(cublasStbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<double> &a,
                          int lda, DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_116(mht_116_v, 2043, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbmv");

  return DoBlasInternal(cublasDtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_117(mht_117_v, 2057, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbmv");

  return DoBlasInternal(cublasCtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_118(mht_118_v, 2071, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbmv");

  return DoBlasInternal(cublasZtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<float> &a,
                          int lda, DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_119(mht_119_v, 2084, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbsv");

  return DoBlasInternal(cublasStbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<double> &a,
                          int lda, DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_120(mht_120_v, 2097, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbsv");

  return DoBlasInternal(cublasDtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_121(mht_121_v, 2111, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbsv");

  return DoBlasInternal(cublasCtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_122(mht_122_v, 2125, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTbsv");

  return DoBlasInternal(cublasZtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &ap,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_123(mht_123_v, 2138, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpmv");

  return DoBlasInternal(cublasStpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_124(mht_124_v, 2151, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpmv");

  return DoBlasInternal(cublasDtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_125(mht_125_v, 2165, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpmv");

  return DoBlasInternal(cublasCtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_126(mht_126_v, 2179, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpmv");

  return DoBlasInternal(cublasZtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &ap,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_127(mht_127_v, 2192, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpsv");

  return DoBlasInternal(cublasStpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_128(mht_128_v, 2205, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpsv");

  return DoBlasInternal(cublasDtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_129(mht_129_v, 2219, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpsv");

  return DoBlasInternal(cublasCtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_130(mht_130_v, 2233, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTpsv");

  return DoBlasInternal(cublasZtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_131(mht_131_v, 2246, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmv");

  return DoBlasInternal(cublasStrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_132(mht_132_v, 2259, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmv");

  return DoBlasInternal(cublasDtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_133(mht_133_v, 2273, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmv");

  return DoBlasInternal(cublasCtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_134(mht_134_v, 2287, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmv");

  return DoBlasInternal(cublasZtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_135(mht_135_v, 2300, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsv");

  return DoBlasInternal(cublasStrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_136(mht_136_v, 2313, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsv");

  return DoBlasInternal(cublasDtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_137(mht_137_v, 2327, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsv");

  return DoBlasInternal(cublasCtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_138(mht_138_v, 2341, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsv");

  return DoBlasInternal(cublasZtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

port::Status CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                  blas::Transpose transb, uint64_t m, uint64 n,
                                  uint64_t k, blas::DataType dtype,
                                  const void *alpha, const DeviceMemoryBase &a,
                                  int lda, const DeviceMemoryBase &b, int ldb,
                                  const void *beta, DeviceMemoryBase *c,
                                  int ldc) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_139(mht_139_v, 2357, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemm");

  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;

#if CUDA_VERSION < 11000
  if (dtype == blas::DataType::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == blas::DataType::kFloat) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
    if (stream->GetCudaComputeCapability().IsAtLeast(
            CudaComputeCapability::AMPERE)) {
      // TODO(reedwm): Remove or make this VLOG(1) once TensorFloat-32 is more
      // well tested.
      if (tensorflow::tensor_float_32_execution_enabled()) {
        LOG_FIRST_N(INFO, 1) << "TensorFloat-32 will be used for the matrix "
                                "multiplication. This will only be logged "
                                "once.";
      }
    }
  }
#endif

  // TODO(cheshire): Return an error instead.
  // TODO(cheshire): Why are these checked only for `half` and `float`?
  if (dtype == blas::DataType::kHalf || dtype == blas::DataType::kFloat) {
    if (transa == blas::Transpose::kNoTranspose) {
      if (lda < static_cast<int64_t>(m)) {
        LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                        "precondition violation";
      }
    } else {
      if (lda < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                     << ") (transpose case); precondition violation";
      }
    }
    if (transb == blas::Transpose::kNoTranspose) {
      if (ldb < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                     << ") (no transpose case); precondition violation";
      }
    } else {
      if (ldb < static_cast<int64_t>(n)) {
        LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                        "precondition violation";
      }
    }
  }

  VLOG(1) << absl::StrFormat(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%u n=%u "
      "k=%u alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);

  switch (dtype) {
    case blas::DataType::kHalf: {
#if CUDA_VERSION < 7050
      return port::InternalError(
          "fp16 sgemm is not implemented in this cuBLAS version "
          "(need at least CUDA 7.5)");
#endif

      return DoBlasInternalImpl(
          cublasSgemmEx, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          static_cast<const float *>(alpha), a.opaque(), SE_CUDA_DATA_HALF, lda,
          b.opaque(), SE_CUDA_DATA_HALF, ldb, static_cast<const float *>(beta),
          c->opaque(), SE_CUDA_DATA_HALF, ldc);
    }
#if CUDA_VERSION > 11000
    case blas::DataType::kBF16: {
      return DoBlasInternalImpl(
          cublasSgemmEx, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          static_cast<const float *>(alpha), a.opaque(), CUDA_R_16BF, lda,
          b.opaque(), CUDA_R_16BF, ldb, static_cast<const float *>(beta),
          c->opaque(), CUDA_R_16BF, ldc);
    }
#endif
    case dnn::kFloat:
      return DoBlasInternalImpl(
          cublasSgemm, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda,
          static_cast<const float *>(b.opaque()), ldb,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc);
    case dnn::kDouble:
      return DoBlasInternalImpl(
          cublasDgemm, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda,
          static_cast<const double *>(b.opaque()), ldb,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc);
    case dnn::kComplexFloat: {
      GpuComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<float> *>(alpha));
      GpuComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemm, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          &cb_alpha, static_cast<const GpuComplexType *>(a.opaque()), lda,
          static_cast<const GpuComplexType *>(b.opaque()), ldb, &cb_beta,
          static_cast<GpuComplexType *>(c->opaque()), ldc);
    }
    case dnn::kComplexDouble: {
      GpuDoubleComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<double> *>(alpha));
      GpuDoubleComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemm, stream, true /* = pointer_mode_host */, math_type,
          CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
          &cb_alpha, static_cast<const GpuDoubleComplexType *>(a.opaque()), lda,
          static_cast<const GpuDoubleComplexType *>(b.opaque()), ldb, &cb_beta,
          static_cast<GpuDoubleComplexType *>(c->opaque()), ldc);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_140(mht_140_v, 2494, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_141(mht_141_v, 2507, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_142(mht_142_v, 2521, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_143(mht_143_v, 2535, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_144(mht_144_v, 2549, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_145(mht_145_v, 2562, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_146(mht_146_v, 2576, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_147(mht_147_v, 2591, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_148(mht_148_v, 2606, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

template <typename T>
bool CUDABlas::DoBlasGemvWithProfilingImpl(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, const T &alpha,
    const DeviceMemory<T> &a, int lda, const DeviceMemory<T> &x, int incx,
    const T &beta, DeviceMemory<T> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_149(mht_149_v, 2620, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemvWithProfilingImpl");

  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  // Call blasGemm
  bool result =
      DoBlasGemv(stream, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

  if (timer != nullptr && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(blas::kDefaultBlasGemv);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

template <typename T, typename ParamType>
bool CUDABlas::DoBlasGemmWithProfilingImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
    int lda, const DeviceMemory<T> &b, int ldb, const ParamType &beta,
    DeviceMemory<T> *c, int ldc, blas::ProfileResult *output_profile_result) {
  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  // Call blasGemm
  bool result =
      DoBlasGemm(stream, transa, transb, m, n, k, blas::ToDataType<T>::value,
                 &alpha, a, lda, b, ldb, &beta, c, ldc)
          .ok();

  if (timer != nullptr && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(blas::kDefaultBlasGemm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

static bool UsesTensorOps(blas::AlgorithmType algo) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_150(mht_150_v, 2684, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "UsesTensorOps");

#if CUDA_VERSION >= 9000
  cublasGemmAlgo_t cublas_algo = static_cast<cublasGemmAlgo_t>(algo);
  return cublas_algo >= CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#else
  return false;
#endif
}

static port::StatusOr<cublasMath_t> GetMathTypeForGemmEx(
    Stream *stream, blas::AlgorithmType algorithm, blas::DataType type_a,
    blas::DataType type_b) {
  if (type_a != type_b) {
    return port::InternalError("Types of inputs mismatch");
  }

  // GPUs < sm_50 don't support cublasGemmEx.
  CudaComputeCapability cc = stream->GetCudaComputeCapability();
  if (cc.major < 5) {
    return port::InternalError(absl::StrCat(
        "sm_", cc.major, " does not support explicit gemm algorithms."));
  }

  bool algo_uses_tensor_ops = UsesTensorOps(algorithm);
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
  if (algo_uses_tensor_ops) {
    if (cc.major < 7) {
      return port::InternalError(absl::StrCat(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available in sm", cc.major,
          "X devices."));
    } else if (type_a == blas::DataType::kFloat) {
#if CUDA_VERSION < 11000
      return port::InternalError(absl::StrCat(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available for fp32"));
#else
      if (cc.major < 8) {
        return port::InternalError(absl::StrCat(
            "Algorithm ", algorithm,
            " uses tensor ops, but tensor ops are not available in sm",
            cc.major, "X devices for float input types."));
      } else if (!tensorflow::tensor_float_32_execution_enabled()) {
        return port::InternalError(absl::StrCat(
            "Algorithm ", algorithm,
            " uses tensor ops, but tensor ops are disabled for fp32 inputs"));
      }
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
#endif
    } else if (type_a == blas::DataType::kHalf) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#endif
    } else {
      return port::InternalError(
          absl::StrCat("Algorithm ", algorithm,
                       " uses tensor ops which are not supported for input"));
    }
  }

  // Return false if we might be hitting a cuBLAS bug that produces the wrong
  // result. See nvbugs/2156201, b/79126339.
#if CUDA_VERSION >= 9000 && CUDA_VERSION < 9020
  if ((algorithm == CUBLAS_GEMM_DEFAULT || algorithm >= CUBLAS_GEMM_ALGO13) &&
      std::max({m, n, k}) >= 2097153 && cc_major < 7) {
    return port::InternalError(
        "DoBlasGemmWithAlgorithm returning false to work around cudnn "
        "<9.2 bug with m, n, or k >= 2097153.  See b/79126339.");
  }
#endif
  return math_type;
}

static port::StatusOr<std::unique_ptr<GpuTimer, GpuTimerDeleter>>
StartGpuTimerForProfile(Stream *stream, GpuExecutor *executor,
                        blas::ProfileResult *output_profile_result) {
  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result) {
    timer.reset(new GpuTimer(executor));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return port::InternalError(
          "output_profile_result given, but unable to create a GpuTimer");
    }
  }
  return timer;
}

static port::Status PopulateProfileFromTimer(
    GpuTimer *timer, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result, Stream *stream) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_151(mht_151_v, 2776, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "PopulateProfileFromTimer");

  if (timer) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return port::InternalError("unable to stop GpuTimer.");
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return port::Status::OK();
}

port::Status CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_152(mht_152_v, 2800, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmWithAlgorithm");

  TF_ASSIGN_OR_RETURN(cublasMath_t math_type,
                      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b));

  TF_ASSIGN_OR_RETURN(auto timer, StartGpuTimerForProfile(
                                      stream, parent_, output_profile_result));

  // Since we are converting 'algorithm' to cublasGemmAlgo_t by static_cast,
  // we do the following compile-time check on the default value:
  static_assert(blas::kDefaultGemmAlgo == CUBLAS_GEMM_DFALT, "");

  TF_RETURN_IF_ERROR(DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true, math_type,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, alpha,
      a.opaque(), GetCUDADataType(type_a), lda, b.opaque(),
      GetCUDADataType(type_b), ldb, beta, c->opaque(), GetCUDADataType(type_c),
      ldc, CUDAComputationType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm)));
  TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                              output_profile_result, stream));
  return port::Status::OK();
}

port::Status CUDABlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_153(mht_153_v, 2833, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmStridedBatchedWithAlgorithm");

  TF_ASSIGN_OR_RETURN(cublasMath_t math_type,
                      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b));
  TF_ASSIGN_OR_RETURN(auto timer, StartGpuTimerForProfile(
                                      stream, parent_, output_profile_result));

  cudaDataType_t cuda_in_type = GetCUDADataType(type_a);

#if CUDA_VERSION >= 11000
  // Workaround CUDA bug where batched GEMM is erroneously marked as
  // unsupported by manually unbatching it on Pascal.
  if (cuda_in_type == CUDA_R_16BF &&
      !stream->GetCudaComputeCapability().IsAtLeast(7)) {
    for (int batch = 0; batch < batch_count; ++batch) {
      const auto *a_matrix = reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<const Eigen::bfloat16 *>(a.opaque()) + batch * stride_a);
      const auto *b_matrix = reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<const Eigen::bfloat16 *>(b.opaque()) + batch * stride_b);
      auto *c_matrix = reinterpret_cast<__nv_bfloat16 *>(
          static_cast<Eigen::bfloat16 *>(c->opaque()) + batch * stride_c);
      TF_RETURN_IF_ERROR(DoBlasInternalImpl(
          AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true,
          math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
          k, static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
          b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
          c_matrix, CUDA_R_16BF, ldc, CUDAComputationType(computation_type),
          static_cast<cublasGemmAlgo_t>(algorithm)));
    }
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                                output_profile_result, stream));
    return port::Status::OK();
  }
#endif

  TF_RETURN_IF_ERROR(DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmStridedBatchedEx), stream, /*pointer_mode_host=*/true,
      math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      alpha, a.opaque(), cuda_in_type, lda, stride_a, b.opaque(), cuda_in_type,
      ldb, stride_b, beta, c->opaque(), GetCUDADataType(type_c), ldc, stride_c,
      batch_count, CUDAComputationType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm)));
  TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                              output_profile_result, stream));
  return port::Status::OK();
}

bool CUDABlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_154(mht_154_v, 2883, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::GetBlasGemmAlgorithms");

  // cublasGemmAlgo_t (and the function that accepts this type, cublasGemmEx)
  // were first introduced in CUDA 8.
  //
  // Note that when CUDA version and compute capability is not sufficient, we
  // still return the out_algorithms. Caller needs to make sure that in this
  // case, the returned vector is empty.
  *out_algorithms = {
    CUBLAS_GEMM_DFALT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
#if CUDA_VERSION >= 9000
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_DFALT_TENSOR_OP,
    CUBLAS_GEMM_ALGO0_TENSOR_OP,
    CUBLAS_GEMM_ALGO1_TENSOR_OP,
    CUBLAS_GEMM_ALGO2_TENSOR_OP,
    CUBLAS_GEMM_ALGO3_TENSOR_OP,
    CUBLAS_GEMM_ALGO4_TENSOR_OP,
#endif
#if CUDA_VERSION >= 9020
    CUBLAS_GEMM_ALGO18,
    CUBLAS_GEMM_ALGO19,
    CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21,
    CUBLAS_GEMM_ALGO22,
    CUBLAS_GEMM_ALGO23,
    CUBLAS_GEMM_ALGO5_TENSOR_OP,
    CUBLAS_GEMM_ALGO6_TENSOR_OP,
    CUBLAS_GEMM_ALGO7_TENSOR_OP,
    CUBLAS_GEMM_ALGO8_TENSOR_OP,
    CUBLAS_GEMM_ALGO9_TENSOR_OP,
    CUBLAS_GEMM_ALGO10_TENSOR_OP,
    CUBLAS_GEMM_ALGO11_TENSOR_OP,
    CUBLAS_GEMM_ALGO12_TENSOR_OP,
    CUBLAS_GEMM_ALGO13_TENSOR_OP,
    CUBLAS_GEMM_ALGO14_TENSOR_OP,
    CUBLAS_GEMM_ALGO15_TENSOR_OP,
#endif
  };
  return true;
}

template <typename T>
struct HalfAsFloat {
  typedef T type;
};

template <>
struct HalfAsFloat<Eigen::half> {
  typedef float type;
};

namespace {
// pass-through for non-complex types that don't need conversion to
// cublas-specific type.
template <typename T>
T inline GpuComplexValue(T v) {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_155(mht_155_v, 2958, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "GpuComplexValue");

  return v;
}
}  // namespace

template <typename T, typename Scalar, typename FuncT>
port::Status CUDABlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64 n, uint64 k, Scalar alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    Scalar beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_156(mht_156_v, 2973, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatchedInternal");

  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename HalfAsFloat<typename GpuComplexT<T>::type>::type CUDA_T;

  const size_t size = batch_count * sizeof(CUDA_T *);

  // Device-side copy of pointers to matrices.
  DeviceMemory<CUDA_T *> a;
  DeviceMemory<CUDA_T *> b;
  DeviceMemory<CUDA_T *> c;

  // If temporary space is allocated for device-side copies of pointers to
  // matrices, that temporary space should not be freed until this function
  // returns. Although the values for these unique_ptrs are not set here, they
  // are declared at this scope so they will be destroyed when the function
  // returns.
  //
  // If a scratch allocator is provided, these pointers will not be used at all.
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> a_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> b_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> c_temporary;

  // Decide how to allocate device-side copy of pointers to matrices based on
  // whether a scratch allocator was passed.
  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
                        scratch_allocator->AllocateBytes(size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
                        scratch_allocator->AllocateBytes(size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
                        scratch_allocator->AllocateBytes(size));
    a = DeviceMemory<CUDA_T *>(a_bytes);
    b = DeviceMemory<CUDA_T *>(b_bytes);
    c = DeviceMemory<CUDA_T *>(c_bytes);
  } else {
    SE_ASSIGN_OR_RETURN(a_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(b_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(c_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    a = DeviceMemory<CUDA_T *>(*a_temporary->mutable_device_memory());
    b = DeviceMemory<CUDA_T *>(*b_temporary->mutable_device_memory());
    c = DeviceMemory<CUDA_T *>(*c_temporary->mutable_device_memory());
  }

  if (!stream->ThenMemcpy(&a, a_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&b, b_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&c, c_raw_ptrs.data(), size).ok()) {
    return port::Status(port::error::INTERNAL,
                        "failed to copy memory from host to device in "
                        "CUDABlas::DoBlasGemmBatched");
  }

  cudaDataType_t data_type = CUDADataType<T>::type;

#if CUDA_VERSION >= 9010
  if (stream->GetCudaComputeCapability().IsAtLeast(5)) {
    cublasMath_t math_type;
    cublasGemmAlgo_t algo;
    if (data_type == CUDA_R_16F) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#else
      math_type = CUBLAS_DEFAULT_MATH;
#endif
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#if CUBLAS_VER_MAJOR >= 11
    } else if (data_type == CUDA_R_32F) {
      // DoBlassInternalImpl will switch math_type back to CUBLAS_DEFAULT_MATH
      // if TensorFloat-32 is disabled.
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
      algo = tensorflow::tensor_float_32_execution_enabled()
                 ? CUBLAS_GEMM_DFALT_TENSOR_OP
                 : CUBLAS_GEMM_DFALT;
#endif
    } else {
      math_type = CUBLAS_DEFAULT_MATH;
      algo = CUBLAS_GEMM_DFALT;
    }
    cudaDataType_t compute_type =
        (data_type == CUDA_R_16F ? CUDA_R_32F : data_type);
    const void **a_void_ptrs = reinterpret_cast<const void **>(
        const_cast<const CUDA_T **>(GpuMemory(a)));
    const void **b_void_ptrs = reinterpret_cast<const void **>(
        const_cast<const CUDA_T **>(GpuMemory(b)));
    void **c_void_ptrs =
        reinterpret_cast<void **>(const_cast<CUDA_T **>(GpuMemory(c)));
    return DoBlasInternalImpl(
        AS_LAMBDA(cublasGemmBatchedEx), stream, true /* = pointer_mode_host */,
        math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
        k, &alpha, a_void_ptrs, data_type, lda, b_void_ptrs, data_type, ldb,
        &beta, c_void_ptrs, data_type, ldc, batch_count, compute_type, algo);
  }
#endif
  // either CUDA_VERSION < 9.1 or SM < 5.0
  if (data_type != CUDA_R_16F) {
    auto cb_alpha = GpuComplexValue(alpha);
    auto cb_beta = GpuComplexValue(beta);
    bool ok = DoBlasInternal(
        cublas_func, stream, true /* = pointer_mode_host */,
        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
        GpuComplex(&cb_alpha), const_cast<const CUDA_T **>(GpuMemory(a)), lda,
        const_cast<const CUDA_T **>(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
        const_cast<CUDA_T **>(GpuMemory(c)), ldc, batch_count);
    if (ok) {
      return port::Status::OK();
    }
    return port::Status(port::error::INTERNAL,
                        "failed BLAS call, see log for details");
  } else {
    // Fall back to a loop for fp16
    for (int b = 0; b < batch_count; ++b) {
      const DeviceMemory<T> &a_matrix = *a_ptrs_to_wrappers[b];
      const DeviceMemory<T> &b_matrix = *b_ptrs_to_wrappers[b];
      DeviceMemory<T> *c_matrix = c_ptrs_to_wrappers[b];
      TF_RETURN_IF_ERROR(DoBlasGemm(
          stream, transa, transb, m, n, k, blas::ToDataType<T>::value, &alpha,
          a_matrix, lda, b_matrix, ldb, &beta, c_matrix, ldc));
    }
    return port::Status::OK();
  }
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b_array, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<Eigen::half> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_157(mht_157_v, 3112, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatched");

  // Note: The func passed here (cublasSgemmBatched) is not actually called,
  // due to special handling of fp16 inside DoBlasGemmBatchedInternal.
  port::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_158(mht_158_v, 3133, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatched");

  port::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_159(mht_159_v, 3152, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatched");

  port::Status status = DoBlasGemmBatchedInternal(
      cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_160(mht_160_v, 3173, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatched");

  port::Status status = DoBlasGemmBatchedInternal(
      cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_161(mht_161_v, 3194, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmBatched");

  port::Status status = DoBlasGemmBatchedInternal(
      cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

port::Status CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_162(mht_162_v, 3212, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasGemmStridedBatched");

  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#if CUDA_VERSION < 11000
  if (dtype == dnn::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == dnn::kFloat) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
  }
#endif

  switch (dtype) {
#if CUDA_VERSION >= 11000
    case dnn::kBF16: {
      CudaComputeCapability cc = stream->GetCudaComputeCapability();
      if (cc.IsAtLeast(7)) {
        cublasGemmAlgo_t algo =
            (cc.major >= 7 ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
        return DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmStridedBatchedEx), stream,
            true /* = pointer_mode_host */, math_type,
            CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
            alpha, a.opaque(), CUDA_R_16BF, lda, stride_a, b.opaque(),
            CUDA_R_16BF, ldb, stride_b, beta, c->opaque(), CUDA_R_16BF, ldc,
            stride_c, batch_count,
            /*compute_type=*/CUDA_R_32F, algo);
      }
      // Fall back to a loop.
      for (int batch = 0; batch < batch_count; ++batch) {
        const auto *a_matrix = reinterpret_cast<const __nv_bfloat16 *>(
            static_cast<const Eigen::bfloat16 *>(a.opaque()) +
            batch * stride_a);
        const auto *b_matrix = reinterpret_cast<const __nv_bfloat16 *>(
            static_cast<const Eigen::bfloat16 *>(b.opaque()) +
            batch * stride_b);
        auto *c_matrix = reinterpret_cast<__nv_bfloat16 *>(
            static_cast<Eigen::bfloat16 *>(c->opaque()) + batch * stride_c);
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            cublasSgemmEx, stream, true /* = pointer_mode_host */,
            CUBLAS_DEFAULT_MATH, CUDABlasTranspose(transa),
            CUDABlasTranspose(transb), m, n, k,
            static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
            b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
            c_matrix, CUDA_R_16BF, ldc));
      }
      return port::Status::OK();
    }
#endif
    case dnn::kHalf: {
#if CUDA_VERSION >= 9010
      CudaComputeCapability cc = stream->GetCudaComputeCapability();
      if (cc.major >= 5) {
        cublasGemmAlgo_t algo =
            (cc.major >= 7 ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
        return DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmStridedBatchedEx), stream,
            true /* = pointer_mode_host */, math_type,
            CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
            alpha, a.opaque(), CUDA_R_16F, lda, stride_a, b.opaque(),
            CUDA_R_16F, ldb, stride_b, beta, c->opaque(), CUDA_R_16F, ldc,
            stride_c, batch_count, CUDA_R_32F, algo);
      }
#endif
      // Either CUDA_VERSION < 9.1 or SM < 5.0. Fall back to a loop.
      for (int batch = 0; batch < batch_count; ++batch) {
        const auto *a_matrix = reinterpret_cast<const __half *>(
            static_cast<const Eigen::half *>(a.opaque()) + batch * stride_a);
        const auto *b_matrix = reinterpret_cast<const __half *>(
            static_cast<const Eigen::half *>(b.opaque()) + batch * stride_b);
        auto *c_matrix = reinterpret_cast<__half *>(
            static_cast<Eigen::half *>(c->opaque()) + batch * stride_c);
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            cublasSgemmEx, stream, true /* = pointer_mode_host */,
            CUBLAS_DEFAULT_MATH, CUDABlasTranspose(transa),
            CUDABlasTranspose(transb), m, n, k,
            static_cast<const float *>(alpha), a_matrix, SE_CUDA_DATA_HALF, lda,
            b_matrix, SE_CUDA_DATA_HALF, ldb, static_cast<const float *>(beta),
            c_matrix, SE_CUDA_DATA_HALF, ldc));
      }
      return port::Status::OK();
    }
    case dnn::kFloat: {
      return DoBlasInternalImpl(
          cublasSgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
          k, static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda, stride_a,
          static_cast<const float *>(b.opaque()), ldb, stride_b,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc, stride_c, batch_count);
    }
    case dnn::kDouble:
      return DoBlasInternalImpl(
          cublasDgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
          k, static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda, stride_a,
          static_cast<const double *>(b.opaque()), ldb, stride_b,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc, stride_c, batch_count);
    case dnn::kComplexFloat: {
      GpuComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<float> *>(alpha));
      GpuComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
          k, GpuComplex(&cb_alpha),
          static_cast<const GpuComplexType *>(a.opaque()), lda, stride_a,
          static_cast<const GpuComplexType *>(b.opaque()), ldb, stride_b,
          GpuComplex(&cb_beta), static_cast<GpuComplexType *>(c->opaque()), ldc,
          stride_c, batch_count);
    }
    case dnn::kComplexDouble: {
      GpuDoubleComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<double> *>(alpha));
      GpuDoubleComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n,
          k, GpuComplex(&cb_alpha),
          static_cast<const GpuDoubleComplexType *>(a.opaque()), lda, stride_a,
          static_cast<const GpuDoubleComplexType *>(b.opaque()), ldb, stride_b,
          GpuComplex(&cb_beta),
          static_cast<GpuDoubleComplexType *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_163(mht_163_v, 3357, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHemm");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChemm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_164(mht_164_v, 3376, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHemm");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhemm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_165(mht_165_v, 3394, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHerk");

  return DoBlasInternal(cublasCherk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuComplex(GpuMemory(a)), lda, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_166(mht_166_v, 3409, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHerk");

  return DoBlasInternal(cublasZherk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuComplex(GpuMemory(a)), lda, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_167(mht_167_v, 3425, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer2k");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCher2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_168(mht_168_v, 3443, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasHer2k");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZher2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_169(mht_169_v, 3459, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymm");

  return DoBlasInternal(cublasSsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_170(mht_170_v, 3473, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymm");

  return DoBlasInternal(cublasDsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_171(mht_171_v, 3489, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymm");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_172(mht_172_v, 3508, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSymm");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_173(mht_173_v, 3524, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyrk");

  return DoBlasInternal(cublasSsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_174(mht_174_v, 3537, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyrk");

  return DoBlasInternal(cublasDsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_175(mht_175_v, 3552, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyrk");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_176(mht_176_v, 3570, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyrk");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_177(mht_177_v, 3587, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2k");

  return DoBlasInternal(cublasSsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_178(mht_178_v, 3601, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2k");

  return DoBlasInternal(cublasDsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_179(mht_179_v, 3617, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2k");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_180(mht_180_v, 3636, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasSyr2k");

  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_181(mht_181_v, 3653, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmm");

  return DoBlasInternal(cublasStrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb,
                        GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_182(mht_182_v, 3668, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmm");

  return DoBlasInternal(cublasDtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb,
                        GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_183(mht_183_v, 3684, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmm");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_184(mht_184_v, 3702, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrmm");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_185(mht_185_v, 3719, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsm");

  return DoBlasInternal(cublasStrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_186(mht_186_v, 3733, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsm");

  return DoBlasInternal(cublasDtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_187(mht_187_v, 3748, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsm");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_188(mht_188_v, 3765, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsm");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_189(mht_189_v, 3782, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsmBatched");

  return DoBlasInternal(cublasStrsmBatched, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_190(mht_190_v, 3798, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsmBatched");

  return DoBlasInternal(cublasDtrsmBatched, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_191(mht_191_v, 3816, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsmBatched");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(
      cublasCtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<float2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<float2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_192(mht_192_v, 3836, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::DoBlasTrsmBatched");

  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(
      cublasZtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<double2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<double2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

// We only use cublasLt from CUDA 11.0 onward.
#if CUDA_VERSION >= 11000

namespace {

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatrixLayout_t handle,
                                    cublasLtMatrixLayoutAttribute_t attr,
                                    const T &value) {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_193(mht_193_v, 3857, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "SetCublasLtAttr");

  cublasStatus_t status =
      cublasLtMatrixLayoutSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulAlgo_t *handle,
                                    cublasLtMatmulAlgoConfigAttributes_t attr,
                                    const T &value) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_194(mht_194_v, 3875, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "SetCublasLtAttr");

  cublasStatus_t status =
      cublasLtMatmulAlgoConfigSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulAlgoConfigSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulPreference_t handle,
                                    cublasLtMatmulPreferenceAttributes_t attr,
                                    const T &value) {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_195(mht_195_v, 3893, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "SetCublasLtAttr");

  cublasStatus_t status =
      cublasLtMatmulPreferenceSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulPreferenceSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline bool GetCublasLtAttr(const cublasLtMatmulAlgo_t *handle,
                            cublasLtMatmulAlgoConfigAttributes_t attr,
                            T *value) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_196(mht_196_v, 3911, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "GetCublasLtAttr");

  auto mutable_handle = const_cast<cublasLtMatmulAlgo_t *>(handle);
  size_t bytes_written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(mutable_handle, attr, value,
                                              sizeof(T), &bytes_written) ==
             CUBLAS_STATUS_SUCCESS &&
         bytes_written == sizeof(T);
}

template <typename T>
inline const T &ValueForStrCat(const T &value) {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_197(mht_197_v, 3924, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ValueForStrCat");

  return value;
}
template <typename T>
inline absl::Hex ValueForStrCat(T *ptr) {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_198(mht_198_v, 3931, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ValueForStrCat");

  return absl::Hex(reinterpret_cast<uintptr_t>(ptr));
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulDesc_t handle,
                                    cublasLtMatmulDescAttributes_t attr,
                                    const T &value) {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_199(mht_199_v, 3941, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "SetCublasLtAttr");

  cublasStatus_t status =
      cublasLtMatmulDescSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescSetAttribute(attr=", attr, ", value=",
                     ValueForStrCat(value), ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

struct MatmulDescDestroyer {
  void operator()(cublasLtMatmulDesc_t matmul_desc) const {
    cublasLtMatmulDescDestroy(matmul_desc);
  }
};
struct LayoutDestroyer {
  void operator()(cublasLtMatrixLayout_t layout) const {
    cublasLtMatrixLayoutDestroy(layout);
  }
};
struct MatmulPreferenceDestroyer {
  void operator()(cublasLtMatmulPreference_t matmul_pref) const {
    cublasLtMatmulPreferenceDestroy(matmul_pref);
  }
};
using UniqueOpDesc =
    std::unique_ptr<std::remove_pointer<cublasLtMatmulDesc_t>::type,
                    MatmulDescDestroyer>;
using UniqueLayoutDesc =
    std::unique_ptr<std::remove_pointer<cublasLtMatrixLayout_t>::type,
                    LayoutDestroyer>;
using UniqueMatmulPreference =
    std::unique_ptr<std::remove_pointer<cublasLtMatmulPreference_t>::type,
                    MatmulPreferenceDestroyer>;

port::StatusOr<UniqueOpDesc> CreateCublasLtOperationDesc(
    blas::ComputationType computation_type, blas::DataType scale_type,
    blas::PointerMode pointer_mode, blas::Epilogue epilogue,
    blas::Transpose transa, blas::Transpose transb) {
  cublasLtMatmulDesc_t desc;
  cublasComputeType_t cublas_compute_type =
      CUBLASComputationType(computation_type);
  cudaDataType_t cuda_scale_type = GetCUDADataType(scale_type);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&desc, cublas_compute_type, cuda_scale_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescCreate(computation_type=",
                     computation_type, ") failed: ", ToString(status)));
  }
  UniqueOpDesc unique_desc(desc);
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                     CUBLASPointerMode(pointer_mode)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     CUBLASEpilogue(epilogue)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     CUDABlasTranspose(transa)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     CUDABlasTranspose(transb)));
  return unique_desc;
}

port::StatusOr<UniqueLayoutDesc> CreateCublasLtLayoutDesc(
    blas::DataType data_type, uint64_t rows, uint64 cols, int64_t ld,
    int64_t stride, int batch_count) {
  cublasLtMatrixLayout_t desc;
  cublasStatus_t status = cublasLtMatrixLayoutCreate(
      &desc, GetCUDADataType(data_type), rows, cols, ld);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutCreate failed: ", ToString(status)));
  }
  UniqueLayoutDesc unique_desc(desc);
  SE_RETURN_IF_ERROR(
      SetCublasLtAttr(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_count));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(
      desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride));
  return unique_desc;
}

// Helper function to allocate workspace.
port::Status AllocateWorkspace(void **workspace,
                               ScratchAllocator *scratch_allocator,
                               size_t num_bytes) {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_200(mht_200_v, 4031, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "AllocateWorkspace");

  SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> workspace_bytes,
                      scratch_allocator->AllocateBytes(num_bytes));
  *workspace = (void *)GpuMemoryMutable(&workspace_bytes);
  return port::Status::OK();
}

template <typename T>
blas::ComputationType ToComputationType();
template <>
blas::ComputationType ToComputationType<Eigen::half>() {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_201(mht_201_v, 4044, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToComputationType<Eigen::half>");

  return blas::ComputationType::kF16;
}
template <>
blas::ComputationType ToComputationType<float>() {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_202(mht_202_v, 4051, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToComputationType<float>");

  return blas::ComputationType::kF32;
}
template <>
blas::ComputationType ToComputationType<double>() {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_203(mht_203_v, 4058, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToComputationType<double>");

  return blas::ComputationType::kF64;
}
template <>
blas::ComputationType ToComputationType<std::complex<float>>() {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_204(mht_204_v, 4065, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToComputationType<std::complex<float>>");

  return blas::ComputationType::kComplexF32;
}
template <>
blas::ComputationType ToComputationType<std::complex<double>>() {
   std::vector<std::string> mht_205_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_205(mht_205_v, 4072, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ToComputationType<std::complex<double>>");

  return blas::ComputationType::kComplexF64;
}

class CUDABlasLtMatmulPlan final : public blas::IBlasLtMatmulPlan {
 public:
  port::Status init(const blas::BlasLtMatmulPlanParams &p) {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_206(mht_206_v, 4081, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "init");

    params_ = p;
    scale_type_ = GetScaleType(p.c_type, p.computation_type);
    SE_ASSIGN_OR_RETURN(
        op_desc_,
        CreateCublasLtOperationDesc(
            p.computation_type, GetScaleType(p.c_type, p.computation_type),
            p.pointer_mode, p.epilogue, p.transa, p.transb));
    uint64_t rows_a = p.transa == blas::Transpose::kNoTranspose ? p.m : p.k;
    uint64_t cols_a = p.transa == blas::Transpose::kNoTranspose ? p.k : p.m;
    uint64_t rows_b = p.transb == blas::Transpose::kNoTranspose ? p.k : p.n;
    uint64_t cols_b = p.transb == blas::Transpose::kNoTranspose ? p.n : p.k;
    SE_ASSIGN_OR_RETURN(
        a_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda,
                                          p.stride_a, capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        b_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb,
                                          p.stride_b, capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        c_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                          capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        d_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                          capped_batch_count()));
    remainder_batch_count_ =
        p.batch_count > kMaxBatchCount ? p.batch_count % kMaxBatchCount : 0;
    if (remainder_batch_count_) {
      SE_ASSIGN_OR_RETURN(
          a_remainder_desc_,
          CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda, p.stride_a,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          b_remainder_desc_,
          CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb, p.stride_b,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          c_remainder_desc_,
          CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          d_remainder_desc_,
          CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                   remainder_batch_count_));
    }
    return port::Status::OK();
  }

  cublasLtMatmulDesc_t op_desc() const {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_207(mht_207_v, 4131, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "op_desc");
 return op_desc_.get(); }
  cublasLtMatrixLayout_t a_desc() const {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_208(mht_208_v, 4135, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "a_desc");
 return a_desc_.get(); }
  cublasLtMatrixLayout_t b_desc() const {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_209(mht_209_v, 4139, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "b_desc");
 return b_desc_.get(); }
  cublasLtMatrixLayout_t c_desc() const {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_210(mht_210_v, 4143, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "c_desc");
 return c_desc_.get(); }
  cublasLtMatrixLayout_t d_desc() const {
   std::vector<std::string> mht_211_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_211(mht_211_v, 4147, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "d_desc");
 return d_desc_.get(); }
  cublasLtMatrixLayout_t a_remainder_desc() const {
   std::vector<std::string> mht_212_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_212(mht_212_v, 4151, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "a_remainder_desc");

    return a_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t b_remainder_desc() const {
   std::vector<std::string> mht_213_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_213(mht_213_v, 4157, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "b_remainder_desc");

    return b_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t c_remainder_desc() const {
   std::vector<std::string> mht_214_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_214(mht_214_v, 4163, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "c_remainder_desc");

    return c_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t d_remainder_desc() const {
   std::vector<std::string> mht_215_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_215(mht_215_v, 4169, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "d_remainder_desc");

    return d_remainder_desc_.get();
  }

  const blas::BlasLtMatmulPlanParams &params() const {
   std::vector<std::string> mht_216_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_216(mht_216_v, 4176, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "params");
 return params_; }
  blas::DataType scale_type() const {
   std::vector<std::string> mht_217_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_217(mht_217_v, 4180, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "scale_type");
 return scale_type_; }
  blas::DataType ab_type() const override {
   std::vector<std::string> mht_218_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_218(mht_218_v, 4184, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "ab_type");
 return params_.ab_type; }
  blas::DataType c_type() const override {
   std::vector<std::string> mht_219_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_219(mht_219_v, 4188, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "c_type");
 return params_.c_type; }
  int capped_batch_count() const {
   std::vector<std::string> mht_220_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_220(mht_220_v, 4192, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "capped_batch_count");

    return std::min(params_.batch_count, kMaxBatchCount);
  }
  int remainder_batch_count() const {
   std::vector<std::string> mht_221_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_221(mht_221_v, 4198, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "remainder_batch_count");
 return remainder_batch_count_; }

  // Note: Must be const to satisfy API. This is always called before the plan
  // is executed, so the state change is not observed in subsequent executions.
  bool SetBiasPointer(const void *bias) const;

 private:
  // In some cases cublasLt does not support large batch sizes, so we need to
  // split up such cases into multiple calls.
  static constexpr int kMaxBatchCount = 65535;
  blas::BlasLtMatmulPlanParams params_;
  blas::DataType scale_type_;
  UniqueOpDesc op_desc_;
  // These have batch count set to capped_batch_count().
  UniqueLayoutDesc a_desc_;
  UniqueLayoutDesc b_desc_;
  UniqueLayoutDesc c_desc_;
  UniqueLayoutDesc d_desc_;
  int remainder_batch_count_;
  // These have batch count set to remainder_batch_count_, and are only created
  // if params_.batch_count > kMaxBatchSize.
  UniqueLayoutDesc a_remainder_desc_;
  UniqueLayoutDesc b_remainder_desc_;
  UniqueLayoutDesc c_remainder_desc_;
  UniqueLayoutDesc d_remainder_desc_;
};

/*static*/ constexpr int CUDABlasLtMatmulPlan::kMaxBatchCount;

bool CUDABlasLtMatmulPlan::SetBiasPointer(const void *bias) const {
   std::vector<std::string> mht_222_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_222(mht_222_v, 4230, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasLtMatmulPlan::SetBiasPointer");

  return SetCublasLtAttr(op_desc_.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                         bias)
      .ok();
}

class CUDABlasLtMatmulAlgorithm final : public blas::IBlasLtMatmulAlgorithm {
 public:
  CUDABlasLtMatmulAlgorithm(blas::AlgorithmType index,
                            cublasLtMatmulAlgo_t algo, size_t workspace_size)
      : index_(index), algo_(algo), workspace_size_(workspace_size) {
   std::vector<std::string> mht_223_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_223(mht_223_v, 4243, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlasLtMatmulAlgorithm");
}

  blas::AlgorithmType index() const override {
   std::vector<std::string> mht_224_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_224(mht_224_v, 4248, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "index");
 return index_; }

  size_t workspace_size() const override {
   std::vector<std::string> mht_225_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_225(mht_225_v, 4253, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "workspace_size");
 return workspace_size_; }

  const cublasLtMatmulAlgo_t *algo() const {
   std::vector<std::string> mht_226_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_226(mht_226_v, 4258, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "algo");
 return &algo_; }

  int algo_id() const {
   std::vector<std::string> mht_227_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_227(mht_227_v, 4263, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "algo_id");

    int id;
    GetCublasLtAttr(&algo_, CUBLASLT_ALGO_CONFIG_ID, &id);
    return id;
  }

 private:
  blas::AlgorithmType index_;
  cublasLtMatmulAlgo_t algo_;
  size_t workspace_size_;
};

port::StatusOr<UniqueMatmulPreference> CreateCublasLtMatmulPreference(
    const blas::IBlasLtMatmulPlan *plan, size_t max_workspace_bytes) {
  cublasLtMatmulPreference_t preference;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("cublasLtMatmulPreferenceCreate failed: ",
                                     ToString(status)));
  }
  UniqueMatmulPreference unique_preference(preference);
  SE_RETURN_IF_ERROR(SetCublasLtAttr(preference,
                                     CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     max_workspace_bytes));

  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  if (cuda_plan.params().batch_count == 0) {
    return unique_preference;
  }
  // This is a workaround for a known issue in cuBlasLt where the heuristic may
  // in rare cases select an algo that does not support the specified stride.
  // Specifying the alignment requirements manually like this avoids the issue.
  auto get_alignment_bytes = [](int64_t stride, blas::DataType dtype) {
   std::vector<std::string> mht_228_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_228(mht_228_v, 4299, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "lambda");

    return (stride & -stride) * GetDataTypeSizeBytes(dtype);
  };
  if (cuda_plan.params().stride_a) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_a,
                                    cuda_plan.params().ab_type)));
  }
  if (cuda_plan.params().stride_b) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_b,
                                    cuda_plan.params().ab_type)));
  }
  if (cuda_plan.params().stride_c) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_c,
                                    cuda_plan.params().c_type)));
  }
  if (cuda_plan.params().stride_c) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_c,
                                    cuda_plan.params().c_type)));
  }
  return unique_preference;
}

}  // namespace

#endif  // CUDA_VERSION >= 11000

port::StatusOr<std::unique_ptr<blas::IBlasLtMatmulPlan>>
CUDABlas::CreateBlasLtMatmulPlan(const blas::BlasLtMatmulPlanParams &p) {
   std::vector<std::string> mht_229_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_229(mht_229_v, 4337, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::CreateBlasLtMatmulPlan");

#if CUDA_VERSION >= 11000
  auto cuda_plan = std::make_unique<CUDABlasLtMatmulPlan>();
  SE_RETURN_IF_ERROR(cuda_plan->init(p));
  return static_cast<std::unique_ptr<blas::IBlasLtMatmulPlan>>(
      std::move(cuda_plan));
#else
  return port::Status(
      port::error::UNIMPLEMENTED,
      "CreateBlasLtMatmulPlan is not supported with this version of CUDA");
#endif
}

#if CUDA_VERSION >= 11000
port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
CUDABlas::GetBlasLtMatmulAlgorithmsInternal(const blas::IBlasLtMatmulPlan *plan,
                                            size_t max_workspace_size,
                                            int max_algorithm_count,
                                            bool for_remainder_batch) {
   std::vector<std::string> mht_230_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_230(mht_230_v, 4358, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::GetBlasLtMatmulAlgorithmsInternal");

  SE_ASSIGN_OR_RETURN(UniqueMatmulPreference preference,
                      CreateCublasLtMatmulPreference(plan, max_workspace_size));

  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    absl::MutexLock lock(&mu_);

    CHECK(blasLt_ != nullptr);

    gpu::ScopedActivateExecutorContext sac{parent_};

    int found_algorithm_count = 0;
    const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
    const auto &a_desc =
        for_remainder_batch ? cuda_plan.a_remainder_desc() : cuda_plan.a_desc();
    const auto &b_desc =
        for_remainder_batch ? cuda_plan.b_remainder_desc() : cuda_plan.b_desc();
    const auto &c_desc =
        for_remainder_batch ? cuda_plan.c_remainder_desc() : cuda_plan.c_desc();
    const auto &d_desc =
        for_remainder_batch ? cuda_plan.d_remainder_desc() : cuda_plan.d_desc();
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        blasLt_, cuda_plan.op_desc(), a_desc, b_desc, c_desc, d_desc,
        preference.get(), max_algorithm_count, results.data(),
        &found_algorithm_count);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return port::Status(
          port::error::INTERNAL,
          absl::StrCat("cublasLtMatmulAlgoGetHeuristic failed: ",
                       ToString(status)));
    }
    results.resize(found_algorithm_count);
  }

  std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>> out_algorithms;
  out_algorithms.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    if (result.state != CUBLAS_STATUS_SUCCESS) continue;  // Skip failed algos
    out_algorithms.emplace_back(std::make_unique<CUDABlasLtMatmulAlgorithm>(
        i, result.algo, result.workspaceSize));
  }
  return out_algorithms;
}
#endif

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
CUDABlas::GetBlasLtMatmulAlgorithms(const blas::IBlasLtMatmulPlan *plan,
                                    size_t max_workspace_size,
                                    int max_algorithm_count) {
   std::vector<std::string> mht_231_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_231(mht_231_v, 4411, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::GetBlasLtMatmulAlgorithms");

#if CUDA_VERSION >= 11000
  return GetBlasLtMatmulAlgorithmsInternal(plan, max_workspace_size,
                                           max_algorithm_count);
#else  // if CUDA_VERSION < 11000
  return port::Status(
      port::error::UNIMPLEMENTED,
      "GetBlasLtMatmulAlgorithms is not supported with this version of CUDA");
#endif
}

#if CUDA_VERSION >= 11000
bool CUDABlas::DoBlasLtMatmulInternal(
    Stream *stream, bool err_on_failure, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, DeviceMemoryBase d, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias) {
  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  const auto &cuda_algo =
      *static_cast<const CUDABlasLtMatmulAlgorithm *>(algorithm);

  if (alpha.data_type() != cuda_plan.scale_type() ||
      beta.data_type() != cuda_plan.scale_type()) {
    VLOG(2) << "DoBlasLtMatmul returning false because alpha and beta types do "
               "not match plan: expected "
            << cuda_plan.c_type() << ", got alpha=" << alpha.data_type()
            << " beta=" << beta.data_type();
    return false;
  }
  if (alpha.is_pointer() != beta.is_pointer()) {
    VLOG(2) << "DoBlasLtMatmul returning false because one of `alpha` "
               "and `beta` is a pointer, but the other is not.";
    return false;
  }
  bool is_pointer_mode_host = !alpha.is_pointer();
  if ((cuda_plan.params().pointer_mode == blas::PointerMode::kHost) !=
      is_pointer_mode_host) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "pointer_mode for the given alpha/beta.";
    return false;
  }
  if ((cuda_plan.params().epilogue == blas::Epilogue::kBias ||
       cuda_plan.params().epilogue == blas::Epilogue::kBiasThenReLU) !=
      (bias != nullptr)) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "epilogue for the given bias pointer.";
    return false;
  }
  const void *alpha_ptr = alpha.is_pointer() ? alpha.opaque_pointer().opaque()
                                             : alpha.opaque_value();
  const void *beta_ptr =
      beta.is_pointer() ? beta.opaque_pointer().opaque() : beta.opaque_value();

  void *workspace = nullptr;
  if (cuda_algo.workspace_size()) {
    port::Status allocation_status = AllocateWorkspace(
        &workspace, scratch_allocator, cuda_algo.workspace_size());
    if (!allocation_status.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR)
            << "Failed to allocate workspace for cublasLtMatmul algo with id: "
            << cuda_algo.algo_id() << " requiring "
            << cuda_algo.workspace_size() << " bytes of workspace";
      }
      return false;
    }
  }

  // This is only used when batch_count > kMaxBatchCount.
  std::unique_ptr<blas::IBlasLtMatmulAlgorithm> unique_remainder_algo;
  if (cuda_plan.remainder_batch_count()) {
    // There is no easy way to get the user-specified max workspace size here,
    // so we just allow a very small amount and don't worry too much about
    // performance because this is only used in rare cases. The same reasoning
    // applies to selection of the algorithm.
    size_t max_workspace_size = 4 * 1024 * 1024;  // 4 MiB
    auto status_or_algorithms =
        GetBlasLtMatmulAlgorithmsInternal(plan, max_workspace_size,
                                          /* max_algorithm_count = */ 1,
                                          /* for_remainder_batch = */ true);
    if (!status_or_algorithms.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "Failed to get algorithms for blasLt remainder batch.";
      }
      return false;
    }
    auto algorithms = status_or_algorithms.ConsumeValueOrDie();
    unique_remainder_algo = std::move(algorithms.front());
  }

  cudaStream_t cuda_stream = CUDAStream(stream);

  absl::MutexLock lock(&mu_);

  if (bias != nullptr) {
    if (!cuda_plan.SetBiasPointer(bias.opaque())) {
      VLOG(2) << "DoBlasLtMatmul returning false because setting the bias "
                 "pointer failed.";
      return false;
    }
  }

  CHECK(blasLt_ != nullptr);

  gpu::ScopedActivateExecutorContext sac{parent_};

  // Plan execution is broken down into repeat calls with capped_batch_count,
  // followed by a final call with remainder_batch_count.
  // Cases where batch_count <= kMaxBatchCount require only a single call (a
  // single loop iteration and no remainder).
  int ab_type_size = GetDataTypeSizeBytes(cuda_plan.params().ab_type);
  int c_type_size = GetDataTypeSizeBytes(cuda_plan.params().c_type);
  const char *a_ptr = static_cast<const char *>(a.opaque());
  const char *b_ptr = static_cast<const char *>(b.opaque());
  const char *c_ptr = static_cast<const char *>(c.opaque());
  char *d_ptr = static_cast<char *>(d.opaque());
  int capped_batch_count = cuda_plan.capped_batch_count();
  for (int batch = 0;
       batch + capped_batch_count <= cuda_plan.params().batch_count;
       batch += capped_batch_count) {
    cublasStatus_t ret = cublasLtMatmul(
        blasLt_, cuda_plan.op_desc(), alpha_ptr, a_ptr, cuda_plan.a_desc(),
        b_ptr, cuda_plan.b_desc(), beta_ptr, c_ptr, cuda_plan.c_desc(), d_ptr,
        cuda_plan.d_desc(), cuda_algo.algo(), workspace,
        cuda_algo.workspace_size(), cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run cublasLtMatmul routine: " << ToString(ret);
      }
      return false;
    }
    a_ptr += capped_batch_count * cuda_plan.params().stride_a * ab_type_size;
    b_ptr += capped_batch_count * cuda_plan.params().stride_b * ab_type_size;
    c_ptr += capped_batch_count * cuda_plan.params().stride_c * c_type_size;
    d_ptr += capped_batch_count * cuda_plan.params().stride_c * c_type_size;
  }
  // This is only used when batch_count > kMaxBatchCount.
  if (cuda_plan.remainder_batch_count()) {
    const auto &remainder_algo =
        *static_cast<const CUDABlasLtMatmulAlgorithm *>(
            unique_remainder_algo.get());
    if (remainder_algo.workspace_size()) {
      port::Status allocation_status = AllocateWorkspace(
          &workspace, scratch_allocator, remainder_algo.workspace_size());
      if (!allocation_status.ok()) {
        if (err_on_failure || VLOG_IS_ON(3)) {
          LOG(ERROR) << "Failed to allocate workspace for cublasLtMatmul algo "
                        "with id: "
                     << remainder_algo.algo_id() << " requiring "
                     << remainder_algo.workspace_size()
                     << " bytes of workspace";
        }
        return false;
      }
    }
    cublasStatus_t ret = cublasLtMatmul(
        blasLt_, cuda_plan.op_desc(), alpha_ptr, a_ptr,
        cuda_plan.a_remainder_desc(), b_ptr, cuda_plan.b_remainder_desc(),
        beta_ptr, c_ptr, cuda_plan.c_remainder_desc(), d_ptr,
        cuda_plan.d_remainder_desc(), remainder_algo.algo(), workspace,
        remainder_algo.workspace_size(), cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run remainder cublasLtMatmul routine: "
                   << ToString(ret);
      }
      return false;
    }
  }
  return true;
}
#endif  // CUDA_VERSION >= 11000

bool CUDABlas::DoBlasLtMatmul(
    Stream *stream, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias,
    blas::ProfileResult *output_profile_result) {
#if CUDA_VERSION >= 11000
  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  HostOrDeviceScalar<void> alpha_cast = alpha;
  HostOrDeviceScalar<void> beta_cast = beta;
  if (cuda_plan.c_type() == blas::DataType::kHalf &&
      cuda_plan.scale_type() == blas::DataType::kFloat) {
    // The given alpha and beta types are F16 (they always match c), but F32*
    // computation type requires that they be F32, so we must cast them.
    if (alpha.is_pointer() || beta.is_pointer()) {
      // We cannot easily convert a pointer to f16 memory to a pointer to f32
      // memory from here, so we don't support this for now.
      return false;
    }
    alpha_cast = HostOrDeviceScalar<void>(
        static_cast<float>(alpha.value<Eigen::half>()));
    beta_cast =
        HostOrDeviceScalar<void>(static_cast<float>(beta.value<Eigen::half>()));
  }

  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  bool err_on_failure = timer != nullptr;
  bool result = DoBlasLtMatmulInternal(stream, err_on_failure, plan, alpha_cast,
                                       a, b, beta_cast, c, c, scratch_allocator,
                                       algorithm, bias);

  if (timer && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm->index());
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
#else  // if CUDA_VERSION < 11000
  return false;
#endif
}

port::Status CUDABlas::GetVersion(std::string *version) {
   std::vector<std::string> mht_232_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_232(mht_232_v, 4644, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "CUDABlas::GetVersion");

  absl::MutexLock lock(&mu_);

  int v;
  auto status = cublasGetVersion(blas_, &v);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::InternalError(ToString(status));
  }
  *version = std::to_string(v);
  return port::Status::OK();
}

}  // namespace gpu

void initialize_cublas() {
   std::vector<std::string> mht_233_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_blasDTcc mht_233(mht_233_v, 4661, "", "./tensorflow/stream_executor/cuda/cuda_blas.cc", "initialize_cublas");

  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          cuda::kCudaPlatformId, gpu::kCuBlasPlugin, "cuBLAS",
          [](internal::StreamExecutorInterface *parent) -> blas::BlasSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuBLAS "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::CUDABlas *blas = new gpu::CUDABlas(cuda_executor);
            if (!blas->Init()) {
              // Note: Init() will log a more specific error.
              delete blas;
              return nullptr;
            }
            return blas;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuBLAS factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kBlas, gpu::kCuBlasPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { stream_executor::initialize_cublas(); });
