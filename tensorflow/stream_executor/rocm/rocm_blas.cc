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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc() {
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

#include "tensorflow/stream_executor/rocm/rocm_blas.h"

#include "tensorflow/stream_executor/rocm/rocblas_wrapper.h"

#define EIGEN_USE_GPU
#include <assert.h>

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

namespace wrap = tensorflow::wrap;

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const DeviceMemory<T> &a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_0(mht_0_v, 224, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "complex_cast");

  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemory(a));
}

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const T &a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_1(mht_1_v, 235, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "complex_cast");

  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(&a);
}
template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    DeviceMemory<T> *a) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_2(mht_2_v, 244, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "complex_cast");

  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemoryMutable(a));
}

static void blas_log(const char *c) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("c: \"" + (c == nullptr ? std::string("nullptr") : std::string((char*)c)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_3(mht_3_v, 254, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "blas_log");
}

static string ToString(rocblas_status status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_4(mht_4_v, 259, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ToString");

  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_5(mht_5_v, 283, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::Init");

  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret = wrap::rocblas_create_handle(&blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_6(mht_6_v, 298, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::ROCMBlas");
}

ROCMBlas::~ROCMBlas() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_7(mht_7_v, 303, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::~ROCMBlas");

  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    wrap::rocblas_destroy_handle(blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_8(mht_8_v, 313, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::SetStream");

  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret =
      wrap::rocblas_set_stream(blas_, AsGpuStreamValue(stream));
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into rocBLAS arguments.

rocblas_operation ROCMBlasTranspose(blas::Transpose trans) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_9(mht_9_v, 335, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlasTranspose");

  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return rocblas_operation_none;
    case blas::Transpose::kTranspose:
      return rocblas_operation_transpose;
    case blas::Transpose::kConjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

rocblas_fill ROCMBlasUpperLower(blas::UpperLower uplo) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_10(mht_10_v, 351, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlasUpperLower");

  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

rocblas_diagonal ROCMBlasDiagonal(blas::Diagonal diag) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_11(mht_11_v, 365, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlasDiagonal");

  switch (diag) {
    case blas::Diagonal::kUnit:
      return rocblas_diagonal_unit;
    case blas::Diagonal::kNonUnit:
      return rocblas_diagonal_non_unit;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

rocblas_side ROCMBlasSide(blas::Side side) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_12(mht_12_v, 379, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlasSide");

  switch (side) {
    case blas::Side::kLeft:
      return rocblas_side_left;
    case blas::Side::kRight:
      return rocblas_side_right;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret = rocblas_func(blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_13(mht_13_v, 417, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAsum");

  return DoBlasInternal(wrap::rocblas_sasum, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_14(mht_14_v, 428, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAsum");

  return DoBlasInternal(wrap::rocblas_dasum, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_15(mht_15_v, 439, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAsum");

  return DoBlasInternal(wrap::rocblas_scasum, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_16(mht_16_v, 450, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAsum");

  return DoBlasInternal(wrap::rocblas_dzasum, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_17(mht_17_v, 461, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAxpy");

  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_18(mht_18_v, 473, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAxpy");

  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_19(mht_19_v, 486, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAxpy");

  return DoBlasInternal(
      wrap::rocblas_caxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_20(mht_20_v, 498, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasAxpy");

  return DoBlasInternal(
      wrap::rocblas_zaxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_21(mht_21_v, 509, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasCopy");

  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_22(mht_22_v, 520, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasCopy");

  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_23(mht_23_v, 531, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasCopy");

  return DoBlasInternal(wrap::rocblas_ccopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_24(mht_24_v, 542, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasCopy");

  return DoBlasInternal(wrap::rocblas_zcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64_t elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_25(mht_25_v, 554, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDot");

  blas_log("DoBlasDot");
  return DoBlasInternal(
      wrap::rocblas_sdot, stream, /* pointer_mode_host = */ false, elem_count,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64_t elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_26(mht_26_v, 567, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDot");

  blas_log("DoBlasDot");
  return DoBlasInternal(
      wrap::rocblas_ddot, stream, /* pointer_mode_host = */ false, elem_count,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_27(mht_27_v, 580, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDotc");

  return DoBlasInternal(
      wrap::rocblas_cdotc, stream, /* pointer_mode_host = */ false, elem_count,
      complex_cast(x), incx, complex_cast(y), incy, complex_cast(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_28(mht_28_v, 592, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDotc");

  return DoBlasInternal(
      wrap::rocblas_zdotc, stream, /* pointer_mode_host = */ false, elem_count,
      complex_cast(x), incx, complex_cast(y), incy, complex_cast(result));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_29(mht_29_v, 604, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDotu");

  return DoBlasInternal(
      wrap::rocblas_cdotu, stream, /* pointer_mode_host = */ false, elem_count,
      complex_cast(x), incx, complex_cast(y), incy, complex_cast(result));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_30(mht_30_v, 616, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasDotu");

  return DoBlasInternal(
      wrap::rocblas_zdotu, stream, /* pointer_mode_host = */ false, elem_count,
      complex_cast(x), incx, complex_cast(y), incy, complex_cast(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_31(mht_31_v, 627, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasNrm2");

  return DoBlasInternal(wrap::rocblas_snrm2, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_32(mht_32_v, 638, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasNrm2");

  return DoBlasInternal(wrap::rocblas_dnrm2, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_33(mht_33_v, 649, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasNrm2");

  return DoBlasInternal(wrap::rocblas_scnrm2, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_34(mht_34_v, 660, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasNrm2");

  return DoBlasInternal(wrap::rocblas_dznrm2, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_35(mht_35_v, 671, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRot");

  return DoBlasInternal(
      wrap::rocblas_srot, stream, /* pointer_mode_host = */ true, elem_count,
      GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_36(mht_36_v, 683, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRot");

  return DoBlasInternal(
      wrap::rocblas_drot, stream, /* pointer_mode_host = */ true, elem_count,
      GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_37(mht_37_v, 695, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRot");

  return DoBlasInternal(wrap::rocblas_csrot, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64_t elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_38(mht_38_v, 707, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRot");

  return DoBlasInternal(wrap::rocblas_zdrot, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_39(mht_39_v, 718, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotg");

  return DoBlasInternal(wrap::rocblas_srotg, stream,
                        /* pointer_mode_host = */ false, GpuMemoryMutable(a),
                        GpuMemoryMutable(b), GpuMemoryMutable(c),
                        GpuMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_40(mht_40_v, 730, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotg");

  return DoBlasInternal(wrap::rocblas_drotg, stream,
                        /* pointer_mode_host = */ false, GpuMemoryMutable(a),
                        GpuMemoryMutable(b), GpuMemoryMutable(c),
                        GpuMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_41(mht_41_v, 743, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotg");

  return DoBlasInternal(wrap::rocblas_crotg, stream,
                        /* pointer_mode_host = */ false, complex_cast(a),
                        complex_cast(b), GpuMemoryMutable(c), complex_cast(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_42(mht_42_v, 755, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotg");

  return DoBlasInternal(wrap::rocblas_zrotg, stream,
                        /* pointer_mode_host = */ false, complex_cast(a),
                        complex_cast(b), GpuMemoryMutable(c), complex_cast(s));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64_t elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_43(mht_43_v, 767, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotm");

  return DoBlasInternal(
      wrap::rocblas_srotm, stream, /* pointer_mode_host = */ false, elem_count,
      GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64_t elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_44(mht_44_v, 779, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotm");

  return DoBlasInternal(
      wrap::rocblas_drotm, stream, /* pointer_mode_host = */ false, elem_count,
      GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_45(mht_45_v, 791, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotmg");

  return DoBlasInternal(wrap::rocblas_srotmg, stream,
                        /* pointer_mode_host = */ false, GpuMemoryMutable(d1),
                        GpuMemoryMutable(d2), GpuMemoryMutable(x1),
                        GpuMemory(y1), GpuMemoryMutable(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_46(mht_46_v, 804, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasRotmg");

  return DoBlasInternal(wrap::rocblas_drotmg, stream,
                        /* pointer_mode_host = */ false, GpuMemoryMutable(d1),
                        GpuMemoryMutable(d2), GpuMemoryMutable(x1),
                        GpuMemory(y1), GpuMemoryMutable(param));
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_47(mht_47_v, 815, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  blas_log("DoBlasScal<float>");
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_48(mht_48_v, 826, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_49(mht_49_v, 836, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  return DoBlasInternal(wrap::rocblas_csscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_50(mht_50_v, 846, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  return DoBlasInternal(wrap::rocblas_zdscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_51(mht_51_v, 857, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  return DoBlasInternal(wrap::rocblas_cscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_52(mht_52_v, 868, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasScal");

  return DoBlasInternal(wrap::rocblas_zscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_53(mht_53_v, 879, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSwap");

  return DoBlasInternal(wrap::rocblas_sswap, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_54(mht_54_v, 890, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSwap");

  return DoBlasInternal(wrap::rocblas_dswap, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_55(mht_55_v, 901, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSwap");

  return DoBlasInternal(wrap::rocblas_cswap, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64_t elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_56(mht_56_v, 912, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSwap");

  return DoBlasInternal(wrap::rocblas_zswap, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_57(mht_57_v, 923, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamax");

  return DoBlasInternal(wrap::rocblas_isamax, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_58(mht_58_v, 934, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamax");

  return DoBlasInternal(wrap::rocblas_idamax, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_59(mht_59_v, 945, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamax");

  return DoBlasInternal(wrap::rocblas_icamax, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_60(mht_60_v, 956, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamax");

  return DoBlasInternal(wrap::rocblas_izamax, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_61(mht_61_v, 967, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamin");

  return DoBlasInternal(wrap::rocblas_isamin, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_62(mht_62_v, 978, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamin");

  return DoBlasInternal(wrap::rocblas_idamin, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_63(mht_63_v, 989, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamin");

  return DoBlasInternal(wrap::rocblas_icamin, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64_t elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_64(mht_64_v, 1000, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasIamin");

  return DoBlasInternal(wrap::rocblas_izamin, stream,
                        /* pointer_mode_host = */ false, elem_count,
                        complex_cast(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_65(mht_65_v, 1013, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGbmv");

  return DoBlasInternal(
      wrap::rocblas_sgbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, GpuMemory(a), lda,
      GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_66(mht_66_v, 1027, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGbmv");

  return DoBlasInternal(
      wrap::rocblas_dgbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, GpuMemory(a), lda,
      GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_67(mht_67_v, 1043, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGbmv");

  return DoBlasInternal(
      wrap::rocblas_cgbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, kl, ku, complex_cast(alpha),
      complex_cast(a), lda, complex_cast(x), incx, complex_cast(beta),
      complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_68(mht_68_v, 1060, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGbmv");

  return DoBlasInternal(
      wrap::rocblas_zgbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, kl, ku, complex_cast(alpha),
      complex_cast(a), lda, complex_cast(x), incx, complex_cast(beta),
      complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_69(mht_69_v, 1074, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemv");

  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_70(mht_70_v, 1089, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemv");

  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_71(mht_71_v, 1105, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemv");

  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_cgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_72(mht_72_v, 1121, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemv");

  blas_log("DoBlasGemv\n");
  return DoBlasInternal(
      wrap::rocblas_zgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64_t m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_73(mht_73_v, 1135, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGer");

  return DoBlasInternal(
      wrap::rocblas_sger, stream, /* pointer_mode_host = */ true, m, n, &alpha,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64_t m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_74(mht_74_v, 1147, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGer");

  return DoBlasInternal(
      wrap::rocblas_dger, stream, /* pointer_mode_host = */ true, m, n, &alpha,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_75(mht_75_v, 1160, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGerc");

  return DoBlasInternal(wrap::rocblas_cgerc, stream,
                        /* pointer_mode_host = */ true, m, n,
                        complex_cast(alpha), complex_cast(x), incx,
                        complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_76(mht_76_v, 1174, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGerc");

  return DoBlasInternal(wrap::rocblas_zgerc, stream,
                        /* pointer_mode_host = */ true, m, n,
                        complex_cast(alpha), complex_cast(x), incx,
                        complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_77(mht_77_v, 1188, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGeru");

  return DoBlasInternal(wrap::rocblas_cgeru, stream,
                        /* pointer_mode_host = */ true, m, n,
                        complex_cast(alpha), complex_cast(x), incx,
                        complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_78(mht_78_v, 1202, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGeru");

  return DoBlasInternal(wrap::rocblas_zgeru, stream,
                        /* pointer_mode_host = */ true, m, n,
                        complex_cast(alpha), complex_cast(x), incx,
                        complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_79(mht_79_v, 1217, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHbmv");

  return DoBlasInternal(
      wrap::rocblas_chbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_80(mht_80_v, 1232, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHbmv");

  return DoBlasInternal(
      wrap::rocblas_zhbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_81(mht_81_v, 1247, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHemv");

  return DoBlasInternal(
      wrap::rocblas_chemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_82(mht_82_v, 1262, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHemv");

  return DoBlasInternal(
      wrap::rocblas_zhemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_83(mht_83_v, 1275, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer");

  return DoBlasInternal(wrap::rocblas_cher, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, complex_cast(alpha),
                        complex_cast(x), incx, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_84(mht_84_v, 1288, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer");

  return DoBlasInternal(wrap::rocblas_zher, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, complex_cast(alpha),
                        complex_cast(x), incx, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_85(mht_85_v, 1302, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer2");

  return DoBlasInternal(
      wrap::rocblas_cher2, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(x), incx,
      complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_86(mht_86_v, 1316, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer2");

  return DoBlasInternal(
      wrap::rocblas_zher2, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(x), incx,
      complex_cast(y), incy, complex_cast(a), lda);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_87(mht_87_v, 1331, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpmv");

  return DoBlasInternal(
      wrap::rocblas_chpmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(ap),
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_88(mht_88_v, 1346, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpmv");

  return DoBlasInternal(
      wrap::rocblas_zhpmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(ap),
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_89(mht_89_v, 1359, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpr");

  return DoBlasInternal(wrap::rocblas_chpr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, complex_cast(alpha),
                        complex_cast(x), incx, complex_cast(ap));
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_90(mht_90_v, 1372, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpr");

  return DoBlasInternal(wrap::rocblas_zhpr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, complex_cast(alpha),
                        complex_cast(x), incx, complex_cast(ap));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_91(mht_91_v, 1386, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpr2");

  return DoBlasInternal(
      wrap::rocblas_chpr2, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(x), incx,
      complex_cast(y), incy, complex_cast(ap));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_92(mht_92_v, 1400, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHpr2");

  return DoBlasInternal(
      wrap::rocblas_zhpr2, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, complex_cast(alpha), complex_cast(x), incx,
      complex_cast(y), incy, complex_cast(ap));
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_93(mht_93_v, 1413, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSbmv");

  return DoBlasInternal(
      wrap::rocblas_ssbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_94(mht_94_v, 1427, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSbmv");

  return DoBlasInternal(
      wrap::rocblas_dsbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_95(mht_95_v, 1440, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpmv");

  return DoBlasInternal(wrap::rocblas_sspmv, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_96(mht_96_v, 1453, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpmv");

  return DoBlasInternal(wrap::rocblas_dspmv, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_97(mht_97_v, 1465, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpr");

  return DoBlasInternal(wrap::rocblas_sspr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_98(mht_98_v, 1477, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpr");

  return DoBlasInternal(wrap::rocblas_dspr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_99(mht_99_v, 1490, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpr2");

  return DoBlasInternal(wrap::rocblas_sspr2, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_100(mht_100_v, 1503, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSpr2");

  return DoBlasInternal(wrap::rocblas_dspr2, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_101(mht_101_v, 1516, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymv");

  return DoBlasInternal(wrap::rocblas_ssymv, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_102(mht_102_v, 1529, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymv");

  return DoBlasInternal(wrap::rocblas_dsymv, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_103(mht_103_v, 1541, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr");

  return DoBlasInternal(wrap::rocblas_ssyr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64_t n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_104(mht_104_v, 1553, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr");

  return DoBlasInternal(wrap::rocblas_dsyr, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_105(mht_105_v, 1566, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2");

  return DoBlasInternal(wrap::rocblas_ssyr2, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_106(mht_106_v, 1579, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2");

  return DoBlasInternal(wrap::rocblas_dsyr2, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<float> &a,
                          int lda, DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_107(mht_107_v, 1592, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbmv");

  return DoBlasInternal(wrap::rocblas_stbmv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<double> &a,
                          int lda, DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_108(mht_108_v, 1606, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbmv");

  return DoBlasInternal(wrap::rocblas_dtbmv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_109(mht_109_v, 1621, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbmv");

  return DoBlasInternal(wrap::rocblas_ctbmv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, complex_cast(a), lda,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_110(mht_110_v, 1636, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbmv");

  return DoBlasInternal(wrap::rocblas_ztbmv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, complex_cast(a), lda,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<float> &a,
                          int lda, DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_111(mht_111_v, 1650, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbsv");

  return DoBlasInternal(wrap::rocblas_stbsv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k, const DeviceMemory<double> &a,
                          int lda, DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_112(mht_112_v, 1664, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbsv");

  return DoBlasInternal(wrap::rocblas_dtbsv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_113(mht_113_v, 1679, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbsv");

  return DoBlasInternal(wrap::rocblas_ctbsv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, complex_cast(a), lda,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, uint64_t k,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_114(mht_114_v, 1694, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTbsv");

  return DoBlasInternal(wrap::rocblas_ztbsv, stream,
                        /* pointer_mode_host = */ false,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
                        ROCMBlasDiagonal(diag), n, k, complex_cast(a), lda,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &ap,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_115(mht_115_v, 1708, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpmv");

  return DoBlasInternal(
      wrap::rocblas_stpmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(ap), GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_116(mht_116_v, 1721, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpmv");

  return DoBlasInternal(
      wrap::rocblas_dtpmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(ap), GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_117(mht_117_v, 1735, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpmv");

  return DoBlasInternal(
      wrap::rocblas_ctpmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(ap), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_118(mht_118_v, 1749, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpmv");

  return DoBlasInternal(
      wrap::rocblas_ztpmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(ap), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &ap,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_119(mht_119_v, 1762, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpsv");

  return DoBlasInternal(
      wrap::rocblas_stpsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(ap), GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_120(mht_120_v, 1775, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpsv");

  return DoBlasInternal(
      wrap::rocblas_dtpsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(ap), GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_121(mht_121_v, 1789, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpsv");

  return DoBlasInternal(
      wrap::rocblas_ctpsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(ap), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_122(mht_122_v, 1803, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTpsv");

  return DoBlasInternal(
      wrap::rocblas_ztpsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(ap), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_123(mht_123_v, 1816, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmv");

  return DoBlasInternal(
      wrap::rocblas_strmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(a), lda, GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_124(mht_124_v, 1829, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmv");

  return DoBlasInternal(
      wrap::rocblas_dtrmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(a), lda, GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_125(mht_125_v, 1843, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmv");

  return DoBlasInternal(
      wrap::rocblas_ctrmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(a), lda, complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_126(mht_126_v, 1857, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmv");

  return DoBlasInternal(
      wrap::rocblas_ztrmv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(a), lda, complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_127(mht_127_v, 1870, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsv");

  return DoBlasInternal(
      wrap::rocblas_strsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(a), lda, GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_128(mht_128_v, 1883, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsv");

  return DoBlasInternal(
      wrap::rocblas_dtrsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, GpuMemory(a), lda, GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_129(mht_129_v, 1897, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsv");

  return DoBlasInternal(
      wrap::rocblas_ctrsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(a), lda, complex_cast(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag,
                          uint64_t n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_130(mht_130_v, 1911, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsv");

  return DoBlasInternal(
      wrap::rocblas_ztrsv, stream, /* pointer_mode_host = */ false,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
      ROCMBlasDiagonal(diag), n, complex_cast(a), lda, complex_cast(x), incx);
}

port::Status ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                  blas::Transpose transb, uint64_t m, uint64 n,
                                  uint64_t k, blas::DataType dtype,
                                  const void *alpha, const DeviceMemoryBase &a,
                                  int lda, const DeviceMemoryBase &b, int ldb,
                                  const void *beta, DeviceMemoryBase *c,
                                  int ldc) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_131(mht_131_v, 1927, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemm");

  blas_log("DoBlasGemm");
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
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

  switch (dtype) {
    case blas::DataType::kHalf: {
      port::StatusOr<bool> maybe_hasXDLOPS = GpuDriver::GetMFMASupport();
      if (maybe_hasXDLOPS.ok() && maybe_hasXDLOPS.ValueOrDie()) {
        VLOG(1) << "Using rocblas_gemm_ex";
        return DoBlasInternalStatus(
            wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
            (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, alpha, a.opaque(),
            rocblas_datatype_f16_r, lda, b.opaque(), rocblas_datatype_f16_r,
            ldb, beta, c->opaque(), rocblas_datatype_f16_r, ldc, c->opaque(),
            rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0);
      } else {
        VLOG(1) << "Using rocblas_hgemm";
        const Eigen::half alpha_half(*static_cast<const float *>(alpha));
        const Eigen::half beta_half(*static_cast<const float *>(beta));
        return DoBlasInternalStatus(
            wrap::rocblas_hgemm, stream, /* pointer_mode_host = */ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
            reinterpret_cast<const rocblas_half *>(&alpha_half),
            reinterpret_cast<const rocblas_half *>(a.opaque()), lda,
            reinterpret_cast<const rocblas_half *>(b.opaque()), ldb,
            reinterpret_cast<const rocblas_half *>(&beta_half),
            reinterpret_cast<rocblas_half *>(c->opaque()), ldc);
      }
    }
    case blas::DataType::kBF16:
      return DoBlasInternalStatus(
          wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), (rocblas_int)m,
          (rocblas_int)n, (rocblas_int)k, alpha, a.opaque(),
          rocblas_datatype_bf16_r, lda, b.opaque(), rocblas_datatype_bf16_r,
          ldb, beta, c->opaque(), rocblas_datatype_bf16_r, ldc, c->opaque(),
          rocblas_datatype_bf16_r, ldc, rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0);
    case blas::DataType::kFloat:
      return DoBlasInternalStatus(
          wrap::rocblas_sgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda,
          static_cast<const float *>(b.opaque()), ldb,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc);
    case blas::DataType::kDouble:
      return DoBlasInternalStatus(
          wrap::rocblas_dgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda,
          static_cast<const double *>(b.opaque()), ldb,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc);
    case blas::DataType::kComplexFloat: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<float> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_cgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          cb_alpha, static_cast<const rocblas_float_complex *>(a.opaque()), lda,
          static_cast<const rocblas_float_complex *>(b.opaque()), ldb, cb_beta,
          static_cast<rocblas_float_complex *>(c->opaque()), ldc);
    }
    case blas::DataType::kComplexDouble: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<double> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_zgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          cb_alpha, static_cast<const rocblas_double_complex *>(a.opaque()),
          lda, static_cast<const rocblas_double_complex *>(b.opaque()), ldb,
          cb_beta, static_cast<rocblas_double_complex *>(c->opaque()), ldc);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_132(mht_132_v, 2051, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_133(mht_133_v, 2064, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_134(mht_134_v, 2078, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_135(mht_135_v, 2092, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemvWithProfiling");

  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_136(mht_136_v, 2106, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_137(mht_137_v, 2119, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_138(mht_138_v, 2133, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_139(mht_139_v, 2148, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_140(mht_140_v, 2163, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithProfiling");

  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

template <typename T>
bool ROCMBlas::DoBlasGemvWithProfilingImpl(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, const T &alpha,
    const DeviceMemory<T> &a, int lda, const DeviceMemory<T> &x, int incx,
    const T &beta, DeviceMemory<T> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_141(mht_141_v, 2177, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemvWithProfilingImpl");

  // ROCM TODO: properly implement the interface
  return false;
}

template <typename T, typename ParamType>
bool ROCMBlas::DoBlasGemmWithProfilingImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
    int lda, const DeviceMemory<T> &b, int ldb, const ParamType &beta,
    DeviceMemory<T> *c, int ldc, blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return false;
}
port::Status ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_142(mht_142_v, 2200, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmWithAlgorithm");

  // ROCM TODO: properly implement the interface
  return port::InternalError("Not implemented on ROCm");
}

port::Status ROCMBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_143(mht_143_v, 2215, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmStridedBatchedWithAlgorithm");

  // ROCM TODO: properly implement the interface
  return port::InternalError("Not implemented on ROCm");
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_144(mht_144_v, 2224, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::GetBlasGemmAlgorithms");

  // ROCM TODO: properly implement the interface
  return true;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request
template <typename MAPPED_T>
port::Status ReorganizeMemory(Stream *stream,
                              DeviceMemory<MAPPED_T> *device_memory,
                              const std::vector<MAPPED_T *> &raw_ptrs,
                              int batch_count, uint64_t batch_stride,
                              bool gather) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_145(mht_145_v, 2241, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ReorganizeMemory");

  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char *src_ptr = reinterpret_cast<char *>(raw_ptrs[0]);
  char *dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  uint64_t cur_stride_size = matrix_byte_size;

  for (int i = 1; i < batch_count; ++i) {
    if (reinterpret_cast<char *>(raw_ptrs[i]) == src_ptr + cur_stride_size) {
      cur_stride_size += matrix_byte_size;
    } else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(src_ptr, cur_stride_size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(dst_ptr, cur_stride_size);
      bool a_status =
          gather
              ? stream->ThenMemcpy(&target_mem, src_mem, cur_stride_size).ok()
              : stream->ThenMemcpy(&src_mem, target_mem, cur_stride_size).ok();
      if (!a_status) {
        return port::Status(
            port::error::INTERNAL,
            "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
      }
      src_ptr = reinterpret_cast<char *>(raw_ptrs[i]);
      dst_ptr = device_memory_ptr + i * matrix_byte_size;
      cur_stride_size = matrix_byte_size;
    }
  }

  DeviceMemoryBase src_mem = DeviceMemoryBase(src_ptr, cur_stride_size);
  DeviceMemoryBase target_mem = DeviceMemoryBase(dst_ptr, cur_stride_size);
  bool a_status =
      gather ? stream->ThenMemcpy(&target_mem, src_mem, cur_stride_size).ok()
             : stream->ThenMemcpy(&src_mem, target_mem, cur_stride_size).ok();
  if (!a_status)
    return port::Status(
        port::error::INTERNAL,
        "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
  return port::Status::OK();
}

template <typename T>
port::Status ROCMBlas::AllocateStridedBuffer(
    const std::vector<typename RocBlasTypeConversionHelper<T>::mapped_type *>
        &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<
        typename RocBlasTypeConversionHelper<T>::mapped_type>> *temp_memory,
    DeviceMemory<typename RocBlasTypeConversionHelper<T>::mapped_type>
        *device_memory,
    bool copy_data, bool &reallocated) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_146(mht_146_v, 2295, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::AllocateStridedBuffer");

  assert(device_memory != nullptr);

  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  bool needs_allocate_strided = false;
  for (int i = 1; i < batch_count; ++i) {
    uint64_t tmp_batch_stride = raw_ptrs[i] - raw_ptrs[i - 1];
    if (tmp_batch_stride != batch_stride) {
      needs_allocate_strided = true;
      break;
    }
  }

  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  size_t matrix_batch_byte_size = matrix_byte_size * batch_count;

  // No need to do re-allocation, take the short cut and return
  if (!needs_allocate_strided) {
    *device_memory = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    reallocated = false;
    return port::Status::OK();
  }

  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(
        DeviceMemory<uint8> batch_matrix_bytes,
        scratch_allocator->AllocateBytes(matrix_batch_byte_size));
    *device_memory = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  } else {
    assert(temp_memory != nullptr);
    SE_ASSIGN_OR_RETURN(*temp_memory, stream->AllocateTemporaryArray<MAPPED_T>(
                                          matrix_batch_byte_size));
    *device_memory =
        DeviceMemory<MAPPED_T>(*(*temp_memory)->mutable_device_memory());
  }

  reallocated = true;

  if (copy_data)
    return ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true);
  return port::Status::OK();
}

template <typename T, typename FuncT>
port::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_147(mht_147_v, 2351, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatchedInternal");

  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  // Sanity checks before making any further progress
  uint64_t batch_stride_a = 0;
  uint64_t batch_stride_b = 0;
  uint64_t batch_stride_c = 0;

  assert(ldc >= m);
  batch_stride_c = ldc * n;

  if (ROCMBlasTranspose(transa) == rocblas_operation_none) {
    assert(lda >= m);
    batch_stride_a = lda * k;
  } else {
    assert(lda >= k);
    batch_stride_a = lda * m;
  }

  if (ROCMBlasTranspose(transb) == rocblas_operation_none) {
    assert(ldb >= k);
    batch_stride_b = ldb * n;
  } else {
    assert(ldb >= n);
    batch_stride_b = ldb * k;
  }

  // Allocate local vectors to hold device pointers to matrices
  std::vector<MAPPED_T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  DeviceMemory<MAPPED_T> a;
  // Make sure the temporary memory are in-scope before the function returns
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> a_temp;
  bool reallocated_a, reallocated_b, reallocated_c;
  port::Status a_allocation_status = AllocateStridedBuffer<T>(
      a_raw_ptrs, batch_count, batch_stride_a, scratch_allocator, stream,
      &a_temp, &a, true, reallocated_a);
  if (a_allocation_status != port::Status::OK()) {
    return a_allocation_status;
  }

  DeviceMemory<MAPPED_T> b;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> b_temp;
  port::Status b_allocation_status = AllocateStridedBuffer<T>(
      b_raw_ptrs, batch_count, batch_stride_b, scratch_allocator, stream,
      &b_temp, &b, true, reallocated_b);
  if (b_allocation_status != port::Status::OK()) {
    return b_allocation_status;
  }

  DeviceMemory<MAPPED_T> c;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> c_temp;
  port::Status c_allocation_status = AllocateStridedBuffer<T>(
      c_raw_ptrs, batch_count, batch_stride_c, scratch_allocator, stream,
      &c_temp, &c, true, reallocated_c);  // can disable copy if beta=0
  if (c_allocation_status != port::Status::OK()) {
    return c_allocation_status;
  }

  MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
  MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);

  bool ok;
  ok = DoBlasInternal(rocblas_func, stream, /* pointer_mode_host = */ true,
                      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m,
                      n, k, GpuComplex(alpha_ptr), GpuMemory(a), lda,
                      batch_stride_a, GpuMemory(b), ldb, batch_stride_b,
                      GpuComplex(beta_ptr), GpuMemoryMutable(&c), ldc,
                      batch_stride_c, batch_count);
  if (!ok)
    return port::Status(port::error::INTERNAL,
                        "failed BLAS call, see log for details");
  if (reallocated_c)
    return ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c,
                            false);
  return port::Status::OK();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_148(mht_148_v, 2448, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatched");

  blas_log("DoBlasGemmBatched");
  const Eigen::half alpha_half(alpha);
  const Eigen::half beta_half(beta);

  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_hgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha_half, a, lda, b, ldb, beta_half, c, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_149(mht_149_v, 2473, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatched");

  blas_log("DoBlasGemmBatched");
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_sgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_150(mht_150_v, 2494, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatched");

  blas_log("DoBlasGemmBatched");
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_dgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_151(mht_151_v, 2517, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatched");

  blas_log("DoBlasGemmBatched");
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_cgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_152(mht_152_v, 2540, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmBatched");

  blas_log("DoBlasGemmBatched");
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_zgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_153(mht_153_v, 2561, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHemm");

  return DoBlasInternal(wrap::rocblas_chemm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_154(mht_154_v, 2578, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHemm");

  return DoBlasInternal(wrap::rocblas_zhemm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_155(mht_155_v, 2594, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHerk");

  return DoBlasInternal(wrap::rocblas_cherk, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, complex_cast(alpha), complex_cast(a), lda,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_156(mht_156_v, 2610, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHerk");

  return DoBlasInternal(wrap::rocblas_zherk, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, complex_cast(alpha), complex_cast(a), lda,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_157(mht_157_v, 2627, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer2k");

  return DoBlasInternal(
      wrap::rocblas_cher2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      complex_cast(alpha), complex_cast(a), lda, complex_cast(b), ldb,
      complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_158(mht_158_v, 2644, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasHer2k");

  return DoBlasInternal(
      wrap::rocblas_zher2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      complex_cast(alpha), complex_cast(a), lda, complex_cast(b), ldb,
      complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_159(mht_159_v, 2659, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymm");

  return DoBlasInternal(
      wrap::rocblas_ssymm, stream, /* pointer_mode_host = */ true,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, GpuMemory(a),
      lda, GpuMemory(b), ldb, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_160(mht_160_v, 2673, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymm");

  return DoBlasInternal(
      wrap::rocblas_dsymm, stream, /* pointer_mode_host = */ true,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, GpuMemory(a),
      lda, GpuMemory(b), ldb, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_161(mht_161_v, 2689, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymm");

  return DoBlasInternal(wrap::rocblas_csymm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_162(mht_162_v, 2706, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSymm");

  return DoBlasInternal(wrap::rocblas_zsymm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_163(mht_163_v, 2720, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyrk");

  return DoBlasInternal(
      wrap::rocblas_ssyrk, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      GpuMemory(a), lda, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_164(mht_164_v, 2733, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyrk");

  return DoBlasInternal(
      wrap::rocblas_dsyrk, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      GpuMemory(a), lda, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_165(mht_165_v, 2748, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyrk");

  return DoBlasInternal(wrap::rocblas_csyrk, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, complex_cast(alpha), complex_cast(a), lda,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64_t n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_166(mht_166_v, 2764, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyrk");

  return DoBlasInternal(wrap::rocblas_zsyrk, stream,
                        /* pointer_mode_host = */ true,
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
                        k, complex_cast(alpha), complex_cast(a), lda,
                        complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_167(mht_167_v, 2779, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2k");

  return DoBlasInternal(
      wrap::rocblas_ssyr2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      GpuMemory(a), lda, GpuMemory(b), ldb, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_168(mht_168_v, 2793, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2k");

  return DoBlasInternal(
      wrap::rocblas_dsyr2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
      GpuMemory(a), lda, GpuMemory(b), ldb, &beta, GpuMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_169(mht_169_v, 2809, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2k");

  return DoBlasInternal(
      wrap::rocblas_csyr2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      complex_cast(alpha), complex_cast(a), lda, complex_cast(b), ldb,
      complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64_t n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_170(mht_170_v, 2826, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasSyr2k");

  return DoBlasInternal(
      wrap::rocblas_zsyr2k, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
      complex_cast(alpha), complex_cast(a), lda, complex_cast(b), ldb,
      complex_cast(beta), complex_cast(c), ldc);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_171(mht_171_v, 2841, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmm");

  return DoBlasInternal(wrap::rocblas_strmm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_172(mht_172_v, 2856, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmm");

  return DoBlasInternal(wrap::rocblas_dtrmm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_173(mht_173_v, 2872, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmm");

  return DoBlasInternal(wrap::rocblas_ctrmm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_174(mht_174_v, 2888, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrmm");

  return DoBlasInternal(wrap::rocblas_ztrmm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_175(mht_175_v, 2903, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsm");

  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_strsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_176(mht_176_v, 2919, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsm");

  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_dtrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_177(mht_177_v, 2936, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsm");

  return DoBlasInternal(wrap::rocblas_ctrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_178(mht_178_v, 2952, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsm");

  return DoBlasInternal(wrap::rocblas_ztrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_179(mht_179_v, 2968, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsmBatched");

  // ROCM TODO: properly implement the interface
  return false;
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_180(mht_180_v, 2981, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsmBatched");

  // ROCM TODO: properly implement the interface
  return false;
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_181(mht_181_v, 2996, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsmBatched");

  // ROCM TODO: properly implement the interface
  return false;
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_182(mht_182_v, 3011, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasTrsmBatched");

  // ROCM TODO: properly implement the interface
  return false;
}

port::Status ROCMBlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_183(mht_183_v, 3024, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasGemmStridedBatched");

  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS SGEMM Strided Batched<float>: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);

  switch (dtype) {
    case blas::DataType::kHalf: {
      const Eigen::half alpha_half(*static_cast<const float *>(alpha));
      const Eigen::half beta_half(*static_cast<const float *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_hgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const rocblas_half *>(&alpha_half),
          reinterpret_cast<const rocblas_half *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const rocblas_half *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const rocblas_half *>(&beta_half),
          reinterpret_cast<rocblas_half *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    case blas::DataType::kBF16:
      return DoBlasInternalStatus(
          wrap::rocblas_gemm_strided_batched_ex, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, alpha,
          a.opaque(), rocblas_datatype_bf16_r, lda, stride_a, b.opaque(),
          rocblas_datatype_bf16_r, ldb, stride_b, beta, c->opaque(),
          rocblas_datatype_bf16_r, ldc, stride_c, c->opaque(),
          rocblas_datatype_bf16_r, ldc, stride_c, batch_count,
          rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    case blas::DataType::kFloat:
      return DoBlasInternalStatus(
          wrap::rocblas_sgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const float *>(alpha),
          reinterpret_cast<const float *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const float *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const float *>(beta),
          reinterpret_cast<float *>(c->opaque()), ldc, stride_c, batch_count);
    case blas::DataType::kDouble:
      return DoBlasInternalStatus(
          wrap::rocblas_dgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const double *>(alpha),
          reinterpret_cast<const double *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const double *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const double *>(beta),
          reinterpret_cast<double *>(c->opaque()), ldc, stride_c, batch_count);
    case blas::DataType::kComplexFloat: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<float> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_cgemm_strided_batched, stream,
          /* pointer_mode_host = */ true, ROCMBlasTranspose(transa),
          ROCMBlasTranspose(transb), m, n, k, cb_alpha,
          static_cast<const rocblas_float_complex *>(a.opaque()), lda, stride_a,
          static_cast<const rocblas_float_complex *>(b.opaque()), ldb, stride_b,
          cb_beta, static_cast<rocblas_float_complex *>(c->opaque()), ldc,
          stride_c, batch_count);
    }
    case blas::DataType::kComplexDouble: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<double> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_zgemm_strided_batched, stream,
          /* pointer_mode_host = */ true, ROCMBlasTranspose(transa),
          ROCMBlasTranspose(transb), m, n, k, cb_alpha,
          static_cast<const rocblas_double_complex *>(a.opaque()), lda,
          stride_a, static_cast<const rocblas_double_complex *>(b.opaque()),
          ldb, stride_b, cb_beta,
          static_cast<rocblas_double_complex *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

port::Status ROCMBlas::GetVersion(string *version) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_184(mht_184_v, 3115, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::GetVersion");

  return port::UnimplementedError("");
}

port::StatusOr<std::unique_ptr<blas::IBlasLtMatmulPlan>>
ROCMBlas::CreateBlasLtMatmulPlan(const blas::BlasLtMatmulPlanParams &p) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_185(mht_185_v, 3123, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::CreateBlasLtMatmulPlan");

  return port::Status(
      port::error::UNIMPLEMENTED,
      "CreateBlasLtMatmulPlan is not supported with this version of ROCM");
}

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
ROCMBlas::GetBlasLtMatmulAlgorithms(const blas::IBlasLtMatmulPlan *plan,
                                    size_t max_workspace_size,
                                    int max_algorithm_count) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_186(mht_186_v, 3135, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::GetBlasLtMatmulAlgorithms");

  return port::Status(
      port::error::UNIMPLEMENTED,
      "GetBlasLtMatmulAlgorithms is not supported with this version of ROCM");
}

bool ROCMBlas::DoBlasLtMatmul(
    Stream *stream, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_187(mht_187_v, 3150, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "ROCMBlas::DoBlasLtMatmul");

  return false;
}

}  // namespace gpu

void initialize_rocblas() {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTcc mht_188(mht_188_v, 3159, "", "./tensorflow/stream_executor/rocm/rocm_blas.cc", "initialize_rocblas");

  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);

  if (!rocBlasAlreadyRegistered) {
    port::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, gpu::kRocBlasPlugin, "rocBLAS",
                [](internal::StreamExecutorInterface *parent)
                    -> blas::BlasSupport * {
                  gpu::GpuExecutor *rocm_executor =
                      dynamic_cast<gpu::GpuExecutor *>(parent);
                  if (rocm_executor == nullptr) {
                    LOG(ERROR)
                        << "Attempting to initialize an instance of the "
                           "rocBLAS "
                        << "support library with a non-ROCM StreamExecutor";
                    return nullptr;
                  }

                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(rocm_executor);
                  if (!blas->Init()) {
                    // Note: Init() will log a more specific error.
                    delete blas;
                    return nullptr;
                  }
                  return blas;
                });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocBLAS factory: "
                 << status.error_message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
