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

// ROCM-specific support for BLAS functionality -- this wraps the rocBLAS
// library capabilities, and is only included into ROCM implementation code --
// it will not introduce rocm headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTh() {
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


#include "absl/synchronization/mutex.h"
#include "rocm/include/rocblas.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"

namespace stream_executor {

class Stream;

namespace gpu {

// Type conversion helper that helps to map non-rocblas types to rocblas types
// Right now, it only converts the Eigen::half type to rocblas_half type
template <typename T>
struct RocBlasTypeConversionHelper {
  using mapped_type = T;
};

template <>
struct RocBlasTypeConversionHelper<Eigen::half> {
  using mapped_type = rocblas_half;
};

template <>
struct RocBlasTypeConversionHelper<std::complex<float>> {
  using mapped_type = rocblas_float_complex;
};

template <>
struct RocBlasTypeConversionHelper<std::complex<double>> {
  using mapped_type = rocblas_double_complex;
};

// Opaque and unique identifier for the rocBLAS plugin.
extern const PluginId kRocBlasPlugin;

class GpuExecutor;

// BLAS plugin for ROCM platform via rocBLAS library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the rocBLAS handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the rocBLAS handle when a
// ROCM context is active.
//
// Thread-safe post-initialization.
class ROCMBlas : public blas::BlasSupport {
 public:
  explicit ROCMBlas(GpuExecutor *parent);

  // Allocates a rocBLAS handle.
  bool Init();

  // Releases the rocBLAS handle, if present.
  ~ROCMBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  // Tells rocBLAS to enqueue the BLAS operation onto a particular Stream.
  //
  // rocBLAS is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into rocBLAS.
  bool SetStream(Stream *stream) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real rocBLAS function together with error
  // handling.
  //
  // rocblas_func:       rocBLAS function pointer.
  // rocblas_name:       rocBLAS function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // err_on_failure:     Whether to print an error if the rocBLAS function
  // fails. args:               Arguments of rocBLAS function.
  template <typename FuncT, typename... Args>
  bool DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                          bool pointer_mode_host, bool err_on_failure,
                          Args... args);

  // Convenience functions that call DoBlasInternalImpl with different values
  // for err_on_failure.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT rocblas_func, Stream *stream,
                      bool pointer_mode_host, Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/true, args...);
  }

  // Same as above, but returns Status.
  template <typename... Args>
  port::Status DoBlasInternalStatus(Args... args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_blasDTh mht_0(mht_0_v, 289, "", "./tensorflow/stream_executor/rocm/rocm_blas.h", "DoBlasInternalStatus");

    if (!DoBlasInternal(args...)) {
      return port::InternalError("Failed calling rocBLAS");
    }
    return port::Status::OK();
  }

  template <typename FuncT, typename... Args>
  bool DoBlasInternalFailureOK(FuncT rocblas_func, Stream *stream,
                               bool pointer_mode_host, Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/false, args...);
  }

  // A helper allocation function to convert raw pointers memory layout to
  // strided flavor
  template <typename T>
  port::Status AllocateStridedBuffer(
      const std::vector<typename RocBlasTypeConversionHelper<T>::mapped_type *>
          &raw_ptrs,
      int batch_count, uint64_t batch_stride,
      ScratchAllocator *scratch_allocator, Stream *stream,
      std::unique_ptr<TemporaryDeviceMemory<
          typename RocBlasTypeConversionHelper<T>::mapped_type>> *temp_memory,
      DeviceMemory<typename RocBlasTypeConversionHelper<T>::mapped_type>
          *device_memory,
      bool copy_data, bool &reallocated);

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  //
  // Note: This function is implemented using gemm_strided_batched interface,
  // NOT gemm_batched interface, because rocblas do not support it. As a
  // result, if the passed in batch matrix are not allocated in strided batched
  // format, it might end up in non-trivial amount of memory allocation and
  // copy. To avoid this, always prioritize to use DoBlasGemmStridedBatched
  // interface.
  //
  // In most use cases, batch matrix do get allocated in strided manner, making
  // calling this interface equivalent with DoBlasGemmStridedBatched. The only
  // use case we see so far that violates this observation is when batch
  // matrix is created by broadcasting from a smaller matrix. When it happens,
  // It will take advantage of the AllocateStridedBuffer subroutine to
  // reallocate the memory layout to be strided batched.
  template <typename T, typename FuncT>
  port::Status DoBlasGemmBatchedInternal(
      FuncT rocblas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64_t m, uint64 n, uint64 k, T alpha,
      const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
      const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
      T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
      int ldc, int batch_count, ScratchAllocator *scratch_allocator);

  // Helper function for implementing DoBlasGemmWithProfiling.
  template <typename T, typename ParamType>
  bool DoBlasGemmWithProfilingImpl(Stream *stream, blas::Transpose transa,
                                   blas::Transpose transb, uint64_t m,
                                   uint64_t n, uint64 k, const ParamType &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &b, int ldb,
                                   const ParamType &beta, DeviceMemory<T> *c,
                                   int ldc,
                                   blas::ProfileResult *output_profile_result);

  // Helper function for implementing DoBlasGemvWithProfiling.
  template <typename T>
  bool DoBlasGemvWithProfilingImpl(Stream *stream, blas::Transpose trans,
                                   uint64_t m, uint64 n, const T &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &x, int incx,
                                   const T &beta, DeviceMemory<T> *y, int incy,
                                   blas::ProfileResult *output_profile_result);

  // mutex that guards the rocBLAS handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this ROCMBlas.
  // Immutable post-initialization.
  GpuExecutor *parent_;

  // rocBLAS library handle on the device.
  rocblas_handle blas_ TF_GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(ROCMBlas);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
