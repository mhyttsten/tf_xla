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
class MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc() {
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
#include "tensorflow/stream_executor/platform/default/dso_loader.h"

#include <stdlib.h>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/cuda_config.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "third_party/tensorrt/tensorrt_config.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace stream_executor {
namespace internal {

namespace {
string GetCudaVersion() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_0(mht_0_v, 206, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCudaVersion");
 return TF_CUDA_VERSION; }
string GetCudaRtVersion() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_1(mht_1_v, 210, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCudaRtVersion");
 return TF_CUDART_VERSION; }
string GetCudnnVersion() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_2(mht_2_v, 214, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCudnnVersion");
 return TF_CUDNN_VERSION; }
string GetCublasVersion() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_3(mht_3_v, 218, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCublasVersion");
 return TF_CUBLAS_VERSION; }
string GetCusolverVersion() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_4(mht_4_v, 222, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCusolverVersion");
 return TF_CUSOLVER_VERSION; }
string GetCurandVersion() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_5(mht_5_v, 226, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCurandVersion");
 return TF_CURAND_VERSION; }
string GetCufftVersion() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_6(mht_6_v, 230, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCufftVersion");
 return TF_CUFFT_VERSION; }
string GetCusparseVersion() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_7(mht_7_v, 234, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetCusparseVersion");
 return TF_CUSPARSE_VERSION; }
string GetTensorRTVersion() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformPSdefaultPSdso_loaderDTcc mht_8(mht_8_v, 238, "", "./tensorflow/stream_executor/platform/default/dso_loader.cc", "GetTensorRTVersion");
 return TF_TENSORRT_VERSION; }

port::StatusOr<void*> GetDsoHandle(const string& name, const string& version) {
  auto filename = port::Env::Default()->FormatLibraryFileName(name, version);
  void* dso_handle;
  port::Status status =
      port::Env::Default()->LoadDynamicLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << filename;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", filename,
                              "'; dlerror: ", status.error_message());
#if !defined(PLATFORM_WINDOWS)
  if (const char* ld_library_path = getenv("LD_LIBRARY_PATH")) {
    message += absl::StrCat("; LD_LIBRARY_PATH: ", ld_library_path);
  }
#endif
  LOG(WARNING) << message;
  return port::Status(port::error::FAILED_PRECONDITION, message);
}
}  // namespace

namespace DsoLoader {
port::StatusOr<void*> GetCudaDriverDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvcuda", "");
#elif defined(__APPLE__)
  // On Mac OS X, CUDA sometimes installs libcuda.dylib instead of
  // libcuda.1.dylib.
  auto handle_or = GetDsoHandle("cuda", "");
  if (handle_or.ok()) {
    return handle_or;
  }
#endif
  return GetDsoHandle("cuda", "1");
}

port::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  return GetDsoHandle("cudart", GetCudaRtVersion());
}

port::StatusOr<void*> GetCublasDsoHandle() {
  return GetDsoHandle("cublas", GetCublasVersion());
}

port::StatusOr<void*> GetCublasLtDsoHandle() {
  return GetDsoHandle("cublasLt", GetCublasVersion());
}

port::StatusOr<void*> GetCufftDsoHandle() {
  return GetDsoHandle("cufft", GetCufftVersion());
}

port::StatusOr<void*> GetCusolverDsoHandle() {
  return GetDsoHandle("cusolver", GetCusolverVersion());
}

port::StatusOr<void*> GetCusparseDsoHandle() {
  return GetDsoHandle("cusparse", GetCusparseVersion());
}

port::StatusOr<void*> GetCurandDsoHandle() {
  return GetDsoHandle("curand", GetCurandVersion());
}

port::StatusOr<void*> GetCuptiDsoHandle() {
  // Load specific version of CUPTI this is built.
  auto status_or_handle = GetDsoHandle("cupti", GetCudaVersion());
  if (status_or_handle.ok()) return status_or_handle;
  // Load whatever libcupti.so user specified.
  return GetDsoHandle("cupti", "");
}

port::StatusOr<void*> GetCudnnDsoHandle() {
  return GetDsoHandle("cudnn", GetCudnnVersion());
}

port::StatusOr<void*> GetNvInferDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer", "");
#else
  return GetDsoHandle("nvinfer", GetTensorRTVersion());
#endif
}

port::StatusOr<void*> GetNvInferPluginDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer_plugin", "");
#else
  return GetDsoHandle("nvinfer_plugin", GetTensorRTVersion());
#endif
}

port::StatusOr<void*> GetRocblasDsoHandle() {
  return GetDsoHandle("rocblas", "");
}

port::StatusOr<void*> GetMiopenDsoHandle() {
  return GetDsoHandle("MIOpen", "");
}

port::StatusOr<void*> GetHipfftDsoHandle() {
  return GetDsoHandle("hipfft", "");
}

port::StatusOr<void*> GetRocrandDsoHandle() {
  return GetDsoHandle("rocrand", "");
}

port::StatusOr<void*> GetRocsolverDsoHandle() {
  return GetDsoHandle("rocsolver", "");
}

#if TF_ROCM_VERSION >= 40500
port::StatusOr<void*> GetHipsolverDsoHandle() {
  return GetDsoHandle("hipsolver", "");
}
#endif

port::StatusOr<void*> GetRoctracerDsoHandle() {
  return GetDsoHandle("roctracer64", "");
}

port::StatusOr<void*> GetHipsparseDsoHandle() {
  return GetDsoHandle("hipsparse", "");
}

port::StatusOr<void*> GetHipDsoHandle() { return GetDsoHandle("amdhip64", ""); }

}  // namespace DsoLoader

namespace CachedDsoLoader {
port::StatusOr<void*> GetCudaDriverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaDriverDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaRuntimeDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCublasDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCublasLtDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasLtDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCurandDsoHandle() {
  static auto result = new auto(DsoLoader::GetCurandDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCufftDsoHandle() {
  static auto result = new auto(DsoLoader::GetCufftDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCusolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusolverDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCusparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusparseDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCuptiDsoHandle() {
  static auto result = new auto(DsoLoader::GetCuptiDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCudnnDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudnnDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocblasDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocblasDsoHandle());
  return *result;
}

port::StatusOr<void*> GetMiopenDsoHandle() {
  static auto result = new auto(DsoLoader::GetMiopenDsoHandle());
  return *result;
}

port::StatusOr<void*> GetHipfftDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipfftDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocrandDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocrandDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRoctracerDsoHandle() {
  static auto result = new auto(DsoLoader::GetRoctracerDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocsolverDsoHandle());
  return *result;
}

#if TF_ROCM_VERSION >= 40500
port::StatusOr<void*> GetHipsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsolverDsoHandle());
  return *result;
}
#endif

port::StatusOr<void*> GetHipsparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsparseDsoHandle());
  return *result;
}

port::StatusOr<void*> GetHipDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipDsoHandle());
  return *result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace stream_executor
