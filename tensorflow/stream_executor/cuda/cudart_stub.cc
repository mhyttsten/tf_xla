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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc() {
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

// This file wraps cuda runtime calls with dso loader so that we don't need to
// have explicit linking to libcuda.

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

namespace {
void* GetDsoHandle() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_0(mht_0_v, 193, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "GetDsoHandle");

  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetCudaRuntimeDsoHandle();
    if (!handle_or.ok()) {
      LOG(INFO) << "Ignore above cudart dlerror if you do not have a GPU set "
                   "up on your machine.";
      return nullptr;
    }
    return handle_or.ValueOrDie();
  }();
  return handle;
}

template <typename T>
T LoadSymbol(const char* symbol_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_1(mht_1_v, 212, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "LoadSymbol");

  void* symbol = nullptr;
  auto env = stream_executor::port::Env::Default();
  env->GetSymbolFromLibrary(GetDsoHandle(), symbol_name, &symbol).IgnoreError();
  return reinterpret_cast<T>(symbol);
}
cudaError_t GetSymbolNotFoundError() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_2(mht_2_v, 221, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "GetSymbolNotFoundError");

  return cudaErrorSharedObjectSymbolNotFound;
}
}  // namespace

#define __dv(v)
#define __CUDA_DEPRECATED

// A bunch of new symbols were introduced in version 10
#if CUDART_VERSION < 10000
#include "tensorflow/stream_executor/cuda/cuda_runtime_9_0.inc"
#elif CUDART_VERSION < 10010
#include "tensorflow/stream_executor/cuda/cuda_runtime_10_0.inc"
#elif CUDART_VERSION < 10020
#include "tensorflow/stream_executor/cuda/cuda_runtime_10_1.inc"
#elif CUDART_VERSION < 11000
#include "tensorflow/stream_executor/cuda/cuda_runtime_10_2.inc"
#elif CUDART_VERSION < 11020
#include "tensorflow/stream_executor/cuda/cuda_runtime_11_0.inc"
#else
#include "tensorflow/stream_executor/cuda/cuda_runtime_11_2.inc"
#endif
#undef __dv
#undef __CUDA_DEPRECATED

extern "C" {

// Following are private symbols in libcudart that got inserted by nvcc.
extern void CUDARTAPI __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("hostFun: \"" + (hostFun == nullptr ? std::string("nullptr") : std::string((char*)hostFun)) + "\"");
   mht_3_v.push_back("deviceFun: \"" + (deviceFun == nullptr ? std::string("nullptr") : std::string((char*)deviceFun)) + "\"");
   mht_3_v.push_back("deviceName: \"" + (deviceName == nullptr ? std::string("nullptr") : std::string((char*)deviceName)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_3(mht_3_v, 258, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaRegisterFunction");

  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle, const char *hostFun,
                                    char *deviceFun, const char *deviceName,
                                    int thread_limit, uint3 *tid, uint3 *bid,
                                    dim3 *bDim, dim3 *gDim, int *wSize);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFunction");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);
}

extern void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_4(mht_4_v, 272, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaUnregisterFatBinary");

  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaUnregisterFatBinary");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                        char *deviceAddress,
                                        const char *deviceName, int ext,
                                        size_t size, int constant, int global) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("hostVar: \"" + (hostVar == nullptr ? std::string("nullptr") : std::string((char*)hostVar)) + "\"");
   mht_5_v.push_back("deviceAddress: \"" + (deviceAddress == nullptr ? std::string("nullptr") : std::string((char*)deviceAddress)) + "\"");
   mht_5_v.push_back("deviceName: \"" + (deviceName == nullptr ? std::string("nullptr") : std::string((char*)deviceName)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_5(mht_5_v, 288, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaRegisterVar");

  using FuncPtr = void(CUDARTAPI *)(
      void **fatCubinHandle, char *hostVar, char *deviceAddress,
      const char *deviceName, int ext, size_t size, int constant, int global);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterVar");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
           constant, global);
}

extern void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_6(mht_6_v, 301, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaRegisterFatBinary");

  using FuncPtr = void **(CUDARTAPI *)(void *fatCubin);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFatBinary");
  if (!func_ptr) return nullptr;
  return (void **)func_ptr(fatCubin);
}

extern cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                        dim3 *blockDim,
                                                        size_t *sharedMem,
                                                        void *stream) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_7(mht_7_v, 314, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaPopCallConfiguration");

  using FuncPtr = cudaError_t(CUDARTAPI *)(dim3 * gridDim, dim3 * blockDim,
                                           size_t * sharedMem, void *stream);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaPopCallConfiguration");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

extern __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_8(mht_8_v, 326, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaPushCallConfiguration");

  using FuncPtr = unsigned(CUDARTAPI *)(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem, void *stream);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaPushCallConfiguration");
  if (!func_ptr) return 0;
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

extern char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScudart_stubDTcc mht_9(mht_9_v, 337, "", "./tensorflow/stream_executor/cuda/cudart_stub.cc", "__cudaInitModule");

  using FuncPtr = char(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaInitModule");
  if (!func_ptr) return 0;
  return func_ptr(fatCubinHandle);
}

#if CUDART_VERSION >= 10010
extern void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFatBinaryEnd");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}
#endif
}  // extern "C"
