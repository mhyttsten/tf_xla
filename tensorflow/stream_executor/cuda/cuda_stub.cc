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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc() {
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
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

// Implements the CUDA driver API by forwarding to CUDA loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc mht_0(mht_0_v, 192, "", "./tensorflow/stream_executor/cuda/cuda_stub.cc", "GetDsoHandle");

#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetCudaDriverDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.ValueOrDie();
  }();
  return handle;
#endif
}

template <typename T>
T LoadSymbol(const char* symbol_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc mht_1(mht_1_v, 211, "", "./tensorflow/stream_executor/cuda/cuda_stub.cc", "LoadSymbol");

  void* symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    stream_executor::port::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

CUresult GetSymbolNotFoundError() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_stubDTcc mht_2(mht_2_v, 224, "", "./tensorflow/stream_executor/cuda/cuda_stub.cc", "GetSymbolNotFoundError");

  return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
}
}  // namespace

#if CUDA_VERSION < 8000
#error CUDA version earlier than 8 is not supported.
#endif

// Forward-declare types introduced in CUDA 9.0.
typedef struct CUDA_LAUNCH_PARAMS_st CUDA_LAUNCH_PARAMS;

#ifndef __CUDA_DEPRECATED
#define __CUDA_DEPRECATED
#endif

#if CUDA_VERSION < 10000
// Define fake enums introduced in CUDA 10.0.
typedef enum CUgraphNodeType_enum {} CUgraphNodeType;
typedef enum CUstreamCaptureStatus_enum {} CUstreamCaptureStatus;
typedef enum CUexternalMemoryHandleType_enum {} CUexternalMemoryHandleType;
typedef enum CUexternalSemaphoreHandleType_enum {
} CUexternalSemaphoreHandleType;
#endif

// Forward-declare types introduced in CUDA 10.0.
typedef struct CUextMemory_st* CUexternalMemory;
typedef struct CUextSemaphore_st* CUexternalSemaphore;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUDA_KERNEL_NODE_PARAMS_st CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_MEMSET_NODE_PARAMS_st CUDA_MEMSET_NODE_PARAMS;
typedef struct CUDA_HOST_NODE_PARAMS_st CUDA_HOST_NODE_PARAMS;
typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;
typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
typedef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS;
typedef void(CUDA_CB* CUhostFn)(void* userData);

#if CUDA_VERSION < 10000
#include "tensorflow/stream_executor/cuda/cuda_9_0.inc"
#elif CUDA_VERSION < 10010
#include "tensorflow/stream_executor/cuda/cuda_10_0.inc"
#elif CUDA_VERSION < 10020
#include "tensorflow/stream_executor/cuda/cuda_10_1.inc"
#elif CUDA_VERSION < 11000
#include "tensorflow/stream_executor/cuda/cuda_10_2.inc"
#elif CUDA_VERSION < 11020
#include "tensorflow/stream_executor/cuda/cuda_11_0.inc"
#else
#include "tensorflow/stream_executor/cuda/cuda_11_2.inc"
#endif
