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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc() {
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
==============================================================================*/

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUcudaMallocAllocator::GPUcudaMallocAllocator(
    PlatformDeviceId platform_device_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.cc", "GPUcudaMallocAllocator::GPUcudaMallocAllocator");

  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
}

void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.cc", "GPUcudaMallocAllocator::AllocateRaw");

#ifdef GOOGLE_CUDA
  // allocate with cudaMalloc
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  CUdeviceptr rv = 0;
  CUresult res = cuMemAlloc(&rv, num_bytes);
  if (res != CUDA_SUCCESS) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(res, &error_name);
    cuGetErrorString(res, &error_string);
    LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes
               << "\n Error name: " << error_name
               << "\n Error string: " << error_string;
    return nullptr;
  }
  VLOG(10) << "AllocateRaw " << Name() << "  " << num_bytes << " "
           << reinterpret_cast<void*>(rv);
  return reinterpret_cast<void*>(rv);
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.cc", "GPUcudaMallocAllocator::DeallocateRaw");

#ifdef GOOGLE_CUDA
  // free with cudaFree
  CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res == CUDA_ERROR_DEINITIALIZED) {
    // It happens with multi-GPU that TF free the GPU allocation after
    // the driver is unloaded. It is safe to ignore this error here.
    // cuGetErrorName and cuGetErrorString doesn't return any useful
    // information here.
    // TODO: Find how to fix the shutdown steps in TF.
    VLOG(1) << "Ignoring CUDA_ERROR_DEINITIALIZED Error";
  } else if (res != CUDA_SUCCESS) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(res, &error_name);
    cuGetErrorString(res, &error_string);
    LOG(ERROR) << "cuMemFree failed to free " << ptr
               << "\n Error name: " << error_name
               << "\n Error string: " << error_string;
  }
  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // GOOGLE_CUDA
}

bool GPUcudaMallocAllocator::TracksAllocationSizes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamalloc_allocatorDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.cc", "GPUcudaMallocAllocator::TracksAllocationSizes");
 return false; }

}  // namespace tensorflow
