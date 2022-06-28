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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_MEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_MEM_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh() {
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


#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// Suballocator for StreamExecutor-based device memory.
class DeviceMemAllocator : public SubAllocator {
 public:
  // 'platform_device_id' refers to the ID of the device within
  // the process and must reference a valid ID in the process.
  // Note: stream_exec cannot be null.
  explicit DeviceMemAllocator(se::StreamExecutor* stream_exec,
                              PlatformDeviceId device_id,
                              bool use_unified_memory,
                              const std::vector<Visitor>& alloc_visitors,
                              const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        device_id_(device_id),
        use_unified_memory_(use_unified_memory) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "DeviceMemAllocator");

    CHECK(stream_exec_ != nullptr);
  }
  ~DeviceMemAllocator() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "~DeviceMemAllocator");
}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_2(mht_2_v, 220, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "Alloc");

    void* ptr = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0) {
      if (use_unified_memory_) {
        ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
      } else {
        ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
      }
      VisitAlloc(ptr, device_id_.value(), num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_3(mht_3_v, 237, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "Free");

    if (ptr != nullptr) {
      VisitFree(ptr, device_id_.value(), num_bytes);
      if (use_unified_memory_) {
        stream_exec_->UnifiedMemoryDeallocate(ptr);
      } else {
        se::DeviceMemoryBase device_ptr(ptr);
        stream_exec_->Deallocate(&device_ptr);
      }
    }
  }

  bool SupportsCoalescing() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_4(mht_4_v, 252, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "SupportsCoalescing");
 return false; }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_mem_allocatorDTh mht_5(mht_5_v, 257, "", "./tensorflow/core/common_runtime/device/device_mem_allocator.h", "GetMemoryType");

    return AllocatorMemoryType::kDevice;
  }

 private:
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  const PlatformDeviceId device_id_;
  const bool use_unified_memory_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMemAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_MEM_ALLOCATOR_H_
