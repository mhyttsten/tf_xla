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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_allocatorDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"

#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

// std::align is not supported, so this method mimic its behavior.
//
// NOTE(aaroey): according to the TensorRT API,
// nvinfer1::IGpuAllocator::allocate() uses uint64_t type for size and alignment
// parameters, so here we use the same type to make it compatible.
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_allocatorDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_allocator.cc", "Align");

  QCHECK_GT(alignment, 0ul) << "alignment must be greater than 0.";
  QCHECK_EQ(0, alignment & (alignment - 1)) << "Alignment must be power of 2.";
  QCHECK_GT(size, 0ul) << "size must be greater than 0.";
  QCHECK(ptr) << "ptr must not be nullptr.";
  QCHECK_GT(space, 0ul) << "space must be greater than 0.";
  const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  QCHECK_GE(ptr_val + space, ptr_val) << "Provided space overflows.";

  if (size > space) return nullptr;
  const uintptr_t aligned_ptr_val = ((ptr_val + alignment - 1) & -alignment);
  if (aligned_ptr_val > ptr_val + space - size) return nullptr;
  ptr = reinterpret_cast<void*>(aligned_ptr_val);
  const uintptr_t diff = aligned_ptr_val - ptr_val;
  space -= diff;
  return ptr;
}

}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

void* TRTDeviceAllocator::allocate(uint64_t size, uint64_t alignment,
                                   uint32_t flags) noexcept {
  if (size == 0) return nullptr;
  // WAR for allocator alignment requirement. Certain cuda API calls require GPU
  // memory with alignment to cudaDeviceProp::textureAlignment.
  // See issue #20856
  alignment = 512;
  assert((alignment & (alignment - 1)) == 0);  // zero or a power of 2.
  uint64_t total_size = size + alignment;
  // TODO(aaroey): AllocateRaw takes size_t size as input, so it'll produce
  // unexpected result when TRT tries to allocate more bytes than size_t can
  // carry. Fix this.
  //
  // Fail immediately if allocation fails, rather than waiting 10 seconds and
  // failing then anyway.
  // TensorRT 7 can also switch to a different algorithm for a layer if an
  // algorithm uses too much memory. If we don't fail immediately building the
  // engine can be *very* slow with TensorRT7 when GPU memory is limited.
  AllocationAttributes attributes;
  attributes.retry_on_failure = false;
  void* mem = allocator_->AllocateRaw(alignment, total_size, attributes);
  if (!mem) return nullptr;

  void* alloc_mem = mem;
  QCHECK(Align(alignment, size, mem, total_size));
  mutex_lock lock(mu_);
  if (mem != alloc_mem) {
    QCHECK(mem_map_.insert({mem, alloc_mem}).second);
  }
  VLOG(2) << "Allocated " << total_size << " bytes memory @" << alloc_mem
          << "; aligned to " << size << " bytes @" << mem << " with alignment "
          << alignment;
  return mem;
}

TRTDeviceAllocator::TRTDeviceAllocator(Allocator* allocator)
    : allocator_(allocator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_allocatorDTcc mht_1(mht_1_v, 266, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_allocator.cc", "TRTDeviceAllocator::TRTDeviceAllocator");

  VLOG(1) << "Using " << allocator->Name() << " allocator from TensorFlow";
}

void TRTDeviceAllocator::free(void* memory) noexcept {
  mutex_lock lock(mu_);
  VLOG(2) << "Deallocating @ " << memory;
  // allocated memory adjusted for alignment, restore the original pointer
  if (memory) {
    auto alloc_mem = mem_map_.find(memory);
    if (alloc_mem != mem_map_.end()) {
      memory = alloc_mem->second;
      mem_map_.erase(alloc_mem->first);
    }
    allocator_->DeallocateRaw(memory);
  }
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
