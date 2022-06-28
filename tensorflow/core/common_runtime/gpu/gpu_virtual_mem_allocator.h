/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// CUDA virtual memory API is only available in CUDA versions greater than 10.2.

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTh() {
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


#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#endif

#if CUDA_VERSION >= 10020

namespace tensorflow {

// GpuVirtualMemAllocator is a SubAllocator for use with BFCAllocator which
// provides contiguous allocations with each call to Alloc. This is done by
// reserving a large chunk of virtual addresses at construction and then mapping
// physical memory pages to this virtual address range as requested.
//
// This class is not thread-safe.
class GpuVirtualMemAllocator : public SubAllocator {
 public:
  static stream_executor::port::StatusOr<
      std::unique_ptr<GpuVirtualMemAllocator>>
  Create(const std::vector<Visitor>& alloc_visitors,
         const std::vector<Visitor>& free_visitors,
         stream_executor::gpu::GpuContext& gpu_context, PlatformDeviceId gpu_id,
         size_t virtual_address_space_size,
         const std::vector<PlatformDeviceId>& peer_gpu_ids);
  ~GpuVirtualMemAllocator() override;

  // Allocates memory at least as large as requested by num_bytes. Will be
  // aligned to the min allocation granularity (typically 2MiB).
  // alignment is ignored by this allocator.
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override;

  // Frees should only happen at the end of the contiguous memory allocations or
  // else we introduce pointless fragmentation...But, this is supported. If the
  // allocation happens at the end, then the next_alloc_offset_ is moved back,
  // otherwise a hole is created.
  //
  // Holes are not re-used, all allocations continue to come at the end of the
  // next_alloc_offset_. To accommodate this, the virtual_address_space_size
  // should be much larger than the max physical size of the allocator.
  //
  // In practice, since the BFC allocator coalesces adjacent AllocationRegions,
  // this free function should never be invoked.
  void Free(void* ptr, size_t num_bytes) override;

  bool SupportsCoalescing() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTh mht_0(mht_0_v, 240, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h", "SupportsCoalescing");
 return true; }

 private:
  GpuVirtualMemAllocator(
      const std::vector<Visitor>& alloc_visitors,
      const std::vector<Visitor>& free_visitors,
      stream_executor::gpu::GpuContext& gpu_context, PlatformDeviceId gpu_id,
      std::vector<stream_executor::gpu::GpuDeviceHandle> access_device_handles,
      stream_executor::gpu::GpuDriver::VmemSpan vmem, size_t granularity);

  stream_executor::gpu::GpuContext& gpu_context_;
  PlatformDeviceId gpu_id_;

  // Peer access is configured at mmap time so the allocator must be aware of
  // all gpus that may want to read the memory. This list also includes the
  // above gpu_id_ to facilitate the invocation of the GpuDriver::MapMemory
  // function.
  const std::vector<stream_executor::gpu::GpuDeviceHandle> access_gpu_handles_;

  // The virtual memory span held by this allocator.
  stream_executor::gpu::GpuDriver::VmemSpan vmem_;
  // The next offset from the vmem base address that will be allocated. This
  // corresponds to the size of physically pinned memory if holes haven't been
  // created with "free".
  size_t next_alloc_offset_ = 0;

  // Smallest allocation as determined by CUDA.
  const size_t granularity_;

  struct Mapping {
    stream_executor::gpu::GpuDevicePtr va;
    stream_executor::gpu::GpuDriver::GenericMemoryHandle physical;
  };
  // List of mappings, sorted by va.
  std::vector<Mapping> mappings_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuVirtualMemAllocator);
};

}  // namespace tensorflow

#endif  // CUDA_VERSION >= 10200

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_
