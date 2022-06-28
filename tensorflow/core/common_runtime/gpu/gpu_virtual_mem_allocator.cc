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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc() {
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

#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/stream_executor/lib/status.h"

#if CUDA_VERSION >= 10020

namespace tensorflow {
namespace {

using ::stream_executor::gpu::GpuContext;
using ::stream_executor::gpu::GpuDeviceHandle;
using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuDriver;
using ::stream_executor::port::Status;
using ::stream_executor::port::StatusOr;

// Rounds value up to the specified power of two alignment.
size_t AlignUp(size_t value, size_t alignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "AlignUp");

  DCHECK_EQ(alignment & (alignment - 1), 0)
      << "Alignment must be a power of two; alignment=" << alignment;
  return (value + alignment - 1) & ~(alignment - 1);
}

StatusOr<bool> SupportsVirtualAddressManagement(GpuDeviceHandle device) {
  return GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);
}

Status CheckVirtualAddressManagementSupport(GpuDeviceHandle device,
                                            PlatformDeviceId gpu_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "CheckVirtualAddressManagementSupport");

  TF_ASSIGN_OR_RETURN(bool supports_virtual_address_management,
                      SupportsVirtualAddressManagement(device));
  if (!supports_virtual_address_management) {
    return stream_executor::port::InternalError(absl::StrFormat(
        "GPU %d does not support virtual memory address management.",
        gpu_id.value()));
  }
  return {};
}

}  // namespace

/* static */ stream_executor::port::StatusOr<
    std::unique_ptr<GpuVirtualMemAllocator>>
GpuVirtualMemAllocator::Create(
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors, GpuContext& gpu_context,
    PlatformDeviceId gpu_id, size_t virtual_address_space_size,
    const std::vector<PlatformDeviceId>& peer_gpu_ids) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "GpuVirtualMemAllocator::Create");

  std::vector<GpuDeviceHandle> access_gpu_handles;
  access_gpu_handles.reserve(peer_gpu_ids.size() + 1);

  GpuDeviceHandle gpu_handle;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(gpu_id.value(), &gpu_handle));
  TF_RETURN_IF_ERROR(CheckVirtualAddressManagementSupport(gpu_handle, gpu_id));

  access_gpu_handles.push_back(gpu_handle);
  for (const auto& peer_id : peer_gpu_ids) {
    GpuDeviceHandle peer_handle;
    TF_RETURN_IF_ERROR(GpuDriver::GetDevice(peer_id.value(), &peer_handle));
    TF_ASSIGN_OR_RETURN(bool supports_virtual_address_management,
                        SupportsVirtualAddressManagement(peer_handle));
    if (GpuDriver::CanEnablePeerAccess(gpu_handle, peer_handle) &&
        supports_virtual_address_management) {
      access_gpu_handles.push_back(peer_handle);
    }
  }

  // Find the min granularity for all devices that have access to this memory;
  // that is, the maximum min granularity among all devices.
  size_t max_granularity = 1;
  for (const auto device_handle : access_gpu_handles) {
    TF_ASSIGN_OR_RETURN(size_t granularity,
                        GpuDriver::GetMinAllocationGranularity(device_handle));
    max_granularity = std::max(max_granularity, granularity);
  }

  // Create the virtual memory reservation. Must be aligned to system page size,
  // and larger than the CUDA min granularity. Empirically, the granularity
  // check is sufficient as the granularity is some multiple of the page size.
  // TODO(imintz): Create OS agnostic page size utility for completeness.
  TF_ASSIGN_OR_RETURN(
      GpuDriver::VmemSpan vmem,
      GpuDriver::ReserveVirtualMemory(
          &gpu_context, AlignUp(virtual_address_space_size, max_granularity)));
  VLOG(1) << "Reserved GPU virtual memory at " << vmem.base << " of size "
          << strings::HumanReadableNumBytes(vmem.size_bytes) << " bytes";

  return std::unique_ptr<GpuVirtualMemAllocator>(new GpuVirtualMemAllocator(
      alloc_visitors, free_visitors, gpu_context, gpu_id,
      std::move(access_gpu_handles), vmem, max_granularity));
}

GpuVirtualMemAllocator::GpuVirtualMemAllocator(
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors, GpuContext& gpu_context,
    PlatformDeviceId gpu_id,
    const std::vector<GpuDeviceHandle> access_gpu_handles,
    GpuDriver::VmemSpan vmem, size_t granularity)
    : SubAllocator(alloc_visitors, free_visitors),
      gpu_context_(gpu_context),
      gpu_id_(gpu_id),
      access_gpu_handles_(access_gpu_handles),
      vmem_(vmem),
      granularity_(granularity) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_3(mht_3_v, 300, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "GpuVirtualMemAllocator::GpuVirtualMemAllocator");
}

GpuVirtualMemAllocator::~GpuVirtualMemAllocator() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "GpuVirtualMemAllocator::~GpuVirtualMemAllocator");

  for (const auto mapping : mappings_) {
    GpuDriver::UnmapMemory(&gpu_context_, mapping.va, mapping.physical.bytes);
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(mapping.physical));
  }
  GpuDriver::FreeVirtualMemory(&gpu_context_, vmem_);
}

void* GpuVirtualMemAllocator::Alloc(size_t alignment, size_t num_bytes,
                                    size_t* bytes_received) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_5(mht_5_v, 317, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "GpuVirtualMemAllocator::Alloc");

  if (num_bytes == 0) return nullptr;
  size_t padded_bytes = (num_bytes + granularity_ - 1) & ~(granularity_ - 1);

  GpuDevicePtr next_va = vmem_.base + next_alloc_offset_;

  // TODO(imintz): Attempt to extend the vmem allocation by reserving additional
  // virtual memory at the specific address at the end of the initial vmem
  // reservation.
  if (next_va + padded_bytes > vmem_.base + vmem_.size_bytes) {
    LOG(ERROR) << "OOM in GPU virtual memory allocator when attempting to "
                  "allocate {request: "
               << strings::HumanReadableNumBytes(num_bytes)
               << ", aligned: " << padded_bytes << "} bytes.";
    return nullptr;
  }

  // Create physical memory backing allocation.
  auto maybe_handle =
      GpuDriver::CreateMemoryHandle(&gpu_context_, padded_bytes);
  if (!maybe_handle.ok()) {
    LOG(ERROR) << maybe_handle.status();
    return nullptr;
  }
  GpuDriver::GenericMemoryHandle handle = std::move(maybe_handle).ValueOrDie();

  // Map VAs for this physical memory.
  auto status =
      GpuDriver::MapMemory(&gpu_context_, next_va, handle, access_gpu_handles_);
  if (!status.ok()) {
    LOG(ERROR) << status;
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(handle));
    return nullptr;
  }
  next_alloc_offset_ += handle.bytes;
  mappings_.push_back({next_va, std::move(handle)});
  VisitAlloc(reinterpret_cast<void*>(next_va), gpu_id_.value(), padded_bytes);
  *bytes_received = padded_bytes;
  return reinterpret_cast<void*>(next_va);
}

void GpuVirtualMemAllocator::Free(void* ptr, size_t num_bytes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_virtual_mem_allocatorDTcc mht_6(mht_6_v, 361, "", "./tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.cc", "GpuVirtualMemAllocator::Free");

  if (ptr == nullptr) return;

  auto mapping_it =
      std::lower_bound(mappings_.begin(), mappings_.end(), ptr,
                       [](const Mapping& mapping, const void* ptr) {
                         return reinterpret_cast<const void*>(mapping.va) < ptr;
                       });
  if (mapping_it == mappings_.end() ||
      (reinterpret_cast<void*>(mapping_it->va) != ptr)) {
    LOG(ERROR) << "Could not find GPU vmem mapping for address at "
               << reinterpret_cast<uintptr_t>(ptr);
    return;
  }

  int num_mappings_to_free = 0;
  int total_bytes = 0;
  for (auto it = mapping_it; it != mappings_.end() && total_bytes < num_bytes;
       ++it) {
    ++num_mappings_to_free;
    total_bytes += it->physical.bytes;
  }
  if (total_bytes != num_bytes) {
    LOG(ERROR) << "Invalid size requested for freeing GPU vmem mapping. Got "
               << strings::HumanReadableNumBytes(num_bytes) << " but expected "
               << strings::HumanReadableNumBytes(mapping_it->physical.bytes);
    return;
  }

  VLOG(1) << "Freeing " << num_mappings_to_free << " mappings for a total of "
          << total_bytes << " bytes";
  for (auto it = mapping_it; it < mapping_it + num_mappings_to_free; ++it) {
    GpuDriver::UnmapMemory(&gpu_context_, it->va, it->physical.bytes);
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(it->physical));
  }

  // Move back the next_alloc_offset_ if this free was at the end.
  if (mapping_it + num_mappings_to_free == mappings_.end()) {
    next_alloc_offset_ = mapping_it->va - vmem_.base;
  }

  mappings_.erase(mapping_it, mapping_it + num_mappings_to_free);
  VisitFree(ptr, gpu_id_.value(), num_bytes);
}

}  // namespace tensorflow

#endif
