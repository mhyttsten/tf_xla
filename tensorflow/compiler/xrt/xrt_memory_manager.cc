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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc() {
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

#include "tensorflow/compiler/xrt/xrt_memory_manager.h"

#include <algorithm>
#include <list>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace {

// We use kDeviceBits to store the device ordinal in the handle. We store the
// device in the upper part of the int64 handle to make sure the random bits are
// in the lower part which is better when storing the handle as a key for
// unordered maps.
const int kDeviceBits = 12;

int64_t MakeDeviceHandle(int64_t device_ordinal, int64_t rnd_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "MakeDeviceHandle");

  const int64_t kUidMask = (static_cast<int64_t>(1) << (64 - kDeviceBits)) - 1;
  return (device_ordinal << (64 - kDeviceBits)) | (rnd_value & kUidMask);
}

int GetDeviceFromHandle(int64_t handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "GetDeviceFromHandle");

  return (handle >> (64 - kDeviceBits)) & ((1 << kDeviceBits) - 1);
}

}  // namespace

class XRTMemoryManager::DeviceContext {
  struct Alloc {
    explicit Alloc(RefPtr<XRTTupleAllocation> tuple)
        : tuple(std::move(tuple)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "Alloc");
}

    RefPtr<XRTTupleAllocation> tuple;
  };

  using AllocList = std::list<Alloc>;

 public:
  int64_t Register(RefPtr<XRTTupleAllocation> tuple) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "Register");

    while (true) {
      int64_t handle = MakeDeviceHandle(tuple->device_ordinal(), CreateUid());
      mutex_lock lock(lock_);
      allocs_.emplace_front(tuple);
      if (alloc_map_.emplace(handle, allocs_.begin()).second) {
        return handle;
      }
      // The chances of hitting an existing handle are so remote, it is much
      // more convenient to add to the list before, and eventually removing.
      allocs_.erase(allocs_.begin());
    }
  }

  bool Release(int64_t handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "Release");

    mutex_lock lock(lock_);
    auto it = alloc_map_.find(handle);
    if (it == alloc_map_.end()) {
      return false;
    }
    allocs_.erase(it->second);
    alloc_map_.erase(it);
    return true;
  }

  RefPtr<XRTTupleAllocation> Lookup(int64_t handle) {
    mutex_lock lock(lock_);
    auto it = alloc_map_.find(handle);
    if (it == alloc_map_.end()) {
      return nullptr;
    }
    // LRU
    allocs_.splice(allocs_.begin(), allocs_, it->second);
    return it->second->tuple;
  }

  void Clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_5(mht_5_v, 279, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "Clear");

    mutex_lock lock(lock_);
    alloc_map_.clear();
    allocs_.clear();
  }

  Status CompactAllocations(XRTMemoryManager* memory_manager,
                            xla::Backend* backend,
                            se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_6(mht_6_v, 290, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "CompactAllocations");

    profiler::TraceMe trace_me("XRTMemoryManager::CompactAllocations",
                               /*level=*/2);
    auto timed = monitoring::MakeTimed(xrt_metrics::GetMemoryCompactCell());
    VLOG(4) << "CompactAllocations started";
    mutex_lock lock(lock_);
    Status status;
    std::vector<AllocList::iterator> swapped;
    // We are swapping out from the most recently used allocations. This is
    // desirable since the most recently used will be finding themselves at the
    // bottom of the allocation space. Since these are more likely to be pinned
    // allocations, a further trim done by following TryFreeMemory() call will
    // eventually drop the higher located allocations, with better chance of
    // reducing fragmentation.
    // Also, by swapping out the pinned allocations first, those will also be
    // the first to be restored, and hence if we will ever find OOM on the way
    // out, we would more likely be swapping in not pinned ones.
    for (auto it = allocs_.begin(); it != allocs_.end(); ++it) {
      // We are compacting all the allocations, so we will temporarily swap out
      // even pinned allocations.
      auto swap_result_or = it->tuple->SwapOut(backend, /*swap_pinned=*/true);
      if (!swap_result_or.ok()) {
        status = swap_result_or.status();
        break;
      }
      if (swap_result_or.ValueOrDie()) {
        swapped.push_back(it);
      }
    }
    // At this point we have released all the device memory we could release.
    // Load back the tuple allocations we have swapped out above.
    for (auto& it : swapped) {
      auto swap_result_or =
          it->tuple->SwapIn(memory_manager, backend, allocator);
      if (!swap_result_or.ok()) {
        // If we failed to restored a pinned allocation, better to CHECK here
        // than wondering why XRTTupleAllocation calls fail with errors about
        // missing buffers.
        CHECK(!it->tuple->IsPinned());  // Crash OK
        if (status.ok()) {
          status = swap_result_or.status();
        }
      }
    }
    VLOG(4) << "CompactAllocations finished: " << status;
    return status;
  }

  // Tries to free size bytes by freeing some unpinned device memory. Returns
  // the amount of memory which was able to free.
  xla::StatusOr<size_t> TryFreeMemory(xla::Backend* backend, size_t size) {
    profiler::TraceMe trace_me("XRTMemoryManager::TryFreeMemory", /*level=*/2);
    auto timed = monitoring::MakeTimed(xrt_metrics::GetTryFreeMemoryCell());
    mutex_lock lock(lock_);
    size_t swapped_size = 0;
    for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
      TF_ASSIGN_OR_RETURN(bool swap_result,
                          it->tuple->SwapOut(backend, /*swap_pinned=*/false));
      if (swap_result) {
        swapped_size += it->tuple->GetDeviceMemorySize();
        if (swapped_size >= size) {
          break;
        }
      }
    }
    VLOG(3) << "Swapped out " << swapped_size << " bytes";
    return swapped_size;
  }

 private:
  static int64_t CreateUid() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_7(mht_7_v, 363, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "CreateUid");

    int64_t uid;
    do {
      uid = random::New64() & INT64_MAX;
    } while (uid == InvalidKey());
    return uid;
  }

  // We store Alloc records inside an std::list<Alloc> so we can LRU it, and
  // store the list iterators within the handle map, as list iterators don't get
  // invalidated by (other elements) removals or position swaps.
  mutex lock_;
  AllocList allocs_;
  std::unordered_map<int64_t, AllocList::iterator> alloc_map_;
};

XRTMemoryManager::WorkingSet::WorkingSet(
    RefPtr<XRTMemoryManager> memory_manager)
    : memory_manager_(std::move(memory_manager)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_8(mht_8_v, 384, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::WorkingSet::WorkingSet");
}

XRTMemoryManager::WorkingSet::~WorkingSet() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_9(mht_9_v, 389, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::WorkingSet::~WorkingSet");

  for (auto& tuple : pinned_tuples_) {
    tuple->Unpin();
  }
}

Status XRTMemoryManager::WorkingSet::LookupAndPin(
    xla::Backend* backend, int64_t handle,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_10(mht_10_v, 400, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::WorkingSet::LookupAndPin");

  TF_ASSIGN_OR_RETURN(auto tuple, memory_manager_->Lookup(handle));
  TF_RETURN_IF_ERROR(
      tuple->PinAndSwapIn(memory_manager_.get(), backend, allocator).status());
  pinned_tuples_.push_back(std::move(tuple));
  return Status::OK();
}

/* static */ RefPtr<XRTMemoryManager> XRTMemoryManager::Get(ResourceMgr* rm) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_11(mht_11_v, 411, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::Get");

  static string* container = new string("XrtState");
  static string* name = new string("MemoryManager");
  XRTMemoryManager* memory_manager = nullptr;
  TF_CHECK_OK(rm->LookupOrCreate<XRTMemoryManager>(
      *container, *name, &memory_manager, [](XRTMemoryManager** ret) {
        *ret = new XRTMemoryManager();
        return Status::OK();
      }));
  return memory_manager;
}

int64_t XRTMemoryManager::Register(RefPtr<XRTTupleAllocation> tuple) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_12(mht_12_v, 426, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::Register");

  DeviceContext* device_context = GetDeviceContext(tuple->device_ordinal(),
                                                   /*create_if_missing=*/true);
  return device_context->Register(std::move(tuple));
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> XRTMemoryManager::Lookup(
    int64_t handle) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_13(mht_13_v, 436, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::Lookup");

  int device_ordinal = GetDeviceFromHandle(handle);
  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  RefPtr<XRTTupleAllocation> tuple = device_context->Lookup(handle);
  if (tuple == nullptr) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  return std::move(tuple);
}

Status XRTMemoryManager::Release(int64_t handle) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_14(mht_14_v, 453, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::Release");

  int device_ordinal = GetDeviceFromHandle(handle);
  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr || !device_context->Release(handle)) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  return Status::OK();
}

Status XRTMemoryManager::CompactAllocations(
    xla::Backend* backend, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_15(mht_15_v, 468, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::CompactAllocations");

  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  return device_context != nullptr
             ? device_context->CompactAllocations(this, backend, allocator)
             : Status::OK();
}

void XRTMemoryManager::ReleaseAllAllocations() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_16(mht_16_v, 479, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::ReleaseAllAllocations");

  mutex_lock lock(lock_);
  for (auto& device_context : device_contexts_) {
    if (device_context != nullptr) {
      device_context->Clear();
    }
  }
}

xla::StatusOr<se::OwningDeviceMemory> XRTMemoryManager::Allocate(
    xla::Backend* backend, int device_ordinal, size_t size,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_17(mht_17_v, 493, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::Allocate");

  auto memory_or =
      allocator->Allocate(device_ordinal, size, /*retry_on_failure=*/false);
  if (memory_or.status().code() == error::RESOURCE_EXHAUSTED) {
    VLOG(4) << "Allocate of " << size << " bytes failed on device "
            << device_ordinal;

    DeviceContext* device_context =
        GetDeviceContext(device_ordinal,
                         /*create_if_missing=*/false);
    if (device_context != nullptr) {
      Status status = device_context->TryFreeMemory(backend, size).status();
      if (status.ok()) {
        // As long as there is no error, we still try again the allocation, even
        // if the TryFreeMemory() call ended up freeing less memory than the
        // required size. Fragmentation could make the memory allocation succeed
        // even if the freed memory is indeed lower.
        memory_or = allocator->Allocate(device_ordinal, size,
                                        /*retry_on_failure=*/false);
      } else if (status.code() != error::RESOURCE_EXHAUSTED) {
        VLOG(4) << "Allocate of " << size << " bytes on device "
                << device_ordinal << ": " << status;
        return status;
      }
    }
  }
  return memory_or;
}

string XRTMemoryManager::DebugString() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_18(mht_18_v, 525, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::DebugString");

  // We might want to emit more detailed information here, like per device
  // memory allocations.
  return "XRTMemoryManager";
}

XRTMemoryManager::DeviceContext* XRTMemoryManager::GetDeviceContext(
    int device_ordinal, bool create_if_missing) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_19(mht_19_v, 535, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::GetDeviceContext");

  mutex_lock lock(lock_);
  if (device_ordinal >= device_contexts_.size()) {
    if (!create_if_missing) {
      return nullptr;
    }
    device_contexts_.resize(device_ordinal + 1);
  }
  DeviceContext* device_context = device_contexts_[device_ordinal].get();
  if (device_context == nullptr && create_if_missing) {
    device_contexts_[device_ordinal] = absl::make_unique<DeviceContext>();
    device_context = device_contexts_[device_ordinal].get();
  }
  return device_context;
}

Status XRTMemoryManager::TryFreeMemoryStep(MemoryReclaimContext* mrctx,
                                           const Status& status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTcc mht_20(mht_20_v, 555, "", "./tensorflow/compiler/xrt/xrt_memory_manager.cc", "XRTMemoryManager::TryFreeMemoryStep");

  DeviceContext* device_context = GetDeviceContext(mrctx->device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr) {
    return status;
  }
  if (!mrctx->done_freeing) {
    // If the caller passed us a zero requested_free_size, we try to free chunks
    // of kMaxFreeSize memory, until either the run function succeeds, or we run
    // out of freeable memory.
    const size_t kMaxFreeSize = 1000000000;
    size_t free_size =
        (mrctx->requested_free_size > 0)
            ? std::min<size_t>(mrctx->requested_free_size - mrctx->free_size,
                               kMaxFreeSize)
            : kMaxFreeSize;
    if (free_size > 0) {
      auto free_size_or =
          device_context->TryFreeMemory(mrctx->backend, free_size);
      if (!free_size_or.ok()) {
        return status;
      }
      size_t size = free_size_or.ValueOrDie();
      mrctx->free_size += size;
      if (size > 0) {
        return Status::OK();
      }
    }
    mrctx->done_freeing = true;
  }
  if (!mrctx->done_compacting) {
    mrctx->done_compacting = true;
    if (device_context
            ->CompactAllocations(this, mrctx->backend, mrctx->allocator)
            .ok()) {
      return Status::OK();
    }
  }
  return status;
}

}  // namespace tensorflow
