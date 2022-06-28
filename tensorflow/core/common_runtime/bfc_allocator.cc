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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc() {
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

#include "tensorflow/core/common_runtime/bfc_allocator.h"

#include <algorithm>
#include <atomic>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#ifdef TENSORFLOW_MEM_DEBUG
#include "tensorflow/core/platform/stacktrace.h"
#endif
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"

namespace tensorflow {

constexpr BFCAllocator::ChunkHandle BFCAllocator::kInvalidChunkHandle;

BFCAllocator::BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator,
                           size_t total_memory, const string& name,
                           const Options& opts)
    : opts_(opts),
      coalesce_regions_(sub_allocator->SupportsCoalescing()),
      sub_allocator_(std::move(sub_allocator)),
      name_(name),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::BFCAllocator");

  if (opts.allow_growth) {
    // 2MiB smallest initial allocation, unless total memory available
    // is less.
    curr_region_allocation_bytes_ =
        RoundedBytes(std::min(total_memory, size_t{2 << 20}));
  } else {
    curr_region_allocation_bytes_ = RoundedBytes(total_memory);
  }

  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  stats_.bytes_limit = static_cast<int64_t>(total_memory);

  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  VLOG(1) << "Creating new BFCAllocator named: " << name;
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    VLOG(1) << "Creating bin of max chunk size "
            << strings::HumanReadableNumBytes(bin_size);
    new (BinFromIndex(b)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size + 255), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size * 2 - 1), BinFromIndex(b));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2), BinFromIndex(b));
    }
  }
}

BFCAllocator::~BFCAllocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::~BFCAllocator");

  // Return memory back.
  VLOG(2) << "Number of regions allocated: "
          << region_manager_.regions().size();
  for (const auto& region : region_manager_.regions()) {
    sub_allocator_->Free(region.ptr(), region.memory_size());
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_2(mht_2_v, 274, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::ChunkFromHandle");

  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

const BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_3(mht_3_v, 283, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::ChunkFromHandle");

  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

bool BFCAllocator::Extend(size_t alignment, size_t rounded_bytes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::Extend");

  size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    return false;
  }

  // If curr_region_allocation_bytes_ is not enough to satisfy the
  // allocation, keep multiplying by a power of two until that is
  // sufficient.
  bool increased_allocation = false;
  while (rounded_bytes > curr_region_allocation_bytes_) {
    curr_region_allocation_bytes_ *= 2;
    increased_allocation = true;
  }

  // Try allocating.
  size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
  size_t bytes_received;
  void* mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
  if (mem_addr == nullptr && !started_backpedal_) {
    // Only backpedal once.
    started_backpedal_ = true;

    static constexpr float kBackpedalFactor = 0.9;

    // Try allocating less memory.
    while (mem_addr == nullptr) {
      bytes = RoundedBytes(bytes * kBackpedalFactor);
      if (bytes < rounded_bytes) break;
      mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
    }
  }

  if (mem_addr == nullptr) {
    return false;
  }

  if (!increased_allocation) {
    // Increase the region size of the next required allocation.
    curr_region_allocation_bytes_ *= 2;
  }

  VLOG(1) << "Extending allocation by "
          << strings::HumanReadableNumBytes(bytes_received) << " bytes for "
          << Name() << ".";

  total_region_allocated_bytes_ += bytes_received;
  VLOG(1) << "Total allocated bytes: "
          << strings::HumanReadableNumBytes(total_region_allocated_bytes_);

  VLOG(1) << "Allocated memory at " << mem_addr << " to "
          << static_cast<void*>(static_cast<char*>(mem_addr) + bytes_received);

  AllocationRegion* maybe_extended_region = nullptr;
  if (coalesce_regions_) {
    maybe_extended_region =
        region_manager_.AddOrExtendAllocationRegion(mem_addr, bytes_received);
  } else {
    region_manager_.AddAllocationRegion(mem_addr, bytes_received);
  }

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes_received;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->freed_at_count = 0;

  region_manager_.set_handle(c->ptr, h);

  // If the region was extended, then there exists a previous chunk that should
  // be linked to the new chunk.
  if (maybe_extended_region != nullptr) {
    ChunkHandle prev =
        maybe_extended_region->get_handle(maybe_extended_region->ptr());
    BFCAllocator::Chunk* prev_chunk = ChunkFromHandle(prev);
    // Find the last recorded chunk in the extended region.
    while (prev_chunk->next != kInvalidChunkHandle) {
      prev = prev_chunk->next;
      prev_chunk = ChunkFromHandle(prev);
    }
    c->prev = prev;
    prev_chunk->next = h;
  }

  // Maybe merge adjacent chunks and insert the chunk into the right bin.
  InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

  return true;
}

BFCAllocator::ChunkHandle BFCAllocator::AllocateChunk() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_5(mht_5_v, 395, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocateChunk");

  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
  }
}

void BFCAllocator::DeallocateChunk(ChunkHandle h) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_6(mht_6_v, 411, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeallocateChunk");

  Chunk* c = ChunkFromHandle(h);
  c->allocation_id = -1;
  c->bin_num = kInvalidBinNum;
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

void* BFCAllocator::AllocateRawInternalWithRetry(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_7(mht_7_v, 424, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocateRawInternalWithRetry");

  // Fast path: Try once to allocate without getting the retry_helper_ involved
  uint64 freed_by_count = 0;
  if (allocation_attr.freed_by_func != nullptr) {
    freed_by_count = (*allocation_attr.freed_by_func)();
  }
  void* r =
      AllocateRawInternal(unused_alignment, num_bytes, false, freed_by_count);
  if (r != nullptr) {
    return r;
  } else {
    static const int64_t kMaxMillisToWait = 10000;  // 10 seconds
    r = retry_helper_.AllocateRaw(
        [this, &allocation_attr](size_t a, size_t nb, bool v) {
          uint64 freed_by_count = 0;
          if (allocation_attr.freed_by_func != nullptr) {
            freed_by_count = (*allocation_attr.freed_by_func)();
          }
          return AllocateRawInternal(a, nb, v, freed_by_count);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
    return r;
  }
}

void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
                                const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_8(mht_8_v, 453, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocateRaw");

  VLOG(3) << "AllocateRaw " << Name() << "  " << num_bytes;
  void* result = [&] {
    if (!opts_.allow_retry_on_failure || !allocation_attr.retry_on_failure) {
      // If we have globally disabled retry-on-failure and fail to allocate an
      // "important" alloc, we want to print a log, because the program may be
      // about to fail due to OOM.
      //
      // Bit of a hack: We deem "important" allocs as those which are retryable.
      // In TF, *non*-retryable allocations are usually those which we can
      // tolerate failing.  For example, we allocate convolution scratch memory
      // as non-retryable; if it fails, we'll just use a fallback algorithm that
      // uses no scratch.
      static std::atomic<int32> log_counter{0};
      constexpr int kMaxFailureLogs = 10;
      bool dump_log_on_failure =
          (/*retry is globally disabled*/ !opts_.allow_retry_on_failure &&
           /*alloc is "important"*/ allocation_attr.retry_on_failure &&
           log_counter.load(std::memory_order_relaxed) < kMaxFailureLogs) ||
          VLOG_IS_ON(2);

      uint64 freed_by_count = 0;
      if (allocation_attr.freed_by_func != nullptr) {
        freed_by_count = (*allocation_attr.freed_by_func)();
      }
      void* res = AllocateRawInternal(unused_alignment, num_bytes,
                                      dump_log_on_failure, freed_by_count);
      if (res == nullptr) {
        int32 counter_value = log_counter.load(std::memory_order_relaxed);
        if (counter_value < kMaxFailureLogs) {
          log_counter.store(counter_value + 1, std::memory_order_relaxed);
          LOG(WARNING)
              << "Allocator (" << Name() << ") ran out of memory trying "
              << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
              << " with freed_by_count=" << freed_by_count << "."
              << (!allocation_attr.retry_on_failure
                      ? " The caller indicates that this is not a failure, but"
                        " this may mean that there could be performance gains "
                        "if more memory were available."
                      : "");
        }
      }
      return res;
    } else {
      return AllocateRawInternalWithRetry(unused_alignment, num_bytes,
                                          allocation_attr);
    }
  }();
  VLOG(3) << "AllocateRaw " << Name() << "  " << num_bytes << " " << result;
  return result;
}

// static
size_t BFCAllocator::RoundedBytes(size_t bytes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_9(mht_9_v, 509, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RoundedBytes");

  size_t rounded_bytes =
      (kMinAllocationSize *
       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
  DCHECK_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

bool BFCAllocator::DeallocateFreeRegions(size_t rounded_bytes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_10(mht_10_v, 521, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeallocateFreeRegions");

  // Do nothing if garbage collection is off.
  if (!opts_.garbage_collection) {
    return false;
  }

  // Searching for free regions.
  absl::flat_hash_set<void*> free_region_ptrs;
  size_t total_free_bytes = 0;
  for (const AllocationRegion& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    bool any_use = false;
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        any_use = true;
        break;
      }
      h = c->next;
    }

    if (!any_use) {
      VLOG(2) << "Found free region with ptr = " << region.ptr();
      free_region_ptrs.insert(region.ptr());
      total_free_bytes += region.memory_size();
    }
  }

  if (total_free_bytes == 0) {
    return false;
  }

  // Rough estimation to check whether deallocation can help.
  size_t available_bytes =
      memory_limit_ - total_region_allocated_bytes_ + total_free_bytes;
  if (rounded_bytes > available_bytes) {
    return false;
  }

  LOG(WARNING) << "Garbage collection: deallocate free memory regions"
               << " (i.e., allocations) so that we can re-allocate a larger"
               << " region to avoid OOM due to memory fragmentation. If you"
               << " see this message frequently, you are running near the"
               << " threshold of the available device memory and re-allocation"
               << " may incur great performance overhead. You may try smaller"
               << " batch sizes to observe the performance impact."
               << " Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to"
               << " disable this feature.";

  // Deallocate free regions.
  DeallocateRegions(free_region_ptrs);

  return true;
}

void BFCAllocator::DeallocateRegions(
    const absl::flat_hash_set<void*>& region_ptrs)
    TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_11(mht_11_v, 581, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeallocateRegions");

  // Explicitly remove the const qualifier as some compilers disallow passing
  // const_iterator to std::vector::erase(), which is used in
  // RemoveAllocationRegion().
  auto regions =
      const_cast<std::vector<AllocationRegion>*>(&region_manager_.regions());
  auto it = regions->begin();
  while (it != regions->end()) {
    if (!region_ptrs.contains(it->ptr())) {
      ++it;
      continue;
    }

    VLOG(2) << "Deallocate region with ptr = " << it->ptr();
    // Remove all chunk registrations from Bins.
    ChunkHandle h = region_manager_.get_handle(it->ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->bin_num != kInvalidBinNum) {
        RemoveFreeChunkFromBin(h);
      }
      auto h_to_delete = h;
      h = c->next;
      DeleteChunk(h_to_delete);
    }

    // Deallocate the memory.
    sub_allocator_->Free(it->ptr(), it->memory_size());
    total_region_allocated_bytes_ -= it->memory_size();
    it = region_manager_.RemoveAllocationRegion(it);
  }
}

void* BFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                        size_t num_bytes,
                                        bool dump_log_on_failure,
                                        uint64 freed_before) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_12(mht_12_v, 620, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocateRawInternal");

  if (num_bytes == 0) {
    VLOG(2) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  mutex_lock l(lock_);
  if (!timestamped_chunks_.empty()) {
    // Merge timestamped chunks whose counts have become safe for general use.
    MergeTimestampedChunks(0);
  }
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
  if (ptr != nullptr) {
    AddTraceMe("MemoryAllocation", ptr);
    return ptr;
  }

  // Try to extend
  if (Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr) {
      AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  if ((freed_before == 0) && (!timestamped_chunks_.empty())) {
    // We're unable to satisfy an allocation request without a specific
    // timestamp requirement.  Rather than fail, try merging any held-out
    // timestamped chunks more aggressively until a free chunk of the necessary
    // size is formed.
    if (MergeTimestampedChunks(rounded_bytes)) {
      ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
      if (ptr != nullptr) {
        AddTraceMe("MemoryAllocation", ptr);
        return ptr;
      }
    }
  }

  // Reaching this point means that no chunks can satisfy the request. Also,
  // the unallocated bytes cannot satisfy the request. Before giving up, let's
  // try deallocating free regions so that suballocator can combine them with
  // the unallocated bytes and form a larger region.
  if (DeallocateFreeRegions(rounded_bytes) &&
      Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr) {
      AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  MaybeWriteMemoryMap();
  if (dump_log_on_failure) {
    LOG(WARNING)
        << "Allocator (" << Name() << ") ran out of memory trying "
        << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
        << " (rounded to " << rounded_bytes << ")"
        << "requested by op "
        << profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation()
               .pending_op_name
        << "\nIf the cause is memory fragmentation maybe the environment "
        << "variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will "
        << "improve the situation. \nCurrent allocation summary follows."
        << "\nCurrent allocation summary follows.";
    DumpMemoryLog(rounded_bytes);
    LOG(WARNING) << RenderOccupancy();
  }
  return nullptr;
}

int64_t BFCAllocator::LargestFreeChunk() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_13(mht_13_v, 705, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::LargestFreeChunk");

  for (int i = kNumBins - 1; i >= 0; i--) {
    if (!BinFromIndex(i)->free_chunks.empty()) {
      return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())->size;
    }
  }
  return 0;
}

double BFCAllocator::GetFragmentation() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_14(mht_14_v, 717, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::GetFragmentation");

  int64_t bytes_available = total_region_allocated_bytes_ - stats_.bytes_in_use;
  DCHECK_GT(bytes_available, 0);
  return static_cast<double>(bytes_available - LargestFreeChunk()) /
         bytes_available;
}

void BFCAllocator::AddTraceMe(absl::string_view traceme_name, const void* ptr) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("traceme_name: \"" + std::string(traceme_name.data(), traceme_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_15(mht_15_v, 728, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AddTraceMe");

  BFCAllocator::Chunk* chunk = ChunkFromHandle(region_manager_.get_handle(ptr));
  AddTraceMe(traceme_name, chunk->ptr, chunk->requested_size, chunk->size);
}

void BFCAllocator::AddTraceMe(absl::string_view traceme_name,
                              const void* chunk_ptr, int64_t req_bytes,
                              int64_t alloc_bytes) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("traceme_name: \"" + std::string(traceme_name.data(), traceme_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_16(mht_16_v, 739, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AddTraceMe");

  tensorflow::profiler::TraceMe::InstantActivity(
      [this, traceme_name, chunk_ptr, req_bytes, alloc_bytes]()
          TF_NO_THREAD_SAFETY_ANALYSIS {
            int64_t bytes_available =
                memory_limit_ - stats_.bytes_reserved - stats_.bytes_in_use;
            const auto& annotation =
                profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
            return tensorflow::profiler::TraceMeEncode(
                traceme_name, {{"allocator_name", name_},
                               {"bytes_reserved", stats_.bytes_reserved},
                               {"bytes_allocated", stats_.bytes_in_use},
                               {"bytes_available", bytes_available},
                               {"fragmentation", GetFragmentation()},
                               {"peak_bytes_in_use", stats_.peak_bytes_in_use},
                               {"requested_bytes", req_bytes},
                               {"allocation_bytes", alloc_bytes},
                               {"addr", reinterpret_cast<uint64>(chunk_ptr)},
                               {"tf_op", annotation.pending_op_name},
                               {"id", annotation.pending_step_id},
                               {"region_type", annotation.pending_region_type},
                               {"data_type", annotation.pending_data_type},
                               {"shape", annotation.pending_shape_func()}});
          },
      /*level=*/profiler::TraceMeLevel::kInfo);
}

void* BFCAllocator::FindChunkPtr(BinNum bin_num, size_t rounded_bytes,
                                 size_t num_bytes, uint64 freed_before) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_17(mht_17_v, 770, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::FindChunkPtr");

  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const BFCAllocator::ChunkHandle h = (*citer);
      BFCAllocator::Chunk* chunk = ChunkFromHandle(h);
      DCHECK(!chunk->in_use());
      if (freed_before > 0 && freed_before < chunk->freed_at_count) {
        continue;
      }
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do don't waste more than max_internal_fragmentation_bytes on
        // padding. If this threshold is not set by the user, then use 128MB as
        // the default.
        const int64_t max_internal_fragmentation_bytes =
            (opts_.fragmentation_fraction > 0.0)
                ? opts_.fragmentation_fraction * memory_limit_
                : 128 << 20;

        if (chunk->size >= rounded_bytes * 2 ||
            static_cast<int64_t>(chunk->size) - rounded_bytes >=
                max_internal_fragmentation_bytes) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
          VLOG(2) << "New Peak memory usage of " << stats_.bytes_in_use
                  << " bytes for " << Name();
        }
        stats_.peak_bytes_in_use =
            std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
        stats_.largest_alloc_size =
            std::max<std::size_t>(stats_.largest_alloc_size, chunk->size);

#ifdef TENSORFLOW_MEM_DEBUG
        if (ShouldRecordOpName()) {
          const auto& annotation =
              profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
          if (annotation.pending_op_name != nullptr) {
            chunk->op_name = annotation.pending_op_name;
          } else {
            LOG(INFO) << "missing pending_op_name for " << Name()
                      << " reading addr "
                      << static_cast<const void*>(&annotation.pending_op_name)
                      << "\n"
                      << CurrentStackTrace();
            chunk->op_name = nullptr;
          }
          chunk->action_count = ++action_counter_;
          chunk->step_id = annotation.pending_step_id;
          int slot = chunk->action_count % MEM_DEBUG_SIZE_HISTORY_SIZE;
          size_history_[slot] = stats_.bytes_in_use;
        }
#endif

        VLOG(4) << "Returning: " << chunk->ptr;
        if (VLOG_IS_ON(4)) {
          LOG(INFO) << "A: " << RenderOccupancy();
        }
        return chunk->ptr;
      }
    }
  }

  return nullptr;
}

void BFCAllocator::SplitChunk(BFCAllocator::ChunkHandle h, size_t num_bytes) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_18(mht_18_v, 860, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::SplitChunk");

  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // It inherits the freed time.
  new_chunk->freed_at_count = c->freed_at_count;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCAllocator::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void BFCAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_19(mht_19_v, 901, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeallocateRaw");

  VLOG(3) << "DeallocateRaw " << Name() << " "
          << (ptr ? RequestedSize(ptr) : 0);
  DeallocateRawInternal(ptr);
  retry_helper_.NotifyDealloc();
}

void BFCAllocator::DeallocateRawInternal(void* ptr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_20(mht_20_v, 911, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeallocateRawInternal");

  if (ptr == nullptr) {
    VLOG(2) << "tried to deallocate nullptr";
    return;
  }
  mutex_lock l(lock_);

  // Find the chunk from the ptr.
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle);
  // Record chunk information before it's freed.
  Chunk* chunk = ChunkFromHandle(h);
  void* chunk_ptr = chunk->ptr;
  int64_t req_bytes = chunk->requested_size;
  int64_t alloc_bytes = chunk->size;

  MarkFree(h);

  // Consider coalescing it.
  if (timing_counter_) {
    InsertFreeChunkIntoBin(h);
    timestamped_chunks_.push_back(h);
  } else {
    InsertFreeChunkIntoBin(TryToCoalesce(h, false));
  }

  // TraceMe needs to be added after MarkFree and InsertFreeChunkIntoBin for
  // correct aggregation stats (bytes_in_use, fragmentation).
  AddTraceMe("MemoryDeallocation", chunk_ptr, req_bytes, alloc_bytes);

  if (VLOG_IS_ON(4)) {
    LOG(INFO) << "F: " << RenderOccupancy();
  }
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void BFCAllocator::Merge(BFCAllocator::ChunkHandle h1,
                         BFCAllocator::ChunkHandle h2) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_21(mht_21_v, 952, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::Merge");

  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCAllocator::ChunkHandle h3 = c2->next;
  c1->next = h3;
  CHECK(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  // Pick latest free time.
  c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

  DeleteChunk(h2);
}

void BFCAllocator::DeleteChunk(ChunkHandle h) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_22(mht_22_v, 986, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DeleteChunk");

  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  //  VLOG(4) << "Removing: " << c->ptr;
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void BFCAllocator::InsertFreeChunkIntoBin(BFCAllocator::ChunkHandle h) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_23(mht_23_v, 997, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::InsertFreeChunkIntoBin");

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void BFCAllocator::RemoveFreeChunkIterFromBin(
    BFCAllocator::Bin::FreeChunkSet* free_chunks,
    const BFCAllocator::Bin::FreeChunkSet::iterator& citer) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_24(mht_24_v, 1011, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RemoveFreeChunkIterFromBin");

  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void BFCAllocator::RemoveFreeChunkFromBin(BFCAllocator::ChunkHandle h) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_25(mht_25_v, 1022, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RemoveFreeChunkFromBin");

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  CHECK_GT(BinFromIndex(c->bin_num)->free_chunks.erase(h), 0)
      << "Could not find chunk in bin";
  c->bin_num = kInvalidBinNum;
}

void BFCAllocator::MarkFree(BFCAllocator::ChunkHandle h) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_26(mht_26_v, 1033, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::MarkFree");

  Chunk* c = ChunkFromHandle(h);
  CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

  // Mark the chunk as no longer in use.
  c->allocation_id = -1;

  // Optionally record the free time.
  if (timing_counter_) {
    c->freed_at_count = timing_counter_->next();
  }

  // Updates the stats.
  stats_.bytes_in_use -= c->size;

#ifdef TENSORFLOW_MEM_DEBUG
  if (ShouldRecordOpName()) {
    c->action_count = ++action_counter_;
    int slot = c->action_count % MEM_DEBUG_SIZE_HISTORY_SIZE;
    size_history_[slot] = stats_.bytes_in_use;
  }
#endif
}

BFCAllocator::ChunkHandle BFCAllocator::TryToCoalesce(ChunkHandle h,
                                                      bool ignore_freed_at) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_27(mht_27_v, 1061, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::TryToCoalesce");

  Chunk* c = ChunkFromHandle(h);
  if ((!ignore_freed_at) && c->freed_at_count > 0) return h;
  ChunkHandle coalesced_chunk = h;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use()) {
    Chunk* n = ChunkFromHandle(c->next);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      VLOG(4) << "Merging c->next " << n->ptr << " with c " << c->ptr;
      RemoveFreeChunkFromBin(c->next);
      Merge(h, c->next);
    }
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use()) {
    Chunk* n = ChunkFromHandle(c->prev);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      VLOG(4) << "Merging c " << c->ptr << " into c->prev " << n->ptr;
      coalesced_chunk = c->prev;
      RemoveFreeChunkFromBin(c->prev);
      Merge(c->prev, h);
    }
  }

  return coalesced_chunk;
}

void BFCAllocator::SetSafeFrontier(uint64 count) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_28(mht_28_v, 1093, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::SetSafeFrontier");

  uint64 current = safe_frontier_.load(std::memory_order_relaxed);
  while (count > current) {
    if (safe_frontier_.compare_exchange_strong(current, count)) {
      retry_helper_.NotifyDealloc();
      return;
    } else {
      current = safe_frontier_.load(std::memory_order_relaxed);
    }
  }
}

bool BFCAllocator::MergeTimestampedChunks(size_t required_bytes) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_29(mht_29_v, 1108, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::MergeTimestampedChunks");

  VLOG(1) << "MergeTimestampedChunks queue_len=" << timestamped_chunks_.size()
          << " required_bytes=" << required_bytes;
  bool satisfied = (required_bytes == 0);
  std::vector<void*> to_merge;
  std::deque<ChunkHandle> new_ts_queue;
  while (!timestamped_chunks_.empty()) {
    ChunkHandle h = timestamped_chunks_.front();
    timestamped_chunks_.pop_front();
    DCHECK_NE(h, kInvalidChunkHandle);
    Chunk* c = ChunkFromHandle(h);
    // It's possible this chunk has already been merged so refetch and retest
    // the handle.
    h = region_manager_.get_handle(c->ptr);
    if (h == kInvalidChunkHandle) {
      continue;
    }
    if (c->in_use() || (c->bin_num == kInvalidBinNum)) {
      // This chunk has already been reallocated.
      continue;
    }
    if (c->freed_at_count == 0) {
      to_merge.push_back(c->ptr);
      continue;
    }
    // Chunk should be free and assigned to a bin.
    DCHECK_NE(c->bin_num, kInvalidBinNum);
    if (c->freed_at_count < safe_frontier_) {
      c->freed_at_count = 0;
      to_merge.push_back(c->ptr);
    } else if (required_bytes > 0) {
      to_merge.push_back(c->ptr);
    } else {
      new_ts_queue.push_back(h);
    }
  }
  DCHECK(timestamped_chunks_.empty());
  std::swap(timestamped_chunks_, new_ts_queue);

  // At this point all candidate chunks have been moved from timestamped_chunks_
  // to to_merge.  If this is a standard merge (required_bytes == 0) then
  // merge them all, otherwise merge just until a Chunk of the required size
  // is produced.
  for (int ci = 0, end = to_merge.size(); ci < end; ++ci) {
    void* ptr = to_merge[ci];
    // It's possible that the Chunk associated with this memory location got
    // merged and deallocated in a prior iteration so refetch the handle and
    // retest.
    ChunkHandle h = region_manager_.get_handle(ptr);
    if (h == kInvalidChunkHandle) continue;
    if (required_bytes == 0 || !satisfied) {
      Chunk* c = ChunkFromHandle(h);
      DCHECK_NE(c->bin_num, kInvalidBinNum);
      DCHECK(!c->in_use());
      RemoveFreeChunkFromBin(h);
      ChunkHandle new_h = TryToCoalesce(h, (required_bytes > 0));
      InsertFreeChunkIntoBin(new_h);
      if (required_bytes > 0) {
        c = ChunkFromHandle(new_h);
        if (new_h != h && c->freed_at_count > 0) {
          timestamped_chunks_.push_back(new_h);
        }
        if (c->size >= required_bytes) {
          satisfied = true;
        }
      }
    } else {
      // We were force merging Chunks with unsafe timestamps, but managed
      // to create a satisfying Chunk so just requeue the rest.
      timestamped_chunks_.push_back(h);
    }
  }
  return satisfied;
}

bool BFCAllocator::TracksAllocationSizes() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_30(mht_30_v, 1186, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::TracksAllocationSizes");
 return true; }

size_t BFCAllocator::RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_31(mht_31_v, 1191, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RequestedSize");

  CHECK(ptr);
  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for requested size of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t BFCAllocator::AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_32(mht_32_v, 1204, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocatedSize");

  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

int64_t BFCAllocator::AllocationId(const void* ptr) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_33(mht_33_v, 1216, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::AllocationId");

  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocation id of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->allocation_id;
}

namespace {

void RenderRegion(char* rendered, const size_t resolution,
                  const size_t total_render_size, const size_t offset,
                  const void* base_ptr, const void* ptr, const size_t size,
                  const char c) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("rendered: \"" + (rendered == nullptr ? std::string("nullptr") : std::string((char*)rendered)) + "\"");
   mht_34_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_34(mht_34_v, 1235, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "RenderRegion");

  const char* base_ptr_c = static_cast<const char*>(base_ptr);
  const char* ptr_c = static_cast<const char*>(ptr);

  size_t start_location =
      ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
  CHECK_GE(start_location, 0);
  CHECK_LT(start_location, resolution);
  size_t end_location =
      ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) /
      total_render_size;
  CHECK_GE(end_location, 0);
  CHECK_LT(end_location, resolution);

  for (size_t i = start_location; i <= end_location; ++i) {
    rendered[i] = c;
  }
}

}  // namespace

string BFCAllocator::RenderOccupancy() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_35(mht_35_v, 1259, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RenderOccupancy");

  // Make a buffer for the ASCII-art representation.
  const size_t resolution = 100;
  char rendered[resolution];

  // Compute the total region size to render over
  size_t total_region_size = 0;
  for (const auto& region : region_manager_.regions()) {
    total_region_size += region.memory_size();
  }

  if (total_region_size == 0) {
    return "<allocator contains no memory>";
  }

  // Start out with everything empty
  RenderRegion(rendered, resolution, total_region_size, 0, nullptr, nullptr,
               total_region_size, '_');

  size_t region_offset = 0;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    // Then render each chunk left to right.
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        // Render the wasted space
        size_t wasted = c->size - c->requested_size;
        if (wasted > 0) {
          RenderRegion(rendered, resolution, total_region_size,
                       region_offset + c->requested_size, region.ptr(), c->ptr,
                       wasted, 'x');
        }
        // Then the occupied space
        RenderRegion(rendered, resolution, total_region_size, region_offset,
                     region.ptr(), c->ptr, c->requested_size, '*');
      }
      h = c->next;
    }
    region_offset += region.memory_size();
  }

  return string(rendered, resolution);
}

void BFCAllocator::DumpMemoryLog(size_t num_bytes) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_36(mht_36_v, 1307, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::DumpMemoryLog");

  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  LOG(INFO) << "BFCAllocator dump for " << Name();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    CHECK_EQ(b->free_chunks.size(),
             bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

    LOG(INFO) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << bin_info.total_chunks_in_bin
              << ", Chunks in use: " << bin_info.total_chunks_in_use << ". "
              << strings::HumanReadableNumBytes(bin_info.total_bytes_in_bin)
              << " allocated for chunks. "
              << strings::HumanReadableNumBytes(bin_info.total_bytes_in_use)
              << " in use in bin. "
              << strings::HumanReadableNumBytes(
                     bin_info.total_requested_bytes_in_use)
              << " client-requested in use in bin.";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOG(INFO) << "Bin for " << strings::HumanReadableNumBytes(num_bytes)
            << " was " << strings::HumanReadableNumBytes(b->bin_size)
            << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOG(INFO) << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    LOG(INFO) << "Next region of size " << region.memory_size();
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      string buf = strings::StrCat(
          (c->in_use() ? "InUse" : "Free "), " at ",
          strings::Hex(reinterpret_cast<uint64>(c->ptr)), " of size ", c->size);
#ifdef TENSORFLOW_MEM_DEBUG
      if (ShouldRecordOpName()) {
        strings::StrAppend(&buf, " by op ", c->op_name, " action_count ",
                           c->action_count, " step ", c->step_id);
      }
#endif
      strings::StrAppend(&buf, " next ", c->next);
      if (timing_counter_) {
        strings::StrAppend(&buf, " freed_at_count ", c->freed_at_count);
      }
      LOG(INFO) << buf;
      h = c->next;
    }
  }

  LOG(INFO) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO) << it.second << " Chunks of size " << it.first << " totalling "
              << strings::HumanReadableNumBytes(it.first * it.second);
    total_bytes += (it.first * it.second);
  }
  LOG(INFO) << "Sum Total of in-use chunks: "
            << strings::HumanReadableNumBytes(total_bytes);
  LOG(INFO) << "total_region_allocated_bytes_: "
            << total_region_allocated_bytes_
            << " memory_limit_: " << memory_limit_ << " available bytes: "
            << (memory_limit_ - total_region_allocated_bytes_)
            << " curr_region_allocation_bytes_: "
            << curr_region_allocation_bytes_;
  LOG(INFO) << "Stats: \n" << stats_.DebugString();
}

void BFCAllocator::MaybeWriteMemoryMap() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_37(mht_37_v, 1391, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::MaybeWriteMemoryMap");

  const char* gpu_memory_map_file = std::getenv("TF_BFC_MEMORY_DUMP");
  if (gpu_memory_map_file != nullptr) {
    std::unique_ptr<WritableFile> dump_file;
    string file_name = strings::StrCat(gpu_memory_map_file, "_", Name(), ".",
                                       Env::Default()->NowMicros());
    Status status = Env::Default()->NewWritableFile(file_name, &dump_file);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open file " << file_name;
      return;
    }
    MemoryDump md = RecordMemoryMapInternal();
    status = dump_file->Append(md.SerializeAsString());
    if (!status.ok()) {
      LOG(ERROR) << "Error on writing to file " << gpu_memory_map_file << ": "
                 << status;
    }
  }
}

MemoryDump BFCAllocator::RecordMemoryMap() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_38(mht_38_v, 1414, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RecordMemoryMap");

  mutex_lock l(lock_);
  return RecordMemoryMapInternal();
}

MemoryDump BFCAllocator::RecordMemoryMapInternal() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_39(mht_39_v, 1422, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::RecordMemoryMapInternal");

  MemoryDump md;
  md.set_allocator_name(Name());

  // Record the general stats
  MemAllocatorStats* mas = md.mutable_stats();
  mas->set_num_allocs(stats_.num_allocs);
  mas->set_bytes_in_use(stats_.bytes_in_use);
  mas->set_peak_bytes_in_use(stats_.peak_bytes_in_use);
  mas->set_largest_alloc_size(stats_.largest_alloc_size);

  // Record summary data for every bin.
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    DCHECK_EQ(b->free_chunks.size(),
              bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);
    BinSummary* bs = md.add_bin_summary();
    bs->set_bin(bin_num);
    bs->set_total_bytes_in_use(bin_info.total_bytes_in_use);
    bs->set_total_bytes_in_bin(bin_info.total_bytes_in_bin);
    bs->set_total_chunks_in_use(bin_info.total_chunks_in_use);
    bs->set_total_chunks_in_bin(bin_info.total_chunks_in_bin);
  }

  // Record state of every defined Chunk.
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      MemChunk* mc = md.add_chunk();
      mc->set_in_use(c->in_use());
      mc->set_address(reinterpret_cast<uint64>(c->ptr));
      mc->set_size(c->size);
      mc->set_requested_size(c->requested_size);
      mc->set_bin(c->bin_num);
#ifdef TENSORFLOW_MEM_DEBUG
      mc->set_op_name(c->op_name ? string(c->op_name) : "UNKNOWN");
      mc->set_step_id(c->step_id);
      mc->set_action_count(c->action_count);
#endif
      if (timing_counter_) {
        mc->set_freed_at_count(c->in_use() ? 0 : c->freed_at_count);
      }
      h = c->next;
    }
  }

  mas->set_fragmentation_metric(GetFragmentation());

#ifdef TENSORFLOW_MEM_DEBUG
  // Record the recent size history
  int history_len = static_cast<int>(std::min(
      action_counter_, static_cast<long long>(MEM_DEBUG_SIZE_HISTORY_SIZE)));
  for (int i = action_counter_ - history_len; i < action_counter_; ++i) {
    SnapShot* ss = md.add_snap_shot();
    ss->set_action_count(i);
    int slot = i % MEM_DEBUG_SIZE_HISTORY_SIZE;
    ss->set_size(size_history_[slot]);
  }
#endif

  return md;
}

absl::optional<AllocatorStats> BFCAllocator::GetStats() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_40(mht_40_v, 1491, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::GetStats");

  mutex_lock l(lock_);
  return stats_;
}

bool BFCAllocator::ClearStats() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_41(mht_41_v, 1499, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::ClearStats");

  mutex_lock l(lock_);
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
  return true;
}

std::array<BFCAllocator::BinDebugInfo, BFCAllocator::kNumBins>
BFCAllocator::get_bin_debug_info() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_42(mht_42_v, 1511, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::get_bin_debug_info");

  std::array<BinDebugInfo, kNumBins> bin_infos;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;
      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        CHECK_EQ(bin->free_chunks.count(h), 1);
        CHECK_EQ(c->bin_num, bin_num);
      }
      h = c->next;
    }
  }
  return bin_infos;
}

AllocatorMemoryType BFCAllocator::GetMemoryType() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbfc_allocatorDTcc mht_43(mht_43_v, 1539, "", "./tensorflow/core/common_runtime/bfc_allocator.cc", "BFCAllocator::GetMemoryType");

  return sub_allocator_->GetMemoryType();
}

}  // namespace tensorflow
