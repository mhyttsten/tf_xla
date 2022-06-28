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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc() {
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

#include "tensorflow/core/common_runtime/pool_allocator.h"

#include <errno.h>
#ifndef _MSC_VER
#include <strings.h>
#include <sys/mman.h>  // for munmap
#endif

#include <map>
#include <utility>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

PoolAllocator::PoolAllocator(size_t pool_size_limit, bool auto_resize,
                             SubAllocator* allocator,
                             RoundUpInterface* size_rounder, string name)
    : name_(std::move(name)),
      has_size_limit_(pool_size_limit > 0),
      auto_resize_(auto_resize),
      pool_size_limit_(pool_size_limit),
      allocator_(allocator),
      size_rounder_(size_rounder) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::PoolAllocator");

  if (auto_resize) {
    CHECK_LT(size_t{0}, pool_size_limit)
        << "size limit must be > 0 if auto_resize is true.";
  }
}

PoolAllocator::~PoolAllocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::~PoolAllocator");
 Clear(); }

namespace {
// Pools contain Chunks allocated from the underlying Allocator.
// Chunk alignment is always on kPoolAlignment boundaries.  Each Chunk
// begins with a descriptor (ChunkPrefix) that gives its size and a
// pointer to itself.  The pointer returned to the user is just past
// the ChunkPrefix.  If the user asks for a larger alignment, we will
// increase the size of the chunk, then adjust the returned user
// pointer and also re-write the ChunkPrefix.chunk_ptr value
// immediately before it.  This way the Chunk address and size can be
// recovered from the returned user pointer, regardless of alignment.
// Note that this dereferencing of the pointers means that we cannot
// handle GPU memory, only CPU memory.
struct ChunkPrefix {
  size_t num_bytes;
  void* chunk_ptr;
};
// kPoolAlignment cannot be less than the size of ChunkPrefix.
static const int kPoolAlignment = sizeof(ChunkPrefix);

void* PrepareChunk(void* chunk, size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PrepareChunk");

  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(chunk);
  cp->num_bytes = num_bytes;
  cp->chunk_ptr = chunk;
  void* user_ptr = reinterpret_cast<void*>(cp + 1);
  if (alignment > kPoolAlignment) {
    // Move user_ptr forward to the first satisfying offset, and write
    // chunk_ptr just before it.
    size_t aligned_ptr = reinterpret_cast<size_t>(user_ptr) + alignment;
    user_ptr = reinterpret_cast<void*>(aligned_ptr & ~(alignment - 1));
    (reinterpret_cast<ChunkPrefix*>(user_ptr) - 1)->chunk_ptr = chunk;
  }
  // Safety check that user_ptr is always past the ChunkPrefix.
  CHECK_GE(user_ptr, reinterpret_cast<ChunkPrefix*>(chunk) + 1);
  return user_ptr;
}

ChunkPrefix* FindPrefix(void* user_ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "FindPrefix");

  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(user_ptr) - 1;
  return reinterpret_cast<ChunkPrefix*>(cp->chunk_ptr);
}
}  // namespace

void* PoolAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::AllocateRaw");

  if (num_bytes == 0) return nullptr;

  // If alignment is larger than kPoolAlignment, increase num_bytes so that we
  // are guaranteed to be able to return an aligned ptr by advancing user_ptr
  // without overrunning the end of the chunk.
  if (alignment > kPoolAlignment) {
    num_bytes += alignment;
  }
  num_bytes += sizeof(ChunkPrefix);
  num_bytes = size_rounder_->RoundUp(num_bytes);
  PtrRecord* pr = nullptr;
  if (has_size_limit_) {
    {
      mutex_lock lock(mutex_);
      auto iter = pool_.find(num_bytes);
      if (iter == pool_.end()) {
        allocated_count_++;
        // Deliberately fall out of lock scope before
        // calling the allocator.  No further modification
        // to the pool will be performed.
      } else {
        get_from_pool_count_++;
        pr = iter->second;
        RemoveFromList(pr);
        pool_.erase(iter);
        // Fall out of lock scope and do the result without the lock held.
      }
    }
  }
  if (pr != nullptr) {
    void* r = pr->ptr;
    delete pr;
    return PrepareChunk(r, alignment, num_bytes);
  } else {
    size_t bytes_received;
    void* ptr = allocator_->Alloc(kPoolAlignment, num_bytes, &bytes_received);
    return PrepareChunk(ptr, alignment, bytes_received);
  }
}

void PoolAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_5(mht_5_v, 321, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::DeallocateRaw");

  if (ptr == nullptr) return;
  ChunkPrefix* cp = FindPrefix(ptr);
  CHECK_LE((void*)cp, (void*)ptr);
  if (!has_size_limit_ && !auto_resize_) {
    allocator_->Free(cp, cp->num_bytes);
  } else {
    mutex_lock lock(mutex_);
    ++put_count_;
    while (pool_.size() >= pool_size_limit_) {
      EvictOne();
    }
    PtrRecord* pr = new PtrRecord;
    pr->num_bytes = cp->num_bytes;
    pr->ptr = cp;
    AddToList(pr);
    pool_.insert(std::make_pair(cp->num_bytes, pr));
  }
}

void PoolAllocator::Clear() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_6(mht_6_v, 344, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::Clear");

  if (has_size_limit_) {
    mutex_lock lock(mutex_);
    for (auto iter : pool_) {
      PtrRecord* pr = iter.second;
      allocator_->Free(pr->ptr, pr->num_bytes);
      delete pr;
    }
    pool_.clear();
    get_from_pool_count_ = 0;
    put_count_ = 0;
    allocated_count_ = 0;
    evicted_count_ = 0;
    lru_head_ = nullptr;
    lru_tail_ = nullptr;
  }
}

void PoolAllocator::RemoveFromList(PtrRecord* pr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_7(mht_7_v, 365, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::RemoveFromList");

  if (pr->prev == nullptr) {
    DCHECK_EQ(lru_head_, pr);
    lru_head_ = nullptr;
  } else {
    pr->prev->next = pr->next;
  }
  if (pr->next == nullptr) {
    DCHECK_EQ(lru_tail_, pr);
    lru_tail_ = pr->prev;
  } else {
    pr->next->prev = pr->prev;
    if (lru_head_ == nullptr) {
      lru_head_ = pr->next;
    }
  }
}

void PoolAllocator::AddToList(PtrRecord* pr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_8(mht_8_v, 386, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::AddToList");

  pr->prev = nullptr;
  if (lru_head_ == nullptr) {
    CHECK(lru_tail_ == nullptr);
    lru_tail_ = pr;
    pr->next = nullptr;
  } else {
    pr->next = lru_head_;
    pr->next->prev = pr;
  }
  lru_head_ = pr;
}

void PoolAllocator::EvictOne() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_9(mht_9_v, 402, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "PoolAllocator::EvictOne");

  DCHECK(lru_tail_ != nullptr);
  PtrRecord* prec = lru_tail_;
  RemoveFromList(prec);
  auto iter = pool_.find(prec->num_bytes);
  while (iter->second != prec) {
    ++iter;
    DCHECK(iter != pool_.end());
  }
  pool_.erase(iter);
  allocator_->Free(prec->ptr, prec->num_bytes);
  delete prec;
  ++evicted_count_;
  // Auto-resizing, and warning messages.
  static const double kTolerable = 2e-3;
  static const int kCheckInterval = 1000;
  static const double kIncreaseFactor = 1.1;
  static const int kMinPoolSize = 100;
  if (0 == evicted_count_ % kCheckInterval) {
    const double eviction_rate =
        evicted_count_ / static_cast<double>(put_count_);
    const int64_t alloc_request_count = allocated_count_ + get_from_pool_count_;
    const double alloc_rate =
        (alloc_request_count == 0)
            ? 0.0
            : allocated_count_ / static_cast<double>(alloc_request_count);
    // Can turn on for debugging purposes.
    const bool kShouldLog = false;
    if (kShouldLog) {
      LOG(INFO) << "PoolAllocator: After " << alloc_request_count
                << " get requests, put_count=" << put_count_
                << " evicted_count=" << evicted_count_
                << " eviction_rate=" << eviction_rate
                << " and unsatisfied allocation rate=" << alloc_rate;
    }
    if (auto_resize_ && (eviction_rate > kTolerable) &&
        (alloc_rate > kTolerable)) {
      size_t new_size_limit = (pool_size_limit_ < kMinPoolSize)
                                  ? kMinPoolSize
                                  : (kIncreaseFactor * pool_size_limit_);
      if (kShouldLog) {
        LOG(INFO) << "Raising pool_size_limit_ from " << pool_size_limit_
                  << " to " << new_size_limit;
      }
      pool_size_limit_ = new_size_limit;
      // Reset all the counters so that ratios are relative to new sizes
      // at next test interval.
      put_count_ = 0;
      allocated_count_ = 0;
      evicted_count_ = 0;
      get_from_pool_count_ = 0;
    }
  }
}

void* BasicCPUAllocator::Alloc(size_t alignment, size_t num_bytes,
                               size_t* bytes_received) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_10(mht_10_v, 461, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "BasicCPUAllocator::Alloc");

  void* ptr = nullptr;
  *bytes_received = num_bytes;
  if (num_bytes > 0) {
    if (numa_node_ == port::kNUMANoAffinity) {
      ptr = port::AlignedMalloc(num_bytes, static_cast<int>(alignment));
    } else {
      ptr =
          port::NUMAMalloc(numa_node_, num_bytes, static_cast<int>(alignment));
    }
    VisitAlloc(ptr, numa_node_, num_bytes);
  }
  return ptr;
}

void BasicCPUAllocator::Free(void* ptr, size_t num_bytes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTcc mht_11(mht_11_v, 479, "", "./tensorflow/core/common_runtime/pool_allocator.cc", "BasicCPUAllocator::Free");

  if (num_bytes > 0) {
    VisitFree(ptr, numa_node_, num_bytes);
    if (numa_node_ == port::kNUMANoAffinity) {
      port::AlignedFree(ptr);
    } else {
      port::NUMAFree(ptr, num_bytes);
    }
  }
}
}  // namespace tensorflow
