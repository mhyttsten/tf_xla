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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh() {
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


// Simple LRU pool allocators for various flavors of CPU RAM.

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Interface of an object that rounds up integers.
class RoundUpInterface {
 public:
  virtual ~RoundUpInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_0(mht_0_v, 207, "", "./tensorflow/core/common_runtime/pool_allocator.h", "~RoundUpInterface");
}
  virtual size_t RoundUp(size_t num_bytes) = 0;
};

// Size-limited pool of memory buffers obtained from a SubAllocator
// instance.  Pool eviction policy is LRU.
class PoolAllocator : public Allocator {
 public:
  // "pool_size_limit" is the maximum number of returned, re-usable
  // memory buffers to keep in the pool.  If pool_size_limit == 0, the
  // pool is effectively a thin wrapper around the allocator.
  // If "auto_resize" is true, then the pool_size_limit will gradually
  // be raised so that deallocations happen very rarely, if at all.
  // Transitory start-up objects may deallocate, but the long-term
  // working-set should not. Auto-resizing can raise pool_size_limit
  // but will never lower it.
  // "allocator" is the object that performs the underlying memory
  // malloc/free operations.  This object takes ownership of allocator.
  PoolAllocator(size_t pool_size_limit, bool auto_resize,
                SubAllocator* allocator, RoundUpInterface* size_rounder,
                string name);
  ~PoolAllocator() override;

  string Name() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_1(mht_1_v, 233, "", "./tensorflow/core/common_runtime/pool_allocator.h", "Name");
 return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  void DeallocateRaw(void* ptr) override;

  // Allocate an unused memory region of size "num_bytes".  Fetch from
  // the pool if available, otherwise call allocator_.
  void* Get(size_t num_bytes);

  // Return a no-longer needed memory region to the pool.  It is an error
  // to deference "ptr" after this call.  If the pool is full, the least
  // recently used region will be deallocated.
  void Put(void* ptr, size_t num_bytes);

  // Reset the pool to empty.
  void Clear();

  // The following accessors permit monitoring the effectiveness of
  // the pool at avoiding repeated malloc/frees on the underlying
  // allocator.  Read locks are not taken on the theory that value
  // consistency with other threads is not important.

  // Number of Get() requests satisfied from pool.
  int64_t get_from_pool_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return get_from_pool_count_;
  }
  // Number of Put() requests.
  int64_t put_count() const TF_NO_THREAD_SAFETY_ANALYSIS { return put_count_; }
  // Number of Get() requests requiring a fresh allocation.
  int64_t allocated_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return allocated_count_;
  }
  // Number of pool evictions.
  int64_t evicted_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return evicted_count_;
  }
  // Current size limit.
  size_t size_limit() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return pool_size_limit_;
  }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_2(mht_2_v, 278, "", "./tensorflow/core/common_runtime/pool_allocator.h", "GetMemoryType");

    return allocator_->GetMemoryType();
  }

 private:
  struct PtrRecord {
    void* ptr;
    size_t num_bytes;
    PtrRecord* prev;
    PtrRecord* next;
  };

  // Remove "pr" from the double-linked LRU list.
  void RemoveFromList(PtrRecord* pr) TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Add "pr" to the head of the double-linked LRU list.
  void AddToList(PtrRecord* pr) TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Delete the least recently used record.
  void EvictOne() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const string name_;
  const bool has_size_limit_;
  const bool auto_resize_;
  size_t pool_size_limit_;
  std::unique_ptr<SubAllocator> allocator_;
  std::unique_ptr<RoundUpInterface> size_rounder_;
  mutex mutex_;
  std::multimap<const size_t, PtrRecord*> pool_ TF_GUARDED_BY(mutex_);
  PtrRecord* lru_head_ TF_GUARDED_BY(mutex_) = nullptr;
  PtrRecord* lru_tail_ TF_GUARDED_BY(mutex_) = nullptr;
  int64_t get_from_pool_count_ TF_GUARDED_BY(mutex_) = 0;
  int64_t put_count_ TF_GUARDED_BY(mutex_) = 0;
  int64_t allocated_count_ TF_GUARDED_BY(mutex_) = 0;
  int64_t evicted_count_ TF_GUARDED_BY(mutex_) = 0;
};

// Do-nothing rounder. Passes through sizes unchanged.
class NoopRounder : public RoundUpInterface {
 public:
  size_t RoundUp(size_t num_bytes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_3(mht_3_v, 321, "", "./tensorflow/core/common_runtime/pool_allocator.h", "RoundUp");
 return num_bytes; }
};

// Power of 2 rounder: rounds up to nearest power of 2 size.
class Pow2Rounder : public RoundUpInterface {
 public:
  size_t RoundUp(size_t num_bytes) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_4(mht_4_v, 330, "", "./tensorflow/core/common_runtime/pool_allocator.h", "RoundUp");

    return 1uLL << Log2Ceiling64(num_bytes);
  }
};

class BasicCPUAllocator : public SubAllocator {
 public:
  BasicCPUAllocator(int numa_node, const std::vector<Visitor>& alloc_visitors,
                    const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors), numa_node_(numa_node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_5(mht_5_v, 342, "", "./tensorflow/core/common_runtime/pool_allocator.h", "BasicCPUAllocator");
}

  ~BasicCPUAllocator() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_6(mht_6_v, 347, "", "./tensorflow/core/common_runtime/pool_allocator.h", "~BasicCPUAllocator");
}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override;

  void Free(void* ptr, size_t num_bytes) override;

  bool SupportsCoalescing() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_7(mht_7_v, 357, "", "./tensorflow/core/common_runtime/pool_allocator.h", "SupportsCoalescing");
 return false; }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpool_allocatorDTh mht_8(mht_8_v, 362, "", "./tensorflow/core/common_runtime/pool_allocator.h", "GetMemoryType");

    return AllocatorMemoryType::kHostPageable;
  }

 private:
  int numa_node_;

  TF_DISALLOW_COPY_AND_ASSIGN(BasicCPUAllocator);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
