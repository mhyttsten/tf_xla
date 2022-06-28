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

// A simple CPU allocator that intercepts malloc/free calls from MKL library
// and redirects them to Tensorflow allocator

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh() {
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


#ifdef INTEL_MKL

#include <cstdlib>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#ifdef _WIN32
typedef unsigned int uint;
#endif

namespace tensorflow {

static bool mkl_small_allocator_collect_stats = false;

class MklSubAllocator : public BasicCPUAllocator {
 public:
  MklSubAllocator() : BasicCPUAllocator(port::kNUMANoAffinity, {}, {}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "MklSubAllocator");
}
  ~MklSubAllocator() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "~MklSubAllocator");
}
};

// CPU allocator that handles small-size allocations by calling
// suballocator directly. Mostly, it is just a wrapper around a suballocator
// (that calls malloc and free directly) with support for bookkeeping.
class MklSmallSizeAllocator : public Allocator {
 public:
  MklSmallSizeAllocator(SubAllocator* sub_allocator, size_t total_memory,
                        const string& name)
      : sub_allocator_(sub_allocator), name_(name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "MklSmallSizeAllocator");

    stats_.bytes_limit = total_memory;
  }
  ~MklSmallSizeAllocator() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_3(mht_3_v, 237, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "~MklSmallSizeAllocator");
}

  TF_DISALLOW_COPY_AND_ASSIGN(MklSmallSizeAllocator);

  inline string Name() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_4(mht_4_v, 244, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "Name");
 return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_5(mht_5_v, 249, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "AllocateRaw");

    void* ptr = port::AlignedMalloc(num_bytes, alignment);
    if (mkl_small_allocator_collect_stats) IncrementStats(num_bytes);
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_6(mht_6_v, 258, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "DeallocateRaw");

    if (ptr == nullptr) {
      LOG(ERROR) << "tried to deallocate nullptr";
      return;
    }

    if (mkl_small_allocator_collect_stats) {
      const size_t alloc_size = port::MallocExtension_GetAllocatedSize(ptr);
      DecrementStats(alloc_size);
    }
    port::AlignedFree(ptr);
  }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mutex_);
    return stats_;
  }

  bool ClearStats() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_7(mht_7_v, 279, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "ClearStats");

    mutex_lock l(mutex_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = 0;
    stats_.largest_alloc_size = 0;
    stats_.bytes_in_use = 0;
    stats_.bytes_limit = 0;
    return true;
  }

 private:
  // Increment statistics for the allocator handling small allocations.
  inline void IncrementStats(size_t alloc_size) TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    ++stats_.num_allocs;
    stats_.bytes_in_use += alloc_size;
    stats_.peak_bytes_in_use =
        std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    stats_.largest_alloc_size =
        std::max(alloc_size, static_cast<size_t>(stats_.largest_alloc_size));
  }

  // Decrement statistics for the allocator handling small allocations.
  inline void DecrementStats(size_t dealloc_size) TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    stats_.bytes_in_use -= dealloc_size;
  }

  SubAllocator* sub_allocator_;  // Not owned by this class.

  // Mutex for protecting updates to map of allocations.
  mutable mutex mutex_;

  // Allocator name
  string name_;

  // Allocator stats for small allocs
  AllocatorStats stats_ TF_GUARDED_BY(mutex_);
};

/// CPU allocator for MKL that wraps BFC allocator and intercepts
/// and redirects memory allocation calls from MKL.
class MklCPUAllocator : public Allocator {
 public:
  // Constructor and other standard functions

  /// Environment variable that user can set to upper bound on memory allocation
  static constexpr const char* kMaxLimitStr = "TF_MKL_ALLOC_MAX_BYTES";

  /// Default upper limit on allocator size - 64GB
  static constexpr size_t kDefaultMaxLimit = 64LL << 30;

  MklCPUAllocator() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_8(mht_8_v, 334, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "MklCPUAllocator");
 TF_CHECK_OK(Initialize()); }

  ~MklCPUAllocator() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_9(mht_9_v, 339, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "~MklCPUAllocator");

    delete small_size_allocator_;
    delete large_size_allocator_;
  }

  Status Initialize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_10(mht_10_v, 347, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "Initialize");

    VLOG(2) << "MklCPUAllocator: In MklCPUAllocator";

    // Set upper bound on memory allocation to physical RAM available on the
    // CPU unless explicitly specified by user
    uint64 max_mem_bytes = kDefaultMaxLimit;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    max_mem_bytes =
        (uint64)sysconf(_SC_PHYS_PAGES) * (uint64)sysconf(_SC_PAGESIZE);
#endif
    char* user_mem_bytes = getenv(kMaxLimitStr);

    if (user_mem_bytes != NULL) {
      uint64 user_val = 0;
      if (!strings::safe_strtou64(user_mem_bytes, &user_val)) {
        return errors::InvalidArgument("Invalid memory limit (", user_mem_bytes,
                                       ") specified for MKL allocator through ",
                                       kMaxLimitStr);
      }
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
      if (user_val > max_mem_bytes) {
        LOG(WARNING) << "The user specified a memory limit " << kMaxLimitStr
                     << "=" << user_val
                     << " greater than available physical memory: "
                     << max_mem_bytes
                     << ". This could significantly reduce performance!";
      }
#endif
      max_mem_bytes = user_val;
    }

    VLOG(1) << "MklCPUAllocator: Setting max_mem_bytes: " << max_mem_bytes;

    sub_allocator_ = new MklSubAllocator();

    // SubAllocator is owned by BFCAllocator, so we do not need to deallocate
    // it in MklSmallSizeAllocator.
    small_size_allocator_ =
        new MklSmallSizeAllocator(sub_allocator_, max_mem_bytes, kName);

    BFCAllocator::Options large_allocator_opts;
    large_allocator_opts.allow_growth = kAllowGrowth;
    large_size_allocator_ =
        new BFCAllocator(absl::WrapUnique(sub_allocator_), max_mem_bytes, kName,
                         large_allocator_opts);
    return Status::OK();
  }

  inline string Name() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_11(mht_11_v, 398, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "Name");
 return kName; }
  inline bool IsSmallSizeAllocation(const void* ptr) const
      TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    return large_allocations_map_.find(ptr) == large_allocations_map_.end();
  }
  // AddLargeAllocMap and RemoveLargeAllocMap are always called with a lock held
  inline void AddLargeAllocMap(void* ptr, size_t num_bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_12(mht_12_v, 409, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "AddLargeAllocMap");

    if (ptr != nullptr) {
      std::pair<void*, size_t> map_val(ptr, num_bytes);
      large_allocations_map_.insert(map_val);
    }
  }
  inline void RemoveLargeAllocMap(void* ptr)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_13(mht_13_v, 419, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "RemoveLargeAllocMap");

    auto map_iter = large_allocations_map_.find(ptr);
    if (map_iter != large_allocations_map_.end()) {
      large_allocations_map_.erase(map_iter);
    } else {
      LOG(ERROR) << "tried to deallocate invalid pointer";
    }
    return;
  }

  inline void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_14(mht_14_v, 432, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "AllocateRaw");

    // If the allocation size is less than threshold, call small allocator,
    // otherwise call large-size allocator (BFC). We found that BFC allocator
    // does not deliver good performance for small allocations when
    // inter_op_parallelism_threads is high.
    if (UseSystemAlloc() || num_bytes < kSmallAllocationsThreshold) {
      return small_size_allocator_->AllocateRaw(alignment, num_bytes);
    } else {
      mutex_lock l(mutex_);
      void* ptr = large_size_allocator_->AllocateRaw(alignment, num_bytes);
      AddLargeAllocMap(ptr, num_bytes);
      return ptr;
    }
  }
  inline void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_15(mht_15_v, 449, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "DeallocateRaw");

    // Check if ptr is for "small" allocation. If it is, then call Free
    // directly. Otherwise, call BFC to handle free.
    if (UseSystemAlloc() || IsSmallSizeAllocation(ptr)) {
      small_size_allocator_->DeallocateRaw(ptr);
    } else {
      mutex_lock l(mutex_);
      RemoveLargeAllocMap(ptr);
      large_size_allocator_->DeallocateRaw(ptr);
    }
  }
  absl::optional<AllocatorStats> GetStats() override {
    auto s_stats = small_size_allocator_->GetStats();
    auto l_stats = large_size_allocator_->GetStats();

    // Combine statistics from small-size and large-size allocator.
    mutex_lock l(mutex_);
    stats_.num_allocs = l_stats->num_allocs + s_stats->num_allocs;
    stats_.bytes_in_use = l_stats->bytes_in_use + s_stats->bytes_in_use;
    stats_.peak_bytes_in_use =
        l_stats->peak_bytes_in_use + s_stats->peak_bytes_in_use;

    // Since small-size allocations go to MklSmallSizeAllocator,
    // max_alloc_size from large_size_allocator would be the maximum
    // size allocated by MklCPUAllocator.
    stats_.largest_alloc_size = l_stats->largest_alloc_size;
    stats_.bytes_limit = std::max(s_stats->bytes_limit, l_stats->bytes_limit);
    return stats_;
  }

  bool ClearStats() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_16(mht_16_v, 482, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "ClearStats");

    bool stats_cleared = small_size_allocator_->ClearStats();
    stats_cleared &= large_size_allocator_->ClearStats();
    return stats_cleared;
  }

 private:
  // Hooks provided by this allocator for memory allocation routines from MKL
  static inline void* MallocHook(size_t size) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_17(mht_17_v, 493, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "MallocHook");

    VLOG(3) << "MklCPUAllocator: In MallocHook";
    return cpu_allocator()->AllocateRaw(kAlignment, size);
  }

  static inline void FreeHook(void* ptr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_18(mht_18_v, 501, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "FreeHook");

    VLOG(3) << "MklCPUAllocator: In FreeHook";
    cpu_allocator()->DeallocateRaw(ptr);
  }

  static inline void* CallocHook(size_t num, size_t size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_19(mht_19_v, 509, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "CallocHook");

    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
    return nullptr;  // return a value and make static code analyzers happy
  }

  static inline void* ReallocHook(void* ptr, size_t size) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_cpu_allocatorDTh mht_20(mht_20_v, 519, "", "./tensorflow/core/common_runtime/mkl_cpu_allocator.h", "ReallocHook");

    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
    return nullptr;  // return a value and make static code analyzers happy
  }

  // Do we allow growth in BFC Allocator
  static const bool kAllowGrowth = true;

  // Name
  static constexpr const char* kName = "mklcpu";

  // The alignment that we need for the allocations
  static constexpr const size_t kAlignment = 64;

  Allocator* large_size_allocator_;              // owned by this class
  MklSmallSizeAllocator* small_size_allocator_;  // owned by this class.

  SubAllocator* sub_allocator_;  // not owned by this class
  mutable mutex mutex_;
  AllocatorStats stats_ TF_GUARDED_BY(mutex_);

  // Hash map to keep track of "BFC" allocations
  // We do not use BFC allocator for small allocations.
  std::unordered_map<const void*, size_t> large_allocations_map_
      TF_GUARDED_BY(mutex_);

  // Size in bytes that defines the upper-bound for "small" allocations.
  // Any allocation below this threshold is "small" allocation.
  static constexpr const size_t kSmallAllocationsThreshold = 4096;

  // Prevent copying and assignment
  TF_DISALLOW_COPY_AND_ASSIGN(MklCPUAllocator);
};

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
