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
class MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc() {
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

#include <atomic>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

// If true, cpu allocator collects more stats.
static bool cpu_allocator_collect_stats = false;

void EnableCPUAllocatorStats() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "EnableCPUAllocatorStats");
 cpu_allocator_collect_stats = true; }
void DisableCPUAllocatorStats() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "DisableCPUAllocatorStats");
 cpu_allocator_collect_stats = false; }
bool CPUAllocatorStatsEnabled() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_2(mht_2_v, 211, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "CPUAllocatorStatsEnabled");
 return cpu_allocator_collect_stats; }

static const int kMaxTotalAllocationWarnings = 1;

static const int kMaxSingleAllocationWarnings = 5;

// If cpu_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_3(mht_3_v, 228, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "LargeAllocationWarningBytes");

  static int64_t value = static_cast<int64_t>(port::AvailableRam() *
                                              kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "TotalAllocationWarningBytes");

  static int64_t value = static_cast<int64_t>(port::AvailableRam() *
                                              kTotalAllocationWarningThreshold);
  return value;
}

namespace {

// A default Allocator for CPU devices.  ProcessState::GetCPUAllocator() will
// return a different version that may perform better, but may also lack the
// optional stats triggered by the functions above.  TODO(tucker): migrate all
// uses of cpu_allocator() except tests to use ProcessState instead.
class CPUAllocator : public Allocator {
 public:
  CPUAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "CPUAllocator");
}

  ~CPUAllocator() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "~CPUAllocator");
}

  string Name() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "Name");
 return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_8(mht_8_v, 271, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "AllocateRaw");

    if (num_bytes > static_cast<size_t>(LargeAllocationWarningBytes()) &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of free system memory.";
    }

    void* p = port::AlignedMalloc(num_bytes, alignment);
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64_t>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64_t>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of free system memory";
      }
      if (p != nullptr) {
        AddTraceMe("MemoryAllocation", p, num_bytes, alloc_size);
      }
    }
    return p;
  }

  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "DeallocateRaw");

    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size =
          port::MallocExtension_GetAllocatedSize(ptr);
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
      AddTraceMe("MemoryDeallocation", ptr, 0, alloc_size);
    }
    port::AlignedFree(ptr);
  }

  void AddTraceMe(absl::string_view traceme_name, const void* chunk_ptr,
                  std::size_t req_bytes, std::size_t alloc_bytes) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("traceme_name: \"" + std::string(traceme_name.data(), traceme_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_10(mht_10_v, 324, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "AddTraceMe");

    tensorflow::profiler::TraceMe::InstantActivity(
        [this, traceme_name, chunk_ptr, req_bytes,
         alloc_bytes]() TF_NO_THREAD_SAFETY_ANALYSIS {
          const auto& annotation =
              profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
          return tensorflow::profiler::TraceMeEncode(
              traceme_name, {{"allocator_name", Name()},
                             {"bytes_reserved", stats_.bytes_reserved},
                             {"bytes_allocated", stats_.bytes_in_use},
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

  absl::optional<AllocatorStats> GetStats() override {
    if (!cpu_allocator_collect_stats) return absl::nullopt;
    mutex_lock l(mu_);
    return stats_;
  }

  bool ClearStats() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_11(mht_11_v, 356, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "ClearStats");

    if (!cpu_allocator_collect_stats) return false;
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
    return true;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_12(mht_12_v, 368, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "AllocatedSizeSlow");

    return port::MallocExtension_GetAllocatedSize(ptr);
  }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_13(mht_13_v, 375, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "GetMemoryType");

    return AllocatorMemoryType::kHostPageable;
  }

 private:
  mutex mu_;
  AllocatorStats stats_ TF_GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
};

class CPUAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_14(mht_14_v, 396, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "CreateAllocator");
 return new CPUAllocator; }

  SubAllocator* CreateSubAllocator(int numa_node) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_15(mht_15_v, 401, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "CreateSubAllocator");

    return new CPUSubAllocator(new CPUAllocator);
  }

 private:
  class CPUSubAllocator : public SubAllocator {
   public:
    explicit CPUSubAllocator(CPUAllocator* cpu_allocator)
        : SubAllocator({}, {}), cpu_allocator_(cpu_allocator) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_16(mht_16_v, 412, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "CPUSubAllocator");
}

    void* Alloc(size_t alignment, size_t num_bytes,
                size_t* bytes_received) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_17(mht_17_v, 418, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "Alloc");

      *bytes_received = num_bytes;
      return cpu_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_18(mht_18_v, 426, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "Free");

      cpu_allocator_->DeallocateRaw(ptr);
    }

    bool SupportsCoalescing() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_19(mht_19_v, 433, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "SupportsCoalescing");
 return false; }

    AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScpu_allocator_implDTcc mht_20(mht_20_v, 438, "", "./tensorflow/core/framework/cpu_allocator_impl.cc", "GetMemoryType");

      return cpu_allocator_->GetMemoryType();
    }

   private:
    CPUAllocator* cpu_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
}  // namespace

}  // namespace tensorflow
