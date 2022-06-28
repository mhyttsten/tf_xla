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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh() {
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


#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class PoolAllocator;

// Singleton that manages per-process state, e.g. allocation of
// shared resources.
class ProcessState : public ProcessStateInterface {
 public:
  static ProcessState* singleton();

  // Descriptor for memory allocation attributes, used by optional
  // runtime correctness analysis logic.
  struct MemDesc {
    enum MemLoc { CPU, GPU };
    MemLoc loc;
    int dev_index;
    bool gpu_registered;
    bool nic_registered;
    MemDesc()
        : loc(CPU),
          dev_index(0),
          gpu_registered(false),
          nic_registered(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_0(mht_0_v, 223, "", "./tensorflow/core/common_runtime/process_state.h", "MemDesc");
}
    string DebugString();
  };

  // If NUMA Allocators are desired, call this before calling any
  // Allocator accessor.
  void EnableNUMA() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/process_state.h", "EnableNUMA");
 numa_enabled_ = true; }

  // Returns what we know about the memory at ptr.
  // If we know nothing, it's called CPU 0 with no other attributes.
  MemDesc PtrType(const void* ptr);

  // Returns the one CPUAllocator used for the given numa_node.
  // Treats numa_node == kNUMANoAffinity as numa_node == 0.
  Allocator* GetCPUAllocator(int numa_node) override;

  // Registers alloc visitor for the CPU allocator(s).
  // REQUIRES: must be called before GetCPUAllocator.
  void AddCPUAllocVisitor(SubAllocator::Visitor v);

  // Registers free visitor for the CPU allocator(s).
  // REQUIRES: must be called before GetCPUAllocator.
  void AddCPUFreeVisitor(SubAllocator::Visitor v);

  typedef std::unordered_map<const void*, MemDesc> MDMap;

 protected:
  ProcessState();
  virtual ~ProcessState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_2(mht_2_v, 257, "", "./tensorflow/core/common_runtime/process_state.h", "~ProcessState");
}
  friend class GPUProcessState;
  friend class PluggableDeviceProcessState;

  // If these flags need to be runtime configurable consider adding
  // them to ConfigProto.
  static constexpr bool FLAGS_brain_mem_reg_gpu_dma = true;
  static constexpr bool FLAGS_brain_gpu_record_mem_types = false;

  // Helper method for unit tests to reset the ProcessState singleton by
  // cleaning up everything. Never use in production.
  void TestOnlyReset();

  static ProcessState* instance_;
  bool numa_enabled_;

  mutex mu_;

  // Indexed by numa_node.  If we want numa-specific allocators AND a
  // non-specific allocator, maybe should index by numa_node+1.
  std::vector<Allocator*> cpu_allocators_ TF_GUARDED_BY(mu_);
  std::vector<SubAllocator::Visitor> cpu_alloc_visitors_ TF_GUARDED_BY(mu_);
  std::vector<SubAllocator::Visitor> cpu_free_visitors_ TF_GUARDED_BY(mu_);

  // A cache of cpu allocators indexed by a numa node. Used as a fast path to
  // get CPU allocator by numa node id without locking the mutex. We can't use
  // `cpu_allocators_` storage in the lock-free path because concurrent
  // operation can deallocate the vector storage.
  std::atomic<int> cpu_allocators_cached_;
  std::array<Allocator*, 8> cpu_allocators_cache_;

  // Optional RecordingAllocators that wrap the corresponding
  // Allocators for runtime attribute use analysis.
  MDMap mem_desc_map_;
  std::vector<Allocator*> cpu_al_ TF_GUARDED_BY(mu_);
};

namespace internal {
class RecordingAllocator : public Allocator {
 public:
  RecordingAllocator(ProcessState::MDMap* mm, Allocator* a,
                     ProcessState::MemDesc md, mutex* mu)
      : mm_(mm), a_(a), md_(md), mu_(mu) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_3(mht_3_v, 302, "", "./tensorflow/core/common_runtime/process_state.h", "RecordingAllocator");
}

  string Name() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_4(mht_4_v, 307, "", "./tensorflow/core/common_runtime/process_state.h", "Name");
 return a_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_5(mht_5_v, 311, "", "./tensorflow/core/common_runtime/process_state.h", "AllocateRaw");

    void* p = a_->AllocateRaw(alignment, num_bytes);
    mutex_lock l(*mu_);
    (*mm_)[p] = md_;
    return p;
  }
  void DeallocateRaw(void* p) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_6(mht_6_v, 320, "", "./tensorflow/core/common_runtime/process_state.h", "DeallocateRaw");

    mutex_lock l(*mu_);
    auto iter = mm_->find(p);
    mm_->erase(iter);
    a_->DeallocateRaw(p);
  }
  bool TracksAllocationSizes() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_7(mht_7_v, 329, "", "./tensorflow/core/common_runtime/process_state.h", "TracksAllocationSizes");

    return a_->TracksAllocationSizes();
  }
  size_t RequestedSize(const void* p) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_8(mht_8_v, 335, "", "./tensorflow/core/common_runtime/process_state.h", "RequestedSize");

    return a_->RequestedSize(p);
  }
  size_t AllocatedSize(const void* p) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_9(mht_9_v, 341, "", "./tensorflow/core/common_runtime/process_state.h", "AllocatedSize");

    return a_->AllocatedSize(p);
  }
  absl::optional<AllocatorStats> GetStats() override { return a_->GetStats(); }
  bool ClearStats() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_10(mht_10_v, 348, "", "./tensorflow/core/common_runtime/process_state.h", "ClearStats");
 return a_->ClearStats(); }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTh mht_11(mht_11_v, 353, "", "./tensorflow/core/common_runtime/process_state.h", "GetMemoryType");

    return a_->GetMemoryType();
  }

  ProcessState::MDMap* mm_;  // not owned
  Allocator* a_;             // not owned
  ProcessState::MemDesc md_;
  mutex* mu_;
};
}  // namespace internal
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
