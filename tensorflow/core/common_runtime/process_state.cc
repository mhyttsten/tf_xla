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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc() {
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

#include "tensorflow/core/common_runtime/process_state.h"

#include <atomic>
#include <cstring>
#include <vector>

#include "absl/base/call_once.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

/*static*/ ProcessState* ProcessState::singleton() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::singleton");

  static ProcessState* instance = new ProcessState;
  static absl::once_flag f;
  absl::call_once(f, []() {
    AllocatorFactoryRegistry::singleton()->process_state_ = instance;
  });

  return instance;
}

ProcessState::ProcessState()
    : numa_enabled_(false), cpu_allocators_cached_(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::ProcessState");
}

string ProcessState::MemDesc::DebugString() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::MemDesc::DebugString");

  return strings::StrCat((loc == CPU ? "CPU " : "GPU "), dev_index,
                         ", dma: ", gpu_registered, ", nic: ", nic_registered);
}

ProcessState::MemDesc ProcessState::PtrType(const void* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_3(mht_3_v, 232, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::PtrType");

  if (FLAGS_brain_gpu_record_mem_types) {
    auto iter = mem_desc_map_.find(ptr);
    if (iter != mem_desc_map_.end()) {
      return iter->second;
    }
  }
  return MemDesc();
}

Allocator* ProcessState::GetCPUAllocator(int numa_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::GetCPUAllocator");

  if (!numa_enabled_ || numa_node == port::kNUMANoAffinity) numa_node = 0;

  // Check if allocator for the numa node is in lock-free cache.
  if (numa_node < cpu_allocators_cached_.load(std::memory_order_acquire)) {
    return cpu_allocators_cache_[numa_node];
  }

  mutex_lock lock(mu_);
  while (cpu_allocators_.size() <= static_cast<size_t>(numa_node)) {
    // If visitors have been defined we need an Allocator built from
    // a SubAllocator.  Prefer BFCAllocator, but fall back to PoolAllocator
    // depending on env var setting.
    const bool alloc_visitors_defined =
        (!cpu_alloc_visitors_.empty() || !cpu_free_visitors_.empty());
    bool use_bfc_allocator = false;
    Status status = ReadBoolFromEnvVar(
        "TF_CPU_ALLOCATOR_USE_BFC", alloc_visitors_defined, &use_bfc_allocator);
    if (!status.ok()) {
      LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
    }
    Allocator* allocator = nullptr;
    SubAllocator* sub_allocator =
        (numa_enabled_ || alloc_visitors_defined || use_bfc_allocator)
            ? new BasicCPUAllocator(
                  numa_enabled_ ? numa_node : port::kNUMANoAffinity,
                  cpu_alloc_visitors_, cpu_free_visitors_)
            : nullptr;
    if (use_bfc_allocator) {
      // TODO(reedwm): evaluate whether 64GB by default is the best choice.
      int64_t cpu_mem_limit_in_mb = -1;
      Status status = ReadInt64FromEnvVar("TF_CPU_BFC_MEM_LIMIT_IN_MB",
                                          1LL << 16 /*64GB max by default*/,
                                          &cpu_mem_limit_in_mb);
      if (!status.ok()) {
        LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
      }
      int64_t cpu_mem_limit = cpu_mem_limit_in_mb * (1LL << 20);
      DCHECK(sub_allocator);

      BFCAllocator::Options allocator_opts;
      allocator_opts.allow_growth = true;
      allocator = new BFCAllocator(
          absl::WrapUnique(sub_allocator), cpu_mem_limit,
          /*name=*/"bfc_cpu_allocator_for_gpu", allocator_opts);

      VLOG(2) << "Using BFCAllocator with memory limit of "
              << cpu_mem_limit_in_mb << " MB for ProcessState CPU allocator";
    } else if (sub_allocator) {
      DCHECK(sub_allocator);
      allocator =
          new PoolAllocator(/*pool_size_limit=*/100, /*auto_resize=*/true,
                            sub_allocator, new NoopRounder, "cpu_pool");
      VLOG(2) << "Using PoolAllocator for ProcessState CPU allocator "
              << "numa_enabled_=" << numa_enabled_
              << " numa_node=" << numa_node;
    } else {
      DCHECK(!sub_allocator);
      allocator = cpu_allocator_base();
    }
    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    cpu_allocators_.push_back(allocator);
    if (cpu_allocators_.size() < cpu_allocators_cache_.max_size()) {
      cpu_allocators_cache_[cpu_allocators_.size() - 1] = allocator;
      cpu_allocators_cached_.fetch_add(1, std::memory_order_release);
    }
    if (!sub_allocator) {
      DCHECK(cpu_alloc_visitors_.empty() && cpu_free_visitors_.empty());
    }
  }
  return cpu_allocators_[numa_node];
}

void ProcessState::AddCPUAllocVisitor(SubAllocator::Visitor visitor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_5(mht_5_v, 325, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::AddCPUAllocVisitor");

  VLOG(1) << "AddCPUAllocVisitor";
  mutex_lock lock(mu_);
  CHECK_EQ(0, cpu_allocators_.size())  // Crash OK
      << "AddCPUAllocVisitor must be called prior to first call to "
         "ProcessState::GetCPUAllocator";
  cpu_alloc_visitors_.push_back(std::move(visitor));
}

void ProcessState::AddCPUFreeVisitor(SubAllocator::Visitor visitor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_6(mht_6_v, 337, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::AddCPUFreeVisitor");

  mutex_lock lock(mu_);
  CHECK_EQ(0, cpu_allocators_.size())  // Crash OK
      << "AddCPUFreeVisitor must be called prior to first call to "
         "ProcessState::GetCPUAllocator";
  cpu_free_visitors_.push_back(std::move(visitor));
}

void ProcessState::TestOnlyReset() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_stateDTcc mht_7(mht_7_v, 348, "", "./tensorflow/core/common_runtime/process_state.cc", "ProcessState::TestOnlyReset");

  mutex_lock lock(mu_);
  // Don't delete this value because it's static.
  Allocator* default_cpu_allocator = cpu_allocator_base();
  mem_desc_map_.clear();
  for (Allocator* a : cpu_allocators_) {
    if (a != default_cpu_allocator) delete a;
  }
  cpu_allocators_.clear();
  for (Allocator* a : cpu_al_) {
    delete a;
  }
  cpu_al_.clear();
}

}  // namespace tensorflow
