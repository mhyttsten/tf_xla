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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"

#include <cstring>
#include <unordered_map>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_simple_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

/*static*/ PluggableDeviceProcessState* PluggableDeviceProcessState::singleton(
    const string& device_type, const string& platform_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_type: \"" + device_type + "\"");
   mht_0_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.cc", "PluggableDeviceProcessState::singleton");

  using ProcessStateMap =
      std::unordered_map<string, PluggableDeviceProcessState*>;
  static ProcessStateMap* process_state_map = new ProcessStateMap;
  auto iter = process_state_map->find(platform_name);
  if (iter != process_state_map->end()) {
    return iter->second;
  }
  (*process_state_map)[platform_name] =
      new PluggableDeviceProcessState(device_type, platform_name);
  return (*process_state_map)[platform_name];
}

PluggableDeviceProcessState::PluggableDeviceProcessState(
    const string& device_type, const string& platform_name)
    : pluggable_device_enabled_(false),
      device_type_(device_type),
      platform_name_(platform_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device_type: \"" + device_type + "\"");
   mht_1_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.cc", "PluggableDeviceProcessState::PluggableDeviceProcessState");

  process_state_ = ProcessState::singleton();
}

int PluggableDeviceProcessState::BusIdForPluggableDevice(
    TfDeviceId tf_device_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.cc", "PluggableDeviceProcessState::BusIdForPluggableDevice");

  // Return the NUMA node associated with the PluggableDevice's StreamExecutor.
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  se::StreamExecutor* se = DeviceIdUtil::ExecutorForTfDeviceId(
                               DeviceType(device_type_), platform, tf_device_id)
                               .ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // `bus_id` must be non-negative. If the `numa_node` is unknown, use 0.
  return numa_node >= 0 ? numa_node : 0;
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.cc", "PluggableDeviceProcessState::GetPluggableDeviceAllocator");

  DCHECK(process_state_);
  const string& allocator_type = options.allocator_type();
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(DeviceType(device_type_), platform,
                                     tf_device_id);

  if (tf_device_id.value() >=
      static_cast<int64_t>(pluggable_device_allocators_.size())) {
    pluggable_device_allocators_.resize(tf_device_id.value() + 1);
  }

  AllocatorParts& allocator_parts =
      pluggable_device_allocators_[tf_device_id.value()];
  if (allocator_parts.allocator == nullptr) {
    if (!allocator_type.empty()) {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType(device_type_), tf_device_id, &platform_device_id));

    int bus_id = BusIdForPluggableDevice(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= pluggable_device_visitors_.size()) {
      pluggable_device_visitors_.push_back({});
    }

    bool use_unified_memory = options.per_process_gpu_memory_fraction() > 1.0 ||
                              options.experimental().use_unified_memory();
    DeviceMemAllocator* sub_allocator = new DeviceMemAllocator(
        DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_device_id)
            .ValueOrDie(),
        platform_device_id, use_unified_memory,
        pluggable_device_visitors_[bus_id], {});

    Allocator* device_allocator = nullptr;
    auto cplatform = dynamic_cast<se::CPlatform*>(platform);
    if (cplatform == nullptr) {
      LOG(FATAL) << "PluggableDevice's platform must be of type "  // Crash OK
                 << "stream_executor::CPlatform";
    }
    if (cplatform->UseBfcAllocator()) {
      device_allocator = new PluggableDeviceBFCAllocator(
          sub_allocator, total_bytes, options,
          strings::StrCat("PluggableDevice_", tf_device_id.value(), "_bfc"),
          cplatform->ForceMemoryGrowth());
    } else {
      device_allocator = new PluggableDeviceSimpleAllocator(sub_allocator);
    }

    allocator_parts = {std::unique_ptr<Allocator>(device_allocator),
                       device_allocator, sub_allocator};
  }
  return allocator_parts.allocator.get();
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceHostAllocator(
    int numa_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_process_stateDTcc mht_4(mht_4_v, 326, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.cc", "PluggableDeviceProcessState::GetPluggableDeviceHostAllocator");

  DCHECK(process_state_);
  if (!HasPluggableDevice()) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where
    // pluggable_device_host_allocators_ have already been populated and since
    // we're only reading these vectors, we can get by with a shared lock. In
    // the slower case, we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);
    if (static_cast<int>(pluggable_device_host_allocators_.size()) >
        numa_node) {
      return pluggable_device_host_allocators_[0].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request PluggableDevice host memory
  // through, since any will work.
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(pluggable_device_allocators_.size());
       ++i) {
    if (pluggable_device_allocators_[i].allocator != nullptr) {
      se = DeviceIdUtil::ExecutorForTfDeviceId(DeviceType(device_type_),
                                               platform, TfDeviceId(i))
               .ValueOrDie();
      break;
    }
  }

  DCHECK_NE(nullptr, se);

  while (static_cast<int>(pluggable_device_host_allocators_.size()) <=
         numa_node) {
    while (pluggable_device_host_alloc_visitors_.size() <= numa_node) {
      pluggable_device_host_alloc_visitors_.push_back({});
    }
    while (pluggable_device_host_free_visitors_.size() <= numa_node) {
      pluggable_device_host_free_visitors_.push_back({});
    }
    SubAllocator* sub_allocator = new DeviceHostAllocator(
        se, numa_node, pluggable_device_host_alloc_visitors_[numa_node],
        pluggable_device_host_free_visitors_[numa_node]);
    int64_t pluggable_device_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_GPU_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &pluggable_device_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetPluggableDeviceHostAllocator: "
                 << status.error_message();
    }
    int64_t pluggable_device_host_mem_limit =
        pluggable_device_host_mem_limit_in_mb << 20;

    BFCAllocator::Options allocator_opts;
    allocator_opts.allow_growth = true;
    Allocator* allocator = new BFCAllocator(
        absl::WrapUnique(sub_allocator), pluggable_device_host_mem_limit,
        /*name=*/"pluggable_device_host_bfc", allocator_opts);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    pluggable_device_host_allocators_.push_back(
        {std::unique_ptr<Allocator>(allocator), nullptr /*bfc_allocator*/,
         sub_allocator});
  }
  return pluggable_device_host_allocators_[0].allocator.get();
}

}  // namespace tensorflow
