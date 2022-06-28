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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc() {
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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"

#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {
namespace {

int64_t MinSystemMemory(int64_t available_memory) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "MinSystemMemory");

  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 225MiB to system memory,
  // Otherwise, allocate max(300MiB, kMinSystemMemoryFraction *
  // available_memory) to system memory.
  //
  // In the future we could be more sophisticated by using a table of devices.
  int64_t min_system_memory;
  constexpr float kMinSystemMemoryFraction = 0.06;
  if (available_memory < (1LL << 31)) {
    // 225MiB
    min_system_memory = 255 * 1024 * 1024;
  } else {
    // max(300 MiB, kMinSystemMemoryFraction * available_memory)
    min_system_memory = std::max(
        int64_t{314572800},
        static_cast<int64_t>(available_memory * kMinSystemMemoryFraction));
  }
#if defined(__GNUC__) && defined(__OPTIMIZE__)
// Do nothing
#elif !defined(__GNUC__) && defined(NDEBUG)
// Do nothing
#else
  // Double the amount of available PluggableDevice memory in non-opt builds
  // (debug builds in windows); because in non-opt builds more system memory is
  // necessary.
  min_system_memory *= 2;
#endif
  VLOG(5) << "available_memory = " << available_memory;
  VLOG(5) << "min_system_memory = " << min_system_memory;
  return min_system_memory;
}

// Get the memory limit for the virtual device being created on PluggableDevice
// with 'platform_device_id', when that virtual device is the only
// virtual device being created on that PluggableDevice.
Status SingleVirtualDeviceMemoryLimit(const string& platform_name,
                                      const GPUOptions& device_options,
                                      PlatformDeviceId platform_device_id,
                                      int64_t* memory_limit) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_1(mht_1_v, 253, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "SingleVirtualDeviceMemoryLimit");

  int64_t total_memory = 0;
  int64_t available_memory = 0;
  se::Platform* platform = PluggableDeviceMachineManager(platform_name);
  se::StreamExecutor* se =
      DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_device_id)
          .ValueOrDie();
  if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
    return errors::Unknown(
        "Failed to query available memory for PluggableDevice ",
        platform_device_id.value());
  }

  int64_t allocated_memory = 0;
  const double per_process_device_memory_fraction =
      device_options.per_process_gpu_memory_fraction();
  if (per_process_device_memory_fraction > 1.0 ||
      device_options.experimental().use_unified_memory()) {
    return errors::Internal("Unified memory is not supported yet.");
  }

  if (per_process_device_memory_fraction == 0) {
    allocated_memory = available_memory;
    const int64_t min_system_memory = MinSystemMemory(available_memory);
    if (min_system_memory < allocated_memory) {
      allocated_memory -= min_system_memory;
    }
  } else {
    allocated_memory = total_memory * per_process_device_memory_fraction;
  }
  *memory_limit = allocated_memory;
  return Status::OK();
}
}  // namespace

PluggableDeviceFactory::PluggableDeviceFactory(const string& device_type,
                                               const string& platform_name)
    : device_type_(device_type), platform_name_(platform_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("device_type: \"" + device_type + "\"");
   mht_2_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_2(mht_2_v, 295, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::PluggableDeviceFactory");
}

Status PluggableDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::ListPhysicalDevices");

  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);

  int device_count = platform->VisibleDeviceCount();
  for (int i = 0; i < device_count; ++i) {
    const string device_name =
        strings::StrCat("/physical_device:", device_type_, ":", i);
    devices->push_back(device_name);
  }

  return Status::OK();
}

Status PluggableDeviceFactory::GetDeviceDetails(
    int device_index, std::unordered_map<string, string>* details) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_4(mht_4_v, 319, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::GetDeviceDetails");

  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return Status::OK();
  }

  int device_count = platform->VisibleDeviceCount();
  if (device_index < 0 || device_index >= device_count) {
    return errors::Internal("Invalid device index: ", device_index);
  }

  auto desc_status = platform->DescriptionForDevice(device_index);
  if (!desc_status.ok()) {
    return desc_status.status();
  }

  auto desc = desc_status.ConsumeValueOrDie();
  (*details)["device_name"] = desc->name();
  return Status::OK();
}

Status PluggableDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_5(mht_5_v, 347, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::CreateDevices");

  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return Status::OK();
  }

  if (platform->VisibleDeviceCount() <= 0) {
    return Status::OK();
  }

  size_t num_tf_devices = INT_MAX;
  auto iter = options.config.device_count().find(device_type_);
  if (iter != options.config.device_count().end()) {
    num_tf_devices = iter->second;
  }
  const auto& device_options = options.config.gpu_options();
  std::vector<PlatformDeviceId> visible_device_order;

  if (num_tf_devices > 0) {
    TF_RETURN_IF_ERROR(DeviceIdUtil::ParseVisibleDeviceList(
        device_options.visible_device_list(), platform->VisibleDeviceCount(),
        &visible_device_order));
  }
  if (num_tf_devices > visible_device_order.size()) {
    num_tf_devices = visible_device_order.size();
  }

  const auto& virtual_devices = device_options.experimental().virtual_devices();
  if (!virtual_devices.empty())
    VLOG(2) << "Pluggable device does not support virtual device setting yet";
  std::vector<int64_t> memory_limit_bytes;
  for (int i = 0; i < num_tf_devices; ++i) {
    const PlatformDeviceId platform_device_id = visible_device_order[i];
    int64_t single_virtual_device_memory_limit = 0;
    TF_RETURN_IF_ERROR(SingleVirtualDeviceMemoryLimit(
        platform_name_, device_options, platform_device_id,
        &single_virtual_device_memory_limit));
    memory_limit_bytes.push_back(single_virtual_device_memory_limit);
    TfDeviceId tf_device_id(i);
    TF_RETURN_IF_ERROR(DeviceIdManager::InsertTfPlatformDeviceIdPair(
        DeviceType(device_type_), tf_device_id, platform_device_id));
  }

  std::vector<DeviceLocality> device_localities;
  TF_RETURN_IF_ERROR(GetDeviceLocalities(num_tf_devices, &device_localities));

  // Build the PluggableDevices.
  for (int di = 0; di < num_tf_devices; ++di) {
    TfDeviceId tf_device_id(di);
    int64_t bytes = memory_limit_bytes[di];
    TF_RETURN_IF_ERROR(CreatePluggableDevice(options, name_prefix, tf_device_id,
                                             bytes, device_localities[di],
                                             devices));
  }
  return Status::OK();
}

static string GetShortDeviceDescription(PlatformDeviceId platform_device_id,
                                        const se::DeviceDescription& desc) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_6(mht_6_v, 409, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "GetShortDeviceDescription");

  return strings::StrCat("device: ", platform_device_id.value(),
                         ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id());
}

Status PluggableDeviceFactory::CreatePluggableDevice(
    const SessionOptions& options, const string& name_prefix,
    TfDeviceId tf_device_id, int64_t memory_limit,
    const DeviceLocality& dev_locality,
    std::vector<std::unique_ptr<Device>>* devices) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_7(mht_7_v, 423, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::CreatePluggableDevice");

  DCHECK_GE(tf_device_id.value(), 0);
  const string device_name = strings::StrCat(
      name_prefix, "/device:", device_type_, ":", tf_device_id.value());

  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  DeviceIdUtil::CheckValidTfDeviceId(DeviceType(device_type_), platform,
                                     tf_device_id);
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
      DeviceType(device_type_), tf_device_id, &platform_device_id));
  int numa_node = dev_locality.numa_node();

  auto desc_status = platform->DescriptionForDevice(platform_device_id.value());
  if (!desc_status.ok()) {
    return desc_status.status();
  }
  auto desc = desc_status.ConsumeValueOrDie();
  PluggableDeviceProcessState* process_state =
      PluggableDeviceProcessState::singleton(device_type_, platform_name_);
  Allocator* device_allocator = process_state->GetPluggableDeviceAllocator(
      options.config.gpu_options(), tf_device_id, memory_limit);
  if (device_allocator == nullptr) {
    return errors::Internal(
        "Failed to get memory allocator for TF PluggableDevice ",
        tf_device_id.value(), " with", memory_limit, " bytes of memory. ");
  }
  absl::optional<AllocatorStats> stats = device_allocator->GetStats();
  if (!stats) {
    return errors::Internal("No allocator statistics");
  }
  // 'memory_limit' is the required memory size, but if the allocator with
  // given 'tf_device_id' was created before, we'll use it instead of creating
  // a new one (as TF Device is a shared resource), in which case the actual
  // memory limit represented by 'stats.bytes_limit' used by that allocator
  // may be different (which should be an error).
  int64_t bytes_limit = stats->bytes_limit ? *stats->bytes_limit : 0;
  auto pluggable_device = absl::make_unique<PluggableDevice>(
      options, device_name, device_type_, platform_name_,
      static_cast<Bytes>(bytes_limit), dev_locality, tf_device_id,
      GetShortDeviceDescription(platform_device_id, *desc), device_allocator,
      ProcessState::singleton()->GetCPUAllocator(numa_node),
      false /*sync every op*/);
  LOG(INFO) << "Created TensorFlow device (" << device_name << " with "
            << (bytes_limit >> 20)
            << " MB memory) -> physical PluggableDevice ("
            << GetShortDeviceDescription(platform_device_id, *desc) << ")";
  TF_RETURN_IF_ERROR(pluggable_device->Init(options));
  devices->push_back(std::move(pluggable_device));
  return Status::OK();
}

Status PluggableDeviceFactory::GetDeviceLocalities(
    int num_tf_devices, std::vector<DeviceLocality>* device_localities) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_factoryDTcc mht_8(mht_8_v, 479, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc", "PluggableDeviceFactory::GetDeviceLocalities");

  for (int i = 0; i < num_tf_devices; ++i) {
    TfDeviceId tf_device_id(i);
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType(device_type_), tf_device_id, &platform_device_id));
    // Get PluggableDevice bus_id from its reported NUMA affinity. Because
    // devices are virtualized in some environment, we can't just use the device
    // id. NUMA locales are indexed from 0, buses are indexed from 1.
    se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
    auto desc_status =
        platform->DescriptionForDevice(platform_device_id.value());
    if (!desc_status.ok()) {
      return desc_status.status();
    }
    auto desc = desc_status.ConsumeValueOrDie();
    int numa_node = desc->numa_node();
    if (numa_node < 0) {
      // For some reason the StreamExecutor couldn't get the NUMA
      // affinity of the device. If this is not a multi-socket mobo with
      // devices local to different buses, it doesn't matter. If it is,
      // we may run into trouble later with data transfer operations.
      // The trouble may manifest as slower than expected performance,
      // or outright failures.
      LOG(INFO) << "Could not identify NUMA node of platform " << device_type_
                << " ID " << platform_device_id
                << ", defaulting to 0. Your kernel may not have been built "
                << "with NUMA support.";
      numa_node = 0;
    }
    DeviceLocality dev_locality;
    dev_locality.set_numa_node(numa_node);
    dev_locality.set_bus_id(numa_node + 1);
    device_localities->push_back(dev_locality);
    VLOG(1) << "PluggableDevice PlatformDeviceId " << platform_device_id
            << " TfDeviceId " << tf_device_id << " on bus "
            << dev_locality.bus_id() << " numa: " << numa_node
            << "DeviceLocality: " << dev_locality.DebugString();
  }
  return Status::OK();
}

}  // namespace tensorflow
