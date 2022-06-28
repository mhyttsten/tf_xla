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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc() {
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
#include <iterator>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

DynamicDeviceMgr::DynamicDeviceMgr() : cpu_device_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::DynamicDeviceMgr");
}

DynamicDeviceMgr::DynamicDeviceMgr(
    std::vector<std::unique_ptr<Device>> devices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::DynamicDeviceMgr");

  Status status = AddDevices(std::move(devices));
  CHECK(status.ok());  // Crash OK
  mutex_lock l(devices_mu_);
  // Initialize cpu_device_.
  for (int i = 0; i < dynamic_devices_.size(); ++i) {
    auto* d = dynamic_devices_[i].get();
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }
}

DynamicDeviceMgr::~DynamicDeviceMgr() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::~DynamicDeviceMgr");

  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  mutex_lock l(devices_mu_);
  for (const auto& d : dynamic_devices_) {
    // TODO(tf-runtime-team): clear devices' resource mgr in devices'
    // destructor.
    d->ClearResourceMgr();
  }
}

void DynamicDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::ListDeviceAttributes");

  tf_shared_lock l(devices_mu_);
  devices->reserve(dynamic_devices_.size());
  for (const auto& d : dynamic_devices_) {
    devices->emplace_back(d->attributes());
  }
}

std::vector<Device*> DynamicDeviceMgr::ListDevices() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_4(mht_4_v, 248, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::ListDevices");

  tf_shared_lock l(devices_mu_);
  std::vector<Device*> devices;
  devices.reserve(dynamic_devices_.size());
  for (const auto& d : dynamic_devices_) {
    devices.emplace_back(d.get());
  }
  return devices;
}

string DynamicDeviceMgr::DebugString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_5(mht_5_v, 261, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::DebugString");

  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& d : dynamic_devices_) {
    strings::StrAppend(&out, d->name(), "\n");
  }
  return out;
}

string DynamicDeviceMgr::DeviceMappingString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::DeviceMappingString");

  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& d : dynamic_devices_) {
    if (!d->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, d->name(), " -> ",
                         d->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}

Status DynamicDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_7(mht_7_v, 288, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::LookupDevice");

  tf_shared_lock l(devices_mu_);
  auto iter = device_map_.find(string(name));
  if (iter == device_map_.end()) {
    std::vector<StringPiece> device_names;
    for (auto&& itr : device_map_) {
      device_names.push_back(itr.first);
    }
    VLOG(1) << "Unknown device: " << name
            << " all devices: " << absl::StrJoin(device_names, ", ");
    return errors::InvalidArgument(name, " unknown device.");
  }
  *device = iter->second;
  return Status::OK();
}

bool DynamicDeviceMgr::ContainsDevice(int64_t device_incarnation) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_8(mht_8_v, 307, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::ContainsDevice");

  tf_shared_lock l(devices_mu_);
  return device_incarnation_set_.contains(device_incarnation);
}

void DynamicDeviceMgr::ClearContainers(
    gtl::ArraySlice<string> containers) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::ClearContainers");

  Status s;
  tf_shared_lock l(devices_mu_);
  for (const auto& d : dynamic_devices_) {
    if (containers.empty()) {
      s.Update(d->resource_manager()->Cleanup(
          d->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(d->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}

int DynamicDeviceMgr::NumDeviceType(const string& type) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_10(mht_10_v, 338, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::NumDeviceType");

  tf_shared_lock l(devices_mu_);
  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}

Status DynamicDeviceMgr::AddDevices(
    std::vector<std::unique_ptr<Device>> devices) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_11(mht_11_v, 349, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::AddDevices");

  mutex_lock l(devices_mu_);
  for (auto& d : devices) {
    if (device_map_.find(d->name()) != device_map_.end()) {
      return errors::InvalidArgument(
          "Trying to add device ", d->name(),
          " to manager but its name conflicts with an existing device.");
    }
    // Register under the (1) full name and (2) canonical name.
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    // Register under the (3) local name and (4) legacy local name.
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    device_type_counts_[d->device_type()]++;
    device_incarnation_set_.insert(d->attributes().incarnation());
    dynamic_devices_.push_back(std::move(d));
  }
  return Status::OK();
}

Status DynamicDeviceMgr::RemoveDevices(const std::vector<Device*>& devices) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_12(mht_12_v, 377, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::RemoveDevices");

  mutex_lock l(devices_mu_);

  for (const auto& d : devices) {
    if (d == cpu_device_) {
      TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Can not remove HostCPU device ", d->name()));
    }
    int i = 0;
    for (; i < dynamic_devices_.size(); ++i) {
      if (d == dynamic_devices_[i].get()) break;
    }
    if (i >= dynamic_devices_.size()) {
      return errors::InvalidArgument("Unknown device ", d->name());
    }
  }

  for (const auto& d : devices) {
    // Clear registration of (1) full name and (2) canonical name
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    // Clear registration of (3) local name and (4) legacy local name
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    device_type_counts_[d->device_type()]--;
    device_incarnation_set_.erase(d->attributes().incarnation());

    int i = 0;
    for (; i < dynamic_devices_.size(); ++i) {
      if (d == dynamic_devices_[i].get()) break;
    }
    // There shouldn't be unknown devices at this point.
    CHECK(i < dynamic_devices_.size());  // Crash OK
    stale_devices_.add(std::move(dynamic_devices_[i]));
    dynamic_devices_.erase(dynamic_devices_.begin() + i);
  }
  return Status::OK();
}

Status DynamicDeviceMgr::RemoveDevicesByName(
    const std::vector<string>& device_names) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_13(mht_13_v, 424, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::RemoveDevicesByName");

  std::vector<Device*> devices_to_remove;
  for (const string& name : device_names) {
    Device* device;
    TF_RETURN_IF_ERROR(LookupDevice(name, &device));
    devices_to_remove.emplace_back(device);
  }
  return RemoveDevices(devices_to_remove);
}

Device* DynamicDeviceMgr::HostCPU() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgrDTcc mht_14(mht_14_v, 437, "", "./tensorflow/core/common_runtime/dynamic_device_mgr.cc", "DynamicDeviceMgr::HostCPU");

  Device* device = cpu_device_.load(std::memory_order_relaxed);

  // Host CPU device can't be removed, so if we found valid device once, we
  // do not need to check that it is still in the device list.
  if (device != nullptr) return device;

  mutex_lock l(devices_mu_);
  for (int i = 0; i < dynamic_devices_.size(); ++i) {
    Device* d = dynamic_devices_[i].get();
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }

  return cpu_device_.load(std::memory_order_relaxed);
}

}  // namespace tensorflow
