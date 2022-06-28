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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc() {
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

#include "tensorflow/core/common_runtime/device_mgr.h"

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

DeviceMgr::~DeviceMgr() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/common_runtime/device_mgr.cc", "DeviceMgr::~DeviceMgr");
}

StaticDeviceMgr::StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)),
      name_backing_store_(128),
      cpu_device_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::StaticDeviceMgr");

  for (auto& d : devices_) {
    // Register under the (1) full name and (2) canonical name.
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
    // Register under the (3) local name and (4) legacy local name.
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
    const auto& t = d->device_type();
    device_type_counts_[t]++;
    device_incarnation_set_.insert(d->attributes().incarnation());
    if (cpu_device_ == nullptr && t == "CPU" && d->parsed_name().id == 0) {
      cpu_device_ = d.get();
    }
  }
}

StaticDeviceMgr::StaticDeviceMgr(std::unique_ptr<Device> device)
    : StaticDeviceMgr([&device] {
        std::vector<std::unique_ptr<Device>> vector;
        vector.push_back(std::move(device));
        return vector;
      }()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::StaticDeviceMgr");
}

StaticDeviceMgr::~StaticDeviceMgr() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::~StaticDeviceMgr");

  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  for (auto& device : devices_) {
    device->ClearResourceMgr();
  }
}

StringPiece StaticDeviceMgr::CopyToBackingStore(StringPiece s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::CopyToBackingStore");

  size_t n = s.size();
  char* space = name_backing_store_.Alloc(n);
  memcpy(space, s.data(), n);
  return StringPiece(space, n);
}

void StaticDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_5(mht_5_v, 262, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::ListDeviceAttributes");

  devices->reserve(devices_.size());
  for (const auto& dev : devices_) {
    devices->emplace_back(dev->attributes());
  }
}

std::vector<Device*> StaticDeviceMgr::ListDevices() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_6(mht_6_v, 272, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::ListDevices");

  std::vector<Device*> devices(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    devices[i] = devices_[i].get();
  }
  return devices;
}

string StaticDeviceMgr::DebugString() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_7(mht_7_v, 283, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::DebugString");

  string out;
  for (const auto& dev : devices_) {
    strings::StrAppend(&out, dev->name(), "\n");
  }
  return out;
}

string StaticDeviceMgr::DeviceMappingString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_8(mht_8_v, 294, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::DeviceMappingString");

  string out;
  for (const auto& dev : devices_) {
    if (!dev->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, dev->name(), " -> ",
                         dev->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}

Status StaticDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::LookupDevice");

  auto iter = device_map_.find(name);
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

bool StaticDeviceMgr::ContainsDevice(int64_t device_incarnation) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_10(mht_10_v, 326, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::ContainsDevice");

  return device_incarnation_set_.contains(device_incarnation);
}

void StaticDeviceMgr::ClearContainers(
    gtl::ArraySlice<string> containers) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_11(mht_11_v, 334, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::ClearContainers");

  Status s;
  for (const auto& dev : devices_) {
    if (containers.empty()) {
      s.Update(dev->resource_manager()->Cleanup(
          dev->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(dev->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}

int StaticDeviceMgr::NumDeviceType(const string& type) const {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_12(mht_12_v, 355, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::NumDeviceType");

  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}

Device* StaticDeviceMgr::HostCPU() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_mgrDTcc mht_13(mht_13_v, 364, "", "./tensorflow/core/common_runtime/device_mgr.cc", "StaticDeviceMgr::HostCPU");
 return cpu_device_; }

}  // namespace tensorflow
