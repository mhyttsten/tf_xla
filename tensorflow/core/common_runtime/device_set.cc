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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc() {
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

#include "tensorflow/core/common_runtime/device_set.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

DeviceSet::DeviceSet() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::DeviceSet");
}

DeviceSet::~DeviceSet() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::~DeviceSet");
}

void DeviceSet::AddDevice(Device* device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_2(mht_2_v, 208, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::AddDevice");

  mutex_lock l(devices_mu_);
  devices_.push_back(device);
  prioritized_devices_.clear();
  prioritized_device_types_.clear();
  for (const string& name :
       DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
    device_by_name_.insert({name, device});
  }
}

void DeviceSet::FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                                    std::vector<Device*>* devices) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::FindMatchingDevices");

  // TODO(jeff): If we are going to repeatedly lookup the set of devices
  // for the same spec, maybe we should have a cache of some sort
  devices->clear();
  for (Device* d : devices_) {
    if (DeviceNameUtils::IsCompleteSpecification(spec, d->parsed_name())) {
      devices->push_back(d);
    }
  }
}

Device* DeviceSet::FindDeviceByName(const string& name) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::FindDeviceByName");

  return gtl::FindPtrOrNull(device_by_name_, name);
}

// static
int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_5(mht_5_v, 246, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::DeviceTypeOrder");

  return DeviceFactory::DevicePriority(d.type_string());
}

static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_6(mht_6_v, 253, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceTypeComparator");

  // First sort by prioritized device type (higher is preferred) and
  // then by device name (lexicographically).
  auto a_priority = DeviceSet::DeviceTypeOrder(a);
  auto b_priority = DeviceSet::DeviceTypeOrder(b);
  if (a_priority != b_priority) {
    return a_priority > b_priority;
  }

  return StringPiece(a.type()) < StringPiece(b.type());
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_7(mht_7_v, 268, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::PrioritizedDeviceTypeList");

  std::vector<DeviceType> result;
  std::set<string> seen;
  for (Device* d : devices_) {
    const auto& t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(t);
    }
  }
  std::sort(result.begin(), result.end(), DeviceTypeComparator);
  return result;
}

void DeviceSet::SortPrioritizedDeviceTypeVector(
    PrioritizedDeviceTypeVector* vector) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_8(mht_8_v, 285, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::SortPrioritizedDeviceTypeVector");

  if (vector == nullptr) return;

  auto device_sort = [](const PrioritizedDeviceTypeVector::value_type& a,
                        const PrioritizedDeviceTypeVector::value_type& b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_9(mht_9_v, 292, "", "./tensorflow/core/common_runtime/device_set.cc", "lambda");

    // First look at set priorities.
    if (a.second != b.second) {
      return a.second > b.second;
    }
    // Then fallback to default priorities.
    return DeviceTypeComparator(a.first, b.first);
  };

  std::sort(vector->begin(), vector->end(), device_sort);
}

void DeviceSet::SortPrioritizedDeviceVector(PrioritizedDeviceVector* vector) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_10(mht_10_v, 307, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::SortPrioritizedDeviceVector");

  auto device_sort = [](const std::pair<Device*, int32>& a,
                        const std::pair<Device*, int32>& b) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_11(mht_11_v, 312, "", "./tensorflow/core/common_runtime/device_set.cc", "lambda");

    if (a.second != b.second) {
      return a.second > b.second;
    }

    const string& a_type_name = a.first->device_type();
    const string& b_type_name = b.first->device_type();
    if (a_type_name != b_type_name) {
      auto a_priority = DeviceFactory::DevicePriority(a_type_name);
      auto b_priority = DeviceFactory::DevicePriority(b_type_name);
      if (a_priority != b_priority) {
        return a_priority > b_priority;
      }
    }

    if (a.first->IsLocal() != b.first->IsLocal()) {
      return a.first->IsLocal();
    }

    return StringPiece(a.first->name()) < StringPiece(b.first->name());
  };
  std::sort(vector->begin(), vector->end(), device_sort);
}

namespace {

void UpdatePrioritizedVectors(
    const std::vector<Device*>& devices,
    PrioritizedDeviceVector* prioritized_devices,
    PrioritizedDeviceTypeVector* prioritized_device_types) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_12(mht_12_v, 344, "", "./tensorflow/core/common_runtime/device_set.cc", "UpdatePrioritizedVectors");

  if (prioritized_devices->size() != devices.size()) {
    for (Device* d : devices) {
      prioritized_devices->emplace_back(
          d, DeviceSet::DeviceTypeOrder(DeviceType(d->device_type())));
    }
    DeviceSet::SortPrioritizedDeviceVector(prioritized_devices);
  }

  if (prioritized_device_types != nullptr &&
      prioritized_device_types->size() != devices.size()) {
    std::set<DeviceType> seen;
    for (const std::pair<Device*, int32>& p : *prioritized_devices) {
      DeviceType t(p.first->device_type());
      if (seen.insert(t).second) {
        prioritized_device_types->emplace_back(t, p.second);
      }
    }
  }
}

}  // namespace

const PrioritizedDeviceVector& DeviceSet::prioritized_devices() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_13(mht_13_v, 370, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::prioritized_devices");

  mutex_lock l(devices_mu_);
  UpdatePrioritizedVectors(devices_, &prioritized_devices_,
                           /* prioritized_device_types */ nullptr);
  return prioritized_devices_;
}

const PrioritizedDeviceTypeVector& DeviceSet::prioritized_device_types() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTcc mht_14(mht_14_v, 380, "", "./tensorflow/core/common_runtime/device_set.cc", "DeviceSet::prioritized_device_types");

  mutex_lock l(devices_mu_);
  UpdatePrioritizedVectors(devices_, &prioritized_devices_,
                           &prioritized_device_types_);
  return prioritized_device_types_;
}

}  // namespace tensorflow
