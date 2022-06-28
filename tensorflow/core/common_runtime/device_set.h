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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh() {
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


#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

typedef std::vector<std::pair<Device*, int32>> PrioritizedDeviceVector;

// DeviceSet is a container class for managing the various types of
// devices used by a model.
class DeviceSet {
 public:
  DeviceSet();
  ~DeviceSet();

  // Does not take ownership of 'device'.
  void AddDevice(Device* device) TF_LOCKS_EXCLUDED(devices_mu_);

  // Set the device designated as the "client".  This device
  // must also be registered via AddDevice().
  void set_client_device(Device* device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/device_set.h", "set_client_device");

    DCHECK(client_device_ == nullptr);
    client_device_ = device;
  }

  // Returns a pointer to the device designated as the "client".
  Device* client_device() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/common_runtime/device_set.h", "client_device");
 return client_device_; }

  // Return the list of devices in this set.
  const std::vector<Device*>& devices() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_setDTh mht_2(mht_2_v, 228, "", "./tensorflow/core/common_runtime/device_set.h", "devices");
 return devices_; }

  // Given a DeviceNameUtils::ParsedName (which may have some
  // wildcards for different components), fills "*devices" with all
  // devices in "*this" that match "spec".
  void FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                           std::vector<Device*>* devices) const;

  // Finds the device with the given "fullname". Returns nullptr if
  // not found.
  Device* FindDeviceByName(const string& fullname) const;

  // Return the list of unique device types in this set, ordered
  // with more preferable devices earlier.
  std::vector<DeviceType> PrioritizedDeviceTypeList() const;

  // Return the prioritized list of devices in this set.
  // Devices are prioritized first by `DeviceTypeOrder`, then by name.
  const PrioritizedDeviceVector& prioritized_devices() const
      TF_LOCKS_EXCLUDED(devices_mu_);

  // Return the prioritized list of unique device types in this set.
  //
  // The list will be ordered by decreasing priority. The priorities (the second
  // element in the list's `std::pair<DeviceType, int32>`) will be initialized
  // to the value of `DeviceTypeOrder` for the device types.
  const PrioritizedDeviceTypeVector& prioritized_device_types() const
      TF_LOCKS_EXCLUDED(devices_mu_);

  // An order to sort by device types according to system-determined
  // priority.
  //
  // Higher result implies higher priority.
  static int DeviceTypeOrder(const DeviceType& d);

  // Sorts a PrioritizedDeviceVector according to devices and explicit
  // priorities.
  //
  // After a call to this function, the argument vector will be sorted by
  // explicit priority (the second element in the `std::pair<DeviceType,
  // int32>`), then by `DeviceTypeOrder` of the device type, then by device
  // locality, and lastly by device name.
  static void SortPrioritizedDeviceVector(PrioritizedDeviceVector* vector);

  // Sorts a PrioritizedDeviceTypeVector according to types and explicit
  // priorities.
  //
  // After a call to this function, the argument vector will be sorted by
  // explicit priority (the second element in the `std::pair<DeviceType,
  // int32>`), then by `DeviceTypeOrder` of the device type.
  static void SortPrioritizedDeviceTypeVector(
      PrioritizedDeviceTypeVector* vector);

 private:
  mutable mutex devices_mu_;

  // Not owned.
  std::vector<Device*> devices_;

  // Cached prioritized vector, created on-the-fly when
  // prioritized_devices() is called.
  mutable PrioritizedDeviceVector prioritized_devices_
      TF_GUARDED_BY(devices_mu_);

  // Cached prioritized vector, created on-the-fly when
  // prioritized_device_types() is called.
  mutable PrioritizedDeviceTypeVector prioritized_device_types_
      TF_GUARDED_BY(devices_mu_);

  // Fullname -> device* for device in devices_.
  std::unordered_map<string, Device*> device_by_name_;

  // client_device_ points to an element of devices_ that we consider
  // to be the client device (in this local process).
  Device* client_device_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceSet);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
