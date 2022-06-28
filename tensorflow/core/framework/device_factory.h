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

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh() {
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


#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
struct SessionOptions;

class DeviceFactory {
 public:
  virtual ~DeviceFactory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/framework/device_factory.h", "~DeviceFactory");
}
  static void Register(const std::string& device_type,
                       std::unique_ptr<DeviceFactory> factory, int priority,
                       bool is_pluggable_device);
  ABSL_DEPRECATED("Use the `Register` function above instead")
  static void Register(const std::string& device_type, DeviceFactory* factory,
                       int priority, bool is_pluggable_device) {
    Register(device_type, std::unique_ptr<DeviceFactory>(factory), priority,
             is_pluggable_device);
  }
  static DeviceFactory* GetFactory(const std::string& device_type);

  // Append to "*devices" CPU devices.
  static Status AddCpuDevices(const SessionOptions& options,
                              const std::string& name_prefix,
                              std::vector<std::unique_ptr<Device>>* devices);

  // Append to "*devices" all suitable devices, respecting
  // any device type specific properties/counts listed in "options".
  //
  // CPU devices are added first.
  static Status AddDevices(const SessionOptions& options,
                           const std::string& name_prefix,
                           std::vector<std::unique_ptr<Device>>* devices);

  // Helper for tests.  Create a single device of type "type".  The
  // returned device is always numbered zero, so if creating multiple
  // devices of the same type, supply distinct name_prefix arguments.
  static std::unique_ptr<Device> NewDevice(const string& type,
                                           const SessionOptions& options,
                                           const string& name_prefix);

  // Iterate through all device factories and build a list of all of the
  // possible physical devices.
  //
  // CPU is are added first.
  static Status ListAllPhysicalDevices(std::vector<string>* devices);

  // Iterate through all device factories and build a list of all of the
  // possible pluggable physical devices.
  static Status ListPluggablePhysicalDevices(std::vector<string>* devices);

  // Get details for a specific device among all device factories.
  // 'device_index' indexes into devices from ListAllPhysicalDevices.
  static Status GetAnyDeviceDetails(
      int device_index, std::unordered_map<string, string>* details);

  // For a specific device factory list all possible physical devices.
  virtual Status ListPhysicalDevices(std::vector<string>* devices) = 0;

  // Get details for a specific device for a specific factory. Subclasses
  // can store arbitrary device information in the map. 'device_index' indexes
  // into devices from ListPhysicalDevices.
  virtual Status GetDeviceDetails(int device_index,
                                  std::unordered_map<string, string>* details) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh mht_1(mht_1_v, 259, "", "./tensorflow/core/framework/device_factory.h", "GetDeviceDetails");

    return Status::OK();
  }

  // Most clients should call AddDevices() instead.
  virtual Status CreateDevices(
      const SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) = 0;

  // Return the device priority number for a "device_type" string.
  //
  // Higher number implies higher priority.
  //
  // In standard TensorFlow distributions, GPU device types are
  // preferred over CPU, and by default, custom devices that don't set
  // a custom priority during registration will be prioritized lower
  // than CPU.  Custom devices that want a higher priority can set the
  // 'priority' field when registering their device to something
  // higher than the packaged devices.  See calls to
  // REGISTER_LOCAL_DEVICE_FACTORY to see the existing priorities used
  // for built-in devices.
  static int32 DevicePriority(const std::string& device_type);

  // Returns true if 'device_type' is registered from plugin. Returns false if
  // 'device_type' is a first-party device.
  static bool IsPluggableDevice(const std::string& device_type);
};

namespace dfactory {

template <class Factory>
class Registrar {
 public:
  // Multiple registrations for the same device type with different priorities
  // are allowed.  Priorities are used in two different ways:
  //
  // 1) When choosing which factory (that is, which device
  //    implementation) to use for a specific 'device_type', the
  //    factory registered with the highest priority will be chosen.
  //    For example, if there are two registrations:
  //
  //      Registrar<CPUFactory1>("CPU", 125);
  //      Registrar<CPUFactory2>("CPU", 150);
  //
  //    then CPUFactory2 will be chosen when
  //    DeviceFactory::GetFactory("CPU") is called.
  //
  // 2) When choosing which 'device_type' is preferred over other
  //    DeviceTypes in a DeviceSet, the ordering is determined
  //    by the 'priority' set during registration.  For example, if there
  //    are two registrations:
  //
  //      Registrar<CPUFactory>("CPU", 100);
  //      Registrar<GPUFactory>("GPU", 200);
  //
  //    then DeviceType("GPU") will be prioritized higher than
  //    DeviceType("CPU").
  //
  // The default priority values for built-in devices is:
  // GPU: 210
  // GPUCompatibleCPU: 70
  // ThreadPoolDevice: 60
  // Default: 50
  explicit Registrar(const std::string& device_type, int priority = 50) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_factoryDTh mht_2(mht_2_v, 326, "", "./tensorflow/core/framework/device_factory.h", "Registrar");

    DeviceFactory::Register(device_type, std::make_unique<Factory>(), priority,
                            /*is_pluggable_device*/ false);
  }
};

}  // namespace dfactory

#define REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, ...) \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory,   \
                                         __COUNTER__, ##__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, \
                                               ctr, ...)                    \
  static ::tensorflow::dfactory::Registrar<device_factory>                  \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr)(device_type, ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_
