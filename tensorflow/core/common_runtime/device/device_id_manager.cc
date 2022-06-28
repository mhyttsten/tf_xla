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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc() {
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

#include "tensorflow/core/common_runtime/device/device_id_manager.h"

#include <unordered_map>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {
// Manages the map between TfDeviceId and platform device id.
class TfToPlatformDeviceIdMap {
 public:
  static TfToPlatformDeviceIdMap* singleton() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/common_runtime/device/device_id_manager.cc", "singleton");

    static auto* id_map = new TfToPlatformDeviceIdMap;
    return id_map;
  }

  Status Insert(const DeviceType& type, TfDeviceId tf_device_id,
                PlatformDeviceId platform_device_id) TF_LOCKS_EXCLUDED(mu_) {
    std::pair<IdMapType::iterator, bool> result;
    {
      mutex_lock lock(mu_);
      TypeIdMapType::iterator device_id_map_iter =
          id_map_.insert({type.type_string(), IdMapType()}).first;
      result = device_id_map_iter->second.insert(
          {tf_device_id.value(), platform_device_id.value()});
    }
    if (!result.second && platform_device_id.value() != result.first->second) {
      return errors::AlreadyExists(
          "TensorFlow device (", type, ":", tf_device_id.value(),
          ") is being mapped to multiple devices (", platform_device_id.value(),
          " now, and ", result.first->second,
          " previously), which is not supported. "
          "This may be the result of providing different ",
          type, " configurations (ConfigProto.gpu_options, for example ",
          "different visible_device_list) when creating multiple Sessions in ",
          "the same process. This is not currently supported, see ",
          "https://github.com/tensorflow/tensorflow/issues/19083");
    }
    return Status::OK();
  }

  bool Find(const DeviceType& type, TfDeviceId tf_device_id,
            PlatformDeviceId* platform_device_id) const TF_LOCKS_EXCLUDED(mu_) {
    // TODO(mrry): Consider replacing this with an atomic `is_initialized` bit,
    // to avoid writing to a shared cache line in the tf_shared_lock.
    tf_shared_lock lock(mu_);
    auto type_id_map_iter = id_map_.find(type.type_string());
    if (type_id_map_iter == id_map_.end()) return false;
    auto id_map_iter = type_id_map_iter->second.find(tf_device_id.value());
    if (id_map_iter == type_id_map_iter->second.end()) return false;
    *platform_device_id = id_map_iter->second;
    return true;
  }

 private:
  TfToPlatformDeviceIdMap() = default;

  void TestOnlyReset() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    id_map_.clear();
  }

  // Map from physical device id to platform device id.
  using IdMapType = std::unordered_map<int32, int32>;
  // Map from DeviceType to IdMapType.
  // We use std::string instead of DeviceType because the key should
  // be default-initializable.
  using TypeIdMapType = std::unordered_map<std::string, IdMapType>;
  mutable mutex mu_;
  TypeIdMapType id_map_ TF_GUARDED_BY(mu_);

  friend class ::tensorflow::DeviceIdManager;
  TF_DISALLOW_COPY_AND_ASSIGN(TfToPlatformDeviceIdMap);
};
}  // namespace

Status DeviceIdManager::InsertTfPlatformDeviceIdPair(
    const DeviceType& type, TfDeviceId tf_device_id,
    PlatformDeviceId platform_device_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc mht_1(mht_1_v, 271, "", "./tensorflow/core/common_runtime/device/device_id_manager.cc", "DeviceIdManager::InsertTfPlatformDeviceIdPair");

  return TfToPlatformDeviceIdMap::singleton()->Insert(type, tf_device_id,
                                                      platform_device_id);
}

Status DeviceIdManager::TfToPlatformDeviceId(
    const DeviceType& type, TfDeviceId tf_device_id,
    PlatformDeviceId* platform_device_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc mht_2(mht_2_v, 281, "", "./tensorflow/core/common_runtime/device/device_id_manager.cc", "DeviceIdManager::TfToPlatformDeviceId");

  if (TfToPlatformDeviceIdMap::singleton()->Find(type, tf_device_id,
                                                 platform_device_id)) {
    return Status::OK();
  }
  return errors::NotFound("TensorFlow device ", type, ":", tf_device_id.value(),
                          " was not registered");
}

void DeviceIdManager::TestOnlyReset() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_managerDTcc mht_3(mht_3_v, 293, "", "./tensorflow/core/common_runtime/device/device_id_manager.cc", "DeviceIdManager::TestOnlyReset");

  TfToPlatformDeviceIdMap::singleton()->TestOnlyReset();
}

}  // namespace tensorflow
