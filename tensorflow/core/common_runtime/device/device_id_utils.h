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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_utilsDTh() {
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


#include <numeric>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

// Utility methods for translation between TensorFlow device ids and platform
// device ids.
class DeviceIdUtil {
 public:
  // Convenient methods for getting the associated executor given a TfDeviceId
  // or PlatformDeviceId.
  static se::port::StatusOr<se::StreamExecutor*> ExecutorForPlatformDeviceId(
      se::Platform* device_manager, PlatformDeviceId platform_device_id) {
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }
  static se::port::StatusOr<se::StreamExecutor*> ExecutorForTfDeviceId(
      const DeviceType& type, se::Platform* device_manager,
      TfDeviceId tf_device_id) {
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
        type, tf_device_id, &platform_device_id));
    return ExecutorForPlatformDeviceId(device_manager, platform_device_id);
  }

  // Verify that the platform_device_id associated with a TfDeviceId is
  // legitimate.
  static void CheckValidTfDeviceId(const DeviceType& type,
                                   se::Platform* device_manager,
                                   TfDeviceId tf_device_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_utilsDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/common_runtime/device/device_id_utils.h", "CheckValidTfDeviceId");

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(type, tf_device_id,
                                                      &platform_device_id));
    const int visible_device_count = device_manager->VisibleDeviceCount();
    CHECK_LT(platform_device_id.value(), visible_device_count)
        << "platform_device_id is outside discovered device range."
        << " TF " << type << " id: " << tf_device_id << ", platform " << type
        << " id: " << platform_device_id
        << ", visible device count: " << visible_device_count;
  }

  // Parse `visible_device_list` into a list of platform Device ids.
  static Status ParseVisibleDeviceList(
      const string& visible_device_list, const int visible_device_count,
      std::vector<PlatformDeviceId>* visible_device_order) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("visible_device_list: \"" + visible_device_list + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_utilsDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/common_runtime/device/device_id_utils.h", "ParseVisibleDeviceList");

    visible_device_order->clear();

    // If the user wants to remap the visible to virtual Device mapping,
    // check for that here.
    if (visible_device_list.empty()) {
      visible_device_order->resize(visible_device_count);
      // By default, visible to virtual mapping is unchanged.
      std::iota(visible_device_order->begin(), visible_device_order->end(), 0);
    } else {
      const std::vector<string> order_str =
          str_util::Split(visible_device_list, ',');
      for (const string& platform_device_id_str : order_str) {
        int32_t platform_device_id;
        if (!strings::safe_strto32(platform_device_id_str,
                                   &platform_device_id)) {
          return errors::InvalidArgument(
              "Could not parse entry in 'visible_device_list': '",
              platform_device_id_str,
              "'. visible_device_list = ", visible_device_list);
        }
        if (platform_device_id < 0 ||
            platform_device_id >= visible_device_count) {
          return errors::InvalidArgument(
              "'visible_device_list' listed an invalid Device id '",
              platform_device_id, "' but visible device count is ",
              visible_device_count);
        }
        visible_device_order->push_back(PlatformDeviceId(platform_device_id));
      }
    }

    // Validate no repeats.
    std::set<PlatformDeviceId> visible_device_set(visible_device_order->begin(),
                                                  visible_device_order->end());
    if (visible_device_set.size() != visible_device_order->size()) {
      return errors::InvalidArgument(
          "visible_device_list contained a duplicate entry: ",
          visible_device_list);
    }
    return Status::OK();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
