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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_manager_testDTcc() {
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

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PlatformDeviceId TfToPlatformDeviceId(const DeviceType& type, TfDeviceId tf) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_id_manager_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/common_runtime/device/device_id_manager_test.cc", "TfToPlatformDeviceId");

  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(
      DeviceIdManager::TfToPlatformDeviceId(type, tf, &platform_device_id));
  return platform_device_id;
}

TEST(DeviceIdManagerTest, Basics) {
  DeviceType device_type("GPU");
  TfDeviceId key_0(0);
  PlatformDeviceId value_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_0,
                                                             value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type, key_0));

  // Multiple calls to map the same value is ok.
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_0,
                                                             value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type, key_0));

  // Map a different TfDeviceId to a different value.
  TfDeviceId key_1(3);
  PlatformDeviceId value_1(2);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_1,
                                                             value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type, key_1));

  // Mapping a different TfDeviceId to the same value is ok.
  TfDeviceId key_2(10);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_2,
                                                             value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type, key_2));

  // Mapping the same TfDeviceId to a different value.
  ASSERT_FALSE(
      DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_2, value_0)
          .ok());

  // Getting a nonexistent mapping.
  ASSERT_FALSE(DeviceIdManager::TfToPlatformDeviceId(device_type,
                                                     TfDeviceId(100), &value_0)
                   .ok());
}

TEST(DeviceIdManagerTest, TwoDevices) {
  // Setup 0 --> 0 mapping for device GPU.
  DeviceType device_type0("GPU");
  TfDeviceId key_0(0);
  PlatformDeviceId value_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type0,
                                                             key_0, value_0));
  // Setup 2 --> 3 mapping for device XPU.
  DeviceType device_type1("XPU");
  TfDeviceId key_1(2);
  PlatformDeviceId value_1(3);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type1,
                                                             key_1, value_1));

  // Key 0 is available for device GPU.
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type0, key_0));
  // Key 2 is available for device XPU.
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type1, key_1));
  // Key 2 is *not* available for device GPU
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId(device_type0, key_1, &value_0)
          .ok());
  // Key 0 is not available for device XPU.
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId(device_type1, key_0, &value_1)
          .ok());
  // Key 0 is not available for device FOO.
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId("FOO", key_0, &value_0).ok());
}
}  // namespace
}  // namespace tensorflow
