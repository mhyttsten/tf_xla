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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

using ::testing::Property;
using ::testing::UnorderedElementsAre;

// Create a fake 'Device' whose only interesting attribute is a non-default
// DeviceLocality and incarnation.
std::unique_ptr<Device> NewDevice(const string& type, const string& name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/distributed_runtime/device_resolver_distributed_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/distributed_runtime/device_resolver_distributed_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/distributed_runtime/device_resolver_distributed_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  attr.set_incarnation(random::New64());
  return absl::make_unique<FakeDevice>(attr);
}

class DeviceResDistTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSdevice_resolver_distributed_testDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/distributed_runtime/device_resolver_distributed_test.cc", "SetUp");

    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:0"));
    devices.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:1"));
    dev_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    dev_resolver_ =
        absl::make_unique<DeviceResolverDistributed>(dev_mgr_.get());

    std::vector<DeviceAttributes> attributes;
    attributes.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:1/device:CPU:0")
            ->attributes());
    attributes.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:1/device:CPU:1")
            ->attributes());
    TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
  }

  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::unique_ptr<DeviceResolverDistributed> dev_resolver_;
};

TEST_F(DeviceResDistTest, GetDeviceAttributesLocal) {
  DeviceAttributes attributes;
  TF_ASSERT_OK(dev_resolver_->GetDeviceAttributes(
      "/job:worker/replica:0/task:0/device:CPU:0", &attributes));
  EXPECT_EQ(attributes.name(), "/job:worker/replica:0/task:0/device:CPU:0");
}

TEST_F(DeviceResDistTest, GetDeviceAttributesLocalUnknown) {
  DeviceAttributes attributes;
  EXPECT_TRUE(errors::IsNotFound(dev_resolver_->GetDeviceAttributes(
      "/job:worker/replica:0/task:0/device:CPU:9", &attributes)));
}

TEST_F(DeviceResDistTest, GetAllDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:1")));
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:1", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:1/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:1/device:CPU:1")));
}

TEST_F(DeviceResDistTest, GetAllDeviceAttributesUnknown) {
  std::vector<DeviceAttributes> attributes;
  EXPECT_TRUE(errors::IsNotFound(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:3", &attributes)));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:2/device:CPU:0")
          ->attributes());
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:2/device:CPU:1")
          ->attributes());
  TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
  // Get the new task.
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:2", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:2/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:2/device:CPU:1")));
  // Get an existing task.
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:1")));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributesExisting) {
  std::vector<DeviceAttributes> attributes;
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributesDifferentIncarnation) {
  std::vector<DeviceAttributes> attributes;
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:0")
          ->attributes());
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:1")
          ->attributes());
  EXPECT_TRUE(errors::IsFailedPrecondition(
      dev_resolver_->UpdateDeviceAttributes(attributes)));
}

}  // namespace
}  // namespace tensorflow
