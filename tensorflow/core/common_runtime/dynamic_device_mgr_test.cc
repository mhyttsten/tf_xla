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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc() {
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

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// Return a fake device with the specified type and name.
static Device* CreateDevice(const char* type, const char* name,
                            Notification* n = nullptr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "CreateDevice");

  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_3(mht_3_v, 219, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "GetAllocator");
 return nullptr; }
  };

  class FakeDeviceWithDestructorNotification : public FakeDevice {
   public:
    FakeDeviceWithDestructorNotification(const DeviceAttributes& attr,
                                         Notification* n)
        : FakeDevice(attr), n_(n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_4(mht_4_v, 229, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "FakeDeviceWithDestructorNotification");
}
    ~FakeDeviceWithDestructorNotification() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdynamic_device_mgr_testDTcc mht_5(mht_5_v, 233, "", "./tensorflow/core/common_runtime/dynamic_device_mgr_test.cc", "~FakeDeviceWithDestructorNotification");
 n_->Notify(); }

   private:
    Notification* n_;
  };

  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  do {
    attr.set_incarnation(random::New64());
  } while (attr.incarnation() == 0);

  if (n) {
    return new FakeDeviceWithDestructorNotification(attr, n);
  }
  return new FakeDevice(attr);
}

TEST(DynamicDeviceMgrTest, AddDeviceToMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));

  auto dm = MakeUnique<DynamicDeviceMgr>();
  EXPECT_EQ(dm->ListDevices().size(), 0);

  std::vector<std::unique_ptr<Device>> added_devices;
  added_devices.emplace_back(std::move(d0));
  added_devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(added_devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);
}

TEST(DynamicDeviceMgrTest, RemoveDeviceFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  Device* d1_ptr = d1.get();
  const int64_t d1_incarnation = d1->attributes().incarnation();

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);

  std::vector<Device*> removed_devices{d1_ptr};
  TF_CHECK_OK(dm->RemoveDevices(removed_devices));
  EXPECT_EQ(dm->ListDevices().size(), 1);
  EXPECT_FALSE(dm->ContainsDevice(d1_incarnation));

  // Device still accessible shortly through the raw pointer after removal.
  EXPECT_EQ(d1_ptr->name(), "/device:CPU:1");
  EXPECT_EQ(d1_ptr->device_type(), "CPU");
}

TEST(DynamicDeviceMgrTest, RemoveDeviceFromMgrBuffer) {
  // Create a device whose destructor will send a notification.
  Notification n;
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0", &n));
  Device* d0_ptr = d0.get();
  std::vector<std::unique_ptr<Device>> added_devices;
  added_devices.emplace_back(std::move(d0));
  auto dm = MakeUnique<DynamicDeviceMgr>();
  TF_CHECK_OK(dm->AddDevices(std::move(added_devices)));
  std::vector<Device*> removed_devices{d0_ptr};
  TF_CHECK_OK(dm->RemoveDevices(removed_devices));

  // Repeatedly add and remove devices to fill up the stale devices buffer.
  for (int i = 0; i < kStaleDeviceBufferSize; i++) {
    added_devices.clear();
    removed_devices.clear();
    std::unique_ptr<Device> d(CreateDevice("CPU", "/device:CPU:0"));
    Device* d_ptr = d.get();
    added_devices.emplace_back(std::move(d));
    TF_CHECK_OK(dm->AddDevices(std::move(added_devices)));
    removed_devices.emplace_back(d_ptr);
    TF_CHECK_OK(dm->RemoveDevices(removed_devices));
  }
  // Verify that d0 destructor is called after the buffer is full.
  n.WaitForNotification();
}

TEST(DynamicDeviceMgrTest, RemoveDeviceByNameFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  string d1_name = "/device:CPU:1";

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);

  std::vector<string> removed_devices{d1_name};
  TF_CHECK_OK(dm->RemoveDevicesByName(removed_devices));
  EXPECT_EQ(dm->ListDevices().size(), 1);
}

TEST(DynamicDeviceMgrTest, AddRepeatedDeviceToMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:0"));

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<std::unique_ptr<Device>> added_devices;
  added_devices.emplace_back(std::move(d1));
  Status s = dm->AddDevices(std::move(added_devices));
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "name conflicts with an existing device"));
}

TEST(DynamicDeviceMgrTest, RemoveNonExistingDeviceFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  Device* d0_ptr = d0.get();
  Device* d1_ptr = d1.get();

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<Device*> removed_devices{d0_ptr, d1_ptr};
  Status s = dm->RemoveDevices(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "Unknown device"));
  EXPECT_EQ(dm->ListDevices().size(), 1);  // d0 *not* removed.
}

TEST(DynamicDeviceMgrTest, RemoveNonExistingDeviceByNameFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  string d0_name = "/device:GPU:0";
  string d1_name = "/device:CPU:0";

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<string> removed_devices{d0_name, d1_name};
  Status s = dm->RemoveDevicesByName(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "unknown device"));
  EXPECT_EQ(dm->ListDevices().size(), 1);  // d0 *not* removed
}

TEST(DynamicDeviceMgrTest, HostCPU) {
  auto dm = MakeUnique<DynamicDeviceMgr>();

  // If there are no CPU devices, HostCPU() should return nullptr.
  std::unique_ptr<Device> gpu(CreateDevice("GPU", "/device:GPU:0"));
  Device* gpu_ptr = gpu.get();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(gpu));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);
  EXPECT_EQ(dm->HostCPU(), nullptr);

  // After adding a CPU device, it should return that device.
  std::unique_ptr<Device> cpu0(CreateDevice("CPU", "/device:CPU:0"));
  Device* cpu0_ptr = cpu0.get();
  devices.clear();
  devices.emplace_back(std::move(cpu0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // If we add another CPU device, HostCPU() should remain the same.
  std::unique_ptr<Device> cpu1(CreateDevice("CPU", "/device:CPU:1"));
  Device* cpu1_ptr = cpu1.get();
  devices.clear();
  devices.emplace_back(std::move(cpu1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 3);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // Once we have a HostCPU() device, we can't remove it ...
  std::vector<Device*> removed{gpu_ptr, cpu0_ptr};
  EXPECT_TRUE(absl::StrContains(dm->RemoveDevices(removed).error_message(),
                                "Can not remove HostCPU device"));
  EXPECT_EQ(dm->ListDevices().size(), 3);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // ... but we should be able to remove another CPU device.
  removed = std::vector<Device*>{cpu1_ptr};
  TF_CHECK_OK(dm->RemoveDevices(removed));
  EXPECT_EQ(dm->ListDevices().size(), 2);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);
}

}  // namespace
}  // namespace tensorflow
