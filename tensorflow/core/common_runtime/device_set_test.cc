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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc() {
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

#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Return a fake device with the specified type and name.
static Device* Dev(const char* type, const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/common_runtime/device_set_test.cc", "Dev");

  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/common_runtime/device_set_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/common_runtime/device_set_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_3(mht_3_v, 213, "", "./tensorflow/core/common_runtime/device_set_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class DeviceSetTest : public ::testing::Test {
 public:
  Device* AddDevice(const char* type, const char* name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_4(mht_4_v, 228, "", "./tensorflow/core/common_runtime/device_set_test.cc", "AddDevice");

    Device* d = Dev(type, name);
    owned_.emplace_back(d);
    devices_.AddDevice(d);
    return d;
  }

  const DeviceSet& device_set() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_5(mht_5_v, 238, "", "./tensorflow/core/common_runtime/device_set_test.cc", "device_set");
 return devices_; }

  std::vector<DeviceType> types() const {
    return devices_.PrioritizedDeviceTypeList();
  }

 private:
  DeviceSet devices_;
  std::vector<std::unique_ptr<Device>> owned_;
};

class DummyFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_6(mht_6_v, 254, "", "./tensorflow/core/common_runtime/device_set_test.cc", "ListPhysicalDevices");

    return Status::OK();
  }
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevice_set_testDTcc mht_7(mht_7_v, 262, "", "./tensorflow/core/common_runtime/device_set_test.cc", "CreateDevices");

    return Status::OK();
  }
};

// Assumes the default priority is '50'.
REGISTER_LOCAL_DEVICE_FACTORY("d1", DummyFactory);
REGISTER_LOCAL_DEVICE_FACTORY("d2", DummyFactory, 51);
REGISTER_LOCAL_DEVICE_FACTORY("d3", DummyFactory, 49);

TEST_F(DeviceSetTest, PrioritizedDeviceTypeList) {
  EXPECT_EQ(50, DeviceSet::DeviceTypeOrder(DeviceType("d1")));
  EXPECT_EQ(51, DeviceSet::DeviceTypeOrder(DeviceType("d2")));
  EXPECT_EQ(49, DeviceSet::DeviceTypeOrder(DeviceType("d3")));

  EXPECT_EQ(std::vector<DeviceType>{}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:1");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  // D2 is prioritized higher than D1.
  AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ((std::vector<DeviceType>{DeviceType("d2"), DeviceType("d1")}),
            types());

  // D3 is prioritized below D1.
  AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ((std::vector<DeviceType>{
                DeviceType("d2"),
                DeviceType("d1"),
                DeviceType("d3"),
            }),
            types());
}

TEST_F(DeviceSetTest, prioritized_devices) {
  Device* d1 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  Device* d2 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ(device_set().prioritized_devices(),
            (PrioritizedDeviceVector{std::make_pair(d2, 51),
                                     std::make_pair(d1, 50)}));

  // Cache is rebuilt when a device is added.
  Device* d3 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ(
      device_set().prioritized_devices(),
      (PrioritizedDeviceVector{std::make_pair(d2, 51), std::make_pair(d1, 50),
                               std::make_pair(d3, 49)}));
}

TEST_F(DeviceSetTest, prioritized_device_types) {
  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ(
      device_set().prioritized_device_types(),
      (PrioritizedDeviceTypeVector{std::make_pair(DeviceType("d2"), 51),
                                   std::make_pair(DeviceType("d1"), 50)}));

  // Cache is rebuilt when a device is added.
  AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ(
      device_set().prioritized_device_types(),
      (PrioritizedDeviceTypeVector{std::make_pair(DeviceType("d2"), 51),
                                   std::make_pair(DeviceType("d1"), 50),
                                   std::make_pair(DeviceType("d3"), 49)}));
}

TEST_F(DeviceSetTest, SortPrioritizedDeviceVector) {
  Device* d1_0 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  Device* d2_0 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  Device* d3_0 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  Device* d1_1 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:1");
  Device* d2_1 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:1");
  Device* d3_1 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:1");

  PrioritizedDeviceVector sorted{
      std::make_pair(d3_1, 30), std::make_pair(d1_0, 10),
      std::make_pair(d2_0, 20), std::make_pair(d3_0, 30),
      std::make_pair(d1_1, 20), std::make_pair(d2_1, 10)};

  device_set().SortPrioritizedDeviceVector(&sorted);

  EXPECT_EQ(sorted, (PrioritizedDeviceVector{
                        std::make_pair(d3_0, 30), std::make_pair(d3_1, 30),
                        std::make_pair(d2_0, 20), std::make_pair(d1_1, 20),
                        std::make_pair(d2_1, 10), std::make_pair(d1_0, 10)}));
}

TEST_F(DeviceSetTest, SortPrioritizedDeviceTypeVector) {
  PrioritizedDeviceTypeVector sorted{std::make_pair(DeviceType("d3"), 20),
                                     std::make_pair(DeviceType("d1"), 20),
                                     std::make_pair(DeviceType("d2"), 30)};

  device_set().SortPrioritizedDeviceTypeVector(&sorted);

  EXPECT_EQ(sorted, (PrioritizedDeviceTypeVector{
                        std::make_pair(DeviceType("d2"), 30),
                        std::make_pair(DeviceType("d1"), 20),
                        std::make_pair(DeviceType("d3"), 20)}));
}

}  // namespace
}  // namespace tensorflow
