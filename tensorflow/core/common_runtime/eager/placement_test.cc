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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::tensorflow::test::function::NDef;

constexpr char kFullCPU[] = "/job:a/replica:0/task:0/device:CPU:0";
constexpr char kFullGPU[] = "/job:a/replica:0/task:0/device:FakeGPU:0";

////////////////////////////////////////////////////////////////////////////////
//
// Op, kernel to set up the environment.
//
// The Placer uses information about the op (input types),
// kernel (device constraints). To avoid depending on the full runtime, we
// define dummy implementations of these, and register them with the
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

// A dummy OpKernel that is used to register ops on different devices.
class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "DummyOp");
}
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "Compute");
}
};

// Register the following ops so they can be added to a Graph, and
// kernels so that they can be placed on particular device types.
REGISTER_OP("InvalidOp").Output("o: Ref(float)");

REGISTER_OP("TestOp").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU).Priority(1), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestOp").Device("FakeGPU").Priority(2), DummyOp);

static Device* CreateDevice(const char* type, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "CreateDevice");

  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_4(mht_4_v, 243, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_5(mht_5_v, 247, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class PlacementTest : public ::testing::Test {
 public:
  PlacementTest() : device_manager_(nullptr), context_(nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_6(mht_6_v, 260, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "PlacementTest");
}

  ~PlacementTest() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_7(mht_7_v, 265, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "~PlacementTest");

    delete device_manager_;
    if (context_) {
      context_->Unref();
    }
  }

  EagerContext* context() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_8(mht_8_v, 275, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "context");
 return context_; }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_9(mht_9_v, 281, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "InitContext");

    ASSERT_EQ(context_, nullptr);
    InitDeviceManager();
    context_ =
        new EagerContext(opts, policy,
                         /* async */ false, device_manager_,
                         /* device_mgr_owned */ false, /* rendezvous */ nullptr,
                         /* cluster_flr */ nullptr);
  }

 protected:
  void InitDeviceManager() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_testDTcc mht_10(mht_10_v, 295, "", "./tensorflow/core/common_runtime/eager/placement_test.cc", "InitDeviceManager");

    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = new DynamicDeviceMgr();
    std::vector<std::unique_ptr<Device>> added_devices;
    SessionOptions opts;

    // Have to use real CPU device. Other, ctx->HostCPU() will return invalid
    // device.
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, kFullCPU));
    added_devices.emplace_back(CreateDevice("FakeGPU", kFullGPU));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  DynamicDeviceMgr* device_manager_;
  EagerContext* context_;
};

TEST_F(PlacementTest, SelectDeviceExplicitHardPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(false);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;

  // No supported devices should result in an error.
  requested.Clear();
  NodeDef invalid_op = NDef("invalid_op", "InvalidOp", {}, {});

  Status status = context()->SelectDevice(requested, invalid_op, &dev);
  LOG(ERROR) << status.ToString();
  EXPECT_TRUE(errors::IsNotFound(status));
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "Could not find device for node"))
      << "unexpected error message " << status.error_message();

  // An invalid requested device should also cause an error.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("FakeGPU:99", &requested));
  NodeDef node = NDef("x", "TestOp", {}, {});
  status = context()->SelectDevice(requested, node, &dev);

  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "Could not satisfy device specification"))
      << "unexpected error message " << status.error_message();

  // Should pick the device with higher priority if given no constraints.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, node, &dev));
  EXPECT_EQ(dev->device_type(), "FakeGPU");

  // Should pick a CPU if asked to.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("CPU:0", &requested));
  TF_ASSERT_OK(context()->SelectDevice(requested, node, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_CPU);
}

TEST_F(PlacementTest, SelectDeviceExplicitSoftPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(true);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;

  // No supported devices should result in an error.
  requested.Clear();
  NodeDef invalid_op = NDef("invalid_op", "InvalidOp", {}, {});

  Status status = context()->SelectDevice(requested, invalid_op, &dev);
  LOG(ERROR) << status.ToString();
  EXPECT_TRUE(errors::IsNotFound(status));
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "Could not find device for node"))
      << "unexpected error message " << status.error_message();

  // An invalid requested device should be replaced by the "best" one.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("FakeGPU:99", &requested));
  NodeDef node = NDef("x", "TestOp", {}, {});
  status = context()->SelectDevice(requested, node, &dev);
  EXPECT_EQ(dev->device_type(), "FakeGPU");
}

}  // namespace
}  // namespace tensorflow
