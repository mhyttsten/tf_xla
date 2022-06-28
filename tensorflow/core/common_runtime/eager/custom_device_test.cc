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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc() {
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

#include "tensorflow/core/common_runtime/eager/custom_device.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace eager {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

class TestCustomDevice : public CustomDevice {
 public:
  explicit TestCustomDevice(std::string name) : name_(name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "TestCustomDevice");
}
  const std::string& name() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "name");
 return name_; }
  Status CopyTensorToDevice(ImmediateExecutionTensorHandle* tensor,
                            ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "CopyTensorToDevice");

    tensor->Ref();
    *result = tensor;
    return Status::OK();
  }
  Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* tensor,
      const std::string& target_device_name,
      ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target_device_name: \"" + target_device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "CopyTensorFromDevice");

    tensor->Ref();
    *result = tensor;
    return Status::OK();
  }
  Status Execute(const ImmediateExecutionOperation* op,
                 ImmediateExecutionTensorHandle** retvals,
                 int* num_retvals) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "Execute");

    return errors::Unimplemented("Not implemented");
  }

  Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
              ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_5(mht_5_v, 245, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "Pack");

    return errors::Unimplemented("Packing is not implemented");
  }

 private:
  std::string name_;
};

class TestCustomDeviceTensorHandle : public CustomDeviceTensorHandle {
 public:
  TestCustomDeviceTensorHandle(ImmediateExecutionContext* context,
                               TestCustomDevice* device,
                               tensorflow::DataType dtype, int64_t length)
      : CustomDeviceTensorHandle(context, device, dtype), length_(length) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "TestCustomDeviceTensorHandle");
}

  void* DevicePointer() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "DevicePointer");
 return nullptr; }
  Status NumDims(int* num_dims) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_8(mht_8_v, 270, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "NumDims");

    *num_dims = 1;
    return Status::OK();
  }
  Status Dim(int dim_index, int64_t* dim) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_9(mht_9_v, 277, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "Dim");

    if (dim_index == 0) {
      *dim = length_;
      return Status::OK();
    } else {
      return errors::Internal("Dim out of bounds");
    }
  }

  Status SummarizeValue(std::string& summary) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_testDTcc mht_10(mht_10_v, 289, "", "./tensorflow/core/common_runtime/eager/custom_device_test.cc", "SummarizeValue");

    summary = std::string("TestValue");
    return Status::OK();
  }

 private:
  const int64_t length_;
};

TEST(CustomDevice, TestTensorHandle) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice device(device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &device, DT_FLOAT,
                                       /*length=*/3));
  Status s;
  std::string device_type = tensor->DeviceType(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ("CUSTOM", device_type);
  int device_index = tensor->DeviceId(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(15, device_index);
  int64_t num_elements = 0;
  s = tensor->NumElements(&num_elements);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(3, num_elements);
  EXPECT_THAT(
      tensor->DebugString(),
      ContainsRegex(
          R"re(TensorHandle\(TestValue, shape=\[3\], dtype=DT_FLOAT, device=.*\))re"));
}

TEST(CustomDevice, TestTensorHandleUnknownDimNumElements) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice device(device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &device, DT_FLOAT,
                                       /*length=*/-1));
  int64_t num_elements;
  Status s = tensor->NumElements(&num_elements);
  EXPECT_FALSE(s.ok());
  EXPECT_THAT(s.error_message(), HasSubstr("representing varying shapes"));
}

TEST(CustomDevice, TestResourcePlacement) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string custom_device_name =
      "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice custom_device(custom_device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> custom_float_tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &custom_device, DT_FLOAT,
                                       /*length=*/3));
  core::RefCountPtr<TestCustomDeviceTensorHandle> custom_resource_tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &custom_device, DT_RESOURCE,
                                       /*length=*/3));

  Tensor resource_tensor(DT_RESOURCE, {});
  Device* physical_device = device_mgr.ListDevices().at(0);
  core::RefCountPtr<TensorHandle> physical_resource_tensor(
      TensorHandle::CreateLocalHandle(std::move(resource_tensor),
                                      physical_device, physical_device,
                                      physical_device, ctx.get()));
  Tensor float_tensor(DT_FLOAT, {});
  core::RefCountPtr<TensorHandle> physical_float_tensor(
      TensorHandle::CreateLocalHandle(std::move(float_tensor), physical_device,
                                      physical_device, physical_device,
                                      ctx.get()));
  EagerOperation op(ctx.get());
  TF_ASSERT_OK(op.Reset("AssignVariableOp", ""));
  TF_ASSERT_OK(op.AddInput(physical_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(custom_float_tensor.get()));
  CustomDevice* placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // MaybePinToCustomDevice has no opinion about ops which have physical
  // resource-dtype inputs. They'll get placed on physical devices.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(op.Reset("AssignVariableOp", custom_device_name.c_str()));
  TF_ASSERT_OK(op.AddInput(physical_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(custom_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Explicit placement onto a custom device also doesn't trigger custom device
  // placement if there's a physical device resource input.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(
      op.Reset("Identity", "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(op.AddInput(physical_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Explicit placements typically override input-based placement onto a custom
  // device.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(op.Reset("AssignVariableOp",
                        "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(op.AddInput(custom_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(physical_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Even with an explicit physical device placement, custom device resource
  // inputs place the op on the custom device.
  ASSERT_NE(placed_device, nullptr);
  EXPECT_EQ(&custom_device, placed_device);
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow
