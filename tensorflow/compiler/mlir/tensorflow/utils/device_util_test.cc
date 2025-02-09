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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_util_testDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_util_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util_test.cc", "FakeDevice");
}

  Status Sync() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_util_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util_test.cc", "Sync");
 return errors::Unimplemented("FakeDevice::Sync()"); }

  static std::unique_ptr<Device> Make(const string& name,
                                      const string& desc = "") {
    DeviceNameUtils::ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(name, &parsed_name);

    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(parsed_name.type);
    device_attributes.set_physical_device_desc(desc);
    return std::make_unique<FakeDevice>(device_attributes);
  }
};

TEST(DeviceUtilTest, AddDeviceToOp) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  const std::string cpu0 = "/job:worker/replica:0/task:0/device:CPU:0";
  const std::string gpu0 = "/job:worker/replica:1/task:2/device:GPU:0";
  const std::string gpu1 = "/job:worker/replica:1/task:2/device:GPU:1";

  llvm::SmallVector<std::unique_ptr<Device>, 2> devices;
  devices.push_back(FakeDevice::Make(cpu0));
  devices.push_back(FakeDevice::Make(gpu0, "compute capability: 7.0"));
  devices.push_back(FakeDevice::Make(gpu1));

  DeviceSet device_set;
  for (auto& device : devices) device_set.AddDevice(device.get());
  AddDevicesToOp(*module_ref, &device_set);

  auto devices_attr =
      (*module_ref)->getAttrOfType<mlir::DictionaryAttr>("tf.devices");
  ASSERT_NE(devices_attr, nullptr);
  ASSERT_EQ(devices_attr.size(), 3);

  // CPU device added with an empty metadata.
  auto device_meta_0 = devices_attr.get(cpu0).dyn_cast<mlir::DictionaryAttr>();
  ASSERT_NE(device_meta_0, nullptr);
  ASSERT_EQ(device_meta_0.size(), 0);

  // GPU device successfully parsed compute capability from description.
  auto device_meta_1 =
      devices_attr.get(gpu0).dyn_cast<mlir::TF::GpuDeviceMetadata>();
  ASSERT_NE(device_meta_1, nullptr);
  ASSERT_EQ(device_meta_1.cc_major().getInt(), 7);
  ASSERT_EQ(device_meta_1.cc_minor().getInt(), 0);

  // If description is empty GPU devices added with an empty metadata.
  auto device_meta_2 = devices_attr.get(gpu1).dyn_cast<mlir::DictionaryAttr>();
  ASSERT_NE(device_meta_2, nullptr);
  ASSERT_EQ(device_meta_2.size(), 0);
}

TEST(DeviceUtilTest, AddDeviceToOpNullDeviceSet) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  AddDevicesToOp(*module_ref, /*device_set=*/nullptr);
  EXPECT_EQ((*module_ref)->getAttr("tf.devices"), nullptr);
}

TEST(DeviceUtilTest, GetDevicesFromOpNoDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeType) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)->setAttr("tf.devices", builder.getBoolAttr(false));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeArraySubtype) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)->setAttr("tf.devices", builder.getI32ArrayAttr({8}));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)
      ->setAttr("tf.devices",
                builder.getDictionaryAttr(builder.getNamedAttr(
                    "bad_device", builder.getDictionaryAttr({}))));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpValidDeviceInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);

  auto device_dict = builder.getDictionaryAttr(
      {builder.getNamedAttr("/job:worker/replica:0/task:0/device:CPU:0",
                            builder.getDictionaryAttr({}))});
  (*module_ref)->setAttr("tf.devices", device_dict);

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));

  ASSERT_EQ(devices.NumDevices(), 1);
  ASSERT_EQ(devices.device_names().size(), 1);
  ASSERT_EQ(DeviceNameUtils::ParsedNameToString(devices.device_names()[0]),
            "/job:worker/replica:0/task:0/device:CPU:0");
}

TEST(DeviceUtilTest, GetGpuDeviceMetadata) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  mlir::Builder builder(*module_ref);

  const std::string gpu0 = "/job:worker/replica:0/task:0/device:GPU:0";
  const std::string gpu1 = "/job:worker/replica:0/task:0/device:GPU:1";

  llvm::SmallVector<mlir::NamedAttribute, 2> metadata;
  metadata.push_back(builder.getNamedAttr(
      gpu0, mlir::TF::GpuDeviceMetadata::get(builder.getI32IntegerAttr(1),
                                             builder.getI32IntegerAttr(2),
                                             module_ref->getContext())));

  (*module_ref)->setAttr("tf.devices", builder.getDictionaryAttr(metadata));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));

  DeviceNameUtils::ParsedName parsed_name;
  DeviceNameUtils::ParseFullName(gpu0, &parsed_name);
  auto meta_0 = devices.GetGpuDeviceMetadata(parsed_name);
  ASSERT_TRUE(meta_0.hasValue());
  ASSERT_EQ(meta_0->cc_major().getInt(), 1);
  ASSERT_EQ(meta_0->cc_minor().getInt(), 2);

  DeviceNameUtils::ParseFullName(gpu1, &parsed_name);
  auto meta_1 = devices.GetGpuDeviceMetadata(parsed_name);
  ASSERT_FALSE(meta_1.hasValue());
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceString) {
  const std::string tpu0 = "/job:worker/replica:0/task:0/device:TPU:0";
  const std::string tpu1 = "/job:worker/replica:0/task:0/device:TPU:1";

  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal0 = -1;
  mlir::LogicalResult result0 =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu0, &device_ordinal0);
  EXPECT_TRUE(mlir::succeeded(result0));
  EXPECT_EQ(device_ordinal0, 0);

  int64_t device_ordinal1 = -1;
  mlir::LogicalResult result1 =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu1, &device_ordinal1);
  EXPECT_TRUE(mlir::succeeded(result1));
  EXPECT_EQ(device_ordinal1, 1);
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceStringInvalid) {
  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal = -1;
  mlir::LogicalResult result = GetDeviceOrdinalFromDeviceString(
      unknown_loc, "bad_device", &device_ordinal);
  EXPECT_TRUE(mlir::failed(result));
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceStringNoId) {
  const std::string tpu_no_id = "/job:worker/replica:0/task:0/device:TPU";

  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal = -1;
  mlir::LogicalResult result =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu_no_id, &device_ordinal);
  EXPECT_TRUE(mlir::failed(result));
}

}  // anonymous namespace
}  // namespace tensorflow
