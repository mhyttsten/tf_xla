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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

#include <cstdint>
#include <tuple>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

using Device = DeviceNameUtils::ParsedName;

bool DeviceNamesToParsedNames(llvm::ArrayRef<std::string> device_names,
                              llvm::SmallVectorImpl<Device>* parsed_devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util_test.cc", "DeviceNamesToParsedNames");

  parsed_devices->reserve(device_names.size());
  for (const auto& device_name : device_names) {
    Device parsed_name;
    if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name))
      return false;

    parsed_devices->push_back(parsed_name);
  }
  return true;
}

using DeviceNames = llvm::SmallVector<std::string, 8>;

struct ParameterizedDeviceSetTest
    : ::testing::TestWithParam<std::tuple<DeviceNames, std::string>> {};

TEST_P(ParameterizedDeviceSetTest, BadDeviceSet) {
  llvm::SmallVector<Device, 8> devices;
  ASSERT_TRUE(DeviceNamesToParsedNames(std::get<0>(GetParam()), &devices));
  std::string topology_attr;
  std::vector<int64_t> device_assignment_attr;

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/1, /*num_cores_per_replica=*/1, topology_attr,
      device_assignment_attr);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().error_message(), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    BadDeviceSet, ParameterizedDeviceSetTest,
    ::testing::Values(
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:CPU:0"},
            "no TPU_SYSTEM devices found"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0"},
            "found TPU_SYSTEM devices with conflicting jobs 'localhost' and "
            "'worker'"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:localhost/replica:1/task:0/device:TPU_SYSTEM:0"},
            "found TPU_SYSTEM devices with conflicting replicas '0' and '1'"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:localhost/replica:0/task:0/device:TPU:0",
             "/job:localhost/replica:0/task:0/device:TPU:1",
             "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0",
             "/job:localhost/replica:0/task:1/device:TPU:0"},
            "expected the number of TPU devices per host to be 2, got 1")));

struct ParameterizedMetadataTest
    : ::testing::TestWithParam<std::tuple<int, int, std::string,
                                          std::vector<int64_t>, std::string>> {
};

TEST_P(ParameterizedMetadataTest, BadMetadata) {
  llvm::SmallVector<Device, 8> devices;
  ASSERT_TRUE(DeviceNamesToParsedNames(
      {"/job:worker/replica:0/task:0/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:0/device:TPU:0",
       "/job:worker/replica:0/task:1/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:1/device:TPU:0"},
      &devices));
  std::string compilation_device;
  llvm::SmallVector<llvm::SmallVector<std::string, 8>, 8> execution_devices;
  llvm::Optional<xla::DeviceAssignmentProto> xla_device_assignment;

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, std::get<0>(GetParam()), std::get<1>(GetParam()),
      std::get<2>(GetParam()), std::get<3>(GetParam()));
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().error_message(), std::get<4>(GetParam()));
}

std::string TopologyWithMeshShape(llvm::ArrayRef<int> mesh_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc mht_1(mht_1_v, 287, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util_test.cc", "TopologyWithMeshShape");

  tpu::TopologyProto topology_proto;
  for (int mesh_dim : mesh_shape) topology_proto.add_mesh_shape(mesh_dim);
  return topology_proto.SerializeAsString();
}

std::string TopologyWithMeshShapeAndTasks(llvm::ArrayRef<int> mesh_shape,
                                          int num_tasks,
                                          int num_tpu_devices_per_task) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc mht_2(mht_2_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util_test.cc", "TopologyWithMeshShapeAndTasks");

  tpu::TopologyProto topology_proto;
  for (int mesh_dim : mesh_shape) topology_proto.add_mesh_shape(mesh_dim);
  topology_proto.set_num_tasks(num_tasks);
  topology_proto.set_num_tpu_devices_per_task(num_tpu_devices_per_task);
  return topology_proto.SerializeAsString();
}

std::string TopologyWithDeviceCoordinates(
    llvm::ArrayRef<int> device_coordinates) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_util_testDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util_test.cc", "TopologyWithDeviceCoordinates");

  tpu::TopologyProto topology_proto;
  topology_proto.add_mesh_shape(2);
  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(1);
  topology_proto.set_num_tasks(2);
  topology_proto.set_num_tpu_devices_per_task(1);
  for (int device_coordinate : device_coordinates)
    topology_proto.add_device_coordinates(device_coordinate);
  return topology_proto.SerializeAsString();
}

INSTANTIATE_TEST_SUITE_P(
    BadFullMeshMetadata, ParameterizedMetadataTest,
    ::testing::Values(
        std::make_tuple(
            2, 1, "", std::vector<int64_t>{0},
            "'device_assignment' must not be set when 'topology' is not set"),
        std::make_tuple(8, 1, "", std::vector<int64_t>(),
                        "'num_replicas' must be equal to 1 or 2, got 8"),
        std::make_tuple(2, 2, "", std::vector<int64_t>(),
                        "'num_cores_per_replica' must be equal to 1, got 2")));

INSTANTIATE_TEST_SUITE_P(
    BadGeneralTopologyMetadata, ParameterizedMetadataTest,
    ::testing::Values(
        std::make_tuple(
            2, 1, "BAD_TOPOLOGY", std::vector<int64_t>(),
            "failed to parse 'topology' attribute to TopologyProto"),
        std::make_tuple(4, 2, TopologyWithMeshShape({0}),
                        std::vector<int64_t>(),
                        "'topology' 'mesh_shape' must be rank 4, got rank 1"),
        std::make_tuple(
            2, 1, TopologyWithMeshShape({2, 0, 1, 2}), std::vector<int64_t>(),
            "'topology' 'mesh_shape' dimension 1 must be positive, got 0"),
        std::make_tuple(2, 1, TopologyWithMeshShapeAndTasks({1, 1, 1, 1}, 1, 1),
                        std::vector<int64_t>(),
                        "number of tasks from available TPU devices must be "
                        "'num_tasks' in 'topology' (1), got 2"),
        std::make_tuple(2, 1, TopologyWithMeshShapeAndTasks({1, 1, 1, 1}, 2, 2),
                        std::vector<int64_t>(),
                        "number of TPU devices available per task must be "
                        "'num_tpu_devices_per_task' in 'topology' (2), got 1"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({}), std::vector<int64_t>(),
            "length of 'device_coordinates' in 'topology' must be 'num_tasks' "
            "* 'num_tpus_per_task' * 4 (2 * 1 * 4), got 0"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({-1, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (-1, 0, 0, 0) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({2, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (2, 0, 0, 0) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, -1, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (0, -1, 0, 0) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 1, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (0, 1, 0, 0) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, -1, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (0, 0, 0, -1) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 1, 1, 0, 0, 0}),
            std::vector<int64_t>(),
            "device coordinate (0, 0, 0, 1) in 'topology' is outside "
            "of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 0, 0, 0, 0}),
            std::vector<int64_t>(),
            "'topology' has duplicate device coordinate (0, 0, 0, 0)")));

INSTANTIATE_TEST_SUITE_P(
    BadGeneralDeviceAssignmentMetadata, ParameterizedMetadataTest,
    ::testing::Values(
        std::make_tuple(2, 1,
                        TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
                        std::vector<int64_t>(),
                        "length of 'device_assignment' must be 'num_replicas' "
                        "* 'num_cores_per_replica' * 4 (2 * 1 * 4), got 0"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{-1, 0, 0, 0, 0, 0, 0, 0},
            "device coordinate (-1, 0, 0, 0) in 'device_assignment' "
            "is outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{2, 0, 0, 0, 0, 0, 0, 0},
            "device coordinate (2, 0, 0, 0) in 'device_assignment' is "
            "outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{0, -1, 0, 0, 0, 0, 0, 0},
            "device coordinate (0, -1, 0, 0) in 'device_assignment' "
            "is outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{0, 1, 0, 0, 0, 0, 0, 0},
            "device coordinate (0, 1, 0, 0) in 'device_assignment' is "
            "outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{0, 0, 0, -1, 0, 0, 0, 0},
            "device coordinate (0, 0, 0, -1) in 'device_assignment' "
            "is outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(
            2, 1, TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
            std::vector<int64_t>{0, 0, 0, 1, 0, 0, 0, 0},
            "device coordinate (0, 0, 0, 1) in 'device_assignment' is "
            "outside of mesh shape (2, 1, 1, 1)"),
        std::make_tuple(2, 1,
                        TopologyWithDeviceCoordinates({0, 0, 0, 0, 1, 0, 0, 0}),
                        std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0},
                        "'device_assignment' has duplicate device coordinate "
                        "(0, 0, 0, 0)")));

std::vector<std::string> MakeDeviceSet(int num_tasks,
                                       int num_devices_per_task) {
  std::vector<std::string> devices{
      "/job:localhost/replica:0/task:0/device:CPU:0"};
  devices.reserve(num_tasks * num_devices_per_task + num_tasks + 1);

  for (int task = 0; task < num_tasks; ++task) {
    devices.push_back(
        llvm::formatv("/job:worker/replica:0/task:{0}/device:CPU:0", task)
            .str());
    devices.push_back(
        llvm::formatv("/job:worker/replica:0/task:{0}/device:TPU_SYSTEM:0",
                      task)
            .str());
    for (int device = 0; device < num_devices_per_task; ++device)
      devices.push_back(
          llvm::formatv("/job:worker/replica:0/task:{0}/device:TPU:{1}", task,
                        device)
              .str());
  }

  return devices;
}

TEST(TPURewriteDeviceUtilTest,
     BadGeneralDeviceAssignmentMetadataMissingDevice) {
  tpu::TopologyProto topology_proto;
  {
    topology_proto.add_mesh_shape(2);
    topology_proto.add_mesh_shape(1);
    topology_proto.add_mesh_shape(1);
    topology_proto.add_mesh_shape(1);
    topology_proto.set_num_tasks(1);
    topology_proto.set_num_tpu_devices_per_task(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
  }

  std::string topology_attr = topology_proto.SerializeAsString();
  std::vector<int64_t> device_assignment_attr{1, 0, 0, 0};

  llvm::SmallVector<Device, 8> devices;
  std::vector<std::string> device_names =
      MakeDeviceSet(/*num_tasks=*/1, /*num_devices_per_task=*/1);
  ASSERT_TRUE(DeviceNamesToParsedNames(device_names, &devices));

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/1, /*num_cores_per_replica=*/1, topology_attr,
      device_assignment_attr);

  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().error_message(),
            "no TPU device found for 'device_assignment' device coordinate (1, "
            "0, 0, 0)");
}

TEST(TPURewriteDeviceUtilTest, ValidFullMeshDeviceAssignment) {
  llvm::SmallVector<Device, 8> devices;
  std::vector<std::string> device_names =
      MakeDeviceSet(/*num_tasks=*/2, /*num_devices_per_task=*/4);
  ASSERT_TRUE(DeviceNamesToParsedNames(device_names, &devices));
  std::string topology_attr;
  std::vector<int64_t> device_assignment_attr;

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/8, /*num_cores_per_replica=*/1, topology_attr,
      device_assignment_attr);

  TF_ASSERT_OK(status_or.status());

  const auto& tpu_device_assignment = status_or.ValueOrDie();
  EXPECT_EQ(tpu_device_assignment.compilation_device,
            "/job:worker/replica:0/task:0/device:CPU:0");
  const auto& tpu_devices = tpu_device_assignment.tpu_devices;
  ASSERT_EQ(tpu_devices.size(), 8);
  for (const auto& replica_tpu_devices : tpu_devices)
    ASSERT_EQ(replica_tpu_devices.size(), 1);

  EXPECT_EQ(tpu_devices[0][0].device,
            "/job:worker/replica:0/task:0/device:TPU:0");
  EXPECT_EQ(tpu_devices[0][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][0].device,
            "/job:worker/replica:0/task:0/device:TPU:1");
  EXPECT_EQ(tpu_devices[1][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[2][0].device,
            "/job:worker/replica:0/task:0/device:TPU:2");
  EXPECT_EQ(tpu_devices[2][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[3][0].device,
            "/job:worker/replica:0/task:0/device:TPU:3");
  EXPECT_EQ(tpu_devices[3][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[4][0].device,
            "/job:worker/replica:0/task:1/device:TPU:0");
  EXPECT_EQ(tpu_devices[4][0].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[5][0].device,
            "/job:worker/replica:0/task:1/device:TPU:1");
  EXPECT_EQ(tpu_devices[5][0].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[6][0].device,
            "/job:worker/replica:0/task:1/device:TPU:2");
  EXPECT_EQ(tpu_devices[6][0].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[7][0].device,
            "/job:worker/replica:0/task:1/device:TPU:3");
  EXPECT_EQ(tpu_devices[7][0].host,
            "/job:worker/replica:0/task:1/device:CPU:0");

  EXPECT_FALSE(tpu_device_assignment.xla_device_assignment.hasValue());
}

TEST(TPURewriteDeviceUtilTest, ValidGeneralDeviceAssignmentMesh2x2x2) {
  tpu::TopologyProto topology_proto;
  {
    topology_proto.add_mesh_shape(2);
    topology_proto.add_mesh_shape(2);
    topology_proto.add_mesh_shape(1);
    topology_proto.add_mesh_shape(2);
    topology_proto.set_num_tasks(2);
    topology_proto.set_num_tpu_devices_per_task(4);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
  }

  std::string topology_attr = topology_proto.SerializeAsString();
  std::vector<int64_t> device_assignment_attr{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                              0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0,
                                              0, 1, 1, 1, 0, 0, 1, 1, 0, 1};

  llvm::SmallVector<Device, 8> devices;
  std::vector<std::string> device_names =
      MakeDeviceSet(/*num_tasks=*/2, /*num_devices_per_task=*/4);
  ASSERT_TRUE(DeviceNamesToParsedNames(device_names, &devices));

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/4, /*num_cores_per_replica=*/2, topology_attr,
      device_assignment_attr);

  TF_ASSERT_OK(status_or.status());

  const auto& tpu_device_assignment = status_or.ValueOrDie();
  EXPECT_EQ(tpu_device_assignment.compilation_device,
            "/job:worker/replica:0/task:0/device:CPU:0");
  const auto& tpu_devices = tpu_device_assignment.tpu_devices;
  ASSERT_EQ(tpu_devices.size(), 4);
  for (const auto& replica_tpu_devices : tpu_devices)
    ASSERT_EQ(replica_tpu_devices.size(), 2);

  EXPECT_EQ(tpu_devices[0][0].device,
            "/job:worker/replica:0/task:0/device:TPU:0");
  EXPECT_EQ(tpu_devices[0][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[0][1].device,
            "/job:worker/replica:0/task:1/device:TPU:3");
  EXPECT_EQ(tpu_devices[0][1].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][0].device,
            "/job:worker/replica:0/task:0/device:TPU:1");
  EXPECT_EQ(tpu_devices[1][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][1].device,
            "/job:worker/replica:0/task:1/device:TPU:2");
  EXPECT_EQ(tpu_devices[1][1].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[2][0].device,
            "/job:worker/replica:0/task:0/device:TPU:3");
  EXPECT_EQ(tpu_devices[2][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[2][1].device,
            "/job:worker/replica:0/task:1/device:TPU:0");
  EXPECT_EQ(tpu_devices[2][1].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[3][0].device,
            "/job:worker/replica:0/task:0/device:TPU:2");
  EXPECT_EQ(tpu_devices[3][0].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[3][1].device,
            "/job:worker/replica:0/task:1/device:TPU:1");
  EXPECT_EQ(tpu_devices[3][1].host,
            "/job:worker/replica:0/task:1/device:CPU:0");

  auto& xla_device_assignment = tpu_device_assignment.xla_device_assignment;
  ASSERT_TRUE(xla_device_assignment.hasValue());
  EXPECT_EQ(xla_device_assignment->replica_count(), 4);
  EXPECT_EQ(xla_device_assignment->computation_count(), 2);
  ASSERT_EQ(xla_device_assignment->computation_devices_size(), 2);
  const auto& computation_device_0 =
      xla_device_assignment->computation_devices(0);
  ASSERT_EQ(computation_device_0.replica_device_ids_size(), 4);
  const auto& computation_device_1 =
      xla_device_assignment->computation_devices(1);
  ASSERT_EQ(computation_device_1.replica_device_ids_size(), 4);

  EXPECT_EQ(computation_device_0.replica_device_ids(0), 0);
  EXPECT_EQ(computation_device_0.replica_device_ids(1), 4);
  EXPECT_EQ(computation_device_0.replica_device_ids(2), 2);
  EXPECT_EQ(computation_device_0.replica_device_ids(3), 6);
  EXPECT_EQ(computation_device_1.replica_device_ids(0), 1);
  EXPECT_EQ(computation_device_1.replica_device_ids(1), 5);
  EXPECT_EQ(computation_device_1.replica_device_ids(2), 3);
  EXPECT_EQ(computation_device_1.replica_device_ids(3), 7);
}

TEST(TPURewriteDeviceUtilTest, ValidGeneralDeviceAssignmentMesh1x2x1x3) {
  tpu::TopologyProto topology_proto;
  {
    topology_proto.add_mesh_shape(1);
    topology_proto.add_mesh_shape(2);
    topology_proto.add_mesh_shape(1);
    topology_proto.add_mesh_shape(3);
    topology_proto.set_num_tasks(3);
    topology_proto.set_num_tpu_devices_per_task(2);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(2);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(0);
    topology_proto.add_device_coordinates(2);
  }

  std::string topology_attr = topology_proto.SerializeAsString();
  std::vector<int64_t> device_assignment_attr{
      0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0};

  llvm::SmallVector<Device, 8> devices;
  std::vector<std::string> device_names =
      MakeDeviceSet(/*num_tasks=*/3, /*num_devices_per_task=*/2);
  ASSERT_TRUE(DeviceNamesToParsedNames(device_names, &devices));

  auto status_or = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/2, /*num_cores_per_replica=*/3, topology_attr,
      device_assignment_attr);

  TF_ASSERT_OK(status_or.status());

  auto& tpu_device_assignment = status_or.ValueOrDie();
  EXPECT_EQ(tpu_device_assignment.compilation_device,
            "/job:worker/replica:0/task:0/device:CPU:0");

  auto& tpu_devices = tpu_device_assignment.tpu_devices;
  ASSERT_EQ(tpu_devices.size(), 2);
  for (const auto& replica_tpu_devices : tpu_devices)
    ASSERT_EQ(replica_tpu_devices.size(), 3);

  EXPECT_EQ(tpu_devices[0][0].device,
            "/job:worker/replica:0/task:1/device:TPU:1");
  EXPECT_EQ(tpu_devices[0][0].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[0][1].device,
            "/job:worker/replica:0/task:1/device:TPU:0");
  EXPECT_EQ(tpu_devices[0][1].host,
            "/job:worker/replica:0/task:1/device:CPU:0");
  EXPECT_EQ(tpu_devices[0][2].device,
            "/job:worker/replica:0/task:2/device:TPU:0");
  EXPECT_EQ(tpu_devices[0][2].host,
            "/job:worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][0].device,
            "/job:worker/replica:0/task:2/device:TPU:1");
  EXPECT_EQ(tpu_devices[1][0].host,
            "/job:worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][1].device,
            "/job:worker/replica:0/task:0/device:TPU:0");
  EXPECT_EQ(tpu_devices[1][1].host,
            "/job:worker/replica:0/task:0/device:CPU:0");
  EXPECT_EQ(tpu_devices[1][2].device,
            "/job:worker/replica:0/task:0/device:TPU:1");
  EXPECT_EQ(tpu_devices[1][2].host,
            "/job:worker/replica:0/task:0/device:CPU:0");

  auto& xla_device_assignment = tpu_device_assignment.xla_device_assignment;
  ASSERT_TRUE(xla_device_assignment.hasValue());
  EXPECT_EQ(xla_device_assignment->replica_count(), 2);
  EXPECT_EQ(xla_device_assignment->computation_count(), 3);
  ASSERT_EQ(xla_device_assignment->computation_devices_size(), 3);
  const auto& computation_device_0 =
      xla_device_assignment->computation_devices(0);
  ASSERT_EQ(computation_device_0.replica_device_ids_size(), 2);
  const auto& computation_device_1 =
      xla_device_assignment->computation_devices(1);
  ASSERT_EQ(computation_device_1.replica_device_ids_size(), 2);
  const auto& computation_device_2 =
      xla_device_assignment->computation_devices(2);
  ASSERT_EQ(computation_device_2.replica_device_ids_size(), 2);

  EXPECT_EQ(computation_device_0.replica_device_ids(0), 1);
  EXPECT_EQ(computation_device_0.replica_device_ids(1), 5);
  EXPECT_EQ(computation_device_1.replica_device_ids(0), 4);
  EXPECT_EQ(computation_device_1.replica_device_ids(1), 0);
  EXPECT_EQ(computation_device_2.replica_device_ids(0), 2);
  EXPECT_EQ(computation_device_2.replica_device_ids(1), 3);
}

TEST(TPURewriteDeviceUtilTest, TestGetDeviceCoordinates) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  auto device_assignment_attr = builder.getI64ArrayAttr({1, 2, 3});
  auto status_or_device_coodinates =
      GetDeviceCoordinates(device_assignment_attr);
  ASSERT_TRUE(status_or_device_coodinates.ok());
  auto device_coordinates = status_or_device_coodinates.ConsumeValueOrDie();
  EXPECT_EQ(device_coordinates[0], 1);
  EXPECT_EQ(device_coordinates[1], 2);
  EXPECT_EQ(device_coordinates[2], 3);
}

TEST(TPURewriteDeviceUtilTest, TestInvalidAttrForDeviceAssignmentDisallowed) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  auto device_assignment_attr = builder.getF32ArrayAttr({1.0, 2.0, 3.0});
  auto status_or_device_coodinates =
      GetDeviceCoordinates(device_assignment_attr);
  ASSERT_TRUE(!status_or_device_coodinates.ok());
  EXPECT_EQ(status_or_device_coodinates.status().error_message(),
            "bad 'device_assignment' attribute at index 0, not an int");
}

TEST(TPURewriteDeviceUtilTest, TestHasModelParallelismFalse) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  EXPECT_FALSE(HasModelParallelism(cluster));
}

TEST(TPURewriteDeviceUtilTest, TestHasModelParallelismTrue) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 5));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  EXPECT_TRUE(HasModelParallelism(cluster));
}

TEST(TPURewriteDeviceUtilTest,
     TestHasModelParallelismFalseMissingCoresPerReplicaAttr) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  EXPECT_FALSE(HasModelParallelism(cluster));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostFailDeviceMissingAttributes) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());
  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);

  mlir::TF::RuntimeDevices devices;
  std::string host_device;
  EXPECT_TRUE(mlir::failed(
      GetHostDeviceOutsideComputation(devices, cluster, &host_device)));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceFailMissingTopology) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  mlir::TF::RuntimeDevices runtime_devices;
  std::string host_device;
  EXPECT_TRUE(mlir::failed(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceFailMissingDeviceAssignment) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));

  mlir::TF::RuntimeDevices runtime_devices;
  std::string host_device;
  EXPECT_TRUE(mlir::failed(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceFailBadDeviceAssignment) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr,
                   builder.getStrArrayAttr(llvm::ArrayRef<llvm::StringRef>(
                       {"bad_device_assigment"})));

  mlir::TF::RuntimeDevices runtime_devices;
  std::string host_device;
  EXPECT_TRUE(mlir::failed(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceFailBadDeviceName) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());
  (*module_ref)
      ->setAttr("tf.devices",
                builder.getStrArrayAttr(
                    llvm::ArrayRef<llvm::StringRef>({"bad_device_name"})));

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  mlir::TF::RuntimeDevices runtime_devices;
  (void)GetDevicesFromOp(*module_ref, &runtime_devices);
  std::string host_device;
  EXPECT_TRUE(mlir::failed(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceTPUReplicate) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<llvm::StringRef, 4>>
      devices;
  auto replicate = builder.create<mlir::tf_device::ReplicateOp>(
      mlir::UnknownLoc::get(&context), /*num_replicas=*/2, devices,
      llvm::ArrayRef<std::pair<mlir::ValueRange, mlir::Type>>{},
      mlir::ValueRange{}, mlir::TypeRange{});
  builder.setInsertionPoint(&replicate.body().front(),
                            replicate.body().front().begin());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);

  mlir::TF::RuntimeDevices runtime_devices;
  std::string host_device;
  EXPECT_TRUE(mlir::succeeded(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
  EXPECT_EQ(host_device, kTPUReplicatedHost);
}

TEST(TPURewriteDeviceUtilTest, TestGetHostDeviceNotReplicated) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());
  (*module_ref)
      ->setAttr("tf.devices",
                builder.getStrArrayAttr(llvm::ArrayRef<llvm::StringRef>(
                    {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
                     "/job:localhost/replica:0/task:0/device:TPU:0",
                     "/job:worker/replica:0/task:0/device:CPU:0"})));

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->setAttr(kNumCoresPerReplicaAttr,
                   builder.getIntegerAttr(builder.getIntegerType(64), 1));
  cluster->setAttr(kTopologyAttr, builder.getStringAttr(""));
  cluster->setAttr(kDeviceAssignmentAttr, builder.getArrayAttr({}));

  mlir::TF::RuntimeDevices runtime_devices;
  (void)GetDevicesFromOp(*module_ref, &runtime_devices);
  std::string host_device;
  EXPECT_TRUE(mlir::succeeded(
      GetHostDeviceOutsideComputation(runtime_devices, cluster, &host_device)));
  EXPECT_EQ(host_device, "/job:localhost/replica:0/task:0/device:CPU:0");
}

TEST(TPURewriteDeviceUtilTest, TestIsTPUDevice) {
  EXPECT_TRUE(IsTPUDevice("/job:localhost/replica:0/task:0/device:TPU:0"));
  EXPECT_FALSE(IsTPUDevice("/job:localhost/replica:0/task:0/device:CPU:0"));
  EXPECT_FALSE(IsTPUDevice("INVALID_DEVICE"));
}

}  // anonymous namespace
}  // namespace tensorflow
