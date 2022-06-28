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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh() {
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


#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
using stream_executor::port::StatusOr;

extern const char* const kTPUReplicatedHost;
extern const char* const kNumCoresPerReplicaAttr;
extern const char* const kTopologyAttr;
extern const char* const kDeviceAssignmentAttr;

// A TPU device for execution alongside its associated host CPU device.
struct TPUDeviceAndHost {
  TPUDeviceAndHost() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h", "TPUDeviceAndHost");
}
  TPUDeviceAndHost(llvm::StringRef device, llvm::StringRef host)
      : device(device), host(host) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h", "TPUDeviceAndHost");
}

  std::string device;
  std::string host;
};

// TPU devices to be used for execution (e.g. devices for TPUExecute ops) and
// their associated host CPU devices (for outside compilation). They are ordered
// by `num_replicas` followed by `num_cores_per_replica`.
using TPUDevicesAndHosts =
    llvm::SmallVector<llvm::SmallVector<TPUDeviceAndHost, 8>, 8>;

// TPU compilation device, execution and associated host devices, and optionally
// execution device IDs. Execution device IDs are populated if `topology` and
// `device_assignment` are provided.
struct TPUDeviceAssignment {
  TPUDeviceAssignment(llvm::StringRef compilation_device,
                      TPUDevicesAndHosts&& tpu_devices)
      : compilation_device(compilation_device),
        tpu_devices(std::move(tpu_devices)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh mht_2(mht_2_v, 240, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h", "TPUDeviceAssignment");
}

  TPUDeviceAssignment(llvm::StringRef compilation_device,
                      TPUDevicesAndHosts&& tpu_devices,
                      xla::DeviceAssignmentProto&& xla_device_assignment)
      : compilation_device(compilation_device),
        tpu_devices(std::move(tpu_devices)),
        xla_device_assignment(std::move(xla_device_assignment)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_rewrite_device_utilDTh mht_3(mht_3_v, 250, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h", "TPUDeviceAssignment");
}

  std::string compilation_device;
  TPUDevicesAndHosts tpu_devices;
  llvm::Optional<xla::DeviceAssignmentProto> xla_device_assignment;
};

// Extracts device coordinates from a device assignment attribute on an op.
StatusOr<llvm::SmallVector<int64_t, 8>> GetDeviceCoordinates(
    mlir::ArrayAttr device_assignment_attr);

// Finds the TPU compilation device and execution devices from `devices` for a
// TPU computation subgraph. Compilation device is determined from looking up
// all TPU_SYSTEM:0 devices and choosing the CPU device associated to the first
// TPU_SYSTEM device sorted lexicographically by replica and task. Execution
// devices are determined by looking up all TPU devices associated with each
// TPU_SYSTEM:0 device found, alongside associated `topology_attr` and
// `device_assignment_attr`. If `topology_attr` not an empty string (parsable to
// TopologyProto), `device_assignment_attr` must not be empty also. When
// `topology_attr` and `device_assignment_attr` are not empty, a general device
// assignment based on those two attributes are used. Otherwise when
// `topology_attr` and `device_assignment_attr` are empty, a full mesh device
// assignment is used instead. A failure will be returned if it is not possible
// (e.g. invalid devices or invalid parameters).
//
//
// For example, for `devices`:
//   {
//     /job:localhost/replica:0/task:0/device:CPU:0,
//     /job:worker/replica:0/task:0/device:CPU:0,
//     /job:worker/replica:0/task:0/device:TPU_SYSTEM:0,
//     /job:worker/replica:0/task:0/device:TPU:0,
//     /job:worker/replica:0/task:0/device:TPU:1,
//     /job:worker/replica:0/task:0/device:TPU:2,
//     /job:worker/replica:0/task:0/device:TPU:3,
//     /job:worker/replica:0/task:1/device:CPU:0,
//     /job:worker/replica:0/task:1/device:TPU_SYSTEM:0,
//     /job:worker/replica:0/task:1/device:TPU:0,
//     /job:worker/replica:0/task:1/device:TPU:1,
//     /job:worker/replica:0/task:1/device:TPU:2,
//     /job:worker/replica:0/task:1/device:TPU:3
//   }
//
//
// With the following parameters (full mesh device assignment):
//   `num_replicas` = 8
//   `num_cores_per_replica` = 1
//   `topology_attr` = ""
//   `device_assignment_attr` = {}
//
// The `compilation_device` will be:
//   /job:worker/replica:0/task:0/device:CPU:0
//
// `execution_devices` will be:
//   {
//     {
//       /job:worker/replica:0/task:0/device:TPU:0
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:1
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:2
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:3
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:0
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:1
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:2
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:3
//     }
//   }
//
// and `xla_device_assignment` will not be set.
//
//
// With the following parameters (general device assignment):
//   `num_replicas` = 4
//   `num_cores_per_replica` = 2
//   `topology_attr` (in proto debug string format) =
//     {
//       mesh_shape: 2
//       mesh_shape: 2
//       mesh_shape: 2
//       num_tasks: 2
//       num_tpu_devices_per_task: 4
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//     }
//   `device_assignment` =
//     {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1}
//
// The `compilation_device` will be:
//   /job:worker/replica:0/task:0/device:CPU:0
//
// `execution_devices` will be:
//   {
//     {
//       "/job:worker/replica:0/task:0/device:TPU:0",
//       "/job:worker/replica:0/task:1/device:TPU:3"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:1",
//       "/job:worker/replica:0/task:1/device:TPU:2"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:3",
//       "/job:worker/replica:0/task:1/device:TPU:0"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:2",
//       "/job:worker/replica:0/task:1/device:TPU:1"
//     }
//   }
//
// and `xla_device_assignment` will be:
//   {
//     replica_count: 4
//     computation_count: 2
//     computation_devices {
//       replica_device_ids: 0
//       replica_device_ids: 4
//       replica_device_ids: 2
//       replica_device_ids: 6
//     }
//     computation_devices {
//       replica_device_ids: 1
//       replica_device_ids: 5
//       replica_device_ids: 3
//       replica_device_ids: 7
//     }
//   }
StatusOr<TPUDeviceAssignment> GetTPUCompilationAndExecutionDevices(
    llvm::ArrayRef<DeviceNameUtils::ParsedName> devices, int num_replicas,
    int num_cores_per_replica, llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr);

// Virtual device is used for evice assignment for executing ops on a specified
// logical core.
std::string GetDeviceAliasForLogicalCore(int core_index);

// Returns true if cluster contains model parallelism based on
// `num_cores_per_replica_attribute`. Otherwise returns false.
bool HasModelParallelism(mlir::tf_device::ClusterOp cluster);

// Parses TPU compilation and execution devices from a TPU cluster and returns
// the host device for the head and tail computations. If the TPU computation is
// replicated, kTPUReplicatedHost is returned instead.
mlir::LogicalResult GetHostDeviceOutsideComputation(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string* host_device);

// Checks if a device string is a TPU device.
bool IsTPUDevice(llvm::StringRef device);

// Checks if a device string is a TPU replicated core device.
bool IsTPUReplicatedCore(llvm::StringRef device);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
