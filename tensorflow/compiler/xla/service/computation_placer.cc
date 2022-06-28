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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/computation_placer.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

using absl::StrAppend;
using absl::StrCat;

namespace xla {

StatusOr<DeviceAssignment::LogicalID> DeviceAssignment::LogicalIdForDevice(
    GlobalDeviceId device_id) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::LogicalIdForDevice");

  absl::optional<DeviceAssignment::LogicalID> logical_id;
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      if ((*this)(r, c) == device_id.value()) {
        if (logical_id.has_value()) {
          return InternalError(
              "Device %d appears twice in DeviceAssignment: %s",
              device_id.value(), ToString());
        }
        logical_id.emplace(DeviceAssignment::LogicalID{r, c});
      }
    }
  }
  if (logical_id.has_value()) {
    return *logical_id;
  } else {
    return InternalError("Device %d doesn't appear in DeviceAssignment: %s",
                         device_id.value(), ToString());
  }
}

StatusOr<int> DeviceAssignment::ReplicaIdForDevice(
    GlobalDeviceId device_id) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::ReplicaIdForDevice");

  TF_ASSIGN_OR_RETURN(const LogicalID logical_id,
                      LogicalIdForDevice(device_id));
  return logical_id.replica_id;
}

absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
DeviceAssignment::GetDeviceToLogicalIdMap() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::GetDeviceToLogicalIdMap");

  absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
      device_to_logical_id;
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      GlobalDeviceId device_id((*this)(r, c));
      device_to_logical_id[device_id] = DeviceAssignment::LogicalID{r, c};
    }
  }
  return device_to_logical_id;
}

Status DeviceAssignment::Serialize(DeviceAssignmentProto* proto) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::Serialize");

  proto->set_replica_count(replica_count());
  proto->set_computation_count(computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    DeviceAssignmentProto::ComputationDevice* computation_device =
        proto->add_computation_devices();
    for (int replica = 0; replica < replica_count(); ++replica) {
      computation_device->add_replica_device_ids((*this)(replica, computation));
    }
  }
  return Status::OK();
}

/* static */ StatusOr<std::unique_ptr<DeviceAssignment>>
DeviceAssignment::Deserialize(const DeviceAssignmentProto& proto) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_4(mht_4_v, 283, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::Deserialize");

  TF_RET_CHECK(proto.computation_devices_size() == proto.computation_count());
  if (proto.replica_count() <= 0 || proto.computation_count() <= 0) {
    return InvalidArgument(
        "Invalid device assignment topology: replica_count=%d, "
        "computation_count=%d",
        proto.replica_count(), proto.computation_count());
  }
  auto assignment = absl::make_unique<DeviceAssignment>(
      proto.replica_count(), proto.computation_count());
  for (int computation = 0; computation < proto.computation_count();
       ++computation) {
    const auto& computation_device = proto.computation_devices(computation);
    TF_RET_CHECK(computation_device.replica_device_ids_size() ==
                 proto.replica_count());
    for (int replica = 0; replica < proto.replica_count(); ++replica) {
      (*assignment)(replica, computation) =
          computation_device.replica_device_ids(replica);
    }
  }
  return std::move(assignment);
}

std::string DeviceAssignment::ToString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_5(mht_5_v, 309, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "DeviceAssignment::ToString");

  std::string output = StrCat("Computations: ", computation_count(),
                              " Replicas: ", replica_count(), "\n");
  for (int computation = 0; computation < computation_count(); ++computation) {
    StrAppend(&output, "Computation ", computation, ": ");
    for (int replica = 0; replica < replica_count(); ++replica) {
      StrAppend(&output, operator()(replica, computation), " ");
    }
    StrAppend(&output, "\n");
  }
  return output;
}

StatusOr<int> ComputationPlacer::DeviceId(int replica, int computation,
                                          int replica_count,
                                          int computation_count) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_6(mht_6_v, 327, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "ComputationPlacer::DeviceId");

  TF_RET_CHECK(replica < replica_count);
  TF_RET_CHECK(computation < computation_count);

  return computation * replica_count + replica;
}

StatusOr<DeviceAssignment> ComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_7(mht_7_v, 338, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "ComputationPlacer::AssignDevices");

  DeviceAssignment assignment(replica_count, computation_count);
  for (int replica = 0; replica < replica_count; ++replica) {
    for (int computation = 0; computation < computation_count; ++computation) {
      TF_ASSIGN_OR_RETURN(
          int device_id,
          DeviceId(replica, computation, replica_count, computation_count));
      assignment(replica, computation) = device_id;
    }
  }
  return std::move(assignment);
}

/* static */ void ComputationPlacer::RegisterComputationPlacer(
    se::Platform::Id platform_id,
    ComputationPlacerCreationFunction creation_function) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_8(mht_8_v, 356, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "ComputationPlacer::RegisterComputationPlacer");

  absl::MutexLock lock(&ComputationPlacer::platform_computation_placer_mutex_);
  auto* computation_placers = GetPlatformComputationPlacers();
  CHECK(computation_placers->find(platform_id) == computation_placers->end());
  (*computation_placers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<ComputationPlacer*> ComputationPlacer::GetForPlatform(
    const se::Platform* platform) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_9(mht_9_v, 367, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "ComputationPlacer::GetForPlatform");

  absl::MutexLock lock(&ComputationPlacer::platform_computation_placer_mutex_);
  auto* computation_placers = GetPlatformComputationPlacers();

  auto it = computation_placers->find(platform->id());
  if (it == computation_placers->end()) {
    return NotFound(
        "could not find registered computation placer for platform %s -- check "
        "target linkage",
        platform->Name());
  }

  if (it->second.placer == nullptr) {
    // Lazily create the computation placer the first time it is needed.
    it->second.placer = (*it->second.creation_function)();
  }

  return it->second.placer.get();
}

/* static */ absl::Mutex ComputationPlacer::platform_computation_placer_mutex_(
    absl::kConstInit);

/* static */ std::map<se::Platform::Id, ComputationPlacer::State>*
ComputationPlacer::GetPlatformComputationPlacers() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_10(mht_10_v, 394, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "ComputationPlacer::GetPlatformComputationPlacers");

  static auto* r = new std::map<se::Platform::Id, ComputationPlacer::State>;
  return r;
}

}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool InitModule() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTcc mht_11(mht_11_v, 408, "", "./tensorflow/compiler/xla/service/computation_placer.cc", "InitModule");

  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::host::kHostPlatformId, &CreateComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::cuda::kCudaPlatformId, &CreateComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::rocm::kROCmPlatformId, &CreateComputationPlacer);
  return true;
}
static bool module_initialized = InitModule();
