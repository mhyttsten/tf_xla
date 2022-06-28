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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh() {
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


#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {

// Class that represents the device assignment for a set of XLA replicated
// computations. For R replicas and C computations, R * C devices are required
// execute the computation in parallel. The assigned device ids can be accessed
// by assignment(replica, computation).
class DeviceAssignment : public Array2D<int> {
 public:
  DeviceAssignment() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/computation_placer.h", "DeviceAssignment");
}
  DeviceAssignment(int replica_count, int computation_count)
      : Array2D<int>(replica_count, computation_count, -1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/service/computation_placer.h", "DeviceAssignment");

    CHECK_GT(replica_count, 0);
    CHECK_GT(computation_count, 0);
  }

  int replica_count() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_2(mht_2_v, 223, "", "./tensorflow/compiler/xla/service/computation_placer.h", "replica_count");
 return height(); }
  int computation_count() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_3(mht_3_v, 227, "", "./tensorflow/compiler/xla/service/computation_placer.h", "computation_count");
 return width(); }

  // The logical ID of a device is its (replica ID, computation ID) pair.
  struct LogicalID {
    int replica_id;
    int computation_id;
  };

  // Finds the (replica ID, computation ID) pair for the given device.
  StatusOr<LogicalID> LogicalIdForDevice(GlobalDeviceId device_id) const;
  // Finds the replica ID for the given device.
  StatusOr<int> ReplicaIdForDevice(GlobalDeviceId device_id) const;
  // Returns a map from device ID to logical ID. Querying this map is much more
  // efficient than `LogicalIdForDevice` if queried repeatedly.
  absl::flat_hash_map<GlobalDeviceId, LogicalID> GetDeviceToLogicalIdMap()
      const;

  // Protocol buffer serialization and deserialization.
  Status Serialize(DeviceAssignmentProto* proto) const;

  // Return a std::unique_ptr<DeviceAssignment> instead of a DeviceAssignment
  // directly because one of the supported TF platforms (mac) does not compile
  // due to a StatusOr of an incomplete type (DeviceAssignment).
  static StatusOr<std::unique_ptr<DeviceAssignment>> Deserialize(
      const DeviceAssignmentProto& proto);

  std::string ToString() const;
};

// A generic implementation of the XLA computation placer, which assigns device
// ids to a set of replicated computations.
class ComputationPlacer {
 public:
  ComputationPlacer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_4(mht_4_v, 263, "", "./tensorflow/compiler/xla/service/computation_placer.h", "ComputationPlacer");
}
  virtual ~ComputationPlacer() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_placerDTh mht_5(mht_5_v, 267, "", "./tensorflow/compiler/xla/service/computation_placer.h", "~ComputationPlacer");
}

  // Returns the device id assigned to the given replica and computation
  // instance for [replica_count x computation_count] setup. The returned device
  // id must match the assignment from PlaceReplicatedComputation().
  virtual StatusOr<int> DeviceId(int replica, int computation,
                                 int replica_count, int computation_count);

  // Returns the device ids assigned to a set of replicated computations, given
  // the number of replicas and the number of computations.
  virtual StatusOr<DeviceAssignment> AssignDevices(int replica_count,
                                                   int computation_count);

  using ComputationPlacerCreationFunction =
      std::unique_ptr<ComputationPlacer> (*)();

  // Registers a computation placer creation function for a particular platform.
  static void RegisterComputationPlacer(
      se::Platform::Id platform_id,
      ComputationPlacerCreationFunction creation_function);

  // Returns the computation placer singleton pointer if it is available for the
  // given platform, or an error status if it is not.
  static StatusOr<ComputationPlacer*> GetForPlatform(
      const se::Platform* platform);

 private:
  // The mutex that guards the platform-to-computation placer map.
  static absl::Mutex platform_computation_placer_mutex_;

  // State kept for each kind of ComputationPlacer. Registration functions set
  // up creation_function, and then we use that to lazily create "placer" the
  // first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<ComputationPlacer> placer;
    ComputationPlacerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to computation placer singleton.
  static std::map<se::Platform::Id, State>* GetPlatformComputationPlacers();

  ComputationPlacer(const ComputationPlacer&) = delete;
  ComputationPlacer& operator=(const ComputationPlacer&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_
