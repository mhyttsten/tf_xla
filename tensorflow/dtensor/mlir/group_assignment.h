/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_
#define TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_
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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh() {
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


#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

// Arranges all replica IDs in a DTensor mesh in groups, used as an attribute
// on collective operations.
//
// A group assignment has two views:
//
// - The global mesh view contains replica IDs from all participant TPU slices.
//   These replica IDs are identical to global device IDs in a DTensor mesh.
// - The local slice view contains per-slice device IDs understood and used by
//   the TPU runtime on each slice. These device IDs are used to set replica
//   IDs on each slice.
//
// Some notable common cases:
//
// - In a single-slice case, `slice_size` is set to the actual slice size
//   (e.g., 32 for 4x4 DF). The global and local views are identical.
// - In a special topology case, `slice_size` is set to 8.
// - In a multi-topology case, `slice_size` is set to the size of a single
// topology.
//   All topologies must have the same size.
class GroupAssignment {
 public:
  using ReplicaId = int;

  struct DeviceId {
   public:
    int slice_id;
    int core_id;  // within `slice_id`
  };

  // Maps global replica IDs to local device IDs consisting of a slice ID and a
  // core-on-slice ID.
  class ReplicaToDeviceMap {
   public:
    // Creates a default map that orders devices according to TF task IDs
    // followed by device ordinals.
    static ReplicaToDeviceMap DefaultReplicaToDeviceMap(int num_slices,
                                                        int slice_size);

    // Constructs a map directly, checking it's valid.
    explicit ReplicaToDeviceMap(absl::flat_hash_map<ReplicaId, DeviceId> map);

    int num_slices() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_0(mht_0_v, 246, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_slices");
 return num_slices_; }
    int num_cores() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_1(mht_1_v, 250, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_cores");
 return map_.size(); }
    DeviceId device_id(ReplicaId replica_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_2(mht_2_v, 254, "", "./tensorflow/dtensor/mlir/group_assignment.h", "device_id");
 return map_[replica_id]; }

   private:
    absl::flat_hash_map<ReplicaId, DeviceId> map_;
    int num_slices_;
  };

  // Creates a group assignment by converting from an MLIR attribute.
  static StatusOr<GroupAssignment> FromMLIR(
      const mlir::DenseIntElementsAttr& group_assignment_attr,
      ReplicaToDeviceMap replica_to_device_map);

  // Creates an MLIR attribute using the global view.
  mlir::DenseIntElementsAttr GlobalToMLIR(mlir::MLIRContext& context) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_3(mht_3_v, 270, "", "./tensorflow/dtensor/mlir/group_assignment.h", "GlobalToMLIR");

    return global_.ToMLIR(context);
  }

  // Creates an MLIR attribute for a particular slice.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  StatusOr<mlir::DenseIntElementsAttr> SliceToMLIR(mlir::MLIRContext& context,
                                                   int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].ToMLIR(context);
  }

  // Returns a string representation for debugging.
  std::string ToString() const;

  // Returns true if every group in the global view only has replica IDs from
  // the same slice.
  bool IsWithinSlices() const;

  // Returns the number of slices in the local view.
  int num_slices() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_4(mht_4_v, 294, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_slices");
 return slices_.size(); }

  // These methods return attributes of the global view.
  int num_groups() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_5(mht_5_v, 300, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_groups");
 return global_.num_groups(); }
  int group_size() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_6(mht_6_v, 304, "", "./tensorflow/dtensor/mlir/group_assignment.h", "group_size");
 return global_.group_size(); }
  int num_replica_ids() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_7(mht_7_v, 308, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_replica_ids");
 return global_.num_replica_ids(); }
  const std::vector<std::vector<int>>& replica_ids() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_8(mht_8_v, 312, "", "./tensorflow/dtensor/mlir/group_assignment.h", "replica_ids");

    return global_.replica_ids();
  }

  // These methods return attributes of a particular slice.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  StatusOr<int> num_groups(int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].num_groups();
  }
  StatusOr<int> group_size(int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].group_size();
  }
  const std::vector<std::vector<int>>& replica_ids(int slice_id) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_9(mht_9_v, 331, "", "./tensorflow/dtensor/mlir/group_assignment.h", "replica_ids");

    return slices_[slice_id].replica_ids();
  }

  // Returns the replica groups for collectives running on a particular host.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  const std::vector<std::vector<int>>& host_replica_ids(int slice_id) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_10(mht_10_v, 340, "", "./tensorflow/dtensor/mlir/group_assignment.h", "host_replica_ids");

    return hosts_[slice_id].replica_ids();
  }

 private:
  // Groups of consecutive replica IDs starting at 0.
  class ReplicaGroups {
   public:
    // Creates an object, enforcing the requirements on `replica_ids_`.
    explicit ReplicaGroups(std::vector<std::vector<int>> replica_ids);

    mlir::DenseIntElementsAttr ToMLIR(mlir::MLIRContext& context) const;

    std::string ToString() const;

    int num_groups() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_11(mht_11_v, 358, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_groups");
 return replica_ids_.size(); }
    int group_size() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_12(mht_12_v, 362, "", "./tensorflow/dtensor/mlir/group_assignment.h", "group_size");
 return replica_ids_.front().size(); }
    int num_replica_ids() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_13(mht_13_v, 366, "", "./tensorflow/dtensor/mlir/group_assignment.h", "num_replica_ids");
 return num_groups() * group_size(); }
    const std::vector<std::vector<int>>& replica_ids() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_14(mht_14_v, 370, "", "./tensorflow/dtensor/mlir/group_assignment.h", "replica_ids");

      return replica_ids_;
    }

   private:
    // N groups of replica IDs, N > 0. All groups have the same size G, G > 0.
    // All replica IDs are distinct values >= 0;
    std::vector<std::vector<int>> replica_ids_;  // replica ID order matters
  };

  // Creates an object but leaves `slices_` empty. `GlobalToSlices` should be
  // called next to fill in `slices_`.
  explicit GroupAssignment(ReplicaGroups global,
                           ReplicaToDeviceMap replica_to_device_map)
      : global_(std::move(global)),
        replica_to_device_map_(std::move(replica_to_device_map)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTh mht_15(mht_15_v, 388, "", "./tensorflow/dtensor/mlir/group_assignment.h", "GroupAssignment");
}

  // Divides the global view along slice boundaries and fill in the slice view.
  Status GlobalToSlices();

  ReplicaGroups global_;
  std::vector<ReplicaGroups> hosts_;   // sorted by increasing slice ID
  std::vector<ReplicaGroups> slices_;  // sorted by increasing slice ID
  ReplicaToDeviceMap replica_to_device_map_;
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_
