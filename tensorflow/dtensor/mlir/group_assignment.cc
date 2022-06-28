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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc() {
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

#include "tensorflow/dtensor/mlir/group_assignment.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

GroupAssignment::ReplicaToDeviceMap
GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(int num_slices,
                                                               int slice_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap");

  absl::flat_hash_map<ReplicaId, DeviceId> map;
  for (int i = 0; i < num_slices; ++i) {
    for (int j = 0; j < slice_size; ++j) {
      map[ReplicaId{i * slice_size + j}] = DeviceId{i, j};
    }
  }
  return ReplicaToDeviceMap(std::move(map));
}

GroupAssignment::ReplicaToDeviceMap::ReplicaToDeviceMap(
    absl::flat_hash_map<ReplicaId, DeviceId> map)
    : map_(std::move(map)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_1(mht_1_v, 226, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ReplicaToDeviceMap::ReplicaToDeviceMap");

  std::set<int> slice_ids;
  for (const auto& entry : map_) {
    slice_ids.insert(entry.second.slice_id);
  }
  CHECK_GT(slice_ids.size(), 0);                // Crash OK
  CHECK_EQ(map_.size() % slice_ids.size(), 0);  // Crash OK
  num_slices_ = slice_ids.size();
}

GroupAssignment::ReplicaGroups::ReplicaGroups(
    std::vector<std::vector<int>> replica_ids)
    : replica_ids_(std::move(replica_ids)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_2(mht_2_v, 241, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ReplicaGroups::ReplicaGroups");

  int n = replica_ids_.size();
  CHECK_GT(n, 0);  // Crash OK
  int g = replica_ids_.front().size();
  CHECK_GT(g, 0);  // Crash OK
  std::set<int> seen_replica_ids;
  for (std::vector<int>& group : replica_ids_) {
    CHECK_EQ(group.size(), g);  // Crash OK
    for (int replica_id : group) {
      CHECK_GE(replica_id, 0);  // Crash OK
      bool inserted = seen_replica_ids.insert(replica_id).second;
      CHECK(inserted);  // Crash OK
    }
  }
}

mlir::DenseIntElementsAttr GroupAssignment::ReplicaGroups::ToMLIR(
    mlir::MLIRContext& context) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_3(mht_3_v, 261, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ReplicaGroups::ToMLIR");

  auto shaped_type = mlir::RankedTensorType::get(
      {num_groups(), group_size()}, mlir::IntegerType::get(&context, 32));

  llvm::SmallVector<int32, 4> flat_replica_ids;
  flat_replica_ids.reserve(num_replica_ids());
  for (const std::vector<int>& group : replica_ids()) {
    flat_replica_ids.insert(flat_replica_ids.end(), group.begin(), group.end());
  }

  return mlir::DenseIntElementsAttr::get(shaped_type, flat_replica_ids);
}

std::string GroupAssignment::ReplicaGroups::ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_4(mht_4_v, 277, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ReplicaGroups::ToString");

  return strings::StrCat(
      "[",
      str_util::Join(replica_ids(), ", ",
                     [](std::string* str, const std::vector<int>& group) {
                       strings::StrAppend(str, "[", str_util::Join(group, ", "),
                                          "]");
                     }),
      "]");
}

StatusOr<GroupAssignment> GroupAssignment::FromMLIR(
    const mlir::DenseIntElementsAttr& group_assignment_attr,
    ReplicaToDeviceMap replica_to_device_map) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_5(mht_5_v, 293, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::FromMLIR");

  mlir::ShapedType shaped_type = group_assignment_attr.getType();
  if (!shaped_type.hasRank()) {
    return errors::InvalidArgument("group_assignment_attr must have a rank");
  }
  if (shaped_type.getRank() != 2) {
    return errors::InvalidArgument(
        "group_assignment_attr must have a rank of 2, got ",
        shaped_type.getRank());
  }
  llvm::ArrayRef<int64_t> shape = shaped_type.getShape();
  int num_groups = shape[0];
  if (num_groups <= 0) {
    return errors::InvalidArgument(
        "group_assignment_attr must have at least 1 group, got ", num_groups);
  }
  int group_size = shape[1];
  if (group_size <= 0) {
    return errors::InvalidArgument(
        "group_assignment_attr must have non-empty groups, got ", group_size,
        " replica IDs per group");
  }
  int num_replica_ids = num_groups * group_size;
  if (num_replica_ids != replica_to_device_map.num_cores()) {
    return errors::InvalidArgument("group_assignment_attr must have ",
                                   replica_to_device_map.num_cores(),
                                   " replica IDs, got ", num_replica_ids);
  }

  // Translate the flat group assignment to a 2D array.
  std::vector<std::vector<int>> replica_ids;
  replica_ids.resize(num_groups, std::vector<int>(group_size));
  std::set<int> seen_replica_ids;
  if (group_assignment_attr.getNumElements() != num_replica_ids) {
    return errors::InvalidArgument(
        "group_assignments_attr num elements was not equal to the number of "
        "replica ids.");
  }
  for (const auto& it :
       llvm::enumerate(group_assignment_attr.getValues<llvm::APInt>())) {
    int index = it.index();
    int replica_id = it.value().getSExtValue();

    // If all replica IDs are within this range and distinct, they must be a
    // permutation of [0, ..., num_replica_ids).
    if (replica_id < 0 || replica_id >= num_replica_ids) {
      return errors::InvalidArgument("Out of range replica ID: ", replica_id);
    }
    if (!seen_replica_ids.insert(replica_id).second) {
      return errors::InvalidArgument(
          "All replica IDs in group_assigment must be distinct, seeing ",
          replica_id, " more than once");
    }

    replica_ids[index / group_size][index % group_size] = replica_id;
  }

  GroupAssignment group_assignment(
      /*global=*/ReplicaGroups(std::move(replica_ids)),
      std::move(replica_to_device_map));
  TF_RETURN_IF_ERROR(group_assignment.GlobalToSlices());
  return group_assignment;
}

std::string GroupAssignment::ToString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_6(mht_6_v, 360, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::ToString");

  return strings::StrCat(
      "GroupAssignment global: ", global_.ToString(), "; hosts: ",
      hosts_.empty()
          ? "<none>"
          : str_util::Join(hosts_, ", ",
                           [](std::string* str, const ReplicaGroups& groups) {
                             strings::StrAppend(str, groups.ToString());
                           }),
      "; slices: ",
      slices_.empty()
          ? "<none>"
          : str_util::Join(slices_, ", ",
                           [](std::string* str, const ReplicaGroups& groups) {
                             strings::StrAppend(str, groups.ToString());
                           }));
}

bool GroupAssignment::IsWithinSlices() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_7(mht_7_v, 381, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::IsWithinSlices");

  // This function returns true iff no group in the global view gets split in
  // `GlobalToSlices`, i.e., the total group count remains the same.
  int total_num_groups = 0;
  for (int i = 0; i < num_slices(); i++) {
    total_num_groups += num_groups(i).ValueOrDie();
  }
  if (total_num_groups != num_groups()) return false;
  return total_num_groups == num_groups();
}

Status GroupAssignment::GlobalToSlices() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignmentDTcc mht_8(mht_8_v, 395, "", "./tensorflow/dtensor/mlir/group_assignment.cc", "GroupAssignment::GlobalToSlices");

  VLOG(2) << "Original group assignment: " << ToString();

  int num_slices = replica_to_device_map_.num_slices();
  if (num_slices == 0) {
    return errors::InvalidArgument("Unexpectedly empty replica_to_device_map.");
  }

  // For each replica group in global replica groups, divide its replicas based
  // on which slices they come from. Then, for each slice, collect subgroups
  // from every such division and form a new ReplicaGroup for that slice.
  std::vector<std::vector<std::vector<int>>> replica_groups_per_host;
  std::vector<std::vector<std::vector<int>>> replica_groups_per_slice;
  replica_groups_per_host.resize(num_slices, {});
  replica_groups_per_slice.resize(num_slices, {});

  for (const std::vector<int>& replica_group : replica_ids()) {
    std::vector<std::vector<int>> replica_group_divided_by_host;
    replica_group_divided_by_host.resize(num_slices, {});
    std::vector<std::vector<int>> replica_group_divided_by_slice;
    replica_group_divided_by_slice.resize(num_slices, {});

    for (int replica_id : replica_group) {
      // TODO(b/183426911): Use DeviceId::core_id in ReplicaGroup directly for
      // now. Integrate with device assignment with proper typing.
      DeviceId device_id = replica_to_device_map_.device_id(replica_id);
      replica_group_divided_by_host[device_id.slice_id].push_back(replica_id);
      replica_group_divided_by_slice[device_id.slice_id].push_back(
          device_id.core_id);
    }

    for (int i = 0; i < num_slices; ++i) {
      if (!replica_group_divided_by_host[i].empty()) {
        // Host meshes have the same global device and replica IDs as TPU
        // meshes. Let the first replica in every group do a host collective.
        replica_groups_per_host[i].push_back(
            std::vector<int>(1, replica_group_divided_by_host[i].front()));
      }
      if (!replica_group_divided_by_slice[i].empty()) {
        replica_groups_per_slice[i].push_back(
            std::move(replica_group_divided_by_slice[i]));
      }
    }
  }

  hosts_.reserve(num_slices);
  slices_.reserve(num_slices);
  for (int i = 0; i < num_slices; ++i) {
    hosts_.push_back(ReplicaGroups(std::move(replica_groups_per_host[i])));
    slices_.push_back(ReplicaGroups(std::move(replica_groups_per_slice[i])));
  }

  VLOG(2) << "Divided group assignment: " << ToString();
  return Status::OK();
}

}  // namespace dtensor
}  // namespace tensorflow
