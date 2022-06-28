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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utilsDTcc() {
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

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

// Match the instruction to a reduction kind. We can represent and/or of pred as
// min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
absl::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo) {
  PrimitiveType type = hlo->shape().element_type();
  switch (hlo->opcode()) {
    case HloOpcode::kAdd:
      return ReductionKind::SUM;
    case HloOpcode::kMultiply:
      return ReductionKind::PRODUCT;
    case HloOpcode::kMinimum:
      return ReductionKind::MIN;
    case HloOpcode::kMaximum:
      return ReductionKind::MAX;
    case HloOpcode::kAnd:
      return type == PRED ? absl::optional<ReductionKind>(ReductionKind::MIN)
                          : absl::nullopt;
    case HloOpcode::kOr:
      return type == PRED ? absl::optional<ReductionKind>(ReductionKind::MAX)
                          : absl::nullopt;
    default:
      return absl::nullopt;
  }
}

absl::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation) {
  namespace m = match;
  const HloInstruction* root = computation->root_instruction();
  auto kind = MatchReductionInstruction(root);
  if (kind && !Match(root, m::Op()
                               .WithBinaryOperandsAnyOrder(m::Parameter(0),
                                                           m::Parameter(1))
                               .WithShape(m::Shape().IsEffectiveScalar()))) {
    kind = absl::nullopt;
  }
  return kind;
}

StatusOr<std::vector<int>> GetParticipatingIDs(
    int current_id, absl::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups) {
  // Empty replica_groups() means that all replicas participate.
  if (groups.empty()) {
    TF_RET_CHECK(total_participant_count.has_value());
    std::vector<int> all_participants(*total_participant_count);
    absl::c_iota(all_participants, 0);
    return all_participants;
  }

  // Figure out the other replicas that go together with this one.
  absl::optional<ReplicaGroup> group;
  for (const ReplicaGroup& g : groups) {
    if (absl::c_linear_search(g.replica_ids(), current_id)) {
      TF_RET_CHECK(!group.has_value())
          << "ID " << current_id << " appears twice in replica groups";
      group = g;
    }
  }
  TF_RET_CHECK(group.has_value())
      << "ID " << current_id << " doesn't appear in replica groups";
  return std::vector<int>(group->replica_ids().begin(),
                          group->replica_ids().end());
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, absl::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (!use_global_device_ids.has_value() || !*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplica;
    } else {
      return InvalidArgument(
          "Invalid combination of has_channel_id and use_global_device_ids");
    }
  } else {
    if (!use_global_device_ids.has_value()) {
      return CollectiveOpGroupMode::kCrossPartition;
    } else if (!*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplicaAndPartition;
    } else {
      return CollectiveOpGroupMode::kFlattenedID;
    }
  }
}

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utilsDTcc mht_0(mht_0_v, 284, "", "./tensorflow/compiler/xla/service/collective_ops_utils.cc", "CollectiveOpGroupModeToString");

  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return "kCrossReplica";
    case CollectiveOpGroupMode::kCrossPartition:
      return "kCrossPartition";
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return "kCrossReplicaAndPartition";
    case CollectiveOpGroupMode::kFlattenedID:
      return "kFlattenedID";
  }
}

StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const DeviceAssignment& device_assignment,
                              absl::Span<const ReplicaGroup> replica_groups,
                              CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  std::vector<ReplicaGroup> participating_replica_groups =
      SpanToVector(replica_groups);

  // If replica groups are empty, assume a group with all replicas.
  if (replica_groups.empty()) {
    if (group_mode == CollectiveOpGroupMode::kFlattenedID) {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";
    }

    int total_participant_count;
    if (group_mode == CollectiveOpGroupMode::kCrossPartition) {
      // replica group are partition ids.
      total_participant_count = partition_count;
    } else {
      // replica group are replica ids.
      total_participant_count = replica_count;
    }

    ReplicaGroup replica_group = ReplicaGroup();
    for (int id = 0; id < total_participant_count; id++) {
      replica_group.add_replica_ids(id);
    }
    participating_replica_groups.push_back(replica_group);
  }

  std::vector<std::vector<GlobalDeviceId>> groups;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      for (const auto& replica_group : participating_replica_groups) {
        // replica_group contains replica id, participants contains all
        // replica_group's replica_ids for the current partition.
        for (int partition_id = 0; partition_id < partition_count;
             partition_id++) {
          std::vector<GlobalDeviceId> participants;
          participants.reserve(replica_group.replica_ids().size());

          for (int replica_id : replica_group.replica_ids()) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
          groups.push_back(participants);
        }
      }
      return groups;
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        // replica_group contains partition id, participants contains all
        // replica_group's partition_ids for the current replica_id.
        for (int replica_id = 0; replica_id < replica_count; replica_id++) {
          std::vector<GlobalDeviceId> participants;
          participants.reserve(replica_group.replica_ids().size());

          for (int partition_id : replica_group.replica_ids()) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
          groups.push_back(participants);
        }
      }
      return groups;
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        std::vector<GlobalDeviceId> participants;
        participants.reserve(replica_group.replica_ids().size() *
                             partition_count);

        // replica_group contains replica id, participants contains all
        // replica_group's replica_ids for all partitions.
        for (int replica_id : replica_group.replica_ids()) {
          for (int partition_id = 0; partition_id < partition_count;
               partition_id++) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
        }
        groups.push_back(participants);
      }
      return groups;
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      for (const auto& replica_group : participating_replica_groups) {
        std::vector<GlobalDeviceId> participants;
        participants.reserve(replica_group.replica_ids().size());

        for (int flattened_id : replica_group.replica_ids()) {
          // Map from flattened id back to replica_id, partition_id.
          int replica_id = flattened_id / partition_count;
          int partition_id = flattened_id % partition_count;
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
        groups.push_back(participants);
      }
      return groups;
    }
  }
}

StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      device_assignment.LogicalIdForDevice(device_id));
  int current_replica_id = logical_id.replica_id;
  int current_partition_id = logical_id.computation_id;

  std::vector<GlobalDeviceId> participants;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      // This is a cross replica operation. replica group contains replica id.
      // use current replica id to find the set of participating replicas. If
      // replica groups are empty, assume a group with all replicas.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(current_replica_id, replica_count,
                                              replica_groups));

      // The set of participating devices is the replicas from the current
      // partition.
      participants.reserve(participating_replicas.size());
      for (int replica_id : participating_replicas) {
        participants.emplace_back(
            device_assignment(replica_id, current_partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossPartition: {
      // replica_groups contain partition_id, group contains all partitions for
      // the current replica.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_partitions,
                          GetParticipatingIDs(current_partition_id,
                                              partition_count, replica_groups));
      participants.reserve(participating_partitions.size());
      for (int partition_id : participating_partitions) {
        participants.emplace_back(
            device_assignment(current_replica_id, partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      // replica_groups contain replica_ids. Group contains replicas for all
      // partitions.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(current_replica_id, replica_count,
                                              replica_groups));
      participants.reserve(participating_replicas.size() * partition_count);
      for (int replica_id : participating_replicas) {
        for (int partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
      }
      return participants;
    }

    case CollectiveOpGroupMode::kFlattenedID: {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";

      int current_flattened_id =
          current_replica_id * partition_count + current_partition_id;

      // Find participants based on flattened id. replica_groups cannot be empty
      // so no need to pass in total_participant_count.
      TF_ASSIGN_OR_RETURN(
          std::vector<int> participating_flattened_ids,
          GetParticipatingIDs(current_flattened_id,
                              /*total_participant_count=*/absl::nullopt,
                              replica_groups));

      participants.reserve(participating_flattened_ids.size());
      for (int flattened_id : participating_flattened_ids) {
        // Map from flattened id back to replica_id, partition_id.
        int replica_id = flattened_id / partition_count;
        int partition_id = flattened_id % partition_count;
        participants.emplace_back(device_assignment(replica_id, partition_id));
      }
      return participants;
    }
  }
}

bool ReplicaGroupsOrthogonal(absl::Span<const ReplicaGroup> first,
                             absl::Span<const ReplicaGroup> second) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utilsDTcc mht_1(mht_1_v, 501, "", "./tensorflow/compiler/xla/service/collective_ops_utils.cc", "ReplicaGroupsOrthogonal");

  if (first.size() != second[0].replica_ids_size()) {
    return false;
  }
  if (first[0].replica_ids_size() != second.size()) {
    return false;
  }
  for (int64_t i = 0; i < first.size(); ++i) {
    for (int64_t j = 0; j < first[i].replica_ids_size(); ++j) {
      if (first[i].replica_ids(j) != second[j].replica_ids(i)) {
        return false;
      }
    }
  }
  return true;
}

}  // end namespace xla
