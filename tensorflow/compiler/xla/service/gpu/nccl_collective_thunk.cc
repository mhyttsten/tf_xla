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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

namespace xla {
namespace gpu {

// This file runs collective ops (i.e. ops that communicate between multiple
// GPUs) using NCCL.
//
// Here's a high-level overview of how running an op works.
//
//  - Multiple threads call ExecuteOnStream.
//  - All threads that "go together" (i.e. are participating in the "same"
//    collective op) choose the same Rendezvous object from a global map.
//  - Once all threads have arrived at the Rendezvous, we know exactly which
//    GPUs are participating in the op, so we get or create a NcclClique
//    containing those GPUs.
//  - We perform the NCCL operation using the clique.

NcclCollectiveConfig::NcclCollectiveConfig() = default;
NcclCollectiveConfig::NcclCollectiveConfig(NcclCollectiveConfig&&) = default;
NcclCollectiveConfig::~NcclCollectiveConfig() = default;
NcclCollectiveConfig& NcclCollectiveConfig::operator=(NcclCollectiveConfig&&) =
    default;

// Returns if the collective communication operation is degenerate because all
// the groups formed by the operation are singleton. A given op can be
// degenerate under several conditions, corresponding to the modes supported
// in GetParticipatingDevices().
//   1. no channel id, use_global_device_ids = false:
//         degenerate if replica_groups are singleton, or groups empty and
//         replica_count == 1.
//   2. channel_id is set, use_global_device_ids = false:
//         degenerate if replica_groups are singleton and num_partitions == 1,
//         or groups empty and num_replicas == 1 && num_partitions == 1.
//   3. channel_id is set, use_global_device_ids = true (flattened-ids):
//         degenerate if replica_groups are singleton (groups cannot be empty).
//   4. no channel_id, no use_global_device_ids:
//         identical to 1.
//   5. channel_id is set, no use_global_device_ids:
//         degenerate if replica_groups are singleton or group emty and
//         num_partitions == 1 (since replica groups contain partition ids).
//
bool NcclCollectiveConfig::IsDegenerate(int64_t replica_count,
                                        int64_t partition_count) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc mht_0(mht_0_v, 244, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.cc", "NcclCollectiveConfig::IsDegenerate");

  bool groups_empty = replica_groups.empty();

  // check if all replica_groups are singleton. If not, then the operation is
  // not degenerate.
  bool all_groups_singleton =
      !groups_empty &&
      absl::c_all_of(replica_groups, [](const ReplicaGroup& group) {
        return group.replica_ids_size() == 1;
      });

  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return all_groups_singleton || (groups_empty && replica_count == 1);
    case CollectiveOpGroupMode::kCrossPartition:
      return all_groups_singleton || (groups_empty && partition_count == 1);
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return (all_groups_singleton && partition_count == 1) ||
             (groups_empty && replica_count == 1 && partition_count == 1);
    case CollectiveOpGroupMode::kFlattenedID:
      CHECK(!groups_empty)
          << "replica groups cannot be empty if use_global_device_ids = true";
      return all_groups_singleton;
    default:
      CHECK(0) << "Invalid collective op mode";
      return false;
  }
}

/* static */ bool NcclCollectiveThunk::NcclIsEnabled() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc mht_1(mht_1_v, 276, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.cc", "NcclCollectiveThunk::NcclIsEnabled");

#if XLA_ENABLE_XCCL
  return true;
#else
  return false;
#endif
}

Status NcclCollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc mht_2(mht_2_v, 287, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.cc", "NcclCollectiveThunk::ExecuteOnStream");

#if XLA_ENABLE_XCCL
  VLOG(1) << absl::StreamFormat("Starting %s.", Thunk::KindToString(kind()));
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              config().replica_groups, config().group_mode));

  if (IsGlobalNcclConfig() &&
      (participants.size() != params.device_assn->replica_count())) {
    return InvalidArgument(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  auto it = absl::c_find(participants, global_device_id);
  TF_RET_CHECK(it != participants.end());
  int rank = it - participants.begin();

  OpId op_id(config().op_id);
  size_t num_local_participants = GetNumLocalParticipants(
      participants, /*local_devices=*/params.gpu_global_device_ids);

  bool is_local = participants.size() == num_local_participants;
  TF_ASSIGN_OR_RETURN(
      const NcclUniqueIdCallback* unique_id_callback,
      GetNcclUniqueIdCallback(params.nccl_unique_id_callback, is_local));

  se::StreamExecutor* executor = params.stream->parent();
  se::gpu::ScopedActivateExecutorContext scoped_context(executor);

  TF_ASSIGN_OR_RETURN(
      NcclComm::Lock comm,
      AcquireNcclComm(params.run_id, op_id, std::move(participants),
                      num_local_participants, *unique_id_callback, rank));

  TF_RETURN_IF_ERROR(RunNcclCollective(params, *comm));

  // Block host on the first call to ensure that all devices have allocated the
  // required buffers for their communicators before allowing any device to
  // continue enqueuing operations. Otherwise, the allocations can cause
  // deadlock in the CUDA driver (b/215649390).
  if (first_call_to_execute_) {
    TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
    first_call_to_execute_ = false;
  }
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

std::string NcclCollectiveThunk::GetDeviceString(
    const ExecuteParams& params) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc mht_3(mht_3_v, 348, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.cc", "NcclCollectiveThunk::GetDeviceString");

  int device_ordinal = params.stream->parent()->device_ordinal();
  GlobalDeviceId global_device_id = params.GetGlobalDeviceId().ValueOrDie();
  DeviceAssignment::LogicalID logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id).ValueOrDie();
  return absl::StrFormat("(r%d, p%d) : GlobalID %d, ord %d",
                         logical_id.replica_id, logical_id.computation_id,
                         global_device_id.value(), device_ordinal);
}

bool IsTypeSupportedByNccl(PrimitiveType element_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTcc mht_4(mht_4_v, 361, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.cc", "IsTypeSupportedByNccl");

  switch (element_type) {
    case S8:
    case PRED:
    case U8:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case BF16:
#endif
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
