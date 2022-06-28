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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"

#include <map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

/*static*/ NcclCollectivePermuteConfig
NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig(
    mlir::lmhlo::CollectivePermuteOp op, int64_t replica_count,
    int64_t partition_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.cc", "NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig");

  NcclCollectivePermuteConfig config;

  config.operand_count = 1;
  const Shape shape = GetShape(op.operand());
  config.operand_element_type.push_back(shape.element_type());
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetGroupMode(op);

  // With a collective permute, all execution instances together form one
  // replica group.
  const int64_t num_participants =
      config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? replica_count
          : partition_count;
  config.replica_groups.emplace_back();
  ReplicaGroup& replica_group = config.replica_groups.front();
  for (int i = 0; i < num_participants; ++i) {
    replica_group.add_replica_ids(i);
  }

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
      ConvertNx2Attribute(op.source_target_pairs()).ValueOrDie();

  for (const std::pair<int64_t, int64_t>& source_target : source_target_pairs) {
    int64_t source = source_target.first;
    int64_t target = source_target.second;

    config.id_to_source_target.insert({target, {}}).first->second.source =
        source;
    config.id_to_source_target.insert({source, {}}).first->second.target =
        target;
  }

  return config;
}

// The collective permute is degenerate if all source-target pairs are identity,
// and all the IDs appear in the list.
/*static*/ bool NcclCollectivePermuteThunk::IsDegenerate(
    mlir::lmhlo::CollectivePermuteOp op, int64_t replica_count,
    int64_t partition_count) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc mht_1(mht_1_v, 254, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.cc", "NcclCollectivePermuteThunk::IsDegenerate");

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
      ConvertNx2Attribute(op.source_target_pairs()).ValueOrDie();
  // Each ID can appear only once as a source and as a target. So if all pairs
  // are identity, all IDs must appear in the list is the size == number of
  // replicas/partitions.
  const int64_t expected_size =
      op.channel_id() ? partition_count : replica_count;
  return source_target_pairs.size() == expected_size &&
         absl::c_all_of(source_target_pairs,
                        [](const std::pair<int64_t, int64_t>& source_target) {
                          return source_target.first == source_target.second;
                        });
}

/*static*/ bool NcclCollectivePermuteThunk::CanImplement(
    mlir::lmhlo::CollectivePermuteOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc mht_2(mht_2_v, 273, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.cc", "NcclCollectivePermuteThunk::CanImplement");

  const Shape shape = GetShape(op.operand());
  return IsTypeSupportedByNccl(shape.element_type());
}

NcclCollectivePermuteThunk::NcclCollectivePermuteThunk(
    ThunkInfo thunk_info, mlir::lmhlo::CollectivePermuteOp op,
    int64_t replica_count, int64_t partition_count, const Buffer& buffer)
    : NcclCollectiveThunk(Thunk::kCollectivePermute, thunk_info),
      config_(
          GetNcclCollectivePermuteConfig(op, replica_count, partition_count)),
      buffer_(buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.cc", "NcclCollectivePermuteThunk::NcclCollectivePermuteThunk");
}

Status NcclCollectivePermuteThunk::RunNcclCollective(
    const ExecuteParams& params, ncclComm_t comm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_permute_thunkDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.cc", "NcclCollectivePermuteThunk::RunNcclCollective");

#if XLA_ENABLE_XCCL
  // Determine the source and target IDs for this instance. The source ID is the
  // ID which will copy its data to this instance. The destination ID is the ID
  // to which this instance will copy its data. Either are optional.
  //
  // No source and no dest:
  //  - this instance does not actually participate, no one send it any data and
  //    it does not have to send any data as well. Since there is no dest,
  //    just memzero() the dest buffer as required by the collective permute
  //    semantics.
  //
  // No source, dest present:
  //  - This instance has to send data to 'dest' Issue an send of the input.
  //    Since there is no source, memzero the dest buffer.
  //
  // Source present, no destination:
  //  - This instance received data from the source, does not have to send data
  //    to anyone, Issue a receive.
  //
  // Source and dest both present:
  //   - Issue a send of the input to dest, receive for the output from the
  //     src.
  //
  //

  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing collective permute from device ordinal: "
          << device_ordinal;

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID current_logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  const int64_t current_id =
      config_.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;

  const NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      config_.GetSourceTarget(current_id);
  const absl::optional<int64_t> source_id = source_target.source;
  const absl::optional<int64_t> target_id = source_target.target;

  // NCCL 2.8.x has an issue with point-to-point communication primitives if
  // different ranks process different amounts of data. This can happen in the
  // case of a collective permute as certain nodes may not do any send or
  // receives, or do only send or only receive. Sending and receiving to self
  // as well (identity pair) causes this imbalance. NCCL 2.8.x requires the
  // use of NCCL_LAUNCH_MODE=PARALLEL to avoid these issues. See
  // https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-8-4.html#rel_2-8-4
  if (!IsNcclLaunchModeParallel()) {
    static absl::once_flag log_once;
    absl::call_once(log_once, [] {
      LOG(WARNING) << "NCCL based collective permute may not work correctly if "
                      "NCCL_LAUNCH_MODE is not set to PARALLEL";
    });
  }

  se::DeviceMemoryBase src_addr =
      params.buffer_allocations->GetDeviceAddress(buffer_.source_buffer);
  se::DeviceMemoryBase dest_addr =
      params.buffer_allocations->GetDeviceAddress(buffer_.destination_buffer);

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                GetDeviceString(params), current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());

  PrimitiveType element_type = config_.operand_element_type[0];
  TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                      ToNcclDataTypeAndCountMultiplier(element_type));
  ncclDataType_t dtype = dtype_and_multiplier.first;
  int element_count = buffer_.element_count * dtype_and_multiplier.second;

  se::gpu::GpuStreamHandle gpu_stream =
      se::gpu::AsGpuStreamValue(params.stream);

  // send source buffer to target peer if needed.
  if (target_id) {
    VLOG(3) << absl::StreamFormat(
        "%s : Calling ncclSend(sendbuff=%p, count=%d, peer=%d "
        "comm=%p, stream=%p)",
        GetDeviceString(params), src_addr.opaque(), element_count, *target_id,
        static_cast<const void*>(comm), gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclSend(src_addr.opaque(), element_count, dtype,
                                      *target_id, comm, gpu_stream));
  }

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    VLOG(3) << absl::StreamFormat(
        "%s : Calling ncclRecv(recvbuff=%p, count=%d, peer=%d comm=%p, "
        "stream=%p)",
        GetDeviceString(params), dest_addr.opaque(), element_count, *source_id,
        static_cast<const void*>(comm), gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclRecv(dest_addr.opaque(), element_count, dtype,
                                      *source_id, comm, gpu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  if (!source_id) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  GetDeviceString(params));
    params.stream->ThenMemZero(&dest_addr, dest_addr.size());
  }
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla
