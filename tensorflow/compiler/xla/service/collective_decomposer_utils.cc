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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_decomposer_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_decomposer_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_decomposer_utilsDTcc() {
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

#include "tensorflow/compiler/xla/service/collective_decomposer_utils.h"

#include <limits>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

// Create the start indices for decompositing the given collective.
StatusOr<std::vector<HloInstruction *>>
CreateStartIndicesForCollectiveDecomposition(
    CollectiveOpGroupMode group_mode,
    absl::Span<const ReplicaGroup> replica_groups, const Shape &shard_shape,
    int64_t shard_dimension, HloComputation *computation) {
  HloInstruction *zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
  std::vector<HloInstruction *> start_indices(shard_shape.rank(), zero);
  const Shape &scalar_shape = zero->shape();

  auto create_flattened_id = [&](HloInstruction *replica_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_decomposer_utilsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/collective_decomposer_utils.cc", "lambda");

    if (replica_index == zero) {
      // special case for 0 * num_partitions + partition_id
      return computation->AddInstruction(HloInstruction::CreatePartitionId());
    }
    const HloModuleConfig &config = computation->parent()->config();
    HloInstruction *partition_count =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32_t>(config.num_partitions())));
    HloInstruction *mul = computation->AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kMultiply,
                                     replica_index, partition_count));
    return computation->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, mul,
        computation->AddInstruction(HloInstruction::CreatePartitionId())));
  };

  HloInstruction *participant_id;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      participant_id =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      // For this mode, the replica groups contain replica_id's, but the
      // participant are replicas with the given replica_id across all
      // partitions (ordered in partition id order, see
      // GetParticipatingDevicesGroups). So replica group {0, 3} corresponds to
      // the participants {r0p0, r0p1, ..., r0pn, r3p0, r3p1, ... r3pn} where
      // number of partitions = n + 1. So the slice index for a given execution
      // instance can be computed by first computing its replica index (using
      // replica_id) and then accounting for partition_id:
      //    replica_index = map replica_id to index using the replica_groups.
      //    index = replica_index * num_partitions + partition_id;
      participant_id =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossPartition:
      participant_id =
          computation->AddInstruction(HloInstruction::CreatePartitionId());
      break;
    case CollectiveOpGroupMode::kFlattenedID:
      participant_id = create_flattened_id(
          computation->AddInstruction(HloInstruction::CreateReplicaId()));
      break;
  }

  auto is_trivial_group = [](absl::Span<const ReplicaGroup> replica_groups) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_decomposer_utilsDTcc mht_1(mht_1_v, 261, "", "./tensorflow/compiler/xla/service/collective_decomposer_utils.cc", "lambda");

    if (replica_groups.empty()) {
      return true;
    }
    if (replica_groups.size() == 1) {
      for (int64_t index = 0; index < replica_groups[0].replica_ids_size();
           ++index) {
        if (index != replica_groups[0].replica_ids(index)) {
          return false;
        }
      }
      return true;
    }
    return false;
  };

  HloInstruction *index;
  if (is_trivial_group(replica_groups)) {
    if (replica_groups.size() == 1 &&
        replica_groups[0].replica_ids_size() == 1) {
      // If there is a single replica group with a single ID, it has to be 0 and
      // the index therefore has to be 1
      TF_RET_CHECK(replica_groups[0].replica_ids(0) == 0);
      index = zero;
    } else {
      index = participant_id;
    }
  } else {
    size_t num_participants =
        replica_groups.size() * replica_groups.front().replica_ids_size();
    std::vector<uint32_t> index_values(num_participants,
                                       std::numeric_limits<uint32_t>::max());
    for (const ReplicaGroup &rg : replica_groups) {
      for (uint64_t idx = 0; idx < rg.replica_ids_size(); ++idx) {
        int64_t id = rg.replica_ids(idx);
        TF_RET_CHECK(index_values[id] == std::numeric_limits<uint32_t>::max());
        index_values[id] = idx;
      }
    }

    // create a u32 constant table of index values and use dynamic-slice to
    // index into it.
    HloInstruction *table =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR1<uint32_t>(index_values)));
    HloInstruction *ds =
        computation->AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::MakeShape(U32, {1}), table, {participant_id}, {1}));
    index = computation->AddInstruction(
        HloInstruction::CreateReshape(scalar_shape, ds));
  }

  // For cross-replica and partition mode, we need to scale the index (which is
  // the replica index) by num_partitions and add partition_id;
  if (group_mode == CollectiveOpGroupMode::kCrossReplicaAndPartition) {
    index = create_flattened_id(index);
  }

  // scale index by the shard size, which is the size of the shard_dimension.
  HloInstruction *scale = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(
          shard_shape.dimensions(shard_dimension))));
  index = computation->AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kMultiply, index, scale));
  start_indices[shard_dimension] = index;
  return start_indices;
}

}  // namespace xla
