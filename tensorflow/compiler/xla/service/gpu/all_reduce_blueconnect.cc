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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnectDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnectDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnectDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

std::vector<HloInstruction*> GetOutputs(HloInstruction& instruction) {
  if (!instruction.shape().IsTuple()) {
    return {&instruction};
  }

  std::vector<HloInstruction*> outputs;
  outputs.reserve(instruction.shape().tuple_shapes_size());

  HloComputation& computation = *instruction.parent();  // never null
  for (int i = 0; i < instruction.shape().tuple_shapes_size(); ++i) {
    outputs.push_back(computation.AddInstruction(
        HloInstruction::CreateGetTupleElement(&instruction, i)));
  }
  return outputs;
}

struct DecomposedReplicaGroups {
  std::vector<ReplicaGroup> scatter_gather_groups;
  std::vector<ReplicaGroup> new_all_reduce_groups;
};

StatusOr<absl::optional<DecomposedReplicaGroups>> TryDecomposeReplicaGroup(
    const ReplicaGroup& replica_group,
    const DeviceAssignment& device_assignment, size_t num_devices_per_host) {
  int group_size = replica_group.replica_ids_size();
  TF_RET_CHECK(group_size > 0);

  absl::btree_map<int, std::vector<int64_t>> replica_ids_by_host;
  for (int64_t replica_id : replica_group.replica_ids()) {
    int device_id = device_assignment(replica_id, /*computation_id=*/0);
    TF_RET_CHECK(device_id >= 0);
    // We assume that devices are ordered by host.
    int host_id = device_id / num_devices_per_host;
    replica_ids_by_host[host_id].push_back(replica_id);
  }

  size_t num_local_devices = replica_ids_by_host.begin()->second.size();
  bool same_num_devices_on_each_host =
      absl::c_all_of(replica_ids_by_host, [&](const auto& entry) {
        return entry.second.size() == num_local_devices;
      });

  if (!same_num_devices_on_each_host) {
    return {absl::nullopt};
  }

  std::vector<int64_t> sorted_replica_group;
  sorted_replica_group.reserve(group_size);
  for (const auto& entry : replica_ids_by_host) {
    absl::c_copy(entry.second, std::back_inserter(sorted_replica_group));
  }

  size_t scatter_group_size = std::max(num_local_devices, size_t(2));
  size_t num_scatter_groups = group_size / scatter_group_size;

  if ((group_size % scatter_group_size != 0) || (num_scatter_groups < 2)) {
    return {absl::nullopt};
  }

  std::vector<ReplicaGroup> scatter_gather_groups(num_scatter_groups);
  std::vector<ReplicaGroup> new_all_reduce_groups(scatter_group_size);

  for (size_t i = 0; i < group_size; ++i) {
    int64_t replica_id = sorted_replica_group[i];
    scatter_gather_groups[i / scatter_group_size].add_replica_ids(replica_id);
    new_all_reduce_groups[i % scatter_group_size].add_replica_ids(replica_id);
  }

  return {DecomposedReplicaGroups{std::move(scatter_gather_groups),
                                  std::move(new_all_reduce_groups)}};
}

StatusOr<absl::optional<DecomposedReplicaGroups>> TryDecomposeReplicaGroups(
    const HloAllReduceInstruction& all_reduce, size_t num_devices_per_host) {
  const DeviceAssignment& device_assignment =
      all_reduce.parent()->parent()->config().static_device_assignment();

  absl::Span<const ReplicaGroup> replica_groups = all_reduce.replica_groups();

  ReplicaGroup all_replicas;  // only populated if replica groups not present.
  if (replica_groups.empty()) {
    for (int i = 0; i < device_assignment.replica_count(); ++i) {
      all_replicas.add_replica_ids(i);
    }
    replica_groups = absl::MakeSpan(&all_replicas, 1);
  }

  std::vector<ReplicaGroup> scatter_gather_groups;
  std::vector<ReplicaGroup> new_all_reduce_groups;

  // Try to find a valid decomposition for each replica group.
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_ASSIGN_OR_RETURN(
        absl::optional<DecomposedReplicaGroups> decomposed_groups,
        TryDecomposeReplicaGroup(replica_group, device_assignment,
                                 num_devices_per_host));

    if (!decomposed_groups) return {absl::nullopt};

    int scatter_group_size =
        decomposed_groups->scatter_gather_groups[0].replica_ids_size();

    if (scatter_gather_groups.empty()) {
      // Check that every operand is exactly divisible by scatter group sizes.
      for (const HloInstruction* operand : all_reduce.operands()) {
        TF_RET_CHECK(operand->shape().IsArray());
        int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
        if (num_elements % scatter_group_size != 0) {
          return {absl::nullopt};
        }
      }

      scatter_gather_groups.reserve(
          replica_groups.size() *
          decomposed_groups->scatter_gather_groups.size());
      new_all_reduce_groups.reserve(
          replica_groups.size() *
          decomposed_groups->new_all_reduce_groups.size());
    } else if (scatter_group_size !=
               scatter_gather_groups[0].replica_ids_size()) {
      // Reduce-scatter would have different output shapes on different devices.
      return {absl::nullopt};
    }

    absl::c_move(decomposed_groups->scatter_gather_groups,
                 std::back_inserter(scatter_gather_groups));
    absl::c_move(decomposed_groups->new_all_reduce_groups,
                 std::back_inserter(new_all_reduce_groups));
  }

  return {DecomposedReplicaGroups{std::move(scatter_gather_groups),
                                  std::move(new_all_reduce_groups)}};
}

// Attempts to decompose all-reduces as described by the BlueConnect paper.
//
// If possible, the all-reduce will be transformed into:
// 1. reduce-scatter
// 2. all-reduce
// 3. all-gather
//
// If the all-reduce replica groups have more than one device within the same
// host, the reduce-scatter will be performed over all devices with each host.
// Otherwise, the reduce-scatter will be performed between pairs of devices on
// different hosts.
//
// When applied repeatedly, this transformation will reproduce the same pattern
// as described in the BlueConnect paper.
StatusOr<bool> TryDecomposeAllReduce(HloAllReduceInstruction* all_reduce,
                                     size_t num_devices_per_host) {
  TF_RET_CHECK(all_reduce);
  TF_RET_CHECK(!all_reduce->has_sharding());

  HloComputation& computation = *all_reduce->parent();  // never null
  PrimitiveType element_type = all_reduce->operand(0)->shape().element_type();

  TF_ASSIGN_OR_RETURN(
      absl::optional<DecomposedReplicaGroups> decomposed_groups,
      TryDecomposeReplicaGroups(*all_reduce, num_devices_per_host));

  if (!decomposed_groups) return false;

  // Bitcast operands to 1D to guarantee that first dimension is divisible by
  // scatter group size (we checked num elements was divisible above).
  std::vector<HloInstruction*> flat_operands;
  flat_operands.reserve(all_reduce->operand_count());
  std::vector<Shape> flat_shapes;
  flat_shapes.reserve(all_reduce->operand_count());
  std::vector<Shape> scattered_shapes;
  scattered_shapes.reserve(all_reduce->operand_count());

  int scatter_group_size =
      decomposed_groups->scatter_gather_groups[0].replica_ids_size();

  for (HloInstruction* operand : all_reduce->operands()) {
    TF_RET_CHECK(operand->shape().IsArray());
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flat_shape = ShapeUtil::MakeShape(element_type, {num_elements});
    flat_operands.push_back(computation.AddInstruction(
        HloInstruction::CreateBitcast(flat_shape, operand)));
    flat_shapes.push_back(std::move(flat_shape));
    scattered_shapes.push_back(ShapeUtil::MakeShape(
        element_type, {num_elements / scatter_group_size}));
  }

  Shape reduce_scatter_shape = ShapeUtil::MakeMaybeTupleShape(scattered_shapes);

  HloInstruction* reduce_scatter =
      computation.AddInstruction(HloInstruction::CreateReduceScatter(
          reduce_scatter_shape, flat_operands, all_reduce->to_apply(),
          decomposed_groups->scatter_gather_groups, /*constrain_layout=*/false,
          all_reduce->channel_id(), all_reduce->use_global_device_ids(),
          /*scatter_dimension=*/0));

  HloInstruction* new_all_reduce =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          reduce_scatter_shape, GetOutputs(*reduce_scatter),
          all_reduce->to_apply(), decomposed_groups->new_all_reduce_groups,
          /*constrain_layout=*/false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));

  HloInstruction* all_gather =
      computation.AddInstruction(HloInstruction::CreateAllGather(
          ShapeUtil::MakeMaybeTupleShape(flat_shapes),
          GetOutputs(*new_all_reduce),
          /*all_gather_dimension=*/0, decomposed_groups->scatter_gather_groups,
          /*constrain_layout=*/false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));

  // Bitcast back to the original shapes and replace all-reduce with decomposed
  // implementation.
  std::vector<HloInstruction*> outputs = GetOutputs(*all_gather);
  for (int64_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = computation.AddInstruction(HloInstruction::CreateBitcast(
        all_reduce->operand(i)->shape(), outputs[i]));
  }

  TF_RETURN_IF_ERROR(computation.ReplaceInstruction(
      all_reduce,
      (outputs.size() == 1)
          ? outputs[0]
          : computation.AddInstruction(HloInstruction::CreateTuple(outputs))));

  // Try to apply decomposition recursively.
  TF_RETURN_IF_ERROR(
      TryDecomposeAllReduce(Cast<HloAllReduceInstruction>(new_all_reduce),
                            num_devices_per_host)
          .status());
  return true;
}

}  // namespace

StatusOr<bool> AllReduceBlueConnect::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnectDTcc mht_0(mht_0_v, 439, "", "./tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.cc", "AllReduceBlueConnect::Run");

  VLOG(1) << "Running AllReduceBlueConnect";

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceBlueConnect because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }
  if (!module->config().has_static_device_assignment()) {
    VLOG(1)
        << "Skip AllReduceBlueConnect because the module doesn't have static "
           "device assignment";
    return false;
  }

  std::vector<HloAllReduceInstruction*> all_reduces;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        all_reduces.push_back(Cast<HloAllReduceInstruction>(instruction));
      }
    }
  }

  bool changed = false;
  for (HloAllReduceInstruction* all_reduce : all_reduces) {
    TF_ASSIGN_OR_RETURN(
        bool all_reduce_changed,
        TryDecomposeAllReduce(all_reduce, num_devices_per_host_));
    changed |= all_reduce_changed;
  }

  return changed;
}

}  // namespace xla
