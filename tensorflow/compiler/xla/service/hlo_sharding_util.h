/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_utilDTh() {
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
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace hlo_sharding_util {

struct GatherParallelDims {
  absl::InlinedVector<int64_t, 1> indices_parallel_dims;
  absl::InlinedVector<int64_t, 1> operand_parallel_dims;
  std::vector<int64_t> index_parallel_in_dim;
};

// Returns true if the lhs sharding is preferable over the rhs sharding.
// The most specific sharding is tile maximal followed by single device tile
// maximal and finally replicated. This order aims to primarily reduce memory
// usage and secondly reduce total compute.
// Note: This does NOT provide a total ordering as we can have 2 different
// sharding with same preference level.
bool IsShardingMoreSpecific(const HloSharding& lhs, const HloSharding& rhs);

// Tries to refine `to_merge` by combining with `old`. Returns if the final
// `to_merge` is more specific than `old`.
bool MergeSharding(const HloSharding& old, HloSharding* to_merge,
                   bool may_combine_partial_sharding);

// Merges `to_merge` into `dst` only if they are compatible, and the merged
// sharding has >= minimum_tiles tiles. Returns if merging happened.
bool MergeShardingIfCompatible(const HloSharding& to_merge,
                               int64_t minimum_tiles, HloSharding* dst);

// Given a map<device, occurrence_count>, selects the device with higher
// occurrence count (if any). If top_count in not nullptr, it will receive the
// count of the dominant device returned.
absl::optional<int64_t> SelectDominantDevice(
    const std::map<int64_t, int64_t>& device_map, int64_t* top_count);

// Assigns all the instructions of a computation, to a given device.
// This API does not recurse into called computations, and does not assign
// instructions which already have sharding.
Status AssignComputationDevice(HloComputation* computation, int64_t device);

// Given an instruction container, returns the device which is most commonly
// occurring among the instructions.
absl::optional<int64_t> GetMostOccurringDevice(
    absl::Span<HloInstruction* const> instructions);

// Given a set of computations, tries to extract the dominant device. A device
// is dominant if the combined occurrence among all the instructions of the
// input computations, is greater/equal than/to dominant_factor (real number
// from 0 to 1).
// This API does not recurse into called computations.
// If no device exists that satisfies the condition, the returned optional will
// hold no value.
StatusOr<absl::optional<int64_t>> GetDominantDevice(
    absl::Span<HloComputation* const> computations, double dominant_factor);

// Returns the HloSharding with the tile dimensions and tile assignment
// transposed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding TransposeSharding(const HloSharding& sharding,
                              absl::Span<const int64_t> dimensions);

// Returns the HloSharding with the tile shape reshaped based on the source and
// target shapes and the tile assignment adjusted to correspond to the new tile
// shape or absl::nullopt if the resulting reshape would create an invalid
// sharding (non continuous or non uniformly sized tiles). In case of a tile
// maximal sharding returns the original sharding.
absl::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding);

// Returns the HloSharding with the tile dimensions and tile assignment
// reversed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding ReverseSharding(const HloSharding& sharding,
                            absl::Span<const int64_t> dimensions);

// Returns a sharding tiled on unique dimension dim by reshaping the tile
// assignment of the sharding argument. Only dimensions in the dims span
// argument are considered for reshaping, the others are ignored.
// Assumptions: sharding is tile sharded, and dim must be included in dims.
HloSharding ReshapeToTileDimension(const HloSharding& sharding, int64_t dim,
                                   absl::Span<const int64_t> dims);

// Returns true if the provided module includes one or more instructions with
// a tile sharding.
bool ContainsTileSharding(const HloModule& module);

// Returns the preferred output sharding for a gather op based on the sharding
// of the indces.
HloSharding GatherOutputSharding(const HloSharding& index_sharding,
                                 const HloInstruction* hlo);

// Returns the preferred index sharding for a gather op based on the sharding
// of the output.
HloSharding GatherIndexSharding(const HloSharding& output_sharding,
                                const HloInstruction* hlo);

// Returns a new HloSharding for a gather op so that only non offset dimensions
// are sharded. Assume "result" is returned by this function. It is ensured that
// "GetIndexSharding(result, hlo)" will have the same number of elements as
// "result".
HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo);

// Returns the preferred index sharding for a scatter op based on the sharding
// of the data.
HloSharding ScatterIndexSharding(const HloSharding& data_sharding,
                                 const HloInstruction* hlo);

// Returns the preferred data sharding for a scatter op based on the sharding
// of the index.
HloSharding ScatterDataSharding(const HloSharding& index_sharding,
                                const HloInstruction* hlo);

// Returns a new index sharding for a scatter op so that we only shard on first
// "number of scatter_window_dims" dimensions. Assume "result" is returned by
// this function. It is ensured that "ScatterDataSharding(result, hlo)" will
// have the same number of elements as "result".
HloSharding ScatterEffectiveIndexSharding(const HloSharding& index_sharding,
                                          const HloInstruction& hlo);

// Returns a new data sharding for a scatter op so that we only shard on
// scatter_window_dims. Assume "result" is returned by this function. It is
// ensured that "ScatterIndexSharding(result, hlo)" will have the same number of
// elements as "result".
HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloInstruction& hlo);

// Returns an output sharding of gather by passing through the data operand's
// sharding.
absl::optional<HloSharding> GatherOutputShardingFromDataOperand(
    const HloSharding& data_operand_sharding, const HloInstruction& hlo,
    absl::Span<const int64_t> slice_sizes, const Shape& output_shape,
    const Shape& operand_shape);

// Returns a data operand sharding of gather by passing through the output's
// sharding.
absl::optional<HloSharding> GatherDataOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo);

// Returns an output sharding of scatter by passing through the update operand's
// sharding.
absl::optional<HloSharding> ScatterOutputShardingFromUpdate(
    const HloSharding& update_sharding, const HloInstruction& hlo);

// Returns an update operand sharding of scatter by passing through the output's
// sharding.
absl::optional<HloSharding> ScatterUpdateShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo);

// Returns an identity value and an HloOpcode for reduce computation of scatter
// instruction.
// - If computation is add/or, return 0/false with corresponding op code;
// - If computation is multiply/and, return 1/true with corresponding op code.
// - If computation is min/max, return max value/min value with corresponding op
//   code.
// - Otherwise, return error status.
StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter);

// Given a sharding and a list of devices in the topology, return a
// list of the devices that `sharding` applies to.
std::vector<int64_t> DevicesForSharding(
    const HloSharding& sharding, absl::Span<const int64_t> available_devices);

// Returns a sharding that replicates data across devices along the given
// dimensions in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnDims(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_replicate);

// Returns a sharding that replicates data across devices along all dimensions
// but the given ones to keep in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_keep);

// Returns a sharding that replicates all data dimensions, but keep manual
// subgroups. If data_rank is provided >= 0, the result sharding's data rank
// will be set to it.
HloSharding ReplicateAllDataDims(const HloSharding& sharding,
                                 int64_t data_rank = -1);

// Returns a sharding the removes given tile dimensions.
//
// Precondition: if not tile maximal, the size of each tile dimension must be 1.
HloSharding RemoveShapeDimensions(const HloSharding& sharding,
                                  absl::Span<const int64_t> dims_to_remove);

// Similar to TransposeSharding(), but allows removing/adding non-partitioned
// dimensions. In src_to_tgt and tgt_to_src, -1 represents a non-existing
// dimension.
absl::optional<HloSharding> TransposeShardingWithCollapsedDims(
    const HloSharding& source, absl::Span<int64_t const> src_to_tgt,
    absl::Span<int64_t const> tgt_to_src);

// Returns the iota dimension if maybe_iota is an kIota instruction or
// equivalent to kIota.
absl::optional<int64_t> GetDimensionForIota(const HloInstruction* maybe_iota);

// Returns identified parallel dimensions for Gather.
absl::optional<GatherParallelDims> GetGatherBatchParallelDims(
    const HloInstruction& hlo);

// Returns the parallel dimensions of the output of a gather based on the
// parallel dimensions of the input.
absl::InlinedVector<int64_t, 1> GatherParallelOutputDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dim);

// Returns the parallel dimensions of the data operand of a gather with the
// order of the parallel dimensions matching that of the parallel dimensions
// of the output.
absl::InlinedVector<int64_t, 1> GatherOutputAlignedOperandParallelDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dims);

// Represents grouping devices in a tiled sharding along certain dimensions.
// Elements in group dimensions define different device groups, and the sharding
// represents the in-group sharding.
struct GroupedSharding {
  GroupedSharding(std::vector<std::vector<int64_t>> device_groups,
                  std::vector<int64_t> group_dims,
                  std::vector<int64_t> group_dim_sizes, int64_t data_rank,
                  HloSharding grouped_sharding, bool subgroup_manual = false)
      : device_groups(std::move(device_groups)),
        group_dims(std::move(group_dims)),
        group_dim_sizes(std::move(group_dim_sizes)),
        data_rank(data_rank),
        sharding(std::move(grouped_sharding)),
        subgroup_manual(subgroup_manual) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_utilDTh mht_0(mht_0_v, 423, "", "./tensorflow/compiler/xla/service/hlo_sharding_util.h", "GroupedSharding");
}
  std::vector<std::vector<int64_t>> device_groups;
  std::vector<int64_t> group_dims;
  std::vector<int64_t> group_dim_sizes;
  int64_t data_rank;
  HloSharding sharding;
  bool subgroup_manual;
};

// Creates a GroupedSharding for a tiled sharding with group dim shard sizes.
GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    absl::Span<const int64_t> group_dim_shards,
                                    bool subgroup_manual = false);

// Creates a GroupedSharding for a tiled sharding.
GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    bool subgroup_manual = false);

// Get group sharding for each manual subgroup.
GroupedSharding GetManualSubgroupSharding(const HloSharding& sharding);

// Reconstructs the ungrouped sharding from a GroupedSharding.
HloSharding UngroupSharding(const GroupedSharding& grouped_sharding);

// Check if the device groups are match for the LHS or RHS group shardings.
bool DeviceGroupsAreMatch(GroupedSharding& lhs, GroupedSharding& rhs,
                          bool ignore_group_order = true);

}  // namespace hlo_sharding_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_
