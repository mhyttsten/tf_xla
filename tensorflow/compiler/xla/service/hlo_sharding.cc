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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_sharding.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/service/hlo_op_metadata.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

using absl::StrCat;
using absl::StrJoin;

HloSharding HloSharding::AssignDevice(int64_t device_id,
                                      absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::AssignDevice");

  return HloSharding(device_id, metadata);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64_t num_tiles,
                                absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Tile1D");

  CHECK_EQ(1, input_shape.rank());
  CHECK_GT(num_tiles, 1);
  std::vector<int64_t> dimensions(1, num_tiles);
  Array<int64_t> assignment(dimensions);
  std::iota(assignment.begin(), assignment.end(), 0);
  return HloSharding(assignment, /*replicate_on_last_tile_dim=*/false,
                     metadata);
}

HloSharding HloSharding::PartialTile(
    const Array<int64_t>& group_tile_assignment,
    absl::Span<const absl::Span<const int64_t>> replication_groups,
    absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::PartialTile");

  CHECK_EQ(group_tile_assignment.num_elements(), replication_groups.size());
  if (replication_groups.size() == 1) {
    return Replicate(metadata);
  }
  auto new_tile_dims = group_tile_assignment.dimensions();
  new_tile_dims.push_back(replication_groups[0].size());
  auto new_tile_assignment = Array<int64_t>(new_tile_dims);
  new_tile_assignment.Each(
      [&](absl::Span<const int64_t> indices, int64_t* device) {
        std::vector<int64_t> group_index(indices.begin(), indices.end());
        group_index.pop_back();
        int64_t group = group_tile_assignment(group_index);
        *device = replication_groups[group][indices.back()];
      });
  return PartialTile(new_tile_assignment, metadata);
}

HloSharding HloSharding::PartialTile(
    const Array<int64_t>& tile_assignment_last_dim_replicate,
    absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::PartialTile");

  if (tile_assignment_last_dim_replicate.num_dimensions() == 1 ||
      tile_assignment_last_dim_replicate.dimensions().back() ==
          tile_assignment_last_dim_replicate.num_elements()) {
    return Replicate(metadata);
  }
  if (tile_assignment_last_dim_replicate.dimensions().back() == 1) {
    auto new_tile_dims = tile_assignment_last_dim_replicate.dimensions();
    new_tile_dims.pop_back();
    auto fully_tiled = tile_assignment_last_dim_replicate;
    fully_tiled.Reshape(new_tile_dims);
    return HloSharding(fully_tiled, /*replicate_on_last_tile_dim=*/false,
                       metadata);
  }
  std::vector<std::set<int64_t>> sorted_groups(
      tile_assignment_last_dim_replicate.num_elements() /
      tile_assignment_last_dim_replicate.dimensions().back());
  auto get_group_id = [&](absl::Span<const int64_t> indices) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "lambda");

    int64_t group_id = 0;
    for (int64_t i = 0; i < indices.size() - 1; ++i) {
      group_id *= tile_assignment_last_dim_replicate.dim(i);
      group_id += indices[i];
    }
    return group_id;
  };
  tile_assignment_last_dim_replicate.Each(
      [&](absl::Span<const int64_t> indices, const int64_t device) {
        sorted_groups[get_group_id(indices)].insert(device);
      });
  Array<int64_t> sorted_tile(tile_assignment_last_dim_replicate.dimensions());
  sorted_tile.Each([&](absl::Span<const int64_t> indices, int64_t* device) {
    const int64_t group_id = get_group_id(indices);
    auto begin = sorted_groups[group_id].begin();
    *device = *begin;
    sorted_groups[group_id].erase(begin);
  });
  return HloSharding(sorted_tile, /*replicate_on_last_tile_dim=*/true,
                     metadata);
}

HloSharding HloSharding::Subgroup(
    const Array<int64_t>& tile_assignment,
    absl::Span<const OpSharding::Type> subgroup_types,
    absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Subgroup");

  if (subgroup_types.empty()) {
    return HloSharding(tile_assignment, /*replicate_on_last_tile_dim=*/false,
                       metadata);
  }
  // If there is only one type of subgrouping and there is no tiling on data
  // dimensions, it can be canonicalized to a simple manual/replicated sharding.
  if (absl::c_all_of(
          subgroup_types,
          [&](const OpSharding::Type t) { return t == subgroup_types[0]; }) &&
      Product(absl::Span<const int64_t>(tile_assignment.dimensions())
                  .subspan(0, tile_assignment.num_dimensions() -
                                  subgroup_types.size())) == 1) {
    if (subgroup_types[0] == OpSharding::MANUAL) {
      return Manual(metadata);
    }
    if (subgroup_types[0] == OpSharding::REPLICATED) {
      return Replicate(metadata);
    }
  }
  // Normalize the subgroups to simplify two cases:
  //   - Remove trivial dims of size 1.
  //   - Merge dims of the same type.
  //   - Sort types.
  int64_t data_dims = tile_assignment.num_dimensions() - subgroup_types.size();
  std::vector<int64_t> perm(data_dims);
  std::iota(perm.begin(), perm.end(), 0);
  // Make sure the replicate dims are at the end so that we can leverage
  // PartialTile() to sort the elements.
  struct CmpTypeRepliateLast {
    bool operator()(OpSharding::Type a, OpSharding::Type b) const {
      if (a == b) {
        return false;
      }
      if (a == OpSharding::REPLICATED) {
        return false;
      }
      if (b == OpSharding::REPLICATED) {
        return true;
      }
      return a < b;
    }
  };
  std::map<OpSharding::Type, std::vector<int64_t>, CmpTypeRepliateLast>
      type_to_dims;
  bool needs_merging = false;
  for (int64_t i = 0; i < subgroup_types.size(); ++i) {
    if (tile_assignment.dim(i + data_dims) == 1) {
      needs_merging = true;
      continue;
    }
    auto& dims = type_to_dims[subgroup_types[i]];
    needs_merging |= !dims.empty();
    dims.push_back(i + data_dims);
  }
  needs_merging |= type_to_dims.size() > 1;
  auto create_sharding = [](const Array<int64_t> tiles,
                            absl::Span<const OpSharding::Type> types,
                            absl::Span<const OpMetadata> metadata) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_6(mht_6_v, 367, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "lambda");

    if (types.size() == 1 && types.back() == OpSharding::REPLICATED) {
      // Normalize to partial tile.
      return PartialTile(tiles, metadata);
    }
    if (types.size() == 1 && types.back() == OpSharding::MANUAL &&
        tiles.num_elements() == tiles.dimensions().back()) {
      // Normalize to manual.
      return Manual(metadata);
    }
    if (!types.empty() && types.back() == OpSharding::REPLICATED) {
      // If the last type is REPLICATED, we first create a partially replicated
      // sharding without other subgroups so that the elements are sorted. Then
      // we fix the subgroup types.
      HloSharding sharding = PartialTile(tiles, metadata);
      sharding.replicate_on_last_tile_dim_ = false;
      for (const OpSharding::Type type : types) {
        sharding.subgroup_types_.push_back(type);
      }
      return sharding;
    }
    return HloSharding(tiles, types, metadata);
  };
  if (needs_merging) {
    auto data_tile_shape =
        absl::Span<const int64_t>(tile_assignment.dimensions())
            .subspan(0, data_dims);
    std::vector<int64_t> merged_shape(data_tile_shape.begin(),
                                      data_tile_shape.end());
    std::vector<int64_t> transposed_shape = merged_shape;
    std::vector<OpSharding::Type> merged_types;
    for (const auto& type_dims : type_to_dims) {
      int64_t dim_size = 1;
      for (int64_t dim : type_dims.second) {
        perm.push_back(dim);
        dim_size *= tile_assignment.dim(dim);
        transposed_shape.push_back(tile_assignment.dim(dim));
      }
      merged_shape.push_back(dim_size);
      merged_types.push_back(type_dims.first);
    }
    Array<int64_t> new_tiles(transposed_shape);
    new_tiles.Each([&](absl::Span<const int64_t> indices, int64_t* value) {
      std::vector<int64_t> src_indices(tile_assignment.num_dimensions(), 0);
      for (int64_t i = 0; i < indices.size(); ++i) {
        src_indices[perm[i]] = indices[i];
      }
      *value = tile_assignment(src_indices);
    });
    new_tiles.Reshape(merged_shape);
    return create_sharding(new_tiles, merged_types, metadata);
  }
  return create_sharding(tile_assignment, subgroup_types, metadata);
}

HloSharding HloSharding::Tuple(const ShapeTree<HloSharding>& sub_shardings) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_7(mht_7_v, 425, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Tuple");

  std::vector<HloSharding> flattened_list;
  flattened_list.reserve(sub_shardings.leaf_count());
  for (const auto& index_to_sharding : sub_shardings.leaves()) {
    flattened_list.push_back(index_to_sharding.second);
  }
  if (flattened_list.empty()) {
    // Empty tuple sharding ends up having no leaves, but we want to allow
    // empty tuple HLO instruction results to have sharding, so we fetch the
    // root ({}) sharding value from the ShapeTree.
    // A ShapeTree created with ShapeTree<HloSharding>(shape, init) will have
    // init as value at its root.
    flattened_list.push_back(sub_shardings.element(ShapeIndex({})));
  }
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Tuple(const Shape& tuple_shape,
                               absl::Span<const HloSharding> shardings) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_8(mht_8_v, 446, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Tuple");

  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  for (auto& sharding : shardings) {
    CHECK(!sharding.IsTuple())
        << sharding.ToString() << ShapeUtil::HumanString(tuple_shape);
  }
  std::vector<HloSharding> flattened_list(shardings.begin(), shardings.end());
  CHECK_EQ(flattened_list.size(), RequiredLeaves(tuple_shape))
      << "Flat list has " << flattened_list.size() << ", required "
      << RequiredLeaves(tuple_shape);
  return HloSharding(flattened_list);
}

HloSharding HloSharding::SingleTuple(const Shape& tuple_shape,
                                     const HloSharding& sharding) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_9(mht_9_v, 463, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::SingleTuple");

  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  CHECK(!sharding.IsTuple()) << sharding.ToString();
  int64_t leaf_count = RequiredLeaves(tuple_shape);
  std::vector<HloSharding> flattened_list;
  flattened_list.resize(leaf_count, sharding);
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Single(const Shape& shape,
                                const HloSharding& sharding) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_10(mht_10_v, 476, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Single");

  return shape.IsTuple() ? SingleTuple(shape, sharding) : sharding;
}

std::string HloSharding::ToString(bool include_metadata) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_11(mht_11_v, 483, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::ToString");

  if (IsTuple()) {
    CHECK(metadata_.empty());
    std::string result = "{";
    for (int i = 0; i < tuple_elements_.size(); ++i) {
      const HloSharding& element = tuple_elements_[i];
      if (i != 0) {
        absl::StrAppend(&result, ", ");
        if (i % 5 == 0) {
          absl::StrAppend(&result, "/*index=", i, "*/");
        }
      }
      absl::StrAppend(&result, element.ToString(include_metadata));
    }
    absl::StrAppend(&result, "}");
    return result;
  }

  std::string metadata;
  if (include_metadata) {
    if (metadata_.size() == 1) {
      metadata =
          StrCat(" metadata={", OpMetadataToString(metadata_.front()), "}");
    } else if (metadata_.size() > 1) {
      std::vector<std::string> metadata_strings;
      metadata_strings.reserve(metadata_.size());
      for (const auto& single_metadata : metadata_) {
        metadata_strings.push_back(
            StrCat("{", OpMetadataToString(single_metadata), "}"));
      }
      metadata = StrCat(" metadata={", StrJoin(metadata_strings, ", "), "}");
    }
  }

  std::string last_tile_dims;
  if (!subgroup_types_.empty()) {
    auto op_sharding_type_to_string = [](OpSharding::Type type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_12(mht_12_v, 522, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "lambda");

      switch (type) {
        case OpSharding::MANUAL:
          return "manual";
        case OpSharding::MAXIMAL:
          return "maximul";
        case OpSharding::REPLICATED:
          return "replicated";
        default:
          return "error_type.";
      }
    };
    std::vector<std::string> sharding_type_strings;
    sharding_type_strings.reserve(subgroup_types_.size());
    for (const auto& single_sharding_type : subgroup_types_) {
      sharding_type_strings.push_back(
          op_sharding_type_to_string(single_sharding_type));
    }
    last_tile_dims =
        StrCat(" last_tile_dims={", StrJoin(sharding_type_strings, ", "), "}");
  }

  if (replicated_) {
    return StrCat("{replicated", metadata, "}");
  }

  if (manual_) {
    return StrCat("{manual", metadata, "}");
  }
  if (maximal_) {
    return StrCat(
        "{maximal device=", static_cast<int64_t>(*tile_assignment_.begin()),
        metadata, "}");
  }
  return StrCat("{devices=[", StrJoin(tile_assignment_.dimensions(), ","), "]",
                StrJoin(tile_assignment_, ","),
                replicate_on_last_tile_dim_ ? " last_tile_dim_replicate" : "",
                last_tile_dims, metadata, "}");
}

bool HloSharding::UsesDevice(int64_t device) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_13(mht_13_v, 565, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::UsesDevice");

  if (IsTuple()) {
    return absl::c_any_of(tuple_elements_, [&](const HloSharding& s) {
      return s.UsesDevice(device);
    });
  }
  const auto& devices = tile_assignment_;
  return replicated_ || manual_ || absl::c_linear_search(devices, device);
}

std::map<int64_t, int64_t> HloSharding::UsedDevices(int64_t* count) const {
  int64_t element_count = 1;
  std::map<int64_t, int64_t> device_map;
  if (IsTuple()) {
    for (auto& tuple_element_sharding : tuple_elements()) {
      auto unique_device = tuple_element_sharding.UniqueDevice();
      if (unique_device) {
        device_map[*unique_device] += 1;
      }
    }
    element_count = tuple_elements().size();
  } else {
    auto unique_device = UniqueDevice();
    if (unique_device) {
      device_map[*unique_device] += 1;
    }
  }
  if (count != nullptr) {
    *count = element_count;
  }
  return device_map;
}

std::vector<int64_t> HloSharding::TileIndexForDevice(int64_t device) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_14(mht_14_v, 601, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::TileIndexForDevice");

  CHECK(!maximal_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  std::vector<int64_t> ret_index;
  tile_assignment_.Each([&](absl::Span<const int64_t> index, int64_t d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  ret_index.resize(TiledDataRank());
  return ret_index;
}

int64_t HloSharding::DeviceForTileIndex(absl::Span<const int64_t> index) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_15(mht_15_v, 619, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::DeviceForTileIndex");

  CHECK(!replicated_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.begin();
  }
  if (index.size() == TiledDataRank() &&
      index.size() < tile_assignment_.num_dimensions()) {
    std::vector<int64_t> first_subgroup_index(index.begin(), index.end());
    for (int64_t i = 0; i < tile_assignment_.num_dimensions() - index.size();
         ++i) {
      first_subgroup_index.push_back(0);
    }
    return tile_assignment_(first_subgroup_index);
  }
  return tile_assignment_(index);
}

std::vector<int64_t> HloSharding::TileOffsetForDevice(const Shape& shape,
                                                      int64_t device) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_16(mht_16_v, 642, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::TileOffsetForDevice");

  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions_size(), 0);
  }
  CHECK_EQ(shape.dimensions_size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
  }
  return index;
}

std::vector<int64_t> HloSharding::TileLimitForDevice(const Shape& shape,
                                                     int64_t device) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_17(mht_17_v, 663, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::TileLimitForDevice");

  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions().begin(),
                                shape.dimensions().end());
  }

  CHECK_EQ(shape.dimensions_size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
  }
  return index;
}

int64_t HloSharding::RequiredLeaves(const Shape& shape) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_18(mht_18_v, 686, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::RequiredLeaves");

  // Empty tuples (with arbitrary nesting) have no leaf nodes as far as
  // ShapeUtil and ShapeTree are concerned, but they do have a single
  // tuple_elements_ entry since we want to allow empty tuple results to
  // have sharding.
  const int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  return (leaf_count == 0) ? 1 : leaf_count;
}

Status HloSharding::CheckLeafCount(const Shape& shape) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_19(mht_19_v, 698, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::CheckLeafCount");

  int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  if (leaf_count == 0 && tuple_elements_.size() == 1) {
    // Allow (but don't require) empty tuples to have a single sharding
    return Status::OK();
  }
  TF_RET_CHECK(leaf_count == tuple_elements_.size())
      << "Shape " << ShapeUtil::HumanString(shape) << " has " << leaf_count
      << " leaf nodes while this sharding has " << tuple_elements_.size();
  return Status::OK();
}

StatusOr<ShapeTree<HloSharding>> HloSharding::AsShapeTree(
    const Shape& shape) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_20(mht_20_v, 714, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::AsShapeTree");

  if (IsTuple()) {
    ShapeTree<HloSharding> result(shape, HloSharding::Replicate());
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    auto it = tuple_elements_.begin();
    for (auto& index_to_sharding : result.leaves()) {
      index_to_sharding.second = *it++;
    }
    if (ShapeUtil::IsEmptyTuple(shape)) {
      // Empty tuples have no leaves, but we want to assign them a sharding
      // anyway, so we use the root element sharding.
      *result.mutable_element(ShapeIndex({})) = *it;
    }
    return std::move(result);
  } else {
    return ShapeTree<HloSharding>(shape, *this);
  }
}

StatusOr<HloSharding> HloSharding::GetTupleSharding(const Shape& shape) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_21(mht_21_v, 736, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::GetTupleSharding");

  if (IsTuple()) {
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    return *this;
  }
  return Tuple(ShapeTree<HloSharding>(shape, *this));
}

absl::optional<int64_t> HloSharding::UniqueDevice() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_22(mht_22_v, 747, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::UniqueDevice");

  if (IsTuple()) {
    if (tuple_elements_.empty()) {
      return absl::nullopt;
    }
    absl::optional<int64_t> unique_device;
    for (auto& tuple_sharding : tuple_elements_) {
      auto device = tuple_sharding.UniqueDevice();
      if (!device || (unique_device && *device != *unique_device)) {
        return absl::nullopt;
      }
      unique_device = device;
    }
    return unique_device;
  }
  if (!replicated_ && maximal_) {
    return static_cast<int64_t>(*tile_assignment_.begin());
  }
  return absl::nullopt;
}

int64_t HloSharding::GetUniqueDevice() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_23(mht_23_v, 771, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::GetUniqueDevice");

  auto device = UniqueDevice();
  CHECK(device) << "Sharding does not have a unique device: " << *this;
  return *device;
}

Status HloSharding::ValidateTuple(const Shape& shape,
                                  int64_t num_devices) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_24(mht_24_v, 781, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::ValidateTuple");

  if (!shape.IsTuple()) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Sharding is tuple-shaped but validation shape is not."));
  }
  TF_RETURN_IF_ERROR(CheckLeafCount(shape));
  if (ShapeUtil::GetLeafCount(shape) == 0 && tuple_elements_.empty()) {
    // Empty tuples are allowed to not have sharding
    return Status::OK();
  }

  // Now we've validated the number of tuple elements, it's safe to request a
  // shape tree.
  ShapeTree<HloSharding> shape_tree = GetAsShapeTree(shape);
  for (const auto& index_to_sharding : shape_tree.leaves()) {
    Status status = index_to_sharding.second.ValidateNonTuple(
        ShapeUtil::GetSubshape(shape, index_to_sharding.first), num_devices);
    if (!status.ok()) {
      tensorflow::errors::AppendToMessage(
          &status, StrCat("Note: While validating sharding tuple element ",
                          index_to_sharding.first.ToString(), " which is ",
                          index_to_sharding.second.ToString()));
      return status;
    }
  }
  return Status::OK();
}

Status HloSharding::Validate(const Shape& shape, int64_t num_devices) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_25(mht_25_v, 812, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::Validate");

  if (shape.IsToken()) {
    return Status::OK();
  }
  Status status = IsTuple() ? ValidateTuple(shape, num_devices)
                            : ValidateNonTuple(shape, num_devices);
  if (!status.ok()) {
    tensorflow::errors::AppendToMessage(
        &status, StrCat("Note: While validating sharding ", ToString(),
                        " against shape ", ShapeUtil::HumanString(shape)));
  }
  return status;
}

Status HloSharding::ValidateNonTuple(const Shape& shape,
                                     int64_t num_devices) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_26(mht_26_v, 830, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::ValidateNonTuple");

  if (shape.IsTuple()) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Validation shape is a tuple but sharding is not."));
  }
  if (replicated_) {
    return Status::OK();
  }

  // All tile assignments must be less than the number of available cores and
  // unique.
  Status status = Status::OK();
  absl::flat_hash_set<int64_t> seen_cores;
  tile_assignment_.Each([&](absl::Span<const int64_t> indices, int32_t core) {
    // Don't overwrite a bad status, so we report the first error.
    if (status.ok()) {
      if (core >= num_devices) {
        status = tensorflow::errors::InvalidArgument(
            StrCat("core ", core, " > ", num_devices, " in tile assignment"));
      } else if (seen_cores.contains(core)) {
        status = tensorflow::errors::InvalidArgument(
            StrCat("core ", core, " is not unique in tile assignment"));
      }
      seen_cores.insert(core);
    }
  });
  if (!status.ok()) {
    return status;
  }

  if (IsTileMaximal() || IsManual()) {
    return Status::OK();
  }

  // The tile assignment tensor must have the same rank as the input, or input
  // rank + 1 for replicate_on_last_tile_dim_.
  if (shape.rank() + (replicate_on_last_tile_dim_ ? 1 : 0) +
          subgroup_types_.size() !=
      tile_assignment_.num_dimensions()) {
    return tensorflow::errors::InvalidArgument(
        "Number of tile assignment dimensions is different to the input rank. "
        "sharding=",
        ToString(), ", input_shape=", ShapeUtil::HumanString(shape));
  }

  // The correct constructor has to be used to create tile maximal shardings.
  if (tile_assignment_.num_elements() == 1) {
    return tensorflow::errors::InvalidArgument(
        "Tile assignment only contains a single device. If a replicated "
        "sharding was intended, use HloSharding::Replicated(). If a device "
        "placement was intended, use HloSharding::AssignDevice()");
  }
  return Status::OK();
}

/*static*/ StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_27(mht_27_v, 889, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::FromProto");

  std::vector<OpMetadata> metadata(proto.metadata().begin(),
                                   proto.metadata().end());
  std::vector<int> subgroup_types_int(proto.last_tile_dims().begin(),
                                      proto.last_tile_dims().end());
  std::vector<OpSharding::Type> subgroup_types;
  absl::c_transform(
      subgroup_types_int, std::back_inserter(subgroup_types),
      [](const int type) { return static_cast<OpSharding::Type>(type); });
  if (proto.type() == OpSharding::TUPLE) {
    TF_RET_CHECK(metadata.empty())
        << "Tuple sharding is expected to have no metadata.";
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(sharding);
    }
    return HloSharding(tuple_shardings);
  } else if (proto.type() == OpSharding::REPLICATED) {
    return Replicate(metadata);
  } else if (proto.type() == OpSharding::MANUAL) {
    return Manual(metadata);
  } else if (proto.tile_assignment_devices().size() == 1) {
    return HloSharding(proto.tile_assignment_devices(0), metadata);
  }

  TF_RET_CHECK(proto.type() != OpSharding::MAXIMAL)
      << "Maximal sharding is expected to have single device assignment, but "
      << proto.tile_assignment_devices().size() << " has provided.";

  TF_RET_CHECK(proto.tile_assignment_devices().size() > 1);
  TF_RET_CHECK(!proto.tile_assignment_dimensions().empty());

  // RE: the product of tile assignment tensor dimensions must be
  // equal to tile_assignment_devices.size().
  int64_t product_of_dimensions = 1;
  for (auto dimension : proto.tile_assignment_dimensions()) {
    TF_RET_CHECK(dimension > 0);
    product_of_dimensions =
        MultiplyWithoutOverflow(product_of_dimensions, dimension);
    TF_RET_CHECK(product_of_dimensions > 0);
  }
  TF_RET_CHECK(product_of_dimensions == proto.tile_assignment_devices().size());

  // Some versions of gcc cannot infer the TileAssignment constructor from a
  // braced initializer-list, so create one manually.
  std::vector<int64_t> devices(proto.tile_assignment_devices().begin(),
                               proto.tile_assignment_devices().end());
  Array<int64_t> tile_assignment(
      std::vector<int64_t>(proto.tile_assignment_dimensions().begin(),
                           proto.tile_assignment_dimensions().end()));
  std::copy(proto.tile_assignment_devices().begin(),
            proto.tile_assignment_devices().end(), tile_assignment.begin());
  if (!subgroup_types.empty()) {
    TF_RET_CHECK(!proto.replicate_on_last_tile_dim());
    return Subgroup(tile_assignment, subgroup_types, metadata);
  }
  return proto.replicate_on_last_tile_dim()
             ? PartialTile(tile_assignment, metadata)
             : HloSharding(tile_assignment,
                           /*replicate_on_last_tile_dim=*/false, metadata);
}

OpSharding HloSharding::ToProto() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_28(mht_28_v, 957, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::ToProto");

  OpSharding result;

  if (IsTuple()) {
    CHECK(metadata_.empty());
    for (const HloSharding& element : tuple_elements_) {
      *result.add_tuple_shardings() = element.ToProto();
    }
    result.set_type(OpSharding::TUPLE);
    return result;
  }

  result.mutable_metadata()->Reserve(metadata_.size());
  for (const auto& metadata : metadata_) {
    *result.add_metadata() = metadata;
  }

  for (int64_t dim : tile_assignment_.dimensions()) {
    result.add_tile_assignment_dimensions(dim);
  }
  for (auto device : tile_assignment_) {
    result.add_tile_assignment_devices(device);
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::REPLICATED);
    result.clear_tile_assignment_dimensions();
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else if (IsManual()) {
    result.set_type(OpSharding::MANUAL);
    result.clear_tile_assignment_dimensions();
  } else {
    result.set_type(OpSharding::OTHER);
    result.set_replicate_on_last_tile_dim(ReplicateOnLastTileDim());
    for (auto type : subgroup_types_) {
      result.add_last_tile_dims(type);
    }
  }
  return result;
}

Shape HloSharding::TileShape(const Shape& shape) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_29(mht_29_v, 1001, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::TileShape");

  if (IsTileMaximal() || IsManual()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64_t i = 0; i < TiledDataRank(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64_t>(shape.dimensions(i), tile_assignment_.dim(i)));
  }
  return result_shape;
}

Shape HloSharding::TileShape(const Shape& shape, int64_t device) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_30(mht_30_v, 1016, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::TileShape");

  if (IsTileMaximal() || IsManual()) {
    return shape;
  }

  std::vector<int64_t> index = TileIndexForDevice(device);
  Shape result_shape = shape;
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    int64_t offset = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
    int64_t limit = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
    result_shape.set_dimensions(i, limit - offset);
  }
  return result_shape;
}

int64_t HloSharding::NumTiles() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_31(mht_31_v, 1038, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::NumTiles");

  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions())
                     .subspan(0, TiledDataRank()));
}

int64_t HloSharding::NumTiles(absl::Span<const int64_t> dims) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_32(mht_32_v, 1050, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::NumTiles");

  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!ReplicateOnLastTileDim() ||
        !absl::c_linear_search(dims, tile_assignment().num_dimensions() - 1));
  int64_t num_tiles = 1;
  for (auto d : dims) {
    CHECK(d < tile_assignment().num_dimensions());
    num_tiles *= tile_assignment().dim(d);
  }
  return num_tiles;
}

HloSharding HloSharding::GetSubSharding(const Shape& shape,
                                        const ShapeIndex& index) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_33(mht_33_v, 1069, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::GetSubSharding");

  CHECK(IsTuple());
  int64_t sharding_index = 0;
  const Shape* sub_shape = &shape;
  for (int64_t idx : index) {
    for (int64_t i = 0; i < idx; ++i) {
      sharding_index +=
          ShapeUtil::GetLeafCount(ShapeUtil::GetSubshape(*sub_shape, {i}));
    }
    sub_shape = &ShapeUtil::GetSubshape(*sub_shape, {idx});
  }
  if (sub_shape->IsTuple()) {
    auto begin_it = tuple_elements_.begin() + sharding_index;
    std::vector<HloSharding> sub_shardings(
        begin_it, begin_it + ShapeUtil::GetLeafCount(*sub_shape));
    return HloSharding::Tuple(*sub_shape, sub_shardings);
  } else {
    return tuple_elements_[sharding_index];
  }
}

absl::optional<HloSharding> HloSharding::ExtractSingleSharding() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_34(mht_34_v, 1093, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::ExtractSingleSharding");

  if (!IsTuple()) {
    return *this;
  }
  if (tuple_elements_.empty()) {
    return absl::nullopt;
  }
  for (int64_t i = 1; i < tuple_elements_.size(); ++i) {
    if (tuple_elements_[0] != tuple_elements_[i]) {
      return absl::nullopt;
    }
  }
  return tuple_elements_.front();
}

HloSharding HloSharding::WithMetadata(absl::Span<const OpMetadata> metadata,
                                      bool overwrite) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_35(mht_35_v, 1112, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::WithMetadata");

  auto assign_metadata = [&](HloSharding& sharding) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_36(mht_36_v, 1116, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "lambda");

    if (sharding.metadata_.empty() || overwrite) {
      sharding.metadata_.assign(metadata.begin(), metadata.end());
    }
  };

  HloSharding sharding = *this;
  if (sharding.IsTuple()) {
    for (HloSharding& sub_sharding : sharding.tuple_elements()) {
      assign_metadata(sub_sharding);
    }
  } else {
    assign_metadata(sharding);
  }
  return sharding;
}

HloSharding HloSharding::WithoutMetadata() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_37(mht_37_v, 1136, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "HloSharding::WithoutMetadata");

  HloSharding sharding = *this;
  sharding.metadata_.clear();
  for (HloSharding& sub_sharding : sharding.tuple_elements()) {
    sub_sharding.metadata_.clear();
  }
  return sharding;
}

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_shardingDTcc mht_38(mht_38_v, 1148, "", "./tensorflow/compiler/xla/service/hlo_sharding.cc", "operator<<");

  out << sharding.ToString();
  return out;
}

}  // namespace xla
