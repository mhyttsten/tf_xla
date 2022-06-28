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

#ifndef TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_
#define TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_
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
class MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh {
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
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh() {
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


#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

// Definitions for DTensor mesh & layout.
//
// A mesh describes how a set of devices is partitioned.
// A layout describes how a distributed tensor is partitioned across a mesh (and
// thus across devices). Defining tensor layouts in terms of mesh dimensions
// allows us to efficiently determine the communication required when computing
// an operation with tensors of different layouts.
namespace tensorflow {
namespace dtensor {

// The location of a device in a mesh.
//
// Each device has a unique location in the mesh, which is indicated by the
// offset in each mesh dimension. e.g. a mesh:
//
// [x:4, y:3, z:2]
//
// Must consist of 24 devices placed densely into the corresponding 3D space.
using DeviceLocation = absl::InlinedVector<int64, 4>;

// A shard refers to a partition of a tensor. Shards are arranged in
// ShardVectors that contains a list of Shards and a list of integers
// representing the number of shards in each dimension.
//
// Example: layout = sharding_specs:x,y, mesh:|x=2,y=2|. This can be represented
// with a ShardVector:
//          - shards = (1,1), (1,2), (2,1), (2,2)
//          - num_shards_per_dim = (2,2).
//
// The number of elements in each shard matches the tensor rank.
using Shard = std::vector<int>;

struct ShardVector {
  bool operator==(const ShardVector& other) const;
  bool operator!=(const ShardVector& other) const { return !(*this == other); }
  std::string ToString() const;

  bool ContainsShard(const Shard& shard) const;

  std::vector<Shard> shards;
  std::vector<int> num_shards_per_dim;
};

struct MeshDimension {
  MeshDimension(const std::string& name, int64 size)
      : name(std::move(name)), size(size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_0(mht_0_v, 251, "", "./tensorflow/dtensor/cc/tensor_layout.h", "MeshDimension");
}
  MeshDimension() = default;

  std::string name;
  int64 size;
};

class Mesh {
 public:
  // Failed serialized strings are represented with en empty string, therefore
  // we use this string representation of an empty mesh instead to avoid
  // confusion.
  static constexpr const char* kEmptyMeshString = "empty_mesh";
  static Mesh Empty();
  // Parses from MeshProto.
  static StatusOr<Mesh> ParseFromProto(const MeshProto& proto);
  // Parses from a human readable string version of the mesh, currently used
  // to represent meshes in MLIR:
  //  mesh = <name|List[MeshDim]|List[GlobalId]|List[LocalId]|List[Devices]>
  //
  // Example:
  //  mesh =
  //  <name|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  static StatusOr<Mesh> FromString(const std::string& str);
  // Creates mesh without specific devices associated to it (aka abstract mesh).
  // This is an experimental API. Use only if strictly needed.
  static StatusOr<Mesh> GetAbstractMesh(
      const std::string& name, const std::vector<MeshDimension>& mesh_dims);
  // Creates fully defined mesh.
  static StatusOr<Mesh> GetMesh(
      const std::string& name, const std::vector<MeshDimension>& mesh_dims,
      const std::vector<std::int64_t>& global_device_ids,
      const std::vector<std::int64_t>& local_device_ids,
      const std::vector<std::string>& local_devices,
      const std::vector<std::string>& global_devices);

  Mesh() = default;

  bool IsEmpty() const;

  // Device information methods.
  std::string device_type() const;
  std::vector<std::string> hosts() const;

  bool is_cpu_mesh() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_1(mht_1_v, 298, "", "./tensorflow/dtensor/cc/tensor_layout.h", "is_cpu_mesh");
 return device_type() == "CPU"; }
  bool is_epu_mesh() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_2(mht_2_v, 302, "", "./tensorflow/dtensor/cc/tensor_layout.h", "is_epu_mesh");
 return device_type() == "EPU"; }
  bool is_tpu_mesh() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_3(mht_3_v, 306, "", "./tensorflow/dtensor/cc/tensor_layout.h", "is_tpu_mesh");
 return device_type() == "TPU"; }

  // Returns the index of MeshDimension in mesh where the mesh dimension name is
  // `mesh_name`.
  int GetMeshDimIndexWithName(const std::string& mesh_name) const;
  bool IsMeshDim(const std::string& dim_name) const;

  const std::string& name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_4(mht_4_v, 316, "", "./tensorflow/dtensor/cc/tensor_layout.h", "name");
 return name_; }
  const MeshDimension& dim(int64 index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_5(mht_5_v, 320, "", "./tensorflow/dtensor/cc/tensor_layout.h", "dim");
 return mesh_dims_[index]; }
  const std::string& dim_name(int64 index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_6(mht_6_v, 324, "", "./tensorflow/dtensor/cc/tensor_layout.h", "dim_name");

    return mesh_dims_[index].name;
  }

  // Consumes a location in the mesh and returns its corresponding index in
  // the flattened list of devices.
  int64 GetFlattenedCoordinate(const DeviceLocation& loc) const;
  // Takes an index in the flattened list of devices and returns a location
  // in the mesh.
  StatusOr<const DeviceLocation> device_location(int offset) const;

  std::vector<MeshDimension> dims() const { return mesh_dims_; }

  // Parses names of local_devices according to TF's Device Name Utils.
  StatusOr<const std::vector<DeviceNameUtils::ParsedName>> ParsedDevices()
      const;
  absl::Span<const std::string> local_devices() const { return local_devices_; }
  absl::Span<const int64_t> local_device_ids() const {
    return local_device_ids_;
  }

  int64_t min_global_device_id() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_7(mht_7_v, 348, "", "./tensorflow/dtensor/cc/tensor_layout.h", "min_global_device_id");

    DCHECK(!global_device_ids_.empty());
    return *std::min_element(global_device_ids_.begin(),
                             global_device_ids_.end());
  }

  absl::Span<const int64_t> global_device_ids() const {
    return global_device_ids_;
  }

  const std::vector<std::string>& global_devices() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_8(mht_8_v, 361, "", "./tensorflow/dtensor/cc/tensor_layout.h", "global_devices");

    return global_devices_;
  }

  // Returns whether the mesh is a remote mesh.
  bool is_remote() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_9(mht_9_v, 369, "", "./tensorflow/dtensor/cc/tensor_layout.h", "is_remote");

    return local_device_ids_.empty() && !global_device_ids_.empty();
  }

  // Returns index of given dim_name in the mesh.
  StatusOr<int32> idx_for_dim(absl::string_view dim_name) const;

  // Returns size of mesh dimension.
  StatusOr<int64> dim_size(absl::string_view name) const;

  // Returns list of mesh dimension sizes.
  std::vector<int64> dim_sizes() const;

  int64 num_devices() const;
  int64 rank() const;
  int64 size() const;

  std::string ToString() const;

  // Global unique fingerprint. Same on different workers.
  uint64 GlobalFingerprint() const;

  bool operator==(const Mesh& b) const;
  bool operator!=(const Mesh& b) const { return !((*this) == b); }
  bool operator<(const Mesh& b) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_10(mht_10_v, 396, "", "./tensorflow/dtensor/cc/tensor_layout.h", "operator<");

    return this->ToString() < b.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const Mesh& m) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_11(mht_11_v, 404, "", "./tensorflow/dtensor/cc/tensor_layout.h", "AbslHashValue");

    return H::combine(std::move(h), m.ToString());
  }

  MeshProto ToProto() const;

  // A map from mesh names to their corresponding core ID mappings. The core ID
  // mapping is stored as a vector. The i-th element in the vector is the ID of
  // the core represented by global device ID of i in this mesh.
  //
  // The entry stored under the empty name key (the so-called "default mapping"
  // in some comments) is special. Is is always set at the end of TPU
  // initialization. It represents the mapping for any mesh whose global device
  // IDs follow TF task-device ordinals. Legacy and test meshes created without
  // using the `create_tpu_mesh` helper follow that rule and can use this entry.
  static std::map<std::string, std::vector<int>>& tpu_core_ids();

  // The host mesh associated with any user-defined TPU mesh.
  static std::string& tpu_host_mesh();

 private:
  std::string name_;
  std::vector<MeshDimension> mesh_dims_;
  std::vector<std::string> local_devices_;
  std::vector<int64_t> local_device_ids_;
  std::vector<int64_t> global_device_ids_;
  std::vector<std::string> global_devices_;
};

class Layout {
 public:
  static constexpr const char* kUnshardedDim = "unsharded";
  // This spec should only be used to express no preferred sharding in the
  // Layout propagation algorithm.
  static constexpr const char* kAny = "any";
  // Failed serialized strings are represented with en empty string, therefore
  // we use this string representation of an empty layout instead to avoid
  // confusion.
  static constexpr const char* kEmptyLayoutString = "empty_layout";
  // Used for the relayout operation, to allow relayout act as an identity on
  // the layout for the given dimension.
  static constexpr const char* kMatch = "match";

  // Returns empty layout.
  static Layout Empty();

  // Parses from LayoutProto.
  static StatusOr<Layout> FromProto(const LayoutProto& proto);
  // Parses from a human readable string version of the layout, currently used
  // to represent layouts in MLIR:
  //  layout = <sharding_specs:List[specs] mesh:name|List[MeshDim]|
  //  List[GlobalId]|List[LocalId]|List[Devices]>
  //
  // Example:
  //  layout = <sharding_specs:x,not_sharded mesh:name|x=2,y=2|0,1,2,3|0,1,2,3|
  //  /job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,
  //  /job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  static StatusOr<Layout> FromString(std::string layout_str);
  static Layout ReplicatedOnMesh(const Mesh& mesh, int rank);
  static Layout AnyOnMesh(const Mesh& mesh, int rank);

  // Returns a layout for the transposed matrix for given layout. This assumes
  // that only the last two dimensions are used for matrix computation and all
  // dimensions before are batch dimensions.
  static StatusOr<Layout> Transposed2D(const Layout& layout);
  static bool IsUnshardedDimension(const absl::string_view name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_12(mht_12_v, 473, "", "./tensorflow/dtensor/cc/tensor_layout.h", "IsUnshardedDimension");

    return name == kUnshardedDim;
  }
  static bool IsShardedDimension(const absl::string_view name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_13(mht_13_v, 480, "", "./tensorflow/dtensor/cc/tensor_layout.h", "IsShardedDimension");

    return !IsUnshardedDimension(name);
  }
  static bool IsUnshardedSpec(const ShardingSpec& spec) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_14(mht_14_v, 486, "", "./tensorflow/dtensor/cc/tensor_layout.h", "IsUnshardedSpec");

    return IsUnshardedDimension(spec.sharding_spec());
  }
  static bool IsShardedSpec(const ShardingSpec& spec) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_15(mht_15_v, 492, "", "./tensorflow/dtensor/cc/tensor_layout.h", "IsShardedSpec");

    return !IsUnshardedDimension(spec.sharding_spec());
  }
  static StatusOr<Layout> GetLayout(
      const std::vector<std::string>& sharding_spec_strs, const Mesh& mesh);
  static StatusOr<Layout> GetLayout(
      const std::vector<ShardingSpec>& sharding_specs, const Mesh& mesh);

  // Makes a new layout from this one dropping the given dimensions.
  // If keep_dims is true, the dimensions are replicated rather than
  // deleted.
  Layout GetLayoutWithReducedDims(const absl::flat_hash_set<int>& reduced_dims,
                                  bool keep_dims) const;

  // Truncates a layout at the front or back, depending on the value of end.
  // end = false returns the layout upto the split point,
  // end = true returns the layout from the split point.
  Layout Truncate(int64 split_point, bool end = false) const;

  // Left or right pad the layout to a max rank.
  Layout LeftPad(int64 rank) const;

  bool IsFullyReplicated() const;
  bool IsLastDimReplicated() const;
  // Checks that the last N-1 dimensions are replicated
  bool IsBatchParallel() const;
  // Checks that the dimensions from [-non_batch_rank, end) are replicaed
  bool IsBatchParallel(int non_batch_rank) const;
  bool IsEmpty() const;

  // Compute global shape using the layout and provided local_shape.
  std::vector<int64_t> GlobalShapeFromLocalShape(
      const std::vector<int64_t>& local_shape) const;

  std::vector<int64_t> LocalShapeFromGlobalShape(
      absl::Span<const int64_t> global_shape) const;
  PartialTensorShape LocalShapeFromGlobalShape(
      const PartialTensorShape& global_shape) const;

  int64 rank() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_16(mht_16_v, 534, "", "./tensorflow/dtensor/cc/tensor_layout.h", "rank");
 return sharding_specs_.size(); }
  int64 num_devices() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_17(mht_17_v, 538, "", "./tensorflow/dtensor/cc/tensor_layout.h", "num_devices");
 return mesh_.num_devices(); }
  size_t num_shards_for_dim(const ShardingSpec& dim) const;
  std::vector<int32> num_shards() const;

  const ShardingSpec& dim(int64 idx) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_18(mht_18_v, 545, "", "./tensorflow/dtensor/cc/tensor_layout.h", "dim");
 return sharding_specs_[idx]; }
  absl::Span<const ShardingSpec> sharding_specs() const {
    return sharding_specs_;
  }

  // Creates a mesh of unique shards.
  Mesh ReducedMesh() const;

  // Computes the corresponding shard vector to this layout.
  ShardVector GetShardVector() const;

  // Map hosts to shards.
  std::map<std::string, ShardVector> HostShardMap() const;

  // Returns sharding specs in string form.
  std::vector<std::string> sharding_spec_strs() const;

  // Creates human readable string version of a layout.
  std::string ToString() const;
  LayoutProto ToProto() const;

  StatusOr<const DeviceLocation> device_location(int64 device_id) const {
    return mesh_.device_location(device_id);
  }

  void set_mesh(Mesh mesh) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_19(mht_19_v, 573, "", "./tensorflow/dtensor/cc/tensor_layout.h", "set_mesh");
 mesh_ = mesh; }

  const Mesh& mesh() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_20(mht_20_v, 578, "", "./tensorflow/dtensor/cc/tensor_layout.h", "mesh");
 return mesh_; }
  const std::string& sharding_spec(int idx) const;

  bool operator==(const Layout& b) const;
  bool operator!=(const Layout& b) const { return !((*this) == b); }
  bool operator<(const Layout& b) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSdtensorPSccPStensor_layoutDTh mht_21(mht_21_v, 586, "", "./tensorflow/dtensor/cc/tensor_layout.h", "operator<");

    return this->ToString() < b.ToString();
  }

 private:
  std::vector<ShardingSpec> sharding_specs_;
  Mesh mesh_;
};

// Takes two layouts and concatenates their TensorDimensions. If the meshes for
// the two layouts are different or both layouts are using the same mesh
// dimension returns an error rather than a layout.
StatusOr<Layout> ConcatenateLayouts(const Layout& layout_a,
                                    const Layout& layout_b);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_
