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
class MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTcc() {
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

#include "tensorflow/dtensor/cc/save_restore_util.h"

namespace tensorflow {
namespace dtensor {

namespace {
// A map that is keyed by the index of tensor_name.
// For example, {2 : <"spec_a", "spec_b"> } means that the
// save_v2.tensor_names[2] should have "spec_a" and "spec_b" saved.
using SliceSpecByName = absl::flat_hash_map<int64_t, std::vector<std::string>>;

// Builds a map from tensor slice spec to saving device_id for the given Tensor
// and layout. The output would record the saving device and the slices it needs
// to save.
//
// For each sharded Tensor, each device would hold a slice of the Tensor - but
// it isn't necessary a unique copy. For a 2 way sharded Tensor in a (2,4) mesh
// on the first dimension, device [0-3] and device [4-7] will hold the same
// slice data. To avoid saving duplicated copies of the Tensor slice, the map
// would only contain the min(device_id) that occupies the slice and save from
// there.
//
// Furthermore, to save a Tensor that isn't on CPU mesh, send/recv is necessary
// from saving device to its corresponding host(CPU) devices. Since we don't
// have multi-mesh execution yet, this isn't implemented yet.
StatusOr<SliceSpecByName> BuildSliceSpecDeviceMap(
    absl::Span<const int64_t> global_shape, Layout layout) {
  if (!layout.mesh().is_cpu_mesh())
    return errors::Unimplemented(
        "Saving tensors on non CPU mesh needs explicit send/receive and isn't "
        "implemented yet");

  // Result map that records the minimum device_id that occupies the unique
  // copy.
  // Note that llvm::SmallDenseMap won't accept std::string as a key.
  absl::flat_hash_map<std::string, int64_t> min_device_for_slice_spec;
  // Records the map of device_ids and a list of slice_spec that it needs to
  // save.
  SliceSpecByName device_slices;

  const auto& mesh = layout.mesh();
  // Construct SliceSpec for each device in the mesh.
  for (int device_id = 0; device_id < mesh.size(); ++device_id) {
    TF_ASSIGN_OR_RETURN(const DeviceLocation& coords,
                        mesh.device_location(device_id));
    // Prefill with full spec on each dim.
    TF_ASSIGN_OR_RETURN(std::vector<std::string> slice_specs,
                        SliceSpecOnDevice(layout, mesh, coords, global_shape));

    // Build the real slice_spec from string pieces.
    std::string slice_spec = absl::StrJoin(slice_specs, ":");
    // Get local shape from the global shape.
    std::string shape_spec = absl::StrJoin(global_shape, " ");
    // Concat shape spec and slice spec to form a complete shape_and_slice.
    std::string shape_and_slice = absl::StrCat(shape_spec, " ", slice_spec);

    // Only record the min device_id for the unique slice_spec on a given
    // Tensor.
    if (min_device_for_slice_spec.find(shape_and_slice) ==
            min_device_for_slice_spec.end() ||
        device_id < min_device_for_slice_spec[shape_and_slice]) {
      min_device_for_slice_spec[shape_and_slice] = device_id;
    }
  }

  // Constructs device_id keyed map for future save operation conditioned on
  // device_ids.
  for (const auto& spec_and_id : min_device_for_slice_spec) {
    device_slices[spec_and_id.second].push_back(spec_and_id.first);
  }

  return device_slices;
}

}  // namespace

StatusOr<absl::flat_hash_map<
    int64_t, absl::flat_hash_map<int64_t, std::vector<std::string>>>>
BuildSavingSpec(absl::Span<const SavingTensorMetadata> tensor_metadatas) {
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<int64_t, std::vector<std::string>>>
      saving_specs;
  for (const SavingTensorMetadata& tensor_metadata : tensor_metadatas) {
    // We use index to select the tensor names and shape_and_slices from the
    // inputs. This is generic regardless whether the inputs are constants or
    // just arguments.
    int index = tensor_metadata.tensor_index;
    const Layout& layout = tensor_metadata.layout;
    absl::Span<const int64_t> tensor_shape = tensor_metadata.shape;

    if (layout.IsFullyReplicated()) {
      // Push a fully replicated save on device 0, where slice_spec is simply
      // empty string.
      saving_specs[0][index].push_back("");
    } else {
      // Calculate shape_and_slices for sharded case here.
      TF_ASSIGN_OR_RETURN(const auto& slice_specs,
                          BuildSliceSpecDeviceMap(tensor_shape, layout));
      // Push specs for each device into the global map.
      for (const auto& slice_spec : slice_specs) {
        int64_t saving_device_id = slice_spec.first;
        for (const std::string& slice : slice_spec.second) {
          saving_specs[saving_device_id][index].push_back(slice);
        }
      }
    }
  }

  return saving_specs;
}

SaveOpSpecs BuildPerDeviceSave(
    const absl::flat_hash_map<int64_t, std::vector<std::string>>& saving_spec,
    const int device_id, absl::string_view prefix) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTcc mht_0(mht_0_v, 299, "", "./tensorflow/dtensor/cc/save_restore_util.cc", "BuildPerDeviceSave");

  std::vector<std::string> new_prefixes;
  std::vector<std::vector<int>> tensor_indices;
  std::vector<std::vector<std::string>> shape_and_slice_specs;
  for (const auto& tensor_name_index_and_slice_specs : saving_spec) {
    int tensor_index = tensor_name_index_and_slice_specs.first;
    const std::vector<std::string> specs =
        tensor_name_index_and_slice_specs.second;
    // For each tensor_name, we save its first slice_spec in the first
    // save_op, second slice_spec in the second save op, etc.
    // This allows us to group save ops together without running into
    // duplicated tensor_names (which save_v2 op doesn't support).
    for (int save_op_index = 0; save_op_index < specs.size(); ++save_op_index) {
      if (save_op_index >= tensor_indices.size()) {
        tensor_indices.push_back({});
        shape_and_slice_specs.push_back({});
        // Generate new prefix based on device_id and save op index, only when
        // we need a new save_op.
        new_prefixes.push_back(absl::StrCat(prefix, "_device_", device_id,
                                            "_save_op_", save_op_index));
      }
      tensor_indices[save_op_index].push_back(tensor_index);
      shape_and_slice_specs[save_op_index].push_back(specs[save_op_index]);
    }
  }

  return SaveOpSpecs(new_prefixes, tensor_indices, shape_and_slice_specs);
}

StatusOr<std::vector<std::string>> SliceSpecOnDevice(
    const Layout& layout, const Mesh& mesh, const DeviceLocation& device_coords,
    absl::Span<const int64_t> global_shape) {
  // Prefill the slice with replicated layouts.
  std::vector<std::string> slice_specs(global_shape.size(), "-");

  const std::vector<std::string>& sharding_spec_strs =
      layout.sharding_spec_strs();
  for (int tensor_dim_index = 0; tensor_dim_index < sharding_spec_strs.size();
       ++tensor_dim_index) {
    const std::string& mesh_dim = sharding_spec_strs[tensor_dim_index];
    if (layout.IsShardedDimension(mesh_dim)) {
      TF_ASSIGN_OR_RETURN(int mesh_dim_index, mesh.idx_for_dim(mesh_dim));
      TF_ASSIGN_OR_RETURN(int64_t dim_size, mesh.dim_size(mesh_dim));
      int64_t per_slice_size = global_shape[tensor_dim_index] / dim_size;
      int start = device_coords[mesh_dim_index] * per_slice_size;
      slice_specs[tensor_dim_index] = absl::StrCat(start, ",", per_slice_size);
    }
  }
  return slice_specs;
}

}  // namespace dtensor
}  // namespace tensorflow
