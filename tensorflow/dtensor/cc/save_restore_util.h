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

#ifndef TENSORFLOW_DTENSOR_CC_SAVE_RESTORE_UTIL_H_
#define TENSORFLOW_DTENSOR_CC_SAVE_RESTORE_UTIL_H_
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
class MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTh {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTh() {
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


#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

// Defines an Metadata entry when saving a Tensor.
struct SavingTensorMetadata {
  // Tracks index from the original save op.
  int64_t tensor_index;
  // The global shape of the saving tensor.
  std::vector<int64_t> shape;
  // The layout of the saving tensor.
  Layout layout;

  SavingTensorMetadata(int64_t index, std::vector<int64_t> global_shape,
                       Layout tensor_layout)
      : tensor_index(index),
        shape(std::move(global_shape)),
        layout(std::move(tensor_layout)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTh mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/cc/save_restore_util.h", "SavingTensorMetadata");
}
};

// Tracks a complete specification for a particular save op.
// The users would build out multiple save ops using the following manner for
// the given fields:
//
// save_op[i] = tf.SaveV2(
//              prefix = new_prefixes[i],
//              tensor_indices = tensor_indies[i],
//              shape_and_slices = shape_and_slice_spec[i])
struct SaveOpSpecs {
  std::vector<std::string> new_prefixes;
  std::vector<std::vector<int>> tensor_indices;
  std::vector<std::vector<std::string>> shape_and_slice_spec;

  SaveOpSpecs(std::vector<std::string> prefixes,
              std::vector<std::vector<int>> indices,
              std::vector<std::vector<std::string>> specs)
      : new_prefixes(std::move(prefixes)),
        tensor_indices(std::move(indices)),
        shape_and_slice_spec(std::move(specs)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSsave_restore_utilDTh mht_1(mht_1_v, 235, "", "./tensorflow/dtensor/cc/save_restore_util.h", "SaveOpSpecs");
}
};

// Builds a complete saving specification for each device on the mesh.
//
// The returned map contains a map of <device_id, SavingSpec>.
// Device_id is where the saving should happen, and SavingSpec is a
// mapping of <tensor_index -> shape_and_slices>. e.g.,
//
// A map of {device_id : 0 ->  {
//     0 : "2 0,1",
//     1 : ""
//   }
// }
//
// Means that device_0 is responsible for saving tensor 0 and 1 from the passed
// in tensors list. For tensor[0], it saves the only the first element in that
// 1d vector with 2 elements. For tensor[1], it saves all elements.
//
// We accept another map as input, that records the mapping of
// <tensor_index -> (tensor_global_shape, tensor_layout)>.
//
// (tensor_global_shape, tensor_layout & tensor_layout.mesh) defines which
// device saves what slices of the Tensor.
//
// For a complete definition of shape_and_slices field, please see:
// third_party/tensorflow/core/framework/tensor_slice.h
StatusOr<absl::flat_hash_map<
    int64_t, absl::flat_hash_map<int64_t, std::vector<std::string>>>>
BuildSavingSpec(absl::Span<const SavingTensorMetadata> tensor_metadatas);

// For a given per device saving spec, find out the counts of SaveV2 ops
// needed and their corresponding inputs.
//
// Current SaveV2 op requires tensor_names to be unique in the list, which is a
// contract that distributed saving would break. For example, if the saving spec
// decides that device 0 is responsible for saving two slices of tensor[a], then
// a single SaveV2 op can't fufill. The setup is very likely to happen when
// saving on TPU - where 8 cores maps to 1 host. In that case, the CPU host will
// be responsible for saving slices on the same tensor across 8 TPU cores.
// TODO(b/179126981): Investigate whether we can make TF core API run with
// different slice spec on a same tensor key.
//
// That said, building one SaveV2 op for each save is wasteful, when a single
// SaveV2 op is capable of saving different tensors. Instead, we simply need to
// break the SaveV2 op to be able to track the longest saving specs for a single
// tensor happening on the device,  e.g.,
//
// For given saving specs:
//
// { 'tensor_name_a' : <"spec_a", "spec_a_2"> }
// { 'tensor_name_b' : <"spec_b"> }
//
// would result into two save ops, where:
//
// SaveOp1 (tensor_names = <"tensor_name_a", tensor_name_b">,
//                           slice_spec = <"spec_a", "spec_b">)
//
// SaveOp2 (tensor_names = "<tensor_name_a>", slice_spec = <"spec_a_2">.
//
// The output vectors tracks the new SaveV2 op parameters and they must agree on
// size and indexing for saving tensors.
//
// tensor_indices trackes a list of indices of tensors that are being saved for
// each Save op, e.g.,
//
// tensor_indices[0] is a list of tensors (in index form) that needs to be saved
// on the first SaveV2 op.
//
// shape_and_slice_specs tracks a list of shape_and_slice_specs being saved for
// each Save op, e.g.,
//
// shape_and_slice_spec[0] is a list of shape_and_slices parameters for SaveV2
// op.
SaveOpSpecs BuildPerDeviceSave(
    const absl::flat_hash_map<int64_t, std::vector<std::string>>& saving_spec,
    int device_id, absl::string_view prefix);

// Figures out the tensor slice_spec for a given layout and mesh device
// location.
StatusOr<std::vector<std::string>> SliceSpecOnDevice(
    const Layout& layout, const Mesh& mesh, const DeviceLocation& device_coords,
    absl::Span<const int64_t> global_shape);
}  // namespace dtensor

}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_SAVE_RESTORE_UTIL_H_
