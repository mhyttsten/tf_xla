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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh() {
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


#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/types.h"

// TODO(jblespiau): The current implementation moves the Python logic to C++,
// as a preliminary step to executing the `pmap` execution path from C++.
// It implements the current Python behavior (thus, it may not be optimal, and
// we will be able to modify it later).

namespace jax {

// High level introduction.
//
// pmap and other parallel computation functions distribute some computation on
// several devices. On December 2020, the devices mesh (i.e. N-dimentional array
// of devices on which we map the computation) is defined by the user.
//
// We describe how to shard the inputs, and how to map it to the mesh of devices
// using `ShardingSpec`. It's mainly based on 2 components:
// - `sharding`, which specifies how to shard the inputs.
// - `mesh_mapping`, which specifies how to map shards to devices.
//
// The 3 following structs define how to shard one dimension of an ndarry.
//
// `NoSharding` (`None` in Python) means no sharding.
struct NoSharding {
  bool operator==(const NoSharding& other) const { return true; }
  bool operator!=(const NoSharding& other) const { return false; }
};

template <typename H>
H AbslHashValue(H h, const NoSharding& key) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_0(mht_0_v, 228, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  return h;
}

// `Chunked` means that the dimension is split into np.prod(chunks) chunks
// and the split dimension itself is preserved inside the map.
// Those chunks are distributed over `len(chunks)` ShardedAxes axes
// (major-to-minor).
// For example, for a tensor `t` of shape [N] sharded using [Chunked([p])] (with
// p  dividing N, let S = N // p) the tensor will be split into p chunks of
// shape [S], such sharded_t[k] = t[k * S: (k+1)*S] (left included, right
// excluded) for k in {0, ... p-1}.
struct Chunked {
 public:
  explicit Chunked(std::vector<int> chunks_) : chunks(std::move(chunks_)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_1(mht_1_v, 245, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "Chunked");
}
  // The number of chunks per axis.
  std::vector<int> chunks;

  bool operator==(const Chunked& other) const { return chunks == other.chunks; }
  bool operator!=(const Chunked& other) const { return chunks != other.chunks; }
};

template <typename H>
H AbslHashValue(H h, const Chunked& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_2(mht_2_v, 257, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  h = H::combine(std::move(h), key.chunks);
  return h;
}

// `Unstacked` means that the dimension is split into chunks of size 1, and
// doesn't appear inside the map. `size` is always the dimension size.
// For example, a Tensor t of shape [N] will be sharded into N tensors of shape
// [], when using `Unstacked(N)`.
struct Unstacked {
 public:
  explicit Unstacked(int sz) : size(sz) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "Unstacked");
}
  int size;

  bool operator==(const Unstacked& other) const { return size == other.size; }
  bool operator!=(const Unstacked& other) const { return size != other.size; }
};

template <typename H>
H AbslHashValue(H h, const Unstacked& key) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_4(mht_4_v, 282, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  h = H::combine(std::move(h), key.size);
  return h;
}

using AvalDimSharding = absl::variant<NoSharding, Chunked, Unstacked>;

// Assigns sharded axes to mesh dimensions.
//
// The devices will be for each dimension which has a sharded `AvalDimSharding`
// When no axis is assigned, the data is replicated.
// As indices are 0-indexed, `ShardedAxis(1)` refers to the second actually
// sharded axis (i.e. counting as if the None dimensions of sharding were
// filtered out).
// For example, given the sharding `[Unstacked(n), None, Chunked(m)]`, an entry
// of `ShardedAxis(1)` refers to the `Chunked(m)` axis, not the `None`.

struct ShardedAxis {
  int axis;
  bool operator==(const ShardedAxis& other) const { return axis == other.axis; }
  bool operator!=(const ShardedAxis& other) const { return axis != other.axis; }
};

template <typename H>
H AbslHashValue(H h, const ShardedAxis& key) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_5(mht_5_v, 309, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  h = H::combine(std::move(h), key.axis);
  return h;
}

struct Replicated {
  int replicas;
  bool operator==(const Replicated& other) const {
    return replicas == other.replicas;
  }
  bool operator!=(const Replicated& other) const {
    return replicas != other.replicas;
  }
};

template <typename H>
H AbslHashValue(H h, const Replicated& key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_6(mht_6_v, 328, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  h = H::combine(std::move(h), key.replicas);
  return h;
}

using MeshDimAssignment = absl::variant<ShardedAxis, Replicated>;

// Describes how each axis is sharded (if it is), and how it's mapped to the
// devices mesh. See Jax pxla.py for the documentation.
//
// ShardingSpec is shared across pmap, pjit and xpmap. For pmap, an input
// `sharding`  is composed of `NoSharding` and at most one `Unstacked`.
// If `axis_size=None`, at least one the inputs has a dimension associated to
// `Unstacked`.
//
// Examples:
//
// 1. For pmap, with a tensor of shape [8, 2, 2], to unstack along the first
//    dimension into [8] devices:
//
//    sharding = [Unstacked(8), NoSharding, NoSharding]
//    mesh_mapping = [ShardedAxis(0)]
//
// 2. With an input array of shape [6], that we want to chunk into [2, 3]
//    Assuming an device mesh [3, 4, 2] of devices, we will have:
//
//    sharding = [Chunked([2, 3])]
//    mesh_mapping = [ShardedAxis(1), Replicated, ShardedAxis(0)]
//
//    In particular, in the above example, the ShardedAxis refers to indices
//    of the sharded shape [2, 3]. (only the `Chunked` sharding can produce more
//    than one dimension).
class ShardingSpec {
 public:
  ShardingSpec(std::vector<AvalDimSharding> sharding,
               std::vector<MeshDimAssignment> mesh_mapping)
      : sharding_(std::move(sharding)),
        mesh_mapping_(std::move(mesh_mapping)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_7(mht_7_v, 368, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "ShardingSpec");
}
  ShardingSpec(pybind11::iterable py_sharding,
               pybind11::iterable py_mesh_mapping)
      : sharding_(xla::IterableToVector<AvalDimSharding>(py_sharding)),
        mesh_mapping_(
            xla::IterableToVector<MeshDimAssignment>(py_mesh_mapping)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_8(mht_8_v, 376, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "ShardingSpec");
}

  const std::vector<AvalDimSharding>& GetSharding() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_9(mht_9_v, 381, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "GetSharding");
 return sharding_; }
  const std::vector<MeshDimAssignment>& GetMeshMapping() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_10(mht_10_v, 385, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "GetMeshMapping");

    return mesh_mapping_;
  }

  bool operator==(const ShardingSpec& other) const {
    return sharding_ == other.sharding_ && mesh_mapping_ == other.mesh_mapping_;
  }

  bool operator!=(const ShardingSpec& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const ShardingSpec& key);

 private:
  //  `sharding` specifies how the array is supposed to get partitioned into
  //  chunks. Its length matchs the rank of the array. See the docstring
  //  of `AvalDimSharding` for the supported partitioning schemes.
  std::vector<AvalDimSharding> sharding_;
  //  `mesh_mapping` describes an assignments of the array chunks created by
  //  `sharding` to a logical device mesh. The length of the tuple is equal to
  //  the rank of the mesh. Each mesh dimension can either get partitions of
  //  data varying along one of the sharded dimensions, or the data can be
  //  replicated.
  std::vector<MeshDimAssignment> mesh_mapping_;
};

template <typename H>
H AbslHashValue(H h, const ShardingSpec& key) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_11(mht_11_v, 415, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "AbslHashValue");

  h = H::combine(std::move(h), key.sharding_);
  h = H::combine(std::move(h), key.mesh_mapping_);
  return h;
}

// A ShardedDeviceArray is an ndarray sharded across devices.
//
// The purpose of a ShardedDeviceArray is to reduce the number of transfers when
// executing replicated computations, by allowing results to persist on the
// devices that produced them. That way dispatching a similarly replicated
// computation that consumes the same sharded memory layout does not incur any
// transfers.

// A ShardedDeviceArray represents one logical ndarray value, and simulates the
// behavior of an ndarray so that it can be treated by user code as an ndarray;
// that is, it is only an optimization to reduce transfers.

// Design note: We move to C++, only what will need to be accessed by C++ to
// execute a pmap computation. A large part of the logic is still in Python.
class ShardedDeviceArray {
 public:
  ShardedDeviceArray(const ShardedDeviceArray&) = delete;
  ShardedDeviceArray& operator=(const ShardedDeviceArray&) = delete;
  ShardedDeviceArray(ShardedDeviceArray&&) = default;
  ShardedDeviceArray& operator=(ShardedDeviceArray&&) = default;

  // Delete all the underlying buffers (freeing memory on device).
  // The Numpy value on the host, if it exists, will also be deleted.
  void Delete();
  const ShardingSpec& GetShardingSpec() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_12(mht_12_v, 448, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "GetShardingSpec");
 return sharding_spec_; }
  // Returns an error status iff the object has been deleted.
  xla::StatusOr<absl::Span<xla::PjRtBuffer* const>> GetPjRtBuffers();

  bool is_deleted() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_13(mht_13_v, 455, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "is_deleted");
 return is_deleted_; }
  bool weak_type() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_14(mht_14_v, 459, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "weak_type");
 return weak_type_; }
  absl::optional<pybind11::list> device_buffers() const {
    return device_buffers_;
  }
  pybind11::object aval() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_15(mht_15_v, 466, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "aval");
 return aval_; }
  pybind11::object indices() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_16(mht_16_v, 470, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "indices");
 return indices_; }

  absl::optional<pybind11::object> npy_value() const { return npy_value_; }
  void set_npy_value(pybind11::object npy_value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_17(mht_17_v, 476, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "set_npy_value");
 npy_value_ = npy_value; }

  absl::optional<pybind11::object> one_replica_buffer_indices() const {
    return one_replica_buffer_indices_;
  }
  void set_one_replica_buffer_indices(pybind11::object obj) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_18(mht_18_v, 484, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "set_one_replica_buffer_indices");

    one_replica_buffer_indices_ = obj;
  }

  // Python-wrapper definitions.

  // pybind11::object typed subclass for PyBuffer objects.
  class pyobject : public pybind11::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    pybind11::object, ShardedDeviceArray::IsShardedDeviceArray);
    pyobject() = default;
    ShardedDeviceArray* sda() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_19(mht_19_v, 499, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "sda");

      return ShardedDeviceArray::AsShardedDeviceArrayUnchecked(*this);
    }
  };
  using object = pyobject;

  // Returns true if `handle` is a IsShardedDeviceArray.
  static bool IsShardedDeviceArray(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Does not do any checking.
  static ShardedDeviceArray* AsShardedDeviceArrayUnchecked(
      pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Returns an error status if
  // !IsPyBuffer(handle)
  static xla::StatusOr<ShardedDeviceArray*> AsShardedDeviceArray(
      pybind11::handle handle);

  // Gets a Python handle to an existing ShardedDeviceArray. Assumes the
  // PyObject was allocated on the Python heap, which is the case if Make() was
  // used.
  pybind11::handle AsHandle();

  static object Make(pybind11::object aval, ShardingSpec sharding_spec,
                     pybind11::list device_buffers, pybind11::object indices,
                     bool weak_type);

  static xla::Status RegisterTypes(pybind11::module& m);
  static PyObject* base_type() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_20(mht_20_v, 528, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "base_type");
 return base_type_; }
  static PyObject* type() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_21(mht_21_v, 532, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "type");
 return type_; }

 private:
  // Buffers are expected to be xla::PyBuffer objects, but as there are
  // alternative backend implementations, this may not be guaranteed.
  // TODO(jblespiau): As soon as PjRtBuffer is supported by all
  // implementations, we should be able to store this with the C++ objects.
  ShardedDeviceArray(pybind11::object aval, ShardingSpec sharding_spec,
                     pybind11::list device_buffers, pybind11::object indices,
                     bool weak_type)
      : aval_(std::move(aval)),
        sharding_spec_(std::move(sharding_spec)),
        indices_(std::move(indices)),
        device_buffers_(std::move(device_buffers)),
        weak_type_(weak_type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTh mht_22(mht_22_v, 549, "", "./tensorflow/compiler/xla/python/sharded_device_array.h", "ShardedDeviceArray");
}
  static PyObject* base_type_;
  static PyObject* type_;

  // A ShapedArray indicating the shape and dtype of this array.
  pybind11::object aval_;
  // Describes how this array is sharded across `device_buffers`.
  ShardingSpec sharding_spec_;
  // The `indices` used to slice numpy array into the underlying list of
  // buffers. See the Python pxla.py:spec_to_indices function.
  pybind11::object indices_;
  // The buffers containing the data for this array. Each buffer is the same
  // shape and on a different device. Buffers are in row-major order, with
  // replication treated as an extra innermost dimension.
  absl::optional<pybind11::list> device_buffers_;

  absl::optional<pybind11::object> npy_value_ = absl::nullopt;
  absl::optional<pybind11::object> one_replica_buffer_indices_ = absl::nullopt;

  // The device_buffers as a C++ object. As this is what we consume from C++
  // and this is also what we generate from C++, cache the result so that
  // we don't have to perform casts.
  // TODO(jblespiau): Make this the default, and have `device_buffers_` the
  // the optional Python value if it's accessed from Python.
  absl::optional<std::vector<xla::PjRtBuffer*>> cpp_device_buffers_ =
      absl::nullopt;

  // The weak_type to prevent accessing the "aval_.weak_type" attribute which
  // is significantly slower.
  bool weak_type_;
  bool is_deleted_ = false;
};

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
