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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh() {
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


#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace jit {
class DeviceInfoCache;
class DeviceSet;

// Instances of DeviceId represent TensorFlow devices as integers.
//
// This helps avoid having to manipulate device names as strings when
// auto-clustering.
class DeviceId {
 public:
  DeviceId(DeviceId&&) = default;
  DeviceId(const DeviceId&) = default;
  DeviceId& operator=(const DeviceId&) = default;

  bool operator==(const DeviceId& other) const { return id() == other.id(); }
  bool operator!=(const DeviceId& other) const { return !(*this == other); }

 private:
  int id_;

  explicit DeviceId(int id) : id_(id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_0(mht_0_v, 220, "", "./tensorflow/compiler/jit/device_util.h", "DeviceId");
}

  int id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_1(mht_1_v, 225, "", "./tensorflow/compiler/jit/device_util.h", "id");
 return id_; }

  friend class DeviceInfoCache;
  friend class DeviceSet;
};

// A set of DeviceIds, represented as a bitmap.
class DeviceSet {
 public:
  void Insert(DeviceId device_id);
  void UnionWith(const DeviceSet& other);
  bool IsEmpty() const;

  // Calls `func` on each DeviceId in the set.  Stops iterating early if `func`
  // return false.
  //
  // TODO(sanjoy): Change this to take a typed std::function if that's
  // performance neutral.
  template <typename FnTy>
  void ForEach(FnTy func) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_2(mht_2_v, 247, "", "./tensorflow/compiler/jit/device_util.h", "ForEach");

    // This is really a poor man's iterator, we should consider writing a proper
    // iterator if this ends up being used widely.
    for (int word_index = 0, end = storage_.size(); word_index < end;
         word_index++) {
      uint64 word = storage_[word_index];
      while (word != 0) {
        uint64 only_lowest_bit_set = word & -word;
        // The number of trailing zeros in a non-zero word is the index of the
        // least significant 1.
        int bit_index = ctz_uint64(word);
        if (!func(DeviceId(word_index * kWordSize + bit_index))) {
          return;
        }
        word ^= only_lowest_bit_set;
      }
    }
  }

 private:
  static int ctz_uint64(uint64 x) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_3(mht_3_v, 270, "", "./tensorflow/compiler/jit/device_util.h", "ctz_uint64");

    DCHECK_NE(x, 0);
#ifdef __GNUC__
    return __builtin_ctzl(x);
#else
    int result = 0u;
    while ((x & 1u) == 0u) {
      x >>= 1;
      ++result;
    }
    return result;
#endif
  }

  absl::InlinedVector<uint64, 1> storage_;

  const int kWordSize = 64;
};

// Caches some miscellaneous information about TF devices.  Thread compatible.
class DeviceInfoCache {
 public:
  bool IsGpu(DeviceId device) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_4(mht_4_v, 295, "", "./tensorflow/compiler/jit/device_util.h", "IsGpu");
 return is_gpu_[device.id()]; }
  bool IsCpu(DeviceId device) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_5(mht_5_v, 299, "", "./tensorflow/compiler/jit/device_util.h", "IsCpu");
 return is_cpu_[device.id()]; }

  absl::string_view GetNameFor(DeviceId device) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_6(mht_6_v, 304, "", "./tensorflow/compiler/jit/device_util.h", "GetNameFor");

    return names_[device.id()];
  }

  StatusOr<DeviceId> GetIdFor(absl::string_view name);

  using DeviceRegistration = const XlaOpRegistry::DeviceRegistration;

  DeviceRegistration* GetCompilationDevice(DeviceId device) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_7(mht_7_v, 315, "", "./tensorflow/compiler/jit/device_util.h", "GetCompilationDevice");

    return id_to_compilation_device_[device.id()];
  }

  StatusOr<DeviceRegistration*> GetCompilationDevice(absl::string_view name) {
    TF_ASSIGN_OR_RETURN(DeviceId device_id, GetIdFor(name));
    return GetCompilationDevice(device_id);
  }

  const DeviceType& GetDeviceTypeFor(DeviceId device) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTh mht_8(mht_8_v, 327, "", "./tensorflow/compiler/jit/device_util.h", "GetDeviceTypeFor");

    return *id_to_device_type_[device.id()];
  }

  using DeviceTypeConstRef = std::reference_wrapper<const DeviceType>;

  StatusOr<DeviceTypeConstRef> GetDeviceTypeFor(absl::string_view device_name) {
    TF_ASSIGN_OR_RETURN(DeviceId device_id, GetIdFor(device_name));
    return std::cref(*id_to_device_type_[device_id.id()]);
  }

  string DebugString(const DeviceSet& device_set) const;

 private:
  absl::flat_hash_map<string, DeviceId> name_to_id_;

  // These fields are populated for a device in GetIdFor, *before* we give out a
  // DeviceId.
  std::vector<const XlaOpRegistry::DeviceRegistration*>
      id_to_compilation_device_;
  std::vector<std::unique_ptr<DeviceType>> id_to_device_type_;
  std::vector<string> names_;
  std::vector<bool> is_cpu_;
  std::vector<bool> is_gpu_;
};

}  // namespace jit

// Returns the DeviceType corresponding to 'device'.
Status DeviceNameToDeviceType(const string& device, DeviceType* device_type);

// Picks the device for which XLA should compile a cluster that contains
// operations placed in devices in `devices`.  For instance a cluster that
// contains operations solely placed on the CPU will be compiled into a CPU
// executable by XLA, whereas a cluster that contains operations placed on the
// CPU and also operations placed on the GPU will be compiled into a GPU
// executable.
//
// Returns a non-OK Status if no unambiguous choice of device exists.
//
// We choose the device using the following rules:
//
//  - It is an error for `device_names` to contain more than one device of the
//    same type.
//  - GPU is preferred over CPU.
//  - If `allow_mixing_unknown_and_cpu` is true then unknown devices are
//    preferred over CPU.
//  - XLA devices count as "unrecognized devices".
//
// This set of rules above implicitly assume that XLA:GPU can compile all
// operations in the cluster that XLA:CPU can compile, and if
// `allow_mixing_unknown_and_cpu` then the unrecognized device can also compile
// all operations in the cluster that XLA:CPU can compile.
//
// We provide the `allow_mixing_unknown_and_cpu` knob so that we can do both of
// the following things:
//
// - Let MarkForCompilationPass not inject CPU-placed operations into clusters
//   that will run on unknown devices (because the unknown XLA backend may not
//   support every operation supported by CPU).
// - Let BuildXlaOpsPass successfully infer a compilation device for a cluster
//   that contains nodes placed on both the CPU and on unknown devices.  In this
//   case it is the responsibility of the optimization pass that injected the
//   CPU nodes into the cluster to ensure that these nodes can be compiled by
//   the unknown XLA backend.
StatusOr<jit::DeviceId> PickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu);

// This is like `PickDeviceForXla` except that it returns nullopt (instead of a
// non-OK Status) if no unambiguous choice of device exists.
//
// We return a failing Status for errors unrelated to the device choice
// algorithm itself.
StatusOr<absl::optional<jit::DeviceId>> MaybePickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
