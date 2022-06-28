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
class MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc() {
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

#include "tensorflow/compiler/jit/device_util.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {
namespace jit {

void DeviceSet::Insert(DeviceId device_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceSet::Insert");

  int word_index = device_id.id() / kWordSize;
  int bit_index = device_id.id() % kWordSize;
  const int storage_size = storage_.size();
  if (word_index >= storage_size) {
    storage_.resize(word_index + 1, 0);
  }

  storage_[word_index] |= (1ull << bit_index);
}

void DeviceSet::UnionWith(const DeviceSet& other) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceSet::UnionWith");

  if (other.storage_.size() > storage_.size()) {
    storage_.resize(other.storage_.size(), 0);
  }

  for (int i = 0, end = other.storage_.size(); i < end; i++) {
    storage_[i] |= other.storage_[i];
  }
}

bool DeviceSet::IsEmpty() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceSet::IsEmpty");

  return absl::c_all_of(storage_, [&](uint64 val) { return val == 0; });
}

StatusOr<DeviceId> DeviceInfoCache::GetIdFor(absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_3(mht_3_v, 229, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceInfoCache::GetIdFor");

  TF_RET_CHECK(!name.empty());

  auto it = name_to_id_.find(name);
  if (it != name_to_id_.end()) {
    return it->second;
  }

  int new_id = names_.size();
  names_.push_back(string(name));
  id_to_device_type_.push_back(absl::make_unique<DeviceType>(""));
  DeviceType* device_type = id_to_device_type_.back().get();
  TF_RETURN_IF_ERROR(DeviceNameToDeviceType(names_.back(), device_type));

  is_cpu_.push_back(device_type->type_string() == DEVICE_CPU);
  is_gpu_.push_back(device_type->type_string() == DEVICE_GPU);

  name_to_id_.emplace(string(name), DeviceId(new_id));

  const XlaOpRegistry::DeviceRegistration* compilation_device;
  if (!XlaOpRegistry::GetCompilationDevice(device_type->type(),
                                           &compilation_device)) {
    compilation_device = nullptr;
  }
  id_to_compilation_device_.push_back(compilation_device);

  return DeviceId(new_id);
}

string DeviceInfoCache::DebugString(const DeviceSet& device_set) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceInfoCache::DebugString");

  std::vector<string> names;
  device_set.ForEach([&](DeviceId device_id) {
    names.push_back(string(GetNameFor(device_id)));
    return true;
  });

  return absl::StrCat("[", absl::StrJoin(names, ","), "]");
}
}  // namespace jit

Status DeviceNameToDeviceType(const string& device, DeviceType* device_type) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_utilDTcc mht_5(mht_5_v, 276, "", "./tensorflow/compiler/jit/device_util.cc", "DeviceNameToDeviceType");

  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device, &parsed)) {
    return errors::Internal("Malformed assigned device '", device, "'");
  }
  *device_type = DeviceType(parsed.type);
  return Status::OK();
}

StatusOr<absl::optional<jit::DeviceId>> PickDeviceForXlaImpl(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu,
    bool failure_to_pick_is_error) {
#define FAILED_TO_PICK_DEVICE(failing_status) \
  do {                                        \
    if (failure_to_pick_is_error) {           \
      return failing_status;                  \
    } else {                                  \
      return {absl::nullopt};                 \
    }                                         \
  } while (false)

  absl::optional<jit::DeviceId> maybe_gpu_device;
  absl::optional<jit::DeviceId> maybe_cpu_device;
  absl::optional<jit::DeviceId> maybe_unknown_device;

  bool multiple_cpu_devices = false;
  bool multiple_gpu_devices = false;
  bool multiple_unknown_devices = false;

  // Returns 'true' if d0 and d1 are conflicting devices. If they are
  // compatible, update d1 with a more specific one.
  // TODO(sanjoy): Cache DeviceNameUtils::ParsedName inside device_info_cache.
  const auto is_multiple_devices =
      [&](const jit::DeviceId& d0, absl::optional<jit::DeviceId>* d1) -> bool {
    const absl::string_view name0 = device_info_cache.GetNameFor(d0);
    const absl::string_view name1 = device_info_cache.GetNameFor(d1->value());

    DeviceNameUtils::ParsedName parsed0, parsed1;
    if (!DeviceNameUtils::ParseFullName(name0, &parsed0) ||
        !DeviceNameUtils::ParseFullName(name1, &parsed1) ||
        !DeviceNameUtils::AreCompatibleDevNames(parsed0, parsed1)) {
      return true;
    }

    if (DeviceNameUtils::IsSpecification(parsed0, parsed1)) {
      return false;
    }

    if (DeviceNameUtils::IsSpecification(parsed1, parsed0)) {
      *d1 = d0;
      return false;
    }

    return true;
  };

  devices.ForEach([&](jit::DeviceId device) {
    if (device_info_cache.IsGpu(device)) {
      if (maybe_gpu_device) {
        multiple_gpu_devices = is_multiple_devices(device, &maybe_gpu_device);
        if (multiple_gpu_devices) return false;
      } else {
        maybe_gpu_device = device;
      }
    } else if (device_info_cache.IsCpu(device)) {
      if (maybe_cpu_device) {
        multiple_cpu_devices = is_multiple_devices(device, &maybe_cpu_device);
        if (multiple_cpu_devices) return false;
      } else {
        maybe_cpu_device = device;
      }
    } else {
      if (maybe_unknown_device) {
        multiple_unknown_devices = true;
        return false;
      }
      maybe_unknown_device = device;
    }

    return true;
  });

  if (multiple_cpu_devices) {
    FAILED_TO_PICK_DEVICE(errors::Internal(
        "Multiple CPU devices ", device_info_cache.DebugString(devices)));
  }

  if (multiple_gpu_devices) {
    FAILED_TO_PICK_DEVICE(errors::Internal(
        "Multiple GPU devices ", device_info_cache.DebugString(devices)));
  }

  if (multiple_unknown_devices) {
    FAILED_TO_PICK_DEVICE(errors::Internal(
        "Multiple unknown devices ", device_info_cache.DebugString(devices)));
  }

  if (maybe_unknown_device && maybe_gpu_device) {
    FAILED_TO_PICK_DEVICE(errors::Internal(
        "Found both unknown and GPU devices: ",
        device_info_cache.GetNameFor(*maybe_unknown_device), ", ",
        device_info_cache.GetNameFor(*maybe_gpu_device)));
  }

  if (!allow_mixing_unknown_and_cpu) {
    if (maybe_unknown_device && maybe_cpu_device) {
      FAILED_TO_PICK_DEVICE(errors::Internal(
          "Found both unknown and CPU devices: ",
          device_info_cache.GetNameFor(*maybe_unknown_device), ", ",
          device_info_cache.GetNameFor(*maybe_cpu_device)));
    }
  }

  if (maybe_gpu_device) {
    return {*maybe_gpu_device};
  } else if (maybe_unknown_device) {
    return {*maybe_unknown_device};
  } else if (maybe_cpu_device) {
    return {*maybe_cpu_device};
  }

  FAILED_TO_PICK_DEVICE(errors::Internal("Empty device set!"));

#undef FAILED_TO_PICK_DEVICE
}

StatusOr<jit::DeviceId> PickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu) {
  TF_ASSIGN_OR_RETURN(absl::optional<jit::DeviceId> device_id,
                      PickDeviceForXlaImpl(device_info_cache, devices,
                                           allow_mixing_unknown_and_cpu,
                                           /*failure_to_pick_is_error=*/true));
  return *device_id;
}

StatusOr<absl::optional<jit::DeviceId>> MaybePickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu) {
  return PickDeviceForXlaImpl(device_info_cache, devices,
                              allow_mixing_unknown_and_cpu,
                              /*failure_to_pick_is_error=*/false);
}
}  // namespace tensorflow
