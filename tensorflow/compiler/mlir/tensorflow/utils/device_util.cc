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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

constexpr char kDevicesAttr[] = "tf.devices";

namespace {

// Parse GPU compute capability from physical device description. If compute
// capability is not found in device description, return an empty dictionary
// attribute.
mlir::DictionaryAttr ParseGpuDeviceMetadata(const Device& device,
                                            mlir::Builder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "ParseGpuDeviceMetadata");

  // Parse GPU device compute capability from physical device description.
  static auto* r = new llvm::Regex("compute capability: ([0-9]+)\\.([0-9]+)");

  llvm::SmallVector<llvm::StringRef, 3> cc;
  if (r->match(device.attributes().physical_device_desc(), &cc)) {
    return mlir::TF::GpuDeviceMetadata::get(
        builder->getI32IntegerAttr(std::stoi(cc[1].str())),
        builder->getI32IntegerAttr(std::stoi(cc[2].str())),
        builder->getContext());
  }

  return builder->getDictionaryAttr({});
}

// Get devices from an array of string attributes.
// TODO(ezhulenev): Update all tests to use dictionary attribute for
// `tf.devices` and remove this function.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::ArrayAttr array_attr,
                                     mlir::TF::RuntimeDevices* devices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "GetDevicesFromOp");

  DeviceNameUtils::ParsedName device;

  for (auto& kv : llvm::enumerate(array_attr)) {
    const int idx = kv.index();

    auto string_attr = kv.value().dyn_cast<mlir::StringAttr>();
    if (!string_attr)
      return op->emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, not a string", kDevicesAttr, idx));

    if (DeviceNameUtils::ParseFullName(string_attr.getValue().str(), &device)) {
      devices->AddDevice(device);
    } else {
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, string_attr.getValue()));
    }
  }

  return mlir::success();
}

// Get devices from a dictionary attribute.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::DictionaryAttr dict_attr,
                                     mlir::TF::RuntimeDevices* devices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "GetDevicesFromOp");

  DeviceNameUtils::ParsedName device;

  // Parse device names and metadata from dictionary attribute.
  for (auto& kv : dict_attr) {
    const mlir::StringAttr name = kv.getName();
    const mlir::Attribute attr = kv.getValue();

    if (!DeviceNameUtils::ParseFullName(name.str(), &device))
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, name.strref()));

    if (auto gpu_metadata = attr.dyn_cast<mlir::TF::GpuDeviceMetadata>()) {
      devices->AddGpuDevice(device, gpu_metadata);
    } else {
      devices->AddDevice(device);
    }
  }

  return mlir::success();
}

}  // namespace

void AddDevicesToOp(mlir::Operation* op, const DeviceSet* device_set) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_3(mht_3_v, 296, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "AddDevicesToOp");

  if (!device_set) return;

  mlir::MLIRContext* ctx = op->getContext();
  mlir::Builder builder(ctx);

  // Collect devices with attached metadata.
  llvm::SmallVector<mlir::NamedAttribute, 8> devices;
  devices.reserve(device_set->devices().size());

  // For device that do not have any metadata, or if we failed to parse metadata
  // from the DeviceSet, we add empty dictionary to the `tf.devices` attribute.
  for (Device* device : device_set->devices()) {
    string name = DeviceNameUtils::ParsedNameToString(device->parsed_name());

    if (device->device_type() == DEVICE_GPU) {
      auto metadata = ParseGpuDeviceMetadata(*device, &builder);
      devices.push_back(builder.getNamedAttr(name, metadata));
    } else {
      auto metadata = builder.getDictionaryAttr({});
      devices.push_back(builder.getNamedAttr(name, metadata));
    }
  }

  op->setAttr(kDevicesAttr, builder.getDictionaryAttr(devices));
}

mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::TF::RuntimeDevices* devices) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_4(mht_4_v, 327, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "GetDevicesFromOp");

  auto devices_attr = op->getAttr(kDevicesAttr);
  if (!devices_attr) return mlir::success();

  if (auto array_attr = devices_attr.dyn_cast<mlir::ArrayAttr>()) {
    return GetDevicesFromOp(op, array_attr, devices);

  } else if (auto dict_attr = devices_attr.dyn_cast<mlir::DictionaryAttr>()) {
    return GetDevicesFromOp(op, dict_attr, devices);
  }

  return op->emitOpError(
      llvm::formatv("unsupported '{0}' attribute", kDevicesAttr));
}

mlir::LogicalResult GetDeviceOrdinalFromDeviceString(mlir::Location loc,
                                                     llvm::StringRef device,
                                                     int64_t* device_ordinal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdevice_utilDTcc mht_5(mht_5_v, 347, "", "./tensorflow/compiler/mlir/tensorflow/utils/device_util.cc", "GetDeviceOrdinalFromDeviceString");

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(
          absl::string_view(device.data(), device.size()), &parsed_name))
    return mlir::emitError(loc) << "invalid device '" << device << "'";

  if (!parsed_name.has_id)
    return mlir::emitError(loc) << "device '" << device << "' has no id";

  *device_ordinal = parsed_name.id;
  return mlir::success();
}

}  // namespace tensorflow
