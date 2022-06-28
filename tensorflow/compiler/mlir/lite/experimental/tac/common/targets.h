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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh() {
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
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

// Device attribute string on the TFL dialect.
constexpr char kDevice[] = "tac.device";

// Inference type.
constexpr char kInferenceType[] = "tac.inference_type";

// TODO(renjieliu): Add more inference types.
enum InferenceType {
  UNKNOWN = 0,
  FLOAT = 1,
  QUANTIZED_INT8 = 2,
  QUANTIZED_UINT8 = 3,
  HYBRID = 4
};

inline InferenceType GetInferenceTypeEnum(llvm::StringRef inference_type_str) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h", "GetInferenceTypeEnum");

  if (inference_type_str == "FLOAT") {
    return FLOAT;
  } else if (inference_type_str == "QUANTIZED_INT8") {
    return QUANTIZED_INT8;
  } else if (inference_type_str == "QUANTIZED_UINT8") {
    return QUANTIZED_UINT8;
  } else if (inference_type_str == "HYBRID") {
    return HYBRID;
  } else {
    return UNKNOWN;
  }
}

inline std::string GetInferenceString(InferenceType inference_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh mht_1(mht_1_v, 233, "", "./tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h", "GetInferenceString");

  if (inference_type == FLOAT) {
    return "FLOAT";
  } else if (inference_type == QUANTIZED_INT8) {
    return "QUANTIZED_INT8";
  } else if (inference_type == QUANTIZED_UINT8) {
    return "QUANTIZED_UINT8";
  } else if (inference_type == HYBRID) {
    return "HYBRID";
  } else {
    return "UNKNOWN";
  }
}

// Returns canonical representation for hardware name (All uppercase).
// TODO(b/177376459): Remove this in favor of the string defined by hardwares
// MyHardware::kId.
inline std::string GetCanonicalHardwareName(const std::string& hardware_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("hardware_name: \"" + hardware_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPScommonPStargetsDTh mht_2(mht_2_v, 254, "", "./tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h", "GetCanonicalHardwareName");

  std::string name = hardware_name;
  std::transform(
      name.begin(), name.end(), name.begin(),
      [](unsigned char c) -> unsigned char { return std::toupper(c); });
  return name;
}

// Get the target annotation form the op.
inline llvm::Optional<std::string> GetTargetAnnotation(Operation* op) {
  auto device = op->getAttrOfType<StringAttr>(kDevice);
  if (device == nullptr || device.getValue().empty()) return llvm::None;

  return GetCanonicalHardwareName(device.getValue().str());
}

// Get inference type attribute from the operation if available.
inline llvm::Optional<InferenceType> GetInferenceTypeAnnotation(Operation* op) {
  auto inference_type = op->getAttrOfType<StringAttr>(kInferenceType);
  if (inference_type == nullptr) return llvm::None;

  llvm::StringRef device_name_str = inference_type.getValue();
  return GetInferenceTypeEnum(device_name_str);
}

// InferenceDeviceType is a combination of the hardware with inference type.
struct InferenceDeviceType {
  std::string hardware;
  InferenceType inference_type;

  bool operator==(const InferenceDeviceType& other) const {
    return (hardware == other.hardware) &&
           (inference_type == other.inference_type);
  }

  bool operator!=(const InferenceDeviceType& other) const {
    return !(*this == other);
  }

  struct inference_device_type_hash {
    size_t operator()(const InferenceDeviceType& p) const {
      auto hash1 = std::hash<std::string>{}(p.hardware);
      auto hash2 = std::hash<InferenceType>{}(p.inference_type);
      return hash1 ^ hash2;
    }
  };
};

// Get InferenceDeviceType attribute from the operation if available.
inline llvm::Optional<InferenceDeviceType> GetInferenceDeviceTypeForOp(
    Operation* op) {
  auto hardware = GetTargetAnnotation(op);
  if (!hardware.hasValue()) return llvm::None;

  auto inference_type = GetInferenceTypeAnnotation(op);
  if (!inference_type.hasValue()) return llvm::None;

  InferenceDeviceType inference_device_type;
  inference_device_type.hardware = hardware.getValue();
  inference_device_type.inference_type = inference_type.getValue();
  return inference_device_type;
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_
