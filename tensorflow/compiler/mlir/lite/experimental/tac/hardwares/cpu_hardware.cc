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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc() {
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

#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
// CPU
constexpr float kCPUArithmeticUnitCost = 1.0;

// This basically assumes pure load/store. This is just fake data.
constexpr float kCPUCopyUnitCost = 0.5;

// Default values.
constexpr float kCPUDefaultFixedValuedCost = 10000.0;

// Quantized inference cost efficiency.
// For CPU, quantized inference is ~3x faster than the float alternative, this
// is just an estimation.
constexpr float kQuantizedInferenceEfficiency = 0.3;

inline float InferenceTypeEfficiency(InferenceType inference_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "InferenceTypeEfficiency");

  if (inference_type == QUANTIZED_INT8 || inference_type == QUANTIZED_UINT8) {
    return kQuantizedInferenceEfficiency;
  }
  return 1.0;
}

// CPU hardware class which handles CPU capabilities in TFLite.
// This is used by TAC to get op supported/ op cost estimates on CPU.
class CpuHardware : public TargetHardware {
 public:
  // String Identifier for CPU hardware.
  static constexpr char kId[] = "CPU";

  double GetHardwareSwitchingCost(const TargetHardware* from,
                                  size_t buffer_size) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetHardwareSwitchingCost");

    auto from_type = from->GetTypeId();
    auto to_type = GetTypeId();
    if (from_type == to_type) return 0.0f;

    // TODO(renjieliu): Implement a better version for different hardware cases.
    return buffer_size * kCrossHardwareTransferPerByteCost / 8.0 +
           kCrossHardwareTransferFixedCost;
  }

  mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetTransformations");

    return {context};
  }

  mlir::TypeID GetTypeId() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_3(mht_3_v, 248, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetTypeId");

    return mlir::TypeID::get<CpuHardware>();
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_4(mht_4_v, 255, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "IsOpSupported");

    // All ops in TFL dialect are supported on CPU.
    if (op->getDialect() == nullptr) return false;
    if (op->getDialect()->getNamespace() != "tfl") return false;
    return true;
  }
};

constexpr char CpuHardware::kId[];  // Define kId.

std::unique_ptr<TargetHardware> CreateCpuHardware() {
  return std::make_unique<CpuHardware>();
}

TargetHardwareRegistration<CpuHardware> cpu_hardware("Target device for CPU",
                                                     CreateCpuHardware);

#define TAC_REGISTER_CPU_OP(Op, Create)                                    \
  TargetHardwareOpRegistration<CpuHardware, Op> Op##_CpuHardware_hardware( \
      Create);

// Operation costs on CPU

// Currently used for these ops:
// tfl.conv_2d / tfl.depthwise_conv_2d / tfl.fully_connected
class CpuConvOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_5(mht_5_v, 284, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetOpCost");

    float cost = 0.0;
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      cost = arithmetic_count * kCPUArithmeticUnitCost;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "IsOpSupported");
 return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConvOp() {
  return std::make_unique<CpuConvOp>();
}

// Currently used for these ops:
// tfl.Add / tfl.mul
class CpuArithmeticOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_7(mht_7_v, 311, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetOpCost");

    float cost = 0.0;
    int64_t count;
    if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
      cost = kCPUArithmeticUnitCost * count;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_8(mht_8_v, 325, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "IsOpSupported");
 return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateArithmeticOp() {
  return std::make_unique<CpuArithmeticOp>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape / tfl.pack
class CpuConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_9(mht_9_v, 337, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "GetOpCost");

    float cost = 0.0;
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count)) {
      cost = kCPUCopyUnitCost * count;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPScpu_hardwareDTcc mht_10(mht_10_v, 351, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/cpu_hardware.cc", "IsOpSupported");
 return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<CpuConcatOp>();
}

TAC_REGISTER_CPU_OP(Conv2DOp, CreateConvOp);
TAC_REGISTER_CPU_OP(DepthwiseConv2DOp, CreateConvOp);
TAC_REGISTER_CPU_OP(FullyConnectedOp, CreateConvOp);
TAC_REGISTER_CPU_OP(AddOp, CreateArithmeticOp);
TAC_REGISTER_CPU_OP(MulOp, CreateArithmeticOp);
TAC_REGISTER_CPU_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_CPU_OP(ReshapeOp, CreateConcatOp);
TAC_REGISTER_CPU_OP(PackOp, CreateConcatOp);

#undef TAC_REGISTER_CPU_OP
}  // namespace
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
