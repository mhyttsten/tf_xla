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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc() {
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

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.h"

#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/generated_transform_patterns.inc"
}  // namespace

constexpr char GpuHardware::kId[];  // Define kId.

mlir::RewritePatternSet GpuHardware::GetTransformations(
    MLIRContext* context) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GpuHardware::GetTransformations");

  mlir::RewritePatternSet patterns(context);

  patterns.add<LowerPackIntoConcatReshape, UnrollSplit, UnrollSplitV, SubToAdd,
               EnsureBiasForConv2d, PadSlice, FullyConnectedToConv, PadConcat,
               SquaredDifference>(context);
  return patterns;
}

double GpuHardware::GetHardwareSwitchingCost(const TargetHardware* from,
                                             size_t buffer_size) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GpuHardware::GetHardwareSwitchingCost");

  auto from_type = from->GetTypeId();
  auto to_type = GetTypeId();
  if (from_type == to_type) return 0.0f;

  // TODO(renjieliu): Implement a better version for different hardware cases.
  return buffer_size * kCrossHardwareTransferPerByteCost / 8.0 +
         kCrossHardwareTransferFixedCost;
}

namespace {
// GPU
constexpr float kGPUArithmeticUnitCost = 0.2;

// The copy can be non-consectutive copy. This is just fake data.
constexpr float kGPUCopyUnitCost = 0.2;

// Default values.
constexpr float kGPUDefaultFixedValuedCost = 10000.0;

std::unique_ptr<TargetHardware> CreateGpuHardware() {
  return std::make_unique<GpuHardware>();
}

TargetHardwareRegistration<GpuHardware> gpu_hardware("Target device for GPU",
                                                     CreateGpuHardware);

#define TAC_REGISTER_GPU_OP(Op, Create)                                    \
  TargetHardwareOpRegistration<GpuHardware, Op> Op##_GpuHardware_hardware( \
      Create);

// Currently used for these ops:
// tfl.Abs / tfl.Average_pool_2d / tfl.Cos / tfl.div / tfl.exp / tfl.hardswish /
// tfl.log / tfl.logistic / tfl.max_pool_2d / tfl.mirror_pad / tfl.maximum /
// tfl.custom / tfl.mean / tfl.minimum / tfl.pad / tfl.pow / tfl.prelu /
// tfl.relu / tfl.relu6 / tfl.rsqrt / tfl.sin / tfl.slice / tfl.softmax /
// tfl.space_to_depth / tfl.sqrt / tfl.square / tfl.squared_difference /
// tfl.strided_slice / tfl.tanh / tfl.transpose / tfl.transpose_conv
class GpuBasicSupportedOpNoCost : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GetOpCost");
 return 0; }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "IsOpSupported");

    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateBasicOpNoCost() {
  return std::make_unique<GpuBasicSupportedOpNoCost>();
}

// Currently used for these ops:
// tfl.Add / tfl.mul
class GpuArithmeticOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GetOpCost");

    int64_t count;
    if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
      return kGPUArithmeticUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_5(mht_5_v, 291, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "IsOpSupported");

    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateArithmeticOp() {
  return std::make_unique<GpuArithmeticOp>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape
class GpuConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_6(mht_6_v, 309, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GetOpCost");

    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kGPUCopyUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_7(mht_7_v, 319, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "IsOpSupported");

    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<GpuConcatOp>();
}

// Currently used for these ops:
// tfl.conv_2d / tfl.depthwise_conv_2d / tfl.fully_connected
class GpuConvOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_8(mht_8_v, 337, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "GetOpCost");

    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      return arithmetic_count * kGPUArithmeticUnitCost;
    }
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSgpu_hardwareDTcc mht_9(mht_9_v, 349, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.cc", "IsOpSupported");

    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateConvOp() {
  return std::make_unique<GpuConvOp>();
}

// Op registrations
TAC_REGISTER_GPU_OP(AbsOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(AveragePool2DOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(CosOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(DivOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(ExpOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(HardSwishOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(LogOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(LogisticOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MaxPool2DOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MirrorPadOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MaximumOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MinimumOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MeanOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(CustomOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PadOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PowOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PReluOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(ReluOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(Relu6Op, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(RsqrtOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SinOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SliceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SoftmaxOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SpaceToDepthOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SqrtOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SquareOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SquaredDifferenceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(StridedSliceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TanhOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TransposeOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TransposeConvOp, CreateBasicOpNoCost);

TAC_REGISTER_GPU_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_GPU_OP(ReshapeOp, CreateConcatOp);

TAC_REGISTER_GPU_OP(Conv2DOp, CreateConvOp);
TAC_REGISTER_GPU_OP(DepthwiseConv2DOp, CreateConvOp);
TAC_REGISTER_GPU_OP(FullyConnectedOp, CreateConvOp);

TAC_REGISTER_GPU_OP(AddOp, CreateArithmeticOp);
TAC_REGISTER_GPU_OP(MulOp, CreateArithmeticOp);

#undef TAC_REGISTER_GPU_OP
}  // namespace
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
