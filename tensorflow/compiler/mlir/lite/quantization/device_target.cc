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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"

#include <algorithm>

#include "absl/types/optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

namespace mlir {
namespace quant {

constexpr int k8Bits = 8;
constexpr int k32Bits = 32;
constexpr unsigned kSigned = quant::QuantizationFlags::Signed;

DeviceTarget::DeviceTarget(MLIRContext* ctx) : ctx_(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::DeviceTarget");

  f32_ = FloatType::getF32(ctx_);
  i8_ = IntegerType::get(ctx_, k8Bits);
  i8_min_ = QuantizedType::getDefaultMinimumForInteger(kSigned, k8Bits);
  i8_max_ = QuantizedType::getDefaultMaximumForInteger(kSigned, k8Bits);
  i32_ = IntegerType::get(ctx_, k32Bits);
  i32_min_ = QuantizedType::getDefaultMinimumForInteger(kSigned, k32Bits);
  i32_max_ = QuantizedType::getDefaultMaximumForInteger(kSigned, k32Bits);
  any_ = AnyQuantizedType();
  qi8_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_, i8_max_);
  qi8n_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_ + 1, i8_max_);
  qi32_ = AnyQuantizedType::get(kSigned, i32_, f32_, i32_min_, i32_max_);
  assert(qi8n_ == qi8n_);
}

Optional<KernelSpec> DeviceTarget::GetKernelSpec(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::GetKernelSpec");

  auto kernel_specs_it = specs_.find(kernel);
  if (kernel_specs_it == specs_.end()) return llvm::None;
  return kernel_specs_it->getValue().Find(signature);
}

ScaleDecomposeFn DeviceTarget::GetDecomposeFn(QuantizeRegionOp op) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::GetDecomposeFn");

  auto kernel_specs_it = specs_.find(op.logical_kernel());
  if (kernel_specs_it == specs_.end()) return ScaleDecomposeFn(nullptr);
  return kernel_specs_it->second.GetDecomposeFn();
}

void DeviceTarget::AppendToSignature(Type spec,
                                     KernelSpecs::Signature* signature) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_3(mht_3_v, 245, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::AppendToSignature");

  if (auto quant = spec.dyn_cast_or_null<UniformQuantizedType>()) {
    signature->push_back(AnyQuantizedType::get(
        quant.getFlags(), quant.getStorageType(), quant.getExpressedType(),
        quant.getStorageTypeMin(), quant.getStorageTypeMax()));
  } else if (auto any = spec.dyn_cast_or_null<AnyQuantizedType>()) {
    signature->push_back(any);
  } else {  // float
    signature->push_back(AnyQuantizedType());
  }
}

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleFn& fn, const ScaleDecomposeFn& dfn) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_4(mht_4_v, 262, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::RegisterKernel");

  return specs_[kernel].Add(signature, {ScaleConstraintType::CustomScale, fn});
}

namespace ph = std::placeholders;

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleConstraintType constraint) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::RegisterKernel");

  if (failed(specs_[kernel].Add(signature, {constraint, {}}))) return failure();
  switch (constraint) {
    case ScaleConstraintType::OutputInputSameScale:
      specs_[kernel].WithImpl(std::bind(&DeviceTarget::DecomposeSameScale,
                                        ph::_1, ph::_2, ph::_3, ph::_4));
      return success();
    default:
      return failure();
  }
}

LogicalResult DeviceTarget::DecomposeMultiplyAccumulateScale(
    Operation* op, quant::QuantizedMultipliers* input_multipliers,
    quant::QuantizedMultipliers* output_multipliers,
    quant::QuantizedRanges* output_ranges) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_6(mht_6_v, 291, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::DecomposeMultiplyAccumulateScale");

  auto rop = llvm::dyn_cast<quant::QuantizeRegionOp>(op);
  if (!rop) return failure();

  llvm::SmallVector<Type, 4> input_specs, out_specs;
  for (auto spec : rop.input_specs()) {
    input_specs.push_back(spec.cast<TypeAttr>().getValue());
  }
  for (auto spec : rop.output_specs()) {
    out_specs.push_back(spec.cast<TypeAttr>().getValue());
  }

  auto in_spec = input_specs[0].dyn_cast<quant::UniformQuantizedType>();
  // TODO(fengliuai): handles the PerAxis QuantizedType.
  auto w_spec = input_specs[1].dyn_cast<quant::UniformQuantizedType>();
  auto b_spec = input_specs[2].dyn_cast<quant::UniformQuantizedType>();
  auto o_spec = out_specs[0].dyn_cast<quant::UniformQuantizedType>();
  if (!in_spec || !w_spec || !b_spec || !o_spec) return failure();

  double scale_product = in_spec.getScale() * w_spec.getScale();
  if (fabs(scale_product - b_spec.getScale()) >= 1e-6) return failure();

  // input multipliers
  input_multipliers->append(3, kUnitQuantizedMultiplier);

  // output multipliers
  double real_multiplier = scale_product / o_spec.getScale();
  output_multipliers->push_back(quant::QuantizeMultiplier(real_multiplier));

  // output ranges
  auto min = rop->getAttrOfType<FloatAttr>("min");
  auto max = rop->getAttrOfType<FloatAttr>("max");
  output_ranges->push_back(quant::CalculateQuantizedRange(
      o_spec.getScale(), o_spec.getZeroPoint(),
      (min ? absl::optional<double>(min.getValueAsDouble()) : absl::nullopt),
      (max ? absl::optional<double>(max.getValueAsDouble()) : absl::nullopt),
      o_spec.getStorageTypeMin(), o_spec.getStorageTypeMax()));

  return success();
}

LogicalResult DeviceTarget::DecomposeSameScale(
    Operation* op, quant::QuantizedMultipliers* input_multipliers,
    quant::QuantizedMultipliers* output_multipliers,
    quant::QuantizedRanges* output_ranges) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTcc mht_7(mht_7_v, 338, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.cc", "DeviceTarget::DecomposeSameScale");

  auto rop = llvm::dyn_cast<quant::QuantizeRegionOp>(op);
  if (!rop) return failure();

  // input multipliers
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_multipliers->push_back(kUnitQuantizedMultiplier);
  }

  // output multipliers
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_multipliers->push_back(kUnitQuantizedMultiplier);
  }

  auto o_spec = rop.output_specs()[0]
                    .cast<TypeAttr>()
                    .getValue()
                    .dyn_cast<quant::UniformQuantizedType>();
  if (!o_spec) return failure();

  // output ranges
  auto min = rop->getAttrOfType<FloatAttr>("min");
  auto max = rop->getAttrOfType<FloatAttr>("max");
  output_ranges->push_back(quant::CalculateQuantizedRange(
      o_spec.getScale(), o_spec.getZeroPoint(),
      (min ? absl::optional<double>(min.getValueAsDouble()) : absl::nullopt),
      (max ? absl::optional<double>(max.getValueAsDouble()) : absl::nullopt),
      o_spec.getStorageTypeMin(), o_spec.getStorageTypeMax()));

  return success();
}

}  // namespace quant
}  // namespace mlir
