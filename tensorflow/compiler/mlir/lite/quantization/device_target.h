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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

namespace mlir {
namespace quant {

class QuantizeContext;

using AdjacentOperations = llvm::SmallVectorImpl<Operation*>;
using QuantizedMultipliers = llvm::SmallVector<QuantizedMultiplier, 4>;
using QuantizedRanges = llvm::SmallVector<QuantizedRange, 4>;
using ScaleFn = std::function<LogicalResult(QuantizeContext*, Operation*,
                                            AdjacentOperations*, bool*)>;

using ScaleDecomposeFn =
    std::function<LogicalResult(Operation*, QuantizedMultipliers*,
                                QuantizedMultipliers*, QuantizedRanges*)>;

static const QuantizedMultiplier kUnitQuantizedMultiplier{1, 0};

enum class ScaleConstraintType {
  OutputInputSameScale,
  OutputInputFreeScale,
  CustomScale,
};

// Each kernel signature has its own specification for scales.
struct KernelSpec {
  // Scale constraint
  ScaleConstraintType type;

  // Custom function to derive the scales. Only available when the scale
  // constraint is `CustomScale`.
  ScaleFn scale_fn;
};

class KernelSpecs {
 public:
  using Signature = llvm::SmallVector<quant::AnyQuantizedType, 4>;

  // Returns the kernel specification for the kernel signature.
  Optional<KernelSpec> Find(const Signature& signature) const {
    auto spec_it = all_signatures_.find(signature);
    if (spec_it != all_signatures_.end()) {
      return spec_it->second;
    } else {
      return llvm::None;
    }
  }

  ScaleDecomposeFn GetDecomposeFn() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_0(mht_0_v, 254, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "GetDecomposeFn");
 return decompose_fn_; }

  // Adds the kernel signature with the kernel specification.
  LogicalResult Add(const Signature& signature, const KernelSpec& spec) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_1(mht_1_v, 260, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "Add");

    if (all_signatures_.insert({signature, spec}).second) return success();
    return failure();
  }

  KernelSpecs& WithSignature(const KernelSpecs::Signature& signature,
                             const ScaleFn& fn) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_2(mht_2_v, 269, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "WithSignature");

    (void)Add(signature, {ScaleConstraintType::CustomScale, fn});
    return *this;
  }

  KernelSpecs& WithImpl(const ScaleDecomposeFn& dfn) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_3(mht_3_v, 277, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "WithImpl");

    decompose_fn_ = dfn;
    return *this;
  }

 private:
  // The signature is pattern match based.
  struct SignatureInfo : public llvm::DenseMapInfo<Signature> {
    static inline Signature getEmptyKey() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_4(mht_4_v, 288, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "getEmptyKey");
 return {}; }
    static inline Signature getTombstoneKey() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_5(mht_5_v, 292, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "getTombstoneKey");
 return {nullptr}; }
    static unsigned getHashValue(Signature val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_6(mht_6_v, 296, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "getHashValue");

      return llvm::hash_combine_range(val.begin(), val.end());
    }
    static bool isEqual(Signature LHS, Signature RHS) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_7(mht_7_v, 302, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "isEqual");

      if (RHS == getEmptyKey()) return LHS == getEmptyKey();
      if (RHS == getTombstoneKey()) return LHS == getTombstoneKey();
      if (LHS.size() != RHS.size()) return false;
      for (auto arg : llvm::zip(LHS, RHS)) {
        if (std::get<0>(arg) != std::get<1>(arg)) return false;
      }
      return true;
    }
  };

  // Maps the signature to the kernel spec. Note that the matching is
  // pattern match based.
  llvm::DenseMap<Signature, KernelSpec, SignatureInfo> all_signatures_;

  // A method to compute the effective multipliers. This is independent on the
  // bits of the ports, thus all the signature shares the same here.
  ScaleDecomposeFn decompose_fn_;
};

class DeviceTarget {
 public:
  explicit DeviceTarget(MLIRContext* ctx);

  // Retrieves the kernel spec for the quant region op.
  Optional<KernelSpec> GetKernelSpec(
      llvm::StringRef kernel, const KernelSpecs::Signature& signature) const;

  // Retrieves the scale decomposition function for the quant region op.
  ScaleDecomposeFn GetDecomposeFn(quant::QuantizeRegionOp op) const;

  // converts specification to signature:
  // - UniformedQuantizedType -> AnyQuantizedType
  // - AnyQuantizedType (int) -> AnyQuantizedType
  // - Float -> {}
  static void AppendToSignature(Type spec, KernelSpecs::Signature* signature);

 protected:
  // Adds the kernel spec with the custom scale function for the kernel.
  LogicalResult RegisterKernel(llvm::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               const ScaleFn& fn, const ScaleDecomposeFn& dfn);

  // Adds the kernel spec with the scale constraint type for the kernel.
  LogicalResult RegisterKernel(llvm::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               const ScaleConstraintType constraint);

  // Adds the kernel with the name. Retrun an existing one if it has been
  // added before.
  KernelSpecs& RegisterKernel(llvm::StringRef kernel) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSdevice_targetDTh mht_8(mht_8_v, 355, "", "./tensorflow/compiler/mlir/lite/quantization/device_target.h", "RegisterKernel");
 return specs_[kernel]; }

  // For "mulmat->add" type of kernels, convert the scales of all the ports to
  // multipliers.
  static LogicalResult DecomposeMultiplyAccumulateScale(
      Operation* op, quant::QuantizedMultipliers* input_multipliers,
      quant::QuantizedMultipliers* output_multipliers,
      quant::QuantizedRanges* output_ranges);

  // For "reshape" type of kernels.
  static LogicalResult DecomposeSameScale(
      Operation* op, quant::QuantizedMultipliers* input_multipliers,
      quant::QuantizedMultipliers* output_multipliers,
      quant::QuantizedRanges* output_ranges);

  // A set of parameters are required to build the signatures.
  FloatType f32_;
  IntegerType i8_, i32_;
  int64_t i8_min_, i8_max_, i32_min_, i32_max_;
  AnyQuantizedType any_, qi8_, qi8n_, qi32_;

 private:
  // Maps the kernel names to all the available kernels.
  llvm::StringMap<KernelSpecs> specs_;

  // Points to the global MLIRContext.
  MLIRContext* ctx_;
};

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
