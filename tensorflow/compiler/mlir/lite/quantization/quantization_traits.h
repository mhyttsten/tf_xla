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

// This file defines the op traits used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_TRAITS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh() {
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


#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

using QuantizedType = mlir::quant::QuantizedType;
using UniformQuantizedType = mlir::quant::UniformQuantizedType;

namespace mlir {
namespace quant {
// Verify that the op satisfies the same operands and results scales
// constraints. Note that this constraint can only be applied on some
// storage types of the op.
LogicalResult VerifySameScales(Operation* op);
}  // namespace quant

// This includes the interface class definition. It couldn't be in a namespace
// because the table gen doesn't emit the namespace when it is used.
#include "tensorflow/compiler/mlir/lite/quantization/quantization_interface.h.inc"

namespace OpTrait {
namespace quant {

// The base class that all the quantization related OpTrait implements.
template <typename ConcreteType, template <typename> class TraitType>
struct QuantizationSpecTraitBase : public TraitBase<ConcreteType, TraitType> {
  static bool IsBias(int index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "IsBias");
 return false; }
  static bool IsQuantizable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "IsQuantizable");
 return true; }
};

// This class provides the API for TFL ops that has a fixed output value range.
// This is used as a trait like this:
//
//   class SoftmaxOp
//       : public Op<SoftmaxOp,
//           OpTrait::quant::FixedResultUniformScale<
//               8, -128, 390625, -8, 0, 255, false>::Impl> {
//
// TODO(fengliuai): create a better way to express floating point scale in the
// template argument list.
template <unsigned BitWidth, int ZeroPoint, int ScaleMantissa, int ScaleExp,
          int64_t StorageTypeMin, int64_t StorageTypeMax, bool Sign>
class FixedResultUniformScale {
 public:
  template <typename ConcreteType>
  class Impl
      : public QuantizationSpecTraitBase<
            ConcreteType, FixedResultUniformScale<
                              BitWidth, ZeroPoint, ScaleMantissa, ScaleExp,
                              StorageTypeMin, StorageTypeMax, Sign>::Impl> {
   public:
    QuantizedType GetResultQuantizedType(int index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "GetResultQuantizedType");

      auto op = this->getOperation();
      auto result_type =
          op->getResult(index).getType().template cast<ShapedType>();
      if (!result_type.getElementType().template isa<FloatType>()) return {};
      Builder builder(op->getContext());
      IntegerType storage_type = builder.getIntegerType(BitWidth);
      const double scale = static_cast<double>(ScaleMantissa) *
                           ::pow(10.0, static_cast<double>(ScaleExp));
      return UniformQuantizedType::getChecked(
          Sign, storage_type, result_type.getElementType(), scale, ZeroPoint,
          StorageTypeMin, StorageTypeMax, builder.getUnknownLoc());
    }
  };
};

// This class provides the API for TFL ops that has input as bias. This is used
// as a trait like this:
//
//   class Conv2DOp
//       : public Op<Conv2DOp, OpTrait::quant::AccumulatorScale<2, 0, 1>::Impl>
//
// TODO(fengliuai): supports a configurable accumulator bit width.
template <int Bias, int... Operands>
class AccumulatorUniformScale {
 public:
  template <typename ConcreteType>
  class Impl
      : public QuantizationSpecTraitBase<
            ConcreteType, AccumulatorUniformScale<Bias, Operands...>::Impl> {
   public:
    // Whether the index-th operand is a bias.
    static bool IsBias(int index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_3(mht_3_v, 281, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "IsBias");
 return index == Bias; }

    // Returns the indexes of all the non-bias operands.
    static std::vector<int> GetAllNonBiasOperands() {
      return std::vector<int>({Operands...});
    }
  };
};

// The trait to specify the operand index of the coefficient for an affine op
// and also the quantization dimension if per-axis quantization is support.
// If the quantization dimension is -1, per-axis quantization isn't supported.
//
//   class Conv2DOp
//       : public Op<Conv2DOp, OpTrait::quant::AffineOpCoefficient<0>::Impl>
//
template <int QuantDim, int OperandIndex = 1>
class AffineOpCoefficient {
 public:
  template <typename ConcreteType>
  class Impl
      : public TraitBase<ConcreteType,
                         AffineOpCoefficient<QuantDim, OperandIndex>::Impl> {
   public:
    static int GetCoefficientOperandIndex() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_4(mht_4_v, 308, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "GetCoefficientOperandIndex");
 return OperandIndex; }
    static int GetQuantizationDim() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_traitsDTh mht_5(mht_5_v, 312, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_traits.h", "GetQuantizationDim");
 return QuantDim; }
  };
};

// This class provides the API for TFL ops that can be quantized.
// This is as a trait like this:
//
//   class LessOp : public Op<LessOp, OpTrait::quant::QuantizableResult> {
//
template <typename ConcreteType>
class QuantizableResult
    : public QuantizationSpecTraitBase<ConcreteType, QuantizableResult> {};

}  // namespace quant
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_TRAITS_H_
