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
class MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc() {
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

// This pass converts a TFLite uint8 graph to the int8 domain, with adaptors at
// input and output tensors. This is needed because TOSA precision is
// implemented in the int8 domain. This pass does:
// 1. match TFL::QConst with uint8, generate TFL::QConst with int8 with value
// remapped.
// 2. insert tosa.RESCALE uint8 -> int8 if block argument (placeholder of graph)
// is uint8 typed.
// 3. insert tosa.RESCALE int8 -> uint8 if original returned tensor is uint8
// typed.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-convert-tfl-uint8"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class StripQuantTypes : public TosaStripQuantTypesPassBase<StripQuantTypes> {
 public:
  explicit StripQuantTypes() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_0(mht_0_v, 226, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "StripQuantTypes");
}
  void runOnOperation() override;
};

class QuantTypeConverter : public TypeConverter {
 public:
  static Type convertType(Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "convertType");

    if (auto qType = type.dyn_cast<quant::QuantizedType>()) {
      if (qType.isSigned() || qType.getStorageTypeIntegralWidth() != 8) {
        return IntegerType::get(type.getContext(),
                                qType.getStorageTypeIntegralWidth());
      }

      return IntegerType::get(type.getContext(),
                              qType.getStorageTypeIntegralWidth(),
                              IntegerType::SignednessSemantics::Unsigned);
    }
    return type;
  }
  static Type convertTensor(RankedTensorType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "convertTensor");

    auto newType = RankedTensorType::get(type.getShape(),
                                         convertType(type.getElementType()));
    return newType;
  }
  explicit QuantTypeConverter() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "QuantTypeConverter");

    addConversion([](Type type) { return convertType(type); });
    addConversion(convertTensor);
  }
};

// Handles the type conversion component of the TypeConversion. This updates
// conversion patterns that used the original Quant types to be updated to
// the non-quant variants.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(MLIRContext* context, TypeConverter& converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_4(mht_4_v, 274, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "GenericTypeConvert");
}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_5(mht_5_v, 280, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 4> newResults;
    if (isa<FuncOp>(op)) {
      return failure();
    }

    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, op->getAttrs(), op->getSuccessors());
    for (Region& r : op->getRegions()) {
      Region* newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static bool isIllegalType(Type type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_6(mht_6_v, 306, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "isIllegalType");

  if (type.isa<quant::QuantizedType>()) return true;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return isIllegalType(shapedType.getElementType());
  }
  return false;
}

void StripQuantTypes::runOnOperation() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSstrip_quant_typesDTcc mht_7(mht_7_v, 317, "", "./tensorflow/compiler/mlir/tosa/transforms/strip_quant_types.cc", "StripQuantTypes::runOnOperation");

  QuantTypeConverter converter;
  ConversionTarget target(getContext());

  target.addIllegalDialect<quant::QuantizationDialect>();
  // Operations are legal if they don't contain any illegal type.
  target.markUnknownOpDynamicallyLegal([](Operation* op) {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : funcOp.getFunctionType().getResults()) {
        if (isIllegalType(type)) return false;
      }
    }
    for (Type type : op->getResultTypes()) {
      if (type && isIllegalType(type)) return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (type && isIllegalType(type)) return false;
    }
    return true;
  });

  auto* ctx = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);

  if (failed(applyFullConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createStripQuantTypesPass() {
  return std::make_unique<StripQuantTypes>();
}
}  // namespace tosa
}  // namespace mlir
