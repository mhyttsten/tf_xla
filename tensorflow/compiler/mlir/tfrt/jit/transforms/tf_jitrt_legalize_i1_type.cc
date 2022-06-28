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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc() {
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

#include <algorithm>
#include <utility>

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {

using llvm::APInt;
using llvm::ArrayRef;
using llvm::dyn_cast;
using llvm::Optional;
using llvm::SmallVector;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DenseElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::Type;
using mlir::TypeConverter;
using mlir::Value;
using mlir::func::FuncOp;

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

static Optional<Type> PromoteI1ToI8(Type input_type) {
  if (auto integer_type = input_type.dyn_cast<IntegerType>()) {
    if (integer_type.getWidth() == 1)
      return integer_type.scaleElementBitwidth(8);
  }

  return llvm::None;
}

/// TypeConverter that turns 'i1' tensors into 'i8' tensors.
class I1TypeConverter : public mlir::TypeConverter {
 public:
  using TypeConverter::convertType;

  I1TypeConverter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_0(mht_0_v, 239, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "I1TypeConverter");

    // Catch-all type conversion.
    addConversion([](Type type) { return type; });

    addConversion([](RankedTensorType tensor_type) -> Optional<Type> {
      auto maybe_promoted_i8_type = PromoteI1ToI8(tensor_type.getElementType());
      if (!maybe_promoted_i8_type) return tensor_type;
      return RankedTensorType::get(tensor_type.getShape(),
                                   *maybe_promoted_i8_type);
    });
  }
};

static bool isLegalType(const Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_1(mht_1_v, 255, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "isLegalType");

  if (auto tensor_type = type.dyn_cast<RankedTensorType>()) {
    if (auto integer_type =
            tensor_type.getElementType().dyn_cast<IntegerType>()) {
      return integer_type.getWidth() != 1;
    }
  }

  return true;
}

static bool isLegalAttribute(NamedAttribute attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_2(mht_2_v, 269, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "isLegalAttribute");

  if (auto int_attr = attr.getValue().dyn_cast<DenseIntElementsAttr>()) {
    // Only RankedTensorType is expected.
    ShapedType shaped_type = int_attr.getType();
    if (!shaped_type.isa<RankedTensorType>()) return true;
    return !shaped_type.getElementType().isInteger(/*width=*/1);
  }

  // TODO(diegocaballero): Add support for TypeAttr if/when we have a use case.

  return true;
}

static NamedAttribute convertAttribute(NamedAttribute attr,
                                       ConversionPatternRewriter &rewriter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_3(mht_3_v, 286, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "convertAttribute");

  if (auto int_attr = attr.getValue().dyn_cast<DenseIntElementsAttr>()) {
    ShapedType shaped_type = int_attr.getType();
    // Only RankedTensorType is expected.
    if (!shaped_type.isa<RankedTensorType>()) return attr;
    if (!shaped_type.getElementType().isInteger(/*width=*/1)) return attr;

    // Convert internal bool attribute representation to 8-bit integer.
    SmallVector<APInt, 4> new_i8_values;
    for (bool bool_val : int_attr.getValues<bool>()) {
      new_i8_values.push_back(
          bool_val ? APInt::getOneBitSet(/*numBits=*/8, /*bitNo=*/0)
                   : APInt::getZero(/*numBits=*/8));
    }

    auto i8_tensor_type =
        RankedTensorType::get(shaped_type.getShape(), rewriter.getI8Type());
    return NamedAttribute(
        attr.getName(), DenseElementsAttr::get(i8_tensor_type, new_i8_values));
  }

  // TODO(diegocaballero): Add support for TypeAttr if/when we have a use case.

  return attr;
}

/// Generic conversion pattern that replaces any operation (except FuncOp) using
/// 'i1' tensors with the same operation using 'i8' tensors.
struct I1ToI8GenericConversionPattern : public ConversionPattern {
  using ConversionPattern::ConversionPattern;

  I1ToI8GenericConversionPattern(I1TypeConverter &type_converter,
                                 MLIRContext *context)
      : ConversionPattern(type_converter, MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_4(mht_4_v, 323, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "I1ToI8GenericConversionPattern");
}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> converted_operands,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "matchAndRewrite");

    // Convert attributes.
    SmallVector<NamedAttribute, 4> new_attrs;
    for (NamedAttribute attr : op->getAttrs())
      new_attrs.push_back(convertAttribute(attr, rewriter));

    // Convert result types.
    SmallVector<Type, 4> new_result_types;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           new_result_types)))
      return mlir::failure();

    // Create a new op using the converted attributes, operands and result
    // types. If the existing op has regions, we move them to the new op and
    // convert their signature.
    OperationState new_op_state(op->getLoc(), op->getName().getStringRef(),
                                converted_operands, new_result_types, new_attrs,
                                op->getSuccessors());

    for (Region &region : op->getRegions()) {
      Region *new_region = new_op_state.addRegion();
      rewriter.inlineRegionBefore(region, *new_region, new_region->begin());

      TypeConverter::SignatureConversion signature_conv(
          new_region->getNumArguments());
      if (failed(typeConverter->convertSignatureArgs(
              new_region->getArgumentTypes(), signature_conv)))
        return mlir::failure();
      rewriter.applySignatureConversion(new_region, signature_conv);
    }

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
  }
};

static void populateI1TypeConversionPatterns(I1TypeConverter &type_converter,
                                             RewritePatternSet &patterns) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_6(mht_6_v, 371, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "populateI1TypeConversionPatterns");

  patterns.add<I1ToI8GenericConversionPattern>(type_converter,
                                               patterns.getContext());
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(
      patterns, type_converter);
}

struct JitRtLegalizeI1TypesPass
    : public JitRtLegalizeI1TypesBase<JitRtLegalizeI1TypesPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_legalize_i1_typeDTcc mht_7(mht_7_v, 383, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_legalize_i1_type.cc", "runOnOperation");

    MLIRContext &context = getContext();
    I1TypeConverter type_converter;

    ConversionTarget target(context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      // Check legality of attributes.
      auto attrs = op->getAttrs();
      if (std::any_of(attrs.begin(), attrs.end(), [&](NamedAttribute attr) {
            return !isLegalAttribute(attr);
          }))
        return false;

      // Check legality of FuncOp.
      if (FuncOp func_op = dyn_cast<FuncOp>(op)) {
        auto input_types = func_op.getFunctionType().getInputs();
        auto result_types = func_op.getFunctionType().getResults();
        return std::all_of(
                   input_types.begin(), input_types.end(),
                   [&](const Type type) { return isLegalType(type); }) &&
               std::all_of(result_types.begin(), result_types.end(),
                           [&](const Type type) { return isLegalType(type); });
      }

      // Check legality of any other op.
      auto operand_types = op->getOperandTypes();
      auto result_types = op->getResultTypes();
      return std::all_of(operand_types.begin(), operand_types.end(),
                         [](Type type) { return isLegalType(type); }) &&
             std::all_of(result_types.begin(), result_types.end(),
                         [](Type type) { return isLegalType(type); });
    });

    RewritePatternSet patterns(&context);
    populateI1TypeConversionPatterns(type_converter, patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
CreateJitRtLegalizeI1TypesPass() {
  return std::make_unique<JitRtLegalizeI1TypesPass>();
}

}  // namespace tensorflow
