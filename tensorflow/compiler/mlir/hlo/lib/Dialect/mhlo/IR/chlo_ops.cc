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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc() {
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

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"

#include "llvm/ADT/APFloat.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace chlo {

Value getConstantLikeMaxFiniteValue(OpBuilder& b, Location loc, Value val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "getConstantLikeMaxFiniteValue");

  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

Value getConstantLikeInfValue(OpBuilder& b, Location loc, Value val,
                              bool negative) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "getConstantLikeInfValue");

  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

Value getConstantLikeSmallestFiniteValue(OpBuilder& b, Location loc,
                                         Value val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "getConstantLikeSmallestFiniteValue");

  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getSmallest(ty.getFloatSemantics()), val);
}

Value getConstantLike(OpBuilder& b, Location loc, const APFloat& constant,
                      Value val) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "getConstantLike");

  Type ty = getElementTypeOrSelf(val.getType());
  return b.create<ConstantLikeOp>(loc, b.getFloatAttr(ty, constant), val);
}

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
ShapedTypeComponents GetBroadcastType(
    Type x, Type y, Type element_type,
    DenseIntElementsAttr broadcast_dimensions_attr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "GetBroadcastType");

  auto x_ranked = x.dyn_cast<RankedTensorType>();
  auto y_ranked = y.dyn_cast<RankedTensorType>();
  if (!x_ranked || !y_ranked) {
    return {element_type};
  }

  auto shape_x = x_ranked.getShape();
  auto shape_y = y_ranked.getShape();

  // If no broadcast dimensions, assume "numpy" broadcasting.
  if (shape_x.size() == shape_y.size() || !broadcast_dimensions_attr) {
    llvm::SmallVector<int64_t, 4> out_shape;
    if (!mlir::OpTrait::util::getBroadcastedShape(shape_x, shape_y,
                                                  out_shape)) {
      // Signal illegal broadcast_dimensions as unranked.
      return {element_type};
    }
    return {out_shape, element_type};
  }

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  auto broadcast_dimensions = broadcast_dimensions_attr.getValues<APInt>();
  if (broadcast_dimensions.size() != shape_small.size()) {
    // Signal illegal broadcast_dimensions as unranked.
    return {element_type};
  }

  llvm::SmallVector<int64_t, 4> shape_large_filtered;
  shape_large_filtered.reserve(shape_small.size());
  for (const auto& dim : broadcast_dimensions) {
    if (dim.getZExtValue() >= shape_large.size()) return {element_type};
    shape_large_filtered.push_back(shape_large[dim.getZExtValue()]);
  }
  llvm::SmallVector<int64_t, 4> out_shape_filtered;
  if (!mlir::OpTrait::util::getBroadcastedShape(
          shape_small, shape_large_filtered, out_shape_filtered)) {
    // Signal illegal broadcast_dimensions as unranked.
    return {element_type};
  }

  // Update according to the broadcast dimensions.
  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
                                          shape_large.end());
  for (const auto& index_pair : llvm::enumerate(broadcast_dimensions)) {
    auto new_value = out_shape_filtered[index_pair.index()];
    out_shape[index_pair.value().getZExtValue()] = new_value;
  }

  return {out_shape, element_type};
}

LogicalResult InferBroadcastBinaryOpReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, Type element_type,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "InferBroadcastBinaryOpReturnTypeComponents");

  // Find broadcast_dimensions.
  DenseIntElementsAttr broadcast_dimensions =
      attributes.get("broadcast_dimensions")
          .dyn_cast_or_null<DenseIntElementsAttr>();

  ShapedType lhs_type = operands[0].getType().dyn_cast<ShapedType>();
  ShapedType rhs_type = operands[1].getType().dyn_cast<ShapedType>();
  if (!lhs_type || !rhs_type ||
      lhs_type.getElementType() != rhs_type.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }
  if (!element_type) element_type = lhs_type.getElementType();
  inferredReturnShapes.push_back(
      GetBroadcastType(lhs_type, rhs_type, element_type, broadcast_dimensions));
  return success();
}

LogicalResult ReifyBroadcastBinaryOpReturnTypeShapes(
    OpBuilder& builder, Operation* op, ValueRange operands,
    SmallVectorImpl<Value>& result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_6(mht_6_v, 329, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "ReifyBroadcastBinaryOpReturnTypeShapes");

  assert(operands.size() == 2 && "expect binary op");
  auto loc = op->getLoc();
  auto lhs = operands[0];
  auto rhs = operands[1];

  // Check for "numpy"-style rank broadcast.
  auto broadcast_dimensions = op->getAttr("broadcast_dimensions")
                                  .dyn_cast_or_null<DenseIntElementsAttr>();
  if (broadcast_dimensions &&
      !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, broadcast_dimensions)) {
    // Note: It is unclear whether the general specification of explicit
    // broadcast_dimensions on binary ops is a feature we want to carry
    // forward. While it can technically be implemented for ranked-dynamic,
    // it is incompatible with unranked inputs. If this warning is emitted
    // in real programs, it is an indication that the feature should be
    // implemented versus just falling back on the more standard definition
    // of numpy-like prefix-padding.
    return op->emitWarning()
           << "unsupported non prefix-padded dynamic rank "
           << "broadcast_dimensions = " << broadcast_dimensions;
  }

  result.push_back(hlo::ComputeBinaryElementwiseBroadcastingResultExtents(
      loc, lhs, rhs, builder));
  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// BroadcastComplexOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

LogicalResult BroadcastComplexOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_7(mht_7_v, 368, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastComplexOp::inferReturnTypeComponents");

  ShapedType lhs_type = operands[0].getType().dyn_cast<ShapedType>();
  if (!lhs_type) {
    return emitOptionalError(location, "expected ShapedType");
  }
  Type element_type = ComplexType::get(lhs_type.getElementType());
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, element_type,
                                                    inferedReturnShapes);
}
LogicalResult BroadcastComplexOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_8(mht_8_v, 383, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastComplexOp::reifyReturnTypeShapes");

  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                operands, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// BroadcastCompareOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

void BroadcastCompareOp::build(OpBuilder& builder, OperationState& result,
                               Value lhs, Value rhs,
                               DenseIntElementsAttr broadcast_dimensions,
                               mhlo::ComparisonDirection comparison_direction,
                               mhlo::ComparisonType compare_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_9(mht_9_v, 399, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastCompareOp::build");

  build(builder, result, lhs, rhs, broadcast_dimensions,
        mhlo::ComparisonDirectionAttr::get(builder.getContext(),
                                           comparison_direction),
        mhlo::ComparisonTypeAttr::get(builder.getContext(), compare_type));
}

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_10(mht_10_v, 412, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastCompareOp::inferReturnTypeComponents");

  Type element_type = IntegerType::get(context, 1);
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, element_type,
                                                    inferedReturnShapes);
}

LogicalResult BroadcastCompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_11(mht_11_v, 424, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastCompareOp::reifyReturnTypeShapes");

  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                operands, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

static Type getIsInfLikeReturnType(Value operand) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_12(mht_12_v, 436, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "getIsInfLikeReturnType");

  Builder b(operand.getContext());
  return mhlo::getSameShapeTensorType(operand.getType().cast<TensorType>(),
                                      b.getI1Type());
}

LogicalResult IsInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_13(mht_13_v, 447, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "IsInfOp::inferReturnTypes");

  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsNegInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsNegInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_14(mht_14_v, 461, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "IsNegInfOp::inferReturnTypes");

  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsPosInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsPosInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_15(mht_15_v, 475, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "IsPosInfOp::inferReturnTypes");

  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// Macros for method definitions that are common to most broadcasting ops.
//===----------------------------------------------------------------------===//

#define BROADCAST_BINARY_OP_DEFS(Op)                                       \
  LogicalResult Op::inferReturnTypeComponents(                             \
      MLIRContext* context, Optional<Location> location,                   \
      ValueShapeRange operands, DictionaryAttr attributes,                 \
      RegionRange regions,                                                 \
      SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {        \
    return InferBroadcastBinaryOpReturnTypeComponents(                     \
        context, location, operands, attributes, /*element_type=*/nullptr, \
        inferedReturnShapes);                                              \
  }                                                                        \
  LogicalResult Op::reifyReturnTypeShapes(                                 \
      OpBuilder& builder, ValueRange operands,                             \
      SmallVectorImpl<Value>& reifiedReturnShapes) {                       \
    return ReifyBroadcastBinaryOpReturnTypeShapes(                         \
        builder, getOperation(), operands, reifiedReturnShapes);           \
  }

BROADCAST_BINARY_OP_DEFS(BroadcastAddOp);
BROADCAST_BINARY_OP_DEFS(BroadcastAndOp);
BROADCAST_BINARY_OP_DEFS(BroadcastAtan2Op);
BROADCAST_BINARY_OP_DEFS(BroadcastDivOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMaxOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMinOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMulOp);
BROADCAST_BINARY_OP_DEFS(BroadcastNextAfterOp);
BROADCAST_BINARY_OP_DEFS(BroadcastOrOp);
BROADCAST_BINARY_OP_DEFS(BroadcastPolygammaOp);
BROADCAST_BINARY_OP_DEFS(BroadcastPowOp);
BROADCAST_BINARY_OP_DEFS(BroadcastRemOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftLeftOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightArithmeticOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightLogicalOp);
BROADCAST_BINARY_OP_DEFS(BroadcastSubOp);
BROADCAST_BINARY_OP_DEFS(BroadcastXorOp);
BROADCAST_BINARY_OP_DEFS(BroadcastZetaOp);

#undef BROADCAST_BINARY_OP_DEFS

LogicalResult ConstantLikeOp::verify() {
  if (value().getType() != getType().cast<ShapedType>().getElementType())
    return emitOpError() << "value's type doesn't match element return type";
  return success();
}

//===----------------------------------------------------------------------===//
// MinimumBroadcastShapesOp
//===----------------------------------------------------------------------===//
LogicalResult MinimumBroadcastShapesOp::verify() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_16(mht_16_v, 534, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "MinimumBroadcastShapesOp::verify");

  // Check that the number of operands matches the number of outputs.
  unsigned result_shapes_count = results().size();
  unsigned operand_shapes_count = shapes().size();
  if (operand_shapes_count != result_shapes_count) {
    return emitOpError() << "number of operand shapes (" << operand_shapes_count
                         << ") does not match number of result shapes ("
                         << result_shapes_count << ")";
  }
  if (operand_shapes_count < 2) {
    return emitOpError() << "number of operand shapes (" << operand_shapes_count
                         << ") should be >= 2";
  }
  return success();
}

LogicalResult ConstantLikeOp::inferReturnTypeComponents(
    MLIRContext* /*context*/, Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_17(mht_17_v, 557, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "ConstantLikeOp::inferReturnTypeComponents");

  ConstantLikeOp::Adaptor op(operands, attributes);
  if (failed(op.verify(location.getValue()))) return failure();
  Type element_type = op.value().getType();
  Type operand_type = op.operand().getType();
  if (operand_type.isa<UnrankedTensorType>()) {
    inferedReturnShapes.emplace_back(element_type);
  } else {
    const auto& shape = operand_type.cast<RankedTensorType>().getShape();
    inferedReturnShapes.emplace_back(shape, element_type);
  }
  return success();
}

LogicalResult ConstantLikeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_18(mht_18_v, 576, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "ConstantLikeOp::reifyReturnTypeShapes");

  return ::mlir::mhlo::deriveShapeFromOperand(
      &builder, getOperation(), operands.front(), &reifiedReturnShapes);
}

OpFoldResult ConstantLikeOp::fold(ArrayRef<Attribute> /*operands*/) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_19(mht_19_v, 584, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "ConstantLikeOp::fold");

  auto op_type = operand().getType().cast<ShapedType>();
  if (!op_type.hasStaticShape()) return {};
  auto type = RankedTensorType::get(op_type.getShape(), value().getType());
  return DenseElementsAttr::get(type, value());
}

LogicalResult BroadcastSelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_20(mht_20_v, 597, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastSelectOp::inferReturnTypeComponents");

  BroadcastSelectOp::Adaptor op(operands.getValues());
  auto pred_type = op.pred().getType().dyn_cast<ShapedType>();
  auto on_true_type = op.on_true().getType().dyn_cast<ShapedType>();
  auto on_false_type = op.on_false().getType().dyn_cast<ShapedType>();

  if (!pred_type || !on_true_type || !on_false_type ||
      on_true_type.getElementType() != on_false_type.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }

  Type element_type = on_true_type.getElementType();

  // Compute the result shape as two binary broadcasts.
  ShapedTypeComponents& components = inferredReturnShapes.emplace_back(
      GetBroadcastType(on_true_type, on_false_type, element_type, nullptr));
  if (components.hasRank()) {
    components = GetBroadcastType(
        RankedTensorType::get(components.getDims(), element_type), pred_type,
        element_type, nullptr);
  }
  return success();
}

LogicalResult BroadcastSelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands, SmallVectorImpl<Value>& result) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_21(mht_21_v, 625, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "BroadcastSelectOp::reifyReturnTypeShapes");

  result.push_back(hlo::ComputeNaryElementwiseBroadcastingResultExtents(
      getLoc(), operands, builder));
  return success();
}

//===----------------------------------------------------------------------===//
// RankSpecializationClusterOp
//===----------------------------------------------------------------------===//

void RankSpecializationClusterOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> /*operands*/,
    SmallVectorImpl<RegionSuccessor>& regions) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_22(mht_22_v, 640, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "RankSpecializationClusterOp::getSuccessorRegions");

  // RankSpecializationClusterOp has unconditional control flows into the region
  // and back to the parent, so return the correct RegionSuccessor purely based
  // on the index being None or 0.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }
  regions.push_back(RegionSuccessor(&body()));
}

LogicalResult RankSpecializationClusterOp::verify() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_23(mht_23_v, 654, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "RankSpecializationClusterOp::verify");

  if (body().getArgumentTypes() != getOperandTypes())
    return emitOpError() << "block argument types must match operand types";

  // All operands of nested ops must be defined in the body or declared by the
  // cluster.
  Block* body = getBody();
  for (Operation& nested : body->without_terminator()) {
    if (!llvm::all_of(nested.getOpOperands(), [&](OpOperand& operand) {
          Operation* def = operand.get().getDefiningOp();
          if (def != nullptr && def->getBlock() == body) return true;
          return llvm::is_contained(body->getArguments(), operand.get());
        })) {
      return emitOpError() << "nested ops must not depend on implicit operands";
    }
  }

  return success();
}

}  // namespace chlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// chlo Dialect Constructor
//===----------------------------------------------------------------------===//

Operation* HloClientDialect::materializeConstant(OpBuilder& builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_24(mht_24_v, 692, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "HloClientDialect::materializeConstant");

  // Mirror MHLO dialect here.
  if (value.isa<ElementsAttr>())
    return builder.create<mhlo::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

void HloClientDialect::initialize() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPSIRPSchlo_opsDTcc mht_25(mht_25_v, 702, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/chlo_ops.cc", "HloClientDialect::initialize");

  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"
      >();
}

}  // namespace chlo
}  // namespace mlir
