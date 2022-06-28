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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc() {
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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <algorithm>
#include <utility>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

Value InsertDynamicAlloc(Location loc, Value result, Value shape_operand,
                         ConversionPatternRewriter* rewriter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "InsertDynamicAlloc");

  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamic_operands;
  for (const auto& shape_element : llvm::enumerate(result_type.getShape())) {
    if (shape_element.value() != ShapedType::kDynamicSize) continue;
    Value index =
        rewriter->create<arith::ConstantIndexOp>(loc, shape_element.index());
    Value alloc_operand =
        rewriter->create<tensor::ExtractOp>(loc, shape_operand, index);
    if (!alloc_operand.getType().isIndex()) {
      alloc_operand = rewriter->create<arith::IndexCastOp>(
          loc, rewriter->getIndexType(), alloc_operand);
    }
    dynamic_operands.push_back(alloc_operand);
  }

  return rewriter->create<memref::AllocOp>(loc, memref_type, dynamic_operands);
}

Value InsertAlloc(Location loc, OpResult result,
                  ConversionPatternRewriter* rewriter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_1(mht_1_v, 258, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "InsertAlloc");

  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type || !result_type.hasStaticShape()) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());
  OpBuilder::InsertionGuard guard(*rewriter);
  rewriter->setInsertionPoint(result.getDefiningOp());
  auto alloc = rewriter->create<memref::AllocOp>(loc, memref_type);
  return alloc;
}

/// Converts the results of the operation `op` to memref types and append them
/// to the `results` vector.
LogicalResult ConvertResults(Operation* op, SmallVectorImpl<Value>& results,
                             ConversionPatternRewriter& rewriter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "ConvertResults");

  size_t num_operands = results.size();
  SmallVector<Value, 2> tensor_operands;
  for (const auto& result : llvm::enumerate(op->getResults())) {
    RankedTensorType resultType =
        result.value().getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    if (resultType.hasStaticShape()) {
      results.push_back(InsertAlloc(op->getLoc(), result.value(), &rewriter));
      continue;
    }
    auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
    if (!shape_type_op) return failure();

    if (tensor_operands.empty()) {
      for (auto operand : ArrayRef<Value>(results).take_front(num_operands)) {
        auto operand_type = operand.getType().dyn_cast<MemRefType>();
        if (!operand_type) return failure();
        tensor_operands.push_back(rewriter.create<bufferization::ToTensorOp>(
            op->getLoc(),
            RankedTensorType::get(operand_type.getShape(),
                                  operand_type.getElementType()),
            operand));
      }
    }

    SmallVector<Value, 1> results_shape;
    auto status = shape_type_op.reifyReturnTypeShapes(rewriter, tensor_operands,
                                                      results_shape);
    if (failed(status)) return failure();
    results.push_back(InsertDynamicAlloc(op->getLoc(), result.value(),
                                         results_shape[result.index()],
                                         &rewriter));
  }
  return success();
}

template <typename HloOpTy>
class HloToLhloOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_3(mht_3_v, 325, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    Operation* op = hloOp.getOperation();
    SmallVector<Value, 4> buffer_args(adaptor.getOperands());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(op->getLoc(), llvm::None,
                                                buffer_args, op->getAttrs());
    rewriter.replaceOp(op, llvm::makeArrayRef(buffer_args)
                               .drop_front(adaptor.getOperands().size()));
    return success();
  }
};

// This specialization exists so that LMHLO's Dot can be given a specific set of
// dimension numbers, when lowering from MHLO's Dot, which does not have
// dimension numbers (it uses DotGeneral for this generalized notion of dot
// products). When these two dialects are in sync with respect to the
// Dot/DotGeneral issue, this specialization should be deleted.
template <>
class HloToLhloOpConverter<mhlo::DotOp> : public BaseOpConversion<mhlo::DotOp> {
 public:
  using BaseOpConversion<mhlo::DotOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      mhlo::DotOp hloOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_4(mht_4_v, 351, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> buffer_args(adaptor.getOperands());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();

    auto dotOp = rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None,
                                               buffer_args, op->getAttrs());
    // MHLO's Dot uses rank-2 operands, of the form ([N, M], [M, O]) -> [N, O].
    auto dimension_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
        /*rhsContractingDimensions=*/{0});
    dotOp.dot_dimension_numbersAttr(dimension_numbers);
    rewriter.replaceOp(
        op, ArrayRef<Value>(buffer_args).slice(adaptor.getOperands().size()));
    return success();
  }
};

struct HloToLhloCustomCallOpConverter
    : public BaseOpConversion<mhlo::CustomCallOp> {
 public:
  using BaseOpConversion<mhlo::CustomCallOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp hloOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_5(mht_5_v, 380, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> buffer_args(adaptor.getOperands());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();

    auto lhloOp = rewriter.create<lmhlo::CustomCallOp>(
        op->getLoc(), llvm::None, buffer_args, op->getAttrs());
    // Setup AttrSizedOperandSegments attribute to indicate number of operands
    // for args and outputs.
    const int32_t segments[2] = {
        static_cast<int32_t>(adaptor.getOperands().size()),
        static_cast<int32_t>(op->getNumResults())};
    lhloOp->setAttr(lhloOp.getOperandSegmentSizeAttr(),
                    rewriter.getI32VectorAttr(segments));

    rewriter.replaceOp(
        op, ArrayRef<Value>(buffer_args).slice(adaptor.getOperands().size()));
    return success();
  }
};

struct HloToLhloDotGeneralOpConverter
    : public BaseOpConversion<mhlo::DotGeneralOp> {
  using BaseOpConversion<mhlo::DotGeneralOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp dotGeneralOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_6(mht_6_v, 409, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    Operation* op = dotGeneralOp.getOperation();

    if (op->getResults().empty()) return failure();
    OpResult result = op->getResults()[0];
    RankedTensorType resultType = result.getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    // The third buffer argument will be filled with what used to be the return
    // type of the DotGeneral.
    if (adaptor.getOperands().size() != 2) return failure();
    std::array<Value, 3> bufferArgs = {
        adaptor.getOperands()[0], adaptor.getOperands()[1], {}};

    if (resultType.hasStaticShape()) {
      bufferArgs[2] = InsertAlloc(op->getLoc(), result, &rewriter);
    } else {
      SmallVector<Value, 1> results_shape;
      auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
      if (failed(shape_type_op.reifyReturnTypeShapes(
              rewriter, adaptor.getOperands(), results_shape)))
        return failure();

      bufferArgs[2] = InsertDynamicAlloc(op->getLoc(), result,
                                         results_shape.front(), &rewriter);
    }

    rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None, bufferArgs,
                                  op->getAttrs());
    rewriter.replaceOp(op, bufferArgs[2]);
    return success();
  }
};

template <typename HloOpTy>
struct HloToLhloReduceLikeOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_7(mht_7_v, 453, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    Operation* op = hloOp.getOperation();
    auto loc = op->getLoc();
    if (!llvm::hasSingleElement(hloOp.body())) {
      return op->emitOpError()
             << "tensor to buffer conversion expects a single block "
                "in the region containing the operation";
    }
    SmallVector<Value, 4> buffer_args(adaptor.getOperands());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    auto new_op = rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(
        loc, llvm::None, buffer_args, op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(hloOp.body(), new_op.body(),
                                new_op.body().end());

    // Convert the region signature to memref and add extra result.
    auto& entry_block = new_op.body().front();
    TypeConverter::SignatureConversion sig_conversion(
        adaptor.getOperands().size());
    for (auto arg : entry_block.getArguments()) {
      auto old_type = arg.getType().template cast<TensorType>();
      auto new_type =
          MemRefType::get(old_type.getShape(), old_type.getElementType());
      sig_conversion.addInputs(arg.getArgNumber(), new_type);
    }
    auto return_op = cast<mhlo::ReturnOp>(entry_block.getTerminator());
    if (auto tuple_ty = return_op.results()
                            .front()
                            .getType()
                            .template dyn_cast<TupleType>()) {
      auto* tuple_op = return_op.getODSOperands(0).front().getDefiningOp();
      return_op.getOperation()->dropAllReferences();
      rewriter.eraseOp(tuple_op);
      return_op.getOperation()->setOperands(tuple_op->getOperands());
      for (auto ty : tuple_ty) {
        auto tensor_ty = ty.template cast<TensorType>();
        sig_conversion.addInputs(
            MemRefType::get(tensor_ty.getShape(), tensor_ty.getElementType()));
      }
    } else {
      for (auto result : return_op.results()) {
        auto result_type = result.getType().template cast<TensorType>();
        sig_conversion.addInputs({MemRefType::get(
            result_type.getShape(), result_type.getElementType())});
      }
    }
    rewriter.applySignatureConversion(&new_op.body(), sig_conversion);

    rewriter.replaceOp(
        op, ArrayRef<Value>(buffer_args).slice(adaptor.getOperands().size()));

    return success();
  }
};

// Legalize mhlo.return to a lmhlo.copy and lmhlo.terminator.
struct HloToLhloReturnOpConverter : public BaseOpConversion<mhlo::ReturnOp> {
 public:
  using BaseOpConversion<mhlo::ReturnOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_8(mht_8_v, 520, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "matchAndRewrite");

    auto loc = op.getLoc();
    auto& entry_block = op->getParentRegion()->front();
    auto num_arguments = entry_block.getNumArguments();
    if (adaptor.getOperands().size() > num_arguments) {
      return op.emitError(
          "The number of operands that need Copy operations is more "
          "than the number of target function arguments.");
    }

    // The index of the first output block argument.
    auto dest_arg_idx = num_arguments - adaptor.getOperands().size();

    // Create a lmhlo.copy for each operand of mhlo.return.
    for (Value operand : adaptor.getOperands()) {
      rewriter.create<lmhlo::CopyOp>(loc, operand,
                                     entry_block.getArgument(dest_arg_idx));
      ++dest_arg_idx;
    }
    rewriter.replaceOpWithNewOp<lmhlo::TerminatorOp>(op);
    return success();
  }
};

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
//
// Example fusion with HLO ops.
//
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "lmhlo.fusion"() ({
//     %0 = bufferization.to_tensor %arg1 : memref<2x2xf32>
//     %1 = bufferization.to_tensor %arg2 : memref<2x2xf32>
//     %2 = "mhlo.add"(%0, %1) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = bufferization.to_tensor %arg0 : memref<2x2xf32>
//     %4 = "mhlo.multiply"(%2, %3) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     tensor_store %4, %arg3 : memref<2x2xf32>
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
// }
//
// Transformed fusion with LHLO ops.
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "lmhlo.fusion"() ({
//     %0 = alloc() : memref<2x2xf32>
//     "lmhlo.add"(%arg1, %arg2, %0) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "lmhlo.multiply"(%0, %arg0, %arg3) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
// }
//
// FuncOp signature conversion example:
//
// func @func_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
//   %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) ->
//   tensor<4xf32> %1 = "mhlo.add"(%arg0, %0)  : (tensor<4xf32>,
//   tensor<4xf32>) -> tensor<4xf32> return %1 : tensor<4xf32>
// }
//
// Transformed function with an extra argument for the result. The types have
// been converted from tensor to memref.
//
// func @func_op(%arg0: memref<4xf32>,
//               %arg1: memref<4xf32>,
//               %arg2: memref<4xf32>) {
//   %0 = alloc() : memref<4xf32>

//   "lmhlo.maximum"(%arg0, %arg1, %0) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   %1 = alloc() : memref<4xf32>
//   "lmhlo.add"(%arg0, %0, %1) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   "lmhlo.copy"(%1, %arg2) : (memref<4xf32>, memref<4xf32>) -> ()
//   "lmhlo.terminator"() : () -> ()
// }

struct HloLegalizeToLhlo : public HloLegalizeToLhloPassBase<HloLegalizeToLhlo> {
  using HloLegalizeToLhloPassBase<HloLegalizeToLhlo>::HloLegalizeToLhloPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_9(mht_9_v, 614, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "getDependentDialects");

    registry.insert<bufferization::BufferizationDialect, lmhlo::LmhloDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
    shape::registerBufferizableOpInterfaceExternalModels(registry);
  }

 public:
  HloLegalizeToLhlo() = default;

  LogicalResult runOpInterfaceBufferization() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_10(mht_10_v, 626, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "runOpInterfaceBufferization");

    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    RewritePatternSet patterns(&getContext());
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<shape::ShapeDialect>();
    return bufferization::bufferizeOp(getOperation(), options);
  }

  void runOnOperation() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_11(mht_11_v, 641, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "runOnOperation");

    if (failed(runOpInterfaceBufferization())) {
      signalPassFailure();
      return;
    }

    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        lmhlo::LmhloDialect, memref::MemRefDialect, shape::ShapeDialect,
        func::FuncDialect, tensor::TensorDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    // bufferization.to_memref is illegal if it has uses.
    // TODO(b/175670649) Make bufferization.to_memref illegal.
    target.addDynamicallyLegalOp<mlir::bufferization::ToMemrefOp>(
        [](auto op) { return op->use_empty(); });

    bufferization::BufferizeTypeConverter converter;
    auto isMemRefType = [](Type type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_12(mht_12_v, 664, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "lambda");
 return type.isa<BaseMemRefType>(); };
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType) &&
             std::all_of(op.result_type_begin(), op.result_type_end(),
                         isMemRefType);
    });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                             isMemRefType);
        });

    populateHLOToLHLOConversionPattern(&context, &converter, &patterns);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

// Simply lowers all mhlo ops to their lmhlo counterparts.
void populateDynamicHLOToLHLOConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_13(mht_13_v, 702, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "populateDynamicHLOToLHLOConversionPattern");

  // clang-format off
  patterns->add<HloToLhloOpConverter<mhlo::DynamicBroadcastInDimOp>,
                   HloToLhloOpConverter<mhlo::DynamicGatherOp>,
                   HloToLhloOpConverter<mhlo::DynamicIotaOp>,
                   HloToLhloOpConverter<mhlo::DynamicPadOp>,
                   HloToLhloOpConverter<mhlo::DynamicReshapeOp>,
                   HloToLhloOpConverter<mhlo::RealDynamicSliceOp>
  >(*converter, context);
  // clang-format on
}

void populateHLOToLHLOConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_lhloDTcc mht_14(mht_14_v, 719, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc", "populateHLOToLHLOConversionPattern");

  populateDynamicHLOToLHLOConversionPattern(context, converter, patterns);

  // clang-format off
  patterns->add<
      HloToLhloCustomCallOpConverter,
      HloToLhloDotGeneralOpConverter,
      HloToLhloOpConverter<mhlo::AbsOp>,
      HloToLhloOpConverter<mhlo::AddOp>,
      HloToLhloOpConverter<mhlo::AndOp>,
      HloToLhloOpConverter<mhlo::Atan2Op>,
      HloToLhloOpConverter<mhlo::BroadcastInDimOp>,
      HloToLhloOpConverter<mhlo::CeilOp>,
      HloToLhloOpConverter<mhlo::ClampOp>,
      HloToLhloOpConverter<mhlo::CompareOp>,
      HloToLhloOpConverter<mhlo::ComplexOp>,
      HloToLhloOpConverter<mhlo::ConcatenateOp>,
      HloToLhloOpConverter<mhlo::ConstOp>,
      HloToLhloOpConverter<mhlo::ConvOp>,
      HloToLhloOpConverter<mhlo::ConvertOp>,
      HloToLhloOpConverter<mhlo::CopyOp>,
      HloToLhloOpConverter<mhlo::CosOp>,
      HloToLhloOpConverter<mhlo::DivOp>,
      HloToLhloOpConverter<mhlo::DotOp>,
      HloToLhloOpConverter<mhlo::ExpOp>,
      HloToLhloOpConverter<mhlo::Expm1Op>,
      HloToLhloOpConverter<mhlo::FloorOp>,
      HloToLhloOpConverter<mhlo::GatherOp>,
      HloToLhloOpConverter<mhlo::ImagOp>,
      HloToLhloOpConverter<mhlo::IotaOp>,
      HloToLhloOpConverter<mhlo::IsFiniteOp>,
      HloToLhloOpConverter<mhlo::LogOp>,
      HloToLhloOpConverter<mhlo::LogisticOp>,
      HloToLhloOpConverter<mhlo::MaxOp>,
      HloToLhloOpConverter<mhlo::MinOp>,
      HloToLhloOpConverter<mhlo::MulOp>,
      HloToLhloOpConverter<mhlo::NegOp>,
      HloToLhloOpConverter<mhlo::NotOp>,
      HloToLhloOpConverter<mhlo::OrOp>,
      HloToLhloOpConverter<mhlo::PowOp>,
      HloToLhloOpConverter<mhlo::RealOp>,
      HloToLhloOpConverter<mhlo::RemOp>,
      HloToLhloOpConverter<mhlo::RsqrtOp>,
      HloToLhloOpConverter<mhlo::ReshapeOp>,
      HloToLhloOpConverter<mhlo::SelectOp>,
      HloToLhloOpConverter<mhlo::ShiftLeftOp>,
      HloToLhloOpConverter<mhlo::ShiftRightArithmeticOp>,
      HloToLhloOpConverter<mhlo::ShiftRightLogicalOp>,
      HloToLhloOpConverter<mhlo::SignOp>,
      HloToLhloOpConverter<mhlo::SinOp>,
      HloToLhloOpConverter<mhlo::SliceOp>,
      HloToLhloOpConverter<mhlo::SqrtOp>,
      HloToLhloOpConverter<mhlo::SubOp>,
      HloToLhloOpConverter<mhlo::TanhOp>,
      HloToLhloOpConverter<mhlo::TransposeOp>,
      HloToLhloOpConverter<mhlo::XorOp>,
      HloToLhloReduceLikeOpConverter<mhlo::ReduceOp>,
      HloToLhloReduceLikeOpConverter<mhlo::ReduceWindowOp>,
      HloToLhloReturnOpConverter
  >(*converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass() {
  return std::make_unique<HloLegalizeToLhlo>();
}

}  // namespace mhlo
}  // namespace mlir
