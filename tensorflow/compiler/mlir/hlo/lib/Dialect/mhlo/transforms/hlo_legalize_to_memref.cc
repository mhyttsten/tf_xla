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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc() {
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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <functional>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mhlo {
namespace {

using bufferization::BufferizableOpInterface;
using bufferization::BufferizationState;
using bufferization::BufferRelation;
using bufferization::replaceOpWithNewBufferizedOp;

struct ReshapeOpInterface
    : public BufferizableOpInterface::ExternalModel<ReshapeOpInterface,
                                                    mhlo::ReshapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryRead");

    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryWrite");

    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferRelation");

    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferize");

    auto reshape_op = cast<mhlo::ReshapeOp>(op);
    auto unranked_operand_type =
        reshape_op.operand().getType().dyn_cast<UnrankedTensorType>();
    if (unranked_operand_type == nullptr) return failure();

    // The buffer still has the old (pre-reshape) type.
    FailureOr<Value> operand_buffer =
        state.getBuffer(rewriter, reshape_op->getOpOperand(0) /*operand*/);
    if (failed(operand_buffer)) return failure();

    auto result_type = reshape_op.getType().cast<RankedTensorType>();
    auto dest_type =
        MemRefType::get(result_type.getShape(), result_type.getElementType());
    replaceOpWithNewBufferizedOp<memref::CastOp>(rewriter, op, dest_type,
                                                 *operand_buffer);
    return success();
  }
};

struct DynamicReshapeOpInterface
    : public BufferizableOpInterface::ExternalModel<DynamicReshapeOpInterface,
                                                    mhlo::DynamicReshapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryRead");

    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_5(mht_5_v, 285, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryWrite");

    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferRelation");

    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_7(mht_7_v, 307, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferize");

    auto reshape_op = cast<mhlo::DynamicReshapeOp>(op);

    // The buffer still has the old (pre-reshape) type.
    FailureOr<Value> operand_buffer =
        state.getBuffer(rewriter, reshape_op->getOpOperand(0) /*operand*/);
    if (failed(operand_buffer)) return failure();
    FailureOr<Value> output_shape_buffer =
        state.getBuffer(rewriter, reshape_op->getOpOperand(1) /*output_shape*/);
    if (failed(output_shape_buffer)) return failure();

    ShapedType result_type;
    TensorType op_result_type = reshape_op.getType();
    if (auto ranked_type = op_result_type.dyn_cast<RankedTensorType>()) {
      result_type =
          MemRefType::get(ranked_type.getShape(), ranked_type.getElementType());
    } else if (auto unranked_type =
                   op_result_type.dyn_cast<UnrankedTensorType>()) {
      result_type = UnrankedMemRefType::get(unranked_type.getElementType(), 0);
    }
    bufferization::replaceOpWithNewBufferizedOp<memref::ReshapeOp>(
        rewriter, op, result_type, *operand_buffer, *output_shape_buffer);
    return success();
  }
};

// Inserts dynamic memref to change the layout of the memref to put 0-stride
// and size of the target dimension if size-1 dimension expansion is
// necessary.
memref::ReinterpretCastOp InsertDynamicMemrefCastOp(
    mhlo::DynamicBroadcastInDimOp op, Value operand, OpBuilder *b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_8(mht_8_v, 340, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "InsertDynamicMemrefCastOp");

  auto loc = op.getLoc();
  auto operand_type = operand.getType().cast<MemRefType>();
  auto operand_shape = operand_type.getShape();
  auto operand_rank = operand_type.getRank();

  auto result_type = op.getType().cast<RankedTensorType>();
  auto result_rank = result_type.getRank();

  Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
  Value one = b->create<arith::ConstantIndexOp>(loc, 1);

  // Compute a reversed scan product. Compute the stride for the dimensions so
  // far, working from minor to major dimensions. Additionally, save the
  // operand shape Values to use in the next loop.
  SmallVector<Value, 2> operand_strides(operand_rank, one);
  SmallVector<Value, 2> operand_sizes(operand_rank, one);
  Value stride_so_far = one;
  for (int i = operand_rank - 1; i >= 0; --i) {
    Value operand_dim_size =
        ShapedType::isDynamic(operand_shape[i])
            ? b->create<memref::DimOp>(loc, operand, i).getResult()
            : b->create<arith::ConstantIndexOp>(loc, operand_shape[i])
                  .getResult();
    operand_sizes[i] = operand_dim_size;

    operand_strides[i] = stride_so_far;
    if (i > 0) {
      stride_so_far =
          b->create<arith::MulIOp>(loc, stride_so_far, operand_dim_size);
    }
  }

  SmallVector<OpFoldResult, 2> sizes, strides;
  sizes.reserve(result_rank);
  strides.reserve(result_rank);

  DenseMap<int, int> output_to_input_dim;
  for (const auto &dim : llvm::enumerate(op.broadcast_dimensions())) {
    output_to_input_dim[dim.value().getSExtValue()] = dim.index();
  }
  for (int i = 0; i < result_rank; ++i) {
    Value i_val = b->create<arith::ConstantIndexOp>(loc, i);
    Value result_dim_size =
        b->create<tensor::ExtractOp>(loc, op.output_dimensions(), i_val);
    if (!result_dim_size.getType().isIndex()) {
      result_dim_size = b->create<arith::IndexCastOp>(loc, b->getIndexType(),
                                                      result_dim_size);
    }
    if (result_type.isDynamicDim(i)) {
      sizes.push_back(result_dim_size);
    } else {
      sizes.push_back(b->getIndexAttr(result_type.getDimSize(i)));
    }

    auto it = output_to_input_dim.find(i);
    // If the rank of the output is greater than the rank of the input, i.e.
    // there was no output dimension in the inverse broadcast_dimensions map
    // we also set stride to 0 to emulate padding of the shape with 1s and the
    // corresponding expansion.
    if (it == output_to_input_dim.end()) {
      strides.push_back(zero);
      continue;
    }

    // There can be two cases:
    // 1) Operand dim == result dim => expansion is not needed
    //    => stride flattened buffer stride
    // 2) Operand dim < result dim => expansion is needed => stride := 0.
    int dim = it->second;
    Value is_expansion = b->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, operand_sizes[dim], result_dim_size);
    Value select = b->create<mlir::arith::SelectOp>(loc, is_expansion, zero,
                                                    operand_strides[dim]);
    strides.push_back(select);
  }

  // Type-erased memref type with static rank and dynamic strides.
  SmallVector<int64_t, 2> dynamic_layout(result_rank,
                                         ShapedType::kDynamicStrideOrOffset);
  auto type_erased_memref_type = MemRefType::get(
      result_type.getShape(), operand_type.getElementType(),
      makeStridedLinearLayoutMap(dynamic_layout,
                                 /*offset=*/0, b->getContext()));

  auto transformed_operand = b->create<memref::ReinterpretCastOp>(
      loc, type_erased_memref_type, operand,
      /*offset=*/b->getI64IntegerAttr(0), sizes, strides);
  return transformed_operand;
}

Value CreateCopy(mhlo::DynamicBroadcastInDimOp op, Value broadcasted,
                 OpBuilder *b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_9(mht_9_v, 435, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "CreateCopy");

  MemRefType result_type = broadcasted.getType().cast<MemRefType>();
  auto loc = op.getLoc();
  SmallVector<Value, 4> dynamic_operands;
  for (int i = 0; i < result_type.getRank(); ++i) {
    if (!result_type.isDynamicDim(i)) continue;
    auto index = b->createOrFold<arith::ConstantIndexOp>(loc, i);
    Value size =
        b->create<tensor::ExtractOp>(loc, op.output_dimensions(), index);
    if (!size.getType().isIndex()) {
      size = b->create<arith::IndexCastOp>(loc, b->getIndexType(), size);
    }
    dynamic_operands.push_back(size);
  }
  auto identity_map_memref =
      MemRefType::get(result_type.getShape(), result_type.getElementType());
  auto copy = b->create<memref::AllocOp>(op.getLoc(), identity_map_memref,
                                         dynamic_operands);
  b->create<memref::CopyOp>(loc, broadcasted, copy);

  return copy;
}

struct DynamicBroadcastInDimOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DynamicBroadcastInDimOpInterface, mhlo::DynamicBroadcastInDimOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_10(mht_10_v, 465, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryRead");

    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_11(mht_11_v, 473, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferizesToMemoryWrite");

    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_12(mht_12_v, 487, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferRelation");

    // The op may allocate.
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_13(mht_13_v, 496, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "bufferize");

    auto broadcast_in_dim_op = cast<mhlo::DynamicBroadcastInDimOp>(op);
    auto result_type =
        broadcast_in_dim_op.getType().dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    // The buffer still has the old (pre-reshape) type.
    FailureOr<Value> operand_buffer = state.getBuffer(
        rewriter, broadcast_in_dim_op->getOpOperand(0) /*operand*/);
    if (failed(operand_buffer)) return failure();

    Value result = InsertDynamicMemrefCastOp(broadcast_in_dim_op,
                                             *operand_buffer, &rewriter);

    // Evaluate `enforce_identity_map_fn` and maybe create a copy.
    Optional<const MhloBufferizationState *> dialect_state =
        state.getAnalysisState().getDialectState<MhloBufferizationState>(
            mhlo::MhloDialect::getDialectNamespace());
    assert(dialect_state.hasValue() && "mhlo dialect state not initialized");
    if ((*dialect_state)->enforce_identity_map_fn(op)) {
      result = CreateCopy(broadcast_in_dim_op, result, &rewriter);
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, result);
    return success();
  }
};

struct HloLegalizeToMemrefPass
    : public HloLegalizeToMemrefPassBase<HloLegalizeToMemrefPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_14(mht_14_v, 529, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "getDependentDialects");

    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    mhlo::MhloDialect, tensor::TensorDialect>();
    registerBufferizableOpInterfaceExternalModels(registry);
  }

 public:
  void runOnOperation() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_15(mht_15_v, 539, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "runOnOperation");

    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    options.allowDialectInFilter<mhlo::MhloDialect>();
    // mhlo dialect state must be explicitly initialized to ease debugging.
    options.addDialectStateInitializer(
        mhlo::MhloDialect::getDialectNamespace(),
        []() { return std::make_unique<MhloBufferizationState>(); });
    if (failed(bufferizeOp(getOperation(), options))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMemrefPass() {
  return std::make_unique<HloLegalizeToMemrefPass>();
}

}  // namespace mhlo
}  // namespace mlir

void mlir::mhlo::registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_memrefDTcc mht_16(mht_16_v, 564, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc", "mlir::mhlo::registerBufferizableOpInterfaceExternalModels");

  registry.addExtension(+[](MLIRContext *ctx,
                            mlir::mhlo::MhloDialect *dialect) {
    mlir::mhlo::ReshapeOp::attachInterface<ReshapeOpInterface>(*ctx);
    mlir::mhlo::DynamicReshapeOp::attachInterface<DynamicReshapeOpInterface>(
        *ctx);
    mlir::mhlo::DynamicBroadcastInDimOp::attachInterface<
        DynamicBroadcastInDimOpInterface>(*ctx);
  });
}
