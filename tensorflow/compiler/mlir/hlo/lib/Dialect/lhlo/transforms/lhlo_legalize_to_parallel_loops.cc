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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace lmhlo {
namespace {

// Clones and adapts the code in `lhlo_block` that works on buffers and has a
// single output buffer to make it compatible with `operands` that have element
// types of the respective buffers. Returns the computed value.
//
// Example. For `operands` with (f32, i32) types and a block with LHLO ops and
// with signature:
//   ^bb(%lhs: memref<f32>, %rhs: memref<i32>, %res: memref<i1>):
//     <LHLO_ops>
//
// inserts necessary alloc and store ops to compute and return result that has
// `i1` type.
Value ApplySingleResultLhloCode(Location loc, ValueRange operands,
                                Block* lhlo_block, OpBuilder* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "ApplySingleResultLhloCode");

  SmallVector<Value, 2> arg_bufs;
  for (auto arg_type : lhlo_block->getArgumentTypes()) {
    arg_bufs.push_back(
        b->create<memref::AllocOp>(loc, arg_type.cast<MemRefType>()));
  }
  for (const auto& operand : llvm::enumerate(operands)) {
    b->create<memref::StoreOp>(loc, operand.value(), arg_bufs[operand.index()]);
  }
  // Clone the ops from `lhlo_block`.
  BlockAndValueMapping mapping;
  mapping.map(lhlo_block->getArguments(), arg_bufs);
  for (auto& nested : lhlo_block->without_terminator()) {
    auto* clone = b->clone(nested, mapping);
    mapping.map(nested.getResults(), clone->getResults());
  }
  return b->create<memref::LoadOp>(loc, arg_bufs.back());
}

// Converts a block with LHLO ops and with signature:
//   ^bb(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
// into a reduction operator of scf.reduce by doing buffer allocation for
// scalar arguments and the result of `scf.reduce` to make it compatible with
// LHLO ops.
void ConvertToReductionOperator(Location loc, scf::ReduceOp reduce_op,
                                Block* lhlo_block, OpBuilder* b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_1(mht_1_v, 243, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "ConvertToReductionOperator");

  Block& loop_reduce_op_body = reduce_op.getReductionOperator().front();
  OpBuilder::InsertionGuard guard(*b);
  b->setInsertionPointToStart(&loop_reduce_op_body);
  b->create<scf::ReduceReturnOp>(
      loc, ApplySingleResultLhloCode(loc, loop_reduce_op_body.getArguments(),
                                     lhlo_block, b));
}

// Returns result of arith::ConstantOp if `dim` is static, otherwise uses DimOp
// to extract dimension at runtime.
Value GetStaticOrDynamicDim(mlir::Location loc, Value shaped_value,
                            size_t dim_index, int64_t dim, OpBuilder* b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "GetStaticOrDynamicDim");

  return dim == ShapedType::kDynamicSize
             ? b->create<memref::DimOp>(loc, shaped_value, dim_index)
                   .getResult()
             : b->create<arith::ConstantIndexOp>(loc, dim);
}

struct MappedIvs {
  // False if the mapped indices are in the padding area, true otherwise.
  Value in_bounds;
  // Mapped indices.
  SmallVector<Value, 2> ivs;
};

template <typename OpTy>
MappedIvs MapWindowIvsToInput(OpTy op, Value operand, ValueRange ivs,
                              ValueRange window_ivs, OpBuilder* b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_3(mht_3_v, 277, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "MapWindowIvsToInput");

  MappedIvs mapped_ivs;

  if (!op.window_strides().hasValue()) {
    op.emitOpError("No window strides specified.");
  }
  auto window_strides = op.window_strides().getValue();

  if (!op.padding().hasValue()) {
    op.emitOpError("No padding specified.");
  }
  auto padding = op.padding().getValue();

  auto loc = op.getLoc();
  auto operand_shape = operand.getType().template cast<MemRefType>().getShape();

  // `in_bounds` is false when the mapped indices are in the padding area.
  mapped_ivs.in_bounds = b->create<mlir::arith::ConstantOp>(
      loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));
  for (unsigned i = 0, e = ivs.size(); i < e; ++i) {
    auto stride = window_strides.template getValues<llvm::APInt>()[i];
    auto pad_low = padding.template getValues<llvm::APInt>()[{i, 0}];

    Value stride_val =
        b->create<arith::ConstantIndexOp>(loc, stride.getSExtValue());
    Value pad_low_val =
        b->create<arith::ConstantIndexOp>(loc, pad_low.getSExtValue());

    Value center = b->create<arith::MulIOp>(loc, ivs[i], stride_val);
    Value offset = b->create<arith::SubIOp>(loc, window_ivs[i], pad_low_val);
    Value index = b->create<arith::AddIOp>(loc, center, offset);
    Value upper_bound =
        GetStaticOrDynamicDim(loc, operand, i, operand_shape[i], b);
    // We must check whether 0 <= index_i < shape_i, as otherwise we are in
    // the pad and then we have to use the neutral element for reduction.
    // Equivalently, it can be computed as the unsigned comparison index_i <
    // shape_i, since a negative value wraps to a large positive value.
    mapped_ivs.in_bounds = b->create<mlir::arith::AndIOp>(
        loc, mapped_ivs.in_bounds,
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, index,
                                 upper_bound));
    mapped_ivs.ivs.push_back(index);
  }
  return mapped_ivs;
}

// Returns scf::Parallel over a shaped value with static or dynamic shape.
scf::ParallelOp MakeLoopOverShape(Location loc, Value shaped_value,
                                  OpBuilder* b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_4(mht_4_v, 328, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "MakeLoopOverShape");

  Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
  Value one = b->create<arith::ConstantIndexOp>(loc, 1);

  ArrayRef<int64_t> shape =
      shaped_value.getType().cast<ShapedType>().getShape();
  SmallVector<Value, 2> lower, upper, step;
  for (const auto& dim : llvm::enumerate(shape)) {
    upper.push_back(
        GetStaticOrDynamicDim(loc, shaped_value, dim.index(), dim.value(), b));
    lower.push_back(zero);
    step.push_back(one);
  }
  return b->create<scf::ParallelOp>(loc, lower, upper, step);
}

// Converts `lmhlo.ReduceOp` into two scf::ParallelOp and a scf::ReduceOp.
// The outper `ParallelOp` refers to the parallel loops if there are
// any. The inner `ParalleOp` refers to the reduction loops and `ReduceOp`
// contains the reduction operator.
//
// Example:
//
//  "lmhlo.reduce"(%buffer, %init_buf, %result) ({
//    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
//      <LHLO ops>
//    } ) {dimensions = dense<[1]> : tensor<1xi64>}
//      : (memref<100x10x5xf32>, memref<f32>, memref<100x5xf32>) -> ()
//
//  is roughly converted into:
//
//  %init = load %init_buf[] : memref<f32>
//  scf.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
//    %result = scf.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
//      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
//      scf.reduce(%elem_to_reduce)  {
//        ^bb0(%elem: f32, %acc: f32):
//          elem_buf = alloc() : memref<f32>
//          store %elem, elem_buf[] : memref<f32>
//          acc_buf = alloc() : memref<f32>
//          store %acc, acc_buf[] : memref<f32>
//          <LHLO_ops>
//          %acc_result = load acc_buf[] : memref<f32>
//          scf.reduce.return %acc_result : f32
//      } : f32
//      scf.yield
//    } : f32
//    scf.yield
//  }
class ReduceOpConverter : public OpConversionPattern<lmhlo::ReduceOp> {
 public:
  using OpConversionPattern<lmhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ReduceOp reduce_op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_5(mht_5_v, 386, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "matchAndRewrite");

    // TODO(b/183977252) : Handle variadic ReduceOp/ReduceWindowOp
    if (reduce_op.out().size() != 1) return failure();

    scf::ReduceOp scf_reduce_op =
        CreateReduceOpInNestedParallelLoops(reduce_op, &rewriter);
    ConvertToReductionOperator(reduce_op.getLoc(), scf_reduce_op,
                               &reduce_op.body().front(), &rewriter);
    rewriter.replaceOp(reduce_op, llvm::None);
    return success();
  }

 private:
  // Creates nested `scf.parallel` ops with `scf.reduce`. The outer ParallelOp
  // refers to the parallel dimensions of `reduce_op` if any and the inner
  // ParallelOp refers to the reduction dimensions. The scf.reduce op is
  // returned.
  //
  // If the reduction argument is a memref<100x10x5xf32> and the
  // reduction is performed along dimension 1 then this method will generate
  //
  //  %init = load %init_buf[] : memref<f32>
  //  scf.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
  //    %result = scf.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
  //      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
  //      scf.reduce(%elem_to_reduce)  {
  //        <THE BLOCK PTR TO BE RETURNED>
  //      } : f32
  //      scf.yield
  //    } : f32
  //    scf.yield
  //  }
  scf::ReduceOp CreateReduceOpInNestedParallelLoops(
      lmhlo::ReduceOp reduce_op, ConversionPatternRewriter* rewriter) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_6(mht_6_v, 422, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "CreateReduceOpInNestedParallelLoops");

    auto loc = reduce_op.getLoc();
    DenseSet<int> reducing_dims;
    for (const auto& rdim : reduce_op.dimensions().getValues<APInt>()) {
      reducing_dims.insert(rdim.getSExtValue());
    }

    Value operand = reduce_op.inputs().front();
    Value out = reduce_op.out().front();
    SmallVector<Value, 2> parallel_lower, parallel_upper, parallel_step;
    SmallVector<Value, 2> reduce_lower, reduce_upper, reduce_step;
    auto operand_shape = operand.getType().cast<MemRefType>().getShape();
    for (const auto& dim : llvm::enumerate(operand_shape)) {
      const bool is_reducing_dim = reducing_dims.count(dim.index());

      Value ub = GetStaticOrDynamicDim(loc, operand, dim.index(), dim.value(),
                                       rewriter);
      Value lb = rewriter->create<arith::ConstantIndexOp>(loc, 0);
      Value step = rewriter->create<arith::ConstantIndexOp>(loc, 1);
      (is_reducing_dim ? reduce_lower : parallel_lower).push_back(lb);
      (is_reducing_dim ? reduce_upper : parallel_upper).push_back(ub);
      (is_reducing_dim ? reduce_step : parallel_step).push_back(step);
    }
    // Load initial value from memref<element_type>.
    SmallVector<Value, 1> init_value = {rewriter->create<memref::LoadOp>(
        loc, *reduce_op.init_values().begin())};
    // Outer ParallelOp is not needed if it is a reduction across all dims.
    scf::ParallelOp outer;
    if (!parallel_lower.empty()) {
      outer = rewriter->create<scf::ParallelOp>(loc, parallel_lower,
                                                parallel_upper, parallel_step);
      rewriter->setInsertionPointToStart(outer.getBody());
    }
    scf::ParallelOp inner = rewriter->create<scf::ParallelOp>(
        loc, reduce_lower, reduce_upper, reduce_step, ValueRange(init_value));
    Value reduction_result = *inner.getResults().begin();

    SmallVector<Value, 1> out_indices;
    if (outer != nullptr) {
      out_indices.reserve(outer.getNumLoops());
      for (Value iv : outer.getInductionVars()) {
        out_indices.push_back(iv);
      }
    } else {
      out_indices.push_back(rewriter->create<arith::ConstantIndexOp>(loc, 0));
    }

    rewriter->create<memref::StoreOp>(loc, reduction_result, out, out_indices);

    // Load the element to reduce.
    SmallVector<Value, 2> indices;
    indices.reserve(operand_shape.size());

    if (outer) {
      auto inner_ivs_it = inner.getInductionVars().begin();
      auto outer_ivs_it = outer.getInductionVars().begin();
      for (unsigned i = 0, e = operand_shape.size(); i < e; ++i) {
        indices.push_back(reducing_dims.count(i) ? *inner_ivs_it++
                                                 : *outer_ivs_it++);
      }
    } else {
      indices = inner.getInductionVars();
    }

    rewriter->setInsertionPointToStart(inner.getBody());
    Value elem = rewriter->create<mlir::memref::LoadOp>(
        loc, reduce_op.inputs().front(), indices);
    return rewriter->create<scf::ReduceOp>(loc, elem);
  }
};

// Pseudocode:
// for each index O in output
//   accumulator = neutral_value
//   in_bounds = true
//   for each index W in window
//     for each dimension i from 0 to rank - 1
//       index = O[i] * stride[i] + W[i] - pad_low[i]
//       in_bounds = inbounds && (index `ult` shape[i])
//       I[i] = index
//     if (in_bounds)
//       value = input[I]
//     else
//       value = neutral_value
//     accumulator = reduction_operator(accumulator, value)
//   output[O] = accumulator
//
// Converts `lmhlo.ReduceWindowOp` into two scf::ParallelOp and a
// scf::ReduceOp.
// The outper `ParallelOp` refers to the parallel loops that traverese output
// buffer. The inner `ParalleOp` refers to the reduction loops that traverse
// reduction windows and `ReduceOp` contains the reduction operator.
//
// Example:
//
// func @reduce_window(%arg: memref<112x112xf32>,
//              %init: memref<f32>,
//              %result: memref<56x56xf32>) {
//   "lmhlo.reduce_window"(%arg, %init, %result) ({
//     ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
//       "lmhlo.maximum"(%lhs, %rhs, %res)
//         : (memref<f32>, memref<f32>, memref<f32>) -> ()
//       "lmhlo.terminator"() : () -> ()
//     }) {
//       padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
//       window_dimensions = dense<[3, 3]> : tensor<2xi64>,
//       window_strides = dense<[2, 2]> : tensor<2xi64>
//     } : (memref<112x112xf32>, memref<f32>, memref<56x56xf32>) -> ()
//   return
// }
//
// is roughly converted into:
//
//    %neutral_elem = load %init_buf[] : memref<f32>
//    scf.parallel (%i, %j) = (%c0, %c0) to (%c56, %c56) step (%c1, %c1) {
//      %result = scf.parallel (%iw, %jw) = (%c0, %c0)
//                  to (%c3, %c3) step (%c1, %c1) neutral_elem (%0) -> f32 {
//        %in_bounds = <COMPUTE IF INDEX IS IN OPERAND'S pad>
//        %elem = load %operand[%computed_i, %computed_j]
//        %elem_or_neutral = select %in_bounds, %elem, %neutral_elem : f32
//        scf.reduce(%elem_to_reduce)  : f32 {
//          ^bb0(%arg7: f32, %arg8: f32):
//            <LHLO ops>
//        }
//        scf.yield
//      }
//      store %result, %output_buffer[%i, %j] : memref<56x56xf32>
//      scf.yield
//    }
//    return
//  }
class ReduceWindowOpConverter
    : public OpConversionPattern<lmhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern<lmhlo::ReduceWindowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ReduceWindowOp reduce_window_op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_7(mht_7_v, 563, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "matchAndRewrite");

    // TODO(b/183977252) : Handle variadic ReduceOp/ReduceWindowOp
    if (reduce_window_op.out().size() != 1) return failure();

    scf::ParallelOp output_loop, window_loop;
    std::tie(output_loop, window_loop) =
        CreateParallelLoopsToTraverseOutputAndWindow(reduce_window_op,
                                                     &rewriter);

    scf::ReduceOp reduce_op = CreateReduceOpInNestedParallelLoops(
        reduce_window_op, output_loop, window_loop, &rewriter);

    ConvertToReductionOperator(reduce_window_op.getLoc(), reduce_op,
                               &reduce_window_op.body().front(), &rewriter);
    rewriter.replaceOp(reduce_window_op, llvm::None);
    return success();
  }

 private:
  std::pair<scf::ParallelOp, scf::ParallelOp>
  CreateParallelLoopsToTraverseOutputAndWindow(
      lmhlo::ReduceWindowOp reduce_window_op,
      ConversionPatternRewriter* rewriter) const {
    auto loc = reduce_window_op.getLoc();
    Value init_value = rewriter->create<memref::LoadOp>(
        loc, reduce_window_op.init_values()[0]);

    Value zero = rewriter->create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter->create<arith::ConstantIndexOp>(loc, 1);

    // Create an outer parallel loop that spans the output of ReduceWindowOp.
    Value output = reduce_window_op.out()[0];
    auto output_loop = MakeLoopOverShape(loc, output, rewriter);

    // Create a nested loop that traverses the window.
    SmallVector<Value, 2> window_lower, window_upper, window_step;
    rewriter->setInsertionPointToStart(output_loop.getBody());
    for (const auto& window_dim : reduce_window_op.window_dimensions()) {
      window_step.push_back(one);
      window_lower.push_back(zero);
      window_upper.push_back(rewriter->create<arith::ConstantIndexOp>(
          loc, window_dim.getSExtValue()));
    }
    auto window_loop = rewriter->create<scf::ParallelOp>(
        loc, window_lower, window_upper, window_step, ValueRange(init_value));

    Value reduction_result = *window_loop.getResults().begin();
    auto output_ivs = output_loop.getInductionVars();
    rewriter->create<memref::StoreOp>(loc, reduction_result, output,
                                      output_ivs);
    return std::make_pair(output_loop, window_loop);
  }

  scf::ReduceOp CreateReduceOpInNestedParallelLoops(
      lmhlo::ReduceWindowOp reduce_window_op, scf::ParallelOp output_loop,
      scf::ParallelOp window_loop, ConversionPatternRewriter* rewriter) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_8(mht_8_v, 621, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "CreateReduceOpInNestedParallelLoops");

    rewriter->setInsertionPointToStart(window_loop.getBody());
    auto loc = reduce_window_op.getLoc();

    if (reduce_window_op.base_dilations().hasValue() ||
        reduce_window_op.window_dilations().hasValue()) {
      reduce_window_op.emitRemark(
          "Lowering to parallel loops does not support `base_dilations` or "
          "`window_dilations` attributes yet. The attributes will be ignored.");
    }

    Value input = reduce_window_op.inputs()[0];
    auto input_type = input.getType().cast<MemRefType>();

    // Compute ivs in 'arg' buffer and whether these ivs are in pad area or not.
    MappedIvs mapped_ivs = MapWindowIvsToInput(
        reduce_window_op, input, output_loop.getInductionVars(),
        window_loop.getInductionVars(), rewriter);

    auto elem_or_init = rewriter->create<scf::IfOp>(
        loc, input_type.getElementType(), mapped_ivs.in_bounds,
        /*withElseRegion=*/true);

    OpBuilder then_builder =
        elem_or_init.getThenBodyBuilder(rewriter->getListener());
    Value elem =
        then_builder.create<mlir::memref::LoadOp>(loc, input, mapped_ivs.ivs);
    then_builder.create<scf::YieldOp>(loc, elem);

    OpBuilder else_builder =
        elem_or_init.getElseBodyBuilder(rewriter->getListener());
    else_builder.create<scf::YieldOp>(loc, *window_loop.getInitVals().begin());

    return rewriter->create<scf::ReduceOp>(loc,
                                           *elem_or_init.getResults().begin());
  }
};

// See the operation semantics in
// https://www.tensorflow.org/xla/operation_semantics#selectandscatter
//
// Pseudocode:
//  scf.parallel(coordinates O in the output):
//    output[O] = init
//  scf.parallel(coordinates S in the source):
//    selected_ivs = 0
//    selected_val = 0
//    initialized_flag = false
//    scf.for (first dim W_1 in the window)
//         iter_args (selected_ivs, selected_val, initialized_flag):
//    ...
//      scf.for (last dim W_N in the window):
//           iter_args (selected_ivs, selected_val, initialized_flag):
//        I = S * stride + W - pad_low
//        if I within bounds of operand:
//          if (initialized_flag):
//            pred = select(selected_value, operand(I))):
//            if (pred)
//              selected_value = operand(I)
//              selected_index = I
//          else
//              selected_value = operand(I)
//              selected_index = I
//              initialized_flag = true
//    output(selected_index) = scatter(output(selected_index), source(S))
class SelectAndScatterOpConverter
    : public OpConversionPattern<lmhlo::SelectAndScatterOp> {
 public:
  using OpConversionPattern<lmhlo::SelectAndScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::SelectAndScatterOp s_and_s_op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_9(mht_9_v, 696, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "matchAndRewrite");

    auto loc = s_and_s_op.getLoc();
    InitializeOutput(s_and_s_op, &rewriter);
    scf::ParallelOp loop_over_src =
        MakeLoopOverShape(loc, s_and_s_op.source(), &rewriter);
    rewriter.setInsertionPointToStart(loop_over_src.getBody());

    // Compute indices of the selected element in the window.
    auto selected_ivs = SelectIvs(s_and_s_op, loop_over_src, &rewriter);

    // Load `source[selected_ivs]`.
    auto src_elem = rewriter.create<memref::LoadOp>(
        loc, s_and_s_op.source(), loop_over_src.getInductionVars());

    // Compute `out[selected_ivs]` = scatter(out[selected_ivs], src_element)`.
    auto rmw = rewriter.create<memref::GenericAtomicRMWOp>(
        loc, s_and_s_op.out(), selected_ivs);
    OpBuilder rmw_builder = OpBuilder::atBlockEnd(rmw.getBody());
    auto acc_result =
        ApplySingleResultLhloCode(loc, {src_elem, rmw.getCurrentValue()},
                                  &s_and_s_op.scatter().front(), &rmw_builder);
    rmw_builder.create<memref::AtomicYieldOp>(loc, acc_result);

    rewriter.replaceOp(s_and_s_op, llvm::None);
    return success();
  }

 private:
  void InitializeOutput(lmhlo::SelectAndScatterOp s_and_s_op,
                        OpBuilder* b) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_10(mht_10_v, 728, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "InitializeOutput");

    auto loc = s_and_s_op.getLoc();
    Value init_value = b->create<memref::LoadOp>(loc, s_and_s_op.init_value());

    scf::ParallelOp loop_over_output =
        MakeLoopOverShape(loc, s_and_s_op.out(), b);
    OpBuilder::InsertionGuard guard(*b);
    b->setInsertionPointToStart(loop_over_output.getBody());
    b->create<memref::StoreOp>(loc, init_value, s_and_s_op.out(),
                               loop_over_output.getInductionVars());
  }

  struct WindowLoops {
    SmallVector<Value, 2> selected_ivs;
    SmallVector<Value, 2> window_ivs;
    scf::ForOp inner_loop;
  };
  WindowLoops InsertWindowLoops(lmhlo::SelectAndScatterOp s_and_s_op,
                                scf::ParallelOp loop_over_src,
                                OpBuilder* b) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_11(mht_11_v, 750, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "InsertWindowLoops");

    auto loc = s_and_s_op.getLoc();
    Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
    Value one = b->create<arith::ConstantIndexOp>(loc, 1);

    auto element_type =
        s_and_s_op.out().getType().cast<MemRefType>().getElementType();
    auto rank = loop_over_src.getNumLoops();

    // `iter_args` = [iv_1, ..., iv_N, selected_value, is_initialized]
    SmallVector<Value, 4> iter_args(rank, zero);
    iter_args.push_back(b->create<mlir::arith::ConstantOp>(
        loc, element_type, b->getFloatAttr(element_type, 0)));
    iter_args.push_back(b->create<mlir::arith::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 0)));

    // Create a nested loop that traverses the window.
    OpBuilder::InsertPoint ip;
    WindowLoops result;
    for (const auto& window_dim :
         s_and_s_op.window_dimensions()->getValues<APInt>()) {
      Value upper =
          b->create<arith::ConstantIndexOp>(loc, window_dim.getSExtValue());
      result.inner_loop =
          b->create<scf::ForOp>(loc, zero, upper, one, iter_args);
      if (b->getInsertionBlock() == loop_over_src.getBody()) {
        ip = b->saveInsertionPoint();
        result.selected_ivs = result.inner_loop.getResults().take_front(rank);
      } else {
        b->create<scf::YieldOp>(loc, result.inner_loop.getResults());
      }
      b->setInsertionPointToStart(result.inner_loop.getBody());
      iter_args = ValueRange{result.inner_loop.getRegionIterArgs()};
      result.window_ivs.push_back(result.inner_loop.getInductionVar());
    }
    b->restoreInsertionPoint(ip);
    return result;
  }

  // Adapter to store iteration arguments of sequential loops that perform
  // select in a window.
  class IterArgs {
   public:
    explicit IterArgs(ValueRange ivs_val_flag) : ivs_val_flag_(ivs_val_flag) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_12(mht_12_v, 796, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "IterArgs");
}
    IterArgs(ValueRange ivs, Value value, Value flag) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_13(mht_13_v, 800, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "IterArgs");

      ivs_val_flag_ = ivs;
      ivs_val_flag_.push_back(value);
      ivs_val_flag_.push_back(flag);
    }

    ArrayRef<Value> to_vector() const { return ivs_val_flag_; }

    // Indices of the currently selected value.
    ArrayRef<Value> ivs() const { return to_vector().drop_back(2); }
    // Currently selected value w.r.t. select() function.
    Value value() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_14(mht_14_v, 814, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "value");
 return ivs_val_flag_.end()[-2]; }
    // i1 flag if value() and ivs() were initialized.
    Value is_init() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_15(mht_15_v, 819, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "is_init");
 return ivs_val_flag_.back(); }

   private:
    // Vector that stores iv_1, ..., iv_N, value, init.
    SmallVector<Value, 4> ivs_val_flag_;
  };

  SmallVector<Value, 2> SelectIvs(lmhlo::SelectAndScatterOp s_and_s_op,
                                  scf::ParallelOp loop_over_src,
                                  OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();

    WindowLoops window_loops = InsertWindowLoops(s_and_s_op, loop_over_src, b);
    auto inner_loop_b =
        OpBuilder::atBlockEnd(window_loops.inner_loop.getBody());

    // Compute ivs in 'arg' buffer and whether these ivs are in the pad area.
    MappedIvs mapped_ivs = MapWindowIvsToInput(
        s_and_s_op, s_and_s_op.operand(), loop_over_src.getInductionVars(),
        window_loops.window_ivs, &inner_loop_b);

    IterArgs ivs_val_flag(window_loops.inner_loop.getRegionIterArgs());

    auto if_in_bounds = inner_loop_b.create<scf::IfOp>(
        loc, window_loops.inner_loop.getResultTypes(), mapped_ivs.in_bounds,
        /*withElseRegion=*/true);

    // Case when we are inside boundaries of 'arg' and not in the pad area.
    {
      OpBuilder in_bounds_then_b =
          if_in_bounds.getThenBodyBuilder(b->getListener());
      auto select_or_init_results = SelectOrInitialize(
          s_and_s_op, mapped_ivs.ivs, &ivs_val_flag, &in_bounds_then_b);
      in_bounds_then_b.create<scf::YieldOp>(loc, select_or_init_results);
    }

    // Case when we are in the pad.
    {
      OpBuilder in_bounds_else_b =
          if_in_bounds.getElseBodyBuilder(b->getListener());
      in_bounds_else_b.create<scf::YieldOp>(loc, ivs_val_flag.to_vector());
    }

    inner_loop_b.create<scf::YieldOp>(loc, if_in_bounds.getResults());
    return window_loops.selected_ivs;
  }

  SmallVector<Value, 4> SelectOrInitialize(lmhlo::SelectAndScatterOp s_and_s_op,
                                           ArrayRef<Value> operand_ivs,
                                           IterArgs* ivs_val_flag,
                                           OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();
    Value true_i1 = b->create<mlir::arith::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));

    TypeRange iter_arg_types{ivs_val_flag->to_vector()};
    Value operand_elem =
        b->create<memref::LoadOp>(loc, s_and_s_op.operand(), operand_ivs);
    auto if_init =
        b->create<scf::IfOp>(loc, iter_arg_types, ivs_val_flag->is_init(),
                             /*withElseRegion=*/true);
    // Init == true, i.e. iter args are already initialized with a selected
    // element in boundaries of the operand. Select function has to be computed
    // here.
    {
      OpBuilder if_init_then_b = if_init.getThenBodyBuilder(b->getListener());

      auto& lhlo_select = s_and_s_op.select().front();
      Value pred =
          ApplySingleResultLhloCode(loc, {operand_elem, ivs_val_flag->value()},
                                    &lhlo_select, &if_init_then_b);

      auto if_pred = if_init_then_b.create<scf::IfOp>(loc, iter_arg_types, pred,
                                                      /*withElseRegion=*/true);

      // Pred == true, therefore pack newly selected ivs, val and init flag back
      // to iter_args and return.
      {
        OpBuilder if_pred_then_b = if_pred.getThenBodyBuilder(b->getListener());
        if_pred_then_b.create<scf::YieldOp>(
            loc, IterArgs{operand_ivs, operand_elem, true_i1}.to_vector());
      }

      // Pred == false, therefore return old iter_args.
      {
        OpBuilder if_pred_else_b = if_pred.getElseBodyBuilder(b->getListener());
        if_pred_else_b.create<scf::YieldOp>(loc, ivs_val_flag->to_vector());
      }

      if_init_then_b.create<scf::YieldOp>(loc, if_pred.getResults());
    }
    // Init == false, i.e. only pad was visited before and this is the first
    // element in the boundaries of the operand.
    {
      OpBuilder if_init_else_b = if_init.getElseBodyBuilder(b->getListener());

      if_init_else_b.create<scf::YieldOp>(
          loc, IterArgs{operand_ivs, operand_elem, true_i1}.to_vector());
    }
    return if_init.getResults();
  }
};

struct LhloLegalizeToParallelLoopsPass
    : public LhloLegalizeToParallelLoopsPassBase<
          LhloLegalizeToParallelLoopsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_16(mht_16_v, 928, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "getDependentDialects");

    registry.insert<arith::ArithmeticDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_parallel_loopsDTcc mht_17(mht_17_v, 936, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc", "runOnOperation");

    auto func = getOperation();

    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<
        ReduceOpConverter,
        ReduceWindowOpConverter,
        SelectAndScatterOpConverter
      >(func.getContext());
    // clang-format on

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           scf::SCFDialect, LmhloDialect>();
    target.addIllegalOp<lmhlo::ReduceOp, lmhlo::ReduceWindowOp,
                        lmhlo::SelectAndScatterOp>();

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToParallelLoopsPass() {
  return std::make_unique<LhloLegalizeToParallelLoopsPass>();
}

}  // namespace lmhlo
}  // namespace mlir
