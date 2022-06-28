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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc() {
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

// This file implements logic for translating mixed IR to buffer form.

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project

#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

struct BufferizeConstantOp : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "matchAndRewrite");

    // We only need to bufferize tensor constants.
    Location loc = op.getLoc();
    auto result_type = op.getType().dyn_cast<RankedTensorType>();
    int64_t result_rank = result_type.getRank();
    if (!result_type || !result_type.hasStaticShape() || result_rank > 1)
      return failure();

    auto element_type = result_type.getElementType();
    auto memref_type = MemRefType::get(result_type.getShape(), element_type);
    auto elements_attr = op.getValue().cast<DenseElementsAttr>();

    // arith.constant doesn't handle scalar complex types.
    // TODO(kramerb): Should this use materializeConstant instead?
    auto make_constant = [&](Attribute attr, Type type) -> Value {
      if (complex::ConstantOp::isBuildableWith(attr, type))
        return rewriter.create<complex::ConstantOp>(loc, type,
                                                    attr.cast<ArrayAttr>());
      return rewriter.create<arith::ConstantOp>(loc, attr);
    };

    if (result_rank == 0) {
      Value buffer = rewriter.create<memref::AllocOp>(loc, memref_type);
      Value constant =
          make_constant(elements_attr.getValues<Attribute>()[0], element_type);
      rewriter.create<memref::StoreOp>(loc, constant, buffer);
      rewriter.replaceOp(op, {buffer});
      return success();
    }

    Value buffer = rewriter.create<memref::AllocaOp>(loc, memref_type);

    bool all_same_elems = elements_attr.isSplat();
    Value value;
    if (all_same_elems)
      value = make_constant(elements_attr.getSplatValue<mlir::Attribute>(),
                            element_type);
    for (auto &en : llvm::enumerate(elements_attr.getValues<Attribute>())) {
      if (!all_same_elems) value = make_constant(en.value(), element_type);
      Value index = rewriter.create<arith::ConstantIndexOp>(loc, en.index());
      rewriter.create<memref::StoreOp>(loc, value, buffer, index);
    }
    rewriter.replaceOp(op, {buffer});
    return success();
  }
};

struct BufferizeAndConvertMinimumBroadcastShapesOp
    : public OpConversionPattern<chlo::MinimumBroadcastShapesOp> {
  using OpConversionPattern<
      chlo::MinimumBroadcastShapesOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      chlo::MinimumBroadcastShapesOp broadcast_shapes_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_1(mht_1_v, 269, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "matchAndRewrite");

    auto loc = broadcast_shapes_op.getLoc();
    ImplicitLocOpBuilder lb(loc, rewriter);
    Value zero = lb.create<arith::ConstantIndexOp>(0);
    SmallVector<Value> shapes = adaptor.shapes();
    size_t k = shapes.size();
    SmallVector<Value> ranks;
    ranks.reserve(k);

    // Determine the maximum rank of the operands.
    Value max_rank;
    for (size_t i = 0; i < k; ++i) {
      Value rank = lb.create<memref::DimOp>(loc, shapes[i], zero);
      ranks.push_back(rank);
      if (i) {
        Value rank_is_greater = lb.create<arith::CmpIOp>(
            arith::CmpIPredicate::ugt, ranks[i], max_rank);
        max_rank =
            lb.create<arith::SelectOp>(rank_is_greater, ranks[i], max_rank);
      } else {
        max_rank = ranks[0];
      }
    }

    // Allocate buffers for the return values and initialize them with 1's.
    SmallVector<Value> result_shapes;
    result_shapes.reserve(k);
    auto result_type =
        MemRefType::get({ShapedType::kDynamicSize}, lb.getIndexType());
    Value one = lb.create<arith::ConstantIndexOp>(1);
    for (size_t i = 0; i < k; ++i) {
      // We assume the buffer will be small, so we allocate it on the stack.
      // TODO(b/181654096): Replace AllocaOp with AllocOp.
      auto result = lb.create<memref::AllocaOp>(result_type, ranks[i]);
      lb.create<scf::ForOp>(zero, ranks[i], one, llvm::None,
                            [&one, &result](OpBuilder &b, Location l, Value idx,
                                            ValueRange /*vr*/) {
                              b.create<memref::StoreOp>(l, one, result, idx);
                              b.create<scf::YieldOp>(l, llvm::None);
                            });
      result_shapes.push_back(result);
    }

    // Iterate through the dimensions and determine which adjacent dimensions
    // can be combined. Keep a running product of the dimensions that can be
    // combined as iteration variable (initialized to 1), and the current
    // dimension offset in the result shapes. We iterate through the shapes
    // backward, because the broadcasting semantics mean that the last
    // dimensions of each shape (the least significant ones) are matched
    // together.
    Value two = lb.create<arith::ConstantIndexOp>(2);
    Value max_rank_plus_two = lb.create<arith::AddIOp>(loc, max_rank, two);
    Value constant_false =
        lb.create<arith::ConstantOp>(lb.getI1Type(), lb.getBoolAttr(false));
    SmallVector<Value> init_values;
    init_values.reserve(k + 3);
    // Initially, all values are marked as not broadcasted.
    for (int i = 0; i < k; ++i) {
      init_values.push_back(constant_false);
    }
    // The running product is initially 1.
    init_values.push_back(one);
    // The current dimension offset is initially 0.
    init_values.push_back(zero);
    // Whether the broadcasting is invalid.
    init_values.push_back(constant_false);

    // Iterate from 1 to max_rank + 1 (inclusive). This iteration variable is
    // used as an offset from the end of each shape vector. We iterate until
    // max_rank + 1 to handle the case that we have a running_product > 1 left
    // when we have processed all dimensions of the largest shape.
    auto main_loop = lb.create<scf::ForOp>(
        one, max_rank_plus_two, one, init_values,
        [&](OpBuilder &b, Location l, Value v, ValueRange vr) {
          // 'same_size' should track what the size of the dimension is to which
          // the 1-sized dimensions are broadcasted. If all of the dimensions
          // are 1, it will stay 1.
          Value same_size = one;
          // 'result_dimensions' stores the current dimension with an offset of
          // 'leading_ones' to make it easier to check whether we are in-bounds
          // with respect to the "real" shape with leading 1's removed.
          SmallVector<Value> result_dimensions;
          result_dimensions.reserve(k);
          // 'no_broadcasting' stores boolean flags that encode whether the
          // corresponding shape does not need broadcasting at the current
          // position.
          SmallVector<Value> no_broadcasting;
          no_broadcasting.reserve(k + 3);
          // The first k loop carried values are the previous broadcasting
          // state.
          auto prev_no_broadcasting = vr.take_front(k);

          // This loop checks which shapes need broadcasting at the current
          // dimension. A shape needs broadcasting if it is indexed out of
          // bounds, or its current dimension size is 1.
          Value current_dimension_has_invalid_broadcast = constant_false;
          for (size_t i = 0; i < k; ++i) {
            // Determine the size of the current dimension. If the dimension is
            // out of bounds, we choose the value 'one'.
            Value is_out_of_bounds = b.create<arith::CmpIOp>(
                l, arith::CmpIPredicate::ult, ranks[i], v);
            Value dimension = b.create<arith::SubIOp>(l, ranks[i], v);
            result_dimensions.push_back(dimension);
            Value current_size =
                b.create<scf::IfOp>(
                     l, TypeRange{b.getIndexType()}, is_out_of_bounds,
                     [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_2(mht_2_v, 378, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                       b.create<scf::YieldOp>(l, one);
                     },
                     [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_3(mht_3_v, 384, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                       // Using IfOp instead of SelectOp makes sure that we
                       // don't try to load if the dimension is out of bounds.
                       Value size =
                           b.create<memref::LoadOp>(l, shapes[i], dimension);
                       b.create<scf::YieldOp>(l, size);
                     })
                    .getResult(0);
            // Compute whether the current dimension does require broadcasting.
            Value current_size_is_not_one = b.create<arith::CmpIOp>(
                l, arith::CmpIPredicate::ne, current_size, one);
            no_broadcasting.push_back(current_size_is_not_one);
            Value new_same_size = b.create<arith::SelectOp>(
                l, current_size_is_not_one, current_size, same_size);
            Value same_size_was_not_one = b.create<arith::CmpIOp>(
                l, arith::CmpIPredicate::ne, same_size, one);
            Value is_different_size = b.create<arith::CmpIOp>(
                l, arith::CmpIPredicate::ne, same_size, new_same_size);
            // The broadcast is invalid if the size of the current dimension
            // is not equal to the expected size, unless the expected size was
            // still the initial value 1.
            Value is_invalid = b.create<arith::AndIOp>(l, same_size_was_not_one,
                                                       is_different_size);
            current_dimension_has_invalid_broadcast = b.create<arith::OrIOp>(
                l, current_dimension_has_invalid_broadcast, is_invalid);
            same_size = new_same_size;
          }

          // Check whether we have at least one shape that has a different
          // status regarding whether it needs broadcasting at the current
          // dimension versus whether it needs broadcasting at the previous
          // dimension.
          Value same_size_is_one = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::eq, same_size, one);
          Value different_broadcasting_set = constant_false;
          for (size_t i = 0; i < k; ++i) {
            // If all dimensions are 1, we preserve the status whether a shape
            // needs broadcasting or not, because in that case the dimension can
            // just be ignored.
            no_broadcasting[i] = b.create<arith::SelectOp>(
                l, same_size_is_one, prev_no_broadcasting[i],
                no_broadcasting[i]);
            // Compare whether the current shape changes its status regarding
            // whether it needs broadcasting at the current dimension.
            Value broadcasting_is_different = b.create<arith::CmpIOp>(
                l, arith::CmpIPredicate::ne, prev_no_broadcasting[i],
                no_broadcasting[i]);
            different_broadcasting_set = b.create<arith::OrIOp>(
                l, different_broadcasting_set, broadcasting_is_different);
          }
          Value running_product = vr[k];
          Value current_dimension_offset = vr[k + 1];

          // We need to stop combining dimensions if the set of shapes which
          // need broadcasting at the current dimension changes compared to the
          // set of shapes needing broadcasting at the previous dimension.
          Value is_last_iteration = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::sgt, v, max_rank);
          Value stop_combining_dimensions = b.create<arith::OrIOp>(
              l, is_last_iteration, different_broadcasting_set);
          auto if_stop_combining_dimensions = b.create<scf::IfOp>(
              l, TypeRange{b.getIndexType(), b.getIndexType()},
              stop_combining_dimensions,
              [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_4(mht_4_v, 450, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                // If the running product is not 1, add one dimension of size
                // 'running_product' to each shape that didn't need
                // broadcasting, otherwise add a 1 dimension if it was
                // previously indexed in-bounds.
                Value running_product_not_one = b.create<arith::CmpIOp>(
                    l, arith::CmpIPredicate::ne, running_product, one);
                Value new_dimension_offset =
                    b.create<scf::IfOp>(
                         l, TypeRange{b.getIndexType()},
                         running_product_not_one,
                         [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_5(mht_5_v, 464, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                           Value new_dimension_offset = b.create<arith::AddIOp>(
                               l, current_dimension_offset, one);
                           Value minus_one =
                               lb.create<arith::ConstantIndexOp>(-1);
                           for (size_t i = 0; i < k; ++i) {
                             Value was_in_bounds = b.create<arith::CmpIOp>(
                                 l, arith::CmpIPredicate::sge,
                                 result_dimensions[i], minus_one);
                             Value should_store_dimension =
                                 b.create<arith::OrIOp>(
                                     l, was_in_bounds, prev_no_broadcasting[i]);
                             b.create<scf::IfOp>(
                                 l, should_store_dimension,
                                 [&](OpBuilder &b, Location l) {
                                   Value output_dimension =
                                       b.create<arith::SubIOp>(
                                           l, ranks[i], new_dimension_offset);
                                   // If the shape needed broadcasting at the
                                   // previous dimension, we set the output size
                                   // to 1, otherwise to 'running_product'.
                                   Value output_size =
                                       b.create<arith::SelectOp>(
                                           l, prev_no_broadcasting[i],
                                           running_product, one);
                                   b.create<memref::StoreOp>(l, output_size,
                                                             result_shapes[i],
                                                             output_dimension);
                                   b.create<scf::YieldOp>(l, llvm::None);
                                 });
                           }
                           b.create<scf::YieldOp>(l, new_dimension_offset);
                         },
                         [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_6(mht_6_v, 500, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                           b.create<scf::YieldOp>(l, current_dimension_offset);
                         })
                        .getResult(0);
                b.create<scf::YieldOp>(
                    l, ValueRange{same_size, new_dimension_offset});
              },
              [&](OpBuilder &b, Location l) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_7(mht_7_v, 510, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

                Value new_running_product =
                    b.create<arith::MulIOp>(l, running_product, same_size);
                b.create<scf::YieldOp>(l, ValueRange{new_running_product,
                                                     current_dimension_offset});
              });
          // Add the remaining results.
          no_broadcasting.push_back(if_stop_combining_dimensions.getResult(0));
          no_broadcasting.push_back(if_stop_combining_dimensions.getResult(1));
          Value is_invalid = vr.back();
          is_invalid = b.create<arith::OrIOp>(
              l, is_invalid, current_dimension_has_invalid_broadcast);
          no_broadcasting.push_back(is_invalid);
          b.create<scf::YieldOp>(l, no_broadcasting);
        });
    Value is_invalid = main_loop.getResults().back();
    for (size_t i = 0; i < k; ++i) {
      result_shapes[i] =
          RemoveLeadingOnesFrom1DMemref(lb, result_shapes[i], ranks[i]);
      result_shapes[i] =
          lb.create<arith::SelectOp>(is_invalid, shapes[i], result_shapes[i]);
    }
    rewriter.replaceOp(broadcast_shapes_op, result_shapes);
    return success();
  }

 private:
  Value CountLeadingOnes(ImplicitLocOpBuilder &lb, Value extent_memref,
                         Value rank) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_8(mht_8_v, 541, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "CountLeadingOnes");

    // Count leading 1's. Use two iteration variables for that: one with a
    // boolean flag for whether every size so far was 1, one with the number of
    // leading 1's.
    Value constant_true =
        lb.create<arith::ConstantOp>(lb.getI1Type(), lb.getBoolAttr(true));
    Value zero = lb.create<arith::ConstantIndexOp>(0);
    Value one = lb.create<arith::ConstantIndexOp>(1);
    auto leading_ones_loop = lb.create<scf::ForOp>(
        zero, rank, one, ValueRange{constant_true, zero},
        [&](OpBuilder &b, Location l, Value idx, ValueRange vr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_9(mht_9_v, 554, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "lambda");

          auto size = b.create<memref::LoadOp>(l, extent_memref, idx);
          auto is_equal_to_one =
              b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, size, one);
          auto all_ones =
              b.create<arith::AndIOp>(l, vr.front(), is_equal_to_one);
          auto increased_value = b.create<arith::AddIOp>(l, vr.back(), one);
          auto number_of_leading_ones = b.create<arith::SelectOp>(
              l, all_ones, increased_value, vr.back());
          b.create<scf::YieldOp>(l,
                                 ValueRange{all_ones, number_of_leading_ones});
        });
    return leading_ones_loop.getResults()[1];
  }

  Value RemoveLeadingOnesFrom1DMemref(ImplicitLocOpBuilder &lb,
                                      Value extent_memref, Value rank) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_10(mht_10_v, 573, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "RemoveLeadingOnesFrom1DMemref");

    Value leading_ones = CountLeadingOnes(lb, extent_memref, rank);
    Value new_rank = lb.create<arith::SubIOp>(rank, leading_ones);
    auto result_type =
        MemRefType::get({ShapedType::kDynamicSize}, lb.getIndexType());
    // We cannot use SubView here to return a MemRef with 'leading_ones' as
    // offset, because that also changes the size, so the result type would need
    // to have an affine map to change the layout. This is incompatible to our
    // other MemRef types without affine map. So instead we just allocate
    // another buffer of the desired size and copy the elements over. We assume
    // the buffer will be small, so we allocate it on the stack.
    // TODO(b/181654096): Replace AllocaOp with AllocOp.
    Value result = lb.create<memref::AllocaOp>(result_type, new_rank);
    Value zero = lb.create<arith::ConstantIndexOp>(0);
    Value one = lb.create<arith::ConstantIndexOp>(1);
    lb.create<scf::ForOp>(
        zero, new_rank, one, llvm::None,
        [&](OpBuilder &b, Location l, Value idx, ValueRange /*vr*/) {
          Value idx_with_offset = b.create<arith::AddIOp>(l, idx, leading_ones);
          auto size =
              b.create<memref::LoadOp>(l, extent_memref, idx_with_offset);
          b.create<memref::StoreOp>(l, size, result, idx);
          b.create<scf::YieldOp>(l, llvm::None);
        });
    return result;
  }
};

struct BufferizeJITExecuteOp
    : public OpConversionPattern<tf_framework::JITExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tf_framework::JITExecuteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_11(mht_11_v, 610, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "matchAndRewrite");

    SmallVector<Type, 2> result_types;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                result_types))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tf_framework::JITExecuteOp>(
        op, result_types, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

}  // namespace

void populateExtraBufferizePatterns(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferizeDTcc mht_12(mht_12_v, 629, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize.cc", "populateExtraBufferizePatterns");

  // clang-format off
  patterns->add<
      BufferizeAndConvertMinimumBroadcastShapesOp,
      BufferizeConstantOp,
      BufferizeJITExecuteOp
  >(*converter, context);
  // clang-format on
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
