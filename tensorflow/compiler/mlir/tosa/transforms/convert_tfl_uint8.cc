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
class MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc() {
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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
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

// Performs lowering to TOSA dialect.
class ConvertUint8ToInt8
    : public TosaConvertTFLUint8PassBase<ConvertUint8ToInt8> {
 public:
  explicit ConvertUint8ToInt8() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc mht_0(mht_0_v, 229, "", "./tensorflow/compiler/mlir/tosa/transforms/convert_tfl_uint8.cc", "ConvertUint8ToInt8");
}
  void runOnOperation() override;
};

struct ConvertUint8QConstOp : public RewritePattern {
  explicit ConvertUint8QConstOp(MLIRContext *context)
      : RewritePattern(TFL::QConstOp::getOperationName(), 1, context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/mlir/tosa/transforms/convert_tfl_uint8.cc", "ConvertUint8QConstOp");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &builder) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc mht_2(mht_2_v, 244, "", "./tensorflow/compiler/mlir/tosa/transforms/convert_tfl_uint8.cc", "matchAndRewrite");

    auto tfl_qconst_op = cast<TFL::QConstOp>(op);

    // Skip if it's not ranked tensor type.
    auto output_type =
        tfl_qconst_op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
    if (!output_type)
      return builder.notifyMatchFailure(op, "not ranked tensor");

    // Skip if output is not per-tensor quantized type.
    auto output_element_type =
        output_type.getElementType()
            .dyn_cast<mlir::quant::UniformQuantizedType>();
    if (!output_element_type) return failure();

    // Skip if output is not uint8.
    if (output_element_type.isSigned() ||
        output_element_type.getStorageTypeIntegralWidth() != 8) {
      return failure();
    }

    mlir::DenseElementsAttr src_dense_attr =
        tfl_qconst_op.value().cast<DenseElementsAttr>();

    double type_range_min =
        static_cast<double>(output_element_type.getStorageTypeMin() -
                            output_element_type.getZeroPoint()) *
        output_element_type.getScale();
    double type_range_max =
        static_cast<double>(output_element_type.getStorageTypeMax() -
                            output_element_type.getZeroPoint()) *
        output_element_type.getScale();
    bool narrow_range =
        output_element_type.getStorageTypeMin() == 1 ? true : false;

    auto dst_qconst_type = TypeAttr::get(RankedTensorType::get(
        output_type.getShape(),
        buildQTypeFromMinMax(
            builder, output_element_type.getExpressedType(),
            builder.getF64FloatAttr(type_range_min),
            builder.getF64FloatAttr(type_range_max),
            builder.getI32IntegerAttr(
                output_element_type.getStorageTypeIntegralWidth()),
            0, true /* signed */, builder.getBoolAttr(narrow_range))));

    Type dst_dense_element_type = builder.getIntegerType(8);
    llvm::function_ref<APInt(const APInt &)> mapping =
        [](const APInt &in) -> APInt {
      int64_t in_i64 = in.getLimitedValue();
      int64_t out_i64 = in_i64 - 128;
      return APInt(8, out_i64, true);
    };

    auto dst_dense_attr =
        src_dense_attr.mapValues(dst_dense_element_type, mapping);

    builder.replaceOpWithNewOp<TFL::QConstOp>(op, dst_qconst_type,
                                              dst_dense_attr);

    return success();
  }
};

LogicalResult convert_graph_uint8_tensor(mlir::MLIRContext &context,
                                         mlir::func::FuncOp &function) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc mht_3(mht_3_v, 311, "", "./tensorflow/compiler/mlir/tosa/transforms/convert_tfl_uint8.cc", "convert_graph_uint8_tensor");

  size_t num_blocks_in_main = 0;
  mlir::Region *region = function.getCallableRegion();
  OpBuilder builder(&context);

  auto tmp_const_type = RankedTensorType::get({1}, builder.getIntegerType(8));
  auto tmp_const_attr =
      DenseElementsAttr::get(tmp_const_type, {static_cast<uint8_t>(0)});

  for (mlir::Block &bb : region->getBlocks()) {
    // Always have one block for each region right now.
    num_blocks_in_main++;
    if (num_blocks_in_main > 1) {
      return function.emitError("Invalid MLIR: multiple blocks in a region");
    }

    if (!bb.isEntryBlock()) {
      return function.emitError("Invalid MLIR: block must be entry block");
    }

    // Insert rescale uint8->int8 after placeholders.
    for (Value arg : bb.getArguments()) {
      auto uint8_type = arg.getType().dyn_cast<mlir::ShapedType>();
      if (!uint8_type) continue;

      auto uint8_element_type =
          uint8_type.getElementType()
              .dyn_cast<mlir::quant::UniformQuantizedType>();
      if (!uint8_element_type) continue;

      if (uint8_element_type.isSigned() ||
          uint8_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      double type_range_min =
          static_cast<double>(uint8_element_type.getStorageTypeMin() -
                              uint8_element_type.getZeroPoint()) *
          uint8_element_type.getScale();
      double type_range_max =
          static_cast<double>(uint8_element_type.getStorageTypeMax() -
                              uint8_element_type.getZeroPoint()) *
          uint8_element_type.getScale();
      bool narrow_range =
          uint8_element_type.getStorageTypeMin() == 1 ? true : false;

      Type int8_type = uint8_type.clone(buildQTypeFromMinMax(
          builder, uint8_element_type.getExpressedType(),
          builder.getF64FloatAttr(type_range_min),
          builder.getF64FloatAttr(type_range_max),
          builder.getI32IntegerAttr(
              uint8_element_type.getStorageTypeIntegralWidth()),
          0, true /* signed */, builder.getBoolAttr(narrow_range)));

      int32_t uint8_zp = uint8_element_type.getZeroPoint();
      int32_t int8_zp = uint8_zp - 128;

      // Keep original input_val use with tmp_val.
      Value tmp_val = builder.create<TFL::ConstOp>(
          function.getLoc(), tmp_const_type, tmp_const_attr);
      arg.replaceAllUsesWith(tmp_val);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          function.getLoc(), int8_type, arg,
          builder.getI32IntegerAttr(uint8_zp),
          builder.getI32IntegerAttr(int8_zp),
          builder.getI32ArrayAttr({1 << 30}), builder.getI32ArrayAttr({30}),
          builder.getBoolAttr(true), builder.getBoolAttr(false),
          builder.getBoolAttr(false));

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_front(op_rescale_op);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
    }

    // Record types of original graph output before we convert intermediate
    // tensor.
    auto terminator = bb.getTerminator();
    SmallVector<Type, 4> output_types;
    for (Value val : terminator->getOperands()) {
      output_types.push_back(val.getType());
    }

    // Convert intermediate tensor.
    for (auto &op : bb) {
      for (Value output_val : op.getResults()) {
        // Skip if output value is not RankedTensorType.
        auto output_type = output_val.getType().dyn_cast<mlir::ShapedType>();
        if (!output_type) continue;

        // Skip if output value is not per-tensor quantized element type.
        auto output_element_type =
            output_type.getElementType()
                .dyn_cast<mlir::quant::UniformQuantizedType>();
        if (!output_element_type) continue;

        // Skip if output is not uint8.
        if (output_element_type.isSigned() ||
            output_element_type.getStorageTypeIntegralWidth() != 8)
          continue;

        double type_range_min =
            static_cast<double>(output_element_type.getStorageTypeMin() -
                                output_element_type.getZeroPoint()) *
            output_element_type.getScale();
        double type_range_max =
            static_cast<double>(output_element_type.getStorageTypeMax() -
                                output_element_type.getZeroPoint()) *
            output_element_type.getScale();
        bool narrow_range =
            output_element_type.getStorageTypeMin() == 1 ? true : false;

        Type new_type = output_type.clone(buildQTypeFromMinMax(
            builder, output_element_type.getExpressedType(),
            builder.getF64FloatAttr(type_range_min),
            builder.getF64FloatAttr(type_range_max),
            builder.getI32IntegerAttr(
                output_element_type.getStorageTypeIntegralWidth()),
            0, true /* signed */, builder.getBoolAttr(narrow_range)));

        output_val.setType(new_type);
      }
    }

    if (terminator->getNumOperands() != output_types.size()) {
      return function.emitError(
          "Terminator's operand mismatch with number of outputs in graph");
    }

    // Insert int8->uint8 rescale before all terminator's operand.
    for (int32_t i = 0; i < terminator->getNumOperands(); i++) {
      auto defining_op = terminator->getOperand(i).getDefiningOp();
      // skip if operand of terminator is block arg (nullptr in this case) or
      // not
      if (!defining_op) continue;
      Value input_val = defining_op->getResult(0);

      // Check if graph output is uint8 type.
      auto uint8_output_type = output_types[i].dyn_cast<mlir::ShapedType>();
      if (!uint8_output_type) continue;

      auto uint8_output_element_type =
          uint8_output_type.getElementType()
              .dyn_cast<mlir::quant::UniformQuantizedType>();
      if (!uint8_output_element_type) continue;

      if (uint8_output_element_type.isSigned() ||
          uint8_output_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      // Check if output coming into terminator is int8 type.
      auto int8_output_type =
          terminator->getOperand(i).getType().dyn_cast<mlir::ShapedType>();
      if (!int8_output_type) continue;

      auto int8_output_element_type =
          int8_output_type.getElementType()
              .dyn_cast<mlir::quant::UniformQuantizedType>();
      if (!int8_output_element_type) continue;

      if (!int8_output_element_type.isSigned() ||
          int8_output_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      int32_t int8_zp = int8_output_element_type.getZeroPoint();
      int32_t uint8_zp = uint8_output_element_type.getZeroPoint();

      // Sanity check if uint8/int8's scale and zeropoint match.
      if (((uint8_zp - int8_zp) != 128) ||
          (int8_output_element_type.getScale() !=
           uint8_output_element_type.getScale())) {
        return terminator->emitError(
            "convert_uint8_to_int8: scale mismatch at the output tensors");
      }

      // Keep original input_val use with tmp_val.
      Value tmp_val = builder.create<TFL::ConstOp>(
          function.getLoc(), tmp_const_type, tmp_const_attr);
      input_val.replaceAllUsesWith(tmp_val);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          function.getLoc(), uint8_output_type, input_val,
          builder.getI32IntegerAttr(int8_zp),
          builder.getI32IntegerAttr(uint8_zp),
          builder.getI32ArrayAttr({1 << 30}), builder.getI32ArrayAttr({30}),
          builder.getBoolAttr(true), builder.getBoolAttr(false),
          builder.getBoolAttr(false));

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_back(op_rescale_op);
      op_rescale_op->moveBefore(terminator);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
    }
  }

  return success();
}

void ConvertUint8ToInt8::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSconvert_tfl_uint8DTcc mht_4(mht_4_v, 511, "", "./tensorflow/compiler/mlir/tosa/transforms/convert_tfl_uint8.cc", "ConvertUint8ToInt8::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto &ctx = getContext();
  mlir::func::FuncOp func = getOperation();

  // Convert uint8 const tensor. const needs to be handled specifically.
  patterns.add<ConvertUint8QConstOp>(&ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Replace uint8 tensor in the graph and insert rescale as needed.
  (void)convert_graph_uint8_tensor(ctx, func);
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertTFLUint8Pass() {
  return std::make_unique<ConvertUint8ToInt8>();
}

}  // namespace tosa

}  // namespace mlir
