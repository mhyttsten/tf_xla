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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc() {
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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.h"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_gpu.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/generated_transform_patterns.inc"
}  // namespace

RewritePatternSet GetHardwareRewritePatterns(MLIRContext* context,
                                             const std::string& hardware) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hardware: \"" + hardware + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "GetHardwareRewritePatterns");

  auto* devce_hardware = GetTargetHardware(hardware);
  if (devce_hardware == nullptr) return {context};
  return devce_hardware->GetTransformations(context);
}

bool IsSupported(Operation* op, const std::string& hardware) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("hardware: \"" + hardware + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "IsSupported");

  auto* devce_hardware = GetTargetHardware(hardware);
  if (devce_hardware == nullptr) return {};
  return devce_hardware->IsOpSupported(op);
}

// ================== Convert Quantized Op ============================

// Walk through the func and convert the quantize ops to their float version.
void ConvertQuantizedOpToFloat(mlir::func::FuncOp func, OpBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "ConvertQuantizedOpToFloat");

  func.walk([&](Operation* op) {
    // TODO(renjieliu): Find a generic way to deal with const ops.
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp>(op) ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, TF::ConstOp, ConstOp>(op))
      return;

    bool int8_type_observed = false;
    bool uint8_type_observed = false;
    for (auto& input : op->getOpOperands()) {
      auto input_type = input.get().getType();
      if (IsQI8Type(input_type)) {
        int8_type_observed = true;
      } else if (IsQUI8Type(input_type)) {
        uint8_type_observed = true;
      }
    }

    // TODO(renjieliu): We probably should check whether the op supports float
    // execution to be safe. Although normally they should support float
    // execution. Not Quantized ops.
    if (!int8_type_observed && !uint8_type_observed) return;

    // Insert dequantize ops for every quantized input.
    SmallVector<Value, 4> dequantized_inputs;
    for (auto& input : op->getOpOperands()) {
      auto input_type = input.get().getType();
      if (IsQI8Type(input_type) || IsQUI8Type(input_type) ||
          IsQI32Type(input_type)) {
        auto dequantized_input_type =
            mlir::quant::QuantizedType::castToExpressedType(input_type);
        builder->setInsertionPoint(op);
        auto dequantize_op = builder->create<TFL::DequantizeOp>(
            op->getLoc(), dequantized_input_type, input.get());
        dequantized_inputs.push_back(dequantize_op);
      } else {
        dequantized_inputs.push_back(input.get());
      }
    }

    // Result types.
    SmallVector<Type, 4> result_types;
    for (auto result_type : op->getResultTypes()) {
      if (IsQI8Type(result_type) || IsQUI8Type(result_type)) {
        auto dequantized_result_type =
            mlir::quant::QuantizedType::castToExpressedType(result_type);
        result_types.push_back(dequantized_result_type);
      } else {
        result_types.push_back(result_type);
      }
    }

    // Build the new float-versioned op.
    OperationState state(op->getLoc(), op->getName());
    state.operands = dequantized_inputs;
    state.types = result_types;
    state.attributes = op->getAttrs();
    state.successors = op->getSuccessors();
    builder->setInsertionPoint(op);
    Operation* new_op = builder->create(state);

    // Insert quantize ops for every outputs and rewrite.
    for (int i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      auto result_type = result.getType();

      Value new_result = new_op->getResult(i);
      if (IsQI8Type(result_type) || IsQUI8Type(result_type)) {
        builder->setInsertionPoint(op);
        TFL::QuantizeOp quant_op = builder->create<TFL::QuantizeOp>(
            op->getLoc(), result_type, new_result, TypeAttr::get(result_type));
        new_result = quant_op.getResult();
      }

      // Rewire the outputs.
      result.replaceAllUsesWith(new_result);
    }

    // Remove the old op.
    op->erase();
  });
}

// Fold quantized i32 (normally bias) into their float values.
struct FoldQuantizedI32ToFloat : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern<TFL::DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequant_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_3(mht_3_v, 327, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "matchAndRewrite");

    // We only fold i32 -> float pattern.
    auto input = dequant_op.input().getDefiningOp();
    if (!input) return failure();

    auto input_dequant = llvm::dyn_cast_or_null<TFL::QConstOp>(input);
    if (!input_dequant) return failure();

    if (!IsQI32Type(input_dequant.getType())) return failure();

    auto output_type =
        dequant_op.output().getType().dyn_cast_or_null<ShapedType>();
    if (!output_type || !output_type.getElementType().isF32()) return failure();

    auto input_type = input_dequant.getType().dyn_cast<ShapedType>();
    // TODO(renjieliu): support UniformQuantizedPerAxisType.
    auto q_type = input_type.getElementType()
                      .dyn_cast_or_null<quant::UniformQuantizedType>();
    if (!q_type) return failure();

    const float scale = q_type.getScale();
    const float zp = q_type.getZeroPoint();

    auto input_values = input_dequant.value();

    // mapValues always takes a function returning APInt, even when the output
    // is actually float.
    using DequantizeFuncType = llvm::APInt(const llvm::APInt&);
    auto dequantize_func = [&](const APInt& ap_int_value) -> APInt {
      const int64_t int_value = ap_int_value.getSExtValue();

      const float real = (int_value - zp) * scale;

      auto real_int = absl::bit_cast<int32_t>(real);
      return APInt(/*numBits=*/32, real_int);
    };

    auto dequant_values =
        input_values.cast<DenseIntOrFPElementsAttr>().mapValues(
            FloatType::getF32(rewriter.getContext()),
            llvm::function_ref<DequantizeFuncType>(dequantize_func));
    rewriter.replaceOpWithNewOp<TFL::ConstOp>(dequant_op, dequant_op.getType(),
                                              dequant_values);

    return success();
  }
};

// If the quant op has no consumer, we will remove them.
struct RemoveUnusedQuant : public OpRewritePattern<TFL::QuantizeOp> {
  using OpRewritePattern<TFL::QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::QuantizeOp quant_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_4(mht_4_v, 383, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "matchAndRewrite");

    if (!quant_op.getResult().use_empty()) return failure();

    rewriter.eraseOp(quant_op);
    return success();
  }
};

void OptimizeQuantizedOpToFloat(FuncOp func, MLIRContext* context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSdevice_transformDTcc mht_5(mht_5_v, 394, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.cc", "OptimizeQuantizedOpToFloat");

  RewritePatternSet patterns(func.getContext());
  patterns
      .add<FoldQuantizedI32ToFloat, FoldQuantizeDequantize, RemoveUnusedQuant>(
          context);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
