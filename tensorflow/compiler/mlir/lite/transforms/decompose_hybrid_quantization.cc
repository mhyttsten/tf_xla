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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc() {
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

// This transformation pass decomposes dense operations that assume
// support for hybrid quantization. These cases cover when a dense operation
// (e.g. matmul) has both quantized and unquantized inputs by dequantizing
// the quantized inputs, performing the operation in the expressed type, then
// requantizing if a quantized output is required.
//
// The motivation behind these changes is for Dialects that assume only float
// or quantized computation, and do not support a mixture of these types on
// dense operations. Decomposition allows TFLite to be compiled to these
// dialects, such as TOSA.

#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

class DecomposeHybridQuantizationPass
    : public PassWrapper<DecomposeHybridQuantizationPass,
                         OperationPass<FuncOp>> {
 public:
  DecomposeHybridQuantizationPass() = default;
  DecomposeHybridQuantizationPass(const DecomposeHybridQuantizationPass &) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "DecomposeHybridQuantizationPass");
}

  StringRef getArgument() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "getArgument");

    return "tfl-decompose-hybrid-quantization";
  }

  StringRef getDescription() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "getDescription");

    return "Decomposes (with explicit quantize/dequantize ops) selected math "
           "operations which exist in the model with hybrid quantization "
           "(some arguments/results left in floating point).";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect>();
  }
};

template <typename SrcOp>
class DequantizeConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp srcop,
                                PatternRewriter &rewriter) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "matchAndRewrite");

    Operation *op = srcop.getOperation();
    bool allTypesFp = true;
    bool allTypesQuantizedOrInt = true;
    for (auto operand : op->getOperands()) {
      ShapedType type = operand.getType().template dyn_cast<ShapedType>();
      if (!type) continue;
      allTypesFp &= !type.getElementType().isa<quant::QuantizedType>();
      allTypesQuantizedOrInt &=
          (type.getElementType().isa<quant::QuantizedType>() ||
           type.getElementType().isa<IntegerType>());
    }

    for (auto result : op->getResults()) {
      ShapedType type = result.getType().template cast<ShapedType>();
      allTypesFp &= !type.getElementType().isa<quant::QuantizedType>();
      allTypesQuantizedOrInt &=
          (type.getElementType().isa<quant::QuantizedType>() ||
           type.getElementType().isa<IntegerType>());
    }

    // If all quantized or floating point then types are consistent.
    // Int is valid in combination with both quantized and floating point.
    // This occurs when doing qi16 convolution, as bias is passed as a
    // non-quantized int64
    if (allTypesFp || allTypesQuantizedOrInt) return failure();

    Location loc = op->getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (auto operand : op->getOperands()) {
      if (QuantizedType::getQuantizedElementType(operand.getType())) {
        auto newTy = QuantizedType::castToExpressedType(operand.getType());
        newOperands.push_back(
            rewriter.create<TFL::DequantizeOp>(loc, newTy, operand));
        continue;
      }

      newOperands.push_back(operand);
    }

    SmallVector<Type> newResultTys;
    for (auto result : op->getResults()) {
      Type resultTy = result.getType();
      if (QuantizedType::getQuantizedElementType(resultTy)) {
        resultTy = QuantizedType::castToExpressedType(resultTy);
      }
      newResultTys.push_back(resultTy);
    }

    auto newResults = rewriter
                          .create<SrcOp>(loc, newResultTys, newOperands,
                                         op->getAttrDictionary().getValue())
                          .getOperation()
                          ->getResults();

    SmallVector<Value> replaceResults;
    for (int i = 0; i < newResults.size(); i++) {
      Value result = newResults[i];
      Type resultTy = op->getOpResult(i).getType();
      if (QuantizedType::getQuantizedElementType(resultTy)) {
        replaceResults.push_back(rewriter.create<TFL::QuantizeOp>(
            loc, resultTy, result, TypeAttr::get(resultTy)));
        continue;
      }

      replaceResults.push_back(result);
    }

    rewriter.replaceOp(op, replaceResults);

    return success();
  }
};

void DecomposeHybridQuantizationPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdecompose_hybrid_quantizationDTcc mht_5(mht_5_v, 328, "", "./tensorflow/compiler/mlir/lite/transforms/decompose_hybrid_quantization.cc", "DecomposeHybridQuantizationPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto *ctx = &getContext();
  auto func = getOperation();
  patterns.add<DequantizeConverter<TFL::Conv2DOp>,
               DequantizeConverter<TFL::Conv3DOp>,
               DequantizeConverter<TFL::DepthwiseConv2DOp>,
               DequantizeConverter<TFL::FullyConnectedOp>,
               DequantizeConverter<TFL::TransposeConvOp>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateDecomposeHybridQuantizationPass() {
  return std::make_unique<DecomposeHybridQuantizationPass>();
}

static PassRegistration<DecomposeHybridQuantizationPass> pass;

}  // namespace TFL
}  // namespace mlir
