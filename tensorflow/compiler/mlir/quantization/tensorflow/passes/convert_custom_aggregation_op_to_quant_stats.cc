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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class ConvertCustomAggregationOpToQuantStatsPass
    : public PassWrapper<ConvertCustomAggregationOpToQuantStatsPass,
                         OperationPass<FuncOp>> {
 public:
  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "getArgument");

    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-convert-tf-custom-aggregator-op-to-quant-stats";
  }

  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "getDescription");

    // This is a brief description of the pass.
    return "Convert tf.CustomAggregator op to quant.Stats";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "getDependentDialects");

    registry.insert<TF::TensorFlowDialect>();
    registry.insert<QuantizationDialect>();
  }

  void runOnOperation() override;
};

class ConvertCustomAggregationOpToQuantStats : public RewritePattern {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit ConvertCustomAggregationOpToQuantStats(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_3(mht_3_v, 244, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "ConvertCustomAggregationOpToQuantStats");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "matchAndRewrite");

    // Return early if the given operator isn't the custom aggregator op.
    if (op->getName().getStringRef() != "tf.CustomAggregator") return failure();

    FloatAttr min = op->getAttr("min").dyn_cast_or_null<FloatAttr>();
    FloatAttr max = op->getAttr("max").dyn_cast_or_null<FloatAttr>();

    // When there are no min and max attributes, remove op.
    if (min == nullptr || max == nullptr) {
      op->replaceAllUsesWith(op->getOperands());
      rewriter.eraseOp(op);
      return success();
    }

    // The layer stats contain only the first min/max pairs.
    ElementsAttr layer_stats = DenseFPElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getF32Type()),
        {static_cast<float>(min.getValueAsDouble()),
         static_cast<float>(max.getValueAsDouble())});
    ElementsAttr axis_stats;
    IntegerAttr axis;

    rewriter.replaceOpWithNewOp<StatisticsOp>(op, op->getOperand(0),
                                              layer_stats, axis_stats, axis);
    return success();
  }
};

static PassRegistration<ConvertCustomAggregationOpToQuantStatsPass> pass;

void ConvertCustomAggregationOpToQuantStatsPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSconvert_custom_aggregation_op_to_quant_statsDTcc mht_5(mht_5_v, 283, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_custom_aggregation_op_to_quant_stats.cc", "ConvertCustomAggregationOpToQuantStatsPass::runOnOperation");

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  FuncOp func = getOperation();

  patterns.add<ConvertCustomAggregationOpToQuantStats>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError()
        << "quant-convert-tf-custom-aggregator-op-to-quant-stats failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass() {
  return std::make_unique<ConvertCustomAggregationOpToQuantStatsPass>();
}

}  // namespace quant
}  // namespace mlir
