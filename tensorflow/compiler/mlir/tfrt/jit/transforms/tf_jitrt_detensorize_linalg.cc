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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc() {
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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::AffineMap;
using mlir::ConversionPatternRewriter;
using mlir::failure;
using mlir::LogicalResult;
using mlir::OpConversionPattern;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::success;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;
using mlir::linalg::GenericOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::FromElementsOp;

bool IsNotZeroRankTensor(RankedTensorType tensor_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_detensorize_linalg.cc", "IsNotZeroRankTensor");

  return !tensor_type || tensor_type.getRank() > 0;
}

/// A conversion patttern for detensoring Linalg ops.
struct DetensorizeLinalgOp : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenericOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc mht_1(mht_1_v, 233, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_detensorize_linalg.cc", "matchAndRewrite");

    mlir::Location loc = op.getLoc();
    mlir::SmallVector<AffineMap, 3> indexing_maps = op.getIndexingMaps();

    mlir::SmallVector<Value, 3> inputs;
    bool found_zero_dim_tensor = false;
    for (auto& en : llvm::enumerate(op.getInputOperands())) {
      auto tensor_type =
          en.value()->get().getType().dyn_cast<RankedTensorType>();
      if (IsNotZeroRankTensor(tensor_type)) {
        inputs.push_back(en.value()->get());
        continue;
      }
      found_zero_dim_tensor = true;
      indexing_maps[en.index()] =
          AffineMap::get(op.getNumLoops(), 0, llvm::None, op.getContext());
      inputs.push_back(rewriter.create<ExtractOp>(loc, en.value()->get(),
                                                  mlir::ValueRange{}));
    }
    if (!found_zero_dim_tensor) return failure();

    auto linalg_op = rewriter.create<GenericOp>(
        loc, op.getResultTypes(), inputs, op.outputs(),
        rewriter.getAffineMapArrayAttr(indexing_maps), op.iterator_types(),
        mlir::StringAttr(), mlir::StringAttr());
    mlir::Region& region = linalg_op.region();
    rewriter.inlineRegionBefore(op.getBodyRegion(), region, region.end());
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

struct DetensorizeLinalgPass
    : public DetensorizeLinalgBase<DetensorizeLinalgPass> {
  DetensorizeLinalgPass() = default;

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc mht_2(mht_2_v, 272, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_detensorize_linalg.cc", "runOnOperation");

    auto func = getOperation();
    auto* context = &getContext();

    mlir::ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });
    target.addDynamicallyLegalOp<GenericOp>([&](GenericOp op) {
      return llvm::all_of(TypeRange{op.inputs()}, [&](Type type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_detensorize_linalgDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_detensorize_linalg.cc", "lambda");

        return IsNotZeroRankTensor(type.dyn_cast<RankedTensorType>());
      });
    });

    // Detensorize.
    mlir::RewritePatternSet patterns(context);
    patterns.add<DetensorizeLinalgOp>(context);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();

    // Canonicalize.
    mlir::RewritePatternSet canonicalization_patterns(context);
    FromElementsOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            func, std::move(canonicalization_patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDetensorizeLinalgPass() {
  return std::make_unique<DetensorizeLinalgPass>();
}

}  // namespace tensorflow
