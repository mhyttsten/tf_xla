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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc() {
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

#include <utility>

#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

struct InferReturnTypeComponentsPattern : public RewritePattern {
  InferReturnTypeComponentsPattern(MLIRContext *context)
      : RewritePattern("mhlo_test.get_return_type_components", 1, context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "InferReturnTypeComponentsPattern");
}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "matchAndRewrite");

    if (op->getNumOperands() != 1) return failure();
    auto *defining_op = op->getOperand(0).getDefiningOp();
    auto defining_op_int =
        llvm::dyn_cast_or_null<InferShapedTypeOpInterface>(defining_op);
    if (!defining_op_int) return failure();
    SmallVector<ShapedTypeComponents, 4> components;
    if (failed(defining_op_int.inferReturnTypeComponents(
            op->getContext(), op->getLoc(), defining_op->getOperands(),
            defining_op->getAttrDictionary(), defining_op->getRegions(),
            components))) {
      return failure();
    }

    // Replace the op with another pass-through op with attributes added.
    OperationState state(op->getLoc(), "mhlo_test.return_type_components",
                         op->getOperands(), op->getResultTypes(),
                         op->getAttrs());
    auto *new_op = rewriter.create(state);
    for (const auto &it : llvm::enumerate(components)) {
      if (it.value().hasRank()) {
        new_op->setAttr((StringRef("dims") + Twine(it.index())).str(),
                        rewriter.getI64ArrayAttr(it.value().getDims()));
      }
      if (it.value().getElementType()) {
        new_op->setAttr((Twine("element_type") + Twine(it.index())).str(),
                        TypeAttr::get(it.value().getElementType()));
      }
    }
    rewriter.replaceOp(op, {new_op->getResults()});
    return success();
  }
};

struct ReifyReturnTypeShapesPattern : public RewritePattern {
  ReifyReturnTypeShapesPattern(MLIRContext *context)
      : RewritePattern("mhlo_test.reify_return_type_shapes", 1, context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "ReifyReturnTypeShapesPattern");
}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "matchAndRewrite");

    if (op->getNumOperands() != 1) return failure();
    auto defining_op =
        op->getOperand(0).getDefiningOp<InferShapedTypeOpInterface>();
    if (!defining_op) return failure();
    SmallVector<Value, 4> return_shapes;
    if (failed(defining_op.reifyReturnTypeShapes(
            rewriter, defining_op->getOperands(), return_shapes))) {
      return failure();
    }
    rewriter.replaceOp(op, return_shapes);
    return success();
  }
};

struct TestInferShapedTypeMethodsPass
    : public TestInferShapedTypeMethodsPassBase<
          TestInferShapedTypeMethodsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "getDependentDialects");

    registry.insert<shape::ShapeDialect>();
  }
  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPStest_infer_shaped_type_passDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    patterns.add<ReifyReturnTypeShapesPattern>(&getContext());
    patterns.add<InferReturnTypeComponentsPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createTestInferShapedTypeMethodsPass() {
  return std::make_unique<TestInferShapedTypeMethodsPass>();
}

}  // namespace mhlo
}  // namespace mlir
