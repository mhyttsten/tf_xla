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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc() {
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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace mhlo {
namespace {

struct GatherIsTorchIndexSelect : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc", "matchAndRewrite");

    auto start_indices = gather.start_indices();
    auto start_indices_ty = start_indices.getType().cast<ShapedType>();
    if (!start_indices_ty.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked start_indices");
    }

    auto operand = gather.operand();
    auto operand_ty = operand.getType().cast<ShapedType>();
    if (!operand_ty.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked operand");
    }

    int64_t index_vector_dim =
        std::max<int64_t>(0, start_indices_ty.getRank() - 1);

    // We can use torch_index_select if the last dimension represents the
    // gather indices.
    auto dimension_numbers = gather.dimension_numbers();
    if (dimension_numbers.getIndexVectorDim() != index_vector_dim) {
      return rewriter.notifyMatchFailure(
          gather, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (!start_indices_ty.getShape().empty() &&
        start_indices_ty.getShape().back() != 1) {
      return rewriter.notifyMatchFailure(
          gather, "start_indices index vector dimension not 1");
    }

    // Only support the default case for start_index_map.
    if (dimension_numbers.getStartIndexMap().size() != 1 ||
        dimension_numbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "start_index_map != [0]");
    }

    auto result_ty = gather.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (dimension_numbers.getOffsetDims().size() !=
        result_ty.getRank() - index_vector_dim) {
      return rewriter.notifyMatchFailure(
          gather, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (const auto &it : llvm::enumerate(dimension_numbers.getOffsetDims())) {
      if ((it.index() + index_vector_dim) != it.value()) {
        return rewriter.notifyMatchFailure(
            gather, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (const auto &it :
         llvm::enumerate(gather.slice_sizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(gather, "slice_size[0] != 1");
        }
        continue;
      }

      // The gather needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operand_ty.getDimSize(it.index())) {
        return rewriter.notifyMatchFailure(
            gather, "slice_size doesn't match operand dimension");
      }
    }

    llvm::SmallVector<int64_t, 4> index_select_shape =
        llvm::to_vector<4>(start_indices_ty.getShape());

    for (auto dim : operand_ty.getShape().drop_front()) {
      index_select_shape.push_back(dim);
    }

    if (dimension_numbers.getCollapsedSliceDims().size() != 1 ||
        dimension_numbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "collapsed_slice_dims != [0]");
    }

    auto torch_index_select = rewriter.create<TorchIndexSelectOp>(
        gather.getLoc(),
        RankedTensorType::get(index_select_shape, operand_ty.getElementType()),
        operand, gather.start_indices(), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(),
                                           torch_index_select);

    return success();
  }
};

struct LegalizeGatherToTorchIndexSelectPass
    : public LegalizeGatherToTorchIndexSelectPassBase<
          LegalizeGatherToTorchIndexSelectPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc mht_1(mht_1_v, 307, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    PopulateGatherToTorchIndexSelectPatterns(&getContext(), &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

void PopulateGatherToTorchIndexSelectPatterns(mlir::MLIRContext *context,
                                              RewritePatternSet *patterns) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_gather_to_torch_index_selectDTcc mht_2(mht_2_v, 321, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc", "PopulateGatherToTorchIndexSelectPatterns");

  patterns->add<GatherIsTorchIndexSelect>(context);
}

std::unique_ptr<OperationPass<FuncOp>>
createLegalizeGatherToTorchIndexSelectPass() {
  return std::make_unique<LegalizeGatherToTorchIndexSelectPass>();
}

}  // namespace mhlo
}  // namespace mlir
