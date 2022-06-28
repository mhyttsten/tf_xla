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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc() {
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

// This file provides optional optimization patterns for mhlo, canonocalizing
// operations to equivalent but potentially more efficient operations.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::RewritePatternSet;

namespace mlir {
namespace mhlo {
namespace {

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/optimize_mhlo.cc", "GetI64ElementsAttr");

  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

class GatherIsSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/optimize_mhlo.cc", "matchAndRewrite");

    auto dimension_numbers = gather.dimension_numbers();

    // Inputs need to be ranked to lower.
    if (!gather.operand().getType().cast<ShapedType>().hasRank() ||
        !gather.operand().getType().cast<ShapedType>().hasStaticShape() ||
        !gather.start_indices().getType().cast<ShapedType>().hasRank() ||
        !gather.start_indices().getType().cast<ShapedType>().hasStaticShape()) {
      return rewriter.notifyMatchFailure(gather,
                                         "non-static operand or start_indices");
    }

    if (dimension_numbers.getIndexVectorDim() != 0) {
      return rewriter.notifyMatchFailure(gather, "non-zero index_vector_dim");
    }

    // TODO(suderman): Handle start index map != {0}.
    if (dimension_numbers.getStartIndexMap().empty() ||
        dimension_numbers.getStartIndexMap().size() != 1 ||
        dimension_numbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather,
                                         "start_index_map not empty or [0]");
    }

    auto result_ty = gather.getResult().getType().dyn_cast<RankedTensorType>();

    if (!result_ty) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }
    if (dimension_numbers.getOffsetDims().size() != result_ty.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "offset_dims.size != operand.rank");
    }
    for (const auto& it : llvm::enumerate(dimension_numbers.getOffsetDims())) {
      if (it.index() != it.value()) {
        return rewriter.notifyMatchFailure(gather,
                                           "offset_dims != [0, result.rank)");
      }
    }

    if (gather.slice_sizes().size() <= result_ty.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "slices_size.size > result.rank");
    }

    for (const auto& it : llvm::enumerate(result_ty.getShape())) {
      if (gather.slice_sizes().getValues<int64_t>()[it.index() + 1] !=
          it.value()) {
        return failure();
      }
    }

    auto gather_start_indices = gather.start_indices();
    auto gather_start_indices_ty =
        gather_start_indices.getType().cast<ShapedType>();

    llvm::SmallVector<Value, 4> slice_start_indices;

    if (gather_start_indices_ty.getRank() == 0) {
      slice_start_indices.push_back(gather_start_indices);
    } else if (gather_start_indices_ty.getRank() == 1) {
      for (int i = 0; i < gather_start_indices_ty.getDimSize(0); i++) {
        auto start = GetI64ElementsAttr({i}, &rewriter);
        auto limit = GetI64ElementsAttr({i + 1}, &rewriter);
        auto stride = GetI64ElementsAttr({1}, &rewriter);
        auto indicesSlice = rewriter.create<SliceOp>(
            gather.getLoc(), gather_start_indices, start, limit, stride);
        auto reshaped = rewriter.create<ReshapeOp>(
            gather.getLoc(),
            RankedTensorType::get(
                {}, indicesSlice.getType().cast<ShapedType>().getElementType()),
            indicesSlice);
        slice_start_indices.push_back(reshaped);
      }
    } else {
      return rewriter.notifyMatchFailure(gather, "start_indices.rank > 1");
    }

    auto sliceSizesTy = gather.slice_sizes().getType();

    // Start indices have implicit zeros when not specified. This is because
    // Gather occurs similar to slicing where full slices are inferred. Add any
    // missing zeros as necessary.
    auto zero = rewriter.create<ConstOp>(
        gather.getLoc(), rewriter.getZeroAttr(RankedTensorType::get(
                             {}, gather_start_indices_ty.getElementType())));
    while (slice_start_indices.size() < sliceSizesTy.getDimSize(0)) {
      slice_start_indices.push_back(zero);
    }

    SmallVector<int64_t, 5> sliceShape;
    for (auto shapeValue : gather.slice_sizes().getValues<APInt>()) {
      sliceShape.push_back(shapeValue.getSExtValue());
    }

    auto sliceTy =
        RankedTensorType::get(sliceShape, result_ty.getElementType());
    auto slice = rewriter.create<DynamicSliceOp>(
        gather.getLoc(), sliceTy, gather.operand(), slice_start_indices,
        gather.slice_sizes());

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(), slice);

    return success();
  }
};

}  // end anonymous namespace

void PopulateOptimizeMHLOPatterns(MLIRContext* context,
                                  RewritePatternSet* patterns) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSoptimize_mhloDTcc mht_2(mht_2_v, 343, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/optimize_mhlo.cc", "PopulateOptimizeMHLOPatterns");

  patterns->add<GatherIsSlice>(context);
}
}  // end namespace mhlo
}  // end namespace mlir
