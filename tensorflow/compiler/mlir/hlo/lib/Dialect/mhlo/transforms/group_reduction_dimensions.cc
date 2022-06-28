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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc() {
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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

LogicalResult TryLowerToCollapseShape(
    ReduceOp op, RankedTensorType arg_ty, Value arg,
    SmallVector<int64_t>& ordered_reduction_dims, PatternRewriter& rewriter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "TryLowerToCollapseShape");

  // This only works for trivial reductions where all declared reduction
  // dimensiosn are of extent 1.
  if (!llvm::all_of(ordered_reduction_dims,
                    [&](int64_t i) { return arg_ty.getDimSize(i) == 1; })) {
    return failure();
  }

  int64_t arg_rank = arg_ty.getRank();
  int64_t num_reduction_dims = ordered_reduction_dims.size();

  int64_t j = 0;
  auto is_declared_as_reduction_dim = [&](int64_t i) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "lambda");

    if (j < num_reduction_dims && ordered_reduction_dims[j] == i) {
      j++;
      return true;
    }
    return false;
  };

  // Build reassociation indices.
  SmallVector<ReassociationIndices, 4> reassociation;
  int64_t i_begin = 0;
  int64_t i = 0;
  while (i < arg_rank && is_declared_as_reduction_dim(i)) i++;
  while (i < arg_rank) {
    i++;
    while (i < arg_rank && is_declared_as_reduction_dim(i)) i++;
    reassociation.push_back(llvm::to_vector(llvm::seq(i_begin, i)));
    i_begin = i;
  }

  // Lower reduction op to collapse shape op.
  rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, arg, reassociation);
  return success();
}

enum class DimensionKind {
  kParallel,
  kReduction,
  kDegenerate,
};

struct DimensionGroup {
  DimensionKind kind;
  int64_t begin;
  int64_t end;
  int64_t size() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_2(mht_2_v, 260, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "size");
 return end - begin; }
};

// Groups consecutive dimensions of a reduction argument by their kind, i.e. if
// they are reduction or parallel dimensions. Dimensions of size 1 can be
// considered as any kind.
void GroupDimensions(RankedTensorType arg_ty,
                     SmallVector<int64_t> ordered_reduction_dims,
                     SmallVector<DimensionGroup>& groups) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "GroupDimensions");

  int64_t arg_rank = arg_ty.getRank();
  int64_t num_reduction_dims = ordered_reduction_dims.size();
  int64_t j = 0;
  for (int64_t i = 0; i < arg_rank; ++i) {
    // Check if the i-th dimension is one of the declared reduction dimensions.
    bool is_declared_as_reduction_dim = false;
    if (j < num_reduction_dims && i == ordered_reduction_dims[j]) {
      is_declared_as_reduction_dim = true;
      j++;
    }

    // Use the declared dimension kind unless the dimension is of extent 1, in
    // which case we can consider it either kind. We exploit this to form
    // maximal dimension groups.
    DimensionKind kind = is_declared_as_reduction_dim
                             ? DimensionKind::kReduction
                             : DimensionKind::kParallel;
    if (arg_ty.getDimSize(i) == 1) kind = DimensionKind::kDegenerate;

    // Start a new dimension group if the dimenion kind conflicts with the
    // trailing kind.
    if (groups.empty() || (groups.back().kind != kind &&
                           groups.back().kind != DimensionKind::kDegenerate &&
                           kind != DimensionKind::kDegenerate)) {
      groups.push_back({kind, i, i});
    }

    // Include dimension in trailing group and concretize dimension kind if
    // necessary.
    if (groups.back().kind == DimensionKind::kDegenerate)
      groups.back().kind = kind;
    groups.back().end++;
  }
}

LogicalResult TryLowerTo1DOr2DReduction(
    ReduceOp op, RankedTensorType arg_ty, Value arg,
    SmallVector<int64_t>& ordered_reduction_dims,
    bool prefer_columns_reductions, PatternRewriter& rewriter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_4(mht_4_v, 313, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "TryLowerTo1DOr2DReduction");

  // Group the argument dimensions by their kind.
  SmallVector<DimensionGroup> dim_groups;
  GroupDimensions(arg_ty, ordered_reduction_dims, dim_groups);

  // Do not (re-)apply if the dimensions are already fully collapsed.
  if (dim_groups.size() <= 2 &&
      llvm::all_of(dim_groups, [](auto g) { return g.size() == 1; })) {
    return failure();
  }

  // Determine whether or not a dynamic reshape is needed for the final result.
  int64_t num_dyn_parallel_dims = 0;
  for (auto group : dim_groups) {
    if (group.kind != DimensionKind::kParallel) continue;
    for (int64_t i = group.begin; i < group.end; i++) {
      if (arg_ty.isDynamicDim(i)) num_dyn_parallel_dims++;
    }
  }
  bool requires_dynamic_reshape = num_dyn_parallel_dims > 1;

  // Reify the result shape early so that the pattern can fail without altering
  // the IR.
  Optional<Value> result_shape;
  if (requires_dynamic_reshape) {
    llvm::SmallVector<Value, 1> reified_shapes;
    if (failed(llvm::cast<InferShapedTypeOpInterface>(op.getOperation())
                   .reifyReturnTypeShapes(rewriter, op->getOperands(),
                                          reified_shapes))) {
      return failure();
    }
    assert(reified_shapes.size() == 1 && "expect exactly one shape");
    result_shape = reified_shapes.front();
  }

  // Collapse dimension groups so that all adjacent dimensions of the
  // intermediate result are of a different kind.
  Value interm_result = arg;
  auto loc = op.getLoc();
  bool requires_collapse =
      llvm::any_of(dim_groups, [&](auto g) { return g.size() > 1; });
  if (requires_collapse) {
    auto reassociation =
        llvm::to_vector(llvm::map_range(dim_groups, [&](auto g) {
          return llvm::to_vector<2>(llvm::seq<int64_t>(g.begin, g.end));
        }));
    interm_result = rewriter.create<tensor::CollapseShapeOp>(loc, interm_result,
                                                             reassociation);
  }

  // If required, transpose the intermediate result so that dimensions kinds
  // form two partitions, which can be collapsed to a 2D intermediate result.
  bool requires_transpose = dim_groups.size() > 2;
  if (requires_transpose) {
    // Materialize transpose.
    DimensionKind leading_dim_kind = prefer_columns_reductions
                                         ? DimensionKind::kReduction
                                         : DimensionKind::kParallel;
    DimensionKind trailing_dim_kind = prefer_columns_reductions
                                          ? DimensionKind::kParallel
                                          : DimensionKind::kReduction;
    SmallVector<int64_t> perm;
    for (int i = 0; i < dim_groups.size(); i++) {
      if (dim_groups[i].kind == leading_dim_kind) perm.push_back(i);
    }
    int64_t num_leading_dims = perm.size();
    for (int i = 0; i < dim_groups.size(); i++) {
      if (dim_groups[i].kind == trailing_dim_kind) perm.push_back(i);
    }
    auto perm_attr = rewriter.getI64TensorAttr(perm);
    interm_result = rewriter.create<TransposeOp>(loc, interm_result, perm_attr)
                        ->getResults()
                        .front();

    // Collapse intermediate result rank 2.
    SmallVector<ReassociationIndices, 2> reassociation = {
        llvm::to_vector<2>(llvm::seq<int64_t>(0, num_leading_dims)),
        llvm::to_vector<2>(llvm::seq<int64_t>(num_leading_dims, perm.size()))};
    interm_result = rewriter.create<tensor::CollapseShapeOp>(loc, interm_result,
                                                             reassociation);
  }

  // Materialize inner 1D or 2D reduction.
  bool leading_reduction =
      requires_transpose ? prefer_columns_reductions
                         : dim_groups.front().kind == DimensionKind::kReduction;
  int64_t reduction_dim = leading_reduction ? 0 : 1;
  auto reduction_dim_attr = rewriter.getI64VectorAttr({reduction_dim});
  Value init_val = op.init_values().front();
  auto reduction_op = rewriter.create<ReduceOp>(loc, interm_result, init_val,
                                                reduction_dim_attr);
  rewriter.inlineRegionBefore(op.body(), reduction_op.body(),
                              reduction_op.body().begin());
  interm_result = reduction_op->getResults().front();

  // Restore the expected shape by dynamic reshape, if required.
  auto result_ty = op->getResultTypes().front().cast<RankedTensorType>();
  if (requires_dynamic_reshape) {
    assert(result_shape && "expect to have reified the result shape");
    interm_result = rewriter.create<DynamicReshapeOp>(
        loc, result_ty, interm_result, *result_shape);
  }

  // Othwerise, restore the expected shape by shape expansion, if required.
  int64_t result_rank = result_ty.getRank();
  int64_t interm_result_rank =
      interm_result.getType().cast<RankedTensorType>().getRank();
  bool requires_expand =
      !requires_dynamic_reshape && result_rank != interm_result_rank;
  if (requires_expand) {
    assert(interm_result_rank <= 1 &&
           "expect intermediate result to be of rank 0 or 1 before expansion");
    SmallVector<ReassociationIndices, 1> reassociation;
    bool is_scalar_expansion = interm_result_rank == 0;
    if (!is_scalar_expansion)
      reassociation = {llvm::to_vector(llvm::seq<int64_t>(0, result_rank))};
    interm_result = rewriter.create<tensor::ExpandShapeOp>(
        loc, result_ty, interm_result, reassociation);
  }

  rewriter.replaceOp(op, interm_result);
  return success();
}

struct GroupReductionDimensionsPattern : public OpRewritePattern<ReduceOp> {
  GroupReductionDimensionsPattern(MLIRContext* ctx,
                                  bool prefer_columns_reductions)
      : OpRewritePattern<ReduceOp>(ctx, /*benefit=*/1),
        prefer_columns_reductions(prefer_columns_reductions) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_5(mht_5_v, 444, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "GroupReductionDimensionsPattern");
}

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_6(mht_6_v, 450, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "matchAndRewrite");

    // Only apply to reduction of a unique argument.
    if (op.inputs().size() != 1 || op.init_values().size() != 1)
      return failure();
    Value arg = op.inputs().front();
    auto arg_ty = arg.getType().cast<RankedTensorType>();

    // Sort reduction dimensions, which is not an invariant of the op.
    SmallVector<int64_t> ordered_reduction_dims =
        llvm::to_vector<4>(llvm::map_range(op.dimensions(), [](auto d) {
          return static_cast<int64_t>(d.getLimitedValue());
        }));
    std::sort(ordered_reduction_dims.begin(), ordered_reduction_dims.end());

    // If all reduction dimensions are known to be of extent 1 then we can
    // express the reduction through an equivalent collapsing op.
    if (succeeded(TryLowerToCollapseShape(op, arg_ty, arg,
                                          ordered_reduction_dims, rewriter))) {
      return success();
    }

    // Otherwise, try lowering the reduction to an equivalent 1D or 2D
    // reduction, and insert transposes if needed.
    if (succeeded(
            TryLowerTo1DOr2DReduction(op, arg_ty, arg, ordered_reduction_dims,
                                      prefer_columns_reductions, rewriter))) {
      return success();
    }

    return failure();
  }

  bool prefer_columns_reductions;
};

struct GroupReductionDimensionsPass
    : public GroupReductionDimensionsPassBase<GroupReductionDimensionsPass> {
  explicit GroupReductionDimensionsPass(bool prefer_columns_reductions)
      : GroupReductionDimensionsPassBase<
            GroupReductionDimensionsPass>::GroupReductionDimensionsPassBase() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_7(mht_7_v, 492, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "GroupReductionDimensionsPass");

    prefer_columns_reductions_ = prefer_columns_reductions;
  }

  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_8(mht_8_v, 499, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "runOnOperation");

    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateGroupReductionDimensionsPatterns(ctx, &patterns,
                                             prefer_columns_reductions_);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateGroupReductionDimensionsPatterns(MLIRContext* context,
                                              RewritePatternSet* patterns,
                                              bool prefer_columns_reductions) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSgroup_reduction_dimensionsDTcc mht_9(mht_9_v, 518, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc", "populateGroupReductionDimensionsPatterns");

  patterns->add<GroupReductionDimensionsPattern>(context,
                                                 prefer_columns_reductions);
}

std::unique_ptr<OperationPass<FuncOp>> createGroupReductionDimensionsPass(
    bool prefer_columns_reductions) {
  return std::make_unique<GroupReductionDimensionsPass>(
      prefer_columns_reductions);
}

}  // namespace mhlo
}  // namespace mlir
