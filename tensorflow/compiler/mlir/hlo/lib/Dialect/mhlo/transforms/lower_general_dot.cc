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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc() {
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

// This file implements logic for lowering MHLO general dot to a regular dot.

#include <sys/types.h>

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

Value TransposeReshape(Value arg, Location loc,
                       llvm::ArrayRef<int64_t> left_dims,
                       llvm::ArrayRef<int64_t> right_dims,
                       llvm::ArrayRef<int64_t> arg_shape,
                       PatternRewriter &rewriter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "TransposeReshape");

  auto element_type = getElementTypeOrSelf(arg.getType());

  int64_t left_size = 1;
  for (auto dim : left_dims) {
    left_size = (ShapedType::isDynamic(arg_shape[dim]) || left_size < 0)
                    ? ShapedType::kDynamicSize
                    : left_size * arg_shape[dim];
  }

  int64_t right_size = 1;
  for (auto dim : right_dims) {
    right_size = (ShapedType::isDynamic(arg_shape[dim]) || right_size < 0)
                     ? ShapedType::kDynamicSize
                     : right_size * arg_shape[dim];
  }

  // Generate the transpose permutation attribute.
  llvm::SmallVector<int64_t, 5> transpose_permutation(left_dims.begin(),
                                                      left_dims.end());
  transpose_permutation.append(right_dims.begin(), right_dims.end());

  TensorType transpose_permutation_type = RankedTensorType::get(
      {static_cast<int64_t>(transpose_permutation.size())},
      rewriter.getIntegerType(64));

  auto transpose_permutation_attr =
      DenseIntElementsAttr::get(transpose_permutation_type,
                                llvm::makeArrayRef(transpose_permutation))
          .cast<DenseIntElementsAttr>();

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 5> transposed_shape;
  for (auto val : transpose_permutation) {
    transposed_shape.push_back(arg_shape[val]);
  }
  auto transpose_type = RankedTensorType::get(transposed_shape, element_type);
  Value transpose_result = rewriter.create<TransposeOp>(
      loc, transpose_type, arg, transpose_permutation_attr);

  // If there are only a single pair of contracting dimensions and the output
  // rank is two we can skip a needless reshape.
  if (transpose_type.getRank() == 2 && left_dims.size() == 1 &&
      right_dims.size() == 1)
    return transpose_result;

  // Return the final result.
  auto reshaped_type =
      RankedTensorType::get({left_size, right_size}, element_type);

  if (reshaped_type.hasStaticShape()) {
    return rewriter.create<mhlo::ReshapeOp>(loc, reshaped_type,
                                            transpose_result);
  }

  SmallVector<Value> reshape_dims;
  auto multiply_dynamic_dims = [&](llvm::ArrayRef<int64_t> dims) -> Value {
    Value dynamic_size = rewriter.create<mhlo::GetDimensionSizeOp>(
        loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
        rewriter.getI64IntegerAttr(dims.front()));

    for (auto idx : dims.drop_front()) {
      Value dim = rewriter.create<mhlo::GetDimensionSizeOp>(
          loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
          rewriter.getI64IntegerAttr(idx));
      dynamic_size = rewriter.create<mhlo::MulOp>(loc, dynamic_size, dim);
    }
    return dynamic_size;
  };

  if (left_size < 0) {
    reshape_dims.push_back(multiply_dynamic_dims(left_dims));
  } else {
    reshape_dims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(left_size)));
  }

  if (right_size < 0) {
    reshape_dims.push_back(multiply_dynamic_dims(right_dims));
  } else {
    reshape_dims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(right_size)));
  }

  Value reshape_dims_tensor = rewriter.create<mhlo::ConcatenateOp>(
      loc, RankedTensorType::get({2}, rewriter.getI32Type()), reshape_dims,
      rewriter.getI64IntegerAttr(0));

  return rewriter.create<DynamicReshapeOp>(loc, reshaped_type, transpose_result,
                                           reshape_dims_tensor);
}

Value ProcessDotArg(Value arg, Location loc,
                    ArrayRef<int64_t> contract_dims_attr, bool outer_dims_first,
                    PatternRewriter &rewriter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_1(mht_1_v, 312, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "ProcessDotArg");

  auto shape = arg.getType().cast<ShapedType>().getShape();

  llvm::SmallVector<bool, 5> is_outer_dim;
  is_outer_dim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contract_dims;
  for (auto dim : contract_dims_attr) {
    contract_dims.push_back(dim);
    is_outer_dim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outer_dims;
  for (const auto &it : llvm::enumerate(is_outer_dim)) {
    if (it.value()) {
      outer_dims.push_back(it.index());
    }
  }

  if (outer_dims_first) {
    return TransposeReshape(arg, loc, outer_dims, contract_dims, shape,
                            rewriter);
  }

  return TransposeReshape(arg, loc, contract_dims, outer_dims, shape, rewriter);
}

struct GeneralDotConvert : public OpRewritePattern<DotGeneralOp> {
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.

  explicit GeneralDotConvert(MLIRContext *context)
      : OpRewritePattern(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_2(mht_2_v, 354, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "GeneralDotConvert");
}

  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_3(mht_3_v, 360, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "matchAndRewrite");

    Location loc = op.getLoc();

    auto dot_numbers = op.dot_dimension_numbers();
    if (!dot_numbers.getLhsBatchingDimensions().empty() ||
        !dot_numbers.getRhsBatchingDimensions().empty()) {
      return failure();
    }

    auto lhs_contracting_dims = dot_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dot_numbers.getRhsContractingDimensions();

    auto lhs = op.lhs();
    auto rhs = op.rhs();

    RankedTensorType lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_ty || !rhs_ty) return failure();

    lhs = ProcessDotArg(op.lhs(), op.getLoc(),
                        dot_numbers.getLhsContractingDimensions(),
                        /*outer_dims_first=*/true, rewriter);

    rhs = ProcessDotArg(op.rhs(), op.getLoc(),
                        dot_numbers.getRhsContractingDimensions(),
                        /*outer_dims_first=*/false, rewriter);

    // Accept only static shaped types.
    auto lhs_shape_type = lhs.getType().dyn_cast_or_null<ShapedType>();
    auto rhs_shape_type = rhs.getType().dyn_cast_or_null<ShapedType>();
    if (!lhs_shape_type || !rhs_shape_type) return failure();

    ArrayAttr precision_config;
    if (op.precision_config()) precision_config = *op.precision_config();
    Value new_dot_op =
        rewriter.create<DotOp>(op.getLoc(), lhs, rhs, precision_config);
    if (lhs_contracting_dims.size() == (lhs_ty.getRank() - 1) &&
        rhs_contracting_dims.size() == (rhs_ty.getRank() - 1)) {
      rewriter.replaceOp(op, new_dot_op);
      return success();
    }

    ShapedType result_ty = op.getType().cast<ShapedType>();

    // We can avoid all the computation below if we know the static shape.
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, result_ty, new_dot_op);
      return success();
    }

    llvm::SmallVector<int64_t> static_dims;
    llvm::SmallVector<Value> dyn_dims;

    auto getDynamicDims = [&](Value arg,
                              llvm::ArrayRef<int64_t> contracting_dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_4(mht_4_v, 417, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "lambda");

      RankedTensorType ty = arg.getType().cast<RankedTensorType>();
      int index = 0;
      for (auto contracting_dim : contracting_dims) {
        for (; index < contracting_dim; index++) {
          static_dims.push_back(ty.getDimSize(index));
          dyn_dims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
              loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
              rewriter.getI64IntegerAttr(index)));
        }
        index++;
      }

      for (; index < ty.getRank(); index++) {
        static_dims.push_back(ty.getDimSize(index));
        dyn_dims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
            loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
            rewriter.getI64IntegerAttr(index)));
      }
    };

    getDynamicDims(op.lhs(), lhs_contracting_dims);
    getDynamicDims(op.rhs(), rhs_contracting_dims);

    Value reshape_dims_tensor = rewriter.create<mhlo::ConcatenateOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(dyn_dims.size())},
                              rewriter.getI32Type()),
        dyn_dims, rewriter.getI64IntegerAttr(0));

    Value result = rewriter.create<DynamicReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(static_dims, result_ty.getElementType()),
        new_dot_op, reshape_dims_tensor);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LegalizeGeneralDotPass
    : public LegalizeGeneralDotPassBase<LegalizeGeneralDotPass> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_5(mht_5_v, 463, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    PopulateGeneralDotOpLoweringPatterns(&patterns, &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

void mlir::mhlo::PopulateGeneralDotOpLoweringPatterns(
    RewritePatternSet *patterns, MLIRContext *ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_6(mht_6_v, 481, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "mlir::mhlo::PopulateGeneralDotOpLoweringPatterns");

  patterns->add<GeneralDotConvert>(ctx);
}

std::unique_ptr<::mlir::Pass> mlir::mhlo::createLegalizeGeneralDotPass() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlower_general_dotDTcc mht_7(mht_7_v, 488, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/lower_general_dot.cc", "mlir::mhlo::createLegalizeGeneralDotPass");

  return std::make_unique<LegalizeGeneralDotPass>();
}
