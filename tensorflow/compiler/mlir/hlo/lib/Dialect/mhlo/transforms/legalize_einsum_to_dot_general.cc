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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc() {
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

#include <algorithm>
#include <utility>

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

DenseIntElementsAttr Make1DElementsAttr(OpBuilder &b,
                                        ArrayRef<int64_t> integers) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc", "Make1DElementsAttr");

  auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                    b.getI64Type());
  return DenseIntElementsAttr::get(type, integers);
}

struct EinsumToDotGeneralPattern : public OpRewritePattern<EinsumOp> {
  using OpRewritePattern<EinsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EinsumOp einsum,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc", "matchAndRewrite");

    StringRef equation = einsum.einsum_config();
    SmallVector<char> lhs_tokens, rhs_tokens;
    SmallVector<char> result_tokens;
    size_t index = 0;
    enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
    EquationVariable current_variable = kIsLhs;
    while (index < equation.size()) {
      if (std::isalpha(equation[index])) {
        if (current_variable == kIsLhs) {
          lhs_tokens.push_back(equation[index]);
        } else if (current_variable == kIsRhs) {
          rhs_tokens.push_back(equation[index]);
        } else {
          result_tokens.push_back(equation[index]);
        }
      } else if (equation.substr(index, 1).contains(",")) {
        current_variable = kIsRhs;
      } else if ((index < (equation.size() - 1)) &&
                 (equation.substr(index, 2).contains("->"))) {
        current_variable = kIsResult;
        index++;
      } else {
        return einsum.emitError("unexpected character ")
               << equation.substr(index, 1) << " encountered";
      }
      index++;
    }

    auto lhs_type = einsum.lhs().getType().cast<RankedTensorType>();
    auto rhs_type = einsum.rhs().getType().cast<RankedTensorType>();
    assert(static_cast<int64_t>(lhs_tokens.size()) == lhs_type.getRank());
    assert(static_cast<int64_t>(rhs_tokens.size()) == rhs_type.getRank());

    auto collect_operand_dims = [&](RankedTensorType operand_type,
                                    SmallVector<char> operand_tokens,
                                    SmallVector<char> others,
                                    SmallVectorImpl<int64_t> &contracting_dims,
                                    SmallVectorImpl<int64_t> &batching_dims,
                                    SmallVector<char> &dot_result_tokens,
                                    SmallVector<int64_t> &dot_result_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc", "lambda");

      llvm::SmallDenseSet<char> others_set(others.begin(), others.end());
      llvm::SmallDenseSet<char> result_tokens_set(result_tokens.begin(),
                                                  result_tokens.end());
      for (const auto &en : llvm::enumerate(operand_tokens)) {
        bool is_result_token = result_tokens_set.contains(en.value());
        bool is_other_token = others_set.contains(en.value());

        if (!is_result_token) {
          contracting_dims.push_back(en.index());
        } else if (is_other_token) {
          batching_dims.push_back(en.index());
        } else {
          dot_result_tokens.push_back(en.value());
          dot_result_shape.push_back(operand_type.getShape()[en.index()]);
        }
      }
    };
    // Indices of batch and contracting dims, relative to each operand's
    // dimensions.
    SmallVector<int64_t> lhs_contracting_dims, lhs_batching_dims,
        rhs_contracting_dims, rhs_batching_dims;
    // Tokens representing the natural order of the dot_general op (i.e.
    // the lhs non-contracting followed by rhs non-contracting tokens).
    SmallVector<char> dot_result_tokens;
    SmallVector<int64_t> dot_result_shape;

    collect_operand_dims(lhs_type, lhs_tokens, rhs_tokens, lhs_contracting_dims,
                         lhs_batching_dims, dot_result_tokens,
                         dot_result_shape);
    collect_operand_dims(rhs_type, rhs_tokens, lhs_tokens, rhs_contracting_dims,
                         rhs_batching_dims, dot_result_tokens,
                         dot_result_shape);

    // Prepend batch tokens.
    for (const auto &it : llvm::enumerate(lhs_batching_dims)) {
      char batching_token = lhs_tokens[it.value()];
      int64_t batching_shape_dim = lhs_type.getShape()[it.value()];
      dot_result_tokens.insert(dot_result_tokens.begin() + it.index(),
                               batching_token);
      dot_result_shape.insert(dot_result_shape.begin() + it.index(),
                              batching_shape_dim);
    }

    // Lowering to dot_general does not support a mismatch between the number
    // of result dims and the number of non-contracting dims.
    if (dot_result_tokens.size() != result_tokens.size()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "rank reducing einsum not supported");
    }

    // Generate a permutation sequence based on result tokens.
    SmallVector<int64_t> result_perms;
    bool is_natural_order = true;
    for (char result_token : result_tokens) {
      auto *found_it = std::find(dot_result_tokens.begin(),
                                 dot_result_tokens.end(), result_token);
      if (found_it == dot_result_tokens.end()) {
        return rewriter.notifyMatchFailure(
            einsum, "result token not found in operands");
      }
      auto result_index = std::distance(dot_result_tokens.begin(), found_it);
      if (result_perms.empty()) {
        if (result_index != 0) {
          is_natural_order = false;
        }
      } else if (result_index != (result_perms.back() + 1)) {
        is_natural_order = false;
      }
      result_perms.push_back(result_index);
    }

    // Emit the dot_general, using its native result ordering.
    auto dot_general_result_type = RankedTensorType::get(
        ArrayRef<int64_t>(dot_result_shape), lhs_type.getElementType());
    auto dim_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhs_batching_dims, rhs_batching_dims,
        lhs_contracting_dims, rhs_contracting_dims);
    auto dot_general_op =
        rewriter.create<DotGeneralOp>(einsum.getLoc(), dot_general_result_type,
                                      einsum.lhs(), einsum.rhs(), dim_numbers,
                                      /*precision_config=*/ArrayAttr{});

    if (is_natural_order) {
      // The dot_general is already in an appropriate result order.
      rewriter.replaceOp(einsum, ValueRange{dot_general_op});
    } else {
      // Generate a transpose.
      rewriter.replaceOpWithNewOp<TransposeOp>(
          einsum, dot_general_op, rewriter.getI64TensorAttr(result_perms));
    }
    return success();
  }
};

struct LegalizeEinsumToDotGeneralPass
    : public LegalizeEinsumToDotGeneralPassBase<
          LegalizeEinsumToDotGeneralPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc mht_3(mht_3_v, 358, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    PopulateEinsumToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

void PopulateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        RewritePatternSet *patterns) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSlegalize_einsum_to_dot_generalDTcc mht_4(mht_4_v, 373, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc", "PopulateEinsumToDotGeneralPatterns");

  patterns->add<EinsumToDotGeneralPattern>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeEinsumToDotGeneralPass() {
  return std::make_unique<LegalizeEinsumToDotGeneralPass>();
}

}  // namespace mhlo
}  // namespace mlir
