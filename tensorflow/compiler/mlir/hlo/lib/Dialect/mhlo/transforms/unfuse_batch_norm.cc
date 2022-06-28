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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc() {
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

#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

namespace {

// Broadcasts the 1D value tensor 'value_1d' to the shape of 'result_type'. If
// 'shape_value' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value BroadcastToFeatureDim(Location loc, RankedTensorType result_type,
                            Value value_1d, Value shape_value,
                            int64_t feature_dim,
                            PatternRewriter& rewriter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc", "BroadcastToFeatureDim");
  // NOLINT
  Builder b(rewriter.getContext());
  auto dims_type = RankedTensorType::get({1}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, {feature_dim});
  if (shape_value) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, result_type, value_1d, shape_value, dims);
  }
  assert(result_type.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, result_type, value_1d,
                                                 dims);
}

// Calculate the shape value of operand, assuming it is a dynamic shape with
// static rank.
Value CalculateShapeValue(Location loc, Value operand,
                          PatternRewriter& rewriter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc", "CalculateShapeValue");
  // NOLINT
  RankedTensorType result_type = operand.getType().dyn_cast<RankedTensorType>();
  llvm::SmallVector<Value, 4> shape_values;
  int64_t rank = result_type.getRank();
  shape_values.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    shape_values.push_back(
        rewriter.create<mlir::tensor::DimOp>(loc, operand, i));
  }
  return rewriter.create<tensor::FromElementsOp>(loc, shape_values);
}

Value MaterializeEpsilon(Operation* op, FloatAttr epsilon_attr,
                         FloatType fp_type, Value variance,
                         RankedTensorType broadcast_to_type,
                         PatternRewriter& rewriter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc mht_2(mht_2_v, 245, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc", "MaterializeEpsilon");
  // NOLINT
  Builder b(rewriter.getContext());
  if (epsilon_attr.getType() != fp_type) {
    // Need to convert.
    bool loses_info;
    APFloat epsilon_float = epsilon_attr.getValue();
    auto status = epsilon_float.convert(
        fp_type.getFloatSemantics(), APFloat::rmNearestTiesToEven, &loses_info);
    if ((status & (~APFloat::opInexact)) != APFloat::opOK) {
      op->emitWarning() << "Could not convert batch_norm epsilon to target fp "
                           "type: opStatus = "
                        << static_cast<int>(status);
      return nullptr;
    }
    if (loses_info) {
      op->emitWarning("Conversion of epsilon loses precision");
    }
    epsilon_attr = b.getFloatAttr(fp_type, epsilon_float);
  }

  auto scalar_type = RankedTensorType::get({}, fp_type);
  auto epsilon_tensor_attr =
      DenseElementsAttr::get(scalar_type, {epsilon_attr.cast<Attribute>()});
  Value epsilon =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), epsilon_tensor_attr);
  auto dims_type = RankedTensorType::get({0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, SmallVector<int64_t, 1>{});
  if (broadcast_to_type.hasStaticShape()) {
    return rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), broadcast_to_type, epsilon, /*broadcast_dims=*/dims);
  }
  Value shape_value = CalculateShapeValue(op->getLoc(), variance, rewriter);
  return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
      op->getLoc(), broadcast_to_type, epsilon, shape_value,
      /*broadcast_dims=*/dims);
}

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bn_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc mht_3(mht_3_v, 291, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc", "matchAndRewrite");

    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto input_type = bn_op.operand().getType().dyn_cast<RankedTensorType>();
    auto variance_type =
        bn_op.variance().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !variance_type) {
      return failure();
    }
    auto fp_type = variance_type.getElementType().dyn_cast<FloatType>();
    if (!fp_type) {
      return failure();
    }
    int64_t feature_dim = bn_op.feature_index();

    // Add epsilon to the variance and sqrt to get stddev:
    // stddev = sqrt(variance + epsilon)
    auto epsilon =
        MaterializeEpsilon(bn_op.getOperation(), bn_op.epsilonAttr(), fp_type,
                           bn_op.variance(), variance_type, rewriter);
    if (!epsilon) {
      return failure();
    }
    Value stddev =
        rewriter.create<mhlo::AddOp>(bn_op.getLoc(), bn_op.variance(), epsilon);
    stddev = rewriter.create<mhlo::SqrtOp>(bn_op.getLoc(), stddev);

    // Broadcast all terms.
    Value shape_value;
    if (!input_type.hasStaticShape()) {
      shape_value =
          CalculateShapeValue(bn_op.getLoc(), bn_op.operand(), rewriter);
    }
    auto broadcast_scale =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.scale(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_offset =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.offset(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_mean =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.mean(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_stddev = BroadcastToFeatureDim(
        bn_op.getLoc(), input_type, stddev, shape_value, feature_dim, rewriter);

    // Compute:
    // scale * (input - mean) / stddev + offset
    Value result = rewriter.create<mhlo::SubOp>(bn_op.getLoc(), bn_op.operand(),
                                                broadcast_mean);
    result =
        rewriter.create<mhlo::MulOp>(bn_op.getLoc(), result, broadcast_scale);
    result =
        rewriter.create<mhlo::DivOp>(bn_op.getLoc(), result, broadcast_stddev);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(bn_op, result, broadcast_offset);

    return success();
  }
};

}  // namespace

// Populates conversion patterns to unfuse batch normalization operations.
// In combination with marking such ops as illegal, this allows backends that
// do not have special support for fused batchnorm to use simpler arithmetic
// primitives.
void PopulateUnfuseBatchNormPatterns(MLIRContext* context,
                                     RewritePatternSet* patterns) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSunfuse_batch_normDTcc mht_4(mht_4_v, 361, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc", "PopulateUnfuseBatchNormPatterns");

  patterns->add<UnfuseBatchNormInferencePattern>(context);
}

}  // namespace mhlo
}  // namespace mlir
