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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.h"

#include <functional>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tfrt/jitrt/support.h"  // from @tf_runtime

namespace tensorflow {

using mlir::failure;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::success;
using mlir::TensorType;
using mlir::Type;
using mlir::Value;

using mlir::TFDevice::Cluster;
using mlir::TFDevice::ClusteringPolicy;
using mlir::TFDevice::ClusteringPolicySet;
using mlir::TFDevice::ValueConstraint;
using mlir::TFDevice::ValuesConstraintSet;

using mlir::TF::_FusedMatMulOp;
using mlir::TF::BatchMatMulV2Op;
using mlir::TF::BroadcastToOp;
using mlir::TF::ConcatV2Op;
using mlir::TF::ConstOp;
using mlir::TF::ExpandDimsOp;
using mlir::TF::FillOp;
using mlir::TF::MatMulOp;
using mlir::TF::OneHotOp;
using mlir::TF::PackOp;
using mlir::TF::RangeOp;
using mlir::TF::ReshapeOp;
using mlir::TF::ShapeOp;
using mlir::TF::SliceOp;
using mlir::TF::SqueezeOp;
using mlir::TF::StopGradientOp;
using mlir::TF::StridedSliceOp;
using mlir::TF::TransposeOp;

namespace {

// A set of clustering constraints that allow TF -> JitRt compilation pipeline
// to lower Tensorflow operations to MHLO and then to Linalg. Tensorflow
// dynamism is not fully representable at Linalg level, so by providing a
// clustering policy we ensure that we can successfully compile all clustered
// operations (we have enough static information to lower to MHLO, or build
// static Linalg indexing maps).
//
// Some of these constraints gets resolved at constant folding time, and
// operations are completely removed from the IR, and some constraints just
// enable TF->MHLO or MHLO->Linalg lowering.

// Returns true if all types are supported by the Tensorflow -> JitRt
// compilation pipeline and TFRT JIT runtime integration (see jitrt.h).
template <typename TypeRange>
static bool IsSupportedDataTypes(TypeRange&& types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_0(mht_0_v, 253, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsSupportedDataTypes");

  return llvm::all_of(types, [](Type type) -> bool {
    if (auto tensor = type.dyn_cast<TensorType>()) {
      auto elt_type = tensor.getElementType();
      return elt_type.isF32() || elt_type.isInteger(1) ||
             elt_type.isInteger(32) || elt_type.isInteger(64);
    }
    return false;
  });
}

static bool IsSupportedOperandTypes(Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_1(mht_1_v, 267, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsSupportedOperandTypes");

  return IsSupportedDataTypes(op->getOperandTypes());
}

static bool IsSupportedResultTypes(Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_2(mht_2_v, 274, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsSupportedResultTypes");

  return IsSupportedDataTypes(op->getResultTypes());
}

static bool IsSupportedOperandAndResultTypes(Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_3(mht_3_v, 281, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsSupportedOperandAndResultTypes");

  return IsSupportedOperandTypes(op) && IsSupportedResultTypes(op);
}

// Clustering policy for a specific Tensorflow operation type that verifies
// that operation operands and results data types are supported.
template <typename OpTy>
class TensorflowOpClusteringPolicy : public ClusteringPolicy {
 public:
  LogicalResult MatchAndUpdateConstraints(
      Operation* operation, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_4(mht_4_v, 295, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    auto op = mlir::dyn_cast<OpTy>(operation);
    if (op && IsSupportedOperandAndResultTypes(op))
      return MatchAndUpdateConstraints(op, results, operands);
    return failure();
  }

  virtual LogicalResult MatchAndUpdateConstraints(
      OpTy op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const = 0;
};

// -------------------------------------------------------------------------- //
// Default clustering policy for TF -> JitRt compilation.
// -------------------------------------------------------------------------- //

// Default clustering policy for Tensorflow -> TFRT JIT compilation propagates
// the most restrictive constraint from the results to all operands. If results
// do not have any constraints it adds default constraint to all operands if it
// is provided, otherwise just returns `success` without adding any constraints.
class DefaultClusteringPolicy : public ClusteringPolicy {
 public:
  explicit DefaultClusteringPolicy(
      std::function<bool(Operation*)> filter,
      llvm::Optional<ValueConstraint> default_constraint = llvm::None)
      : filter_(std::move(filter)), default_constraint_(default_constraint) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_5(mht_5_v, 323, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "DefaultClusteringPolicy");
}

  LogicalResult MatchAndUpdateConstraints(
      Operation* op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final;

 private:
  // A filter for operations that are supported.
  std::function<bool(Operation*)> filter_;
  // Default constraint for all operands.
  llvm::Optional<ValueConstraint> default_constraint_;
};

template <typename OpTy>
class OpDefaultClusteringPolicy : public DefaultClusteringPolicy {
 public:
  explicit OpDefaultClusteringPolicy(
      llvm::Optional<ValueConstraint> default_constraint = llvm::None)
      : DefaultClusteringPolicy(
            [](Operation* op) -> bool { return mlir::isa<OpTy>(op); },
            default_constraint) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_6(mht_6_v, 346, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "OpDefaultClusteringPolicy");
}
};

LogicalResult DefaultClusteringPolicy::MatchAndUpdateConstraints(
    Operation* op, const ValuesConstraintSet& results,
    ValuesConstraintSet& operands) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_7(mht_7_v, 354, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "DefaultClusteringPolicy::MatchAndUpdateConstraints");

  if (!filter_(op)) return failure();

  if (!IsSupportedOperandAndResultTypes(op)) return failure();

  // Find the most restrictive constraint from the operation results.
  llvm::Optional<ValueConstraint> default_constraint = default_constraint_;

  for (mlir::Value result : op->getResults()) {
    if (auto result_constraint = results.GetConstraint(result)) {
      // TODO(ezhulenev): We can safely propagate value constraints if we know
      // that the value is an integer scalar or a small vector, however in
      // practice all values that we are interested in are defined by constant
      // operations directly. Revisit if this becomes a problem.
      if (*result_constraint == ValueConstraint::kValue) return failure();

      default_constraint = default_constraint.hasValue()
                               ? Merge(*default_constraint, *result_constraint)
                               : *result_constraint;
    }
  }

  // No constraints to propagate.
  if (!default_constraint.hasValue()) return success();

  // Propage constraint to all operands.
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    operands.Insert(op->getOperand(i), *default_constraint);
  return success();
}

// -------------------------------------------------------------------------- //
// tf.BatchMatMulV2
// -------------------------------------------------------------------------- //

class BatchMatMulV2OpClusteringPolicy
    : public OpDefaultClusteringPolicy<BatchMatMulV2Op> {};

// -------------------------------------------------------------------------- //
// tf.BroadcastTo
// -------------------------------------------------------------------------- //

class BroadcastToOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<BroadcastToOp> {
  LogicalResult MatchAndUpdateConstraints(
      BroadcastToOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_8(mht_8_v, 403, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Only ranked inputs are supported.
    operands.Insert(op.input(), ValueConstraint::kRank);

    if (auto result_constraint = results.GetConstraint(op.getResult())) {
      if (*result_constraint == ValueConstraint::kValue) return failure();
      // For a static output shape we need a constant shape operand.
      if (*result_constraint == ValueConstraint::kShape) {
        operands.Insert(op.shape(), ValueConstraint::kValue);
        return success();
      }
    }

    // Producing a ranked output requires a known shape for the shape operand.
    operands.Insert(op.shape(), ValueConstraint::kShape);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// Cwise Binary Operations.
// -------------------------------------------------------------------------- //

class CwiseBinaryOpClusteringPolicy : public DefaultClusteringPolicy {
 public:
  CwiseBinaryOpClusteringPolicy()
      : DefaultClusteringPolicy(IsBinaryOp(), ValueConstraint::kRank) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_9(mht_9_v, 433, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "CwiseBinaryOpClusteringPolicy");
}

 private:
  // TODO(ezhulenev): Use mlir::isa<>() to filter operations.
  std::function<bool(Operation* op)> IsBinaryOp() {
    llvm::StringSet<> binary_ops = {
        "tf.Add",
        "tf.AddV2",
        "tf.ApproximateEqual",
        "tf.Atan2",
        "tf.BiasAdd",
        "tf.BitwiseAnd",
        "tf.BitwiseOr",
        "tf.BitwiseXor",
        "tf.Div",
        "tf.DivNoNan",
        "tf.Equal",
        "tf.FloorDiv",
        "tf.FloorMod",
        "tf.Greater",
        "tf.GreaterEqual",
        "tf.Less",
        "tf.LessEqual",
        "tf.LogicalAnd",
        "tf.LogicalOr",
        "tf.Maximum",
        "tf.Minimum",
        "tf.Mod",
        "tf.Mul",
        "tf.MulNoNan",
        "tf.NotEqual",
        "tf.Pow",
        "tf.RealDiv",
        "tf.SquaredDifference",
        "tf.Sub",
        "tf.TruncateDiv",
        "tf.Xdivy",
        "tf.Xlogy",
    };
    return [binary_ops = std::move(binary_ops)](Operation* op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_10(mht_10_v, 475, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "lambda");

      return binary_ops.contains(op->getName().getStringRef());
    };
  }
};

// -------------------------------------------------------------------------- //
// Cwise Unary Operations.
// -------------------------------------------------------------------------- //

class CwiseUnaryOpClusteringPolicy : public DefaultClusteringPolicy {
 public:
  CwiseUnaryOpClusteringPolicy()
      : DefaultClusteringPolicy(IsUnaryOp(), ValueConstraint::kRank) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_11(mht_11_v, 491, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "CwiseUnaryOpClusteringPolicy");
}

 private:
  std::function<bool(Operation* op)> IsUnaryOp() {
    // TODO(ezhulenev): Use mlir::isa<>() to filter operations.
    llvm::StringSet<> unary_ops = {
        "tf.Abs",      "tf.Acos",        "tf.Acosh",      "tf.Asin",
        "tf.Asinh",    "tf.Atan",        "tf.Atanh",      "tf.Cast",
        "tf.Ceil",     "tf.ClipByValue", "tf.ComplexAbs", "tf.Conj",
        "tf.Cos",      "tf.Cosh",        "tf.Elu",        "tf.Erf",
        "tf.Exp",      "tf.Floor",       "tf.Inv",        "tf.Invert",
        "tf.IsFinite", "tf.IsInf",       "tf.IsNan",      "tf.LeakyRelu",
        "tf.Log",      "tf.Log1p",       "tf.LogicalNot", "tf.Neg",
        "tf.Real",     "tf.Reciprocal",  "tf.Relu",       "tf.Relu6",
        "tf.Rint",     "tf.Round",       "tf.Rsqrt",      "tf.Selu",
        "tf.Sigmoid",  "tf.Sign",        "tf.Sin",        "tf.Sinh",
        "tf.Softplus", "tf.Softsign",    "tf.Sqrt",       "tf.Square",
        "tf.Tan",      "tf.Tanh",        "tf.ZerosLike",
    };
    return [unary_ops = std::move(unary_ops)](Operation* op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_12(mht_12_v, 513, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "lambda");

      return unary_ops.contains(op->getName().getStringRef());
    };
  }
};

// -------------------------------------------------------------------------- //
// Cwise Ternary Operations.
// -------------------------------------------------------------------------- //

class CwiseTernaryOpClusteringPolicy : public DefaultClusteringPolicy {
 public:
  CwiseTernaryOpClusteringPolicy()
      : DefaultClusteringPolicy(IsTernaryOp(), ValueConstraint::kRank) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_13(mht_13_v, 529, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "CwiseTernaryOpClusteringPolicy");
}

 private:
  std::function<bool(Operation* op)> IsTernaryOp() {
    return [](Operation* op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_14(mht_14_v, 536, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "lambda");

      return mlir::isa<mlir::TF::SelectOp, mlir::TF::SelectV2Op>(op);
    };
  }
};

// -------------------------------------------------------------------------- //
// Reduction Operations.
// -------------------------------------------------------------------------- //

// Clustering policy for Tensorflow reduction operations:
//   - shape constraint can be propagated from the result to the input
//   - reduction indices value must be known at compile time
//
// All operations that use this policy must have two operands (input and
// reduction indices) and a single result.
class ReductionOpClusteringPolicy : public ClusteringPolicy {
 public:
  LogicalResult MatchAndUpdateConstraints(
      Operation* op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final;

 private:
  bool IsSupported(Operation* op) const;
};

LogicalResult ReductionOpClusteringPolicy::MatchAndUpdateConstraints(
    Operation* op, const ValuesConstraintSet& results,
    ValuesConstraintSet& operands) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_15(mht_15_v, 567, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "ReductionOpClusteringPolicy::MatchAndUpdateConstraints");

  // Verify that the operation is a reduction with supported operands
  // and results data types.
  if (!IsSupported(op) || !IsSupportedOperandAndResultTypes(op))
    return failure();

  assert(op->getNumOperands() == 2 && "expected two operands");
  assert(op->getNumResults() == 1 && "expected one result");

  // Propagate constraint from the result to the input.
  if (auto result_constraint = results.GetConstraint(op->getResult(0))) {
    if (*result_constraint == ValueConstraint::kValue) return failure();
    operands.Insert(op->getOperand(0), *result_constraint);
  } else {
    operands.Insert(op->getOperand(0), ValueConstraint::kRank);
  }

  // Reduction indices must be known at compile time.
  operands.Insert(op->getOperand(1), ValueConstraint::kValue);

  return success();
}

bool ReductionOpClusteringPolicy::IsSupported(Operation* op) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_16(mht_16_v, 593, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "ReductionOpClusteringPolicy::IsSupported");

  return mlir::isa<mlir::TF::AllOp,   //
                   mlir::TF::AnyOp,   //
                   mlir::TF::MaxOp,   //
                   mlir::TF::MeanOp,  //
                   mlir::TF::MinOp,   //
                   mlir::TF::ProdOp,  //
                   mlir::TF::SumOp>(op);
}

// -------------------------------------------------------------------------- //
// tf.ConcatV2
// -------------------------------------------------------------------------- //

class ConcatV2OpClusteringPolicy
    : public TensorflowOpClusteringPolicy<ConcatV2Op> {
  LogicalResult MatchAndUpdateConstraints(
      ConcatV2Op op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_17(mht_17_v, 614, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    auto result_constraint = results.GetConstraint(op->getResult(0));
    if (result_constraint && *result_constraint == ValueConstraint::kValue)
      return failure();

    // Propagate constraint from the result to the input. All inputs always need
    // a known rank.
    for (auto value : op.values()) {
      operands.Insert(value,
                      result_constraint.getValueOr(ValueConstraint::kRank));
    }

    // Force axis to be a constant.
    operands.Insert(op.axis(), ValueConstraint::kValue);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Const
// -------------------------------------------------------------------------- //

class ConstOpClusteringPolicy : public TensorflowOpClusteringPolicy<ConstOp> {
  LogicalResult MatchAndUpdateConstraints(
      ConstOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_18(mht_18_v, 643, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // We cluster constant operation only if it is required to resolve some of
    // the constraints.
    auto result_constraint = results.GetConstraint(op.getResult());
    if (!result_constraint.hasValue()) return failure();

    return IsCompilableConstant(op.value());
  }
};

// -------------------------------------------------------------------------- //
// tf.ExpandDims
// -------------------------------------------------------------------------- //

class ExpandDimsOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<ExpandDimsOp> {
  LogicalResult MatchAndUpdateConstraints(
      ExpandDimsOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_19(mht_19_v, 664, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Propagate constraint from the result to the input.
    if (auto result_constraint = results.GetConstraint(op->getResult(0))) {
      if (*result_constraint == ValueConstraint::kValue) return failure();
      operands.Insert(op.input(), *result_constraint);
    } else {
      operands.Insert(op.input(), ValueConstraint::kRank);
    }

    // The inserted dimension must be always known at compile time.
    operands.Insert(op.dim(), ValueConstraint::kValue);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf._FusedMatMul
// -------------------------------------------------------------------------- //

class FusedMatMulOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<_FusedMatMulOp> {
  LogicalResult MatchAndUpdateConstraints(
      _FusedMatMulOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_20(mht_20_v, 691, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Check if the default policy accepts the operation.
    OpDefaultClusteringPolicy<_FusedMatMulOp> default_policy;
    if (failed(default_policy.MatchAndUpdateConstraints(op, results, operands)))
      return failure();

    // Check if we do support a set of fused operations.
    size_t n = op.fused_ops().size();

    auto fusion =
        n > 0 ? op.fused_ops()[0].dyn_cast<mlir::StringAttr>() : nullptr;
    auto activation =
        n > 1 ? op.fused_ops()[1].dyn_cast<mlir::StringAttr>() : nullptr;

    if ((n > 0 && !fusion) || (n > 1 && !activation)) return failure();

    // TODO(ezhulenev): Update fission pass to support more fusions and
    // activations.

    // We only support BiasAdd fusion ...
    if (fusion && fusion.getValue() != "BiasAdd") return failure();

    // ... with Relu activation.
    if (activation && activation.getValue() != "Relu") return failure();

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Fill
// -------------------------------------------------------------------------- //

class FillOpClusteringPolicy : public TensorflowOpClusteringPolicy<FillOp> {
  LogicalResult MatchAndUpdateConstraints(
      FillOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_21(mht_21_v, 730, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Fill operation does not have any default constraints.
    auto result_constraint = results.GetConstraint(op->getResult(0));
    if (!result_constraint.hasValue()) return success();

    // To know the result shape we need to know the shape operand value.
    if (*result_constraint == ValueConstraint::kShape)
      operands.Insert(op.dims(), ValueConstraint::kValue);

    // To know the result rank we need to know the shape operand shape.
    if (*result_constraint == ValueConstraint::kRank)
      operands.Insert(op.dims(), ValueConstraint::kShape);

    // Value constraint propagation is not supported.
    if (*result_constraint == ValueConstraint::kValue) return failure();

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.MatMul
// -------------------------------------------------------------------------- //

class MatMulOpClusteringPolicy : public OpDefaultClusteringPolicy<MatMulOp> {};

// -------------------------------------------------------------------------- //
// tf.OneHot
// -------------------------------------------------------------------------- //

class OneHotOpClusteringPolicy : public TensorflowOpClusteringPolicy<OneHotOp> {
  LogicalResult MatchAndUpdateConstraints(
      OneHotOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_22(mht_22_v, 766, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Value constraint propagation is not supported.
    if (auto constraint = results.GetConstraint(op.getResult()))
      if (*constraint == ValueConstraint::kValue) return failure();

    // MHLO lowering needs a static shape for the indices and a constant depth.
    operands.Insert(op.indices(), ValueConstraint::kShape);
    operands.Insert(op.depth(), ValueConstraint::kValue);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Pack
// -------------------------------------------------------------------------- //

class PackOpClusteringPolicy : public OpDefaultClusteringPolicy<PackOp> {};

// -------------------------------------------------------------------------- //
// tf.Range
// -------------------------------------------------------------------------- //

class RangeOpClusteringPolicy : public TensorflowOpClusteringPolicy<RangeOp> {
  LogicalResult MatchAndUpdateConstraints(
      RangeOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_23(mht_23_v, 795, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Range operation does not have any default constraints.
    auto result_constraint = results.GetConstraint(op.getResult());
    if (!result_constraint.hasValue()) return success();

    // To know the result shape we need the input values.
    if (*result_constraint == ValueConstraint::kShape) {
      operands.Insert({op.start(), op.limit(), op.delta()},
                      ValueConstraint::kValue);
    }

    // Value constraint propagation is not supported.
    if (*result_constraint == ValueConstraint::kValue) return failure();

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Reshape
// -------------------------------------------------------------------------- //

class ReshapeOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<ReshapeOp> {
  LogicalResult MatchAndUpdateConstraints(
      ReshapeOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_24(mht_24_v, 824, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // The runtime only supports ranked tensors.
    operands.Insert(op.tensor(), ValueConstraint::kRank);

    // Reshape operation does not have any default constraints.
    auto result_constraint = results.GetConstraint(op.getResult());
    if (!result_constraint.hasValue()) return success();

    // To know the result shape we need to know the shape operand value. We also
    // require a static shape on the input in case there's a -1 in the shape.
    if (*result_constraint == ValueConstraint::kShape) {
      operands.Insert(op.shape(), ValueConstraint::kValue);
      operands.Insert(op.tensor(), ValueConstraint::kShape);
    }

    // To know the result rank we need to know the shape operand shape.
    if (*result_constraint == ValueConstraint::kRank)
      operands.Insert(op.shape(), ValueConstraint::kShape);

    // Value constraint propagation is not supported.
    if (*result_constraint == ValueConstraint::kValue) return failure();

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Shape
// -------------------------------------------------------------------------- //

class ShapeOpClusteringPolicy : public TensorflowOpClusteringPolicy<ShapeOp> {
  LogicalResult MatchAndUpdateConstraints(
      ShapeOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_25(mht_25_v, 860, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Unranked inputs aren't supported by JitRt.
    operands.Insert(op.input(), ValueConstraint::kRank);

    // Check constraint on the result value.
    auto result_constraint = results.GetConstraint(op.getResult());
    if (!result_constraint.hasValue()) return success();

    // To know the result shape we need only the rank of the input.
    if (*result_constraint == ValueConstraint::kShape)
      operands.Insert(op.input(), ValueConstraint::kRank);

    // To know the result value we need to know the shape of the input.
    if (*result_constraint == ValueConstraint::kValue)
      operands.Insert(op.input(), ValueConstraint::kShape);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Softmax
// -------------------------------------------------------------------------- //

class SoftmaxOpClusteringPolicy : public DefaultClusteringPolicy {
 public:
  SoftmaxOpClusteringPolicy()
      : DefaultClusteringPolicy(IsSoftmaxOp(), ValueConstraint::kRank) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_26(mht_26_v, 890, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "SoftmaxOpClusteringPolicy");
}

 private:
  std::function<bool(Operation* op)> IsSoftmaxOp() {
    return [](Operation* op) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_27(mht_27_v, 897, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "lambda");

      return mlir::isa<mlir::TF::SoftmaxOp, mlir::TF::LogSoftmaxOp>(op);
    };
  }
};

// -------------------------------------------------------------------------- //
// tf.Squeeze
// -------------------------------------------------------------------------- //

class SqueezeOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<SqueezeOp> {
  LogicalResult MatchAndUpdateConstraints(
      SqueezeOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_28(mht_28_v, 914, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Propagate static shape constraints.
    auto input_constraint = ValueConstraint::kRank;
    if (auto result_constraint = results.GetConstraint(op.getResult())) {
      if (*result_constraint == ValueConstraint::kValue) return failure();
      input_constraint = *result_constraint;
    }

    // If squeeze_dims is not present we need a static shape.
    if (op.squeeze_dims().empty()) input_constraint = ValueConstraint::kShape;

    operands.Insert(op.input(), input_constraint);
    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.StopGradient
// -------------------------------------------------------------------------- //

class StopGradientOpClusteringPolicy
    : public OpDefaultClusteringPolicy<StopGradientOp> {};

// -------------------------------------------------------------------------- //
// tf.Transpose
// -------------------------------------------------------------------------- //

class TransposeOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<TransposeOp> {
  LogicalResult MatchAndUpdateConstraints(
      TransposeOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_29(mht_29_v, 948, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Propagate result constraints to the input, at minimum require known rank.
    if (auto constraint = results.GetConstraint(op.getResult())) {
      operands.Insert(op.x(), *constraint);
    } else {
      operands.Insert(op.x(), ValueConstraint::kRank);
    }

    // Permutation must be always known at compile time.
    operands.Insert(op.perm(), ValueConstraint::kValue);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.Slice
// -------------------------------------------------------------------------- //

class SliceOpClusteringPolicy : public TensorflowOpClusteringPolicy<SliceOp> {
  LogicalResult MatchAndUpdateConstraints(
      SliceOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_30(mht_30_v, 973, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // Value constraint propagation is not supported.
    if (auto constraint = results.GetConstraint(op.getResult()))
      if (*constraint == ValueConstraint::kValue) return failure();

    // We must know the shape of the input.
    operands.Insert(op.input(), ValueConstraint::kShape);

    // Force begin and size to be constants. The restriction on begin could be
    // lifted if we know that there are no `-1` sizes.
    // TODO(kramerb): Revisit this when mhlo.real_dynamic_slice stabilizes.
    operands.Insert({op.begin(), op.size()}, ValueConstraint::kValue);

    return success();
  }
};

// -------------------------------------------------------------------------- //
// tf.StridedSlice
// -------------------------------------------------------------------------- //

class StridedSliceOpClusteringPolicy
    : public TensorflowOpClusteringPolicy<StridedSliceOp> {
  LogicalResult MatchAndUpdateConstraints(
      StridedSliceOp op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_31(mht_31_v, 1001, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "MatchAndUpdateConstraints");

    // We must know the shape of the input.
    operands.Insert(op.input(), ValueConstraint::kShape);

    // And values of operands that control the slice size.
    operands.Insert({op.begin(), op.end(), op.strides()},
                    ValueConstraint::kValue);

    return success();
  }
};

}  // namespace

void populateTfJitRtClusteringPolicies(ClusteringPolicySet& policies,
                                       JitRtClusteringTier tier) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_32(mht_32_v, 1019, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "populateTfJitRtClusteringPolicies");

  // Returns true if the given jitrt compilation tier is enabled.
  auto is_enabled = [&](JitRtClusteringTier requested) -> bool {
    return (static_cast<uint8_t>(tier) & static_cast<uint8_t>(requested)) ==
           static_cast<uint8_t>(requested);
  };

  if (is_enabled(JitRtClusteringTier::kCwise)) {
    policies.Add<CwiseBinaryOpClusteringPolicy,   //
                 CwiseUnaryOpClusteringPolicy,    //
                 CwiseTernaryOpClusteringPolicy,  //
                 StopGradientOpClusteringPolicy>();
  }

  if (is_enabled(JitRtClusteringTier::kTranspose)) {
    policies.Add<TransposeOpClusteringPolicy>();
  }

  if (is_enabled(JitRtClusteringTier::kReductions)) {
    policies.Add<ReductionOpClusteringPolicy>();
  }

  if (is_enabled(JitRtClusteringTier::kMetadata)) {
    policies.Add<ExpandDimsOpClusteringPolicy,  //
                 ReshapeOpClusteringPolicy,     //
                 ShapeOpClusteringPolicy,       //
                 SqueezeOpClusteringPolicy>();
  }

  if (is_enabled(JitRtClusteringTier::kAll)) {
    policies.Add<BatchMatMulV2OpClusteringPolicy,  //
                 BroadcastToOpClusteringPolicy,    //
                 ConcatV2OpClusteringPolicy,       //
                 FillOpClusteringPolicy,           //
                 FusedMatMulOpClusteringPolicy,    //
                 MatMulOpClusteringPolicy,         //
                 OneHotOpClusteringPolicy,         //
                 PackOpClusteringPolicy,           //
                 RangeOpClusteringPolicy,          //
                 SliceOpClusteringPolicy,          //
                 SoftmaxOpClusteringPolicy,        //
                 StridedSliceOpClusteringPolicy>();
  }
}

void populateTfJitRtConstraintsPolicies(ClusteringPolicySet& policies,
                                        JitRtClusteringTier tier) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_33(mht_33_v, 1068, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "populateTfJitRtConstraintsPolicies");

  populateTfJitRtClusteringPolicies(policies, tier);
  policies.Add<ConstOpClusteringPolicy>();
}

// -------------------------------------------------------------------------- //
// Helper functions.
// -------------------------------------------------------------------------- //

mlir::LogicalResult IsCompilableConstant(mlir::ElementsAttr value) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_34(mht_34_v, 1080, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsCompilableConstant");

  return success(value.getNumElements() <= 16 &&
                 value.getType().getElementType().isIntOrIndexOrFloat());
}

static bool IsI1Integer(Type type) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_35(mht_35_v, 1088, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsI1Integer");

  return mlir::getElementTypeOrSelf(type).isInteger(1);
}

static bool IsUnsignedInteger(Type type) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_36(mht_36_v, 1095, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "IsUnsignedInteger");

  return mlir::getElementTypeOrSelf(type).isUnsignedInteger();
}

mlir::LogicalResult VerifyCluster(const Cluster& cluster) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clusteringDTcc mht_37(mht_37_v, 1102, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.cc", "VerifyCluster");

  llvm::SmallDenseSet<Operation*> ops;
  for (Operation* op : cluster.operations) {
    auto inserted = ops.insert(op);
    assert(inserted.second && "clustered operations must be unique");
    (void)inserted;
  }

  // TODO(ezhulenev): Too large clusters with dynamic shapes can take a very
  // long time to compile. Skip them for now.
  if (ops.size() > 20) return failure();

  // TODO(ezhulenev): This is a temporary workaround to disable forming clusters
  // with known compilation problems.
  for (Operation* op : ops) {
    // TODO(b/205714705): Memory layout of `i1` data type is not defined, and
    // when vectorization is enabled it can lead to crashes.
    bool has_i1_integers = llvm::any_of(op->getOperandTypes(), IsI1Integer) ||
                           llvm::any_of(op->getResultTypes(), IsI1Integer);
    if (has_i1_integers && tensorflow::GetJitRtFlags().vectorize)
      return failure();

    // TODO(b/205905286): Unsigned integers support has a lot of gaps, and
    // similar to handling `i1` we need a type conversion to signless integers.
    bool has_unsigned_integers =
        llvm::any_of(op->getOperandTypes(), IsUnsignedInteger) ||
        llvm::any_of(op->getResultTypes(), IsUnsignedInteger);
    if (has_unsigned_integers) return failure();
  }

  for (auto& pair : cluster.constraints) {
    Value value = pair.getFirst();
    ValueConstraint constraint = pair.getSecond();

    // We can satisfy shape and rank constraints on the compiled function
    // operands.
    if (constraint == ValueConstraint::kRank ||
        constraint == ValueConstraint::kShape)
      continue;

    if (constraint == ValueConstraint::kValue &&
        tfrt::jitrt::SupportsValueSpecialization(value.getType()))
      continue;

    Operation* op = value.getDefiningOp();
    if (!op) return failure();  // we do not support block arguments

    // Operations defined inside the cluster will be constant folded before the
    // compilation. This property is guaranteed by the clustering policy.
    if (ops.contains(op)) continue;

    // Small constants will be sunk into the compiled function body.
    auto const_op = mlir::dyn_cast<mlir::TF::ConstOp>(op);
    if (!const_op || failed(IsCompilableConstant(const_op.value())))
      return failure();
  }

  return success();
}

}  // namespace tensorflow
