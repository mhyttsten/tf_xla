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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc() {
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

#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/lib/monitoring/gauge.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kAllowSoftPlacementAttr[] = "allow_soft_placement";

auto* auto_outside_compilation_gauge =
    tensorflow::monitoring::Gauge<bool, 0>::New(
        "/tensorflow/core/use_auto_outside_compilation",
        "Tracks if auto outside compilation is enabled");

struct MarkOpsForOutsideCompilation
    : public TF::MarkOpsForOutsideCompilationPassBase<
          MarkOpsForOutsideCompilation> {
  void runOnOperation() override;
};

// Adds any canonicalization patterns to list of supported `patterns`.
// TODO(b/161726307): Move or import the relevant patterns to LowerTF pass and
// remove this.
void AddCanonicalizationPatterns(MLIRContext* context,
                                 RewritePatternSet* patterns) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddCanonicalizationPatterns");

  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

// Adds the list of ops that are supported on TPU through constant folding which
// may depend on the inputs shapes not known at this point. Such ops may not
// have any legalization or canonicalization patterns but shouldn't be marked
// for outside compilation.
//
// TODO(b/177523289): Remove manual handling once we support constant folding
// and shape inference through the computation on the host side.
void AddSupportedOpsUsingFolding(MLIRContext* context,
                                 llvm::DenseSet<OperationName>* supported_ops) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddSupportedOpsUsingFolding");

  llvm::SmallDenseSet<OperationName, 8> allowlist_ops = {
      OperationName(TF::BroadcastArgsOp::getOperationName(), context),
      OperationName(TF::BroadcastGradientArgsOp::getOperationName(), context),
      OperationName(TF::ConcatOffsetOp::getOperationName(), context),
      OperationName(TF::EmptyOp::getOperationName(), context),
      OperationName(TF::ListDiffOp::getOperationName(), context),
      OperationName(TF::RankOp::getOperationName(), context),
      OperationName(TF::RangeOp::getOperationName(), context),
      OperationName(TF::ShapeOp::getOperationName(), context),
      OperationName(TF::ShapeNOp::getOperationName(), context),
      OperationName(TF::SizeOp::getOperationName(), context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// Adds the list of ops that are supported through dynamic padder using op by op
// fallback to the TF2XLA bridge.
// TODO(b/168036682): Remove this once ops are supported using dynamic padder
// on MLIR bridge.
void AddSupportedOpsUsingDynamicPadder(
    MLIRContext* context, llvm::DenseSet<OperationName>* supported_ops) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddSupportedOpsUsingDynamicPadder");

  llvm::SmallDenseSet<OperationName, 8> allowlist_ops = {
      OperationName(TF::WhereOp::getOperationName(), context),
      OperationName(TF::UniqueOp::getOperationName(), context),
      OperationName(TF::XlaSetDynamicDimensionSizeOp::getOperationName(),
                    context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// TODO(b/159128666): Check the control flow legalization passes instead once
// added.
void AddSupportedFunctionalOps(MLIRContext* context,
                               llvm::DenseSet<OperationName>* supported_ops) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddSupportedFunctionalOps");

  supported_ops->insert(
      OperationName(TF::CaseRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::IfRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::InplaceAddOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::WhileRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReduceOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReduceWindowOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaRngBitGeneratorOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaScatterOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaSelectAndScatterOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::SymbolicGradientOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicReduceOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicReduceV2Op::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicSortOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReplicaIdOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::YieldOp::getOperationName(), context));
}

// These embedding ops are rewritten when running TPUCompileOp.
void AddRewrittenEmbeddingOps(MLIRContext* context,
                              llvm::DenseSet<OperationName>* supported_ops) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_4(mht_4_v, 326, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddRewrittenEmbeddingOps");

  supported_ops->insert(OperationName(
      TF::RecvTPUEmbeddingActivationsOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      TF::SendTPUEmbeddingGradientsOp::getOperationName(), context));
}

// Stack, TensorList and TensorArray ops are rewritten during the second phase
// of the bridge (compilation of TPUCompile op). They would not match any
// legalization/canonicalization pattern and have to be manually added to the
// list of supported ops.
void AddRewrittenCompositeOps(MLIRContext* context,
                              llvm::DenseSet<OperationName>* supported_ops) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_5(mht_5_v, 341, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "AddRewrittenCompositeOps");

#define GET_OPERATION_NAME(op) OperationName(op::getOperationName(), context)
  llvm::SmallDenseSet<OperationName, 32> allowlist_ops = {
      // Stack ops.
      GET_OPERATION_NAME(TF::StackV2Op),
      GET_OPERATION_NAME(TF::StackPushV2Op),
      GET_OPERATION_NAME(TF::StackPopV2Op),
      // Tensor Array ops.
      GET_OPERATION_NAME(TF::TensorArrayV3Op),
      GET_OPERATION_NAME(TF::TensorArrayReadV3Op),
      GET_OPERATION_NAME(TF::TensorArrayWriteV3Op),
      GET_OPERATION_NAME(TF::TensorArrayConcatV3Op),
      GET_OPERATION_NAME(TF::TensorArraySplitV3Op),
      GET_OPERATION_NAME(TF::TensorArraySizeV3Op),
      GET_OPERATION_NAME(TF::TensorArrayGradV3Op),
      GET_OPERATION_NAME(TF::TensorArrayGatherV3Op),
      GET_OPERATION_NAME(TF::TensorArrayScatterV3Op),
      // Tensor List Ops.
      GET_OPERATION_NAME(TF::EmptyTensorListOp),
      GET_OPERATION_NAME(TF::TensorListReserveOp),
      GET_OPERATION_NAME(TF::TensorListFromTensorOp),
      GET_OPERATION_NAME(TF::TensorListPushBackOp),
      GET_OPERATION_NAME(TF::TensorListPopBackOp),
      GET_OPERATION_NAME(TF::TensorListGetItemOp),
      GET_OPERATION_NAME(TF::TensorListSetItemOp),
      GET_OPERATION_NAME(TF::TensorListLengthOp),
      GET_OPERATION_NAME(TF::TensorListElementShapeOp),
      GET_OPERATION_NAME(TF::TensorListGatherOp),
      GET_OPERATION_NAME(TF::TensorListScatterIntoExistingListOp),
      GET_OPERATION_NAME(TF::TensorListStackOp),
  };
#undef GET_OPERATION_NAME

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

bool IsStringType(Type type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_6(mht_6_v, 380, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "IsStringType");

  if (type.isa<TF::StringType>()) return true;

  auto sub_type = type.dyn_cast<TF::TensorFlowTypeWithSubtype>();
  if (!sub_type) return false;

  bool has_string = llvm::any_of(sub_type.GetSubtypes(), [](TensorType type) {
    return type.getElementType().isa<TF::StringType>();
  });
  return has_string;
}

bool HasStringOperand(Operation& op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_7(mht_7_v, 395, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "HasStringOperand");

  for (auto operand : op.getOperands()) {
    auto operand_type = getElementTypeOrSelf(operand);
    if (IsStringType(operand_type)) return true;
  }
  return false;
}

bool HasStringResult(Operation& op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_8(mht_8_v, 406, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "HasStringResult");

  for (auto result : op.getResults()) {
    auto result_type = getElementTypeOrSelf(result);
    if (IsStringType(result_type)) return true;
  }
  return false;
}

bool MatchesPattern(Operation& op,
                    const llvm::DenseSet<OperationName>& supported_ops) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_9(mht_9_v, 418, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "MatchesPattern");

  return (supported_ops.contains(op.getName()));
}

// Checks if the op is supported inside of a device cluster.  Ops not
// in `tf_dialect` are considered supported.
bool IsSupportedOp(Operation& op,
                   const llvm::DenseSet<OperationName>& supported_ops,
                   const Dialect* tf_dialect) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_10(mht_10_v, 429, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "IsSupportedOp");

  if (op.getDialect() != tf_dialect)
    return true;
  // Assert has a legalization that later removes it so we don't want to outside
  // compile it ever for performance reasons.
  if (llvm::isa<TF::AssertOp>(op)) return true;
  return !HasStringOperand(op) && !HasStringResult(op) &&
         (MatchesPattern(op, supported_ops) ||
          mhlo::IsOpAllowedTf2XlaFallback(&op));
}

// Checks all regions of `op` for captured string operands.
bool HasCapturedStringOperand(Operation* op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_11(mht_11_v, 444, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "HasCapturedStringOperand");

  bool string_operand = false;
  for (auto& region : op->getRegions()) {
    mlir::visitUsedValuesDefinedAbove(
        region, region, [&](mlir::OpOperand* operand) {
          if (getElementTypeOrSelf(operand->get()).isa<TF::StringType>())
            string_operand = true;
        });
    if (string_operand) return string_operand;
  }
  return string_operand;
}

bool IsVariant(Value value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_12(mht_12_v, 460, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "IsVariant");

  return getElementTypeOrSelf(value.getType()).isa<TF::VariantType>();
}

bool HasOutsideCompiledAncestor(Operation* op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_13(mht_13_v, 467, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "HasOutsideCompiledAncestor");

  Operation* parent = op->getParentOp();
  while (parent) {
    if (parent->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

// If any tf.variants are inputs/outputs to the another outside compiled
// Operation, `op`, mark  them for outside compilation unless they are already
// marks with outside compilation attribute.
void MarkVariantInputsOutputs(tf_device::ClusterOp tpu_cluster) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_14(mht_14_v, 483, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "MarkVariantInputsOutputs");

  std::queue<Operation*> outside_compiled_ops;
  tpu_cluster.walk([&](Operation* op) {
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      outside_compiled_ops.push(op);
  });

  while (!outside_compiled_ops.empty()) {
    Operation* host_op = outside_compiled_ops.front();
    outside_compiled_ops.pop();
    host_op->walk([&](Operation* op) {
      // Add any operations that provide variant inputs to the cluster.
      for (auto value : op->getOperands()) {
        Operation* input_defining_op = value.getDefiningOp();
        if (IsVariant(value) && input_defining_op &&
            !HasOutsideCompiledAncestor(input_defining_op) &&
            !input_defining_op->hasAttrOfType<StringAttr>(
                kXlaOutsideCompilationAttr)) {
          input_defining_op->setAttr(
              kXlaOutsideCompilationAttr,
              StringAttr::get(input_defining_op->getContext(), "auto"));
          outside_compiled_ops.push(input_defining_op);
        }
      }
      // Mark for outside compilation any operations that consume variant
      // outputs from an outside compiled operation.
      for (auto value : op->getResults()) {
        if (IsVariant(value)) {
          for (auto user : value.getUsers()) {
            if (!user->hasTrait<OpTrait::IsTerminator>() &&
                !HasOutsideCompiledAncestor(user) &&
                !user->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
              user->setAttr(kXlaOutsideCompilationAttr,
                            StringAttr::get(user->getContext(), "auto"));
              outside_compiled_ops.push(user);
            }
          }
        }
      }
    });
  }
}

// Marks uncompilable ops that are in `tf_dialect` for outside compilation.
LogicalResult MarkUncompilableOps(
    const Dialect* tf_dialect, Block* block,
    llvm::DenseSet<OperationName>& supported_ops) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_15(mht_15_v, 532, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "MarkUncompilableOps");

  // Automatically marked ops for outside compilation have
  // `_xla_outside_compilation` attribute value of "auto" plus
  // an increasing counter.  Manually marked ops for outside compilation only
  // have an increasing counteri for the attribute value.  Therefore there is no
  // collision in
  // `_xla_outside_compilation` attribute between automatically and manually
  // marking ops.
  int outside_compiled_cluster_counter = 0;
  block->walk([&](Operation* op) {
    if (!IsSupportedOp(*op, supported_ops, tf_dialect)) {
      VLOG(3) << "Cloud TPU: Op " << op->getName().getStringRef().str()
              << " isn't compilable, adding outside_compilation attr. "
                 "This op will automatically be placed on CPU.";
      op->setAttr(kXlaOutsideCompilationAttr,
                  StringAttr::get(
                      op->getContext(),
                      llvm::formatv("auto{0}", outside_compiled_cluster_counter)
                          .str()));
      outside_compiled_cluster_counter++;
    }
  });
  if (outside_compiled_cluster_counter > 0) {
    auto_outside_compilation_gauge->GetCell()->Set(true);
  }
  return success();
}

// Check for uncompilable ops that are in `tf_dialect` and are not already
// marked for outside compilation.
bool ContainsUncompilableOps(const Dialect* tf_dialect, Block* block,
                             llvm::DenseSet<OperationName>& supported_ops) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_16(mht_16_v, 566, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "ContainsUncompilableOps");

  int uncompilable_op_count = 0;
  // Check if op or any parent is already marked for outside compilation.
  block->walk([&](Operation* op) {
    Operation* iter_op = op;
    while (iter_op && !llvm::isa<tf_device::ClusterOp>(iter_op)) {
      if (iter_op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
        return;
      }
      iter_op = iter_op->getParentOp();
    }

    if (!IsSupportedOp(*op, supported_ops, tf_dialect)) {
      op->emitOpError() << "isn't compilable for TPU device. enable "
                           "soft_device_placement option to run on CPU";
      ++uncompilable_op_count;
    }
  });
  return uncompilable_op_count > 0;
}

// Unmarks outside compilation for any op that has parents already
// marked for outside compilation since the child will be extracted
// anyways.
void UnmarkChildren(Block* block) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_17(mht_17_v, 593, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "UnmarkChildren");

  block->walk([&](Operation* op) {
    if (!op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) return;
    Operation* iter_op = op;
    bool remove_attr = false;
    while (auto* parent_op = iter_op->getParentOp()) {
      if (parent_op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
        remove_attr = true;
        break;
      }
      iter_op = parent_op;
    }
    if (remove_attr) op->removeAttr(kXlaOutsideCompilationAttr);
  });
}

void MarkOpsForOutsideCompilation::runOnOperation() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSmark_ops_for_outside_compilationDTcc mht_18(mht_18_v, 612, "", "./tensorflow/compiler/mlir/tensorflow/transforms/mark_ops_for_outside_compilation.cc", "MarkOpsForOutsideCompilation::runOnOperation");

  auto module = getOperation();
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }
  RewritePatternSet patterns(&getContext());
  mhlo::PopulateLegalizeTfPatterns(module.getContext(), &patterns);
  TF::PopulateTFLoweringBeforeHLOPatterns(module.getContext(), &patterns);
  TF::PopulateLoweringQuantizedPatterns(module.getContext(), &patterns);
  AddCanonicalizationPatterns(module.getContext(), &patterns);

  // `supported_ops` contains the name of all of the ops that can potentially be
  // lowered into HLO on the device. This doesn't always mean that the op can
  // be lowered in the future passes but if the op is not in this set, it can't
  // be lowered in a subsequent pass.
  llvm::DenseSet<OperationName> supported_ops;
  PatternApplicator(std::move(patterns))
      .walkAllPatterns([&](const Pattern& pattern) {
        Optional<OperationName> root_kind = pattern.getRootKind();
        if (root_kind.hasValue()) supported_ops.insert(root_kind.getValue());
      });
  AddSupportedFunctionalOps(module.getContext(), &supported_ops);
  AddSupportedOpsUsingFolding(module.getContext(), &supported_ops);
  AddSupportedOpsUsingDynamicPadder(module.getContext(), &supported_ops);
  AddRewrittenEmbeddingOps(module.getContext(), &supported_ops);
  AddRewrittenCompositeOps(module.getContext(), &supported_ops);

  auto result = module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we mark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster->getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
    if ((soft_placement_attr && soft_placement_attr.getValue())) {
      if (failed(MarkUncompilableOps(tf_dialect, &cluster.GetBody(),
                                     supported_ops)))
        return WalkResult::interrupt();
    } else {
      if (ContainsUncompilableOps(tf_dialect, &cluster.GetBody(),
                                  supported_ops))
        return WalkResult::interrupt();
    }
    MarkVariantInputsOutputs(cluster);

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we unmark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster->getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
    if (!(soft_placement_attr && soft_placement_attr.getValue())) {
      return;
    }
    UnmarkChildren(&cluster.GetBody());
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOpsForOutsideCompilationPass() {
  return std::make_unique<MarkOpsForOutsideCompilation>();
}

}  // namespace TFDevice
}  // namespace mlir
