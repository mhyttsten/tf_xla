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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc() {
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

namespace mlir {
namespace mhlo {
namespace {

class LegalizeTF : public LegalizeTFBase<LegalizeTF> {
 public:
  explicit LegalizeTF(bool allow_partial_conversion, bool legalize_chlo,
                      llvm::Optional<StringRef> tf2xla_fallback_device_type,
                      bool prefer_tf2xla) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "LegalizeTF");

    allow_partial_conversion_ = allow_partial_conversion;
    legalize_chlo_ = legalize_chlo;
    prefer_tf2xla_ = prefer_tf2xla;
    use_tf2xla_fallback_ = tf2xla_fallback_device_type.hasValue();
    if (tf2xla_fallback_device_type.hasValue()) {
      device_type_ = tf2xla_fallback_device_type.getValue().str();
    }
  }
  /// Performs the lowering to XLA dialect.
  void runOnOperation() override;
};

// Emits debug information which includes the number of ops of each type which
// failed to legalize.
void EmitLegalizationErrors(Operation *op,
                            const DenseSet<Operation *> &nonlegalized_ops) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "EmitLegalizationErrors");

  // Track the legalization failures by mapping op name to information about
  // that failure: the number of unlegalized occurrences of the op, and one
  // example operation that failed.
  std::map<StringRef, std::pair<int, Operation *>> op_name_to_error_info;
  DenseSet<Operation *> error_ops;
  for (Operation *nonlegalized_op : nonlegalized_ops) {
    // Increment count of this legalization failure.
    StringRef op_name = nonlegalized_op->getName().getStringRef();
    // If this emplace is successful, it's the first time we've encountered
    // this op type. Initialize count to 0 so that after increment, it is 1.
    auto insertion_result = op_name_to_error_info.emplace(
        op_name, std::make_pair(0, nonlegalized_op));
    ++insertion_result.first->second.first;
  }
  std::vector<std::string> error_messages;
  error_messages.reserve(op_name_to_error_info.size());
  for (const auto &op_info : op_name_to_error_info) {
    error_messages.push_back(
        llvm::formatv("{0} (count: {1})", op_info.first, op_info.second.first));
  }
  Location loc = op->getLoc();
  emitError(loc) << "The following operations cannot be legalized: "
                 << llvm::join(error_messages, "; ")
                 << ". These legalization failure(s) may be due to missing TF "
                    "to HLO lowerings and/or unsupported attributes, etc.";
  // Emit more information about the missing ops. This error message
  // contains useful details beyond the op name (input and output shapes,
  // attributes, etc.).
  if (!VLOG_IS_ON(1) && nonlegalized_ops.size() != 1) {
    emitError(loc)
        << "Emitting more detail about one op that failed to legalize...";
  } else if (VLOG_IS_ON(1)) {
    emitError(loc) << "Emitting more detail about one of each type of op "
                      "that failed to legalize...";
  }
  for (const auto &op_info : op_name_to_error_info) {
    op_info.second.second->emitOpError() << "is not legalizable";
    if (!VLOG_IS_ON(1)) break;
  }
}

/// Returns ops that should use MLIR legalization only in the case of
/// prefer_tf2xla. All other ops not in this list should use XlaOpKernel
/// legalization only or not be legalized by the new bridge.
const llvm::DenseSet<mlir::TypeID> &MlirPreferredOps() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "MlirPreferredOps");

  // The static variable is a pointer in order to avoid destruction upon thread
  // termination.

  // clang-format off
  static const llvm::DenseSet<mlir::TypeID>* ops =
      new llvm::DenseSet<mlir::TypeID>{
    // Ops that are legalized in the old bridge using MlirXlaOpKernel
    TypeID::get<TF::AbsOp>(),
    TypeID::get<TF::AtanOp>(),
    TypeID::get<TF::AvgPool3DOp>(),
    TypeID::get<TF::BiasAddGradOp>(),
    TypeID::get<TF::CeilOp>(),
    TypeID::get<TF::CheckNumericsOp>(),
    TypeID::get<TF::ComplexOp>(),
    TypeID::get<TF::CosOp>(),
    TypeID::get<TF::DiagPartOp>(),
    TypeID::get<TF::DivOp>(),
    TypeID::get<TF::EinsumOp>(),
    TypeID::get<TF::ExpOp>(),
    TypeID::get<TF::Expm1Op>(),
    TypeID::get<TF::FakeQuantWithMinMaxArgsOp>(),
    TypeID::get<TF::FloorOp>(),
    TypeID::get<TF::GreaterEqualOp>(),
    TypeID::get<TF::IFFTOp>(),
    TypeID::get<TF::ImagOp>(),
    TypeID::get<TF::IsFiniteOp>(),
    TypeID::get<TF::IsInfOp>(),
    TypeID::get<TF::IsNanOp>(),
    TypeID::get<TF::LessEqualOp>(),
    TypeID::get<TF::LgammaOp>(),
    TypeID::get<TF::Log1pOp>(),
    TypeID::get<TF::LogicalOrOp>(),
    TypeID::get<TF::LogSoftmaxOp>(),
    TypeID::get<TF::MatrixBandPartOp>(),
    TypeID::get<TF::MaxPool3DGradOp>(),
    TypeID::get<TF::PreventGradientOp>(),
    TypeID::get<TF::RandomShuffleOp>(),
    TypeID::get<TF::RealOp>(),
    TypeID::get<TF::ReciprocalOp>(),
    TypeID::get<TF::ReluOp>(),
    TypeID::get<TF::Relu6Op>(),
    TypeID::get<TF::ReluGradOp>(),
    TypeID::get<TF::RsqrtOp>(),
    TypeID::get<TF::SelectOp>(),
    TypeID::get<TF::SigmoidOp>(),
    TypeID::get<TF::SignOp>(),
    TypeID::get<TF::SoftmaxOp>(),
    TypeID::get<TF::SqrtOp>(),
    TypeID::get<TF::SqrtGradOp>(),
    TypeID::get<TF::SquaredDifferenceOp>(),
    TypeID::get<TF::TanhOp>(),
    TypeID::get<TF::TanhGradOp>(),
    TypeID::get<TF::XlaDotOp>(),
    TypeID::get<TF::XlaDotV2Op>(),
    TypeID::get<TF::XlaDynamicSliceOp>(),
    TypeID::get<TF::XlaEinsumOp>(),
    TypeID::get<TF::XlaReduceWindowOp>(),
    TypeID::get<TF::XlaReplicaIdOp>(),
    TypeID::get<TF::XlaRngBitGeneratorOp>(),
    TypeID::get<TF::XlaSelectAndScatterOp>(),
    TypeID::get<TF::XlaSortOp>(),
    TypeID::get<TF::XlaVariadicReduceV2Op>(),
    TypeID::get<TF::XlaVariadicSortOp>(),
    TypeID::get<TF::XlogyOp>(),
    TypeID::get<TF::ZetaOp>(),

    // Ops that have no XlaOpKernel.
    TypeID::get<TF::RiscAddOp>(),
    TypeID::get<TF::RiscDotOp>(),

    // Const op has a simple legalization and it is much more efficient to lower
    // within MLIR.
    TypeID::get<TF::ConstOp>(),

    // AssertOp with string types are not supported by the fallback.
    TypeID::get<TF::AssertOp>(),

    // TF2XLA fallback pattern doesn't support these op as MLIR hlo builder
    // doesn't override the necessary builder methods. These ops have simple
    // lowering pattern so this should be safe.
    TypeID::get<TF::CrossReplicaSumOp>(),
    TypeID::get<TF::InfeedDequeueTupleOp>(),
    TypeID::get<TF::OutfeedEnqueueTupleOp>(),
    TypeID::get<TF::XlaShardingOp>(),

    // These ops have undetermined bugs, may not be legalizable with XlaOpKernel
    // legalization in TF2XLA fallback. By legalization with MLIR, we can fix
    // the bug. b/195583695 describes the motivation of this change.
    // See b/216355804 how to reproduce the bug regarding tf.RandomUniform Op
    // See b/216353817 how to reproduce the bug regarding tf.StridedSlice Op
    TypeID::get<TF::RandomUniformOp>(),
    TypeID::get<TF::StridedSliceOp>(),
  };
  // clang-format on
  return *ops;
}

// Patterns whose root op is in the set `include_ops` are moved from the set
// `from` to the returned set. This is used to partition patterns by op so they
// can be cleanly migrated from the old bridge to the MLIR bridge.
RewritePatternSet PatternsIncludeOps(
    RewritePatternSet &from, const llvm::DenseSet<mlir::TypeID> &include_ops) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_3(mht_3_v, 393, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "PatternsIncludeOps");

  RewritePatternSet to(from.getContext());
  // Filter NativePatterns.
  for (auto &pattern : from.getNativePatterns()) {
    Optional<OperationName> pat_op_name = pattern->getRootKind();
    // If the pattern does not have a specific operation, always include it,
    // If the pattern is in include_ops then include it.
    bool include =
        !pat_op_name ||
        include_ops.count(pat_op_name->getRegisteredInfo()->getTypeID());
    if (include) to.add(std::move(pattern));
  }

  // Don't filter PDLPatterns.
  to.add(std::move(from.getPDLPatterns()));

  return to;
}

/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
LogicalResult legalizeTF(Operation *op, bool allow_partial_conversion,
                         bool legalize_chlo,
                         llvm::Optional<StringRef> tf2xla_fallback_device_type,
                         bool prefer_tf2xla) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_4(mht_4_v, 422, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "legalizeTF");

  MLIRContext *context = op->getContext();
  RewritePatternSet legalize_lower_patterns(context);
  // Note that the `OperationConverter` orders patterns lexicographically by:
  // 1) Ascending legalization depth (i.e., minimum number of patterns necessary
  //    to arrive at conversion target). This requires relevant patterns to
  //    specify the list of ops generated by it which most of patterns
  //    implemented in C++ don't do so this comparison doesn't work in those
  //    cases.
  // 2) Descending pattern benefit.
  // 3) Op specific patterns over patterns with MatchAnyOpTypeTag.
  // 4) Order of patterns in `RewritePatternSet`.

  // Add TF->HLO legalization patterns.
  PopulateLegalizeTfPatterns(context, &legalize_lower_patterns);

  // Add TF->TF lowering patterns.
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &legalize_lower_patterns);

  if (tf2xla_fallback_device_type && prefer_tf2xla) {
    VLOG(1) << "TF to XLA legalization patterns are partitioned by op into "
               "either native MLIR legalization, or TF2XLA fallback "
               "legalzation, with a preference toward TF2XLA.";
  } else if (tf2xla_fallback_device_type) {
    VLOG(1) << "TF to XLA legalization patterns include all native patterns "
               "and TF2XLA fallback patterns.";
  } else {
    VLOG(1) << "TF to XLA legalization patterns are native patterns only.";
  }

  // Set patterns to legalize_lower_patters, where in the prefer_tf2xla case
  // only patterns whose ops are in the set MlirPreferredOps are kept.
  RewritePatternSet patterns =
      (tf2xla_fallback_device_type && prefer_tf2xla)
          ? PatternsIncludeOps(legalize_lower_patterns, MlirPreferredOps())
          : std::move(legalize_lower_patterns);

  if (tf2xla_fallback_device_type) {
    // Add TF->HLO legalization patterns via TF2XLA fallback.
    PopulateLegalizeTfWithTf2XlaPatterns(tf2xla_fallback_device_type.getValue(),
                                         patterns, context, prefer_tf2xla);
  }

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // CHLO first.
  if (legalize_chlo) {
    chlo::PopulateDecomposeChloPatterns(context, &patterns);
    chlo::PopulateChloBroadcastingPatterns(context, &patterns);
  }
  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  ConversionTarget target(*context);
  if (legalize_chlo) {
    target.addIllegalDialect<chlo::HloClientDialect>();
  } else {
    target.addLegalDialect<chlo::HloClientDialect>();
  }
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  if (!allow_partial_conversion) {
    // Fully qualify ReturnOp here as mhlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, FuncOp, ::mlir::func::ReturnOp>();
    DenseSet<Operation *> nonlegalized_ops;
    LogicalResult result = applyPartialConversion(
        op, target, std::move(patterns), &nonlegalized_ops);
    // In order to enforce that the conversion result is fully converted,
    // fail if there are any nonlegalized ops in the set.
    if (failed(result) || !nonlegalized_ops.empty()) {
      EmitLegalizationErrors(op, nonlegalized_ops);
      return failure();
    }
    return result;
  }

  return applyPartialConversion(op, target, std::move(patterns));
}

// Performs the lowering to XLA dialect.
void LegalizeTF::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSxla_legalize_tfDTcc mht_5(mht_5_v, 511, "", "./tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf.cc", "LegalizeTF::runOnOperation");

  llvm::Optional<StringRef> tf2xla_fallback_device_type = llvm::None;
  if (use_tf2xla_fallback_) {
    tf2xla_fallback_device_type = device_type_;
  }
  if (failed(legalizeTF(getOperation(), allow_partial_conversion_,
                        legalize_chlo_, tf2xla_fallback_device_type,
                        prefer_tf2xla_))) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion, bool legalize_chlo,
    llvm::Optional<StringRef> tf2xla_fallback_device_type, bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(allow_partial_conversion, legalize_chlo,
                                      tf2xla_fallback_device_type,
                                      prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
