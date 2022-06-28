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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc() {
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
#include <set>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

namespace mlir {
namespace TF {
namespace internal {

// The name prefix of Flex ops.
constexpr absl::string_view kFlexOpNamePrefix = "Flex";
// Don't fallback to Flex op if this attribute is set. This attribute is
// transient and is only used inside this pass. First, the pass looks for
// predefined patterns and set this attribute to ops in the patterns. Then,
// when parsing the function, if find ops with this attribute, the pass
// remove the attribute and skip further processing on those ops.
constexpr char kNoFallbackAttr[] = "no_fallback";
// TF Quantization modes. These constants are defined as char arrays so they
// can parsed by the pass option.
constexpr char kDefaultMode[] = "DEFAULT";
constexpr char kLegacyIntegerMode[] = "LEGACY_INTEGER";

// Checks if the operation is TF FakeQuant ops.
bool IsTfFakeQuantOp(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "IsTfFakeQuantOp");

  return llvm::isa<
      // clang-format off
      TF::FakeQuantWithMinMaxArgsOp,
      TF::FakeQuantWithMinMaxVarsOp,
      TF::FakeQuantWithMinMaxVarsPerChannelOp
      // clang-format on
      >(op);
}

// Checks if the operation is allowlisted in both modes. These ops are not
// quantizable but is necessary to make the conversion successful.
bool IsAlwaysAllowlistedOp(Operation *op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "IsAlwaysAllowlistedOp");

  return llvm::isa<
      // clang-format off
      // go/keep-sorted start
      TF::ConstOp,
      TF::IdentityOp,
      TF::PartitionedCallOp,
      TF::StatefulPartitionedCallOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

// LINT.IfChange
// The list of quantizable ops in the Legacy Integer mode.
ABSL_ATTRIBUTE_NOINLINE const std::set<std::string>
    &QuantizableOpsInLegacyMode() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "QuantizableOpsInLegacyMode");

  static const std::set<std::string> *legacy_op_list =
      new std::set<std::string>({
          // clang-format off
          // go/keep-sorted start
          TF::AbsOp::getOperationName().str(),
          TF::AddOp::getOperationName().str(),
          TF::AddV2Op::getOperationName().str(),
          TF::ArgMaxOp::getOperationName().str(),
          TF::AvgPoolOp::getOperationName().str(),
          TF::BiasAddOp::getOperationName().str(),
          TF::BucketizeOp::getOperationName().str(),
          TF::ConcatV2Op::getOperationName().str(),
          TF::Conv2DBackpropInputOp::getOperationName().str(),
          TF::Conv2DOp::getOperationName().str(),
          TF::DepthwiseConv2dNativeOp::getOperationName().str(),
          TF::FusedBatchNormV3Op::getOperationName().str(),
          TF::GatherV2Op::getOperationName().str(),
          TF::MatMulOp::getOperationName().str(),
          TF::MaxPoolOp::getOperationName().str(),
          TF::MaximumOp::getOperationName().str(),
          TF::MeanOp::getOperationName().str(),
          TF::MinimumOp::getOperationName().str(),
          TF::MulOp::getOperationName().str(),
          TF::PadOp::getOperationName().str(),
          TF::PadV2Op::getOperationName().str(),
          TF::Relu6Op::getOperationName().str(),
          TF::ReluOp::getOperationName().str(),
          TF::ReshapeOp::getOperationName().str(),
          TF::SoftmaxOp::getOperationName().str(),
          TF::SubOp::getOperationName().str(),
          TF::TransposeOp::getOperationName().str(),
          // go/keep-sorted end
          // clang-format on
      });
  return *legacy_op_list;
}

// The list of quantizable ops in the Default mode.
ABSL_ATTRIBUTE_NOINLINE const std::set<std::string>
    &QuantizableOpsInDefaultMode() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "QuantizableOpsInDefaultMode");

  static const std::set<std::string> *default_op_list =
      new std::set<std::string>({
          // clang-format off
          // go/keep-sorted start
          TF::BiasAddOp::getOperationName().str(),
          TF::Conv2DBackpropInputOp::getOperationName().str(),
          TF::Conv2DOp::getOperationName().str(),
          TF::DepthwiseConv2dNativeOp::getOperationName().str(),
          TF::FusedBatchNormV3Op::getOperationName().str(),
          TF::MatMulOp::getOperationName().str(),
          TF::Relu6Op::getOperationName().str(),
          TF::ReluOp::getOperationName().str(),
          // go/keep-sorted end
          // clang-format on
      });
  return *default_op_list;
}
// LINT.ThenChange(Google-internal path)

// Checks if the operation can be fused with bias.
inline bool IsFusibleWithBiasOp(Operation *op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_4(mht_4_v, 316, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "IsFusibleWithBiasOp");

  return llvm::isa<
      // clang-format off
      TF::MatMulOp,
      TF::Conv2DOp,
      TF::DepthwiseConv2dNativeOp,
      TF::Conv2DBackpropInputOp,
      TF::Conv3DOp,
      TF::Conv3DBackpropInputV2Op
      // clang-format on
      >(op);
}

// Creates the custom option of the Flex ops.
inline void CreateFlexOpCustomOptions(const std::string &op_name,
                                      const std::string &node_def_str,
                                      std::string &custom_option_buffer) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + op_name + "\"");
   mht_5_v.push_back("node_def_str: \"" + node_def_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_5(mht_5_v, 337, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "CreateFlexOpCustomOptions");

  auto flex_builder = absl::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(op_name);
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  custom_option_buffer.assign(flex_builder->GetBuffer().begin(),
                              flex_builder->GetBuffer().end());
}

// Creates ElementsAttr for custom option.
inline OpaqueElementsAttr CustomOptionForFlexOp(OpBuilder *builder,
                                                const std::string &content) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_6(mht_6_v, 354, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "CustomOptionForFlexOp");

  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

// Fallbacks ops that are not supported by TF Quantization to TFLite Flex ops.
class FallbackToFlexOps
    : public PassWrapper<FallbackToFlexOps, OperationPass<FuncOp>> {
 public:
  FallbackToFlexOps() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_7(mht_7_v, 369, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "FallbackToFlexOps");
}
  explicit FallbackToFlexOps(const std::string &mode) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("mode: \"" + mode + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_8(mht_8_v, 374, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "FallbackToFlexOps");
 mode_ = mode; }
  FallbackToFlexOps(const FallbackToFlexOps &other) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_9(mht_9_v, 378, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "FallbackToFlexOps");
 mode_ = other.mode_; }

  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_10(mht_10_v, 385, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "getArgument");
 return "quant-raise-flex-fallback"; }

  StringRef getDescription() const final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_11(mht_11_v, 390, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "getDescription");

    return "Fallback TF-Quantization-unsupported ops to TFLite Flex ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_12(mht_12_v, 397, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 private:
  // The mode of TF Quantization, might indicate different users/devices.
  Option<std::string> mode_{*this, "mode",
                            llvm::cl::desc("The mode of TF Quantization."),
                            llvm::cl::init("")};

  // Checks if the operation is allowlisted in the current mode.
  bool IsAllowListedOp(Operation *op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_13(mht_13_v, 411, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "IsAllowListedOp");

    std::string op_name = op->getName().getStringRef().str();
    if (IsAlwaysAllowlistedOp(op) || IsTfFakeQuantOp(op)) {
      return true;
    } else if (mode_ == kDefaultMode) {
      return QuantizableOpsInDefaultMode().count(op_name) > 0;
    } else if (mode_ == kLegacyIntegerMode) {
      return QuantizableOpsInLegacyMode().count(op_name) > 0;
    } else {
      mlir::emitError(getOperation().getLoc(), "Unregconized mode: " + mode_);
      signalPassFailure();
      return true;
    }
  }

  // Converts the operation to a TFLite Flex op.
  bool ConvertToFlexOp(Operation *op);
};

bool FallbackToFlexOps::ConvertToFlexOp(Operation *op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_14(mht_14_v, 433, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "FallbackToFlexOps::ConvertToFlexOp");

  tensorflow::StatusOr<std::unique_ptr<tensorflow::NodeDef>> node_def =
      tensorflow::ConvertTFDialectOpToNodeDef(
          op, /*name=*/"", /*ignore_unregistered_attrs=*/true);
  if (!node_def.ok()) {
    op->emitError("Failed to obtain TensorFlow NodeDef: " +
                  node_def.status().ToString());
    return false;
  }
  std::string node_def_str;
  if (!(*node_def)->SerializeToString(&node_def_str)) {
    op->emitError("Failed to serialize tensorflow NodeDef");
    return false;
  }
  std::string op_name = (*node_def)->op();

  OpBuilder builder(op);
  std::string flex_op_name = std::string(kFlexOpNamePrefix) + op_name;
  std::string custom_option_buffer;
  CreateFlexOpCustomOptions(op_name, node_def_str, custom_option_buffer);
  auto flex_op = builder.create<TFL::CustomOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(), flex_op_name,
      CustomOptionForFlexOp(&builder, custom_option_buffer));
  op->replaceAllUsesWith(flex_op);
  op->erase();
  return true;
}

// Sets the "no_fallback" attribute.
Value SetNoFallbackAttr(PatternRewriter &rewriter, Value val) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_15(mht_15_v, 465, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "SetNoFallbackAttr");

  val.getDefiningOp()->setAttr(kNoFallbackAttr, rewriter.getUnitAttr());
  return val;
}

// Returns true if the attr is a float attribute and be equal to value.
static bool FloatValueEquals(const Attribute &attr, double value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_16(mht_16_v, 474, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "FloatValueEquals");

  auto fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (fp_attr == nullptr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat &f) {
    return f.isExactlyValue(value);
  });
}

// Returns true if the rank of the value equals to the given rank.
bool RankEquals(Value value, int rank) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPSfallback_to_flex_opsDTcc mht_17(mht_17_v, 490, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_ops.cc", "RankEquals");

  auto rank_type = value.getType().template dyn_cast<RankedTensorType>();
  return (rank_type && rank_type.getRank() == rank);
}

#include "tensorflow/compiler/mlir/lite/quantization/tensorflow/fallback_to_flex_patterns.inc"

void FallbackToFlexOps::runOnOperation() {
  if (mode_.empty()) return;

  FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();

  // Convert binary ops to BiasAdd ops if possible.
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Convert unsupported ops to Flex ops.
  auto tf_dialect = ctx->getLoadedDialect<TF::TensorFlowDialect>();
  func.walk([&](Operation *op) {
    if (op->getDialect() != tf_dialect) return;
    if (IsAllowListedOp(op)) return;
    if (op->hasAttr(kNoFallbackAttr)) {
      op->removeAttr(kNoFallbackAttr);
      return;
    }
    if (!ConvertToFlexOp(op)) signalPassFailure();
  });
}
}  // namespace internal

std::unique_ptr<OperationPass<FuncOp>> CreateFallbackToFlexOpsPass(
    const std::string &mode) {
  return std::make_unique<internal::FallbackToFlexOps>(mode);
}

static PassRegistration<internal::FallbackToFlexOps> pass([] {
  return CreateFallbackToFlexOpsPass(/*mode=*/internal::kDefaultMode);
});

}  // namespace TF
}  // namespace mlir
