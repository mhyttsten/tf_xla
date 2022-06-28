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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc() {
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

// This transformation pass applies quantization propagation on TFLite dialect.
#include <iterator>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"

// NOLINTNEXTLINE
static llvm::cl::list<std::string> quantize_allowlist(
    "tfl-test-quantize-allowlist", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of allowlisted functions to be "
                   "quantized. Only used in tests"),
    llvm::cl::CommaSeparated);

// NOLINTNEXTLINE
static llvm::cl::opt<bool> quantize_signed(
    "tfl-test-quantize-signed", llvm::cl::value_desc("bool"),
    llvm::cl::desc("signed inference type. Only used in tests"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> post_training_quantize(
    "tfl-test-post-training-quantize", llvm::cl::value_desc("bool"),
    llvm::cl::desc("enable post training quantization. Only used in tests"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> legacy_float_scale(
    "tfl-test-legacy-float-scale", llvm::cl::value_desc("bool"),
    llvm::cl::desc("calculate quantization scales in float instead of double"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> disable_per_channel(
    "tfl-disable-per-channel", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether disable per-channel quantized weights."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {

auto* tflite_quantizer_usage_stats = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/lite/quantization/transforms/stats",
    "The number of quantization pass invocations.", "path");

// Applies prepare quantization on the model in TFL dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass
    : public PassWrapper<PrepareQuantizePass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_0(mht_0_v, 271, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "getDependentDialects");

    registry
        .insert<TensorFlowLiteDialect, ::mlir::quant::QuantizationDialect>();
  }

 public:
  // Constructor used by the PassRegistration and enforce uint8 quantization.
  // This is only used by test.
  explicit PrepareQuantizePass() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_1(mht_1_v, 282, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass");

    quant_specs_.inference_type =
        quantize_signed ? tensorflow::DT_QINT8 : tensorflow::DT_QUINT8;
    quant_specs_.post_training_quantization = post_training_quantize;
    quant_specs_.legacy_float_scale = legacy_float_scale;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(const quant::QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_2(mht_2_v, 294, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_3(mht_3_v, 299, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-prepare-quantize";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_4(mht_4_v, 307, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Prepare TFL dialect for quantization";
  }

  void runOnOperation() override;

 private:
  // Set the quantization parameters of the input nodes. These parameters are
  // converted from the user specified input value ranges. The input nodes with
  // non-float tensor types will be skipped because they are not quantizable.
  // Return true if number of input nodes doesn't equal to that of the input
  // ranges.
  bool SetInputNodesQuantizationParams(FuncOp func);

  // The function might contain more stats ops than required, and it will
  // introduce requantize if the calibration stats have conflicts. This method
  // tries to remove all the redundant stats ops.
  bool RemoveRedundantStats(FuncOp func);

  // Verify the quantization specification is expected for quantizing the
  // current function.
  bool IsLegalQuantSpecs(FuncOp func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_5(mht_5_v, 332, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "IsLegalQuantSpecs");

    if (func.getName() == quant_specs_.target_func) {
      return func.getNumArguments() == quant_specs_.input_ranges.size();
    }
    return true;
  }

  // Get the min and max values from the quantization specification for the
  // current function and argument index. Uses default values if the function
  // is specified in the `quantize_allowlist`.
  std::pair<llvm::Optional<double>, llvm::Optional<double>>
  GetMinMaxValuesForArgument(llvm::StringRef func_name, int index) {
    if (func_name == quant_specs_.target_func) {
      return quant_specs_.input_ranges[index];
    } else {
      return {0.0, 255.0};
    }
  }

  // Apply some sanity check and report some warnings for those who don't follow
  // the best quantization practice. This also fixes some simple violations.
  void SanityCheckAndAdjustment(FuncOp func);

  // Whether the func contains Quantize ops. This is used to determine whether
  // to use the quantization parameters from the fixed output range property.
  bool ContainsQuantizeOps(FuncOp func);

  quant::QuantizationSpecs quant_specs_;
};

bool PrepareQuantizePass::SetInputNodesQuantizationParams(FuncOp func) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_6(mht_6_v, 365, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass::SetInputNodesQuantizationParams");

  StringRef func_name = func.getName();
  auto& target_func = quant_specs_.target_func;
  // Skip this function because it isn't the target function from the spec or
  // in the function while list.
  if (target_func != func_name &&
      !llvm::is_contained(quantize_allowlist, func_name)) {
    return false;
  }
  auto has_quantize_op = [&](const Value arg) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_7(mht_7_v, 377, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "lambda");

    return (arg.hasOneUse() &&
            llvm::isa<quant::QuantizeCastOp>(*arg.user_begin()));
  };

  bool need_to_set_input_nodes_quantization_params = false;
  for (const BlockArgument arg : func.getArguments()) {
    auto shaped = arg.getType().dyn_cast<ShapedType>();
    if (shaped && shaped.getElementType().isa<FloatType>() &&
        !has_quantize_op(arg)) {
      need_to_set_input_nodes_quantization_params = true;
      break;
    }
  }

  if (!need_to_set_input_nodes_quantization_params) {
    return false;
  }

  // If the validation fails, the pass should stop immediately.
  if (!IsLegalQuantSpecs(func)) {
    return true;
  }

  OpBuilder builder(func);
  bool is_signed = quant_specs_.IsSignedInferenceType();
  IntegerAttr num_bits =
      builder.getI32IntegerAttr(quant_specs_.GetQuantizationTypeWidth());
  BoolAttr narrow_range = builder.getBoolAttr(false);

  auto add_quantize_op = [&](Location loc, Type input_type, Block* block,
                             Block::iterator insertion_point, Value arg,
                             int i) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_8(mht_8_v, 412, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "lambda");

    if (auto shaped = input_type.dyn_cast<ShapedType>()) {
      if (shaped.getElementType().isa<FloatType>()) {
        // If there are existing quantize ops, they are from training and we
        // should respect them.
        if (has_quantize_op(arg)) {
          return;
        }

        auto min_max = GetMinMaxValuesForArgument(func_name, i);
        // The input min/max or mean/std are not specified, then skip.
        if (!min_max.first.hasValue() || !min_max.second.hasValue()) return;

        TypeAttr params = quant::GetQuantizedTypeAttr(
            builder, input_type,
            builder.getF64FloatAttr(min_max.first.getValue()),
            builder.getF64FloatAttr(min_max.second.getValue()),
            /*quant_dim=*/-1, num_bits, narrow_range, is_signed);
        builder.setInsertionPoint(block, insertion_point);
        auto q_op =
            builder.create<quant::QuantizeCastOp>(loc, params.getValue(), arg);
        auto dq_op = builder.create<quant::DequantizeCastOp>(loc, input_type,
                                                             q_op.getResult());
        arg.replaceAllUsesWith(dq_op.getResult());
        q_op.setOperand(arg);
      }
    }
  };

  for (int i = 0, e = func.getNumArguments(); i != e; ++i) {
    BlockArgument arg = func.getArgument(i);
    auto* arg_block = arg.getOwner();
    add_quantize_op(arg.getLoc(), arg.getType(), arg_block,
                    std::next(arg_block->begin(), i), arg, i);
  }

  return false;
}

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

bool PrepareQuantizePass::RemoveRedundantStats(FuncOp func) {
  return RemoveRedundantStatsOps(func, GetOpQuantSpec);
}

static Value Quantized(Operation* user) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_9(mht_9_v, 460, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "Quantized");

  if (auto q = llvm::dyn_cast_or_null<quant::QuantizeCastOp>(user)) {
    if (auto dq = llvm::dyn_cast_or_null<quant::DequantizeCastOp>(
            *q.getResult().user_begin())) {
      return dq.getResult();
    }
  }
  return {};
}

void PrepareQuantizePass::SanityCheckAndAdjustment(FuncOp func) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_10(mht_10_v, 473, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass::SanityCheckAndAdjustment");

  // If an op output has two users: one of them is a quantize op and another
  // one is returned directly, we decide to return the quantized result instead,
  // so this op can be quantized. This is only applied on the returned result
  // because the error will not be accumulated.

  func.walk([&](func::ReturnOp ret) {
    int i = 0;
    for (Value returned : ret.getOperands()) {
      llvm::SmallVector<Value, 4> quantized;
      for (auto user : returned.getUsers()) {
        if (auto q = Quantized(user)) {
          quantized.push_back(q);
        }
      }
      if (quantized.size() == 1) {
        ret.setOperand(i, quantized.front());
      }
      i++;
    }
  });

  // We prefer to placing quantization emulation ops on the results of the
  // concat ops.
  func.walk([&](ConcatenationOp concat) {
    if (concat.output().hasOneUse() &&
        Quantized(*concat.output().user_begin())) {
      return;
    }
    concat.emitWarning(
        "Missing quantization parameter on the output might introduce "
        "quantization error!");
  });

  // Check for  (Quant (Dequant $in), $qA) "qdq" pairs that couldn't be
  // eliminated at this point.  This only occurs for the pattern
  //      (Quant (Dequant (Quant $in, $qB)), $qA)   $qB != $qA
  // where the  qdq pair denotes a non-trivial requantization of an
  // already quantized value. Since this makes little sense (directly quantizing
  // (Quant $in, $qA) would introduce less quantization noise) the likely cause
  // is an minor error in constructing the original network model that
  // introduced back-to-back Fake Quantization operations. Hence: emit a
  // warning. N.b. at this point we're (teporarility) in the quantization
  // dialect (presumably enable re-use in xla etc) quant::*QuantizeCastOp
  // we're matching here.
  //
  func.walk([&](quant::QuantizeCastOp q_op) {
    // If up with end up with
    auto dq_op = dyn_cast_or_null<quant::DequantizeCastOp>(
        q_op.getOperand().getDefiningOp());
    if (!dq_op) {
      return;
    }
    auto dq_arg = dq_op.getOperand();

    if (!dq_arg.hasOneUse()) {
      // The initial quantization is used someplace else ... so it might be
      // reasonable for it to requantized for another purpose.
      // Ideally would want to still check whether requantization narrows
      // rather than widens the representation.
      return;
    }

    // Invariant:
    // isa<quant::QuantizeCastOp>(dq_arg.getDefiningOp()) -->
    // getdq_arg.getType() != q_op.getResult().getType()
    //
    // as otherwise qdq pair would have been optimized away.
    auto qd_arg_def_q_op =
        dyn_cast_or_null<quant::QuantizeCastOp>(dq_arg.getDefiningOp());
    if (!qd_arg_def_q_op) {
      return;
    }

    qd_arg_def_q_op.emitWarning()
        << " quantizer's output has another quantizer (" << q_op.getLoc()
        << ") as consumer - intentional?";
  });
}

bool PrepareQuantizePass::ContainsQuantizeOps(FuncOp func) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_11(mht_11_v, 556, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass::ContainsQuantizeOps");

  for (const auto& op : func.getOps()) {
    if (llvm::isa<quant::DequantizeCastOp>(op)) return true;
  }
  return false;
}

using PrepareQuantStats =
    quant::ConvertStatsToQDQs<quant::QuantizeCastOp, quant::DequantizeCastOp>;

void PrepareQuantizePass::runOnOperation() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSprepare_quantizeDTcc mht_12(mht_12_v, 569, "", "./tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc", "PrepareQuantizePass::runOnOperation");

  FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();
  ScopedTFLQuantOpsToMlirQuantOpsConverter converter(func);

  if (quant_specs_.post_training_quantization) {
    tflite_quantizer_usage_stats->GetCell("post_training")->IncrementBy(1);
    RemoveRedundantStats(func);
  } else {
    tflite_quantizer_usage_stats->GetCell("during_training")->IncrementBy(1);
    // Set the quantization parameters for the quantizable input nodes. If this
    // failed, return the function immediately. This is only required for
    // quantization aware training model conversion.
    if (SetInputNodesQuantizationParams(func)) {
      return;
    }
  }

  bool is_signed = quant_specs_.IsSignedInferenceType();
  int bit_width = quant_specs_.GetQuantizationTypeWidth();
  // When this is true, the quantizer will try its best to extract the
  // quantization parameters from the op quantization property and constant
  // content. This is also set to true when the `quantize_allowlist` and
  // `quantize_signed` test flags are enabled.
  bool eager_quantize = ContainsQuantizeOps(func) ||
                        (!quantize_allowlist.empty() || quantize_signed);
  // Infer the tensor range for the activation ops and weight constants unless
  // it is disabled explicitly.
  bool infer_tensor_range =
      (quant_specs_.post_training_quantization || eager_quantize) &&
      !quant_specs_.disable_infer_tensor_range;

  // LSTM's restrict_scale requirement should be handled before converting stats
  // to Q-DQ ops. The pattern is applied for non-PTQ case to make op ordering
  // consistent. Otherwise some FileCheck tests would fail.
  RewritePatternSet patterns_1(&getContext());
  if (quant_specs_.post_training_quantization) {
    patterns_1.add<PrepareLstmOutputScale<LSTMOp>>(ctx);
    patterns_1.add<PrepareLstmOutputScale<UnidirectionalSequenceLSTMOp>>(ctx);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_1));

  // During the legalization, unsigned quantized type is used, so we have to
  // convert all of them to signed.
  RewritePatternSet patterns_2(&getContext());
  if (is_signed) {
    patterns_2.add<quant::ConvertUnsignedToSigned<quant::QuantizeCastOp>>(ctx);
    // Convert quant stats to int8 quantization parameters.
    // Currently, only activation stats are imported, so narrow_range = false.
    patterns_2.add<PrepareQuantStats>(bit_width, false, true,
                                      quant_specs_.legacy_float_scale, ctx);
  } else {
    // Convert quant stats to uint8 quantization parameters.
    // Currently, only activation stats are imported, so narrow_range = false.
    patterns_2.add<PrepareQuantStats>(bit_width, false, false,
                                      quant_specs_.legacy_float_scale, ctx);
  }

  if (quant_specs_.post_training_quantization) {
    patterns_2.add<ConvertLstmStatsToQDQs<LSTMOp>>(ctx, quant_specs_);
    patterns_2.add<ConvertLstmStatsToQDQs<UnidirectionalSequenceLSTMOp>>(
        ctx, quant_specs_);
    patterns_2.add<ConvertSvdfStatsToQDQs>(ctx, quant_specs_);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));

  SanityCheckAndAdjustment(func);

  // Finally, the quantization parameters can be propagated to the rest of the
  // values (tensors).
  ApplyQuantizationParamsPropagation(
      func, is_signed, disable_per_channel || quant_specs_.disable_per_channel,
      GetOpQuantSpec, infer_tensor_range, quant_specs_.legacy_float_scale);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareQuantizePass(
    const quant::QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareQuantizePass>(quant_specs);
}

static PassRegistration<PrepareQuantizePass> pass;

}  // namespace TFL
}  // namespace mlir
