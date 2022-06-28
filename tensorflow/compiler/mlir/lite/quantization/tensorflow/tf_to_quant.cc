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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc() {
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
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// The pass to legalize the quantization emulation ops from TF.
//
namespace {

// Legalize TF quantization emulation ops to that in Quant ops dialect.
struct LegalizeTFToQuant
    : public PassWrapper<LegalizeTFToQuant, OperationPass<FuncOp>> {
  explicit LegalizeTFToQuant() = default;
  LegalizeTFToQuant(const LegalizeTFToQuant &) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/tf_to_quant.cc", "LegalizeTFToQuant");
}

  /// Performs the lowering to Quant ops dialect.
  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/tf_to_quant.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-to-quant";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/tf_to_quant.cc", "getDescription");

    // This is a brief description of the pass.
    return "Legalize TF to quant ops dialect";
  }
};

// Inserts a "tfl.quantize" and "tfl.dequantize" op pair (QDQs) after the
// "tf.FakeQuantWithMinMaxVarsOp" to be constant folded. Since the constant
// folding logic will use a "arith.constant" op to replace the
// "tf.FakeQuantWithMinMaxVarsOp", the "tfl.quantize" op is used to preserve
// the quantization parameters as a TypeAttr and "tfl.dequantize" op used to
// convert the output type to the next op. Here are the transformations:
//
// input   min cst       max cst          input   min cst       max cst
//  \       |             |                \       |             |
//   \  (tf.Identity) (tf.Identity)   =>    \  (tf.Identity) (tf.Identity)
//    \     |             |                  \     |             |
//       tf.FakeQuantWithMinMaxVars       tf.FakeQuantWithMinMaxVars
//                   |                                 |
//                                                tf.quantize
//                                                     |
//                                                tf.dequantize
//                                                     |
// If the input is a constant, the result pattern will eventually converted to
//
//            quant-emulated input
//                   |
//               tf.quantize
//                   |
//              tf.dequantize
//                   |
template <typename TFFakeQuantOp, bool PerAxis>
struct InsertQuantOpsAfterTFFakeQuantOp
    : public OpRewritePattern<TFFakeQuantOp> {
  using BaseType = InsertQuantOpsAfterTFFakeQuantOp<TFFakeQuantOp, PerAxis>;

  explicit InsertQuantOpsAfterTFFakeQuantOp<TFFakeQuantOp, PerAxis>(
      MLIRContext *ctx)
      : OpRewritePattern<TFFakeQuantOp>(ctx) {}

  LogicalResult matchAndRewrite(TFFakeQuantOp tf_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/tf_to_quant.cc", "matchAndRewrite");

    // We don't want to insert quantize/dequantize if the quantize op exists.
    auto res = tf_op.outputs();
    if (!res.hasOneUse() || isa<quant::QuantizeCastOp>(*res.user_begin()))
      return failure();

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.
    Value min = tf_op.min(), max = tf_op.max();
    DenseFPElementsAttr min_value, max_value;
    if (auto id1 = dyn_cast_or_null<TF::IdentityOp>(min.getDefiningOp())) {
      id1.replaceAllUsesWith(id1.input());
      min = tf_op.min();
      rewriter.eraseOp(id1);
    }
    if (auto id2 = dyn_cast_or_null<TF::IdentityOp>(max.getDefiningOp())) {
      id2.replaceAllUsesWith(id2.input());
      max = tf_op.max();
      rewriter.eraseOp(id2);
    }
    if (!matchPattern(min, m_Constant(&min_value))) return failure();
    if (!matchPattern(max, m_Constant(&max_value))) return failure();

    int quant_dim = -1;
    if (PerAxis) {
      // This is a special case that the quant_dim is the last dimensions
      // according to the tf.FakeQuantWithMinMaxPerChannel.
      quant_dim = res.getType().template cast<ShapedType>().getRank() - 1;
    }
    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPointAfter(tf_op.getOperation());
    IntegerAttr num_bits = rewriter.getI64IntegerAttr(tf_op.num_bits());
    BoolAttr narrow_range = rewriter.getBoolAttr(tf_op.narrow_range());
    Type res_type = tf_op.getType();
    TypeAttr qtype = quant::GetQuantizedTypeAttr(
        rewriter, res_type, min_value, max_value, quant_dim, num_bits,
        narrow_range, /*is_signed=*/true);
    if (!qtype) return failure();

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    Value value = tf_op.outputs();
    auto quantize = rewriter.create<quant::QuantizeCastOp>(
        tf_op.getLoc(), qtype.getValue(), value);
    auto dequantize = rewriter.create<quant::DequantizeCastOp>(
        tf_op.getLoc(), res_type, quantize.getResult());
    value.replaceAllUsesWith(dequantize);
    quantize.getOperation()->replaceUsesOfWith(dequantize, value);

    return success();
  }
};

using PreparePerTensorFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsOp, false>;

using PreparePerChannelFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsPerChannelOp,
                                     true>;

// TODO(fengliuai): add the support of the tf.QuantizeAndDequantize*
// legalization.

void LegalizeTFToQuant::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStensorflowPStf_to_quantDTcc mht_4(mht_4_v, 331, "", "./tensorflow/compiler/mlir/lite/quantization/tensorflow/tf_to_quant.cc", "LegalizeTFToQuant::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto *ctx = func.getContext();
  patterns.add<PreparePerTensorFakeQuant, PreparePerChannelFakeQuant>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

// Creates an instance of the TensorFlow dialect to QuantOps dialect pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeTFToQuantPass() {
  return std::make_unique<LegalizeTFToQuant>();
}

static PassRegistration<LegalizeTFToQuant> pass([] {
  return CreateLegalizeTFToQuantPass();
});

}  // namespace TF
}  // namespace mlir
