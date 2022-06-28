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
class MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc() {
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

// Fuse tf.Op + tf.BiasAdd and legalized to TOSA

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"

#define PASS_NAME "tosa-fuse-bias-tf"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class FuseBiasTF : public TosaFusebiasTFPassBase<FuseBiasTF> {
 public:
  explicit FuseBiasTF() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/tosa/transforms/fuse_bias_tf.cc", "FuseBiasTF");
}
  void runOnOperation() override;
};

struct ConvertTFBiasAddOp : public RewritePattern {
  explicit ConvertTFBiasAddOp(MLIRContext* context)
      : RewritePattern(TF::BiasAddOp::getOperationName(), 1, context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/mlir/tosa/transforms/fuse_bias_tf.cc", "ConvertTFBiasAddOp");
}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

// Replaces the following pattern:
//   %1 = tf.Conv2D (%ifm, %filter)
//   %2 = tf.BiasAdd(%1, %bias)
//   with
//   %1 = tosa.conv2d(%ifm, %filter, %bias)
//   This can also be done using the pair ot Pat<> options in
//   tf_optimize_patterns.td
//   However, this explicit code can handle both when the LHS or RHS is the
//   defining conv2d op.
// TODO: support other pattern. e.g. tf.DepthwiseConv2DNative

LogicalResult ConvertTFBiasAddOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/tosa/transforms/fuse_bias_tf.cc", "ConvertTFBiasAddOp::matchAndRewrite");

  auto tf_biasadd_op = cast<TF::BiasAddOp>(op);

  auto output_type =
      tf_biasadd_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  auto value = tf_biasadd_op.value();
  auto bias = tf_biasadd_op.bias();

  TF::Conv2DOp tf_conv2d_op =
      dyn_cast_or_null<TF::Conv2DOp>(value.getDefiningOp());

  if (!tf_conv2d_op) {
    return failure();
  }

  // Sanity check to confirm rhs() has the expected shape of bias
  auto filter_shape =
      tf_conv2d_op.filter().getType().dyn_cast<RankedTensorType>().getShape();

  auto bias_shape = bias.getType().dyn_cast<RankedTensorType>().getShape();

  // Bias dimension must match filter output channels, where tf.conv2d's filter
  // is [H, W, I, O]
  if (filter_shape.back() != bias_shape.back()) return failure();

  // Bias tensor that feeds into tosa.conv2d must be rank 1
  if (bias_shape.size() != 1) return failure();

  auto result = convertTFConv2DCommon(
      rewriter, op, output_type, tf_conv2d_op.input(), tf_conv2d_op.filter(),
      bias, tf_conv2d_op.strides(), tf_conv2d_op.dilations(),
      tf_conv2d_op.explicit_paddings(), tf_conv2d_op.padding(),
      tf_conv2d_op.data_format());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

void FuseBiasTF::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStosaPStransformsPSfuse_bias_tfDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/mlir/tosa/transforms/fuse_bias_tf.cc", "FuseBiasTF::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto* ctx = &getContext();
  auto func = getOperation();

  // Add the generated patterns to the list.
  patterns.add<ConvertTFBiasAddOp>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createFuseBiasTFPass() {
  return std::make_unique<FuseBiasTF>();
}

}  // namespace tosa

}  // namespace mlir
