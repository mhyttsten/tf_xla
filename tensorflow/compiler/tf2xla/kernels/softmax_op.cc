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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific Ops for softmax.

#include "absl/strings/match.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

REGISTER_XLA_OP(Name("Softmax"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("LogSoftmax"), MlirXlaOpKernel);

std::pair<xla::XlaOp, xla::XlaOp> CrossEntropyWithLogits(
    XlaOpKernelContext* ctx, DataType type, xla::PrimitiveType xla_type,
    xla::XlaOp logits, xla::XlaOp labels) {
  const xla::XlaComputation& max_func = *ctx->GetOrCreateMax(type);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  xla::XlaBuilder* b = ctx->builder();
  // Find the max in each batch, resulting in a tensor of shape [batch]
  auto logits_max =
      xla::Reduce(logits, xla::MinValue(b, xla_type), max_func, {kClassDim});

  // Subtract the max in batch b from every element in batch b.
  // Broadcasts along the batch dimension.
  auto shifted_logits = xla::Sub(logits, logits_max, {kBatchDim});

  // exp(logits - max_logits)
  auto exp_shifted_logits = xla::Exp(shifted_logits);

  // sum_{class} (exp(logits - max_logits))
  const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
  auto converted =
      XlaHelpers::ConvertElementType(exp_shifted_logits, accumulation_type);
  auto reduce =
      xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
  auto sum_exp = XlaHelpers::ConvertElementType(reduce, type);

  // log(sum(exp(logits - max_logits)))
  auto log_sum_exp = xla::Log(sum_exp);

  // sum(-labels *
  //    ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
  // along classes
  // (The subtraction broadcasts along the batch dimension.)
  auto sub = xla::Sub(shifted_logits, log_sum_exp, {kBatchDim});
  // Make sure the multiplication doesn't result in -inf * 0.
  auto safe_sub = xla::Select(xla::Eq(labels, xla::ZerosLike(labels)),
                              xla::ZerosLike(sub), sub);
  auto mul = xla::Mul(xla::Neg(labels), safe_sub);

  auto sum = xla::Reduce(XlaHelpers::ConvertElementType(mul, accumulation_type),
                         XlaHelpers::Zero(b, accumulation_type),
                         *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
  auto loss = XlaHelpers::ConvertElementType(sum, type);

  // backprop: prob - labels, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  //     (where the division broadcasts along the batch dimension)
  xla::XlaOp backprop =
      xla::Sub(xla::Div(exp_shifted_logits, sum_exp, {kBatchDim}), labels);
  return {loss, backprop};
}

class SoftmaxXentWithLogitsOp : public XlaOpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc mht_0(mht_0_v, 272, "", "./tensorflow/compiler/tf2xla/kernels/softmax_op.cc", "SoftmaxXentWithLogitsOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc mht_1(mht_1_v, 277, "", "./tensorflow/compiler/tf2xla/kernels/softmax_op.cc", "Compile");

    const DataType type = input_type(0);
    const xla::PrimitiveType xla_type = ctx->input_xla_type(0);
    auto logits = ctx->Input(0);
    auto labels = ctx->Input(1);

    OP_REQUIRES_OK(ctx, BroadcastOpsToSame(&logits, &labels));

    xla::XlaOp loss, backprop;
    std::tie(loss, backprop) =
        CrossEntropyWithLogits(ctx, type, xla_type, logits, labels);
    ctx->SetOutput(0, loss);
    ctx->SetOutput(1, backprop);
  }
};

REGISTER_XLA_OP(Name("SoftmaxCrossEntropyWithLogits"), SoftmaxXentWithLogitsOp);

class SparseSoftmaxXentWithLogitsOp : public XlaOpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc mht_2(mht_2_v, 301, "", "./tensorflow/compiler/tf2xla/kernels/softmax_op.cc", "SparseSoftmaxXentWithLogitsOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsoftmax_opDTcc mht_3(mht_3_v, 306, "", "./tensorflow/compiler/tf2xla/kernels/softmax_op.cc", "Compile");

    const TensorShape logits_shape = ctx->InputShape(0);
    const TensorShape labels_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_shape),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(0) == labels_shape.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits_shape.DebugString(), " and labels shape ",
                    labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits_shape.DebugString()));

    int64_t batch_size = logits_shape.dim_size(0);
    int64_t depth = logits_shape.dim_size(1);

    const DataType logits_type = input_type(0);
    const xla::PrimitiveType xla_logits_type = ctx->input_xla_type(0);
    const DataType indices_type = input_type(1);

    xla::XlaOp indices = ctx->Input(1);

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp labels;
    OP_REQUIRES_OK(ctx,
                   XlaHelpers::OneHot(
                       builder, depth, /*axis=*/1, input_type(1), labels_shape,
                       indices, XlaHelpers::One(builder, logits_type),
                       XlaHelpers::Zero(builder, logits_type), &labels));

    // If any of the indices are out of range, we must populate the labels with
    // NaNs to obey the interface contract of
    // tf.nn.sparse_softmax_cross_entropy_with_logits.
    // Builds a vector of {batch_size} that is 0 if the index is in range, or
    // NaN otherwise; then add that vector to the labels to force out-of-range
    // values to NaNs.
    xla::XlaOp nan_or_zero = xla::Select(
        xla::And(xla::Le(XlaHelpers::Zero(builder, indices_type), indices),
                 xla::Lt(indices, XlaHelpers::IntegerLiteral(
                                      builder, indices_type, depth))),
        xla::Broadcast(XlaHelpers::Zero(builder, logits_type), {batch_size}),
        xla::Broadcast(XlaHelpers::FloatLiteral(builder, logits_type, NAN),
                       {batch_size}));
    labels = xla::Add(labels, nan_or_zero, {0});

    xla::XlaOp loss, backprop;
    std::tie(loss, backprop) = CrossEntropyWithLogits(
        ctx, logits_type, xla_logits_type, ctx->Input(0), labels);
    ctx->SetOutput(0, loss);
    ctx->SetOutput(1, backprop);
  }
};

REGISTER_XLA_OP(Name("SparseSoftmaxCrossEntropyWithLogits"),
                SparseSoftmaxXentWithLogitsOp);

}  // namespace
}  // namespace tensorflow
