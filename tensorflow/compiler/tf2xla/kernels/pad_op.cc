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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpad_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpad_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpad_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class PadOp : public XlaOpKernel {
 public:
  explicit PadOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpad_opDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/tf2xla/kernels/pad_op.cc", "PadOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSpad_opDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/tf2xla/kernels/pad_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape pad_shape = ctx->InputShape("paddings");
    const int dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(pad_shape) && pad_shape.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                pad_shape.DebugString()));
    OP_REQUIRES(
        ctx, dims == pad_shape.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            pad_shape.DebugString(), " ", input_shape.DebugString()));

    xla::XlaOp input = ctx->Input("input");
    if (dims == 0) {
      // Tensor is rank 0. Return it unchanged.
      ctx->SetOutput(0, input);
      return;
    }

    xla::Literal pad_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(
                            "paddings", &pad_literal,
                            xla::ValueInferenceMode::kUpperBound));

    xla::Literal padding_dynamism_literal;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamism("paddings", &padding_dynamism_literal));

    xla::PaddingConfig config;
    for (int i = 0; i < dims; ++i) {
      auto* dim = config.add_dimensions();
      int before = pad_literal.Get<int64_t>({i, 0});
      int after = pad_literal.Get<int64_t>({i, 1});
      OP_REQUIRES(ctx, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "Paddings must be non-negative: ", before, " ", after));
      dim->set_edge_padding_low(before);
      dim->set_edge_padding_high(after);
    }

    // PadV2 added a "constant_values" input that indicates the pad value.
    xla::XlaOp constant_values;
    xla::XlaOp pad;
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(ctx->InputShape("constant_values")),
          errors::InvalidArgument("constant_values must be a scalar."));
      pad = xla::Pad(input, ctx->Input("constant_values"), config);
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      pad = xla::Pad(input, zero, config);
    }

    for (int i = 0; i < dims; ++i) {
      bool low_pad_is_dynamic = padding_dynamism_literal.Get<bool>({i, 0});

      OP_REQUIRES(
          ctx, !low_pad_is_dynamic,
          errors::InvalidArgument("low_pad in Pad op has to be static."));
      bool high_pad_is_dynamic = padding_dynamism_literal.Get<bool>({i, 1});
      if (high_pad_is_dynamic) {
        // When we have
        // pad_width = MAX_WIDTH - size(t)
        // op = pad(t, /*high_pad=*/pad_width)
        // The bound of the result size should be MAX_WIDTH, instead of
        // `bound(t) + bound(pad_width)`
        //
        // We do this by analyzing the expression
        // size(op) = size(t) + MAX_WIDTH - size(t)
        // and leave value inference to analyze it.
        xla::XlaOp high_pad_size =
            xla::Slice(ctx->Input("paddings"), {i, 1}, {i + 1, 2}, {1, 1});
        high_pad_size = xla::Reshape(high_pad_size, {});
        high_pad_size = xla::ConvertElementType(high_pad_size, xla::S32);
        // Low pad has to be static.
        xla::XlaOp low_pad_size = xla::ConstantR0<int32>(
            ctx->builder(), pad_literal.Get<int64_t>({i, 0}));
        xla::XlaOp input_size = xla::GetDimensionSize(input, i);
        xla::XlaOp total_size = low_pad_size + input_size + high_pad_size;
        auto size_upper_bound_status_or =
            ctx->value_inference().AnalyzeConstant(
                total_size, xla::ValueInferenceMode::kUpperBound);
        OP_REQUIRES_OK(ctx, size_upper_bound_status_or.status());
        auto size_upper_bound =
            size_upper_bound_status_or.ValueOrDie().Get<int32>({});
        OP_REQUIRES(
            ctx, size_upper_bound.has_value(),
            errors::InvalidArgument(
                "Failed to infer upperbound of total size after padding."));
        // If we know a tighter upperbound, trim the output with the new
        // upperbound.
        pad = xla::SliceInDim(pad, 0, size_upper_bound.value(), 1, i);
        pad = xla::SetDimensionSize(pad, total_size, i);
      }
    }
    ctx->SetOutput(0, pad);
  }
};

REGISTER_XLA_OP(Name("Pad").CompileTimeConstantInput("paddings"), PadOp);
REGISTER_XLA_OP(Name("PadV2").CompileTimeConstantInput("paddings"), PadOp);

}  // namespace
}  // namespace tensorflow
