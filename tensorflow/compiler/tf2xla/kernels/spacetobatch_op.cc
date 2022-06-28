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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {
namespace {

void SpaceToBatch(XlaOpKernelContext* ctx, const xla::XlaOp& input,
                  DataType input_dtype, const TensorShape& input_tensor_shape,
                  absl::Span<const int64_t> block_shape,
                  const xla::Literal& paddings) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc", "SpaceToBatch");

  const int input_rank = input_tensor_shape.dims();
  const absl::InlinedVector<int64_t, 4> input_shape =
      input_tensor_shape.dim_sizes();
  const int block_rank = block_shape.size();

  OP_REQUIRES(
      ctx, input_rank >= 1 + block_rank,
      errors::InvalidArgument("input rank should be >= ", 1 + block_rank,
                              " instead of ", input_rank));
  absl::Span<const int64_t> remainder_shape(input_shape);
  remainder_shape.remove_prefix(1 + block_rank);

  OP_REQUIRES(
      ctx,
      paddings.shape().rank() == 2 &&
          block_rank == xla::ShapeUtil::GetDimension(paddings.shape(), 0) &&
          2 == xla::ShapeUtil::GetDimension(paddings.shape(), 1),
      errors::InvalidArgument("paddings should have shape [", block_rank,
                              ", 2] instead of ",
                              xla::ShapeUtil::HumanString(paddings.shape())));

  xla::XlaBuilder* b = ctx->builder();

  // 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
  //  input according to `paddings` to produce `padded` of shape `padded_shape`.
  xla::PaddingConfig padding_config;
  std::vector<int64_t> padded_shape(input_shape.begin(), input_shape.end());
  int64_t block_num_elems = 1LL;
  padding_config.add_dimensions();  // Don't pad the batch dimension.
  for (int i = 0; i < block_rank; ++i) {
    auto* dim = padding_config.add_dimensions();
    int64_t pad_start = paddings.Get<int64_t>({i, 0});
    int64_t pad_end = paddings.Get<int64_t>({i, 1});
    OP_REQUIRES(ctx, pad_start >= 0 && pad_end >= 0,
                errors::InvalidArgument("Paddings must be non-negative"));
    OP_REQUIRES(ctx, block_shape[i] >= 1,
                errors::InvalidArgument(
                    "All values in block_shape must be positive, got value, ",
                    block_shape[i], " at index ", i, "."));
    dim->set_edge_padding_low(pad_start);
    dim->set_edge_padding_high(pad_end);
    padded_shape[1 + i] += pad_start + pad_end;
    block_num_elems = MultiplyWithoutOverflow(block_num_elems, block_shape[i]);
  }
  // Don't pad the remainder dimensions.
  for (int i = 0; i < remainder_shape.size(); ++i) {
    padding_config.add_dimensions();
  }
  OP_REQUIRES(ctx, block_num_elems > 0,
              errors::InvalidArgument(
                  "The product of the block dimensions must be positive"));
  const int64_t batch_size = input_shape[0];
  const int64_t output_dim =
      MultiplyWithoutOverflow(batch_size, block_num_elems);
  if (output_dim < 0) {
    OP_REQUIRES(
        ctx, output_dim >= 0,
        errors::InvalidArgument("Negative output dimension size caused by "
                                "overflow when multiplying ",
                                batch_size, " and ", block_num_elems));
  }

  xla::XlaOp padded =
      xla::Pad(input, XlaHelpers::Zero(b, input_dtype), padding_config);

  // 2. Reshape `padded` to `reshaped_padded` of shape:
  //
  //      [batch] +
  //      [padded_shape[1] / block_shape[0],
  //        block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1],
  //       block_shape[M-1]] +
  //      remaining_shape
  std::vector<int64_t> reshaped_padded_shape(input_rank + block_rank);
  reshaped_padded_shape[0] = batch_size;
  for (int i = 0; i < block_rank; ++i) {
    OP_REQUIRES(ctx, padded_shape[1 + i] % block_shape[i] == 0,
                errors::InvalidArgument("padded_shape[", 1 + i,
                                        "]=", padded_shape[1 + i],
                                        " is not divisible by block_shape[", i,
                                        "]=", block_shape[i]));

    reshaped_padded_shape[1 + i * 2] = padded_shape[1 + i] / block_shape[i];
    reshaped_padded_shape[1 + i * 2 + 1] = block_shape[i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            reshaped_padded_shape.begin() + 1 + 2 * block_rank);

  xla::XlaOp reshaped_padded = xla::Reshape(padded, reshaped_padded_shape);

  // 3. Permute dimensions of `reshaped_padded` to produce
  //    `permuted_reshaped_padded` of shape:
  //
  //      block_shape +
  //      [batch] +
  //      [padded_shape[1] / block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1]] +
  //      remaining_shape
  std::vector<int64_t> permutation(reshaped_padded_shape.size());
  for (int i = 0; i < block_rank; ++i) {
    permutation[i] = 1 + 2 * i + 1;
    permutation[block_rank + 1 + i] = 1 + 2 * i;
  }
  permutation[block_rank] = 0;
  std::iota(permutation.begin() + 1 + block_rank * 2, permutation.end(),
            1 + block_rank * 2);
  xla::XlaOp permuted_reshaped_padded =
      xla::Transpose(reshaped_padded, permutation);

  // 4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the
  //    batch dimension, producing an output tensor of shape:
  //
  //      [batch * prod(block_shape)] +
  //      [padded_shape[1] / block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1]] +
  //      remaining_shape
  // Determine the length of the prefix of block dims that can be combined
  // into the batch dimension due to having no padding and block_shape=1.
  std::vector<int64_t> output_shape(input_rank);
  output_shape[0] = output_dim;
  for (int i = 0; i < block_rank; ++i) {
    output_shape[1 + i] = padded_shape[1 + i] / block_shape[i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            output_shape.begin() + 1 + block_rank);

  xla::XlaOp output = xla::Reshape(permuted_reshaped_padded, output_shape);
  ctx->SetOutput(0, output);
}

class SpaceToBatchNDOp : public XlaOpKernel {
 public:
  explicit SpaceToBatchNDOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc mht_1(mht_1_v, 336, "", "./tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc", "SpaceToBatchNDOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc mht_2(mht_2_v, 341, "", "./tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc", "Compile");

    std::vector<int64_t> block_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &block_shape));

    xla::Literal paddings;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(2, &paddings));

    SpaceToBatch(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 block_shape, paddings);
  }
};
REGISTER_XLA_OP(Name("SpaceToBatchND")
                    .CompileTimeConstantInput("paddings")
                    .CompileTimeConstantInput("block_shape"),
                SpaceToBatchNDOp);

class SpaceToBatchOp : public XlaOpKernel {
 public:
  explicit SpaceToBatchOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc mht_3(mht_3_v, 362, "", "./tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc", "SpaceToBatchOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspacetobatch_opDTcc mht_4(mht_4_v, 372, "", "./tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc", "Compile");

    xla::Literal paddings;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(1, &paddings));

    SpaceToBatch(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 {block_size_, block_size_}, paddings);
  }

 private:
  int block_size_;
};
REGISTER_XLA_OP(Name("SpaceToBatch").CompileTimeConstantInput("paddings"),
                SpaceToBatchOp);

}  // namespace
}  // namespace tensorflow
