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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc() {
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

namespace tensorflow {
namespace {

void BatchToSpace(XlaOpKernelContext* ctx, const xla::XlaOp& input,
                  DataType input_dtype, const TensorShape& input_tensor_shape,
                  absl::Span<const int64_t> block_shape,
                  const xla::Literal& crops) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/tf2xla/kernels/batchtospace_op.cc", "BatchToSpace");

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
      crops.shape().rank() == 2 &&
          block_rank == xla::ShapeUtil::GetDimension(crops.shape(), 0) &&
          2 == xla::ShapeUtil::GetDimension(crops.shape(), 1),
      errors::InvalidArgument("crops should have shape [", block_rank,
                              ", 2] instead of ",
                              xla::ShapeUtil::HumanString(crops.shape())));

  const int64_t batch_size = input_shape[0];

  // Compute the product of the block_shape values.
  int64_t block_num_elems = 1;
  for (int i = 0; i < block_rank; ++i) {
    block_num_elems *= block_shape[i];
  }
  OP_REQUIRES(ctx, block_num_elems > 0,
              errors::InvalidArgument(
                  "The product of the block dimensions must be positive"));

  // 1. Reshape `input` to `reshaped` of shape:
  //      [block_shape[0], ..., block_shape[M-1],
  //       batch / prod(block_shape),
  //       input_shape[1], ..., input_shape[N-1]]

  OP_REQUIRES(
      ctx, batch_size % block_num_elems == 0,
      errors::InvalidArgument("Input batch dimension (", batch_size,
                              ") is not divisible by product of block sizes (",
                              block_num_elems, ")"));
  std::vector<int64_t> reshaped_shape(input_rank + block_rank);
  std::copy(block_shape.begin(), block_shape.end(), reshaped_shape.begin());
  reshaped_shape[block_rank] = batch_size / block_num_elems;
  std::copy(input_shape.begin() + 1, input_shape.end(),
            reshaped_shape.begin() + block_rank + 1);
  xla::XlaOp reshaped = xla::Reshape(input, reshaped_shape);

  // 2. Permute dimensions of `reshaped` to produce `permuted` of shape
  //      [batch / prod(block_shape),
  //
  //       input_shape[1], block_shape[0],
  //       ...,
  //       input_shape[M], block_shape[M-1],
  //
  //       input_shape[M+1], ..., input_shape[N-1]]
  std::vector<int64_t> permutation(reshaped_shape.size());
  permutation[0] = block_rank;
  for (int i = 0; i < block_rank; ++i) {
    permutation[1 + 2 * i] = block_rank + 1 + i;
    permutation[1 + 2 * i + 1] = i;
  }
  std::iota(permutation.begin() + 1 + block_rank * 2, permutation.end(),
            1 + block_rank * 2);
  xla::XlaOp permuted = xla::Transpose(reshaped, permutation);

  // 3. Reshape `permuted` to produce `reshaped_permuted` of shape
  //      [batch / prod(block_shape),
  //
  //       input_shape[1] * block_shape[0],
  //       ...,
  //       input_shape[M] * block_shape[M-1],
  //
  //       input_shape[M+1],
  //       ...,
  //       input_shape[N-1]]
  std::vector<int64_t> reshaped_permuted_shape(input_rank);
  reshaped_permuted_shape[0] = batch_size / block_num_elems;
  for (int i = 0; i < block_rank; ++i) {
    reshaped_permuted_shape[1 + i] = block_shape[i] * input_shape[1 + i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            reshaped_permuted_shape.begin() + 1 + block_rank);

  xla::XlaOp reshaped_permuted =
      xla::Reshape(permuted, reshaped_permuted_shape);

  // 4. Crop the start and end of dimensions `[1, ..., M]` of
  //    `reshaped_permuted` according to `crops` to produce the output of shape:
  //      [batch / prod(block_shape),
  //
  //       input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
  //       ...,
  //       input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
  //
  //       input_shape[M+1], ..., input_shape[N-1]]
  std::vector<int64_t> start_indices(input_rank, 0);
  std::vector<int64_t> end_indices = reshaped_permuted_shape;
  std::vector<int64_t> strides(input_rank, 1);
  for (int i = 0; i < block_rank; ++i) {
    int64_t crop_start = crops.Get<int64_t>({i, 0});
    int64_t crop_end = crops.Get<int64_t>({i, 1});
    OP_REQUIRES(ctx, crop_start >= 0 && crop_end >= 0,
                errors::InvalidArgument("Crops must be non-negative"));
    start_indices[1 + i] = crop_start;
    end_indices[1 + i] -= crop_end;
    OP_REQUIRES(
        ctx, start_indices[1 + i] <= end_indices[1 + i],
        errors::InvalidArgument(
            "Cropped size must be non-negative: start: ", crop_start,
            " end: ", crop_end, " size ", reshaped_permuted_shape[1 + i]));
  }
  xla::XlaOp output =
      xla::Slice(reshaped_permuted, start_indices, end_indices, strides);
  ctx->SetOutput(0, output);
}

class BatchToSpaceNDOp : public XlaOpKernel {
 public:
  explicit BatchToSpaceNDOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc mht_1(mht_1_v, 320, "", "./tensorflow/compiler/tf2xla/kernels/batchtospace_op.cc", "BatchToSpaceNDOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc mht_2(mht_2_v, 325, "", "./tensorflow/compiler/tf2xla/kernels/batchtospace_op.cc", "Compile");

    std::vector<int64_t> block_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &block_shape));

    xla::Literal crops;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(2, &crops));

    BatchToSpace(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 block_shape, crops);
  }
};
REGISTER_XLA_OP(Name("BatchToSpaceND")
                    .CompileTimeConstantInput("block_shape")
                    .CompileTimeConstantInput("crops"),
                BatchToSpaceNDOp);

class BatchToSpaceOp : public XlaOpKernel {
 public:
  explicit BatchToSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc mht_3(mht_3_v, 346, "", "./tensorflow/compiler/tf2xla/kernels/batchtospace_op.cc", "BatchToSpaceOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSbatchtospace_opDTcc mht_4(mht_4_v, 356, "", "./tensorflow/compiler/tf2xla/kernels/batchtospace_op.cc", "Compile");

    xla::Literal crops;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(1, &crops));

    BatchToSpace(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 {block_size_, block_size_}, crops);
  }

 private:
  int block_size_;
};
REGISTER_XLA_OP(Name("BatchToSpace").CompileTimeConstantInput("crops"),
                BatchToSpaceOp);

}  // namespace
}  // namespace tensorflow
