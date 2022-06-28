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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdepthtospace_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdepthtospace_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdepthtospace_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/lib/data_format.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class DepthToSpaceOp : public XlaOpKernel {
 public:
  explicit DepthToSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdepthtospace_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2xla/kernels/depthtospace_op.cc", "DepthToSpaceOp");

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdepthtospace_opDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/depthtospace_op.cc", "Compile");

    xla::XlaOp input = ctx->Input(0);

    TensorFormat data_format = data_format_;
    // If the data is in a vectorized format, reformat it into a non-vectorized
    // version first. We'll undo the transformation later.
    if (data_format == FORMAT_NCHW_VECT_C) {
      data_format = FORMAT_NCHW;
      auto input_reshaped = NCHW_VECT_CToNCHW(input);
      OP_REQUIRES_OK(ctx, input_reshaped.status());
      input = input_reshaped.ValueOrDie();
    }

    OP_REQUIRES(ctx, data_format == FORMAT_NCHW || data_format == FORMAT_NHWC,
                errors::InvalidArgument("Unsupported data format ",
                                        ToString(data_format_)));

    xla::XlaBuilder* builder = input.builder();
    auto input_xla_shape = builder->GetShape(input);
    OP_REQUIRES_OK(ctx, input_xla_shape.status());
    absl::Span<const int64_t> input_shape =
        input_xla_shape.ValueOrDie().dimensions();
    int input_rank = input_shape.size();

    static const int kRequiredDims = 4;
    OP_REQUIRES(ctx, kRequiredDims == input_rank,
                errors::InvalidArgument("Input rank should be ", kRequiredDims,
                                        "; got: ", input_rank));

    int feature_dim = GetTensorFeatureDimIndex(input_rank, data_format);
    int num_spatial_dims = GetTensorSpatialDims(input_rank, data_format);

    std::vector<int64_t> reshaped_shape;
    std::vector<int64_t> transpose_order;
    std::vector<int64_t> output_shape;
    reshaped_shape.reserve(input_rank);
    transpose_order.reserve(input_rank);
    output_shape.reserve(input_rank);
    if (data_format == FORMAT_NHWC) {
      reshaped_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(input_shape[1 + i]);
      }
      int64_t block_elems = 1;
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(block_size_);
        block_elems *= block_size_;
      }
      reshaped_shape.push_back(input_shape[feature_dim] / block_elems);

      transpose_order.push_back(0);
      for (int i = 0; i < num_spatial_dims; ++i) {
        transpose_order.push_back(i + 1);
        transpose_order.push_back(i + 1 + num_spatial_dims);
      }
      transpose_order.push_back(feature_dim + num_spatial_dims);

      output_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dims; ++i) {
        output_shape.push_back(input_shape[1 + i] * block_size_);
      }
      output_shape.push_back(input_shape[feature_dim] / block_elems);
    } else {
      // NCHW format.
      reshaped_shape.push_back(input_shape[0]);
      int64_t block_elems = 1;
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(block_size_);
        block_elems *= block_size_;
      }
      reshaped_shape.push_back(input_shape[feature_dim] / block_elems);
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(input_shape[2 + i]);
      }

      transpose_order.push_back(0);
      transpose_order.push_back(1 + num_spatial_dims);
      for (int i = 0; i < num_spatial_dims; ++i) {
        transpose_order.push_back(2 + num_spatial_dims + i);
        transpose_order.push_back(1 + i);
      }

      output_shape.push_back(input_shape[0]);
      output_shape.push_back(input_shape[feature_dim] / block_elems);
      for (int i = 0; i < num_spatial_dims; ++i) {
        output_shape.push_back(input_shape[2 + i] * block_size_);
      }
    }

    // Note: comments are given in NHWC format; NCHW is similar with a different
    // dimension order.
    // 1. Reshape `input` to `reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1],
    //       input_shape[2],
    //       block_size_,
    //       block_size_,
    //       depth / (block_size_ * block_size_)]
    OP_REQUIRES(ctx,
                input_shape[feature_dim] % (block_size_ * block_size_) == 0,
                errors::InvalidArgument(
                    "Input depth dimension (", input_shape[3],
                    ") is not divisible by square of the block size (",
                    block_size_, ")"));

    xla::XlaOp reshaped = xla::Reshape(input, reshaped_shape);

    // 2. Permute dimensions of `reshaped` to produce
    //    `permuted_reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1],
    //       block_size_,
    //       input_shape[2],
    //       block_size_,
    //       depth / (block_size_ * block_size_)]
    xla::XlaOp permuted_reshaped = xla::Transpose(reshaped, transpose_order);

    // 3. Reshape `permuted_reshaped` to flatten `block_shape` into the
    //    batch dimension, producing an output tensor of shape:
    //
    //      [batch,
    //       input_shape[1] * block_size_,
    //       input_shape[2] * block_size_,
    //       depth / (block_size_ * block_size_)]
    //
    xla::XlaOp output = xla::Reshape(permuted_reshaped, output_shape);

    // If this used to be a vectorized format turn it back now.
    if (data_format != data_format_) {
      DCHECK(data_format == FORMAT_NCHW && data_format_ == FORMAT_NCHW_VECT_C);
      auto output_reshaped = NCHWToNCHW_VECT_C(output);
      OP_REQUIRES_OK(ctx, output_reshaped.status());
      output = output_reshaped.ValueOrDie();
    }

    ctx->SetOutput(0, output);
  }

 private:
  TensorFormat data_format_;
  int block_size_;
};
REGISTER_XLA_OP(Name("DepthToSpace"), DepthToSpaceOp);

}  // namespace
}  // namespace tensorflow
