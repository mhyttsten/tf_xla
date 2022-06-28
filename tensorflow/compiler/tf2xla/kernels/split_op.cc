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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc() {
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

// XLA-specific Ops for split.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class SplitOp : public XlaOpKernel {
 public:
  explicit SplitOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/split_op.cc", "SplitOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/tf2xla/kernels/split_op.cc", "Compile");

    const int32_t num_split = num_outputs();
    const TensorShape split_dim_shape = ctx->InputShape("split_dim");
    const TensorShape input_shape = ctx->InputShape(1);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(split_dim_shape),
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_shape.dims()));
    int64_t split_dim_orig;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &split_dim_orig));

    int32_t split_dim = split_dim_orig < 0 ? split_dim_orig + input_shape.dims()
                                           : split_dim_orig;
    OP_REQUIRES(ctx, 0 <= split_dim && split_dim < input_shape.dims(),
                errors::InvalidArgument("-input rank(-", input_shape.dims(),
                                        ") <= split_dim < input rank (",
                                        input_shape.dims(), "), but got ",
                                        split_dim_orig));

    OP_REQUIRES(
        ctx, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(
        ctx, input_shape.dim_size(split_dim) % num_split == 0,
        errors::InvalidArgument(
            "Number of ways to split should evenly divide the split "
            "dimension, but got split_dim ",
            split_dim_orig, " (size = ", input_shape.dim_size(split_dim), ") ",
            "and num_split ", num_split));

    // All the slices are the same size: this is the size along the
    // split dimension.
    const int32_t slice_size = input_shape.dim_size(split_dim) / num_split;

    // The vectors we will use to define the slice. The entry for the
    // split dimensions varies for each output.
    std::vector<int64_t> begin(input_shape.dims(), 0);
    std::vector<int64_t> limits(input_shape.dims());
    std::vector<int64_t> strides(input_shape.dims(), 1);
    for (int i = 0; i < input_shape.dims(); ++i) {
      // Initially set up the limits to be the full size of the input:
      // the split dimension is filled in below.
      int64_t dim = input_shape.dim_size(i);
      limits[i] = dim;
    }

    auto input = ctx->Input(1);

    // Create each of the outputs.
    for (int i = 0; i < num_split; ++i) {
      // Slice out the ith split from the split dimension.
      begin[split_dim] = i * slice_size;
      limits[split_dim] = (i + 1) * slice_size;
      ctx->SetOutput(i, xla::Slice(input, begin, limits, strides));
    }
  }
};

REGISTER_XLA_OP(Name("Split").CompileTimeConstantInput("split_dim"), SplitOp);

class SplitVOp : public XlaOpKernel {
 public:
  explicit SplitVOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc mht_2(mht_2_v, 277, "", "./tensorflow/compiler/tf2xla/kernels/split_op.cc", "SplitVOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsplit_opDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/tf2xla/kernels/split_op.cc", "Compile");

    const int32_t num_split = num_outputs();
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape index_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx, index_shape.num_elements() == 1,
                errors::InvalidArgument(
                    "split_dim_tensor must have exactly one element."));

    int64_t split_dim_orig;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &split_dim_orig));
    int64_t split_dim = split_dim_orig < 0 ? split_dim_orig + input_shape.dims()
                                           : split_dim_orig;
    OP_REQUIRES(ctx, 0 <= split_dim && split_dim < input_shape.dims(),
                errors::InvalidArgument("-input rank(-", input_shape.dims(),
                                        ") <= split_dim < input rank (",
                                        input_shape.dims(), "), but got ",
                                        split_dim_orig));

    xla::XlaOp input = ctx->Input(0);

    OP_REQUIRES(ctx, input_shape.dims() > 0,
                errors::InvalidArgument("Can't split a 0 dimensional input"));

    OP_REQUIRES(
        ctx, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    // Check that sizes are correct.
    int total_split_size = 0;
    int neg_one_dim = -1;
    const TensorShape split_size_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx,
                split_size_shape.dims() == 1 &&
                    split_size_shape.num_elements() == num_split,
                errors::InvalidArgument(
                    "shape of tensor describing "
                    " the output must have dimension 1 and the same "
                    " number of elements as the output. Got ",
                    split_size_shape.dims(), "-D and ",
                    split_size_shape.num_elements(), " elements"));
    // Get the dimension of this split.
    std::vector<int64_t> split_sizes;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &split_sizes));

    for (int i = 0; i < num_split; ++i) {
      int64_t slice_size = split_sizes[i];
      OP_REQUIRES(ctx, slice_size >= -1,
                  errors::InvalidArgument("Split size at index ", i,
                                          " must be >= -1, Got: ", slice_size));
      if (slice_size == -1) {
        OP_REQUIRES(
            ctx, neg_one_dim == -1,
            errors::InvalidArgument("Only one dimensions can have a value of"
                                    "-1. Second one found at dimension ",
                                    i));
        neg_one_dim = i;
      } else {
        total_split_size += slice_size;
      }
    }

    OP_REQUIRES(
        ctx,
        (neg_one_dim == -1 &&
         total_split_size == input_shape.dim_size(split_dim)) ||
            (neg_one_dim >= 0 &&
             total_split_size <= input_shape.dim_size(split_dim)),
        errors::InvalidArgument("Determined shape must either match "
                                "input shape along split_dim exactly if "
                                "fully specified, or be less than the size of "
                                "the input along split_dim if not fully "
                                "specified.  Got: ",
                                total_split_size));

    if (neg_one_dim >= 0) {
      split_sizes[neg_one_dim] =
          input_shape.dim_size(split_dim) - total_split_size;
    }

    // The vectors we will use to define the slice. The entry for the
    // split dimensions varies for each output.
    std::vector<int64_t> begin(input_shape.dims(), 0);
    auto dim_sizes = input_shape.dim_sizes();
    std::vector<int64_t> limits(dim_sizes.begin(), dim_sizes.end());
    std::vector<int64_t> strides(input_shape.dims(), 1);
    for (int i = 0; i < num_split; ++i) {
      TensorShape output_shape(input_shape);
      int slice_size = split_sizes[i];
      output_shape.set_dim(split_dim, slice_size);

      // Slice out the ith split from the split dimension.
      limits[split_dim] = begin[split_dim] + slice_size;
      ctx->SetOutput(i, xla::Slice(input, begin, limits, strides));
      begin[split_dim] = limits[split_dim];
    }
  }
};

REGISTER_XLA_OP(Name("SplitV")
                    .CompileTimeConstantInput("split_dim")
                    .CompileTimeConstantInput("size_splits"),
                SplitVOp);

}  // namespace
}  // namespace tensorflow
