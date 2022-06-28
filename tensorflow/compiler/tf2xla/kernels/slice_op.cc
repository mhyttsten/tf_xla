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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSslice_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSslice_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSslice_opDTcc() {
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

// XLA-specific Slice Op.

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace {

class SliceOp : public XlaOpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSslice_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/tf2xla/kernels/slice_op.cc", "SliceOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSslice_opDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2xla/kernels/slice_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape begin_tensor_shape = ctx->InputShape(1);
    const TensorShape size_tensor_shape = ctx->InputShape(2);

    const int input_dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(begin_tensor_shape) &&
            TensorShapeUtils::IsVector(size_tensor_shape) &&
            begin_tensor_shape.num_elements() == input_dims &&
            size_tensor_shape.num_elements() == input_dims,
        errors::InvalidArgument(
            "Expected begin and size arguments to be 1-D tensors of size ",
            input_dims, ", but got shapes ", begin_tensor_shape.DebugString(),
            " and ", size_tensor_shape.DebugString(), " instead."));

    std::vector<int64_t> begin;
    std::vector<int64_t> size;
    const bool all_begins_are_constant =
        ctx->ConstantInputAsIntVector(1, &begin).ok();
    const bool all_sizes_are_constant =
        ctx->ConstantInputAsIntVector(2, &size).ok();
    if (all_begins_are_constant && all_sizes_are_constant) {
      std::vector<int64_t> wrapped_size(size.size());
      // `begin` is a compile-time constant.
      for (int i = 0; i < input_dims; ++i) {
        if (size[i] == -1) {
          // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
          wrapped_size[i] = input_shape.dim_size(i) - begin[i];
        } else {
          wrapped_size[i] = size[i];
        }
      }

      for (int i = 0; i < input_dims; ++i) {
        int64_t b = begin[i];
        int64_t s = wrapped_size[i];
        if (input_shape.dim_size(i) == 0) {
          OP_REQUIRES(ctx, b == 0 && s == 0,
                      errors::InvalidArgument(
                          "Expected begin[", i, "] == 0 (got ", b,
                          ") and size[", i, "] == 0 ", "(got ", s, ") when ",
                          "input_shape.dim_size(", i, ") == 0"));
        } else {
          OP_REQUIRES(ctx, 0 <= b && b <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                              input_shape.dim_size(i),
                                              "], but got ", b));
          OP_REQUIRES(ctx, 0 <= s && b + s <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                              input_shape.dim_size(i) - b,
                                              "], but ", "got ", s));
        }
      }

      std::vector<int64_t> limits;
      limits.reserve(begin.size());
      for (int i = 0; i < begin.size(); ++i) {
        limits.push_back(begin[i] + wrapped_size[i]);
      }
      std::vector<int64_t> strides(begin.size(), 1);
      auto slice = xla::Slice(ctx->Input(0), begin, limits, strides);
      // Check for slice on dynamic dimensions.
      std::vector<bool> size_is_dynamic;
      OP_REQUIRES_OK(
          ctx, ctx->ResolveInputDynamismIntoPredVector(2, &size_is_dynamic));

      for (int64_t i = 0; i < size.size(); ++i) {
        if (size_is_dynamic[i]) {
          if (size[i] != -1) {
            // If there is a dynamic dimension, properly set dimension size of
            // the slice.
            auto dynamic_size =
                xla::Reshape(xla::Slice(ctx->Input(2), {i}, {i + 1}, {1}), {});

            slice = xla::SetDimensionSize(slice, dynamic_size, i);
          }
        }
      }
      ctx->SetOutput(0, slice);
    } else {
      // When a size is -1, we take rest of the dimension according to
      // https://www.tensorflow.org/api_docs/python/tf/slice.
      // This essentially makes size as dynamic.
      bool constant_size_is_minus_one = false;
      // `begin` or `size` is not a compile-time constant.
      if (all_sizes_are_constant) {
        for (int i = 0; i < input_dims; ++i) {
          if (size[i] < 0) {
            OP_REQUIRES(ctx, size[i] == -1,
                        errors::InvalidArgument(
                            "Negative size of slice operator can only be -1"));
            constant_size_is_minus_one = true;
          }

          OP_REQUIRES(ctx, size[i] <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                              input_shape.dim_size(i),
                                              "], but ", "got ", size[i]));
        }
      }

      absl::InlinedVector<xla::XlaOp, 4> begin_indices;
      begin_indices.reserve(input_dims);
      xla::XlaOp begin = ctx->Input("begin");
      for (int i = 0; i < input_dims; i++) {
        begin_indices.push_back(
            xla::Reshape(xla::Slice(begin, {i}, {i + 1}, {1}), {}));
      }
      if (all_sizes_are_constant && !constant_size_is_minus_one) {
        xla::XlaOp input = ctx->Input(0);
        ctx->SetOutput(0, xla::DynamicSlice(input, begin_indices, size));
      } else {
        // Size is not constant, use input size as upperbound and then set
        // dimension size on it.

        // First pad input with input size to avoid OOB -- dynamic slice with
        // OOB slice produces undesired results.
        xla::PaddingConfig padding_config;
        xla::XlaOp input = ctx->Input(0);
        for (int64_t i = 0; i < input_dims; ++i) {
          auto* dims = padding_config.add_dimensions();
          dims->set_edge_padding_low(0);
          dims->set_edge_padding_high(input_shape.dim_size(i));
          dims->set_interior_padding(0);
          input = xla::RemoveDynamicDimension(input, i);
        }
        auto padded_input =
            xla::Pad(input, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
                     padding_config);
        // Slice full size out of the input starting from the offsets.
        auto sliced = xla::DynamicSlice(padded_input, begin_indices,
                                        input_shape.dim_sizes());
        for (int i = 0; i < input_dims; i++) {
          xla::XlaOp dynamic_size =
              xla::Reshape(xla::Slice(ctx->Input(2), {i}, {i + 1}, {1}), {});
          if (constant_size_is_minus_one && size[i] == -1) {
            // size = input_.dim_size(i) - begin[i]
            dynamic_size = xla::ConstantR0<int32>(ctx->builder(),
                                                  input_shape.dim_size(i)) -
                           begin_indices[i];
          }
          auto constant_size = ctx->value_inference().AnalyzeConstant(
              dynamic_size, xla::ValueInferenceMode::kValue);
          OP_REQUIRES_OK(ctx, constant_size.status());
          if (constant_size->AllValid()) {
            // Slice size on this dimension is constant. This branch is
            // triggered when some dimensions's slice sizes are constant while
            // some are dynamic.
            sliced = xla::SliceInDim(
                sliced, 0, constant_size->Get<int32>({}).value(), 1, i);
          } else {
            // We gave a generous bound (same as input) to the output, try reset
            // the bound if a tighter one can be found.
            auto status = xla::SetDimensionSizeWithRebound(
                &ctx->value_inference(), sliced, dynamic_size, i);
            OP_REQUIRES_OK(ctx, status.status());
            sliced = status.ValueOrDie();
          }
        }
        ctx->SetOutput(0, sliced);
      }
    }
  }
};

REGISTER_XLA_OP(Name("Slice")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("size"),
                SliceOp);

}  // namespace
}  // namespace tensorflow
