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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreshape_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreshape_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreshape_opDTcc() {
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

// XLA-specific reshape Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ReshapeOp : public XlaOpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreshape_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/reshape_op.cc", "ReshapeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreshape_opDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/tf2xla/kernels/reshape_op.cc", "Compile");

    TensorShape input_shape = ctx->InputShape(0);
    auto input_xla_shape = ctx->InputXlaShape(0);
    const TensorShape sizes_shape = ctx->InputShape(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(sizes_shape),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes_shape.DebugString()));
    const int64_t num_dims = sizes_shape.num_elements();

    std::vector<int64_t> shape_input;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector(
                       1, &shape_input, xla::ValueInferenceMode::kUpperBound));
    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one if there
    // is one.
    TensorShape shape;
    int64_t product = 1;
    int unknown_index = -1;
    bool shape_has_zero_dim = false;
    for (int d = 0; d < num_dims; ++d) {
      const int64_t size = shape_input[d];
      if (size == -1) {
        OP_REQUIRES(
            ctx, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape.AddDim(size);
        shape_has_zero_dim = true;
      } else {
        OP_REQUIRES(ctx, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    auto input = ctx->Input(0);
    if (unknown_index != -1) {
      int64_t input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input_shape.dims(); dim++) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input_shape.dim_size(dim) > 0 || !shape_has_zero_dim) {
          input_num_elements *= input_shape.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      int64_t missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        if (input_xla_shape->is_static() || input_xla_shape->rank() != 1) {
          OP_REQUIRES(
              ctx, product * missing == input_num_elements,
              errors::InvalidArgument(
                  "Input to reshape is a tensor with ", input_num_elements,
                  " values, but the requested shape requires a multiple of ",
                  product));
        } else {
          // For 1D shape, we can safely insert extra padding in the end to make
          // sure the input is multiple of the product of the known dimensions.
          // (We can probably do that for >1D shapes but that involves
          // factorizing the number of missing elements.)
          int64_t padded_input_num =
              xla::CeilOfRatio(input_num_elements, product) * product;
          missing = padded_input_num / product;
          input = xla::PadInDim(
              input, xla::Zero(ctx->builder(), input_xla_shape->element_type()),
              0, 0, padded_input_num - input_num_elements);
          input_shape.set_dim(0, padded_input_num);
        }
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(ctx, shape.num_elements() == input_shape.num_elements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input_shape.num_elements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    VLOG(2) << "Reshape from " << input_shape.DebugString() << " to "
            << shape.DebugString() << ", unknown_index=" << unknown_index;
    if (input_xla_shape->is_static()) {
      ctx->SetOutput(0, xla::Reshape(input, shape.dim_sizes()));
      return;
    }

    std::vector<xla::XlaOp> output_dim_sizes;
    std::vector<bool> dims_are_dynamic;
    const auto& dims = shape.dims();
    dims_are_dynamic.reserve(dims);
    for (int64_t i = 0; i < dims; ++i) {
      output_dim_sizes.push_back(
          xla::Reshape(xla::Slice(ctx->Input(1), {i}, {i + 1}, {1}), {}));
    }
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPredVector(1, &dims_are_dynamic));
    if (unknown_index == -1) {
      // No unknown index.
      ctx->SetOutput(
          0, xla::DynamicReshape(input, output_dim_sizes, shape.dim_sizes(),
                                 dims_are_dynamic));
      return;
    }
    auto common_factors =
        xla::CommonFactors(input_shape.dim_sizes(), shape.dim_sizes());

    // Find common_factors that the input belongs to.
    for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
      auto start = common_factors[i];
      auto end = common_factors[i + 1];
      bool input_is_dynamic = false;
      // product of all input dims in this group. E.g., in
      // reshape(Tensor([2, 3, 3]), [3, -1, 3]) product of the group
      // containing -1 will be 6.
      xla::XlaOp product = xla::One(ctx->builder(), xla::S32);
      for (int64_t dim = start.first; dim < end.first; ++dim) {
        if (input_xla_shape->is_dynamic_dimension(dim)) {
          input_is_dynamic = true;
        }
        product = xla::Mul(product, xla::GetDimensionSize(input, dim));
      }
      bool unknown_dim_in_group = false;
      // The real size for the -1 dimension in a reshape. E.g., in
      // reshape(Tensor([2, 3, 3]), [3, -1, 3]) this will be 2.
      xla::XlaOp unknown_dim_size = product;
      for (int64_t dim = start.second; dim < end.second; ++dim) {
        if (dim == unknown_index) {
          unknown_dim_in_group = true;
        } else {
          unknown_dim_size = xla::Div(unknown_dim_size, output_dim_sizes[dim]);
        }
      }

      if (unknown_dim_in_group) {
        // If input dim is dynamic, output dim at the -1 position must be
        // dynamic. Similarly, if input dim is static, output dim has to be
        // static at the -1 dimension.
        dims_are_dynamic[unknown_index] = input_is_dynamic;
        output_dim_sizes[unknown_index] = unknown_dim_size;

        ctx->SetOutput(
            0, xla::DynamicReshape(input, output_dim_sizes, shape.dim_sizes(),
                                   dims_are_dynamic));
        VLOG(2) << "Reshape from " << ctx->InputXlaShape(0)->ToString()
                << " to " << xla::VectorString(shape.dim_sizes())
                << ", dynamic_dims=" << xla::VectorString(dims_are_dynamic);
        return;
      }
    }
  }
};

REGISTER_XLA_OP(Name("Reshape").CompileTimeConstantInput("shape"), ReshapeOp);

}  // namespace
}  // namespace tensorflow
