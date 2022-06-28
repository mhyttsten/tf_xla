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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc() {
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

// XLA-specific dynamic stitch Op.

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class DynamicStitchOp : public XlaOpKernel {
 public:
  explicit DynamicStitchOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_stitch_op.cc", "DynamicStitchOp");

    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("DynamicStitchOp: Must have some inputs"));
    OP_REQUIRES(ctx, ctx->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    "DynamicStitchOp: Must have even number of arguments"));
    // Compute expected input signature
    const int n = ctx->num_inputs() / 2;
    const DataType dt = ctx->input_type(n);
    DataTypeVector expected;
    for (int i = 0; i < n; i++) {
      expected.push_back(DT_INT32);
    }
    for (int i = 0; i < n; i++) {
      expected.push_back(dt);
    }
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected, {dt}));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_stitch_op.cc", "Compile");

    // Validate that data_shape[i] = indices[i].shape() + constant
    std::vector<xla::Literal> indices_input;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputList("indices", &indices_input));

    std::vector<xla::XlaOp> data;
    std::vector<TensorShape> data_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("data", &data, &data_shapes));

    std::vector<xla::Literal> indices(indices_input.size());

    const TensorShape& data0_shape = data_shapes[0];
    TensorShape indices0_shape;
    OP_REQUIRES_OK(
        ctx, XLAShapeToTensorShape(indices_input[0].shape(), &indices0_shape));
    for (int input_num = 0; input_num < indices_input.size(); input_num++) {
      TensorShape indices_shape;
      OP_REQUIRES_OK(ctx,
                     XLAShapeToTensorShape(indices_input[input_num].shape(),
                                           &indices_shape));
      TensorShape& data_shape = data_shapes[input_num];
      if (!TensorShapeUtils::StartsWith(data_shape, indices_shape)) {
        // This happens when data shape is a dynamic shape with bound with
        // indices_shape is a concrete shape. We use slice to reconcile the
        // mismatch.
        for (int64_t i = 0; i < indices_shape.dims(); ++i) {
          data_shape.set_dim(i, indices_shape.dim_size(i));
          data[input_num] = xla::SliceInDim(data[input_num], 0,
                                            indices_shape.dim_size(i), 1, i);
        }
      }
      OP_REQUIRES(
          ctx, TensorShapeUtils::StartsWith(data_shape, indices_shape),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data_shape.DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices_shape.DebugString()));
      OP_REQUIRES(
          ctx,
          input_num == 0 || SameExtraShape(data0_shape, indices0_shape,
                                           data_shape, indices_shape),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0_shape.dims(), ":] = data[",
              input_num, "].shape[", indices_shape.dims(),
              ":], got data[0].shape = ", data0_shape.DebugString(), ", data[",
              input_num, "].shape = ", data_shape.DebugString(),
              ", indices[0].shape = ", indices0_shape.DebugString(),
              ", indices[", input_num,
              "].shape = ", indices_shape.DebugString()));

      OP_REQUIRES_OK(ctx,
                     XlaHelpers::ReshapeLiteral(indices_input[input_num],
                                                {indices_shape.num_elements()},
                                                &indices[input_num]));
    }

    // Find which slice will be used for each index. If the same index
    // appears in multiple inputs, the last one is used. The logic
    // here is different from that in third_party/tensorflow because
    // it is important for XLA that there be a well-formed Concat
    // operation at the end. The existing CPU/GPU code copies multiple
    // source slices to the same destination slice if there are
    // repeated indices, whereas the XLA code works out which
    // source slice will 'win' and only uses that in the Concat.
    int max_index = -1;
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      for (int i = 0; i < indices[input_num].shape().dimensions(0); ++i) {
        max_index = std::max(max_index, indices[input_num].Get<int>({i}));
      }
    }
    int number_of_indices = max_index + 1;
    int64_t result_rank = 1 + data0_shape.dims() - indices0_shape.dims();
    if (number_of_indices == 0) {
      std::vector<int64_t> result_shape(result_rank);
      for (int d = indices0_shape.dims(); d < data0_shape.dims(); d++) {
        result_shape[d - indices0_shape.dims() + 1] = data0_shape.dim_size(d);
      }
      xla::PrimitiveType element_type =
          ctx->input_xla_type(ctx->num_inputs() - 1);
      xla::Literal empty_literal = xla::Literal::CreateFromShape(
          xla::ShapeUtil::MakeShape(element_type, result_shape));
      ctx->SetOutput(0, xla::ConstantLiteral(ctx->builder(), empty_literal));
      return;
    }

    // Construct the reverse mapping, for each index, of which slice of which
    // input it comes from.
    std::vector<int32> src_input_vector(number_of_indices);
    std::vector<int32> src_slice_vector(number_of_indices);
    std::vector<bool> src_index_used(number_of_indices);
    int index_used_count = 0;
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      for (int i = 0; i < indices[input_num].shape().dimensions(0); ++i) {
        int index = indices[input_num].Get<int>({i});
        src_input_vector[index] = input_num;
        src_slice_vector[index] = i;
        if (!src_index_used[index]) {
          src_index_used[index] = true;
          ++index_used_count;
        }
      }
    }
    OP_REQUIRES(ctx, index_used_count == number_of_indices,
                errors::InvalidArgument("not all indices are used"));

    // Look up all the children expressions that represent the data
    // inputs.
    std::vector<xla::XlaOp> input(indices.size());
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      TensorShape new_shape;
      // first reshaped dimension is the number of indices for this input.
      new_shape.AddDim(indices[input_num].shape().dimensions(0));
      // Then the rest are the common extra shape.
      for (int d = indices0_shape.dims(); d < data0_shape.dims(); d++) {
        new_shape.AddDim(data0_shape.dim_size(d));
      }
      // Get the data, shaped appropriately.
      auto handle = data[input_num];
      if (new_shape == data_shapes[input_num]) {
        input[input_num] = handle;
      } else {
        input[input_num] = xla::Reshape(handle, new_shape.dim_sizes());
      }
    }

    // Set up the vectors for slicing: the first dimension will vary
    // slice by slice, and the rest take the full common extra shape.
    std::vector<int64_t> slice_start(result_rank);
    std::vector<int64_t> slice_limit(result_rank);
    std::vector<int64_t> stride(result_rank, 1);
    for (int d = indices0_shape.dims(); d < data0_shape.dims(); d++) {
      slice_limit[1 + d - indices0_shape.dims()] = data0_shape.dim_size(d);
    }
    std::vector<xla::XlaOp> to_concat(number_of_indices);
    for (int index_num = 0; index_num < number_of_indices; index_num++) {
      const auto& expression = input[src_input_vector[index_num]];
      // Take the appropriate slice of data.
      slice_start[0] = src_slice_vector[index_num];
      slice_limit[0] = src_slice_vector[index_num] + 1;
      // And place it in the concat list in the place indicated by
      // the index.
      to_concat[index_num] =
          xla::Slice(expression, slice_start, slice_limit, stride);
    }

    ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), to_concat, 0));
  }

 private:
  // Check if data0_shape[indices0.dims():] == data1_shape[indices1.dims():]
  static bool SameExtraShape(const TensorShape& data0_shape,
                             const TensorShape& indices0,
                             const TensorShape& data1_shape,
                             const TensorShape& indices1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_stitch_opDTcc mht_2(mht_2_v, 383, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_stitch_op.cc", "SameExtraShape");

    const int extra0 = data0_shape.dims() - indices0.dims();
    const int extra1 = data1_shape.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0_shape.dim_size(indices0.dims() + i) !=
          data1_shape.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }
};

REGISTER_XLA_OP(Name("DynamicStitch").CompileTimeConstantInput("indices"),
                DynamicStitchOp);
REGISTER_XLA_OP(
    Name("ParallelDynamicStitch").CompileTimeConstantInput("indices"),
    DynamicStitchOp);

}  // namespace
}  // namespace tensorflow
