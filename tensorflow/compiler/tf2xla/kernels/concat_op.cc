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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc() {
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

// XLA-specific Concat Ops.

#include <limits>
#include <vector>

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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class ConcatBaseOp : public XlaOpKernel {
 public:
  ConcatBaseOp(OpKernelConstruction* c, int axis_index)
      : XlaOpKernel(c), axis_index_(axis_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "ConcatBaseOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "Compile");

    const TensorShape concat_dim_tensor_shape = ctx->InputShape(axis_index_);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_tensor_shape),
                errors::InvalidArgument(
                    "Concat dim tensor should be a scalar, but got shape ",
                    concat_dim_tensor_shape.DebugString()));
    int64_t concat_dim;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(axis_index_, &concat_dim));

    std::vector<xla::XlaOp> values;
    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
    const int N = values.size();
    const int input_dims = shapes[0].dims();
    const TensorShape& input_shape = shapes[0];

    int32_t axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(ctx, 0 <= axis && axis < input_dims,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    // Make a vector holding the XlaOp for each of the inputs that has non-zero
    // elements.
    std::vector<xla::XlaOp> input_data;
    int output_concat_dim = 0;
    for (int i = 0; i < N; ++i) {
      xla::XlaOp handle = values[i];
      const TensorShape& in_shape = shapes[i];
      OP_REQUIRES(
          ctx, in_shape.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in_shape.DebugString()));
      if (in_shape.dims() == 0) {
        // Inputs that come in as scalars must be reshaped to 1-vectors.
        input_data.push_back(xla::Reshape(handle, {1}));
      } else {
        input_data.push_back(handle);
      }
      output_concat_dim += in_shape.dims() > 0 ? in_shape.dim_size(axis) : 1;
    }

    VLOG(1) << "Concat dim " << concat_dim << " equivalent to " << axis;
    ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), input_data, axis));
  }

 private:
  int axis_index_;
};

class ConcatOp : public ConcatBaseOp {
 public:
  explicit ConcatOp(OpKernelConstruction* c)
      : ConcatBaseOp(c, /* axis_index */ 0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "ConcatOp");
}
};

// ConcatV2 operation is the same as Concat except 'concat_dim'
// is the last input instead of the first and renamed to 'axis'.
class ConcatV2Op : public ConcatBaseOp {
 public:
  explicit ConcatV2Op(OpKernelConstruction* c)
      : ConcatBaseOp(c, /* axis_index */ c->num_inputs() - 1) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_3(mht_3_v, 289, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "ConcatV2Op");
}
};

REGISTER_XLA_OP(Name("Concat").CompileTimeConstantInput("concat_dim"),
                ConcatOp);
REGISTER_XLA_OP(Name("ConcatV2")
                    .TypeConstraint("Tidx", DT_INT32)
                    .CompileTimeConstantInput("axis"),
                ConcatV2Op);

class ConcatOffsetOp : public XlaOpKernel {
 public:
  explicit ConcatOffsetOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_4(mht_4_v, 304, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "ConcatOffsetOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSconcat_opDTcc mht_5(mht_5_v, 309, "", "./tensorflow/compiler/tf2xla/kernels/concat_op.cc", "Compile");

    const TensorShape concat_dim_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_shape),
                errors::InvalidArgument(
                    "Concat dim tensor should be a scalar, but got shape ",
                    concat_dim_shape.DebugString()));
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ctx->InputShape(i)),
                  errors::InvalidArgument("input ", i,
                                          " should be a vector, but got shape ",
                                          ctx->InputShape(i).DebugString()));
    }
    // Suppose a Concat() op needs to Concatenate N tensors, each of
    // which has the same number of dimensions.  Their shapes match
    // except the concat dimension.
    //
    // E.g., say, we want to concatenate 3 tensors in the 2nd
    // dimension, and their shapes are:
    //
    //  [2, 2, 5, 7]
    //  [2, 3, 5, 7]
    //  [2, 4, 5, 7]
    //
    // Here, N=3, cdim=1, dims=4. The concatenated tensor has shape
    // [2,9,5,7]. We will compute the cumulative sum along the 2nd
    // dimension to figure out each input's offset in the concatenated
    // output:
    //  [0, 0, 0, 0]
    //  [0, 2, 0, 0]
    //  [0, 5, 0, 0]
    const int32_t N = ctx->num_inputs() - 1;
    const TensorShape inp0_shape = ctx->InputShape(1);
    std::vector<int64_t> inp0_dims;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector(
                       1, &inp0_dims, xla::ValueInferenceMode::kUpperBound));
    const int64_t inp0_rank = inp0_shape.num_elements();

    int64_t cdim;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &cdim));

    VLOG(1) << "ConcatOffset " << cdim << "," << inp0_rank;
    int32_t axis = cdim < 0 ? cdim + inp0_rank : cdim;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, inp0_rank),
                errors::InvalidArgument("Concat dim is out of range: ", axis,
                                        " vs. ", inp0_rank));
    int32_t offset = 0;
    for (int i = 0; i < N; ++i) {
      const TensorShape inp_shape = ctx->InputShape(1 + i);
      OP_REQUIRES(ctx, inp0_rank == inp_shape.num_elements(),
                  errors::InvalidArgument("input ", i, " should contain ",
                                          inp0_rank, " elements, but got ",
                                          inp_shape.num_elements()));
      std::vector<int64_t> inp_dims;
      OP_REQUIRES_OK(
          ctx, ctx->ConstantInputAsIntVector(
                   1 + i, &inp_dims, xla::ValueInferenceMode::kUpperBound));

      Tensor out_constant(DT_INT32, TensorShape({inp0_rank}));
      auto out_vec = out_constant.vec<int32>();
      for (int64_t j = 0; j < inp0_rank; ++j) {
        if (j == axis) {
          out_vec(j) = offset;
          offset += inp_dims[j];
        } else {
          const int32_t inp0_element = inp0_dims[j];
          const int32_t inp_element = inp_dims[j];
          OP_REQUIRES(ctx, inp0_element == inp_element,
                      errors::InvalidArgument(
                          "All dimensions except ", axis, " must match. Input ",
                          i, " has shape [", absl::StrJoin(inp_dims, " "),
                          "] and doesn't match input 0 with shape [",
                          absl::StrJoin(inp0_dims, " "), "]."));
          out_vec(j) = 0;
        }
      }

      ctx->SetConstantOutput(i, out_constant);
    }
  }
};

REGISTER_XLA_OP(Name("ConcatOffset")
                    .CompileTimeConstantInput("concat_dim")
                    .CompileTimeConstantInput("shape"),
                ConcatOffsetOp);

}  // namespace
}  // namespace tensorflow
