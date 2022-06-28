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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc() {
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

// XLA implementations of Random ops
// TODO(misard,phawkins): handle random number generator seeds/states correctly.
// TODO(misard,phawkins): add tests.

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class RandomUniformOp : public XlaOpKernel {
 public:
  explicit RandomUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "RandomUniformOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(
                            0, &shape, xla::ValueInferenceMode::kUpperBound));

    const DataType dtype = output_type(0);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.uniform with XLA compilation will ignore "
           "seeds; consider using tf.random.stateless_uniform instead if "
           "reproducible behavior is desired. "
        << name();
    xla::XlaOp result = xla::RngUniform(XlaHelpers::Zero(b, dtype),
                                        XlaHelpers::One(b, dtype), xla_shape);
    auto result_status_or =
        SetAllDimensionSizes(&ctx->value_inference(), result, ctx->Input(0));
    OP_REQUIRES_OK(ctx, result_status_or.status());
    result = result_status_or.ValueOrDie();
    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformOp);
};

REGISTER_XLA_OP(Name("RandomUniform").CompileTimeConstantInput("shape"),
                RandomUniformOp);

REGISTER_XLA_OP(Name("RandomShuffle"), MlirXlaOpKernel);

class RandomUniformIntOp : public XlaOpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "RandomUniformIntOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(input_type(1), shape, &xla_shape));

    const TensorShape minval_shape = ctx->InputShape(1);
    const TensorShape maxval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval_shape.DebugString()));

    auto minval = ctx->Input(1);
    auto maxval = ctx->Input(2);
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.uniform with XLA compilation will ignore "
           "seeds; consider using tf.random.stateless_uniform instead if "
           "reproducible behavior is desired. "
        << name();
    ctx->SetOutput(0, xla::RngUniform(minval, maxval, xla_shape));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformIntOp);
};

REGISTER_XLA_OP(Name("RandomUniformInt").CompileTimeConstantInput("shape"),
                RandomUniformIntOp);

class RandomStandardNormalOp : public XlaOpKernel {
 public:
  explicit RandomStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_4(mht_4_v, 301, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "RandomStandardNormalOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "Compile");

    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(
                            0, &shape, xla::ValueInferenceMode::kUpperBound));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    // Normal distribution with a mean of 0 and a standard deviation of 1:
    xla::XlaOp result = xla::RngNormal(XlaHelpers::Zero(b, dtype),
                                       XlaHelpers::One(b, dtype), xla_shape);
    auto result_status_or =
        SetAllDimensionSizes(&ctx->value_inference(), result, ctx->Input(0));
    OP_REQUIRES_OK(ctx, result_status_or.status());
    result = result_status_or.ValueOrDie();
    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomStandardNormalOp);
};

REGISTER_XLA_OP(Name("RandomStandardNormal").CompileTimeConstantInput("shape"),
                RandomStandardNormalOp);

class TruncatedNormalOp : public XlaOpKernel {
 public:
  explicit TruncatedNormalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_6(mht_6_v, 339, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "TruncatedNormalOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_7(mht_7_v, 344, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "Compile");

    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp one = xla::One(b, xla_shape.element_type());
    xla::XlaOp min_positive =
        xla::MinPositiveNormalValue(b, xla_shape.element_type());
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.truncated_normal with XLA "
           "compilation will ignore seeds; consider using "
           "tf.random.stateless_truncated_normal instead if "
           "reproducible behavior is desired. "
        << name();
    auto uniform = xla::RngUniform(min_positive, one, xla_shape);
    ctx->SetOutput(0, TruncatedNormal(uniform));
  }
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_DOUBLE}),
                TruncatedNormalOp);

// Broadcast a ParameterizedTruncatedNormal parameter to the output shape. If
// the parameter is a vector of shape [num_batches], then it is broadcast along
// dimension 0 to ([num_batches] x samples_per_batch). Otherwise it is a scalar
// or has shape [1], in which case the single value is broadcast.
static StatusOr<xla::XlaOp> BroadcastParameters(xla::XlaOp params,
                                                TensorShape& output_shape) {
  // broadcast to [samples1, ..., num_batches]
  int rank = output_shape.dims();
  std::vector<int64_t> bcast_shape;
  for (int i = 1; i < rank; ++i) {
    bcast_shape.push_back(output_shape.dim_size(i));
  }
  bcast_shape.push_back(output_shape.dim_size(0));
  TF_ASSIGN_OR_RETURN(xla::XlaOp bcast_params,
                      BroadcastTo(params, bcast_shape));

  // transpose to [num_batches, samples1, ...]
  std::vector<int64_t> permutation;
  permutation.push_back(rank - 1);
  for (int i = 0; i < rank - 1; ++i) {
    permutation.push_back(i);
  }
  return xla::Transpose(bcast_params, permutation);
}

class ParameterizedTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit ParameterizedTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_8(mht_8_v, 404, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "ParameterizedTruncatedNormalOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSrandom_opsDTcc mht_9(mht_9_v, 409, "", "./tensorflow/compiler/tf2xla/kernels/random_ops.cc", "Compile");

    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    OP_REQUIRES(ctx, xla_shape.rank() >= 1,
                errors::InvalidArgument(
                    "shape parameter must have rank >= 1, received (",
                    xla::ShapeUtil::HumanString(xla_shape), ")"));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp one = xla::One(b, xla_shape.element_type());
    xla::XlaOp min_positive =
        xla::MinPositiveNormalValue(b, xla_shape.element_type());
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.truncated_normal with XLA "
           "compilation will ignore seeds; consider using "
           "tf.random.stateless_truncated_normal instead if "
           "reproducible behavior is desired. "
        << name();
    xla::XlaOp uniform = xla::RngUniform(min_positive, one, xla_shape);

    auto result = b->ReportErrorOrReturn([&]() -> StatusOr<xla::XlaOp> {
      TF_ASSIGN_OR_RETURN(xla::XlaOp means,
                          BroadcastParameters(ctx->Input(1), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp stddevs,
                          BroadcastParameters(ctx->Input(2), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp minvals,
                          BroadcastParameters(ctx->Input(3), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp maxvals,
                          BroadcastParameters(ctx->Input(4), shape));
      return ParameterizedTruncatedNormal(uniform, means, stddevs, minvals,
                                          maxvals);
    });

    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("ParameterizedTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_DOUBLE}),
                ParameterizedTruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
