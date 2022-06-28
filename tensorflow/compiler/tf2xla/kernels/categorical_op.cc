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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc() {
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

// XLA implementations of Categorical op.

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class CategoricalOp : public XlaOpKernel {
 public:
  explicit CategoricalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "CategoricalOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "Compile");

    // Get the logits
    const xla::XlaOp& logits = ctx->Input(0);
    TensorShape logits_shape = ctx->InputShape(0);
    int64_t num_samples;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(
                       1, &num_samples, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits should be a matrix, got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, num_samples >= 0,
                errors::InvalidArgument(
                    "num_samples should be nonnegative, got ", num_samples));

    for (int i = 0; i < 2; i++) {
      const int64_t dim = logits_shape.dim_size(i);
      OP_REQUIRES(
          ctx, static_cast<int>(dim) == dim,
          errors::InvalidArgument("logits.shape = ", logits_shape.DebugString(),
                                  " too large for int"));
    }

    const int64_t batch_size = logits_shape.dim_size(0);
    const int64_t num_classes = logits_shape.dim_size(1);

    xla::Shape uniform_shape;
    int class_dimension;
    bool num_samples_is_dynamic = false;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(1, &num_samples_is_dynamic));
    if (num_samples != 1 || num_samples_is_dynamic) {
      std::array<int64_t, 3> uniform_shape_array = {
          {batch_size, num_samples, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 2;
    } else {
      // Have a special case for when we only need one sample, because
      // dimensions may be padded on architectures with tiled memory layouts, so
      // if the num_classes or batch size is large then this can lead to
      // expensive wasted memory.
      std::array<int64_t, 2> uniform_shape_array = {{batch_size, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 1;
    }
    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(0), &type));
    xla::XlaOp log_uniforms = GetLogUniforms(uniform_shape, type, ctx);

    if (num_samples_is_dynamic) {
      // num_samples is dimension 1 in uniform_shape_array.
      log_uniforms = xla::SetDimensionSize(log_uniforms, ctx->Input(1), 1);
    }

    // Use Gumbel softmax trick to generate categorical samples.
    // See:
    // https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    // TODO(b/68769470): Switch to using a cumulative sum approach.
    auto softmax_entries =
        xla::Sub(logits, log_uniforms,
                 /*broadcast_dimensions=*/{0, class_dimension});

    xla::PrimitiveType xla_output_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(output_type(0), &xla_output_type));
    xla::XlaOp argmax = xla::ArgMax(softmax_entries, xla_output_type,
                                    /*axis=*/class_dimension);

    if (num_samples == 1 && !num_samples_is_dynamic) {
      argmax = xla::Reshape(argmax, {batch_size, 1});
    }

    ctx->SetOutput(0, argmax);
  }

  virtual xla::XlaOp GetLogUniforms(xla::Shape uniform_shape,
                                    xla::PrimitiveType type,
                                    XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_2(mht_2_v, 301, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "GetLogUniforms");

    xla::XlaBuilder* builder = ctx->builder();
    LOG_FIRST_N(WARNING, 1) << "Warning: Using tf.random.categorical with XLA"
                               " compilation will ignore seeds.";
    // We want a number in (0, 1) rather than [0, 1) or (0, 1]:
    // * log(-log(0)) is ∞.
    // * log(-log(1)) is -∞.
    auto uniforms = xla::RngUniform(
        xla::MinPositiveNormalValue(builder, type),
        xla::One(builder, uniform_shape.element_type()), uniform_shape);
    return xla::Log(-xla::Log(uniforms));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CategoricalOp);
};

// TODO(b/68769717): Rename this sampler to Categorical.
REGISTER_XLA_OP(Name("Multinomial").CompileTimeConstantInput("num_samples"),
                CategoricalOp);

class StatelessCategoricalOp : public CategoricalOp {
 public:
  explicit StatelessCategoricalOp(OpKernelConstruction* ctx)
      : CategoricalOp(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_3(mht_3_v, 329, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "StatelessCategoricalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  xla::XlaOp GetLogUniforms(xla::Shape uniform_shape, xla::PrimitiveType type,
                            XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_4(mht_4_v, 337, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "GetLogUniforms");

    xla::XlaOp seed = ctx->Input(2);

    xla::XlaBuilder* builder = ctx->builder();
    if (uniform_shape.element_type() == xla::BF16) {
      uniform_shape.set_element_type(xla::F32);
    }
    // We want a number in (0, 1) rather than [0, 1) or (0, 1]:
    // * log(-log(0)) is ∞.
    // * log(-log(1)) is -∞.
    xla::XlaOp uniforms = StatelessRngUniform(
        device_type_string_, seed, uniform_shape,
        xla::MinPositiveNormalValue(builder, uniform_shape.element_type()),
        xla::One(builder, uniform_shape.element_type()));
    return xla::ConvertElementType(xla::Log(-xla::Log(uniforms)), type);
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScategorical_opDTcc mht_5(mht_5_v, 357, "", "./tensorflow/compiler/tf2xla/kernels/categorical_op.cc", "Compile");

    TensorShape seed_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    CategoricalOp::Compile(ctx);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessCategoricalOp);
};

REGISTER_XLA_OP(Name("StatelessMultinomial")
                    .CompileTimeConstantInput("num_samples")
                    .TypeConstraint("T", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessCategoricalOp);

}  // anonymous namespace
}  // namespace tensorflow
