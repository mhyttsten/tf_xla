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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc() {
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

#include <cmath>

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

namespace {

xla::BitGeneratorTy GetBitGeneratorForDevice(
    absl::string_view device_type_string) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_type_string: \"" + std::string(device_type_string.data(), device_type_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "GetBitGeneratorForDevice");

  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "lambda");

      std::tie(state, key) = xla::ScramblePhiloxKey(key);
      xla::XlaOp philox_state =
          xla::ConcatInDim(key.builder(), {xla::Reshape(key, {1}), state}, 0);
      xla::XlaOp result = xla::RngBitGenerator(xla::RandomAlgorithm::RNG_PHILOX,
                                               philox_state, shape);
      return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                            /*state=*/xla::GetTupleElement(result, 0)};
    };
  }
  return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "lambda");

    state = xla::ConcatScalars(key.builder(), {key, state});
    xla::XlaOp result =
        xla::RngBitGenerator(xla::RandomAlgorithm::RNG_DEFAULT, state, shape);
    return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                          /*state=*/xla::GetTupleElement(result, 0)};
  };
}

}  // namespace

xla::XlaOp MaybeConvertF32ToBF16(xla::XlaOp input, DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_3(mht_3_v, 246, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "MaybeConvertF32ToBF16");

  if (dtype == DT_BFLOAT16) {
    xla::XlaBuilder* builder = input.builder();
    xla::XlaOp output = xla::BitcastConvertType(input, xla::U32) &
                        xla::ConstantR0<uint32>(builder, 0xFFFF0000);
    return xla::ConvertElementType(xla::BitcastConvertType(output, xla::F32),
                                   xla::BF16);
  } else {
    return input;
  }
}

xla::XlaOp StatelessRngUniform(absl::string_view device_type_string,
                               xla::XlaOp seeds, const xla::Shape& shape,
                               xla::XlaOp minval, xla::XlaOp maxval) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device_type_string: \"" + std::string(device_type_string.data(), device_type_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_4(mht_4_v, 264, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRngUniform");

  xla::XlaBuilder* builder = seeds.builder();

  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
  xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(
                 key, initial_state,
                 GetBitGeneratorForDevice(device_type_string), minval, maxval,
                 shape)
          .value;
    case xla::S32:  // fall through
    case xla::S64:
      return UniformIntDistribution(
                 key, initial_state,
                 GetBitGeneratorForDevice(device_type_string), minval, maxval,
                 shape)
          .value;
      break;
    default:
      return builder->ReportError(xla::Unimplemented(
          "Types other than F32, S32 and S64 are not implemented by "
          "StatelessRngUniform; got %s",
          xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

namespace {

xla::XlaOp StatelessRngUniformFullInt(absl::string_view device_type_string,
                                      xla::XlaOp seeds,
                                      const xla::Shape& shape) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device_type_string: \"" + std::string(device_type_string.data(), device_type_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_5(mht_5_v, 304, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRngUniformFullInt");

  xla::XlaBuilder* builder = seeds.builder();

  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
  xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  xla::RngOutput output =
      GetBitGeneratorForDevice(device_type_string)(key, initial_state, shape);
  switch (type) {
    case xla::U32:
    case xla::U64:
      return output.value;
    case xla::S32:
    case xla::S64:
      return BitcastConvertType(output.value, type);
    default:
      return builder->ReportError(xla::Unimplemented(
          "Types other than U32, S32, U64 and S64 are not implemented by "
          "StatelessRngUniformFullInt; got: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

class StatelessRandomUniformOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_6(mht_6_v, 336, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRandomUniformOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_7(mht_7_v, 343, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
    xla::PrimitiveType rng_primitive_type = xla_shape.element_type();

    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::ConstantR0WithType(builder, rng_primitive_type, 0.0),
        xla::ConstantR0WithType(builder, rng_primitive_type, 1.0));
    uniform = MaybeConvertF32ToBF16(uniform, dtype_);
    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformOp);

class StatelessRandomUniformIntOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_8(mht_8_v, 389, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRandomUniformIntOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_9(mht_9_v, 396, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    TensorShape minval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        minval_shape.DebugString()));
    TensorShape maxval_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        maxval_shape.DebugString()));

    xla::XlaOp seed = ctx->Input(1);
    xla::XlaOp minval = ctx->Input(2);
    xla::XlaOp maxval = ctx->Input(3);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));
    xla::XlaOp uniform = StatelessRngUniform(device_type_string_, seed,
                                             xla_shape, minval, maxval);

    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformIntOp);
};

// TODO(phawkins): generalize to non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniformInt")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32, DT_INT64})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformIntOp);

class StatelessRandomUniformFullIntOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformFullIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_10(mht_10_v, 446, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRandomUniformFullIntOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_11(mht_11_v, 453, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));

    xla::XlaOp seed = ctx->Input(1);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));
    xla::XlaOp uniform =
        StatelessRngUniformFullInt(device_type_string_, seed, xla_shape);

    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformFullIntOp);
};

// TODO(phawkins): generalize to non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniformFullInt")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32, DT_INT64})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformFullIntOp);

class StatelessRandomNormalOp : public XlaOpKernel {
 public:
  explicit StatelessRandomNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_12(mht_12_v, 493, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessRandomNormalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_13(mht_13_v, 500, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);
    xla::Shape xla_shape;

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    xla::XlaBuilder* builder = seed.builder();
    xla::XlaOp seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    xla::XlaOp seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
    xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);

    xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
    xla::XlaOp normal =
        xla::NormalFloatingPointDistribution(
            key, initial_state, GetBitGeneratorForDevice(device_type_string_),
            xla_shape)
            .value;
    normal = MaybeConvertF32ToBF16(normal, dtype_);
    ctx->SetOutput(0, normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomNormalOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_14(mht_14_v, 550, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessTruncatedNormalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_15(mht_15_v, 557, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal = TruncatedNormal(uniform);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessTruncatedNormalOp);

class StatelessParameterizedTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessParameterizedTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_16(mht_16_v, 600, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "StatelessParameterizedTruncatedNormalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateless_random_opsDTcc mht_17(mht_17_v, 607, "", "./tensorflow/compiler/tf2xla/kernels/stateless_random_ops.cc", "Compile");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    auto bcasted_means = BroadcastTo(ctx->Input(2), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_means.status());
    auto means = bcasted_means.ValueOrDie();

    auto bcasted_stddevs = BroadcastTo(ctx->Input(3), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_stddevs.status());
    auto stddevs = bcasted_stddevs.ValueOrDie();

    auto bcasted_minvals = BroadcastTo(ctx->Input(4), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_minvals.status());
    auto minvals = bcasted_minvals.ValueOrDie();

    auto bcasted_maxvals = BroadcastTo(ctx->Input(5), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_maxvals.status());
    auto maxvals = bcasted_maxvals.ValueOrDie();

    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal =
        ParameterizedTruncatedNormal(uniform, means, stddevs, minvals, maxvals);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessParameterizedTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("StatelessParameterizedTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessParameterizedTruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
