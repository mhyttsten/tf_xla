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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/stateful_random_ops.h"

#include <cmath>

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
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
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {
namespace {

xla::BitGeneratorTy BitGen(Algorithm alg) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "BitGen");

  if (alg == RNG_ALG_PHILOX) {
    return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "lambda");

      state =
          xla::ConcatInDim(key.builder(), {xla::Reshape(key, {1}), state}, 0);
      xla::XlaOp result =
          xla::RngBitGenerator(xla::RandomAlgorithm::RNG_PHILOX, state, shape);
      xla::XlaOp data = xla::GetTupleElement(result, 1);
      xla::XlaOp new_state =
          xla::Slice(xla::GetTupleElement(result, 0), {1}, {3}, {1});
      return xla::RngOutput{data, new_state};
    };
  } else {
    return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "lambda");

      state = xla::ConcatScalars(key.builder(), {key, state});
      xla::XlaOp result = xla::RngBitGenerator(
          xla::RandomAlgorithm::RNG_THREE_FRY, state, shape);
      xla::XlaOp data = xla::GetTupleElement(result, 1);
      xla::XlaOp new_state = xla::Reshape(
          xla::Slice(xla::GetTupleElement(result, 0), {1}, {2}, {1}), {});
      return xla::RngOutput{data, new_state};
    };
  }
}

xla::RngOutput StatefulRngUniform(Algorithm alg, xla::XlaOp key,
                                  xla::XlaOp initial_state,
                                  const xla::Shape& shape, xla::XlaOp minval,
                                  xla::XlaOp maxval) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulRngUniform");

  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(
          key, initial_state, BitGen(alg), minval, maxval, shape);
    case xla::U32:
    case xla::S32:
    case xla::U64:
    case xla::S64:
      return UniformIntDistribution(key, initial_state, BitGen(alg), minval,
                                    maxval, shape);
    default:
      return {key.builder()->ReportError(xla::Unimplemented(
                  "Types other than F32, U32, S32, U64 and S64 "
                  "are not implemented by "
                  "StatefulRngUniform; got %s",
                  xla::primitive_util::LowercasePrimitiveTypeName(type))),
              initial_state};
  }
}

xla::RngOutput StatefulRngUniformFullInt(Algorithm alg, xla::XlaOp key,
                                         xla::XlaOp initial_state,
                                         const xla::Shape& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_4(mht_4_v, 275, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulRngUniformFullInt");

  xla::PrimitiveType type = shape.element_type();
  xla::RngOutput output = BitGen(alg)(key, initial_state, shape);
  switch (type) {
    case xla::U32:
    case xla::U64:
      return output;
    case xla::S32:
    case xla::S64:
      output.value = BitcastConvertType(output.value, type);
      return output;
    default:
      return {
          key.builder()->ReportError(xla::Unimplemented(
              "Types other than U32, S32, U64 and S64 are not implemented by "
              "StatefulRngUniformFullInt; got: %s",
              xla::primitive_util::LowercasePrimitiveTypeName(type))),
          initial_state};
  }
}

using SamplerReturnType = StatusOr<xla::RngOutput>;

int64_t GetMinStateSize(Algorithm alg) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_5(mht_5_v, 301, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "GetMinStateSize");

  if (alg == RNG_ALG_PHILOX) {
    return PHILOX_MIN_STATE_SIZE;
  }
  return THREEFRY_MIN_STATE_SIZE;
}

Status CheckStateShape(Algorithm alg, const TensorShape& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_6(mht_6_v, 311, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "CheckStateShape");

  if (shape.dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ", shape.dims());
  }
  auto state_size = shape.dim_size(0);
  auto min_state_size = GetMinStateSize(alg);
  if (state_size < min_state_size) {
    return errors::InvalidArgument("The size of the state must be at least ",
                                   min_state_size, "; got ", state_size);
  }
  return Status::OK();
}

std::pair<xla::XlaOp, xla::XlaOp> StateAndKeyFromVariable(Algorithm alg,
                                                          xla::XlaOp var) {
  if (alg == RNG_ALG_THREEFRY) {
    static constexpr int kStateSize = 1;
    auto state = BitcastConvertType(
        xla::Reshape(xla::Slice(var, {0}, {kStateSize}, {1}), {}), xla::U64);
    auto key = BitcastConvertType(
        xla::Reshape(xla::Slice(var, {kStateSize}, {kStateSize + 1}, {1}), {}),
        xla::U64);
    return std::make_pair(state, key);
  } else {
    static constexpr int kStateSize = 2;
    auto state =
        BitcastConvertType(xla::Slice(var, {0}, {kStateSize}, {1}), xla::U64);
    auto key = xla::Reshape(
        BitcastConvertType(xla::Slice(var, {kStateSize}, {kStateSize + 1}, {1}),
                           xla::U64),
        {});
    return std::make_pair(state, key);
  }
}

xla::XlaOp StateAndKeyToVariable(Algorithm alg, xla::XlaOp state,
                                 xla::XlaOp key) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_7(mht_7_v, 351, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StateAndKeyToVariable");

  auto builder = state.builder();
  if (alg == RNG_ALG_THREEFRY) {
    return ConcatScalars(builder, {state, key});
  } else {
    return ConcatInDim(builder, {state, xla::Reshape(key, {1})}, 0);
  }
}

// A helper function containing the common part of several kernels below.
// Precondition: 'algorithm' and 'shape' are compile-time constants.
Status CompileImpl(
    XlaOpKernelContext* ctx, int state_input_idx, int alg_input_idx,
    int shape_input_idx,
    std::function<SamplerReturnType(Algorithm, xla::XlaOp, xla::XlaOp,
                                    TensorShape)> const& sampler) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_8(mht_8_v, 369, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "CompileImpl");

  auto alg_shape = ctx->InputShape(alg_input_idx);
  if (alg_shape.dims() != 0) {
    return errors::InvalidArgument("algorithm must be of shape [], not ",
                                   alg_shape.DebugString());
  }
  xla::Literal alg_literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(alg_input_idx, &alg_literal));
  Algorithm alg = Algorithm(alg_literal.Get<int64_t>({}));
  if (!(alg == RNG_ALG_THREEFRY || alg == RNG_ALG_PHILOX)) {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }

  xla::XlaOp var;
  TensorShape var_shape;
  TF_RETURN_IF_ERROR(ctx->ReadVariableInput(
      state_input_idx, STATE_ELEMENT_DTYPE, &var_shape, &var));
  TF_RETURN_IF_ERROR(CheckStateShape(alg, var_shape));
  TensorShape shape;
  TF_RETURN_IF_ERROR(ctx->ConstantInputAsShape(shape_input_idx, &shape));
  xla::XlaOp state;
  xla::XlaOp key;
  std::tie(state, key) = StateAndKeyFromVariable(alg, var);
  auto status_or_value = sampler(alg, state, key, shape);
  if (!status_or_value.ok()) {
    return status_or_value.status();
  }
  xla::RngOutput value_state = status_or_value.ConsumeValueOrDie();
  state = value_state.state;
  ctx->SetOutput(0, value_state.value);
  var = StateAndKeyToVariable(alg, state, key);
  xla::PrimitiveType state_element_type;
  TF_RETURN_IF_ERROR(
      DataTypeToPrimitiveType(STATE_ELEMENT_DTYPE, &state_element_type));
  var = BitcastConvertType(var, state_element_type);
  TF_RETURN_IF_ERROR(
      ctx->AssignVariable(state_input_idx, STATE_ELEMENT_DTYPE, var));
  return Status::OK();
}

class StatefulUniformOp : public XlaOpKernel {
 public:
  explicit StatefulUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_9(mht_9_v, 414, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulUniformOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_10(mht_10_v, 421, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    auto sampler = [builder, this](Algorithm alg, xla::XlaOp state,
                                   xla::XlaOp key,
                                   TensorShape shape) -> SamplerReturnType {
      xla::Shape xla_shape;
      DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
      xla::PrimitiveType rng_primitive_type = xla_shape.element_type();
      xla::RngOutput uniform_state = StatefulRngUniform(
          alg, key, state, xla_shape,
          xla::ConstantR0WithType(builder, rng_primitive_type, 0.0),
          xla::ConstantR0WithType(builder, rng_primitive_type, 1.0));
      xla::XlaOp uniform = uniform_state.value;
      state = uniform_state.state;
      uniform = MaybeConvertF32ToBF16(uniform, dtype_);
      return {{uniform, state}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sampler));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformOp);
};

// TODO(wangpeng): Support plain float16 to get rid of the `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulUniform")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
                StatefulUniformOp);

class StatefulStandardNormalOp : public XlaOpKernel {
 public:
  explicit StatefulStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_11(mht_11_v, 464, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulStandardNormalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_12(mht_12_v, 471, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    auto sampler =
        // Needs explicit lambda return type because it fails to be inferred.
        [this](Algorithm alg, xla::XlaOp state, xla::XlaOp key,
               TensorShape shape) -> SamplerReturnType {
      xla::Shape xla_shape;
      DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
      xla::RngOutput value_state = xla::NormalFloatingPointDistribution(
          key, state, BitGen(alg), xla_shape);
      xla::XlaOp normal = MaybeConvertF32ToBF16(value_state.value, dtype_);
      return {{normal, value_state.state}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sampler));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulStandardNormalOp);
};

// TODO(wangpeng): Support plain float16 to get rid of the `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulStandardNormalV2")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
                StatefulStandardNormalOp);

class StatefulTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatefulTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_13(mht_13_v, 509, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulTruncatedNormalOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_14(mht_14_v, 516, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    auto sampler =
        // Needs explicit lambda return type because it fails to be inferred.
        [builder, this](Algorithm alg, xla::XlaOp state, xla::XlaOp key,
                        TensorShape shape) -> SamplerReturnType {
      xla::Shape xla_shape;
      DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

      xla::RngOutput uniform_result = StatefulRngUniform(
          alg, key, state, xla_shape,
          xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
          xla::One(builder, xla_shape.element_type()));
      xla::XlaOp uniform = uniform_result.value;
      state = uniform_result.state;
      xla::XlaOp truncated_normal = TruncatedNormal(uniform);
      truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
      return {{truncated_normal, state}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sampler));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulTruncatedNormalOp);
};

// TODO(wangpeng): Support plain float16 to get rid of the `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulTruncatedNormal")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
                StatefulTruncatedNormalOp);

class StatefulUniformIntOp : public XlaOpKernel {
 public:
  explicit StatefulUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_15(mht_15_v, 560, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulUniformIntOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_16(mht_16_v, 567, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    xla::XlaOp minval = ctx->Input(3);
    xla::XlaOp maxval = ctx->Input(4);
    auto sample_with_threefry =
        [minval, maxval, this](Algorithm alg, xla::XlaOp state, xla::XlaOp key,
                               TensorShape shape) -> SamplerReturnType {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype_, shape, &xla_shape));
      return StatefulRngUniform(alg, key, state, xla_shape, minval, maxval);
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformIntOp);
};

REGISTER_XLA_OP(Name("StatefulUniformInt")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}),
                StatefulUniformIntOp);

class StatefulUniformFullIntOp : public XlaOpKernel {
 public:
  explicit StatefulUniformFullIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_17(mht_17_v, 601, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "StatefulUniformFullIntOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_18(mht_18_v, 608, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    auto sample_with_threefry = [this](Algorithm alg, xla::XlaOp state,
                                       xla::XlaOp key,
                                       TensorShape shape) -> SamplerReturnType {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype_, shape, &xla_shape));
      return StatefulRngUniformFullInt(alg, key, state, xla_shape);
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformFullIntOp);
};

REGISTER_XLA_OP(Name("StatefulUniformFullInt")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}),
                StatefulUniformFullIntOp);

xla::XlaOp IncreaseCounter(Algorithm const& alg, xla::XlaOp counter,
                           xla::XlaOp delta) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_19(mht_19_v, 638, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "IncreaseCounter");

  // Multiplying 256 to be consistent with the CPU/GPU kernels
  delta = delta * ConstantR0WithType(delta.builder(), xla::U64, 256);
  if (alg == RNG_ALG_PHILOX) {
    return xla::PhiloxIncreaseCounter(counter, delta);
  } else {
    return counter + delta;
  }
}

xla::XlaOp PadRight(xla::XlaOp a, int n) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_20(mht_20_v, 651, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "PadRight");

  return xla::Pad(a, xla::ScalarLike(a, 0),
                  xla::MakeEdgePaddingConfig({{0, n}}));
}

template <typename AlgEnumType = int64_t, bool read_old_value = false>
class RngSkipOp : public XlaOpKernel {
 public:
  explicit RngSkipOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_21(mht_21_v, 662, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "RngSkipOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSstateful_random_opsDTcc mht_22(mht_22_v, 667, "", "./tensorflow/compiler/tf2xla/kernels/stateful_random_ops.cc", "Compile");

    const int state_input_idx = 0;
    const int alg_input_idx = 1;
    const int delta_input_idx = 2;
    xla::XlaOp var;
    TensorShape var_shape;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(state_input_idx, STATE_ELEMENT_DTYPE,
                                          &var_shape, &var));
    xla::Literal alg_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(alg_input_idx, &alg_literal));
    Algorithm alg = Algorithm(alg_literal.Get<AlgEnumType>({}));
    OP_REQUIRES(ctx, alg == RNG_ALG_THREEFRY || alg == RNG_ALG_PHILOX,
                errors::InvalidArgument("Unsupported algorithm id: ", alg));
    OP_REQUIRES_OK(ctx, CheckStateShape(alg, var_shape));
    if (read_old_value) {
      auto counter_size = GetCounterSize(alg);
      xla::XlaOp output = var;
      if (RNG_MAX_COUNTER_SIZE > counter_size) {
        // Because the size of `var` depends on the algorithm while we want the
        // output to have a fixed size (to help shape inference), we fix the
        // output size to be the maximal state size among algorithms, and right-
        // pad it with zeros if var's size is smaller than that.
        output = PadRight(output, RNG_MAX_COUNTER_SIZE - counter_size);
      }
      ctx->SetOutput(0, output);
    }
    xla::XlaOp counter;
    xla::XlaOp key;
    std::tie(counter, key) = StateAndKeyFromVariable(alg, var);
    xla::XlaOp delta = ctx->Input(delta_input_idx);
    delta = BitcastConvertType(delta, xla::U64);
    auto new_counter = IncreaseCounter(alg, counter, delta);
    var = StateAndKeyToVariable(alg, new_counter, key);
    xla::PrimitiveType state_element_type;
    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(STATE_ELEMENT_DTYPE, &state_element_type));
    var = BitcastConvertType(var, state_element_type);
    OP_REQUIRES_OK(
        ctx, ctx->AssignVariable(state_input_idx, STATE_ELEMENT_DTYPE, var));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RngSkipOp);
};

REGISTER_XLA_OP(Name("RngSkip").CompileTimeConstantInput("algorithm"),
                RngSkipOp<>);

using RngReadAndSkipOp = RngSkipOp<int32, true>;

REGISTER_XLA_OP(Name("RngReadAndSkip").CompileTimeConstantInput("alg"),
                RngReadAndSkipOp);

}  // namespace
}  // namespace tensorflow
