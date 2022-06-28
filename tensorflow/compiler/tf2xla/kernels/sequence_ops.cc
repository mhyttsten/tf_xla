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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc() {
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

// XLA-specific sequence and range Ops.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

// The type-specific part of the implementation of Range.
template <typename T>
StatusOr<xla::XlaOp> CreateRangeTensor(const xla::LiteralSlice& start_literal,
                                       const xla::LiteralSlice& limit_literal,
                                       const xla::LiteralSlice& delta_literal,
                                       xla::XlaBuilder* builder) {
  T start = start_literal.Get<T>({});
  T limit = limit_literal.Get<T>({});
  T delta = delta_literal.Get<T>({});

  if (delta == 0) {
    return errors::InvalidArgument("Requires delta != 0: ", delta);
  }
  if (delta > 0) {
    if (start > limit) {
      return errors::InvalidArgument(
          "Requires start <= limit when delta > 0: ", start, "/", limit);
    }
  } else {
    if (start < limit) {
      return errors::InvalidArgument(
          "Requires start >= limit when delta < 0: ", start, "/", limit);
    }
  }
  int64_t size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));

  return xla::ConstantR0(builder, start) +
         xla::ConstantR0(builder, delta) *
             xla::Iota(builder, xla::primitive_util::NativeToPrimitiveType<T>(),
                       size);
}

class RangeOp : public XlaOpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc mht_0(mht_0_v, 241, "", "./tensorflow/compiler/tf2xla/kernels/sequence_ops.cc", "RangeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/tf2xla/kernels/sequence_ops.cc", "Compile");

    const TensorShape start_in_shape = ctx->InputShape(0);
    const TensorShape limit_in_shape = ctx->InputShape(1);
    const TensorShape delta_in_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(limit_in_shape),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(delta_in_shape),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in_shape.DebugString()));
    xla::Literal start, limit, delta;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            0, &start, xla::ValueInferenceMode::kLowerBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            1, &limit, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &delta));

    DataType type = input_type(0);
    StatusOr<xla::XlaOp> output;
    switch (type) {
      case DT_INT32:
        output = CreateRangeTensor<int32>(start, limit, delta, ctx->builder());
        break;
      case DT_INT64:
        output =
            CreateRangeTensor<int64_t>(start, limit, delta, ctx->builder());
        break;
      case DT_FLOAT:
        output = CreateRangeTensor<float>(start, limit, delta, ctx->builder());
        break;
      case DT_DOUBLE:
        output = CreateRangeTensor<double>(start, limit, delta, ctx->builder());
        break;
      default:
        output = errors::InvalidArgument("Invalid type for Range ",
                                         DataTypeString(type));
    }
    OP_REQUIRES_OK(ctx, output.status());
    bool start_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(0, &start_is_dynamic));
    bool limit_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(1, &limit_is_dynamic));

    if (start_is_dynamic || limit_is_dynamic) {
      xla::XlaOp delta = ctx->Input(2);
      xla::XlaOp limit = ctx->Input(1);
      xla::XlaOp start = ctx->Input(0);
      if (type == DT_INT32 || type == DT_INT64) {
        auto dynamic_size = (xla::Abs(limit - start) + xla::Abs(delta) -
                             xla::One(ctx->builder(), ctx->input_xla_type(0))) /
                            xla::Abs(delta);
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.ValueOrDie(), dynamic_size, 0);
      } else {
        auto dynamic_size = (xla::Ceil(xla::Abs((limit - start) / delta)));
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.ValueOrDie(), dynamic_size, 0);
      }
    }

    ctx->SetOutput(0, output.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("Range")
                    .CompileTimeConstantInput("start")
                    .CompileTimeConstantInput("limit")
                    .CompileTimeConstantInput("delta"),
                RangeOp);

class LinSpaceOp : public XlaOpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc mht_2(mht_2_v, 326, "", "./tensorflow/compiler/tf2xla/kernels/sequence_ops.cc", "LinSpaceOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsequence_opsDTcc mht_3(mht_3_v, 331, "", "./tensorflow/compiler/tf2xla/kernels/sequence_ops.cc", "Compile");

    const TensorShape start_in_shape = ctx->InputShape("start");
    const TensorShape stop_in_shape = ctx->InputShape("stop");
    const TensorShape num_in_shape = ctx->InputShape("num");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(stop_in_shape),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_in_shape),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in_shape.DebugString()));

    int64_t num;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("num", &num));
    OP_REQUIRES(ctx, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    xla::XlaOp start = ctx->Input("start");
    xla::XlaOp stop = ctx->Input("stop");
    xla::XlaOp iota = xla::Iota(ctx->builder(), ctx->output_xla_type(0), num);
    xla::XlaOp step =
        (stop - start) / xla::ScalarLike(start, (num > 1 ? num - 1 : num));
    xla::XlaOp result = iota * step + start;
    if (num > 1) {
      // According to linspace spec, start has to be the first element and end
      // has to be last element.
      xla::XlaOp mask = xla::Iota(ctx->builder(), xla::S64, num);
      xla::XlaOp eq = xla::Eq(mask, xla::ScalarLike(mask, num - 1));
      result = xla::Select(eq, stop, result);
    }
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("LinSpace").CompileTimeConstantInput("num"), LinSpaceOp);

}  // namespace
}  // namespace tensorflow
