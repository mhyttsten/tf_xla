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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc() {
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
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class CastOp : public XlaOpKernel {
 public:
  explicit CastOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/cast_op.cc", "CastOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &dst_dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(src_dtype_, &src_type_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dst_dtype_, &dst_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Truncate", &use_truncation_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/tf2xla/kernels/cast_op.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    if (src_dtype_ == dst_dtype_) {
      output = input;
    } else if (dst_dtype_ == DT_BOOL) {
      output = xla::Ne(input, XlaHelpers::Zero(builder, src_dtype_));
    } else if (xla::primitive_util::IsComplexType(src_type_) &&
               !xla::primitive_util::IsComplexType(dst_type_)) {
      // As in cast_op.h, we replicate the numpy behavior of truncating the
      // imaginary part.
      output = xla::ConvertElementType(xla::Real(input), dst_type_);
    } else {
      if (use_truncation_) {
        OP_REQUIRES(
            ctx,
            xla::primitive_util::IsFloatingPointType(src_type_) &&
                xla::primitive_util::IsFloatingPointType(dst_type_),
            errors::Unimplemented("Truncate attribute is only "
                                  "implemented for floating point datatypes."));
        int mantissa_difference =
            xla::primitive_util::SignificandWidth(src_type_) -
            xla::primitive_util::SignificandWidth(dst_type_);
        OP_REQUIRES(ctx, mantissa_difference > 0,
                    errors::Unimplemented(
                        "Truncate attribute is only implemented in cases where "
                        "dst datatype "
                        "has fewer mantissa bits than the src datatype"));
        int src_bitwidth = xla::primitive_util::BitWidth(src_type_);

        // Bitcast to same-width integer, mask off the LSBs, bitcast back to the
        // source datatype.
        int64_t mask = ~((1L << mantissa_difference) - 1);
        xla::PrimitiveType same_width_int =
            xla::primitive_util::UnsignedIntegralTypeForBitWidth(src_bitwidth);
        OP_REQUIRES(ctx, same_width_int != xla::PRIMITIVE_TYPE_INVALID,
                    errors::Unimplemented("Unexpected type bitwidth"));
        input = xla::BitcastConvertType(
            xla::And(
                xla::BitcastConvertType(input, same_width_int),
                ::tensorflow::IntegerLiteral(builder, same_width_int, mask)),
            src_type_);
      }
      output = xla::ConvertElementType(input, dst_type_);
    }

    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;
  bool use_truncation_;

  TF_DISALLOW_COPY_AND_ASSIGN(CastOp);
};

REGISTER_XLA_OP(Name("Cast"), CastOp);

class BitcastOp : public XlaOpKernel {
 public:
  explicit BitcastOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc mht_2(mht_2_v, 282, "", "./tensorflow/compiler/tf2xla/kernels/cast_op.cc", "BitcastOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("type", &dst_dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(src_dtype_, &src_type_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dst_dtype_, &dst_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScast_opDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/tf2xla/kernels/cast_op.cc", "Compile");

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    if (src_dtype_ == dst_dtype_) {
      output = input;
      ctx->SetOutput(0, output);
      return;
    }
    // Error out if the bitcast has a complex source or destination type and
    // the bitcast is not trivial.
    OP_REQUIRES(ctx,
                !xla::primitive_util::IsComplexType(src_type_) &&
                    !xla::primitive_util::IsComplexType(dst_type_),
                errors::Unimplemented("Complex types not supported."));
    auto input_bit_width = xla::primitive_util::BitWidth(src_type_);
    auto output_bit_width = xla::primitive_util::BitWidth(dst_type_);

    OP_REQUIRES(ctx,
                output_bit_width % input_bit_width == 0 ||
                    input_bit_width % output_bit_width == 0,
                errors::InvalidArgument(
                    "Neither bit width is a multiple of the other."));
    output = xla::BitcastConvertType(input, dst_type_);
    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(BitcastOp);
};

REGISTER_XLA_OP(Name("Bitcast"), BitcastOp);

}  // anonymous namespace
}  // namespace tensorflow
