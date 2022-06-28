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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSarg_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSarg_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSarg_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// This OpKernel implements the _Arg Op for XLA JIT devices. It
// associates its output with one of the arguments to a
// subcomputation.
class XlaArgOp : public XlaOpKernel {
 public:
  explicit XlaArgOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSarg_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/arg_op.cc", "XlaArgOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSarg_opDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/arg_op.cc", "Compile");

    // If 'frame' is non-null, this is a function call inside an outer JIT
    // compilation. Use the usual implementation of _Arg.
    auto frame = ctx->call_frame();
    if (frame != nullptr) {
      const Tensor* val;
      OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
      // Types that cannot be copied using memcpy (like DT_STRING) are wrapped
      // in a DT_UINT8 and hence the type mismatches. Skip the test in such
      // cases. See XlaOpKernelContext::SetOutputExpression for details.
      if (DataTypeCanUseMemcpy(dtype_)) {
        OP_REQUIRES(ctx, val->dtype() == dtype_,
                    errors::InvalidArgument(
                        "Type mismatch: actual ", DataTypeString(val->dtype()),
                        " vs. expect ", DataTypeString(dtype_)));
      }
      // Forwards the argument from the frame.
      ctx->op_kernel_context()->set_output(0, *val);
      return;
    }

    const XlaExpression& arg = ctx->xla_context()->args()[index_];
    OP_REQUIRES(ctx, arg.kind() != XlaExpression::Kind::kInvalid,
                errors::InvalidArgument("Invalid/missing argument expression"));
    if (ctx->expected_output_dtype(0) == DT_VARIANT) {
      ctx->SetTensorListOutput(0, arg.handle());
    } else if (arg.value_bound().has_value()) {
      // The argument has a bound attached to it, call SetBound op on the
      // argument.
      xla::XlaBuilder* builder = ctx->builder();
      auto input_op = arg.AsXlaOp(builder);

      // We pass two pieces of information to SetBound:
      // Bound - The upper-bounds of the argument's values.
      //
      // Dynamism - Whether or not each individual value is dynamic. If this
      // is false, it means value with same tensor index in the argument is
      // static, and it's upper-bound is same as lower-bound and also same as
      // the static value itself.
      //
      // E.g.,:
      // When we have an argument `arg` with shape s32[3], bound = [1, 2, 3] and
      // dynamism = [false, false, true]
      //
      // We know that:
      //  arg[0] is a static value, its value is 1
      //  arg[1] is a static value, its value is 2
      //  arg[2] is a dynamic value, its value is unknown at compile time, but
      //  its upper-bound is known to be 3.
      //
      // Note that `arg` is still considered dynamic as long as one element
      // inside is dynamic, therefore the argument node can't be constant folded
      // into a constant node.
      xla::Literal bound = HostTensorToLiteral(*arg.value_bound()).ValueOrDie();
      xla::Literal dynamism =
          HostTensorToLiteral(*arg.value_dynamism()).ValueOrDie();
      xla::Literal tuple = xla::LiteralUtil::MakeTupleOwned(
          std::move(bound), std::move(dynamism));
      ctx->SetOutput(
          0, xla::CustomCall(builder, "SetBound", {input_op},
                             builder->GetShape(input_op).ValueOrDie(), "",
                             false, {}, &tuple));
      return;
    } else {
      ctx->SetOutputExpression(0, arg);
    }
  }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaArgOp);
};

REGISTER_XLA_OP(
    Name("_Arg").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    XlaArgOp);

}  // namespace tensorflow
