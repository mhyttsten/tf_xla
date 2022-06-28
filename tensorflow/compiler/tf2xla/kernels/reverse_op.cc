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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc() {
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

// XLA-specific reverse Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class ReverseOp : public XlaOpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/tf2xla/kernels/reverse_op.cc", "ReverseOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/reverse_op.cc", "Compile");

    // r = tf.reverse(x, revdims)
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape revd_shape = ctx->InputShape(1);
    // Validate input sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(revd_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        revd_shape.DebugString()));
    OP_REQUIRES(ctx, revd_shape.num_elements() == x_shape.dims(),
                errors::InvalidArgument("axes ", revd_shape.DebugString(),
                                        " must have same number of elements as"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    if (revd_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    // XlaBuilder::Rev() requires concrete values for dimensions arg.
    xla::Literal lax;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &lax));

    std::vector<int64_t> dimensions;
    for (int d = 0; d < x_shape.dims(); ++d) {
      if (lax.Get<bool>({d})) {
        dimensions.push_back(d);
      }
    }

    ctx->SetOutput(0, xla::Rev(ctx->Input(0), dimensions));
  }
};

REGISTER_XLA_OP(Name("Reverse").CompileTimeConstantInput("dims"), ReverseOp);

class ReverseV2Op : public XlaOpKernel {
 public:
  explicit ReverseV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/tf2xla/kernels/reverse_op.cc", "ReverseV2Op");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreverse_opDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/tf2xla/kernels/reverse_op.cc", "Compile");

    // r = tf.reverse(x, axes)
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape axes_shape = ctx->InputShape(1);
    // Validate input sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(axes_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        axes_shape.DebugString()));
    OP_REQUIRES(ctx, axes_shape.num_elements() <= x_shape.dims(),
                errors::InvalidArgument("axes ", axes_shape.DebugString(),
                                        " can not have more elements"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    // Reverse is a no-op if axes argument is empty.
    if (axes_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    // XlaBuilder::Rev() requires concrete values for dimensions arg.
    std::vector<int64_t> axes;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &axes));

    // witnessed_axes is used to ensure that the same axis is not marked to be
    // reversed multiple times.
    absl::InlinedVector<bool, 8> witnessed_axes(x_shape.dims(), false);

    for (int d = 0; d < axes.size(); ++d) {
      OP_REQUIRES(
          ctx, (-x_shape.dims() <= axes[d]) && (axes[d] < x_shape.dims()),
          errors::InvalidArgument(axes[d], " is out of range [-",
                                  x_shape.dims(), ", ", x_shape.dims(), ")."));
      // Axes can be negative and are shifted to the canonical index before
      // being lowered to HLO.
      if (axes[d] < 0) {
        axes[d] += x_shape.dims();
      }
      OP_REQUIRES(ctx, !witnessed_axes[axes[d]],
                  errors::InvalidArgument("canonicalized axis ", axes[d],
                                          " was repeated."));
      witnessed_axes[axes[d]] = true;
    }

    ctx->SetOutput(0, xla::Rev(ctx->Input(0), axes));
  }
};

REGISTER_XLA_OP(Name("ReverseV2").CompileTimeConstantInput("axis"),
                ReverseV2Op);

}  // namespace
}  // namespace tensorflow
