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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc() {
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

// Native XLA implementations of XLA Elu Ops

#include "tensorflow/compiler/tf2xla/kernels/elu_op.h"

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace xla {
XlaOp Elu(XlaOp x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Elu");

  const auto zero = ScalarLike(x, 0);
  const auto pred = Gt(x, zero);
  const auto expm1 = Expm1(x);
  return Select(pred, x, expm1);
}

XlaOp Selu(XlaOp x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Selu");

  const auto zero = ScalarLike(x, 0);
  const auto scale = ScalarLike(x, 1.0507009873554804934193349852946);
  const auto scale_alpha = ScalarLike(x, 1.7580993408473768599402175208123);
  const auto pred = Gt(x, zero);
  const auto expm1 = Expm1(x);
  return Select(pred, Mul(scale, x), Mul(scale_alpha, expm1));
}
}  // namespace xla

namespace tensorflow {
namespace {

class EluOp : public XlaOpKernel {
 public:
  explicit EluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "EluOp");
}
  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Compile");

    ctx->SetOutput(0, xla::Elu(ctx->Input(0)));
  }
};

class EluGradOp : public XlaOpKernel {
 public:
  explicit EluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_4(mht_4_v, 240, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "EluGradOp");
}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return lhs * (1 + rhs).
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_5(mht_5_v, 246, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    const auto zero = XlaHelpers::Zero(b, input_type(0));
    const auto one = XlaHelpers::One(b, input_type(0));
    const auto grad = ctx->Input(0);
    const auto activation = ctx->Input(1);
    const auto exp_grad = xla::Mul(grad, xla::Add(activation, one));
    const auto pred = xla::Gt(activation, zero);
    ctx->SetOutput(0, xla::Select(pred, grad, exp_grad));
  }
};

REGISTER_XLA_OP(Name("Elu"), EluOp);
REGISTER_XLA_OP(Name("EluGrad"), EluGradOp);

class SeluOp : public XlaOpKernel {
 public:
  explicit SeluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_6(mht_6_v, 266, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "SeluOp");
}
  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_7(mht_7_v, 271, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Compile");

    ctx->SetOutput(0, xla::Selu(ctx->Input(0)));
  }
};

class SeluGradOp : public XlaOpKernel {
 public:
  explicit SeluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_8(mht_8_v, 281, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "SeluGradOp");
}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return lhs * (1 + rhs).
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSelu_opDTcc mht_9(mht_9_v, 287, "", "./tensorflow/compiler/tf2xla/kernels/elu_op.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    const auto zero = XlaHelpers::Zero(b, input_type(0));
    const auto scale = XlaHelpers::FloatLiteral(b, input_type(0),
            1.0507009873554804934193349852946);
    const auto scale_alpha = XlaHelpers::FloatLiteral(b, input_type(0),
            1.7580993408473768599402175208123);
    const auto grad = ctx->Input(0);
    const auto activation = ctx->Input(1);
    const auto lin_grad = xla::Mul(grad, scale);
    const auto exp_grad = xla::Mul(grad, xla::Add(activation, scale_alpha));
    const auto pred = xla::Gt(activation, zero);
    ctx->SetOutput(0, xla::Select(pred, lin_grad, exp_grad));
  }
};

REGISTER_XLA_OP(Name("Selu"), SeluOp);
REGISTER_XLA_OP(Name("SeluGrad"), SeluGradOp);

}  // namespace
}  // namespace tensorflow
