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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_opsDTcc() {
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

// Native XLA implementations of simple unary Ops

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

#define XLAJIT_MAKE_UNARY(NAME, COMPUTATION)                           \
  class NAME##Op : public XlaOpKernel {                                \
   public:                                                             \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {} \
    void Compile(XlaOpKernelContext* ctx) {                            \
      xla::XlaBuilder* b = ctx->builder();                             \
      (void)b;                                                         \
      xla::XlaOp x = ctx->Input(0);                                    \
      xla::XlaOp y = COMPUTATION;                                      \
      ctx->SetOutput(0, y);                                            \
    }                                                                  \
  };                                                                   \
  REGISTER_XLA_OP(Name(#NAME), NAME##Op);

XLAJIT_MAKE_UNARY(ComplexAbs, xla::Abs(x));

XLAJIT_MAKE_UNARY(Angle, xla::Atan2(xla::Imag(x), xla::Real(x)));

XLAJIT_MAKE_UNARY(Conj, xla::Conj(x));

// Return x if x>0, otherwise -x.
REGISTER_XLA_OP(Name("Abs"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Acos, xla::Acos(x));
XLAJIT_MAKE_UNARY(Acosh, xla::Acosh(x));
XLAJIT_MAKE_UNARY(Asin, xla::Asin(x))
XLAJIT_MAKE_UNARY(Asinh, xla::Asinh(x));
REGISTER_XLA_OP(Name("Atan"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Atanh, xla::Atanh(x));
REGISTER_XLA_OP(Name("Ceil"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Cos"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Cosh, xla::Cosh(x));
XLAJIT_MAKE_UNARY(Sin, xla::Sin(x));
REGISTER_XLA_OP(Name("Exp"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Expm1"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Floor"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsFinite"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsInf"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsNan"), MlirXlaOpKernel);
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, xla::ScalarLike(x, 1.0) / x);
REGISTER_XLA_OP(Name("Reciprocal"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Log, xla::Log(x));
REGISTER_XLA_OP(Name("Log1p"), MlirXlaOpKernel);

XLAJIT_MAKE_UNARY(Invert, xla::Not(x));
XLAJIT_MAKE_UNARY(LogicalNot, xla::Not(x));
XLAJIT_MAKE_UNARY(PopulationCount,
                  xla::ConvertElementType(xla::PopulationCount(x), xla::U8));
XLAJIT_MAKE_UNARY(Neg, -x);

XLAJIT_MAKE_UNARY(Rint, xla::RoundToEven(x));
XLAJIT_MAKE_UNARY(Round, xla::RoundToEven(x));

REGISTER_XLA_OP(Name("Rsqrt"), MlirXlaOpKernel);

REGISTER_XLA_OP(Name("Sigmoid"), MlirXlaOpKernel);

// Returns NaN if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
REGISTER_XLA_OP(Name("Sign"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Sinh, xla::Sinh(x));

static xla::XlaOp Softplus(xla::XlaBuilder* b, xla::XlaOp features) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_opsDTcc mht_0(mht_0_v, 265, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops.cc", "Softplus");

  return b->ReportErrorOrReturn([&]() -> StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(features));
    xla::XlaOp threshold =
        Log(xla::Epsilon(b, shape.element_type())) + ScalarLike(features, 2.0);
    // Value above which exp(x) may overflow, but softplus(x) == x
    // is within machine epsilon.
    xla::XlaOp too_large = Gt(features, -threshold);
    // Value below which exp(x) may underflow, but softplus(x) == exp(x)
    // is within machine epsilon.
    xla::XlaOp too_small = Lt(features, threshold);
    xla::XlaOp features_exp = Exp(features);
    xla::XlaOp output =
        Select(too_large, features,
               Select(too_small, features_exp, Log1p(features_exp)));
    return output;
  });
}
XLAJIT_MAKE_UNARY(Softplus, Softplus(b, x));

// softsign(x) = x / (abs(x) + 1)
XLAJIT_MAKE_UNARY(Softsign, x / (xla::Abs(x) + xla::ScalarLike(x, 1.0)));
REGISTER_XLA_OP(Name("Sqrt"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Square, x* x);
XLAJIT_MAKE_UNARY(Tan, xla::Tan(x));
REGISTER_XLA_OP(Name("Tanh"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Real"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Imag"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Erf, xla::Erf(x));
XLAJIT_MAKE_UNARY(Erfc, xla::Erfc(x));
XLAJIT_MAKE_UNARY(Erfinv, xla::ErfInv(x));
// ndtri = sqrt(2) * erfinv(2 * x - 1)
XLAJIT_MAKE_UNARY(Ndtri, xla::ScalarLike(x, std::sqrt(2.0)) *
                             xla::ErfInv(xla::ScalarLike(x, 2.0) * x -
                                         xla::ScalarLike(x, 1.0)));
REGISTER_XLA_OP(Name("Lgamma"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Digamma, xla::Digamma(x));
XLAJIT_MAKE_UNARY(BesselI0e, xla::BesselI0e(x));
XLAJIT_MAKE_UNARY(BesselI1e, xla::BesselI1e(x));

}  // namespace
}  // namespace tensorflow
