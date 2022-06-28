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
class MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc {
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
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/gradients/grad_helper.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace ops {
namespace {

// Logical operations have no gradients.
REGISTER_NO_GRADIENT_OP("Less");
REGISTER_NO_GRADIENT_OP("LessEqual");
REGISTER_NO_GRADIENT_OP("Greater");
REGISTER_NO_GRADIENT_OP("GreaterEqual");
REGISTER_NO_GRADIENT_OP("Equal");
REGISTER_NO_GRADIENT_OP("ApproximateEqual");
REGISTER_NO_GRADIENT_OP("NotEqual");
REGISTER_NO_GRADIENT_OP("LogicalAnd");
REGISTER_NO_GRADIENT_OP("LogicalOr");
REGISTER_NO_GRADIENT_OP("LogicalNot");
REGISTER_NO_GRADIENT_OP("Floor");

// Conjugate helper function returns the conjugate of an Output if it
// is complex valued.
Output ConjugateHelper(const Scope& scope, const Output& out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_0(mht_0_v, 213, "", "./tensorflow/cc/gradients/math_grad.cc", "ConjugateHelper");

  DataType dtype = out.type();
  if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
    return Conj(scope, out);
  } else {
    return out;
  }
}

// TODO(andydavis) Add control dependencies to gradient functions (as needed).

Status AbsGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_1(mht_1_v, 229, "", "./tensorflow/cc/gradients/math_grad.cc", "AbsGrad");

  // dx = dy * sign(x)
  grad_outputs->push_back(Mul(scope, grad_inputs[0], Sign(scope, op.input(0))));
  return scope.status();
}
REGISTER_GRADIENT_OP("Abs", AbsGrad);

Status NegGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_2(mht_2_v, 241, "", "./tensorflow/cc/gradients/math_grad.cc", "NegGrad");

  // dx = -dy;
  grad_outputs->push_back(Neg(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Neg", NegGrad);

Status InvGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_3(mht_3_v, 253, "", "./tensorflow/cc/gradients/math_grad.cc", "InvGrad");

  // Use the built-in operator.
  grad_outputs->push_back(
      internal::ReciprocalGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Inv", InvGrad);
REGISTER_GRADIENT_OP("Reciprocal", InvGrad);

Status SquareGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_4(mht_4_v, 267, "", "./tensorflow/cc/gradients/math_grad.cc", "SquareGrad");

  // dy/dx = (2 * x)
  auto two = Cast(scope, Const(scope, 2), op.input(0).type());
  auto dydx = Mul(scope, two, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Square", SquareGrad);

Status SqrtGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_5(mht_5_v, 283, "", "./tensorflow/cc/gradients/math_grad.cc", "SqrtGrad");

  // Use the built-in operator.
  grad_outputs->push_back(
      internal::SqrtGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sqrt", SqrtGrad);

Status RsqrtGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_6(mht_6_v, 296, "", "./tensorflow/cc/gradients/math_grad.cc", "RsqrtGrad");

  // Use the built-in operator.
  grad_outputs->push_back(
      internal::RsqrtGrad(scope, op.output(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Rsqrt", RsqrtGrad);

Status ExpGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_7(mht_7_v, 309, "", "./tensorflow/cc/gradients/math_grad.cc", "ExpGrad");

  // dy/dx = exp(x) = y
  // grad(x) = grad(y) * conj(dy/dx)
  //         = grad(y) * conj(y)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, op.output(0))));
  return scope.status();
}
REGISTER_GRADIENT_OP("Exp", ExpGrad);

Status Expm1Grad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_8(mht_8_v, 324, "", "./tensorflow/cc/gradients/math_grad.cc", "Expm1Grad");

  // y = expm1(x)
  // dy/dx = exp(x)
  auto dydx = Exp(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Expm1", Expm1Grad);

Status LogGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_9(mht_9_v, 340, "", "./tensorflow/cc/gradients/math_grad.cc", "LogGrad");

  // y = log(x)
  // dy/dx = 1 / x
  auto dydx = Reciprocal(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Log", LogGrad);

Status Log1pGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_10(mht_10_v, 356, "", "./tensorflow/cc/gradients/math_grad.cc", "Log1pGrad");

  // y = log1p(x)
  // dy/dx = 1 / (1 + x)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Add(scope, one, op.input(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Log1p", Log1pGrad);

Status SinhGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_11(mht_11_v, 373, "", "./tensorflow/cc/gradients/math_grad.cc", "SinhGrad");

  // y = sinh(x)
  // dy/dx = cosh(x)
  auto dydx = Cosh(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sinh", SinhGrad);

Status CoshGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_12(mht_12_v, 389, "", "./tensorflow/cc/gradients/math_grad.cc", "CoshGrad");

  // y = cosh(x)
  // dy/dx = sinh(x)
  auto dydx = Sinh(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Cosh", CoshGrad);

Status TanhGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_13(mht_13_v, 405, "", "./tensorflow/cc/gradients/math_grad.cc", "TanhGrad");

  // Use the built-in operator.
  // Note that the built-in operator does not return the conjugate of
  // the gradient.
  auto grad = grad_inputs[0];
  // Optimization to avoid calculating conj(y) until the gradient is
  // evaluated.
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto y = ConjugateHelper(grad_scope, op.output(0));
  grad_outputs->push_back(internal::TanhGrad(grad_scope, y, grad));
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Tanh", TanhGrad);

Status AsinhGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_14(mht_14_v, 424, "", "./tensorflow/cc/gradients/math_grad.cc", "AsinhGrad");

  // y = asinh(x)
  // dy/dx = 1 / cosh(y)
  auto dydx = Reciprocal(scope, Cosh(scope, op.output(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Asinh", AsinhGrad);

Status AcoshGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_15(mht_15_v, 440, "", "./tensorflow/cc/gradients/math_grad.cc", "AcoshGrad");

  // y = acosh(x)
  // dy/dx = 1 / sinh(y)
  auto dydx = Reciprocal(scope, Sinh(scope, op.output(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Acosh", AcoshGrad);

Status AtanhGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_16(mht_16_v, 456, "", "./tensorflow/cc/gradients/math_grad.cc", "AtanhGrad");

  // y = atanh(x)
  // dy/dx = 1 / (1 - x^2)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Sub(scope, one, Square(scope, op.input(0))));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Atanh", AtanhGrad);

Status SigmoidGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_17(mht_17_v, 473, "", "./tensorflow/cc/gradients/math_grad.cc", "SigmoidGrad");

  // Use the built-in operator.
  // Note that the built-in operator does not return the conjugate of
  // the gradient.
  auto grad = grad_inputs[0];
  // Optimization to avoid calculating conj(y) until the gradient is
  // evaluated.
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto y = ConjugateHelper(grad_scope, op.output(0));
  grad_outputs->push_back(internal::SigmoidGrad(grad_scope, y, grad));
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Sigmoid", SigmoidGrad);

Status SignGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_18(mht_18_v, 492, "", "./tensorflow/cc/gradients/math_grad.cc", "SignGrad");

  auto shape = Shape(scope, op.input(0));
  auto zero = Cast(scope, Const(scope, 0.0), op.input(0).type());
  auto dx = Fill(scope, shape, zero);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Sign", SignGrad);

Status SinGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_19(mht_19_v, 506, "", "./tensorflow/cc/gradients/math_grad.cc", "SinGrad");

  // y = sin(x)
  // dy/dx = cos(x)
  auto dydx = Cos(scope, op.input(0));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Sin", SinGrad);

Status CosGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_20(mht_20_v, 522, "", "./tensorflow/cc/gradients/math_grad.cc", "CosGrad");

  // y = cos(x)
  // dy/dx = -sin(x)
  auto dydx = Neg(scope, Sin(scope, op.input(0)));
  // grad(x) = grad(y) * conj(dy/dx)
  grad_outputs->push_back(
      Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Cos", CosGrad);

Status AsinGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_21(mht_21_v, 538, "", "./tensorflow/cc/gradients/math_grad.cc", "AsinGrad");

  // y = asin(x)
  // dy/dx = 1 / sqrt(1 - x^2)
  auto x2 = Square(scope, op.input(0));
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Sqrt(scope, Sub(scope, one, x2)));
  // grad(x) = grad(y) * conj(dy/dx)
  auto dx = Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Asin", AsinGrad);

Status AcosGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_22(mht_22_v, 556, "", "./tensorflow/cc/gradients/math_grad.cc", "AcosGrad");

  // y = acos(x)
  // dy/dx = - 1 / (1 - x * x)^1/2
  // dx = dy * (- 1 / (1 - x * x)^1/2)
  auto x2 = Square(scope, op.input(0));
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Neg(scope, Reciprocal(scope, Sqrt(scope, Sub(scope, one, x2))));
  auto dx = Mul(scope, grad_inputs[0], dydx);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Acos", AcosGrad);

Status TanGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_23(mht_23_v, 574, "", "./tensorflow/cc/gradients/math_grad.cc", "TanGrad");

  // y = tan(x)
  // dy/dx = sec(x)^2 = 1 / cos(x)^2
  auto dydx = Square(scope, Reciprocal(scope, Cos(scope, op.input(0))));
  // grad(x) = grad(y) * conj(dy/dx)
  auto dx = Mul(scope, grad_inputs[0], ConjugateHelper(scope, dydx));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Tan", TanGrad);

Status AtanGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_24(mht_24_v, 590, "", "./tensorflow/cc/gradients/math_grad.cc", "AtanGrad");

  // y = arctan(x)
  // dy/dx = 1 / (1 + x^2)
  // dx = dy * (1 / (1 + x^2)
  auto one = Cast(scope, Const(scope, 1.0), op.input(0).type());
  auto dydx = Reciprocal(scope, Add(scope, one, Square(scope, op.input(0))));
  auto dx = Mul(scope, grad_inputs[0], dydx);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Atan", AtanGrad);

Status Atan2Grad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_25(mht_25_v, 607, "", "./tensorflow/cc/gradients/math_grad.cc", "Atan2Grad");

  auto y = op.input(0);
  auto x = op.input(1);
  Output grad_inv = Div(scope, grad_inputs[0],
                        Add(scope, Square(scope, x), Square(scope, y)));
  grad_outputs->push_back(Mul(scope, x, grad_inv));
  grad_outputs->push_back(Mul(scope, Neg(scope, y), grad_inv));
  return scope.status();
}
REGISTER_GRADIENT_OP("Atan2", Atan2Grad);

// BinaryGradCommon handles the setup for binary ops that broadcast
// their inputs.
Status BinaryGradCommon(const Scope& scope, const Operation& op,
                        std::vector<Output>* grad_outputs, const Output& gx_1,
                        const Output& gx_2) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_26(mht_26_v, 625, "", "./tensorflow/cc/gradients/math_grad.cc", "BinaryGradCommon");

  auto sx_1 = Shape(scope, op.input(0));
  auto sx_2 = Shape(scope, op.input(1));
  auto rx = internal::BroadcastGradientArgs(scope, sx_1, sx_2);
  auto dx_1 = Reshape(scope, Sum(scope, gx_1, rx.r0), sx_1);
  auto dx_2 = Reshape(scope, Sum(scope, gx_2, rx.r1), sx_2);
  grad_outputs->push_back(dx_1);
  grad_outputs->push_back(dx_2);
  return scope.status();
}

Status AddGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_27(mht_27_v, 641, "", "./tensorflow/cc/gradients/math_grad.cc", "AddGrad");

  // y = x_1 + x_2
  // dy/dx_1 = dy/dx_2 = 1
  auto gx_1 = Identity(scope, grad_inputs[0]);
  auto gx_2 = Identity(scope, grad_inputs[0]);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Add", AddGrad);
REGISTER_GRADIENT_OP("AddV2", AddGrad);

Status SubGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_28(mht_28_v, 656, "", "./tensorflow/cc/gradients/math_grad.cc", "SubGrad");

  // y = x_1 - x_2
  // dy/dx_1 = 1
  // dy/dx_2 = -1
  auto gx_1 = Identity(scope, grad_inputs[0]);
  auto gx_2 = Neg(scope, grad_inputs[0]);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Sub", SubGrad);

Status MulGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_29(mht_29_v, 671, "", "./tensorflow/cc/gradients/math_grad.cc", "MulGrad");

  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 * x_2
  // dy/dx_1 = x_2
  // dy/dx_2 = x_1
  auto gx_1 = Mul(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0], x_1);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Mul", MulGrad);

Status DivGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_30(mht_30_v, 688, "", "./tensorflow/cc/gradients/math_grad.cc", "DivGrad");

  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 / x_2
  // dy/dx_1 = 1/x_2
  // dy/dx_2 = -x_1/x_2^2
  auto gx_1 = Div(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0],
                  Div(scope, Div(scope, Neg(scope, x_1), x_2), x_2));
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Div", DivGrad);

Status RealDivGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_31(mht_31_v, 706, "", "./tensorflow/cc/gradients/math_grad.cc", "RealDivGrad");

  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 / x_2
  // dy/dx_1 = 1/x_2
  // dy/dx_2 = -x_1/x_2^2
  auto gx_1 = RealDiv(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0],
                  RealDiv(scope, RealDiv(scope, Neg(scope, x_1), x_2), x_2));
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("RealDiv", RealDivGrad);

Status DivNoNanGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_32(mht_32_v, 724, "", "./tensorflow/cc/gradients/math_grad.cc", "DivNoNanGrad");

  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = x_1 / x_2
  // dy/dx_1 = 1/x_2
  // dy/dx_2 = -x_1/x_2^2
  auto gx_1 = DivNoNan(scope, grad_inputs[0], x_2);
  auto gx_2 = Mul(scope, grad_inputs[0],
                  DivNoNan(scope, DivNoNan(scope, Neg(scope, x_1), x_2), x_2));
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("DivNoNan", DivNoNanGrad);

Status SquaredDifferenceGrad(const Scope& scope, const Operation& op,
                             const std::vector<Output>& grad_inputs,
                             std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_33(mht_33_v, 742, "", "./tensorflow/cc/gradients/math_grad.cc", "SquaredDifferenceGrad");

  auto x_1 = ConjugateHelper(scope, op.input(0));
  auto x_2 = ConjugateHelper(scope, op.input(1));
  // y = (x_1 - x_2)^2
  // dy/dx_1 = 2 * (x_1 - x_2)
  // dy/dx_2 = -2 * (x_1 - x_2)
  auto two = Cast(scope, Const(scope, 2), grad_inputs[0].type());
  auto gx_1 = Mul(scope, grad_inputs[0], Mul(scope, two, Sub(scope, x_1, x_2)));
  auto gx_2 = Neg(scope, gx_1);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("SquaredDifference", SquaredDifferenceGrad);

Status AddNGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_34(mht_34_v, 760, "", "./tensorflow/cc/gradients/math_grad.cc", "AddNGrad");

  // AddN doesn't support broadcasting, so all the inputs must be the
  // same shape.
  // Note:
  // dy/dx_k = d(x_1 + x_2 + ... + x_n)/dx_k = 1 for all x_k
  // hence dx_k = dy for all x_k
  // So the gradient for AddN just transfers the incoming gradient to
  // all outgoing gradients.
  auto incoming = Identity(scope, grad_inputs[0]);
  for (int32_t i = 0; i < op.num_inputs(); ++i) {
    grad_outputs->push_back(incoming);
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("AddN", AddNGrad);

Status PowGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_35(mht_35_v, 781, "", "./tensorflow/cc/gradients/math_grad.cc", "PowGrad");

  auto x = ConjugateHelper(scope, op.input(0));
  auto y = ConjugateHelper(scope, op.input(1));
  auto z = ConjugateHelper(scope, op.output(0));
  auto grad = grad_inputs[0];
  // grad * y * pow(x, y - 1)
  auto one = Cast(scope, Const(scope, 1.0), y.type());
  auto gx_1 = Mul(scope,
                  Mul(scope, grad, y),
                  Pow(scope, x, Sub(scope, y, one)));
  // Avoid false singularity at x = 0
  DataType x_dtype = x.type();
  auto zero = Cast(scope, Const(scope, 0.0), x_dtype);
  if (x_dtype == DT_COMPLEX64 || x_dtype == DT_COMPLEX128) {
    // real(x) < 0 is fine for the complex case
    auto log_x = Where3(scope,
                        NotEqual(scope, x, zero),
                        Log(scope, x),
                        ZerosLike(scope, x));
    auto gy_1 = Mul(scope, Mul(scope, grad, z), log_x);
    return BinaryGradCommon(scope, op, grad_outputs, gx_1, gy_1);
  } else {
    // There's no sensible real value to return if x < 0, so return 0
    auto log_x = Where3(scope,
                        Greater(scope, x, zero),
                        Log(scope, x),
                        ZerosLike(scope, x));
    auto gy_1 = Mul(scope, Mul(scope, grad, z), log_x);
    return BinaryGradCommon(scope, op, grad_outputs, gx_1, gy_1);
  }
}
REGISTER_GRADIENT_OP("Pow", PowGrad);

// MaximumMinimumGradCommon adds shared ops to calculate gradients for
// the binary Maximum and Minimum ops.
Status MaximumMinimumGradCommon(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs,
                                const Output& comparator) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_36(mht_36_v, 822, "", "./tensorflow/cc/gradients/math_grad.cc", "MaximumMinimumGradCommon");

  // comparator is a boolean tensor, with
  // y = x_1 at points where comparator is true, and x_2 otherwise
  // Therefore
  // dy/dx_1 = 1 where comparator is true, and 0 otherwise.
  // dy/dx_2 = 0 where comparator is true, and 1 otherwise.
  auto grad = grad_inputs[0];
  auto zeros = ZerosLike(scope, grad);
  auto gx_1 = Where3(scope, comparator, grad, zeros);
  auto gx_2 = Where3(scope, comparator, zeros, grad);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}

Status MaximumGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_37(mht_37_v, 840, "", "./tensorflow/cc/gradients/math_grad.cc", "MaximumGrad");

  auto comparator = GreaterEqual(scope, op.input(0), op.input(1));
  return MaximumMinimumGradCommon(scope, op, grad_inputs, grad_outputs,
                                  comparator);
}
REGISTER_GRADIENT_OP("Maximum", MaximumGrad);

Status MinimumGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_38(mht_38_v, 852, "", "./tensorflow/cc/gradients/math_grad.cc", "MinimumGrad");

  auto comparator = LessEqual(scope, op.input(0), op.input(1));
  return MaximumMinimumGradCommon(scope, op, grad_inputs, grad_outputs,
                                  comparator);
}
REGISTER_GRADIENT_OP("Minimum", MinimumGrad);

Status RealGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_39(mht_39_v, 864, "", "./tensorflow/cc/gradients/math_grad.cc", "RealGrad");

  auto zero = Cast(scope, Const(scope, 0.0), op.output(0).type());
  auto dx = Complex(scope, grad_inputs[0], zero);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Real", RealGrad);

Status ImagGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_40(mht_40_v, 877, "", "./tensorflow/cc/gradients/math_grad.cc", "ImagGrad");

  auto zero = Cast(scope, Const(scope, 0.0), op.output(0).type());
  auto dx = Complex(scope, zero, grad_inputs[0]);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Imag", ImagGrad);

Status ComplexGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_41(mht_41_v, 890, "", "./tensorflow/cc/gradients/math_grad.cc", "ComplexGrad");

  auto gx_1 = Real(scope, grad_inputs[0]);
  auto gx_2 = Imag(scope, grad_inputs[0]);
  return BinaryGradCommon(scope, op, grad_outputs, gx_1, gx_2);
}
REGISTER_GRADIENT_OP("Complex", ComplexGrad);

Status AngleGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_42(mht_42_v, 902, "", "./tensorflow/cc/gradients/math_grad.cc", "AngleGrad");

  // y = Angle(x)
  // dx = -dy / (Im(x) + iRe(x)) = -dy * z
  auto re = Real(scope, op.input(0));
  auto im = Imag(scope, op.input(0));
  auto z_inv = Reciprocal(scope, Complex(scope, im, re));
  auto zero = Cast(scope, Const(scope, 0), grad_inputs[0].type());
  auto grad = Complex(scope, grad_inputs[0], zero);
  auto dx = Neg(scope, Mul(scope, grad, z_inv));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Angle", AngleGrad);

Status ConjGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_43(mht_43_v, 921, "", "./tensorflow/cc/gradients/math_grad.cc", "ConjGrad");

  grad_outputs->push_back(Conj(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Conj", ConjGrad);

// Integer division x / y, assuming x and y >=0, but treats x/0 = x
Output SafeDivHelper(const Scope& scope, const Output& x, const Output& y) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_44(mht_44_v, 931, "", "./tensorflow/cc/gradients/math_grad.cc", "SafeDivHelper");

  return Div(scope, x, Maximum(scope, y, Const(scope, 1)));
}

// SumGradHelper returns the gradient for the Sum operator, and is used
// by SumGrad and MeanGrad.
Output SumGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_45(mht_45_v, 941, "", "./tensorflow/cc/gradients/math_grad.cc", "SumGradHelper");

  // The partial derivative for any input along a "reduced" dimension
  // is just 1, so we only need replicate the output gradient on such a
  // dimension to its "expanded" shape.
  // Running example:
  // input is
  // [[a, b, c],
  //  [d, e, f]]
  // reduction_indices = [1]
  // Sum = [a + b + c, d + e + f]
  // if the gradient is [g1, g2]
  // We want the propagated gradient to be
  // [[g1, g1, g1],
  //  [g2, g2, g2]]

  // input_shape = [2, 3]
  auto input_shape = Shape(scope, op.input(0));

  // output_shape_kept_dims = [2, 1]
  auto output_shape_kept_dims =
      ReducedShapeHelper(scope, input_shape, op.input(1));

  // This step "flips" any 1s with values from the input_shape, and
  // replaces remaining entries with 1. This creates a shape that
  // shows how much each dimension in the incoming gradient should be
  // replicated.
  // tile_scaling = [1, 3]
  auto tile_scaling = SafeDivHelper(scope, input_shape, output_shape_kept_dims);

  // grad = [[g1], [g2]]
  auto grad = Reshape(scope, grad_inputs[0], output_shape_kept_dims);

  // tile(grad, tile_scaling) = [[g1, g1, g1], [g2, g2, g2]]
  return Tile(scope, grad, tile_scaling);
}

Status SumGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_46(mht_46_v, 982, "", "./tensorflow/cc/gradients/math_grad.cc", "SumGrad");

  grad_outputs->push_back(SumGradHelper(scope, op, grad_inputs));

  // Stop propagation along reduction_indices
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Sum", SumGrad);

Status MeanGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_47(mht_47_v, 996, "", "./tensorflow/cc/gradients/math_grad.cc", "MeanGrad");

  // The Mean gradient is just like the Sum gradient, except that
  // all gradients are also divided by the size of reduced groups.
  auto sum_grad = SumGradHelper(scope, op, grad_inputs);

  // The product of all entries in a tensor's shape is the total
  // number of entries in the tensor. This step calculates
  // n_input_entries/n_output_entries
  // = group_size
  auto input_shape = Shape(scope, op.input(0));
  auto output_shape = Shape(scope, op.output(0));
  auto zero = Const(scope, 0);
  auto group_size = SafeDivHelper(scope, Prod(scope, input_shape, zero),
                                  Prod(scope, output_shape, zero));

  // propagate sum_grad/group_size
  grad_outputs->push_back(
      Div(scope, sum_grad, Cast(scope, group_size, sum_grad.type())));

  // Stop propagation along reduction_indices
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Mean", MeanGrad);

Status ErfGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_48(mht_48_v, 1026, "", "./tensorflow/cc/gradients/math_grad.cc", "ErfGrad");

  auto grad = grad_inputs[0];
  auto two_over_root_pi = Cast(scope, Const(scope, 2 / std::sqrt(M_PI)),
                               grad.type());
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto x = ConjugateHelper(grad_scope, op.input(0));
  // grad * 2/sqrt(pi) * exp(-x**2)
  auto dx = Mul(grad_scope,
                Mul(grad_scope, grad, two_over_root_pi),
                Exp(grad_scope, Neg(grad_scope, Square(grad_scope, x))));
  grad_outputs->push_back(dx);
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Erf", ErfGrad);

Status ErfinvGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_49(mht_49_v, 1046, "", "./tensorflow/cc/gradients/math_grad.cc", "ErfinvGrad");

  auto grad = grad_inputs[0];
  auto root_pi_over_two =
      Cast(scope, Const(scope, std::sqrt(M_PI) / 2), grad.type());
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto x = ConjugateHelper(grad_scope, op.input(0));
  // grad * sqrt(pi) / 2 * exp(erfinv(x) ** 2)
  auto dx = Mul(grad_scope, Mul(grad_scope, grad, root_pi_over_two),
                Exp(grad_scope, Square(grad_scope, op.output(0))));
  grad_outputs->push_back(dx);
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Erfinv", ErfinvGrad);

Status NdtriGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_50(mht_50_v, 1065, "", "./tensorflow/cc/gradients/math_grad.cc", "NdtriGrad");

  auto grad = grad_inputs[0];
  auto root_two_pi =
      Cast(scope, Const(scope, std::sqrt(2 * M_PI)), grad.type());
  auto two = Cast(scope, Const(scope, 2), grad.type());
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto x = ConjugateHelper(grad_scope, op.input(0));
  // grad * sqrt(2 * pi) * exp(ndtri(x) ** 2 / 2)
  auto dx = Mul(
      grad_scope, Mul(grad_scope, grad, root_two_pi),
      Exp(grad_scope, Div(grad_scope, Square(grad_scope, op.output(0)), two)));
  grad_outputs->push_back(dx);
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Ndtri", NdtriGrad);

Status LgammaGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_51(mht_51_v, 1086, "", "./tensorflow/cc/gradients/math_grad.cc", "LgammaGrad");

  auto grad = grad_inputs[0];
  Scope grad_scope = scope.WithControlDependencies(grad);
  auto x = ConjugateHelper(grad_scope, op.input(0));
  auto dx = Mul(grad_scope, grad, Digamma(grad_scope, x));
  grad_outputs->push_back(dx);
  return grad_scope.status();
}
REGISTER_GRADIENT_OP("Lgamma", LgammaGrad);

Status MinOrMaxGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_52(mht_52_v, 1101, "", "./tensorflow/cc/gradients/math_grad.cc", "MinOrMaxGrad");

  // The partial derivative for any input along a "reduced" dimension
  // is 1 when it is the min (or max) and 0 everywhere else. So the
  // gradient calculation is identical for both operators.
  //
  // There's a special case for propagating gradients when there are
  // multiple minima (or maxima) - we choose to divide the gradient
  // equally among all matching inputs.
  //
  // Please note this comment
  // https://github.com/tensorflow/tensorflow/issues/4886#issuecomment-256836063
  // for details.

  // Running example:
  // input: [[5, 5, 5],
  //         [1, 2, -3]]
  // reduction_indices: [1]
  auto input = op.input(0);
  auto reduction_indices = op.input(1);

  // [2, 3]
  auto input_shape = Shape(scope, input);

  // [2, 1]
  auto output_shape_kept_dims =
      ReducedShapeHelper(scope, input_shape, reduction_indices);

  // for op=min (say)
  // output = [5, -3]
  // y = [[5],
  //      [-3]]
  auto y = Reshape(scope, op.output(0), output_shape_kept_dims);

  // reshape([g1, g2], [2, 1]) = [[g1],
  //                              [g2]]
  auto grad = Reshape(scope, grad_inputs[0], output_shape_kept_dims);

  // indicators = equal(y, input)
  //  = equal([[5],   [[5, 5, 5],
  //           [-3]],  [1, 2, -3]])
  //  = [[1, 1, 1],
  //     [0, 0, 1]]
  auto indicators = Cast(scope, Equal(scope, y, input), grad_inputs[0].type());

  // [[3],
  //  [1]]
  auto num_selected = Reshape(scope, Sum(scope, indicators, reduction_indices),
                              output_shape_kept_dims);

  // [[1/3, 1/3, 1/3],
  //  [0, 0, 1]]
  auto scale = Div(scope, indicators, num_selected);

  // [[g1/3, g1/3, g1/3],
  //  [0, 0, g2]]
  grad_outputs->push_back(Mul(scope, scale, grad));

  // Stop propagation along reduction_indices
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Min", MinOrMaxGrad);
REGISTER_GRADIENT_OP("Max", MinOrMaxGrad);

Status ProdGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_53(mht_53_v, 1170, "", "./tensorflow/cc/gradients/math_grad.cc", "ProdGrad");

  auto zero = Const(scope, 0);
  auto one = Const(scope, 1);

  // The gradient can be expressed by dividing the product by each entry of
  // the input tensor. If our input is
  // [
  //  [3, 4],
  //  [5, 6],
  //  [7, 8]
  // ]
  // and we do a Prod operation on the axis 1, we will obtain [[105, 192]].
  // The gradient will have the same shape as the input
  //     [
  //       [105/3, 192/4],
  // dz *  [105/5, 192/6],
  //       [105/7, 192/6]
  //     ]
  // If the input contains a zero, the division is impossible but
  // if we take the calculation that gave the first gradient
  // (3 * 5 * 6)/3 is equal to 5 * 6
  // the trick will be to cumprod the elements on the axis without
  // the element at the current position (3 in the example above).
  // We will take as example:
  // [
  //   [
  //     [3.0, 4.0],
  //     [5.0, 6.0],
  //     [7.0, 8.0]
  //   ],
  //   [
  //     [3.0, 5.0],
  //     [0.0, 6.0],
  //     [5.0, 6.0]
  //   ]
  // ]

  // [2, 3, 2]
  auto input_shape = Shape(scope, op.input(0));

  // The Reshape with -1 flattens the reduction indices.
  // [1]
  auto reduction_indices = Reshape(scope, op.input(1), {-1});

  // [2, 1, 2]
  auto output_shape_kept_dims =
      ReducedShapeHelper(scope, input_shape, reduction_indices);

  // [1, 3, 1]
  auto tile_scaling = SafeDivHelper(scope, input_shape, output_shape_kept_dims);

  // [[[105, 192]], [[0, 180]]]
  auto grad = Reshape(scope, grad_inputs[0], output_shape_kept_dims);

  // [[[105, 192], [105, 192], [105, 192]], [[0, 180], [0, 180], [0, 180]]]
  auto grad_tiled = Tile(scope, grad, tile_scaling);

  Scope cpu_scope = scope.WithDevice("/cpu:0");

  // [3]
  auto rank = Rank(cpu_scope, op.input(0));


  // Normalize any negative indices in the reduction_axes to positive values.
  auto reduction_indices_pos = Mod(cpu_scope, Add(cpu_scope, reduction_indices, rank), rank);

  // [1]
  auto reduced = Cast(cpu_scope, reduction_indices_pos, DataType::DT_INT32);

  // [0, 1, 2]
  auto idx = Range(cpu_scope, zero, rank, one);

  // [0, 2]
  auto other = SetDiff1D(cpu_scope, idx, reduced).out;

  // [1, 0, 2]
  auto perm =
      Concat(cpu_scope, std::initializer_list<Input>{reduced, other}, 0);

  // 3 => [3]
  auto reduced_num = Prod(cpu_scope, Gather(scope, input_shape, reduced), 0);

  // 2 * 2 => [2]
  auto other_num = Prod(cpu_scope, Gather(scope, input_shape, other), 0);

  // [
  //    [
  //       [ 3.,  4.],
  //       [ 3.,  5.]
  //   ],
  //   [
  //       [ 5.,  6.],
  //       [ 0.,  6.]
  //   ],
  //   [
  //       [ 7.,  8.],
  //       [ 5.,  6.]
  //   ]
  // ]
  auto permuted = Transpose(scope, op.input(0), perm);

  // [3, 2, 2]
  auto permuted_shape = Shape(scope, permuted);

  // [
  //   [ 3.,  4.,  3.,  5.],
  //   [ 5.,  6.,  0.,  6.],
  //   [ 7.,  8.,  5.,  6.]
  // ]
  auto reshaped = Reshape(
      scope, permuted,
      Stack(scope, std::initializer_list<Input>{reduced_num, other_num}));

  // [
  //   [ 1.,  1.,  1.,  1.],
  //   [ 3.,  4.,  3.,  5.],
  //   [ 15.,  24.,  0.,  30.]
  // ]
  auto left = Cumprod(scope, reshaped, zero, Cumprod::Exclusive(true));

  // [
  //   [ 35.,  48.,  0.,  36.],
  //   [  7.,   8.,   5.,   6.],
  //   [  1.,   1.,   1.,   1.]
  // ]
  auto right =
      Cumprod(scope, reshaped, zero, Cumprod::Exclusive(true).Reverse(true));

  // left * right =
  // [
  //   [ 35.,  48.,  0.,  36.],
  //   [ 21.,  32.,  15.,  30.],
  //   [ 15.,  24.,  0.,  30.]
  // ]
  // y =
  // [
  //   [
  //     [ 35.,  48.],
  //     [ 0.,  36.]
  //   ],
  //   [
  //     [ 21.,  32.],
  //     [ 15.,  30.]
  //   ],
  //   [
  //     [ 15.,  24.],
  //     [ 0.,  30.]
  //   ]
  // ]
  auto y = Reshape(scope, Mul(scope, left, right), permuted_shape);

  // out =
  // [
  //   [
  //     [ 35.,  48.],
  //     [ 21.,  32.],
  //     [ 15.,  24.]
  //   ],
  //   [
  //     [ 0.,   36.],
  //     [ 15.,  30.],
  //     [ 0.,  30.]
  //   ]
  // ]
  auto out =
      Mul(scope, grad_tiled, Transpose(scope, y, InvertPermutation(scope, perm)));

  grad_outputs->push_back(Reshape(scope, out, input_shape));

  // stop propagation along reduction_indices
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Prod", ProdGrad);

Status SegmentSumGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_54(mht_54_v, 1350, "", "./tensorflow/cc/gradients/math_grad.cc", "SegmentSumGrad");

  // The SegmentSum operation sums segments of the Tensor that have the same
  // index in the segment_ids parameter.
  // i.e z = [2, 3, 4, 5], segment_ids [0, 0, 0, 1]
  // will produce [2 + 3 + 4, 5] = [9, 5]
  // The gradient that will flow back to the gather operation will look like
  // [x1, x2], it will have the same shape as the output of the SegmentSum
  // operation. The differentiation step of the SegmentSum operation just
  // broadcast the gradient in order to retrieve the z's shape.
  // dy/dz = [x1, x1, x1, x2]
  grad_outputs->push_back(Gather(scope, grad_inputs[0], op.input(1)));

  // stop propagation along segment_ids
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SegmentSum", SegmentSumGrad);

// MatMulGrad helper function used to compute two MatMul operations
// based on input matrix transposition combinations.
Status MatMulGradHelper(const Scope& scope, const bool is_batch,
                        const Output& x0, const bool adj_x0, const Output& x1,
                        const bool adj_x1, const Output& y0, const bool adj_y0,
                        const Output& y1, const bool adj_y1,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_55(mht_55_v, 1377, "", "./tensorflow/cc/gradients/math_grad.cc", "MatMulGradHelper");

  if (is_batch == false) {
    auto dx =
        MatMul(scope, x0, x1, MatMul::TransposeA(adj_x0).TransposeB(adj_x1));
    grad_outputs->push_back(dx);
    auto dy =
        MatMul(scope, y0, y1, MatMul::TransposeA(adj_y0).TransposeB(adj_y1));
    grad_outputs->push_back(dy);
  } else {
    auto dx =
        BatchMatMul(scope, x0, x1, BatchMatMul::AdjX(adj_x0).AdjY(adj_x1));
    grad_outputs->push_back(dx);
    auto dy =
        BatchMatMul(scope, y0, y1, BatchMatMul::AdjX(adj_y0).AdjY(adj_y1));
    grad_outputs->push_back(dy);
  }
  return scope.status();
}

// MatMulGrad common used to read and check node attr state, and determine
// proper MatMul products for gradients based on input matrix transposition
// combinations.
Status MatMulGradCommon(const Scope& scope, const Operation& op,
                        const bool is_batch,
                        const std::vector<Output>& grad_inputs,
                        const string& attr_adj_x, const string& attr_adj_y,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("attr_adj_x: \"" + attr_adj_x + "\"");
   mht_56_v.push_back("attr_adj_y: \"" + attr_adj_y + "\"");
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_56(mht_56_v, 1408, "", "./tensorflow/cc/gradients/math_grad.cc", "MatMulGradCommon");

  auto a = op.input(0);
  auto b = op.input(1);
  // Use conjugate of the inputs for MatMul
  if (is_batch == false) {
    a = ConjugateHelper(scope, a);
    b = ConjugateHelper(scope, b);
  }
  auto product = op.output(0);

  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(GetNodeAttr(product.node()->attrs(), attr_adj_x, &ta));
  TF_RETURN_IF_ERROR(GetNodeAttr(product.node()->attrs(), attr_adj_y, &tb));

  if (!ta && !tb) {
    return MatMulGradHelper(scope, is_batch, grad_inputs[0], false, b, true, a,
                            true, grad_inputs[0], false, grad_outputs);
  } else if (!ta && tb) {
    return MatMulGradHelper(scope, is_batch, grad_inputs[0], false, b, false,
                            grad_inputs[0], true, a, false, grad_outputs);
  } else if (ta && !tb) {
    return MatMulGradHelper(scope, is_batch, b, false, grad_inputs[0], true, a,
                            false, grad_inputs[0], false, grad_outputs);
  }
  return MatMulGradHelper(scope, is_batch, b, true, grad_inputs[0], true,
                          grad_inputs[0], true, a, true, grad_outputs);
}

Status MatMulGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_57(mht_57_v, 1442, "", "./tensorflow/cc/gradients/math_grad.cc", "MatMulGrad");

  return MatMulGradCommon(scope, op, false, grad_inputs, "transpose_a",
                          "transpose_b", grad_outputs);
}
REGISTER_GRADIENT_OP("MatMul", MatMulGrad);

Status BatchMatMulGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_58(mht_58_v, 1453, "", "./tensorflow/cc/gradients/math_grad.cc", "BatchMatMulGrad");

  return MatMulGradCommon(scope, op, true, grad_inputs, "adj_x", "adj_y",
                          grad_outputs);
}
REGISTER_GRADIENT_OP("BatchMatMul", BatchMatMulGrad);

Status CumsumGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_59(mht_59_v, 1464, "", "./tensorflow/cc/gradients/math_grad.cc", "CumsumGrad");

  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("Cumsum requires 2 arguments");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Cumsum grad requires 1 grad input");
  }

  Cumsum::Attrs attrs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "exclusive", &attrs.exclusive_));
  bool reverse;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "reverse", &reverse));
  attrs.reverse_ = !reverse;

  auto axis = op.input(1);
  auto sum = Cumsum(scope, grad_inputs[0], axis, attrs);
  grad_outputs->push_back(sum.out);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Cumsum", CumsumGrad);

bool IsFloatingPointDtype(DataType dtype) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_60(mht_60_v, 1490, "", "./tensorflow/cc/gradients/math_grad.cc", "IsFloatingPointDtype");

  static constexpr DataType valid_dtypes[] = {
      DT_FLOAT, DT_HALF, DT_DOUBLE, DT_BFLOAT16, DT_COMPLEX64, DT_COMPLEX128};
  return std::find(std::begin(valid_dtypes), std::end(valid_dtypes), dtype) !=
         std::end(valid_dtypes);
}

Status CastGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_61(mht_61_v, 1502, "", "./tensorflow/cc/gradients/math_grad.cc", "CastGrad");

  if (op.num_inputs() != 1) {
    return errors::InvalidArgument("Cast requires 2 arguments");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Cast grad requires 1 grad input");
  }

  auto src_type = op.input_type(0);
  auto dst_type = grad_inputs[0].type();
  if (IsFloatingPointDtype(src_type) && IsFloatingPointDtype(dst_type)) {
    grad_outputs->push_back(Cast(scope, grad_inputs[0], src_type));
  } else {
    grad_outputs->push_back(NoGradient());
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("Cast", CastGrad);

Status SelectGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_62(mht_62_v, 1526, "", "./tensorflow/cc/gradients/math_grad.cc", "SelectGrad");

  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Select requires 3 arguments");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Select grad requires 1 grad input");
  }

  auto c = op.input(0);
  auto zeros = ZerosLike(scope, grad_inputs[0]);
  grad_outputs->push_back(NoGradient());  // Condition
  grad_outputs->push_back(Where3(scope, c, grad_inputs[0], zeros));
  grad_outputs->push_back(Where3(scope, c, zeros, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Select", SelectGrad);

Status SelectV2Grad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSmath_gradDTcc mht_63(mht_63_v, 1548, "", "./tensorflow/cc/gradients/math_grad.cc", "SelectV2Grad");

  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Select requires 3 arguments");
  }

  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Select grad requires 1 grad input");
  }

  auto c = op.input(0);
  auto x = op.input(1);
  auto y = op.input(2);

  auto zeros = ZerosLike(scope, grad_inputs[0]);
  auto gx = SelectV2(scope, c, grad_inputs[0], zeros);
  auto x_shape = Shape(scope, x);
  auto output_shape = Shape(scope, op.output(0));

  // Reduce away broadcasted leading dims.
  auto reduce_x = internal::BroadcastGradientArgs(scope, x_shape, output_shape);
  auto gx_sum =
      ReduceSum(scope, gx, /*axis=*/reduce_x.r0, ReduceSum::KeepDims(true));
  auto gx_sum_reshape = Reshape(scope, gx_sum, x_shape);

  auto gy = SelectV2(scope, c, zeros, grad_inputs[0]);
  auto y_shape = Shape(scope, y);

  // Reduce away broadcasted leading dims.
  auto reduce_y = internal::BroadcastGradientArgs(scope, y_shape, output_shape);
  auto gy_sum =
      ReduceSum(scope, gy, /*axis=*/reduce_y.r0, ReduceSum::KeepDims(true));
  auto gy_sum_reshape = Reshape(scope, gy_sum, y_shape);

  grad_outputs->push_back(NoGradient());  // Condition
  grad_outputs->push_back(gx_sum_reshape);
  grad_outputs->push_back(gy_sum_reshape);
  return scope.status();
}

REGISTER_GRADIENT_OP("SelectV2", SelectV2Grad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
