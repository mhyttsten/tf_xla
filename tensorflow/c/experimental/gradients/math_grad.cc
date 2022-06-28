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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/gradients/math_grad.h"

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"

using std::vector;
using tensorflow::ops::AddV2;
using tensorflow::ops::Div;
using tensorflow::ops::DivNoNan;
using tensorflow::ops::MatMul;
using tensorflow::ops::Mul;
using tensorflow::ops::Neg;
using tensorflow::ops::OnesLike;
using tensorflow::ops::SqrtGrad;

namespace tensorflow {
namespace gradients {
namespace {

static Status SafeConj(AbstractContext* ctx, AbstractTensorHandle* const input,
                       AbstractTensorHandle** output, const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_0(mht_0_v, 208, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "SafeConj");

  auto dtype = input->DataType();
  if (DataTypeIsFloating(BaseType(dtype)) ||
      DataTypeIsInteger(BaseType(dtype))) {
    return tensorflow::ops::Identity(ctx, input, output, name);
  } else if (!DataTypeIsComplex(BaseType(dtype)) &&
             BaseType(dtype) != DT_VARIANT) {
    return errors::InvalidArgument(
        "Expected numeric or variant tensor, got dtype ", dtype);
  }
  return tensorflow::ops::Conj(ctx, input, output, name);
}

class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_1(mht_1_v, 228, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    // TODO(b/161805092): Support broadcasting.

    DCHECK(grad_outputs[0]);
    grad_inputs[0] = grad_outputs[0];
    grad_inputs[1] = grad_outputs[0];

    grad_inputs[0]->Ref();
    grad_inputs[1]->Ref();
    return Status::OK();
  }
  ~AddGradientFunction() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_2(mht_2_v, 242, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~AddGradientFunction");
}
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_3(mht_3_v, 250, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "ExpGradientFunction");

    exp->Ref();
  }
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_4(mht_4_v, 258, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    AbstractTensorHandle* conj_output;
    std::string name = "Conj_Exp_Grad";
    TF_RETURN_IF_ERROR(SafeConj(ctx, exp_.get(), &conj_output, name.c_str()));
    AbstractTensorHandlePtr conj_output_releaser(conj_output);

    name = "Mul_Exp_Grad";
    TF_RETURN_IF_ERROR(
        Mul(ctx, conj_output, grad_outputs[0], &grad_inputs[0], name.c_str()));
    return Status::OK();
  }
  ~ExpGradientFunction() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_5(mht_5_v, 272, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~ExpGradientFunction");
}

 private:
  AbstractTensorHandlePtr exp_;
};

class SqrtGradientFunction : public GradientFunction {
 public:
  explicit SqrtGradientFunction(AbstractTensorHandle* sqrt) : sqrt_(sqrt) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_6(mht_6_v, 283, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "SqrtGradientFunction");

    sqrt->Ref();
  }
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_7(mht_7_v, 291, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    std::string name = "Sqrt_Grad";
    TF_RETURN_IF_ERROR(SqrtGrad(ctx, sqrt_.get(), grad_outputs[0],
                                &grad_inputs[0], name.c_str()));
    return Status::OK();
  }
  ~SqrtGradientFunction() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_8(mht_8_v, 300, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~SqrtGradientFunction");
}

 private:
  AbstractTensorHandlePtr sqrt_;
};

class MatMulGradientFunction : public GradientFunction {
 public:
  explicit MatMulGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                  AttrBuilder f_attrs)
      : forward_inputs_(f_inputs), forward_attrs_(f_attrs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_9(mht_9_v, 313, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "MatMulGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_10(mht_10_v, 326, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    /* Given upstream grad U and a matmul op A*B, the gradients are:
     *
     *    dA = U * B.T
     *    dB = A.T * U
     *
     *    where A.T means `transpose(A)`
     */
    AbstractTensorHandle* upstream_grad = grad_outputs[0];

    // Get transpose attrs
    bool t_a;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("transpose_a", &t_a));

    bool t_b;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("transpose_b", &t_b));

    // Conj each input
    AbstractTensorHandle* conj_output;
    std::string name = "Conj_A_MatMul_Grad";
    TF_RETURN_IF_ERROR(
        SafeConj(ctx, forward_inputs_[0], &conj_output, name.c_str()));

    AbstractTensorHandlePtr A(conj_output);

    name = "Conj_B_MatMul_Grad";
    TF_RETURN_IF_ERROR(
        SafeConj(ctx, forward_inputs_[1], &conj_output, name.c_str()));

    AbstractTensorHandlePtr B(conj_output);

    // Calc Grad
    AbstractTensorHandle* matmul_A_output;
    AbstractTensorHandle* matmul_B_output;
    std::string name_grad_A = "MatMul_Grad_A";
    std::string name_grad_B = "MatMul_Grad_B";
    if (!t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, B.get(), &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, A.get(), upstream_grad, &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ false, name_grad_B.c_str()));
    } else if (!t_a && t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, B.get(), &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ false, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, A.get(), &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ false, name_grad_B.c_str()));

    } else if (t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, B.get(), upstream_grad, &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, A.get(), upstream_grad, &matmul_B_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ false, name_grad_B.c_str()));
    } else {  // t_a && t_b
      TF_RETURN_IF_ERROR(MatMul(ctx, B.get(), upstream_grad, &matmul_A_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, A.get(), &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ true, name_grad_B.c_str()));
    }

    // Gradient for A
    grad_inputs[0] = matmul_A_output;

    // Gradient for B
    grad_inputs[1] = matmul_B_output;
    return Status::OK();
  }
  ~MatMulGradientFunction() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_11(mht_11_v, 407, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~MatMulGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
  AttrBuilder forward_attrs_;
};

class NegGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_12(mht_12_v, 428, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    /* Given upstream grad U and a Neg op Y = -X, the gradients are:
     *
     *    dX =  -U
     *
     */

    std::string name = "Neg_Grad";
    TF_RETURN_IF_ERROR(
        ops::Neg(ctx, grad_outputs[0], &grad_inputs[0], name.c_str()));
    return Status::OK();
  }
  ~NegGradientFunction() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_13(mht_13_v, 443, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~NegGradientFunction");
}
};

class SubGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_14(mht_14_v, 453, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    /* Given upstream grad U and a Sub op A-B, the gradients are:
     *
     *    dA =  U
     *    dB = -U
     *
     */

    // Grad for A
    DCHECK(grad_outputs[0]);
    grad_inputs[0] = grad_outputs[0];
    grad_inputs[0]->Ref();

    // Grad for B
    // negate the upstream grad
    std::string name = "Neg_Sub_Grad_B";
    TF_RETURN_IF_ERROR(
        ops::Neg(ctx, grad_outputs[0], &grad_inputs[1], name.c_str()));

    return Status::OK();
  }
  ~SubGradientFunction() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_15(mht_15_v, 477, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~SubGradientFunction");
}
};

class MulGradientFunction : public GradientFunction {
 public:
  explicit MulGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs_(f_inputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_16(mht_16_v, 486, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "MulGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_17(mht_17_v, 499, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    /* Given upstream grad U and a mul op A*B, the gradients are:
     *
     *    dA = U * B
     *    dB = A * U
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];

    // Gradient for A
    std::string name = "Mul_Grad_A";
    TF_RETURN_IF_ERROR(Mul(ctx, upstream_grad, forward_inputs_[1],
                           &grad_inputs[0], name.c_str()));

    // Gradient for B
    name = "Mul_Grad_B";
    TF_RETURN_IF_ERROR(Mul(ctx, forward_inputs_[0], upstream_grad,
                           &grad_inputs[1], name.c_str()));
    return Status::OK();
  }
  ~MulGradientFunction() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_18(mht_18_v, 523, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~MulGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
};

class Log1pGradientFunction : public GradientFunction {
 public:
  explicit Log1pGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs_(f_inputs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_19(mht_19_v, 542, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Log1pGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_20(mht_20_v, 555, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    // TODO(vnvo2409): Add control dependency
    /* Given upstream grad U and a Log1p op: Y = log(1 + X), the gradients are:
     *
     *    dX = U / (1 + X)
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* X = forward_inputs_[0];

    AbstractTensorHandle* temp_output;

    // Calculate conjugate of X
    std::string name = "Conj_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(SafeConj(ctx, X, &temp_output, name.c_str()));

    AbstractTensorHandlePtr Conj_X(temp_output);

    // Creates Ones
    name = "OnesLike_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(OnesLike(ctx, Conj_X.get(), &temp_output, name.c_str()));

    AbstractTensorHandlePtr Ones_X(temp_output);

    name = "Add_Log1p_Grad_X";
    // Calculate 1 + Conj(X)
    TF_RETURN_IF_ERROR(
        AddV2(ctx, Ones_X.get(), Conj_X.get(), &temp_output, name.c_str()));

    AbstractTensorHandlePtr Conj_XP1(temp_output);

    name = "Div_Log1p_Grad_X";
    // Calculate U / (1 + Conj(X))
    TF_RETURN_IF_ERROR(
        Div(ctx, upstream_grad, Conj_XP1.get(), &grad_inputs[0], name.c_str()));

    return Status::OK();
  }
  ~Log1pGradientFunction() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_21(mht_21_v, 597, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~Log1pGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
};

class DivNoNanGradientFunction : public GradientFunction {
 public:
  explicit DivNoNanGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                    vector<AbstractTensorHandle*> f_outputs)
      : forward_inputs_(f_inputs), forward_outputs_(f_outputs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_22(mht_22_v, 617, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "DivNoNanGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
    for (auto output : forward_outputs_) {
      if (output) {
        output->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_23(mht_23_v, 635, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Compute");

    // TODO(vnvo2409): Add shape broadcasting
    /* Given upstream grad U and a Div op: Z = X/Y, the gradients are:
     *
     *    dX = U / Y
     *    dY = -U*X / Y^2 = (X/Y) * -U / Y = -U*Z / Y
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* Y = forward_inputs_[1];
    AbstractTensorHandle* Z = forward_outputs_[0];

    // Calculate dX =  U / Y
    std::string name = "Div_Grad_X";
    TF_RETURN_IF_ERROR(
        DivNoNan(ctx, upstream_grad, Y, &grad_inputs[0], name.c_str()));

    AbstractTensorHandle* temp_output;
    // Calculate dY = -U*Z / Y
    name = "Neg_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Neg(ctx, upstream_grad, &temp_output,
                           name.c_str()));  // -U
    AbstractTensorHandlePtr MinusU(temp_output);

    name = "Mul_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Mul(ctx, MinusU.get(), Z, &temp_output,
                           name.c_str()));  // -U*Z
    AbstractTensorHandlePtr UZ(temp_output);

    name = "Div_Grad_Y";
    TF_RETURN_IF_ERROR(DivNoNan(ctx, UZ.get(), Y, &grad_inputs[1],
                                name.c_str()));  // -U*Z / Y

    return Status::OK();
  }
  ~DivNoNanGradientFunction() override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_24(mht_24_v, 674, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "~DivNoNanGradientFunction");

    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
    for (auto output : forward_outputs_) {
      if (output) {
        output->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs and outputs.
  vector<AbstractTensorHandle*> forward_inputs_;
  vector<AbstractTensorHandle*> forward_outputs_;
};

}  // namespace

GradientFunction* AddRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_25(mht_25_v, 698, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "AddRegisterer");

  return new AddGradientFunction;
}

GradientFunction* ExpRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_26(mht_26_v, 705, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "ExpRegisterer");

  return new ExpGradientFunction(op.outputs[0]);
}

GradientFunction* MatMulRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_27(mht_27_v, 712, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "MatMulRegisterer");

  return new MatMulGradientFunction(op.inputs, op.attrs);
}

GradientFunction* SqrtRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_28(mht_28_v, 719, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "SqrtRegisterer");

  return new SqrtGradientFunction(op.outputs[0]);
}

GradientFunction* NegRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_29(mht_29_v, 726, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "NegRegisterer");

  return new NegGradientFunction;
}

GradientFunction* SubRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_30(mht_30_v, 733, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "SubRegisterer");

  return new SubGradientFunction;
}

GradientFunction* MulRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_31(mht_31_v, 740, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "MulRegisterer");

  return new MulGradientFunction(op.inputs);
}

GradientFunction* Log1pRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_32(mht_32_v, 747, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "Log1pRegisterer");

  return new Log1pGradientFunction(op.inputs);
}

GradientFunction* DivNoNanRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_gradDTcc mht_33(mht_33_v, 754, "", "./tensorflow/c/experimental/gradients/math_grad.cc", "DivNoNanRegisterer");

  return new DivNoNanGradientFunction(op.inputs, op.outputs);
}

}  // namespace gradients
}  // namespace tensorflow
