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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc() {
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

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status AddModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "AddModel");

  return ops::AddV2(ctx, inputs[0], inputs[1], &outputs[0], "Add");
}

Status ExpModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "ExpModel");

  return ops::Exp(ctx, inputs[0], &outputs[0], "Exp");
}

Status SqrtModel(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> inputs,
                 absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "SqrtModel");

  return ops::Sqrt(ctx, inputs[0], &outputs[0], "Sqrt");
}

Status NegModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_3(mht_3_v, 232, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "NegModel");

  return ops::Neg(ctx, inputs[0], &outputs[0], "Neg");
}

Status SubModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_4(mht_4_v, 241, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "SubModel");

  return ops::Sub(ctx, inputs[0], inputs[1], &outputs[0], "Sub");
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_5(mht_5_v, 250, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "MulModel");

  return ops::Mul(ctx, inputs[0], inputs[1], &outputs[0], "Mul");
}

Status Log1pModel(AbstractContext* ctx,
                  absl::Span<AbstractTensorHandle* const> inputs,
                  absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_6(mht_6_v, 259, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "Log1pModel");

  return ops::Log1p(ctx, inputs[0], &outputs[0], "Log1p");
}

Status DivNoNanModel(AbstractContext* ctx,
                     absl::Span<AbstractTensorHandle* const> inputs,
                     absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_7(mht_7_v, 268, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "DivNoNanModel");

  return ops::DivNoNan(ctx, inputs[0], inputs[1], &outputs[0], "DivNoNan");
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_8(mht_8_v, 278, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "SetUp");

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    status_ = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      status_ =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
      immediate_execution_ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr immediate_execution_ctx_;
  GradientRegistry registry_;
  Status status_;

 public:
  bool UseMlir() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_9(mht_9_v, 306, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "UseMlir");
 return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSmath_grad_testDTcc mht_10(mht_10_v, 310, "", "./tensorflow/c/experimental/gradients/math_grad_test.cc", "UseFunction");
 return std::get<2>(GetParam()); }
};

TEST_P(CppGradients, TestAddGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  // TODO(srbs): Rename ops::Add to ops::AddV2 and AddRegister to
  // AddV2Registerer.
  status_ = registry_.Register("AddV2", AddRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      AddModel, BuildGradModel(AddModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestExpGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Exp", ExpRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      ExpModel, BuildGradModel(ExpModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestMatMulGrad) {
  // TODO(vnvo2409): Figure out why `gradient_checker` does not work very
  // well with `MatMul` and remove `TestMatMul*` in
  // `mnist_gradients_test` when done.
  GTEST_SKIP();

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  int64_t A_dims[] = {3, 3};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    A.reset(A_raw);
  }

  float B_vals[] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t B_dims[] = {3, 3};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), B_vals, B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    B.reset(B_raw);
  }

  status_ = registry_.Register("MatMul", MatMulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  for (bool transpose_a : {false, true}) {
    for (bool transpose_b : {false, true}) {
      Model MatMulModel =
          [transpose_a, transpose_b](
              AbstractContext* ctx,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs) -> Status {
        return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0], transpose_a,
                           transpose_b, "MatMul");
      };
      ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
          MatMulModel, BuildGradModel(MatMulModel, registry_),
          immediate_execution_ctx_.get(), {A.get(), B.get()}, UseFunction()));
    }
  }
}

TEST_P(CppGradients, TestMatMulGradManual) {
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  int64_t A_dims[] = {3, 3};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    A.reset(A_raw);
  }

  float B_vals[] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t B_dims[] = {3, 3};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), B_vals, B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    B.reset(B_raw);
  }

  status_ = registry_.Register("MatMul", MatMulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  bool transpose_a_vals[] = {false, false, true, true};
  bool transpose_b_vals[] = {false, true, false, true};
  float dA_vals[4][9] = {{24, 15, 6, 24, 15, 6, 24, 15, 6},
                         {18, 15, 12, 18, 15, 12, 18, 15, 12},
                         {24, 24, 24, 15, 15, 15, 6, 6, 6},
                         {18, 18, 18, 15, 15, 15, 12, 12, 12}};
  float dB_vals[4][9] = {{12, 12, 12, 15, 15, 15, 18, 18, 18},
                         {12, 15, 18, 12, 15, 18, 12, 15, 18},
                         {6, 6, 6, 15, 15, 15, 24, 24, 24},
                         {6, 15, 24, 6, 15, 24, 6, 15, 24}};

  for (int i{}; i < 4; ++i) {
    bool transpose_a = transpose_a_vals[i];
    bool transpose_b = transpose_b_vals[i];
    Model MatMulModel =
        [transpose_a, transpose_b](
            AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs) -> Status {
      return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0], transpose_a,
                         transpose_b, "MatMul");
    };
    Model MatMulGradModel = BuildGradModel(MatMulModel, registry_);
    std::vector<AbstractTensorHandle*> outputs(2);
    status_ =
        RunModel(MatMulGradModel, immediate_execution_ctx_.get(),
                 {A.get(), B.get()}, absl::MakeSpan(outputs), UseFunction());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], dA_vals[i],
                                             /*dims*/ {3, 3},
                                             /*abs_error*/ 0));
    ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], dB_vals[i],
                                             /*dims*/ {3, 3},
                                             /*abs_error*/ 0));
    outputs[0]->Unref();
    outputs[1]->Unref();
  }
}

TEST_P(CppGradients, TestSqrtGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Sqrt", SqrtRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SqrtModel, BuildGradModel(SqrtModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestNegGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Neg", NegRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      NegModel, BuildGradModel(NegModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestSubGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  status_ = registry_.Register("Sub", SubRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SubModel, BuildGradModel(SubModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestMulGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  status_ = registry_.Register("Mul", MulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      MulModel, BuildGradModel(MulModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestLog1pGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Log1p", Log1pRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      Log1pModel, BuildGradModel(Log1pModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestDivNoNanGrad) {
  status_ = registry_.Register("DivNoNan", DivNoNanRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  auto DivNoNanGradModel = BuildGradModel(DivNoNanModel, registry_);

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      DivNoNanModel, DivNoNanGradModel, immediate_execution_ctx_.get(),
      {x.get(), y.get()}, UseFunction()));

  // `DivNoNanGradModel` should return {`0`, `0`} when the denominator is `0`.
  AbstractTensorHandlePtr z;
  {
    AbstractTensorHandle* z_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 0.0f, &z_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    z.reset(z_raw);
  }
  std::vector<AbstractTensorHandle*> outputs(2);
  status_ =
      RunModel(DivNoNanGradModel, immediate_execution_ctx_.get(),
               {x.get(), z.get()}, absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], {0.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], {0.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
  outputs[0]->Unref();
  outputs[1]->Unref();
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
