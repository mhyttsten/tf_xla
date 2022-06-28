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
class MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc() {
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
#include "tensorflow/c/eager/gradient_checker.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

void CompareNumericalAndManualGradients(
    Model model, AbstractContext* ctx,
    absl::Span<AbstractTensorHandle* const> inputs, int input_index,
    float* expected_grad, int num_grad, bool use_function,
    double abs_error = 1e-2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/c/eager/gradient_checker_test.cc", "CompareNumericalAndManualGradients");

  Status s;
  AbstractTensorHandlePtr numerical_grad;
  {
    AbstractTensorHandle* numerical_grad_raw;
    s = CalcNumericalGrad(ctx, model, inputs, input_index, use_function,
                          &numerical_grad_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    numerical_grad.reset(numerical_grad_raw);
  }

  TF_Tensor* numerical_tensor;
  s = GetValue(numerical_grad.get(), &numerical_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto num_elem_numerical = TF_TensorElementCount(numerical_tensor);
  ASSERT_EQ(num_elem_numerical, num_grad);

  float* dnumerical = new float[num_elem_numerical]{0};
  memcpy(&dnumerical[0], TF_TensorData(numerical_tensor),
         TF_TensorByteSize(numerical_tensor));

  for (int j = 0; j < num_grad; j++) {
    ASSERT_NEAR(dnumerical[j], expected_grad[j], abs_error);
  }
  delete[] dnumerical;
  TF_DeleteTensor(numerical_tensor);
}

Status MatMulModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/c/eager/gradient_checker_test.cc", "MatMulModel");

  return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0],
                     /*transpose_a=*/false,
                     /*transpose_b=*/false, "MatMul");
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/c/eager/gradient_checker_test.cc", "MulModel");

  return ops::Mul(ctx, inputs[0], inputs[1], &outputs[0], "Mul");
}

// TODO(vnvo2409): Add more tests from `python/ops/gradient_checker_v2_test.py`.
// These tests should not be confused with `[*]_grad_test` which compare the
// result of `gradient_checker` and `[*]_grad`. The tests here test the
// functionality of `gradient_checker` by comparing the result with expected
// manual user-provided gradients.
class GradientCheckerTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_3(mht_3_v, 265, "", "./tensorflow/c/eager/gradient_checker_test.cc", "SetUp");

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());

    {
      Status s = StatusFromTF_Status(status.get());
      CHECK_EQ(errors::OK, s.code()) << s.error_message();
    }

    {
      AbstractContext* ctx_raw = nullptr;
      Status s =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr ctx_;

 public:
  bool UseMlir() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_4(mht_4_v, 294, "", "./tensorflow/c/eager/gradient_checker_test.cc", "UseMlir");
 return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradient_checker_testDTcc mht_5(mht_5_v, 298, "", "./tensorflow/c/eager/gradient_checker_test.cc", "UseFunction");
 return std::get<2>(GetParam()); }
};

TEST_P(GradientCheckerTest, TestMatMul) {
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx_.get(), A_vals,
                                                         A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    A.reset(A_raw);
  }
  float B_vals[] = {.5f, -1.0f, 1.0f, 1.0f};
  int64_t B_dims[] = {2, 2};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx_.get(), B_vals,
                                                         B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    B.reset(B_raw);
  }

  float expected_dA[4] = {-.5f, 2.0f, -.5f, 2.0f};
  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndManualGradients(
      MatMulModel, ctx_.get(), {A.get(), B.get()}, 0, expected_dA, 4,
      UseFunction()));
}

TEST_P(GradientCheckerTest, TestMul) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx_.get(), 7.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  float expected_dx[1] = {7.0f};
  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndManualGradients(
      MulModel, ctx_.get(), {x.get(), y.get()}, 0, expected_dx, 1,
      UseFunction()));
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
