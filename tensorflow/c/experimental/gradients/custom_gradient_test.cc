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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc() {
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
#include <memory>

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {
using std::vector;

class CustomGradientTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/c/experimental/gradients/custom_gradient_test.cc", "SetUp");

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();
  }
};

class PassThroughGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/c/experimental/gradients/custom_gradient_test.cc", "Compute");

    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(grad_inputs.size(), 1);
    grad_inputs[0] = grad_outputs[0];
    if (grad_inputs[0]) {
      grad_inputs[0]->Ref();
    }
    return Status::OK();
  }
};

// Computes:
//
// @tf.custom_gradient
// def f(input):
//   def grad(grads):
//     return grads[0]
//   return tf.exp(input), grad
// outputs = [f(inputs[0])]
Status ExpWithPassThroughGrad(AbstractContext* ctx,
                              absl::Span<AbstractTensorHandle* const> inputs,
                              absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPScustom_gradient_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/c/experimental/gradients/custom_gradient_test.cc", "ExpWithPassThroughGrad");

  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]);  // Watch x.
  AbstractTensorHandle* exp_output;
  TF_RETURN_IF_ERROR(ops::Exp(ctx, inputs[0], &exp_output, "Exp"));
  std::unique_ptr<GradientFunction> gradient_function(
      new PassThroughGradientFunction);
  tape.RecordOperation(inputs, {exp_output}, gradient_function.release());
  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx,
                                          /*targets*/ {exp_output},
                                          /*sources=*/inputs,
                                          /*output_gradients=*/{},
                                          /*result=*/outputs));
  exp_output->Unref();
  return Status::OK();
}

TEST_P(CustomGradientTest, ExpWithPassThroughGrad) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = exp(x)
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  Status s = RunModel(ExpWithPassThroughGrad, ctx.get(), {x.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = GetValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    CustomGradientTest, CustomGradientTest,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(true, false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    CustomGradientTest, CustomGradientTest,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
