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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc() {
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
#include "tensorflow/c/experimental/gradients/array_grad.h"

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status IdentityNModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/c/experimental/gradients/array_grad_test.cc", "IdentityNModel");

  std::vector<AbstractTensorHandle*> temp_outputs(2);
  TF_RETURN_IF_ERROR(
      ops::IdentityN(ctx, inputs, absl::MakeSpan(temp_outputs), "IdentityN"));
  // Although, `ops::IdentityN` returns 2 tensors, the first tensor isn't needed
  // for computing gradient so we could safely drop it.
  outputs[0] = temp_outputs[1];
  temp_outputs[0]->Unref();
  return Status::OK();
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/c/experimental/gradients/array_grad_test.cc", "SetUp");

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
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/c/experimental/gradients/array_grad_test.cc", "UseMlir");
 return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSarray_grad_testDTcc mht_3(mht_3_v, 254, "", "./tensorflow/c/experimental/gradients/array_grad_test.cc", "UseFunction");
 return std::get<2>(GetParam()); }
};

TEST_P(CppGradients, TestIdentityNGrad) {
  // This test is interesting because the current implementation of GradientTape
  // would return [0, 1] whereas we use build_default_zeros_grads=false here
  // so we get back [nullptr, 1].

  AbstractTensorHandlePtr x1;
  {
    AbstractTensorHandle* x1_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 1.0f, &x1_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x1.reset(x1_raw);
  }

  AbstractTensorHandlePtr x2;
  {
    AbstractTensorHandle* x2_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 1.0f, &x2_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x2.reset(x2_raw);
  }

  status_ = registry_.Register("IdentityN", IdentityNRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  auto IdentityNGradModel = BuildGradModel(IdentityNModel, registry_);

  std::vector<AbstractTensorHandle*> outputs(2);
  status_ =
      RunModel(IdentityNGradModel, immediate_execution_ctx_.get(),
               {x1.get(), x2.get()}, absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  EXPECT_EQ(outputs[0], nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], {1.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
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
