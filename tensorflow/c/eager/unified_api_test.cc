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
class MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc() {
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
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
class UnifiedAPI
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/c/eager/unified_api_test.cc", "SetUp");

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();
  }

 public:
  bool UseMlir() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/c/eager/unified_api_test.cc", "UseMlir");
 return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc mht_2(mht_2_v, 213, "", "./tensorflow/c/eager/unified_api_test.cc", "UseFunction");
 return std::get<2>(GetParam()); }
};

// Checks that inputs[0] is a scalar.
Status TestScalarShape(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc mht_3(mht_3_v, 222, "", "./tensorflow/c/eager/unified_api_test.cc", "TestScalarShape");

  PartialTensorShape shape;
  TF_RETURN_IF_ERROR(inputs[0]->Shape(&shape));
  if (shape.dims() != 0) {
    return errors::InvalidArgument(
        "Tensor expected to have scalar shape found rank: ", shape.dims());
  }
  return Status::OK();
}

TEST_P(UnifiedAPI, TestTensorShapeScalar) {
  if (UseFunction() && UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
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
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  Status s = RunModel(TestScalarShape, ctx.get(),
                      /*inputs=*/{x.get()},
                      /*outputs=*/{},
                      /*use_function=*/UseFunction());
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
}

// Checks that inputs[0] is a matrix with shape 2x4.
Status TestTensorShape2x4(AbstractContext* ctx,
                          absl::Span<AbstractTensorHandle* const> inputs,
                          absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/c/eager/unified_api_test.cc", "TestTensorShape2x4");

  PartialTensorShape shape;
  TF_RETURN_IF_ERROR(inputs[0]->Shape(&shape));
  if (shape.dims() != 2) {
    return errors::InvalidArgument(
        "Tensor expected to have rank 2 found rank: ", shape.dims());
  }
  int64_t dim_sizes[] = {2, 4};
  for (int i = 0; i < shape.dims(); i++) {
    if (shape.dim_size(i) != dim_sizes[i]) {
      return errors::InvalidArgument("Dim ", i, " expected to be of size ",
                                     dim_sizes[i],
                                     " found: ", shape.dim_size(i));
    }
  }
  return Status::OK();
}

TEST_P(UnifiedAPI, TestTensorShape2x4) {
  if (UseFunction() && UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
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
    float data[] = {0., 0., 0., 0., 0., 0., 0., 0};
    int64_t dim_sizes[] = {2, 4};
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx.get(), data,
                                                         dim_sizes, 2, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  Status s = RunModel(TestTensorShape2x4, ctx.get(),
                      /*inputs=*/{x.get()},
                      /*outputs=*/{},
                      /*use_function=*/UseFunction());
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
}

TEST_P(UnifiedAPI, TestUnknownShapeTracing) {
  if (!UseFunction()) {
    GTEST_SKIP() << "Tracing only test.";
  }
  if (UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx(BuildFunction("test_fn"));
  AbstractTensorHandlePtr x;
  {
    tracing::TracingTensorHandle* x_raw = nullptr;
    PartialTensorShape shape;
    Status s = dyn_cast<tracing::TracingContext>(ctx.get())->AddParameter(
        DT_FLOAT, shape, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  PartialTensorShape shape;
  Status s = x->Shape(&shape);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  ASSERT_TRUE(shape.unknown_rank());
}

TEST_P(UnifiedAPI, TestPartialShapeTracing) {
  if (!UseFunction()) {
    GTEST_SKIP() << "Tracing only test.";
  }
  if (UseMlir()) {
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx(BuildFunction("test_fn"));
  AbstractTensorHandlePtr x;
  {
    tracing::TracingTensorHandle* x_raw = nullptr;
    PartialTensorShape shape;
    int64_t dim_sizes[] = {2, -1};
    Status s = PartialTensorShape::MakePartialShape(dim_sizes, 2, &shape);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    s = dyn_cast<tracing::TracingContext>(ctx.get())->AddParameter(
        DT_FLOAT, shape, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  PartialTensorShape shape;
  Status s = x->Shape(&shape);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  ASSERT_FALSE(shape.unknown_rank());

  ASSERT_EQ(2, shape.dim_size(0));
  ASSERT_EQ(-1, shape.dim_size(1));
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCppAPI, UnifiedAPI,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(true, false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCppAPI, UnifiedAPI,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace tensorflow
