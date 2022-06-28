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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/resource_variable_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/function.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
using tensorflow::AbstractContext;
using tensorflow::AbstractContextPtr;
using tensorflow::AbstractFunctionPtr;
using tensorflow::AbstractTensorHandle;
using tensorflow::DT_FLOAT;
using tensorflow::PartialTensorShape;
using tensorflow::Status;
using tensorflow::TF_StatusPtr;

class VariableTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 public:
  template <class T, TF_DataType datatype>
  impl::TaggedValueTensor CreateScalarTensor(T val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/cc/experimental/libtf/tests/variable_test.cc", "CreateScalarTensor");

    AbstractTensorHandle* raw = nullptr;
    Status s = TestScalarTensorHandle<T, datatype>(ctx_.get(), val, &raw);
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    return impl::TaggedValueTensor(raw, /*add_ref=*/false);
  }

  bool UseTfrt() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc mht_1(mht_1_v, 227, "", "./tensorflow/cc/experimental/libtf/tests/variable_test.cc", "UseTfrt");
 return std::get<1>(GetParam()); }

  AbstractContextPtr ctx_;

 protected:
  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/cc/experimental/libtf/tests/variable_test.cc", "SetUp");

    // Set the tracing impl, GraphDef vs MLIR.
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = tensorflow::StatusFromTF_Status(status.get());
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();

    // Set the runtime impl, Core RT vs TFRT.
    AbstractContext* ctx_raw = nullptr;
    s = BuildImmediateExecutionContext(UseTfrt(), &ctx_raw);
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    ctx_.reset(ctx_raw);
  }
};

template <typename T>
void ExpectEquals(AbstractTensorHandle* t, T expected) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSvariable_testDTcc mht_3(mht_3_v, 254, "", "./tensorflow/cc/experimental/libtf/tests/variable_test.cc", "ExpectEquals");

  TF_Tensor* result_t;
  Status s = tensorflow::GetValue(t, &result_t);
  ASSERT_TRUE(s.ok()) << s.error_message();
  auto value = static_cast<T*>(TF_TensorData(result_t));
  EXPECT_EQ(*value, expected);
  TF_DeleteTensor(result_t);
}

TEST_P(VariableTest, CreateAssignReadDestroy) {
  // Create uninitialized variable.
  tensorflow::AbstractTensorHandlePtr var;
  {
    AbstractTensorHandle* var_ptr = nullptr;
    PartialTensorShape scalar_shape;
    TF_EXPECT_OK(
        PartialTensorShape::MakePartialShape<int32>({}, 0, &scalar_shape));
    TF_EXPECT_OK(tensorflow::ops::VarHandleOp(ctx_.get(), &var_ptr, DT_FLOAT,
                                              scalar_shape));
    var.reset(var_ptr);
  }
  // Assign a value.
  auto x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  TF_EXPECT_OK(
      tensorflow::ops::AssignVariableOp(ctx_.get(), var.get(), x.get()));
  // Read variable.
  tensorflow::AbstractTensorHandlePtr value;
  {
    AbstractTensorHandle* value_ptr = nullptr;
    TF_EXPECT_OK(tensorflow::ops::ReadVariableOp(ctx_.get(), var.get(),
                                                 &value_ptr, DT_FLOAT));
    value.reset(value_ptr);
  }
  ExpectEquals(value.get(), 2.0f);
  // Destroy variable.
  TF_EXPECT_OK(tensorflow::ops::DestroyResourceOp(ctx_.get(), var.get()));
}

INSTANTIATE_TEST_SUITE_P(TF2CAPI, VariableTest,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false, true)));

}  // namespace libtf
}  // namespace tf
