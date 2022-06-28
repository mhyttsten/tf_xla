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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc() {
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
#include "tensorflow/cc/experimental/libtf/function.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
using tensorflow::FunctionDef;
using tensorflow::FunctionDefHelper;
using tensorflow::PartialTensorShape;
using tensorflow::Status;
using tensorflow::StatusOr;
using tensorflow::TF_StatusPtr;
using tensorflow::tracing::graph::GraphFunction;

class FunctionTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 public:
  template <class T, TF_DataType datatype>
  impl::TaggedValueTensor CreateScalarTensor(T val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "CreateScalarTensor");

    AbstractTensorHandle* raw = nullptr;
    Status s = TestScalarTensorHandle<T, datatype>(ctx_.get(), val, &raw);
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    return impl::TaggedValueTensor(raw, /*add_ref=*/false);
  }

  bool UseTfrt() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "UseTfrt");
 return std::get<1>(GetParam()); }

  AbstractContextPtr ctx_;

 protected:
  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "SetUp");

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

// TODO(b/191361582): Use Abstract* APIs for building functions so that we can
// test with MLIR.
FunctionDef SquareFunc() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "SquareFunc");

  return FunctionDefHelper::Define(
      // Function Name
      "SquareFunc",
      // Args
      {"x: float"},
      // Returns
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"y"},
        /*op=*/"Square",
        /*arg=*/{"x"},
        /*attr=*/{{"T", DT_FLOAT}},
        /*dep=*/{},
        /*device=*/"",
        /*name=*/"square"}});
}

FunctionDef AddFunc() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_4(mht_4_v, 280, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "AddFunc");

  return FunctionDefHelper::Define(
      // Function Name
      "AddFunc",
      // Args
      {"x: float", "y: float"},
      // Returns
      {"z: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"z"},
        /*op=*/"Add",
        /*arg=*/{"x", "y"},
        /*attr=*/{{"T", DT_FLOAT}},
        /*dep=*/{},
        /*device=*/"",
        /*name=*/"add"}});
}

FunctionDef IdentityNFunc() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_5(mht_5_v, 303, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "IdentityNFunc");

  return FunctionDefHelper::Define(
      // Function Name
      "IdentityNFunc",
      // Args
      {"x: float", "y: float"},
      // Returns
      {"u: float", "v: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"u", "v"},
        /*op=*/"IdentityN",
        /*arg=*/{"x", "y"},
        /*attr=*/{{"T", tensorflow::DataTypeSlice({DT_FLOAT, DT_FLOAT})}},
        /*dep=*/{},
        /*device=*/""}});
}

template <typename T>
void ExpectEquals(AbstractTensorHandle* t, T expected) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPStestsPSfunction_testDTcc mht_6(mht_6_v, 326, "", "./tensorflow/cc/experimental/libtf/tests/function_test.cc", "ExpectEquals");

  TF_Tensor* result_t;
  Status s = tensorflow::GetValue(t, &result_t);
  ASSERT_TRUE(s.ok()) << s.error_message();
  auto value = static_cast<T*>(TF_TensorData(result_t));
  EXPECT_EQ(*value, expected);
  TF_DeleteTensor(result_t);
}

// TODO(srbs): Add tests for captures.
// TODO(srbs): Add tests for polymorphism (different shapes and dtypes).
TEST_P(FunctionTest, Square) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = SquareFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args(std::move(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().error_message();
  const TaggedValue& result = v.ValueOrDie();
  AbstractTensorHandle* t = result.tensor().get();
  ExpectEquals(t, 4.0f);
}

TEST_P(FunctionTest, Add) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = AddFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(tensor_spec);
  input_signature.tuple().emplace_back(tensor_spec);
  Status s =
      tf_function.RegisterTrace(std::move(trace), input_signature, tensor_spec);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().error_message();
  const TaggedValue& result = v.ValueOrDie();
  ExpectEquals(result.tensor().get(), 4.0f);
}

TEST_P(FunctionTest, IdentityN) {
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  impl::TaggedValueTensor y = CreateScalarTensor<float, TF_FLOAT>(4.0f);
  FunctionDef fdef = IdentityNFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue signature = TaggedValue::Tuple();
  signature.tuple().emplace_back(tensor_spec);
  signature.tuple().emplace_back(tensor_spec);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(y));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().error_message();
  const TaggedValue& result = v.ValueOrDie();
  ExpectEquals(result.tuple()[0].tensor().get(), 2.0f);
  ExpectEquals(result.tuple()[1].tensor().get(), 4.0f);
}

TEST_P(FunctionTest, UnaryFuncCalledWithMultipleArgsFails) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = SquareFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInvalidArgument(v.status()));
  ASSERT_TRUE(absl::StrContains(v.status().error_message(), "No match"));
}

TEST_P(FunctionTest, IncorrectArityOfOutputSignatureFails) {
  if (UseTfrt()) {
    GTEST_SKIP() << "TFRT crashes if expected number of output tensors does not"
                    " match actual.";
  }
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  impl::TaggedValueTensor y = CreateScalarTensor<float, TF_FLOAT>(4.0f);
  FunctionDef fdef = IdentityNFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(tensor_spec);
  input_signature.tuple().emplace_back(tensor_spec);
  // This is wrong!
  TaggedValue output_signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), input_signature,
                                       output_signature);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(y));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInvalidArgument(v.status())) << v.status();
  ASSERT_TRUE(absl::StrContains(v.status().error_message(),
                                "Expecting 2 outputs, but *num_retvals is 1"));
}

TEST_P(FunctionTest, IncorrectDtypeInOutputSignatureFails) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = AddFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue input_tensor_spec(unknown_shape, tensorflow::DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(input_tensor_spec);
  input_signature.tuple().emplace_back(input_tensor_spec);
  // Incorrect type.
  TaggedValue output_tensor_spec(unknown_shape, tensorflow::DT_INT64);
  Status s = tf_function.RegisterTrace(std::move(trace), input_signature,
                                       output_tensor_spec);
  ASSERT_TRUE(s.ok()) << s.error_message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInternal(v.status())) << v.status();
  ASSERT_TRUE(absl::StrContains(v.status().error_message(),
                                "Shape and dtype of tensor"));
  ASSERT_TRUE(absl::StrContains(v.status().error_message(),
                                "does not match that in signature"));
}

INSTANTIATE_TEST_SUITE_P(TF2CAPI, FunctionTest,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false, true)));

}  // namespace libtf
}  // namespace tf
