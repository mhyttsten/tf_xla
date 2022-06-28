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
class MHTracer_DTPStensorflowPScorePSkernelsPSas_string_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSas_string_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSas_string_op_testDTcc() {
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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class AsStringGraphTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type, const string& fill = "", int width = -1,
              int precision = -1, bool scientific = false,
              bool shortest = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSas_string_op_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/as_string_op_test.cc", "Init");

    TF_CHECK_OK(NodeDefBuilder("op", "AsString")
                    .Input(FakeInput(input_type))
                    .Attr("fill", fill)
                    .Attr("precision", precision)
                    .Attr("scientific", scientific)
                    .Attr("shortest", shortest)
                    .Attr("width", width)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(AsStringGraphTest, Int8) {
  TF_ASSERT_OK(Init(DT_INT8));

  AddInputFromArray<int8>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Int64) {
  TF_ASSERT_OK(Init(DT_INT64));

  AddInputFromArray<int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatDefault) {
  TF_ASSERT_OK(Init(DT_FLOAT));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatScientific) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/true));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-4.200000e+01", "0.000000e+00",
                                        "3.141590e+00", "4.200000e+01"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatShortest) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/false, /*shortest=*/true));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42", "0", "3.14159", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatPrecisionOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/2));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", "0.00", "3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatWidthOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Float_5_2_Format) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", " 0.00", " 3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Complex) {
  TF_ASSERT_OK(Init(DT_COMPLEX64, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromArray<complex64>(TensorShape({3}), {{-4, 2}, {0}, {3.14159, -1}});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(
      &expected, {"(-4.00, 2.00)", "( 0.00, 0.00)", "( 3.14,-1.00)"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Bool) {
  TF_ASSERT_OK(Init(DT_BOOL));

  AddInputFromArray<bool>(TensorShape({2}), {true, false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"true", "false"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Variant) {
  TF_ASSERT_OK(Init(DT_VARIANT));

  AddInput(DT_VARIANT, TensorShape({4}));
  auto inputs = mutable_input(0)->flat<Variant>();
  inputs(0) = 2;
  inputs(1) = 3;
  inputs(2) = true;
  inputs(3) = Tensor("hi");
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"Variant<type: int value: 2>", "Variant<type: int value: 3>",
                  "Variant<type: bool value: 1>",
                  ("Variant<type: tensorflow::Tensor value: Tensor<type: string"
                   " shape: [] values: hi>>")});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, String) {
  Status s = Init(DT_STRING);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "Value for attr 'T' of string is not in the list of allowed values"));
}

TEST_F(AsStringGraphTest, OnlyOneOfScientificAndShortest) {
  Status s = Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/true, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(),
                        "Cannot select both scientific and shortest notation"));
}

TEST_F(AsStringGraphTest, NoShortestForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/false, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoScientificForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoPrecisionForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/5);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.error_message(),
                                "precision not supported for datatype"));
}

TEST_F(AsStringGraphTest, LongFill) {
  Status s = Init(DT_INT32, /*fill=*/"asdf");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.error_message(),
                                "Fill string must be one or fewer characters"));
}

TEST_F(AsStringGraphTest, FillWithZero) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"0", /*width=*/4));

  AddInputFromArray<int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-042", "0000", "0042"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithSpace) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/" ", /*width=*/4));

  AddInputFromArray<int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {" -42", "   0", "  42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar1) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"-", /*width=*/4));

  AddInputFromArray<int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42 ", "0   ", "42  "});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar3) {
  Status s = Init(DT_INT32, /*fill=*/"s");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(), "Fill argument not supported"));
}

TEST_F(AsStringGraphTest, FillWithChar4) {
  Status s = Init(DT_INT32, /*fill=*/"n");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(), "Fill argument not supported"));
}

}  // end namespace
}  // end namespace tensorflow
