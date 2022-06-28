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
class MHTracer_DTPStensorflowPScorePSkernelsPSroll_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSroll_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSroll_op_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class RollOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, DataType index_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSroll_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/roll_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "Roll")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RollOpTest, ScalarIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {2, 3, 4, 0, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ScalarIndices_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({5}), {"a", "b", "c", "d", "e"});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5}));
  test::FillValues<tstring>(&expected, {"c", "d", "e", "a", "b"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, DT_INT32);

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_COMPLEX64, TensorShape({5}));
  test::FillValues<std::complex<float>>(
      &expected, {std::complex<float>(2, 12), std::complex<float>(3, 13),
                  std::complex<float>(4, 14), std::complex<float>(0, 10),
                  std::complex<float>(1, 11)});
  test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({3, 5}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({2}), {2, -1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 5}));
  test::FillValues<float>(&expected,
                          {6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 1, 2, 3, 4, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({3, 5}),
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                              "k", "l", "m", "n", "o"});
  AddInputFromArray<int32>(TensorShape({2}), {2, -1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({3, 5}));
  test::FillValues<tstring>(&expected, {"g", "h", "i", "j", "f", "l", "m", "n",
                                        "o", "k", "b", "c", "d", "e", "a"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({3}), {1, -1, -1});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {10, 11, 9, 7, 8, 6, 4, 5, 3, 1, 2, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(
      TensorShape({2, 2, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int32>(TensorShape({3}), {1, -1, -1});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({2, 2, 3}));
  test::FillValues<tstring>(
      &expected, {"k", "l", "j", "h", "i", "g", "e", "f", "d", "b", "c", "a"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64_t>(TensorShape({2}), {-1, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected,
                          {5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 2, 0, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD64_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT64);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({5, 3}),
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                              "k", "l", "m", "n", "o"});
  AddInputFromArray<int64_t>(TensorShape({2}), {-1, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5, 3}));
  test::FillValues<tstring>(&expected, {"f", "d", "e", "i", "g", "h", "l", "j",
                                        "k", "o", "m", "n", "c", "a", "b"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4, 1, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int64_t>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int64_t>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 1, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD64_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT64);

  // Feed and run
  AddInputFromArray<tstring>(
      TensorShape({4, 1, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int64_t>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int64_t>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({4, 1, 3}));
  test::FillValues<tstring>(
      &expected, {"b", "c", "a", "e", "f", "d", "h", "i", "g", "k", "l", "j"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroShift_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroShift_ThreeD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(
      TensorShape({2, 2, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({2, 2, 3}));
  test::FillValues<tstring>(
      &expected, {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroSize_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0, 0}), {});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 0, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroSize_ThreeD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({5, 0, 0}), {});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5, 0, 0}));
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, OneSize_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({1, 1, 1}), {5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1}));
  test::FillValues<float>(&expected, {5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, OneSize_ThreeD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({1, 1, 1}), {"a"});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({1, 1, 1}));
  test::FillValues<tstring>(&expected, {"a"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, MultiShifts_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({3, 5}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {-2, 2, -1, 1});
  AddInputFromArray<int32>(TensorShape({4}), {1, 0, 0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 5}));
  test::FillValues<float>(&expected,
                          {11, 12, 13, 14, 10, 1, 2, 3, 4, 0, 6, 7, 8, 9, 5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, MultiShifts_TwoD32_NoMemcpy) {
  MakeOp(DT_STRING, DT_INT32);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({3, 5}),
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                              "k", "l", "m", "n", "o"});
  AddInputFromArray<int32>(TensorShape({4}), {-2, 2, -1, 1});
  AddInputFromArray<int32>(TensorShape({4}), {1, 0, 0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({3, 5}));
  test::FillValues<tstring>(&expected, {"l", "m", "n", "o", "k", "b", "c", "d",
                                        "e", "a", "g", "h", "i", "j", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Error_InputMustBeVectorOrHigher) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {7});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "input must be 1-D or higher"))
      << s;
}

TEST_F(RollOpTest, Error_AxisMustBeScalarOrVector) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "axis must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftMustBeScalarOrVector) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "shift must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftAndAxisMustBeSameSize) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1}), {1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "shift and axis must have the same size"))
      << s;
}

TEST_F(RollOpTest, Error_AxisOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "is out of range")) << s;
}

// isd - (inner shift dimension) The inner most dimension to be shifted.
//    All outer dimensions will also be shifted for testing.
static Graph* RollGraph(const TensorShape& shape, int isd) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSroll_op_testDTcc mht_1(mht_1_v, 605, "", "./tensorflow/core/kernels/roll_op_test.cc", "RollGraph");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, shape);
  input.flat<float>().setRandom();
  const int dims = static_cast<int>(input.dims());
  Tensor shift(DT_INT32, TensorShape({dims}));
  for (int i = 0; i < dims; i++) {
    // shift the inner shift dimension and all outer dimensions
    shift.flat<int32>()(i) = (i <= isd) ? 2 : 0;
  }
  Tensor axis(DT_INT32, TensorShape({dims}));
  for (int i = 0; i < dims; i++) {
    axis.flat<int32>()(i) = i;
  }
  test::graph::Roll(g, test::graph::Constant(g, input),
                    test::graph::Constant(g, shift),
                    test::graph::Constant(g, axis));
  return g;
}

#define BM_ROLL_OUTER(DEVICE)                                                  \
  static void BM_##DEVICE##_roll_outer(::testing::benchmark::State& state) {   \
    const int rows = state.range(0);                                           \
    const int columns = state.range(1);                                        \
                                                                               \
    TensorShape shape{rows, columns};                                          \
    test::Benchmark(#DEVICE, RollGraph(shape, 0), /*old_benchmark_api*/ false) \
        .Run(state);                                                           \
    const int64_t num_items =                                                  \
        static_cast<int64_t>(state.iterations()) * shape.num_elements();       \
    state.SetItemsProcessed(num_items);                                        \
    state.SetBytesProcessed(num_items * sizeof(float));                        \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_roll_outer)                                          \
      ->UseRealTime()                                                          \
      ->ArgPair(256, 256)                                                      \
      ->ArgPair(512, 512)                                                      \
      ->ArgPair(1024, 1024)                                                    \
      ->ArgPair(2048, 2048)

#define BM_ROLL_ALL(DEVICE)                                                    \
  static void BM_##DEVICE##_roll_all(::testing::benchmark::State& state) {     \
    const int rows = state.range(0);                                           \
    const int columns = state.range(1);                                        \
                                                                               \
    TensorShape shape{rows, columns};                                          \
    test::Benchmark(#DEVICE, RollGraph(shape, 1), /*old_benchmark_api*/ false) \
        .Run(state);                                                           \
    const int64_t num_items =                                                  \
        static_cast<int64_t>(state.iterations()) * shape.num_elements();       \
    state.SetItemsProcessed(num_items);                                        \
    state.SetBytesProcessed(num_items * sizeof(float));                        \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_roll_all)                                            \
      ->UseRealTime()                                                          \
      ->ArgPair(256, 256)                                                      \
      ->ArgPair(512, 512)                                                      \
      ->ArgPair(1024, 1024)                                                    \
      ->ArgPair(2048, 2048)

BM_ROLL_OUTER(cpu);
BM_ROLL_ALL(cpu);
}  // namespace
}  // namespace tensorflow
