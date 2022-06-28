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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePScrop_and_resize_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScrop_and_resize_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePScrop_and_resize_op_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class CropAndResizeOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(float extrapolation_value, const string& method) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScrop_and_resize_op_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/image/crop_and_resize_op_test.cc", "MakeOp");

    TF_EXPECT_OK(NodeDefBuilder("crop_and_resize_op", "CropAndResize")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Attr("extrapolation_value", extrapolation_value)
                     .Attr("method", method)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                               \
  TEST_F(CropAndResizeOpTest, TestCropAndResize##T) {                  \
    MakeOp<T>(0, "bilinear");                                          \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {2.5});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }                                                                    \
                                                                       \
  TEST_F(CropAndResizeOpTest, TestCropAndResize##T##nearest) {         \
    MakeOp<T>(0, "nearest");                                           \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {4.0});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8)
REGISTER_TEST(uint16)
REGISTER_TEST(int8)
REGISTER_TEST(int16)
REGISTER_TEST(int32)
REGISTER_TEST(int64_t)

#undef REGISTER_TEST

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Uint8) {
  MakeOp<uint8>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<uint8>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Uint8NearestNeibor) {
  MakeOp<uint8>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<uint8>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {4.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {4.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3NearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  2,  2,
     3,  4,  4,
     3,  4,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {4,  3.5,  3,
     3,  2.5,  2,
     2,  1.5,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {4,  4,  3,
     4,  4,  3,
     2,  2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {0, 0, 1, 1, 0, 0, 0.5, 0.5});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9,
     1,  2,
     4,  5});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2NearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {0, 0, 1, 1, 0, 0, 0.5, 0.5});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9,
     1,  2,
     4,  5});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {1, 1, 0, 0, 0.5, 0.5, 0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {9,  7,
     3,  1,
     5,  4,
     2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {1, 1, 0, 0, 0.5, 0.5, 0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {9,  7,
     3,  1,
     5,  4,
     2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3Extrapolated) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {-1, -1, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {v,  v,  v,
     v,  1,  2,
     v,  3,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3NoCrop) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<int32>(TensorShape({0}), {});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({0, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected, {});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestInvalidInputShape) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "input image must be 4-D")) << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndexShape) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "box_index has incompatible shape"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndex) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {1});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "box_index has values outside [0, batch_size)"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestWithSharding) {
  MakeOp<float>(0, "bilinear");
  // Generate a relatively large input (999x999) so that sharding happens.
  const int kLength = 999;  // Length of the input. Must use an odd number.
  const int kHalf = (kLength + 1) / 2;  // Half size for the cropped result.

  // Input:
  //  0, 1, 2, ..., 998
  //  0, 1, 2, ..., 998
  //  ... (altogether 999 lines)
  //  0, 1, 2, ..., 998
  AddInput<float>(TensorShape({1, kLength, kLength, 1}),
                  [=](int i) -> float { return i % kLength; });
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {kHalf, kHalf});

  TF_ASSERT_OK(RunOpKernel());

  // Generate result tensor.
  // Result 1:
  //  0, 1, 2, ..., 499
  //  ... (altogether 500 lines)
  //  0, 1, 2, ..., 499
  Tensor result1(allocator(), DT_FLOAT, TensorShape({1, kHalf, kHalf, 1}));
  test::FillFn<float>(&result1, [=](int i) -> float { return i % kHalf; });

  // Result 2:
  //  499, 500, 501, ..., 998
  //  ... (altogether 500 lines)
  //  499, 500, 501, ..., 998
  Tensor result2(allocator(), DT_FLOAT, TensorShape({1, kHalf, kHalf, 1}));
  test::FillFn<float>(&result2,
                      [=](int i) -> float { return i % kHalf + kHalf - 1; });

  // Expected result is the concat of the two tensors.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, kHalf, kHalf, 1}));
  TF_ASSERT_OK(tensor::Concat({result1, result2}, &expected));

  // Compare result.
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace tensorflow
