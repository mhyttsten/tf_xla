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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc() {
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

// TODO(shlens, sherrym): Consider adding additional tests in image_ops.py in
// order to compare the reference implementation for image resizing in Python
// Image Library.
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
enum class TestDevice { kCPU, kGPU };

class ResizeNearestNeighborOpTestBase
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  explicit ResizeNearestNeighborOpTestBase(bool half_pixel_centers)
      : align_corners_(false), half_pixel_centers_(half_pixel_centers) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "ResizeNearestNeighborOpTestBase");
}
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "SetUp");

    if (GetParam() == TestDevice::kGPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice(/*type=*/"GPU", /*options=*/{},
                                   /*name_prefix=*/"/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_nn_op", "ResizeNearestNeighbor")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Attr("half_pixel_centers", half_pixel_centers_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
  bool align_corners_;
  bool half_pixel_centers_;
};

class ResizeNearestNeighborOpTest : public ResizeNearestNeighborOpTestBase {
 protected:
  ResizeNearestNeighborOpTest() : ResizeNearestNeighborOpTestBase(false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "ResizeNearestNeighborOpTest");
}
};

class ResizeNearestNeighborHalfPixelCentersOpTest
    : public ResizeNearestNeighborOpTestBase {
 protected:
  ResizeNearestNeighborHalfPixelCentersOpTest()
      : ResizeNearestNeighborOpTestBase(true) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "ResizeNearestNeighborHalfPixelCentersOpTest");
}
};

// TODO(jflynn): Add some actual tests for the half pixel centers case.

class ResizeNearestNeighborOpAlignCornersTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  ResizeNearestNeighborOpAlignCornersTest() : align_corners_(true) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_4(mht_4_v, 261, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "ResizeNearestNeighborOpAlignCornersTest");
}
  void SetUp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_nearest_neighbor_op_testDTcc mht_5(mht_5_v, 265, "", "./tensorflow/core/kernels/image/resize_nearest_neighbor_op_test.cc", "SetUp");

    if (GetParam() == TestDevice::kGPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice(/*type=*/"GPU", /*options=*/{},
                                   /*name_prefix=*/"/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_nn_op", "ResizeNearestNeighbor")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
  bool align_corners_;
};

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {1});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearest2x2AlignCornersTo1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {1});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2,
     1, 1, 2,
     3, 3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2, 2,
     3, 4, 4,
     3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     4, 5});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestAlignCorners3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 3,
     7, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To2x5) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {2, 5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 5, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1, 2, 2,
     3, 3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearestNeighbor4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  2,  3,
     5,  6,  7,
     9, 10, 11});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestNeighborAlignCorners4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    { 1,  3,  4,
      9, 11, 12,
     13, 15, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To5x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {5, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 5, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     1, 2,
     1, 2,
     3, 4,
     3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2,
     1, 1, 2, 2,
     3, 3, 4, 4,
     3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2],
  //    [ 3, 3 ], [ 4, 4] ],
  //  [ [ 5, 5 ], [ 6, 6],
  //    [ 7, 7 ], [ 8, 8] ]
  AddInputFromArray<float>(TensorShape({2, 2, 2, 2}),
                           {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 2}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1,
     1, 2, 2,
     1, 1, 1,
     1, 2, 2,
     3, 3, 3,
     3, 4, 4,
     5, 5, 5,
     5, 6, 6,
     5, 5, 5,
     5, 6, 6,
     7, 7, 7,
     7, 8, 8});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest5x2To2x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 5, 1}),
                           {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {2, 4, 2, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2, 2,
     3, 4, 4,
     3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 3,
     7, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To2x5) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {2, 5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 5, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2, 2,
     3, 3, 4, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest,
       TestNearestNeighbor4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,  4,
     9, 11, 12,
     13, 15, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To5x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {5, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 5, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     1, 2,
     3, 4,
     3, 4,
     3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2,
     1, 1, 2, 2,
     3, 3, 4, 4,
     3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest,
       TestNearest2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2],
  //    [ 3, 3 ], [ 4, 4] ],
  //  [ [ 5, 5 ], [ 6, 6],
  //    [ 7, 7 ], [ 8, 8] ]
  AddInputFromArray<float>(TensorShape({2, 2, 2, 2}),
                           {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 2}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2, 2, 2,
     3, 3, 4, 4, 4, 4,
     3, 3, 4, 4, 4, 4,
     5, 5, 6, 6, 6, 6,
     7, 7, 8, 8, 8, 8,
     7, 7, 8, 8, 8, 8});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTestCpu,
                         ResizeNearestNeighborOpTest,
                         ::testing::Values(TestDevice::kCPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborHalfPixelCentersOpTestCpu,
                         ResizeNearestNeighborHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::kCPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpAlignCornersTestCpu,
                         ResizeNearestNeighborOpAlignCornersTest,
                         ::testing::Values(TestDevice::kCPU));
#if GOOGLE_CUDA
// Instantiate tests for kGPU.
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTestGpu,
                         ResizeNearestNeighborOpTest,
                         ::testing::Values(TestDevice::kGPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborHalfPixelCentersOpTestGpu,
                         ResizeNearestNeighborHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::kGPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpAlignCornersTestGpu,
                         ResizeNearestNeighborOpAlignCornersTest,
                         ::testing::Values(TestDevice::kGPU));
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
