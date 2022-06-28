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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc() {
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
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
enum class TestDevice { CPU, GPU };

class ResizeBilinearOpTestBase
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  explicit ResizeBilinearOpTestBase()
      : align_corners_(false), half_pixel_centers_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "ResizeBilinearOpTestBase");
}

  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "SetUp");

    if (GetParam() == TestDevice::GPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice("GPU", {}, "/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Attr("half_pixel_centers", half_pixel_centers_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  const Tensor* SetRandomImageInput(const TensorShape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "SetRandomImageInput");

    inputs_.clear();

    CHECK_EQ(shape.dims(), 4) << "All images must have 4 dimensions.";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(allocator(), DataTypeToEnum<float>::v(), shape);
    input->flat<float>().setRandom();
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<float>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<float>::v());
      inputs_.push_back({nullptr, input});
    }
    return input;
  }

  // This is the straight forward unoptimized implementation of resize bilinear
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeBilinearBaseline(TTypes<float, 4>::ConstTensor images,
                              TTypes<float, 4>::Tensor output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "ResizeBilinearBaseline");

    const int batch = images.dimension(0);
    const int64_t in_height = images.dimension(1);
    const int64_t in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64_t out_height = output.dimension(1);
    const int64_t out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const float in_y =
            half_pixel_centers_
                ? (static_cast<float>(y) + 0.5f) * height_scale - 0.5f
                : y * height_scale;
        const int64_t top_y_index = std::max(static_cast<int64_t>(floorf(in_y)),
                                             static_cast<int64_t>(0));
        const int64_t bottom_y_index =
            std::min(static_cast<int64_t>(ceilf(in_y)), in_height - 1);
        const float y_lerp = in_y - std::floor(in_y);
        for (int64_t x = 0; x < out_width; ++x) {
          const float in_x =
              half_pixel_centers_
                  ? (static_cast<float>(x) + 0.5f) * width_scale - 0.5f
                  : x * width_scale;
          const int64_t left_x_index = std::max(
              static_cast<int64_t>(floorf(in_x)), static_cast<int64_t>(0));
          const int64_t right_x_index =
              std::min(static_cast<int64_t>(ceilf(in_x)), in_width - 1);
          const float x_lerp = in_x - std::floor(in_x);
          for (int c = 0; c < channels; ++c) {
            const float top_left = images(b, top_y_index, left_x_index, c);
            const float top_right = images(b, top_y_index, right_x_index, c);
            const float bottom_left =
                images(b, bottom_y_index, left_x_index, c);
            const float bottom_right =
                images(b, bottom_y_index, right_x_index, c);
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * x_lerp;
            output(b, y, x, c) = top + (bottom - top) * y_lerp;
          }
        }
      }
    }
  }

  void TestResize(int batch_size, int input_width, int input_height,
                  int channels, int output_width, int output_height) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_4(mht_4_v, 318, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "TestResize");

    const TensorShape shape({batch_size, input_width, input_height, channels});
    const Tensor* input = SetRandomImageInput(shape);
    AddInputFromArray<int32>(TensorShape({2}), {output_width, output_height});
    TF_ASSERT_OK(RunOpKernel());

    std::unique_ptr<Tensor> expected(new Tensor(
        allocator(), DataTypeToEnum<float>::v(),
        TensorShape({batch_size, output_width, output_height, channels})));
    ResizeBilinearBaseline(input->tensor<float, 4>(),
                           expected->tensor<float, 4>());
    test::ExpectClose(*expected, *GetOutput(0), /*atol=*/4e-5);
  }

  void RunManyRandomTests(int channels) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_5(mht_5_v, 335, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "RunManyRandomTests");

    for (int batch_size : {1, 2, 5}) {
      for (int in_w : {2, 4, 7, 20, 165}) {
        for (int in_h : {1, 3, 5, 8, 100, 233}) {
          for (int target_height : {1, 2, 3, 50, 113}) {
            for (int target_width : {target_height, target_height / 2 + 1}) {
              TestResize(batch_size, in_w, in_h, channels, target_width,
                         target_height);
            }
          }
        }
      }
    }
  }

  bool align_corners_;
  bool half_pixel_centers_;
};

class ResizeBilinearOpTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearOpTest() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_6(mht_6_v, 359, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "ResizeBilinearOpTest");
}
};

class ResizeBilinearHalfPixelCentersOpTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearHalfPixelCentersOpTest() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_7(mht_7_v, 367, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "ResizeBilinearHalfPixelCentersOpTest");
 half_pixel_centers_ = true; }
};

class ResizeBilinearOpAlignCornersTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearOpAlignCornersTest() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_8(mht_8_v, 375, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "ResizeBilinearOpAlignCornersTest");
 align_corners_ = true; }
};

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes1Channel) {
  RunManyRandomTests(1);
}

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes3Channels) {
  RunManyRandomTests(3);
}

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes4Channels) {
  RunManyRandomTests(4);
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinearRandom2x2To1x1) {
  const Tensor* input = SetRandomImageInput(TensorShape({1, 2, 2, 1}));
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor* output = GetOutput(0);
  std::unique_ptr<Tensor> expected(new Tensor(
      allocator(), DataTypeToEnum<float>::v(), TensorShape({1, 1, 1, 1})));
  ResizeBilinearBaseline(input->tensor<float, 4>(),
                         expected->tensor<float, 4>());
  EXPECT_EQ(input->flat<float>()(0), output->flat<float>()(0));
  test::ExpectClose(*expected, *output);
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,        5.0f / 3,  2,
     7.0f / 3, 3,         10.0f / 3,
     3,        11.0f / 3, 4});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // The corners exactly align with the original corners, and we bilinear
  // interpolate the values in between.

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear3x3To2x2) {
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
    {1,   2.5,
     5.5,   7});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners3x3To2x2) {
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
    {1,  3,
     7,  9});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear3x3To4x4) {
  // Input:
  //  1, 2, 3,
  //  4, 5, 6,
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1.75, 2.5, 3,
     3.25, 4, 4.75, 5.25,
     5.5, 6.25, 7, 7.5,
     7,  7.75, 8.5, 9});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear4x4To3x3) {
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
    {1,        7.0f/3, 11.0f/3,
     19.0f/3, 23.0f/3, 27.0f/3,
     35.0f/3, 39.0f/3, 43.0f/3});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearHalfPixelCentersOpTest, TestDownsamples) {
  TestResize(4, 298, 297, 3, 61, 71);
}

TEST_P(ResizeBilinearHalfPixelCentersOpTest, TestUpsamples) {
  TestResize(4, 61, 71, 3, 298, 297);
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners4x4To3x3) {
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
    { 1,  2.5,  4,
      7,  8.5, 10,
     13, 14.5, 16});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To3x3Batch2) {
  // Input:
  //  1, 2
  //  3, 4
  //
  // repeated twice
  AddInputFromArray<float>(TensorShape({2, 2, 2, 1}), {1, 2, 3, 4, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4,
     1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4
    });
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2x2To3x3x2) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 2}),
                           {1, -1, 2, -2, 3, -3, 4, -4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 2}));
  // clang-format off
  test::FillValues<float>(&expected,
    {
      1,       -1,
      5.0f/3,  -5.0f/3,
      2,       -2,
      7.0f/3,  -7.0f/3,
      3,       -3,
      10.0f/3, -10.0f/3,
      3,       -3,
      11.0f/3, -11.0f/3,
      4,       -4
    });
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5, 2, 2,
     2,  2.5, 3, 3,
     3,  3.5, 4, 4,
     3,  3.5, 4, 4});
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

// similar_size case
TEST_P(ResizeBilinearOpTest, Test1_1c) { TestResize(1, 183, 299, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test1_3c) { TestResize(1, 183, 299, 3, 299, 299); }

// Significantly smaller: scale_up case
TEST_P(ResizeBilinearOpTest, Test2_1c) { TestResize(1, 141, 186, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test2_3c) { TestResize(1, 141, 186, 3, 299, 299); }

// Significantly larger: scale_down case
TEST_P(ResizeBilinearOpTest, Test3_1c) { TestResize(1, 749, 603, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test3_3c) { TestResize(1, 749, 603, 3, 299, 299); }

// Exactly the same size
TEST_P(ResizeBilinearOpTest, Test4_1c) { TestResize(1, 299, 299, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test4_3c) { TestResize(1, 299, 299, 3, 299, 299); }

// Slightly smaller: similar_size case
TEST_P(ResizeBilinearOpTest, Test5_1c) { TestResize(1, 298, 297, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test5_3c) { TestResize(1, 298, 297, 3, 299, 299); }

// Slightly bigger: similar_size case
TEST_P(ResizeBilinearOpTest, Test6_1c) { TestResize(1, 304, 303, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test6_3c) { TestResize(1, 304, 303, 3, 299, 299); }

TEST_P(ResizeBilinearOpTest, TestInvalidOutputSize) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "output dimensions must be positive"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidInputShape) {
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "input must be 4-dimensional"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidSizeDim) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "shape_t must be 1-dimensional"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidSizeElements) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {4, 4, 1});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "shape_t must have two elements"))
      << s;
}

INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpTestCpu, ResizeBilinearOpTest,
                         ::testing::Values(TestDevice::CPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearHalfPixelCentersOpTestCpu,
                         ResizeBilinearHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::CPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpAlignCornersTestCpu,
                         ResizeBilinearOpAlignCornersTest,
                         ::testing::Values(TestDevice::CPU));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Instantiate tests for GPU.
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpTestGpu, ResizeBilinearOpTest,
                         ::testing::Values(TestDevice::GPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearHalfPixelCentersOpTestGpu,
                         ResizeBilinearHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::GPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpAlignCornersTestGpu,
                         ResizeBilinearOpAlignCornersTest,
                         ::testing::Values(TestDevice::GPU));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class ResizeBM : public ResizeBilinearOpTest {
 public:
  void TestBody() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_9(mht_9_v, 751, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "TestBody");
}
  void SetUpBenchmark(int input_width, int input_height, int num_channels,
                      int output_width, int output_height) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bilinear_op_testDTcc mht_10(mht_10_v, 756, "", "./tensorflow/core/kernels/image/resize_bilinear_op_test.cc", "SetUpBenchmark");

    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Attr("half_pixel_centers", half_pixel_centers_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    const TensorShape shape(
        {/*batch_size*/ 1, input_width, input_height, num_channels});
    SetRandomImageInput(shape);
    AddInputFromArray<int32>(TensorShape({2}), {output_width, output_height});
  }

  using ResizeBilinearOpTest::RunOpKernel;
};

#ifdef PLATFORM_GOOGLE

void BM_Resize(benchmark::State& state) {
  ResizeBM bench;
  bench.SetUpBenchmark(640, 480, 3, 1024, 768);
  for (const auto _ : state) {
    CHECK(bench.RunOpKernel().ok());
  }
}
BENCHMARK(BM_Resize);

#endif

}  // namespace tensorflow
