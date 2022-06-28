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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc() {
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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// These tests are copied from non_max_suppression_op_test.cc file and modified
// to use GPU ops. See other file for test details.

class NonMaxSuppressionV2GPUOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/image/non_max_suppression_op_gpu_test.cc", "MakeOp");

    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(
        NodeDefBuilder("non_max_suppression_op_gpu", "NonMaxSuppressionV2")
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_INT32))
            .Input(FakeInput(DT_FLOAT))
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,     0, 0,      // score= 0.9
                            0, 0.1f,  1, 1.1f,   // score= 0.75
                            0, .9f,   1, -0.1f,  // score= 0.6
                            0, 10,    1, 11,     // score= 0.95
                            1, 10.1f, 0, 11.1f,  // score=  0.5
                            1, 101,   0, 100});  // score=0.3

  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV3GPUOp Tests
// Copied from CPU tests

class NonMaxSuppressionV3GPUOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc mht_1(mht_1_v, 390, "", "./tensorflow/core/kernels/image/non_max_suppression_op_gpu_test.cc", "MakeOp");

    // TODO(laigd): enable the op once b/140816449 is fixed.
    // SetDevice(DEVICE_GPU,
    //           std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
    //               "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV3")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersWithScoreThresholdZeroScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.1, 0, 0, .3, .2, -5.0});
  // If we ask for more boxes than we actually expect to get back;
  // should still only get 2 boxes back.
  AddInputFromArray<int>(TensorShape({}), {6});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {-3.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});

  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
                            0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV4GPUOp Tests
//

class NonMaxSuppressionV4GPUOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_op_gpu_testDTcc mht_2(mht_2_v, 610, "", "./tensorflow/core/kernels/image/non_max_suppression_op_gpu_test.cc", "MakeOp");

    // TODO(b/143610288): reenable this once the bug is fixed.
    // SetDevice(DEVICE_GPU,
    //           std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
    //               "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV4")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("pad_to_max_output_size", true)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV4GPUOpTest, TestSelectFromThreeClustersPadFive) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {5});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  const auto expected_indices = test::AsTensor<int>({3, 0, 5, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(0));
  Tensor expected_num_valid = test::AsScalar<int>(3);
  test::ExpectTensorEqual<int>(expected_num_valid, *GetOutput(1));
}

TEST_F(NonMaxSuppressionV4GPUOpTest,
       TestSelectFromThreeClustersPadFiveScoreThr) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {6});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  const auto expected_indices = test::AsTensor<int>({3, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(0));
  Tensor expected_num_valid = test::AsScalar<int>(2);
  test::ExpectTensorEqual<int>(expected_num_valid, *GetOutput(1));
}

#endif

}  // namespace tensorflow
