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
class MHTracer_DTPStensorflowPScorePSkernelsPSfused_batch_norm_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_batch_norm_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfused_batch_norm_op_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
class FusedBatchNormOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormOpTest, Training) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 1.0)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_mean, {10, 10});
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_variance, {14.00, 14.00});
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 0.01);
}

TEST_F(FusedBatchNormOpTest, TrainingRunningMean) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 0.5)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {6.0, 6.0});
  AddInputFromArray<float>(TensorShape({2}), {16.0, 16.0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_mean, {8, 8});
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_variance, {15.00, 15.00});
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 0.01);
}

TEST_F(FusedBatchNormOpTest, Inference) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {10, 10});
  AddInputFromArray<float>(TensorShape({2}), {11.67f, 11.67f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(FusedBatchNormOpTest, InferenceIgnoreAvgFactor) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 0.5)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {10, 10});
  AddInputFromArray<float>(TensorShape({2}), {11.67f, 11.67f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(FusedBatchNormOpTest, EmptyInput) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 0, 0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({1, 1, 0, 0}));
}

class FusedBatchNormGradOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormGradOpTest, Simple) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_grad_op", "FusedBatchNormGrad")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {2, 2, 9, 9, -4, -4, 5, 5, 8, 8, 7, 7});
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {1, 1, 7, 7, 4, 4, -3, -3, -11, -11, 13, 13});
  AddInputFromArray<float>(TensorShape({2}), {4, 4});
  AddInputFromArray<float>(TensorShape({2}), {1.833f, 1.833f});
  AddInputFromArray<float>(TensorShape({2}), {57.472f, 57.472f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_x(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected_x, {-1.34, -1.34, 2.47, 2.47, -4.44, -4.44,
                                        0.17, 0.17, 1.60, 1.60, 1.53, 1.53});
  test::ExpectTensorNear<float>(expected_x, *GetOutput(0), 0.01);

  Tensor expected_scale(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_scale, {-1.6488, -1.6488});
  test::ExpectTensorNear<float>(expected_scale, *GetOutput(1), 0.01);

  Tensor expected_offset(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_offset, {27, 27});
  test::ExpectTensorNear<float>(expected_offset, *GetOutput(2), 0.01);
}

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

using fp32 = float;
using fp16 = Eigen::half;

template <typename T>
static Graph* FusedBatchNormInference(int n, int h, int w, int c,
                                      bool is_training,
                                      TensorFormat data_format) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_batch_norm_op_testDTcc mht_0(mht_0_v, 395, "", "./tensorflow/core/kernels/fused_batch_norm_op_test.cc", "FusedBatchNormInference");

  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  Tensor x_t(dtype, data_format == FORMAT_NHWC ? TensorShape({n, h, w, c})
                                               : TensorShape({n, c, h, w}));
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({c}));
  other_t.flat<float>().setRandom();

  Tensor empty_t(DT_FLOAT, TensorShape({0}));

  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");
  Node* empty = test::graph::Constant(g, empty_t, "empty");

  Node* fused_batch_norm;
  TF_CHECK_OK(NodeBuilder(g->NewName("fused_batch_norm"), "FusedBatchNormV3")
                  .Input(x)
                  .Input(other)                        // scale
                  .Input(other)                        // offset
                  .Input(is_training ? empty : other)  // mean
                  .Input(is_training ? empty : other)  // variance
                  .Attr("T", dtype)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("is_training", is_training)
                  .Attr("data_format", ToString(data_format))
                  .Finalize(g, &fused_batch_norm));

  return g;
}

template <typename T>
static Graph* FusedBatchNormGrad(int n, int h, int w, int c, bool is_training,
                                 TensorFormat data_format) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_batch_norm_op_testDTcc mht_1(mht_1_v, 434, "", "./tensorflow/core/kernels/fused_batch_norm_op_test.cc", "FusedBatchNormGrad");

  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  TensorShape shape = data_format == FORMAT_NHWC ? TensorShape({n, h, w, c})
                                                 : TensorShape({n, c, h, w});

  Tensor y_backprop_t(dtype, shape);
  y_backprop_t.flat<T>().setRandom();

  Tensor x_t(dtype, shape);
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({c}));
  other_t.flat<float>().setRandom();

  Node* y_backprop = test::graph::Constant(g, y_backprop_t, "y_backprop");
  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");

  Node* fused_batch_norm;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("fused_batch_norm_grad"), "FusedBatchNormGradV3")
          .Input(y_backprop)
          .Input(x)
          .Input(other)  // scale
          .Input(other)  // saved_mean_or_pop_mean
          .Input(other)  // saved_maybe_inv_var_or_pop_var
          .Input(other)  // reserve_space
          .Attr("T", dtype)
          .Attr("U", DT_FLOAT)
          .Attr("epsilon", 0.001)
          .Attr("is_training", is_training)
          .Attr("data_format", ToString(data_format))
          .Finalize(g, &fused_batch_norm));

  return g;
}

#define BM_NAME(NAME, N, H, W, C, T, IT, FORMAT, DEVICE) \
  BM_##NAME##_##N##_##H##_##W##_##C##_##IT##_##FORMAT##_##T##_##DEVICE

// -------------------------------------------------------------------------- //
// FusedBatchNorm inference
// -------------------------------------------------------------------------- //
// clang-format off
// NOLINTBEGIN
#define BM_FusedBatchNorm(N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)         \
  static void BM_NAME(FusedBatchNorm, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)(::testing::benchmark::State & state) {                     \
    test::Benchmark(                                                          \
        #DEVICE,                                                              \
        FusedBatchNormInference<T>(N, H, W, C, IS_TRAINING, FORMAT_##FORMAT), \
        /*old_benchmark_api*/ false)                                          \
        .Run(state);                                                          \
    state.SetItemsProcessed(state.iterations() * N * H * W * C);              \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_NAME(FusedBatchNorm, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE))    \
      ->UseRealTime();

// NOLINTEND
// clang-format on

BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NHWC, cpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NHWC, cpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NHWC, cpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NHWC, cpu);

#ifdef GOOGLE_CUDA
BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NHWC, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NHWC, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NCHW, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NCHW, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NHWC, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NHWC, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NCHW, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NCHW, gpu);
#endif  // GOOGLE_CUDA

// -------------------------------------------------------------------------- //
// FusedBatchNorm gradient
// -------------------------------------------------------------------------- //

#define BM_FusedBatchNormGrad(N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)      \
  static void BM_NAME(FusedBatchNormGrad, N, H, W, C, T, IS_TRAINING, FORMAT,  \
                      DEVICE)(::testing::benchmark::State & state) {           \
    test::Benchmark(                                                           \
        #DEVICE,                                                               \
        FusedBatchNormGrad<T>(N, H, W, C, IS_TRAINING, FORMAT_##FORMAT),       \
        /*old_benchmark_api*/ false)                                           \
        .Run(state);                                                           \
    state.SetItemsProcessed(state.iterations() * N * H * W * C);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(FusedBatchNormGrad, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)) \
      ->UseRealTime();

#define BM_FusedBatchNormGradResnetShapes(T, IS_TRAINING, FORMAT, DEVICE) \
  BM_FusedBatchNormGrad(64, 56, 56, 64, T, IS_TRAINING, FORMAT, DEVICE);  \
  BM_FusedBatchNormGrad(64, 56, 56, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 56, 56, 256, T, IS_TRAINING, FORMAT, DEVICE); \
                                                                          \
  BM_FusedBatchNormGrad(64, 28, 28, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 28, 28, 256, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 28, 28, 512, T, IS_TRAINING, FORMAT, DEVICE); \
                                                                          \
  BM_FusedBatchNormGrad(64, 14, 14, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 14, 14, 256, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 14, 14, 1024, T, IS_TRAINING, FORMAT, DEVICE)

BM_FusedBatchNormGradResnetShapes(fp32, true, NHWC, cpu);
BM_FusedBatchNormGradResnetShapes(fp32, false, NHWC, cpu);

#ifdef GOOGLE_CUDA
BM_FusedBatchNormGradResnetShapes(fp32, true, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, true, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp32, true, NCHW, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, true, NCHW, gpu);

BM_FusedBatchNormGradResnetShapes(fp32, false, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, false, NHWC, gpu);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
