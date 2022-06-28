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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ResizeBicubicOpTest : public OpsTestBase {
 protected:
  ResizeBicubicOpTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "ResizeBicubicOpTest");

    TF_EXPECT_OK(NodeDefBuilder("resize_bicubic_op", "ResizeBicubic")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", false)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  const Tensor* SetRandomImageInput(const TensorShape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "SetRandomImageInput");

    inputs_.clear();

    CHECK_EQ(shape.dims(), 4) << "All images must have 4 dimensions.";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DataTypeToEnum<float>::v(), shape);
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

 private:
  static constexpr int64_t kTableSize = (1 << 10);

  const float* InitCoeffsTable() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "InitCoeffsTable");

    // Allocate and initialize coefficients table using Bicubic
    // convolution algorithm.
    // https://en.wikipedia.org/wiki/Bicubic_interpolation
    float* coeffs_tab = new float[(kTableSize + 1) * 2];
    static const double A = -0.75;
    for (int i = 0; i <= kTableSize; ++i) {
      float x = i * 1.0 / kTableSize;
      coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
      x += 1.0;
      coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    }
    return coeffs_tab;
  }

  const float* GetCoeffsTable() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "GetCoeffsTable");

    // Static so that we initialize it on first use
    static const float* coeffs_tab = InitCoeffsTable();
    return coeffs_tab;
  }

  // Used in the baseline implementation
  inline int64_t Bound(int64_t val, int64_t limit) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "Bound");

    return std::min(limit - 1, std::max(int64_t{0}, val));
  }

  // Used in the baseline implementation
  inline void GetWeightsAndIndices(float scale, int64_t out_loc, int64_t limit,
                                   std::array<float, 4>* weights,
                                   std::array<int64_t, 4>* indices) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "GetWeightsAndIndices");

    const int64_t in_loc = scale * out_loc;
    const float delta = scale * out_loc - in_loc;
    const int64_t offset = lrintf(delta * kTableSize);
    const float* coeffs_tab = GetCoeffsTable();
    *weights = {{coeffs_tab[offset * 2 + 1], coeffs_tab[offset * 2],
                 coeffs_tab[(kTableSize - offset) * 2],
                 coeffs_tab[(kTableSize - offset) * 2 + 1]}};
    *indices = {{Bound(in_loc - 1, limit), Bound(in_loc, limit),
                 Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)}};
  }

  // Used in the baseline implementation
  inline float Interpolate1D(const std::array<float, 4>& weights,
                             const std::array<float, 4>& values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "Interpolate1D");

    return values[0] * weights[0] + values[1] * weights[1] +
           values[2] * weights[2] + values[3] * weights[3];
  }

  // This is the straight forward unoptimized implementation of resize bicubic
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeBicubicBaseline(TTypes<float, 4>::ConstTensor images,
                             TTypes<float, 4>::Tensor output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "ResizeBicubicBaseline");

    const int batch_size = images.dimension(0);
    const int64_t in_height = images.dimension(1);
    const int64_t in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch_size, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64_t out_height = output.dimension(1);
    const int64_t out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    std::array<float, 4> coeff = {{0.0, 0.0, 0.0, 0.0}};
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        std::array<float, 4> y_weights;
        std::array<int64_t, 4> y_indices;
        GetWeightsAndIndices(height_scale, y, in_height, &y_weights,
                             &y_indices);
        for (int64_t x = 0; x < out_width; ++x) {
          std::array<float, 4> x_weights;
          std::array<int64_t, 4> x_indices;
          GetWeightsAndIndices(width_scale, x, in_width, &x_weights,
                               &x_indices);
          for (int64_t c = 0; c < channels; ++c) {
            // Use a 4x4 patch to compute the interpolated output value at
            // (b, y, x, c).
            for (int64_t i = 0; i < 4; ++i) {
              const std::array<float, 4> values = {
                  {static_cast<float>(images(b, y_indices[i], x_indices[0], c)),
                   static_cast<float>(images(b, y_indices[i], x_indices[1], c)),
                   static_cast<float>(images(b, y_indices[i], x_indices[2], c)),
                   static_cast<float>(
                       images(b, y_indices[i], x_indices[3], c))}};
              coeff[i] = Interpolate1D(x_weights, values);
            }
            output(b, y, x, c) = Interpolate1D(y_weights, coeff);
          }
        }
      }
    }
  }

 protected:
  void RunRandomTest(const int batch_size, const int64_t in_height,
                     const int64_t in_width, const int target_height,
                     const int target_width, int channels) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_8(mht_8_v, 357, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "RunRandomTest");

    LOG(INFO) << "Running random test " << in_height << "x" << in_width << "x"
              << channels << " to " << target_height << "x" << target_width
              << "x" << channels;
    const Tensor* input = SetRandomImageInput(
        TensorShape({batch_size, in_height, in_width, channels}));
    AddInputFromArray<int32>(TensorShape({2}), {target_height, target_width});

    TF_ASSERT_OK(RunOpKernel());

    std::unique_ptr<Tensor> expected(new Tensor(
        device_->GetAllocator(AllocatorAttributes()),
        DataTypeToEnum<float>::v(),
        TensorShape({batch_size, target_height, target_width, channels})));

    ResizeBicubicBaseline(input->tensor<float, 4>(),
                          expected->tensor<float, 4>());
    // Note: the baseline implementation reduces first in the x direction, and
    // then in the y direction. The optimized version reduces first in the y
    // direction, and then the X direction. As a result, there may be
    // some slight floating point inaccuracies. We thus ensure we're within
    // 0.00001 of the previous implementation.
    test::ExpectTensorNear<float>(*expected, *GetOutput(0), 0.00001);
  }

  void RunManyRandomTests(int channels) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_9(mht_9_v, 385, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "RunManyRandomTests");

    for (int batch_size : {1, 2, 5}) {
      for (int in_w : {2, 4, 7, 20, 165}) {
        for (int in_h : {1, 3, 5, 8, 100, 233}) {
          for (int target_height : {1, 2, 3, 50, 113}) {
            for (int target_width : {target_height, target_height / 2 + 1}) {
              RunRandomTest(batch_size, in_h, in_w, target_height, target_width,
                            channels);
            }
          }
        }
      }
    }
  }
};

TEST_F(ResizeBicubicOpTest, TestBicubic2x2To1x1) {
  // Input:
  // 1, 2
  // 3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBicubicOpTest, TestBicubic2x2To0x0) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});

  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "output dimensions must be positive"))
      << s;
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom141x186) {
  RunRandomTest(2, 141, 186, 299, 299, 1 /* channels */);
  RunRandomTest(2, 141, 186, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom183x229) {
  RunRandomTest(2, 183, 229, 299, 299, 1 /* channels */);
  RunRandomTest(2, 183, 229, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom749x603) {
  RunRandomTest(2, 749, 603, 299, 299, 1 /* channels */);
  RunRandomTest(2, 749, 603, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes1Channel) {
  RunManyRandomTests(1);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes3Channels) {
  RunManyRandomTests(3);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes4Channels) {
  RunManyRandomTests(4);
}

static Graph* ResizeBicubic(int batch_size, int size, int channels,
                            float scale_y = 0.3, float scale_x = 0.7) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSresize_bicubic_op_testDTcc mht_10(mht_10_v, 458, "", "./tensorflow/core/kernels/image/resize_bicubic_op_test.cc", "ResizeBicubic");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, TensorShape({batch_size, size, size, channels}));
  input.flat<float>().setRandom();
  Tensor shape(DT_INT32, TensorShape({2}));
  auto shape_t = shape.flat<int32>();
  shape_t(0) = scale_y * size;
  shape_t(1) = scale_x * size;
  test::graph::Binary(g, "ResizeBicubic", test::graph::Constant(g, input),
                      test::graph::Constant(g, shape));
  return g;
}

#define BM_ResizeBicubicDev(BATCH, SIZE, CHANNELS)                             \
  static void BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS(                \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark("cpu", ResizeBicubic(BATCH, SIZE, CHANNELS),               \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * BATCH * \
                            SIZE * SIZE * CHANNELS);                           \
  }                                                                            \
  BENCHMARK(BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS);

BM_ResizeBicubicDev(8, 32, 3);
BM_ResizeBicubicDev(8, 128, 3);
BM_ResizeBicubicDev(8, 512, 3);
BM_ResizeBicubicDev(8, 1024, 3);
BM_ResizeBicubicDev(16, 32, 3);
BM_ResizeBicubicDev(16, 128, 3);
BM_ResizeBicubicDev(16, 512, 3);
BM_ResizeBicubicDev(16, 1024, 3);
BM_ResizeBicubicDev(32, 32, 3);
BM_ResizeBicubicDev(32, 128, 3);
BM_ResizeBicubicDev(32, 512, 3);
BM_ResizeBicubicDev(32, 1024, 3);

#define BM_ResizeBicubicExpand(BATCH, SIZE, CHANNELS)                          \
  static void BM_ResizeBicubicExpand##_##BATCH##_##SIZE##_##CHANNELS(          \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark("cpu", ResizeBicubic(BATCH, SIZE, CHANNELS, 8, 8),         \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * BATCH * \
                            SIZE * SIZE * CHANNELS * 8 * 8);                   \
  }                                                                            \
  BENCHMARK(BM_ResizeBicubicExpand##_##BATCH##_##SIZE##_##CHANNELS);

BM_ResizeBicubicExpand(12, 48, 1);
BM_ResizeBicubicExpand(12, 48, 3);
BM_ResizeBicubicExpand(12, 48, 40);

}  // end namespace tensorflow
