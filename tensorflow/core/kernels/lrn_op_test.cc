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
class MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc() {
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

#include <functional>
#include <memory>

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
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static const float tol_ = 1e-4;

class LRNFloatTest : public OpsTestBase {
 protected:
  LRNFloatTest() : philox_(123, 17), rand_(&philox_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/lrn_op_test.cc", "LRNFloatTest");
}

  int GetIntAttr(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/lrn_op_test.cc", "GetIntAttr");

    int value;
    TF_CHECK_OK(GetNodeAttr(*node_def(), name, &value));
    return value;
  }

  float GetFloatAttr(const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/lrn_op_test.cc", "GetFloatAttr");

    float value;
    TF_CHECK_OK(GetNodeAttr(*node_def(), name, &value));
    return value;
  }

  bool Compare() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/kernels/lrn_op_test.cc", "Compare");

    const auto& input = GetInput(0);
    const int64_t batch_size = input.dim_size(0);
    const int64_t rows = input.dim_size(1);
    const int64_t cols = input.dim_size(2);
    const int64_t depth = input.dim_size(3);
    const int64_t rest = cols * rows * batch_size;

    const int64_t depth_radius = GetIntAttr("depth_radius");
    const float bias = GetFloatAttr("bias");
    const float alpha = GetFloatAttr("alpha");
    const float beta = GetFloatAttr("beta");

    Eigen::Tensor<float, 4, Eigen::RowMajor> expected(batch_size, rows, cols,
                                                      depth);
    auto out = expected.reshape(Eigen::DSizes<Eigen::Index, 2>{rest, depth});
    auto in = input.shaped<float, 2>({rest, depth});

    for (int64_t i = 0; i < rest; ++i) {
      Eigen::Tensor<float, 1, Eigen::RowMajor> out_col(depth);
      for (int64_t d = 0; d < depth; ++d) {
        float denom = 0.0f;
        for (int64_t r = std::max(int64_t{0}, d - depth_radius);
             r < std::min(depth, d + depth_radius + 1); ++r) {
          denom += in(i, r) * in(i, r);
        }
        denom = std::pow(denom * alpha + bias, beta);
        out_col(d) = in(i, d) / denom;
      }
      out.chip<0>(i) = out_col;
    }
    auto actual = GetOutput(0)->tensor<float, 4>();
    Eigen::Tensor<float, 0, Eigen::RowMajor> sum =
        ((expected - actual).abs() > actual.constant(tol_))
            .select(actual.constant(1), actual.constant(0))
            .sum();
    return sum() == 0;
  }

  random::PhiloxRandom philox_;
  random::SimplePhilox rand_;
};

TEST_F(LRNFloatTest, Depth96) {
  TF_ASSERT_OK(NodeDefBuilder("lrn_op", "LRN")
                   .Input(FakeInput())
                   .Attr("depth_radius", 5)
                   .Attr("bias", 1.0f)
                   .Attr("alpha", 0.1f)
                   .Attr("beta", 2.0f)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInput<float>(TensorShape({1, 1, 1, 96}),
                  [](int i) -> float { return i + 1; });
  TF_ASSERT_OK(RunOpKernel());
  auto actual = GetOutput(0)->tensor<float, 4>();

  // Output for Node 0 with Value 1:
  // 1 / (1 + 0.1*(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2))^2
  EXPECT_NEAR(1. / (10.1 * 10.1), actual(0, 0, 0, 0), tol_);

  // Output for Node 5 with Value 6:
  // 6 / (1 + 0.1*(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 ... + 11^2))^2
  EXPECT_NEAR(6. / (51.6 * 51.6), actual(0, 0, 0, 5), tol_);

  // Output for Node 63 with value 64:
  // 64 / (1 + 0.1*(59^2 + 60^2 + 61^2 + 62^2 + 63^2 + 64^2))^2
  EXPECT_NEAR(64. / (2272.1 * 2272.1), actual(0, 0, 0, 63), tol_);

  // Output for Node 64 with value 65:
  // 65 / (1 + 0.1*(65^2 + 66^2 + 67^2 + 68^2 + 69^2 + 70^2))^2
  EXPECT_NEAR(65. / (2736.5 * 2736.5), actual(0, 0, 0, 64), tol_);

  // Output for Node 95 with value 96:
  // 96 / (1 + 0.1*(91^2 + 92^2 + 93^2 + 94^2 + 95^2 + 96^2))^2
  EXPECT_NEAR(96. / (5248.1 * 5248.1), actual(0, 0, 0, 95), tol_);
  EXPECT_TRUE(Compare());
}

TEST_F(LRNFloatTest, Depth16) {
  TF_ASSERT_OK(NodeDefBuilder("lrn_op", "LRN")
                   .Input(FakeInput())
                   .Attr("depth_radius", 5)
                   .Attr("bias", 1.0f)
                   .Attr("alpha", 0.1f)
                   .Attr("beta", 2.0f)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInput<float>(TensorShape({1, 1, 1, 16}),
                  [](int i) -> float { return i + 1; });
  TF_ASSERT_OK(RunOpKernel());
  auto actual = GetOutput(0)->tensor<float, 4>();

  // Output for Node 0 with Value 1:
  // 1 / (1 + 0.1*(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2))^2
  EXPECT_NEAR(1. / (10.1 * 10.1), actual(0, 0, 0, 0), tol_);

  // Output for Node 5 with Value 6:
  // 6 / (1 + 0.1*(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 ... + 11^2))^2
  EXPECT_NEAR(6. / (51.6 * 51.6), actual(0, 0, 0, 5), tol_);

  // Output for Node 15 with value 16:
  // 16 / (1 + 0.1*(11^2 + 12^2 + 13^2 + 14^2 + 15^2 + 16^2))^2
  EXPECT_NEAR(16. / (112.1 * 112.1), actual(0, 0, 0, 15), tol_);
  EXPECT_TRUE(Compare());
}

static double RndGaussian(random::SimplePhilox* rnd) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_op_testDTcc mht_4(mht_4_v, 345, "", "./tensorflow/core/kernels/lrn_op_test.cc", "RndGaussian");

  // Box-Muller transformation.
  // See, for example, http://www.taygeta.com/random/gaussian.html
  double x1, x2;
  double r;
  do {
    x1 = 2 * rnd->RandDouble() - 1;
    x2 = 2 * rnd->RandDouble() - 1;
    r = x1 * x1 + x2 * x2;
  } while (r == 0 || r >= 1.0);
  double w = sqrt(-2.0 * log(r) / r);
  return x1 * w;
}

#define TCASE(NAME, DEPTH, BATCH, DEPTH_RADIUS, BIAS, ALPHA, BETA)           \
  TEST_F(LRNFloatTest, NAME) {                                               \
    TF_ASSERT_OK(NodeDefBuilder("lrn_op", "LRN")                             \
                     .Input(FakeInput())                                     \
                     .Attr("depth_radius", (DEPTH_RADIUS))                   \
                     .Attr("bias", (BIAS))                                   \
                     .Attr("alpha", ((ALPHA) / 10))                          \
                     .Attr("beta", (BETA))                                   \
                     .Finalize(node_def()));                                 \
    TF_ASSERT_OK(InitOp());                                                  \
    AddInput<float>(TensorShape({BATCH, 1, 1, DEPTH}),                       \
                    [this](int i) -> float { return RndGaussian(&rand_); }); \
    TF_ASSERT_OK(RunOpKernel());                                             \
    EXPECT_TRUE(Compare());                                                  \
  }

// clang-format off
//        DEPTH  BATCH  DEPTH_RADIUS  BIAS  ALPHA  BETA
TCASE(T0, 4,     2,     2,            1.0f, 1.0f,  2.0f)
TCASE(T1, 16,    1,     5,            1.0f, 1.0f,  2.0f)
TCASE(T2, 16,    32,    2,            1.0f, 2.0f,  1.0f)
TCASE(T3, 128,   4,     3,            2.0f, 1.0f,  1.0f)
// clang-format on

#undef TCASE

static Graph* MakeRNGrad(int batches, int rows, int cols, int depth,
                         int depth_radius) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor grads(DT_FLOAT, TensorShape({batches, rows, cols, depth}));
  grads.flat<float>().setRandom();

  Tensor in(DT_FLOAT, TensorShape({batches, rows, cols, depth}));
  in.flat<float>().setRandom();

  Tensor out(DT_FLOAT, TensorShape({batches, rows, cols, depth}));

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("lrn_grad_op"), "LRNGrad")
                  .Input(test::graph::Constant(g, grads))
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, out))
                  .Attr("depth_radius", depth_radius)
                  .Attr("bias", 1.0f)
                  .Attr("alpha", 1.0f / 10)
                  .Attr("beta", 2.0f)
                  .Finalize(g, &ret));
  return g;
}

#define BM_LRNGradDev(DEVICE, B, R, C, D, DR)                                  \
  static void BM_LRNGrad_##DEVICE##_##B##_##R##_##C##_##D##_##DR(              \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, MakeRNGrad(B, R, C, D, DR),                       \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * B * R * \
                            C * D * DR * 4);                                   \
  }                                                                            \
  BENCHMARK(BM_LRNGrad_##DEVICE##_##B##_##R##_##C##_##D##_##DR)

BM_LRNGradDev(cpu, 128, 12, 12, 64, 4);
BM_LRNGradDev(cpu, 128, 56, 56, 64, 2);
BM_LRNGradDev(cpu, 128, 27, 27, 192, 2);

}  // namespace tensorflow
