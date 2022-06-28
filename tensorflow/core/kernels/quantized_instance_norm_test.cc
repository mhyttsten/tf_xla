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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc() {
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

#define EIGEN_USE_THREADS

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace ops {
namespace {

void ReferenceImpl(const quint8* inp, float inp_min, float inp_max,
                   const TensorShape& shape, float var_eps, float* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "ReferenceImpl");

  int N = shape.dim_size(0);
  int H = shape.dim_size(1);
  int W = shape.dim_size(2);
  int C = shape.dim_size(3);

  int total = N * H * W * C;
  float inp_scale = (inp_max - inp_min) / 255.0f;
  std::unique_ptr<float[]> dequantized(new float[total]);

  for (int i = 0; i < total; ++i) {
    dequantized[i] = inp_min + inp_scale * static_cast<float>(inp[i]);
  }

  std::unique_ptr<float[]> inp_mean(new float[N * C]);
  std::unique_ptr<float[]> inp_var(new float[N * C]);

  float img_size = static_cast<float>(H) * static_cast<float>(W);

  // Compute mean
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      float sum = 0.0;
      for (int i = 0; i < H * W; ++i) {
        sum += dequantized[n * H * W * C + i * C + c];
      }
      inp_mean[n * C + c] = sum / img_size;
    }
  }

  // Compute var
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      float sum = 0.0;
      for (int i = 0; i < H * W; ++i) {
        float tmp =
            dequantized[n * H * W * C + i * C + c] - inp_mean[n * C + c];
        sum += tmp * tmp;
      }
      inp_var[n * C + c] = sum / img_size;
    }
  }

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H * W; ++i) {
        out[n * H * W * C + i * C + c] =
            (dequantized[n * H * W * C + i * C + c] - inp_mean[n * C + c]) /
            std::sqrt(inp_var[n * C + c] + var_eps);
      }
    }
  }
}

void Expect(const Tensor& input, float x_min, float x_max,
            bool output_range_given, float give_y_min, float given_y_max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_1(mht_1_v, 256, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "Expect");

  Scope root = Scope::NewRootScope();

  auto input_ph = Placeholder(root, DT_QUINT8);

  const float variance_eps = 1e-5;
  auto instance_norm = QuantizedInstanceNorm(
      root, input_ph, x_min, x_max,
      QuantizedInstanceNorm::Attrs().VarianceEpsilon(variance_eps));

  Status s = root.status();
  EXPECT_TRUE(s.ok());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  s = session.Run({{input_ph, input}},
                  {instance_norm.y, instance_norm.y_min, instance_norm.y_max},
                  &outputs);

  EXPECT_TRUE(s.ok());
  Tensor expected(DT_FLOAT, input.shape());

  ReferenceImpl(input.flat<quint8>().data(), x_min, x_max, input.shape(),
                variance_eps, expected.flat<float>().data());

  auto out = outputs[0].flat<quint8>();

  float out_min = outputs[1].flat<float>()(0);
  float out_max = outputs[2].flat<float>()(0);
  float out_scale = (out_max - out_min) / 255.0f;

  Eigen::Tensor<float, 0, Eigen::RowMajor> max_diff =
      (expected.flat<float>() - (out_min + out_scale * out.cast<float>()))
          .abs()
          .maximum();
  EXPECT_LE(max_diff(), 0.1);
  LOG(INFO) << "max diff " << max_diff();
}

void TestBasic() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "TestBasic");

  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  // Random input
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  Expect(input_tensor, 0.0f, 1.0f, false, 0.0f, 0.0f);
}

void TestZeroInput() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_3(mht_3_v, 311, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "TestZeroInput");

  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  // Zero input, but input min > 0. Tests that output min and max should be
  // properly separated.
  input = input.setConstant(0);

  Expect(input_tensor, 2.0f, 3.0f, false, 0.0f, 0.0f);
}

void TestMaxInput() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_4(mht_4_v, 324, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "TestMaxInput");

  Tensor input_tensor(DT_QUINT8, {1, 1, 2, 16});
  auto input = input_tensor.flat<quint8>();
  // Inputs are all FLT_MAX / (number of inputs).
  input = input.setConstant(255);

  Expect(input_tensor, 0.0f,
         std::numeric_limits<float>::max() / static_cast<float>(2 * 16), false,
         0.0f, 0.0f);
}

void TestOutputRangeGiven() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_5(mht_5_v, 338, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "TestOutputRangeGiven");

  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  Expect(input_tensor, -10.0f, 10.0f, true, -1.0f, 1.0f);
}

void TestClamp() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "TestClamp");

  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  // Tests that negative outputs are clamped at 0.0, as the output range is
  // given to be (0.0, 1.0).
  Expect(input_tensor, -10.0f, 10.0f, true, 0.0f, 1.0f);
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow

#define RUN_TEST(t) \
  TEST(QuantizedInstanceNormTest, t) { tensorflow::ops::t(); }

RUN_TEST(TestBasic);
RUN_TEST(TestZeroInput);
RUN_TEST(TestMaxInput);
RUN_TEST(TestOutputRangeGiven);
RUN_TEST(TestClamp);

int main(int argc, char** argv) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_instance_norm_testDTcc mht_7(mht_7_v, 375, "", "./tensorflow/core/kernels/quantized_instance_norm_test.cc", "main");

  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
