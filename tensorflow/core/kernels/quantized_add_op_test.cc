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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc() {
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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace ops {
namespace {

void TestAdd(const std::vector<int64_t>& x_shape,
             const std::vector<float>& x_values, float x_min_value,
             float x_max_value, const std::vector<int64_t>& y_shape,
             const std::vector<float>& y_values, float y_min_value,
             float y_max_value, const std::vector<int64_t>& expected_shape,
             const std::vector<float>& expected_values, double tolerance) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestAdd");

  Scope root = Scope::NewRootScope();

  Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
  test::FillValues<float>(&x_float_tensor, x_values);
  Tensor x_quantized_tensor(DT_QUINT8, x_float_tensor.shape());
  FloatTensorToQuantizedInPlace<quint8>(x_float_tensor, x_min_value,
                                        x_max_value, &x_quantized_tensor);
  Output x =
      Const(root.WithOpName("x"), Input::Initializer(x_quantized_tensor));
  Output x_min = Const(root.WithOpName("x_min"), x_min_value);
  Output x_max = Const(root.WithOpName("x_max"), x_max_value);

  Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
  test::FillValues<float>(&y_float_tensor, y_values);
  Tensor y_quantized_tensor(DT_QUINT8, y_float_tensor.shape());
  FloatTensorToQuantizedInPlace<quint8>(y_float_tensor, y_min_value,
                                        y_max_value, &y_quantized_tensor);
  Output y =
      Const(root.WithOpName("y"), Input::Initializer(y_quantized_tensor));
  Output y_min = Const(root.WithOpName("y_min"), y_min_value);
  Output y_max = Const(root.WithOpName("y_max"), y_max_value);

  ops::QuantizedAdd add = ops::QuantizedAdd(root.WithOpName("add"), x, y, x_min,
                                            x_max, y_min, y_max);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {add.z, add.min_z, add.max_z}, &outputs));

  const Tensor& z_quantized = outputs[0];
  const float z_min = outputs[1].flat<float>()(0);
  const float z_max = outputs[2].flat<float>()(0);

  Tensor z_float = QuantizedTensorToFloat<qint32>(z_quantized, z_min, z_max);
  Tensor expected_z_float(DT_FLOAT, TensorShape(expected_shape));
  test::FillValues<float>(&expected_z_float, expected_values);
  test::ExpectTensorNear<float>(expected_z_float, z_float, tolerance);
}

void TestAddShape(const std::vector<int64_t>& x_shape,
                  const std::vector<int64_t>& y_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestAddShape");

  const size_t x_num_elements = TensorShape(x_shape).num_elements();
  std::vector<float> x_values(x_num_elements);
  for (int i = 0; i < x_num_elements; ++i) {
    x_values[i] = i % 256;
  }
  const float x_min_value = 0.0f;
  const float x_max_value = 256.0f;

  const size_t y_num_elements = TensorShape(y_shape).num_elements();
  std::vector<float> y_values(y_num_elements);
  for (int i = 0; i < y_num_elements; ++i) {
    y_values[i] = ((i + 23) % 123) - 50;
  }
  const float y_min_value = -150.0f;
  const float y_max_value = 150.0f;

  Scope root = Scope::NewRootScope();

  Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
  test::FillValues<float>(&x_float_tensor, x_values);
  Output x = Const(root.WithOpName("x"), Input::Initializer(x_float_tensor));

  Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
  test::FillValues<float>(&y_float_tensor, y_values);
  Output y = Const(root.WithOpName("y"), Input::Initializer(y_float_tensor));

  Add add = Add(root.WithOpName("add"), x, y);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run(ClientSession::FeedType(), {add.z}, &outputs));

  const Tensor& expected_values_tensor = outputs[0];
  const float* expected_values_data =
      expected_values_tensor.flat<float>().data();
  std::vector<float> expected_values(
      expected_values_data,
      expected_values_data + expected_values_tensor.NumElements());
  std::vector<int64_t> expected_shape;
  for (const int64_t dim : expected_values_tensor.shape().dim_sizes()) {
    expected_shape.push_back(dim);
  }
  TestAdd(x_shape, x_values, x_min_value, x_max_value, y_shape, y_values,
          y_min_value, y_max_value, expected_shape, expected_values, 256.0);
}

void TimeAdd(const std::vector<int64_t>& x_shape,
             const std::vector<int64_t>& y_shape, int64_t iterations) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_2(mht_2_v, 313, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TimeAdd");

  TestAddShape(x_shape, y_shape);

  Scope root = Scope::NewRootScope();

  Tensor x_quantized_tensor(DT_QUINT8, TensorShape(x_shape));
  Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_QUINT8);
  Output x_min = Const(root.WithOpName("x_min"), 0.0f);
  Output x_max = Const(root.WithOpName("x_max"), 1.0f);

  Tensor y_quantized_tensor(DT_QUINT8, TensorShape(y_shape));
  Output y =
      Const(root.WithOpName("y"), Input::Initializer(y_quantized_tensor));
  Output y_min = Const(root.WithOpName("y_min"), 0.0f);
  Output y_max = Const(root.WithOpName("y_max"), 1.0f);

  ops::QuantizedAdd add = ops::QuantizedAdd(root.WithOpName("add"), placeholder,
                                            y, x_min, x_max, y_min, y_max);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  int64_t total_duration = 0;
  for (int i = 0; i < iterations; ++i) {
    const int64_t start_time = Env::Default()->NowMicros();
    TF_EXPECT_OK(session.Run({{placeholder, x_quantized_tensor}},
                             {add.z, add.min_z, add.max_z}, &outputs));
    const int64_t end_time = Env::Default()->NowMicros();
    total_duration += end_time - start_time;
  }
  const int64_t one_run_duration = total_duration / iterations;

  const int64_t num_ops = outputs[0].NumElements();

  const double million_ops_per_second =
      (iterations * num_ops) / static_cast<double>(total_duration);

  LOG(INFO) << "TimeAdd: " << TensorShape(x_shape).DebugString() << " * "
            << TensorShape(y_shape).DebugString()
            << ": iterations=" << iterations
            << ", MOps/s=" << million_ops_per_second
            << ", one_run_duration=" << one_run_duration
            << ", total_duration=" << total_duration;
}

void TestManualScalar() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_3(mht_3_v, 363, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestManualScalar");

  TestAdd(
      {10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {1}, {10.0f}, -100.0f, 100.0f, {10},
      {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      1.0f);
  TestAdd(
      {1}, {10.0f}, -100.0f, 100.0f, {10},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {10},
      {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      1.0f);
}

void TestScalar() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_4(mht_4_v, 380, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestScalar");

  TestAddShape({100}, {1});
  TestAddShape({1}, {100});
}

void TestManualVector() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_5(mht_5_v, 388, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestManualVector");

  TestAdd({10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
          0.0f, 10.0f, {10},
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
          10.0f, {10},
          {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f},
          1.0f);
}

void TestVector() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_6(mht_6_v, 400, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestVector");
 TestAddShape({100}, {100}); }

void TestManualVectorPlusTensor() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_7(mht_7_v, 405, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestManualVectorPlusTensor");

  TestAdd(
      {10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {2, 10},
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      0.0f, 20.0f, {2, 10},
      {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f,
       12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
      1.0f);
  TestAdd({2, 10}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                    8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
          0.0f, 20.0f, {10},
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
          10.0f, {2, 10}, {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f,
                           16.0f, 18.0f, 20.0f, 12.0f, 14.0f, 16.0f, 18.0f,
                           20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
          1.0f);
  TestAdd(
      {5, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
      0.0f, 10.0f, {2, 5, 2},
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      0.0f, 20.0f, {2, 5, 2},
      {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f,
       12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
      1.0f);
}

void TestVectorPlusTensor() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_8(mht_8_v, 438, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "TestVectorPlusTensor");

  TestAddShape({100}, {2, 100});
  TestAddShape({2, 100}, {100});
  TestAddShape({5, 2}, {2, 5, 2});
}

void BenchmarkTensorScalar() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_9(mht_9_v, 447, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "BenchmarkTensorScalar");

  TimeAdd({200}, {1}, 1000);
  TimeAdd({10000}, {1}, 100);
  TimeAdd({1000000}, {1}, 10);
  TimeAdd({10000000}, {1}, 1);
}

void BenchmarkVector() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_10(mht_10_v, 457, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "BenchmarkVector");

  TimeAdd({200}, {200}, 1000);
  TimeAdd({10000}, {10000}, 100);
  TimeAdd({1000000}, {1000000}, 10);
  TimeAdd({10000000}, {10000000}, 1);
}

void BenchmarkVectorPlusTensor() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_add_op_testDTcc mht_11(mht_11_v, 467, "", "./tensorflow/core/kernels/quantized_add_op_test.cc", "BenchmarkVectorPlusTensor");

  TimeAdd({10, 20}, {20}, 100);
  TimeAdd({10, 1000}, {1000}, 10);
  TimeAdd({1000, 1000}, {1000}, 1);
  TimeAdd({10000, 1000}, {1000}, 1);
  TimeAdd({100, 100}, {100}, 10);
  TimeAdd({10000, 100}, {100}, 1);
  TimeAdd({100000, 100}, {100}, 1);
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow

#define RUN_TEST(t) \
  TEST(QuantizedAddOpTest, t) { tensorflow::ops::t(); }

RUN_TEST(TestManualScalar);
RUN_TEST(TestManualVector);
RUN_TEST(TestManualVectorPlusTensor);
RUN_TEST(TestScalar);
RUN_TEST(TestVector);
RUN_TEST(TestVectorPlusTensor);

#if defined(__ANDROID__)

RUN_TEST(BenchmarkTensorScalar);
RUN_TEST(BenchmarkVector);
RUN_TEST(BenchmarkVectorPlusTensor);

#endif  // __ANDROID__

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
