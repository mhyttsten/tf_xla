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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

using test::graph::Constant;

class QuantizedConcatTest : public OpsTestBase {
 protected:
  QuantizedConcatTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "QuantizedConcatTest");
}

  void TestSmall8Bit(float first_min, float first_max, float second_min,
                     float second_max);
  void TestSmall32Bit(float first_min, float first_max, float second_min,
                      float second_max);
  void TestSecondDim8Bit(float first_min, float first_max, float second_min,
                         float second_max);
};

TEST_F(QuantizedConcatTest, Small8Bit) {
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 25.0f);
}

TEST_F(QuantizedConcatTest, Small8BitSameRange) {
  // Range for both is the same, so impl can use memcpy.
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 255.0f);
}

void QuantizedConcatTest::TestSmall8Bit(float first_min, float first_max,
                                        float second_min, float second_max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "QuantizedConcatTest::TestSmall8Bit");

  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_batch = first_batch + second_batch;
  Tensor expected_float(DT_FLOAT, {expected_batch, first_height, first_width});
  test::FillValues<float>(&expected_float,
                          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedConcatTest, Small32Bit) {
  TestSmall32Bit(0.0f, 1200.0f, 0.0f, 2400.0f);
}

TEST_F(QuantizedConcatTest, Small32BitSameRange) {
  TestSmall32Bit(-2400.0f, 2400.0f, -2400.0f, 2400.0f);
}

TEST_F(QuantizedConcatTest, Small32BitOneDimSameRangeAsOutput) {
  TestSmall32Bit(-2400.0f, 2400.0f, -1200.0f, 2400.0f);
}

void QuantizedConcatTest::TestSmall32Bit(float first_min, float first_max,
                                         float second_min, float second_max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_2(mht_2_v, 306, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "QuantizedConcatTest::TestSmall32Bit");

  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QINT32))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float, {100, 200, 300, 400, 500, 600, 700, 800,
                                         900, 1000, 1100, 1200});
  Tensor first_quantized =
      FloatTensorToQuantized<qint32>(first_float, first_min, first_max);

  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float, {1300, 1400, 1500, 1600, 1700, 1800,
                                          1900, 2000, 2100, 2200, 2300, 2400});
  Tensor second_quantized =
      FloatTensorToQuantized<qint32>(second_float, second_min, second_max);

  const int expected_batch = first_batch + second_batch;
  Tensor expected_float(DT_FLOAT, {expected_batch, first_height, first_width});
  test::FillValues<float>(
      &expected_float,
      {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
       1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400});

  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<qint32>(first_quantized.shape(),
                            first_quantized.flat<qint32>());
  AddInputFromArray<qint32>(second_quantized.shape(),
                            second_quantized.flat<qint32>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedConcatTest, SecondDim8Bit) {
  TestSecondDim8Bit(-10.0f, 150.0f, 0.0f, 200.0f);
}

TEST_F(QuantizedConcatTest, SecondDim8BitSameRange) {
  TestSecondDim8Bit(-10.0f, 150.0f, -10.0f, 150.0f);
}

void QuantizedConcatTest::TestSecondDim8Bit(float first_min, float first_max,
                                            float second_min,
                                            float second_max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_3(mht_3_v, 372, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "QuantizedConcatTest::TestSecondDim8Bit");

  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_height = first_height + second_height;
  Tensor expected_float(DT_FLOAT, {first_batch, expected_height, first_width});
  test::FillValues<float>(&expected_float,
                          {1, 2, 3, 4,  5,  6,  13, 14, 15, 16, 17, 18,
                           7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension".
// If <same_limits> is true, then both concatenated dimensions have the same
// quantized range; otherwise, they are set to different values.
template <typename T>
static void ConcatHelper(::testing::benchmark::State& state,
                         int concat_dimension, bool same_limits, int dim2) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_4(mht_4_v, 433, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "ConcatHelper");

  Graph* g = new Graph(OpRegistry::Global());

  DataType dt = DataTypeToEnum<T>::v();
  const int kDim1 = 100;
  TensorShape shape({kDim1, dim2});

  Tensor concat_dim = test::AsScalar<int32>(concat_dimension);
  Tensor in0(dt, shape);
  in0.flat<T>().setRandom();
  Tensor in1(dt, shape);
  in1.flat<T>().setRandom();

  Tensor mins0 = test::AsScalar<float>(-1.0);
  Tensor maxes0 = test::AsScalar<float>(1.0);
  Tensor mins1 = test::AsScalar<float>(same_limits ? -1.0 : -255.0);
  Tensor maxes1 = test::AsScalar<float>(same_limits ? 1.0 : 255.0);

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "QuantizedConcat")
                  .Input(Constant(g, concat_dim))
                  .Input({Constant(g, in0), Constant(g, in1)})
                  .Input({Constant(g, mins0), Constant(g, mins1)})
                  .Input({Constant(g, maxes0), Constant(g, maxes1)})
                  .Attr("N", 2)
                  .Attr("T", dt)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          ((kDim1 * dim2) + (kDim1 * dim2)) * sizeof(T));
}

static void BM_QConcatDim0SameLimitQInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_5(mht_5_v, 469, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim0SameLimitQInt32");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1SameLimitQInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_6(mht_6_v, 479, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim1SameLimitQInt32");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim0DifferLimitQInt32(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_7(mht_7_v, 490, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim0DifferLimitQInt32");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1DifferLimitQInt32(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_8(mht_8_v, 501, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim1DifferLimitQInt32");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

BENCHMARK(BM_QConcatDim0SameLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1SameLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim0DifferLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1DifferLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);

static void BM_QConcatDim0SameLimitQUint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_9(mht_9_v, 532, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim0SameLimitQUint8");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1SameLimitQUint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_10(mht_10_v, 542, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim1SameLimitQUint8");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim0DifferLimitQUint8(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_11(mht_11_v, 553, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim0DifferLimitQUint8");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1DifferLimitQUint8(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_concat_op_testDTcc mht_12(mht_12_v, 564, "", "./tensorflow/core/kernels/quantized_concat_op_test.cc", "BM_QConcatDim1DifferLimitQUint8");

  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

BENCHMARK(BM_QConcatDim0SameLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1SameLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim0DifferLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1DifferLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);

}  // namespace tensorflow
