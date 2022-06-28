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
class MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class ReverseOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/reverse_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "Reverse")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput())
                     .Attr("T", data_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  template <typename T>
  void Reverse_0() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/kernels/reverse_op_test.cc", "Reverse_0");

    MakeOp(DataTypeToEnum<T>::value);
    AddInputFromArray<T>(TensorShape({}), {3});
    AddInputFromArray<bool>(TensorShape({}), {true});
    TF_ASSERT_OK(RunOpKernel());

    Tensor* output = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value, TensorShape({}));
    expected.scalar<T>() = expected.scalar<T>().constant(3);
    test::ExpectTensorEqual<T>(expected, *output);
  }

  template <typename T>
  void Reverse_234() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/reverse_op_test.cc", "Reverse_234");

    MakeOp(DataTypeToEnum<T>::value);
    // Feed and run
    // [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
    AddInputFromArray<T>(TensorShape({2, 3, 4}),
                         {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    AddInputFromArray<bool>(TensorShape({3}), {true, false, true});

    TF_ASSERT_OK(RunOpKernel());

    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 3, 4}));
    // Should become
    // [[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]
    test::FillValues<T>(&expected,
                        {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                         3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8});
    test::ExpectTensorEqual<T>(expected, *params_tensor);
  }

  template <typename T>
  void Reverse_1234() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/reverse_op_test.cc", "Reverse_1234");

    MakeOp(DataTypeToEnum<T>::value);
    // Feed and run
    // [[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //   [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]]
    AddInputFromArray<T>(TensorShape({1, 2, 3, 4}),
                         {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    AddInputFromArray<bool>(TensorShape({4}), {true, true, false, true});

    TF_ASSERT_OK(RunOpKernel());

    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({1, 2, 3, 4}));
    // Should become
    // [[[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //   [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]]
    test::FillValues<T>(&expected,
                        {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                         3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8});
    test::ExpectTensorEqual<T>(expected, *params_tensor);
  }
};

TEST_F(ReverseOpTest, Reverse_0_uint8) { Reverse_0<uint8>(); }

TEST_F(ReverseOpTest, Reverse_0_int8) { Reverse_0<int8>(); }

TEST_F(ReverseOpTest, Reverse_0_uint16) { Reverse_0<uint16>(); }

TEST_F(ReverseOpTest, Reverse_0_int16) { Reverse_0<int16>(); }

TEST_F(ReverseOpTest, Reverse_0_float) { Reverse_0<float>(); }

TEST_F(ReverseOpTest, Reverse_0_int32) { Reverse_0<int32>(); }

TEST_F(ReverseOpTest, Reverse_0_int64) { Reverse_0<int64_t>(); }

TEST_F(ReverseOpTest, Reverse_0_double) { Reverse_0<double>(); }

TEST_F(ReverseOpTest, Reverse_0_complex64) { Reverse_0<complex64>(); }

TEST_F(ReverseOpTest, Reverse_0_complex128) { Reverse_0<complex128>(); }

TEST_F(ReverseOpTest, Reverse_234_uint8) { Reverse_234<uint8>(); }

TEST_F(ReverseOpTest, Reverse_234_int8) { Reverse_234<int8>(); }

TEST_F(ReverseOpTest, Reverse_234_uint16) { Reverse_234<uint16>(); }

TEST_F(ReverseOpTest, Reverse_234_int16) { Reverse_234<int16>(); }

TEST_F(ReverseOpTest, Reverse_234_float) { Reverse_234<float>(); }

TEST_F(ReverseOpTest, Reverse_234_int32) { Reverse_234<int32>(); }

TEST_F(ReverseOpTest, Reverse_234_int64) { Reverse_234<int64_t>(); }

TEST_F(ReverseOpTest, Reverse_234_double) { Reverse_234<double>(); }

TEST_F(ReverseOpTest, Reverse_234_complex64) { Reverse_234<complex64>(); }

TEST_F(ReverseOpTest, Reverse_234_complex128) { Reverse_234<complex128>(); }

TEST_F(ReverseOpTest, Reverse_1234_uint8) { Reverse_1234<uint8>(); }

TEST_F(ReverseOpTest, Reverse_1234_int8) { Reverse_1234<int8>(); }

TEST_F(ReverseOpTest, Reverse_1234_uint16) { Reverse_1234<uint16>(); }

TEST_F(ReverseOpTest, Reverse_1234_int16) { Reverse_1234<int16>(); }

TEST_F(ReverseOpTest, Reverse_1234_float) { Reverse_1234<float>(); }

TEST_F(ReverseOpTest, Reverse_1234_int32) { Reverse_1234<int32>(); }

TEST_F(ReverseOpTest, Reverse_1234_int64) { Reverse_1234<int64_t>(); }

TEST_F(ReverseOpTest, Reverse_1234_double) { Reverse_1234<double>(); }

TEST_F(ReverseOpTest, Reverse_1234_complex64) { Reverse_1234<complex64>(); }

TEST_F(ReverseOpTest, Reverse_1234_complex128) { Reverse_1234<complex128>(); }

static SessionOptions GetOptions(int intra_threads) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_4(mht_4_v, 359, "", "./tensorflow/core/kernels/reverse_op_test.cc", "GetOptions");

  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(intra_threads);
  opts.config.set_inter_op_parallelism_threads(1);
  return opts;
}

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
template <typename T>
static Graph* Reverse(const TensorShape& shape, int reverse_axis) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_5(mht_5_v, 372, "", "./tensorflow/core/kernels/reverse_op_test.cc", "Reverse");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, shape);
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = reverse_axis;
  test::graph::Reverse(g, test::graph::Constant(g, data),
                       test::graph::Constant(g, axes));
  return g;
}

template <typename T>
static void RunReverseRowsBenchmark(::testing::benchmark::State& state,
                                    int outer_dim, int middle_dim,
                                    int intra_threads, int channels) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_6(mht_6_v, 389, "", "./tensorflow/core/kernels/reverse_op_test.cc", "RunReverseRowsBenchmark");

  SessionOptions opts = GetOptions(intra_threads);
  TensorShape shape{outer_dim, middle_dim, channels};
  test::Benchmark("cpu", Reverse<T>(shape, 1), &opts, nullptr, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  const int64_t num_items =
      static_cast<int64_t>(state.iterations()) * shape.num_elements();
  state.SetItemsProcessed(num_items);
  state.SetBytesProcessed(num_items * sizeof(T));
}

void BM_ReverseRowsOf1Channel_1T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_7(mht_7_v, 404, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf1Channel_1T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_1T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf1Channel_1T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_8(mht_8_v, 421, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf1Channel_1T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_1T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf1Channel_4T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_9(mht_9_v, 438, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf1Channel_4T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_4T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf1Channel_4T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_10(mht_10_v, 455, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf1Channel_4T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_4T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf3Channels_1T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_11(mht_11_v, 472, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf3Channels_1T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_1T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf3Channels_1T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_12(mht_12_v, 490, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf3Channels_1T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_1T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf3Channels_4T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_13(mht_13_v, 508, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf3Channels_4T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_4T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf3Channels_4T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_14(mht_14_v, 526, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf3Channels_4T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 3 /* channels */);
}
BENCHMARK(BM_ReverseRowsOf3Channels_4T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf4Channels_1T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_15(mht_15_v, 543, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf4Channels_1T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_1T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf4Channels_1T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_16(mht_16_v, 560, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf4Channels_1T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 1 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_1T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf4Channels_4T_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_17(mht_17_v, 577, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf4Channels_4T_float");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<float>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_4T_float)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

void BM_ReverseRowsOf4Channels_4T_uint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_op_testDTcc mht_18(mht_18_v, 594, "", "./tensorflow/core/kernels/reverse_op_test.cc", "BM_ReverseRowsOf4Channels_4T_uint8");

  const int outer_dim = state.range(0);
  const int middle_dim = state.range(1);

  RunReverseRowsBenchmark<uint8>(state, outer_dim, middle_dim,
                                 4 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_4T_uint8)
    ->UseRealTime()
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

}  // namespace
}  // namespace tensorflow
