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
class MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc() {
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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
template <typename T>
static Graph* ToScalar(const string& reduce, int num_x, int num_y) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "ToScalar");

  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, TensorShape({num_x, num_y}));
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ColReduce(const string& reduce, int num_x, int num_y) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "ColReduce");

  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 0;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* RowReduce(const string& reduce, int num_x, int num_y) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "RowReduce");

  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDYReduce(const string& reduce, int num_y, int num_z) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "ThreeDYReduce");

  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDXZReduce(const string& reduce, int num_y, int num_z) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "ThreeDXZReduce");

  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 2;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

// Creates a bench which reduces a 3D tensor with total "num" floats
// into a scalar on a "device". Runs the bench for "iters" times.
template <typename T>
static void ReduceToScalar(::testing::benchmark::State& state,
                           const string& device, const string& reduce,
                           int num_x, int num_y) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device: \"" + device + "\"");
   mht_5_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "ReduceToScalar");

  test::Benchmark(device, ToScalar<T>(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y * sizeof(T));
}

static void DoRowReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("device: \"" + device + "\"");
   mht_6_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "DoRowReduce");

  test::Benchmark(device, RowReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void DoColReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("device: \"" + device + "\"");
   mht_7_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_7(mht_7_v, 314, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "DoColReduce");

  test::Benchmark(device, ColReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void Do3DYReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("device: \"" + device + "\"");
   mht_8_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_8(mht_8_v, 331, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "Do3DYReduce");

  test::Benchmark(device, ThreeDYReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void Do3DXZReduce(::testing::benchmark::State& state,
                         const string& device, const string& reduce, int num_x,
                         int num_y) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device: \"" + device + "\"");
   mht_9_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_9(mht_9_v, 348, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "Do3DXZReduce");

  test::Benchmark(device, ThreeDXZReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void BM_Sum2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_10(mht_10_v, 361, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUComplex(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_11(mht_11_v, 372, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum2DToScalarGPUComplex");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<std::complex<float>>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUComplex)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUHalf(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_12(mht_12_v, 383, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum2DToScalarGPUHalf");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<Eigen::half>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUHalf)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DRowReduceGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_13(mht_13_v, 394, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum2DRowReduceGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  DoRowReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DRowReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DColumnReduceGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_14(mht_14_v, 405, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum2DColumnReduceGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  DoColReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DColumnReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum3DYReduceGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_15(mht_15_v, 416, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum3DYReduceGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  Do3DYReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DYReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Sum3DXZReduceGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_16(mht_16_v, 427, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Sum3DXZReduceGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  Do3DXZReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DXZReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Mean2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_17(mht_17_v, 438, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Mean2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Mean", num_x, num_y);
}
BENCHMARK(BM_Mean2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_EuclideanNorm2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_18(mht_18_v, 449, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_EuclideanNorm2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "EuclideanNorm", num_x, num_y);
}
BENCHMARK(BM_EuclideanNorm2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Max2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_19(mht_19_v, 460, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Max2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Max", num_x, num_y);
}
BENCHMARK(BM_Max2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_20(mht_20_v, 471, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Min2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPUHalf(::testing::benchmark::State& state) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_21(mht_21_v, 482, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Min2DToScalarGPUHalf");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<Eigen::half>(state, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPUHalf)->RangePair(2048, 8192, 2048, 8192);

static void BM_Bool2DToScalarGPU(::testing::benchmark::State& state) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_testDTcc mht_22(mht_22_v, 493, "", "./tensorflow/core/kernels/reduction_ops_test.cc", "BM_Bool2DToScalarGPU");

  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<bool>(state, "gpu", "All", num_x, num_y);
}
BENCHMARK(BM_Bool2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

}  // end namespace tensorflow
