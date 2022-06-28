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
class MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc() {
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
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// We set only the number of threads for intra op thread pool to test how well
// each individual kernel utilize multiple threads.
static SessionOptions* InitMultiThreadingOptions(int num_threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/kernels/training_ops_test.cc", "InitMultiThreadingOptions");

  SessionOptions* opts = new SessionOptions();
  opts->config.set_intra_op_parallelism_threads(num_threads);
  opts->config.set_inter_op_parallelism_threads(1);
  return opts;
}

static SessionOptions* GetOptions() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/kernels/training_ops_test.cc", "GetOptions");

  static SessionOptions* opts = InitMultiThreadingOptions(1);
  return opts;
}

static SessionOptions* GetMultiThreadedOptions() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/core/kernels/training_ops_test.cc", "GetMultiThreadedOptions");

  static SessionOptions* opts = InitMultiThreadingOptions(32);
  return opts;
}

static Node* Var(Graph* g, int n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/kernels/training_ops_test.cc", "Var");

  return test::graph::Var(g, DT_FLOAT, TensorShape({n}));
}

static Node* Var(Graph* g, int m, int n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_4(mht_4_v, 229, "", "./tensorflow/core/kernels/training_ops_test.cc", "Var");

  return test::graph::Var(g, DT_FLOAT, TensorShape({m, n}));
}

static Node* Zeros(Graph* g, int n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_5(mht_5_v, 236, "", "./tensorflow/core/kernels/training_ops_test.cc", "Zeros");

  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

static Node* Zeros(Graph* g, int m, int n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_6(mht_6_v, 245, "", "./tensorflow/core/kernels/training_ops_test.cc", "Zeros");

  Tensor data(DT_FLOAT, TensorShape({m, n}));
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

static Node* Random(Graph* g, int n) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_7(mht_7_v, 254, "", "./tensorflow/core/kernels/training_ops_test.cc", "Random");

  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setRandom();
  return test::graph::Constant(g, data);
}

static Node* Random(Graph* g, int m, int n) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_8(mht_8_v, 263, "", "./tensorflow/core/kernels/training_ops_test.cc", "Random");

  Tensor data(DT_FLOAT, TensorShape({m, n}));
  data.flat<float>().setRandom();
  return test::graph::Constant(g, data);
}

static Node* Iota(Graph* g, int n) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_9(mht_9_v, 272, "", "./tensorflow/core/kernels/training_ops_test.cc", "Iota");

  Tensor data(DT_INT32, TensorShape({n}));
  int32* base = data.flat<int32>().data();
  for (int i = 0; i < n; ++i) base[i] = i;
  return test::graph::Constant(g, data);
}

static Node* Scalar(Graph* g, float val) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_10(mht_10_v, 282, "", "./tensorflow/core/kernels/training_ops_test.cc", "Scalar");

  Tensor data(DT_FLOAT, TensorShape({}));
  data.flat<float>()(0) = val;
  return test::graph::Constant(g, data);
}

static void SGD(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_11(mht_11_v, 291, "", "./tensorflow/core/kernels/training_ops_test.cc", "SGD");

  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    test::graph::Assign(g, var, Zeros(g, n));
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyGradientDescent", {var, lr, grad});
    *train_g = g;
  }
}

static void BM_SGD(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_12(mht_12_v, 311, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_SGD");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  SGD(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_SGD)->Arg(128 << 10)->Arg(256 << 10);

static void Adagrad(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_13(mht_13_v, 329, "", "./tensorflow/core/kernels/training_ops_test.cc", "Adagrad");

  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyAdagrad", {var, accum, lr, grad});
    *train_g = g;
  }
}

static void BM_Adagrad(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_14(mht_14_v, 353, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_Adagrad");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  Adagrad(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_Adagrad)->Arg(128 << 10)->Arg(256 << 10);

static void SparseAdagrad(int32_t m, int32_t n, Graph** init_g,
                          Graph** train_g) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_15(mht_15_v, 372, "", "./tensorflow/core/kernels/training_ops_test.cc", "SparseAdagrad");

  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, m, n);
    auto accum = Var(g, m, n);
    auto zero = Zeros(g, m, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, m, n);
    auto accum = Var(g, m, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, m, n);
    auto indices = Iota(g, m);
    test::graph::Multi(g, "SparseApplyAdagrad",
                       {var, accum, lr, grad, indices});
    *train_g = g;
  }
}
static void BM_SparseAdagrad(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_16(mht_16_v, 397, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_SparseAdagrad");

  const int m = state.range(0);
  const int n = state.range(1);

  Graph* init;
  Graph* train;
  SparseAdagrad(m, n, &init, &train);
  test::Benchmark("cpu", train, GetMultiThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * m * n;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_SparseAdagrad)
    ->UseRealTime()
    ->ArgPair(128, 1 << 10)
    ->ArgPair(128, 4 << 10)
    ->ArgPair(128, 8 << 10)
    ->ArgPair(128, 32 << 10)
    ->ArgPair(128, 128 << 10);

static void Momentum(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_17(mht_17_v, 422, "", "./tensorflow/core/kernels/training_ops_test.cc", "Momentum");

  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    auto mom = Scalar(g, 0.01);
    test::graph::Multi(g, "ApplyMomentum", {var, accum, lr, grad, mom});
    *train_g = g;
  }
}

static void BM_Momentum(::testing::benchmark::State& state) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_18(mht_18_v, 448, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_Momentum");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  Momentum(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_Momentum)->Arg(128 << 10)->Arg(256 << 10);

static void Adam(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_19(mht_19_v, 466, "", "./tensorflow/core/kernels/training_ops_test.cc", "Adam");

  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    test::graph::Assign(g, v, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto beta1_power = Scalar(g, 0.9);
    auto beta2_power = Scalar(g, 0.99);
    auto lr = Scalar(g, 0.01);
    auto beta1 = Scalar(g, 0.9);
    auto beta2 = Scalar(g, 0.99);
    auto epsilon = Scalar(g, 1e-8);
    auto grad = Random(g, n);
    test::graph::Multi(
        g, "ApplyAdam",
        {var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad});
    *train_g = g;
  }
}

static void BM_Adam(::testing::benchmark::State& state) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_20(mht_20_v, 501, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_Adam");

  const int params = state.range(0);
  const int is_multi_threaded = state.range(1);

  Graph* init;
  Graph* train;
  Adam(params, &init, &train);
  if (is_multi_threaded) {
    // Use max thread number if test performance.
    test::Benchmark("cpu", train, nullptr, init, nullptr, "",
                    /*old_benchmark_api*/ false)
        .Run(state);
  } else {
    test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                    /*old_benchmark_api*/ false)
        .Run(state);
  }
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_Adam)->ArgPair(128 << 10, 0)->ArgPair(256 << 10, 0);
BENCHMARK(BM_Adam)->ArgPair(256 << 5, 1)->ArgPair(256 << 16, 1);

static void RMSProp(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_21(mht_21_v, 528, "", "./tensorflow/core/kernels/training_ops_test.cc", "RMSProp");

  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto ms = Var(g, n);
    auto mom = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, ms, zero);
    test::graph::Assign(g, mom, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto ms = Var(g, n);
    auto mom = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto rho = Scalar(g, 0.9);
    auto momentum = Scalar(g, 0.9);
    auto epsilon = Scalar(g, 1e-8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyRMSProp",
                       {var, ms, mom, lr, rho, momentum, epsilon, grad});
    *train_g = g;
  }
}

static void BM_RMSProp(::testing::benchmark::State& state) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_22(mht_22_v, 560, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_RMSProp");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  RMSProp(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benhcmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_RMSProp)->Arg(128 << 10)->Arg(256 << 10);

static void AddSign(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_23(mht_23_v, 578, "", "./tensorflow/core/kernels/training_ops_test.cc", "AddSign");

  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto alpha = Scalar(g, 0.1);
    auto sign_decay = Scalar(g, 0.9);
    auto beta = Scalar(g, 0.8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyAddSign",
                       {var, m, lr, alpha, sign_decay, beta, grad});
    *train_g = g;
  }
}

static void BM_AddSign(::testing::benchmark::State& state) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_24(mht_24_v, 607, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_AddSign");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  AddSign(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benhcmark_api*/ false)
      .Run(state);
  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_AddSign)->Arg(128 << 10)->Arg(256 << 10);

static void PowerSign(int32_t n, Graph** init_g, Graph** train_g) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_25(mht_25_v, 625, "", "./tensorflow/core/kernels/training_ops_test.cc", "PowerSign");

  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto logbase = Scalar(g, 2);
    auto sign_decay = Scalar(g, 0.9);
    auto beta = Scalar(g, 0.8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyPowerSign",
                       {var, m, lr, logbase, sign_decay, beta, grad});
    *train_g = g;
  }
}

static void BM_PowerSign(::testing::benchmark::State& state) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStraining_ops_testDTcc mht_26(mht_26_v, 654, "", "./tensorflow/core/kernels/training_ops_test.cc", "BM_PowerSign");

  const int params = state.range(0);

  Graph* init;
  Graph* train;
  PowerSign(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init, nullptr, "",
                  /*old_benhcmark_api*/ false)
      .Run(state);

  const int64_t tot = static_cast<int64_t>(state.iterations()) * params;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * sizeof(float));
}
BENCHMARK(BM_PowerSign)->Arg(128 << 10)->Arg(256 << 10);

}  // end namespace tensorflow
