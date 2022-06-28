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
class MHTracer_DTPStensorflowPScorePSkernelsPSclustering_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSclustering_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSclustering_ops_testDTcc() {
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

// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// ==============================================================================

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

constexpr int k100Dim = 100;
// Number of points for tests.
constexpr int k10Points = 10;
constexpr int k100Points = 100;
constexpr int k1kPoints = 1000;
constexpr int k10kPoints = 10000;
constexpr int k1MPoints = 1000000;
// Number of centers for tests.
constexpr int k2Centers = 2;
constexpr int k5Centers = 5;
constexpr int k10Centers = 10;
constexpr int k20Centers = 20;
constexpr int k50Centers = 50;
constexpr int k100Centers = 100;
constexpr int k200Centers = 200;
constexpr int k500Centers = 500;
constexpr int k1kCenters = 1000;
constexpr int k10kCenters = 10000;
// Number of retries for tests.
constexpr int k0RetriesPerSample = 0;
constexpr int k3RetriesPerSample = 3;

Graph* SetUpKmeansPlusPlusInitialization(int num_dims, int num_points,
                                         int num_to_sample,
                                         int retries_per_sample) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSclustering_ops_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/clustering_ops_test.cc", "SetUpKmeansPlusPlusInitialization");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor points(DT_FLOAT, TensorShape({num_points, num_dims}));
  Tensor sample_size(DT_INT64, TensorShape({}));
  Tensor seed(DT_INT64, TensorShape({}));
  Tensor num_retries_per_sample(DT_INT64, TensorShape({}));
  points.flat<float>().setRandom();
  sample_size.flat<int64_t>().setConstant(num_to_sample);
  seed.flat<int64_t>().setConstant(12345);
  num_retries_per_sample.flat<int64_t>().setConstant(retries_per_sample);

  TF_CHECK_OK(NodeBuilder("kmeans_plus_plus_initialization_op",
                          "KmeansPlusPlusInitialization")
                  .Input(test::graph::Constant(g, points))
                  .Input(test::graph::Constant(g, sample_size))
                  .Input(test::graph::Constant(g, seed))
                  .Input(test::graph::Constant(g, num_retries_per_sample))
                  .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_points, int num_to_sample, int num_dims,
          int retries_per_sample>
void BM_KmeansPlusPlusInitialization(::testing::benchmark::State& state) {
  Graph* g = SetUpKmeansPlusPlusInitialization(
      num_dims, num_points, num_to_sample, retries_per_sample);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_to_sample);
}

#define BENCHMARK_KMEANS_PLUS_PLUS(p, c, d, r)                     \
  void BM_KmeansPlusPlusInitialization_##p##_##c##_##d##_##r(      \
      ::testing::benchmark::State& state) {                        \
    BM_KmeansPlusPlusInitialization<p, c, d, r>(state);            \
  }                                                                \
  BENCHMARK(BM_KmeansPlusPlusInitialization_##p##_##c##_##d##_##r) \
      ->UseRealTime();

#define RUN_BM_KmeansPlusPlusInitialization(retries)                     \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k2Centers, k100Dim, retries);    \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k5Centers, k100Dim, retries);    \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k10Centers, k100Dim, retries);   \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k10Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k20Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k50Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k100Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k100Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k200Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k500Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k1kCenters, k100Dim, retries);   \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k100Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k200Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k500Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k1kCenters, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k100Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k200Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k500Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k1kCenters, k100Dim, retries)

RUN_BM_KmeansPlusPlusInitialization(k0RetriesPerSample);
RUN_BM_KmeansPlusPlusInitialization(k3RetriesPerSample);

#undef RUN_BM_KmeansPlusPlusInitialization
#undef BENCHMARK_KMEANS_PLUS_PLUS

Graph* SetUpKMC2Initialization(int num_points) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor distances(DT_FLOAT, TensorShape({num_points}));
  Tensor seed(DT_INT64, TensorShape({}));
  distances.flat<float>().setRandom();
  seed.flat<int64_t>().setConstant(12345);

  TF_CHECK_OK(
      NodeBuilder("KMC2ChainInitializationOp", "KMC2ChainInitialization")
          .Input(test::graph::Constant(g, distances))
          .Input(test::graph::Constant(g, seed))
          .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_points, int num_to_sample, int num_dims>
void BM_KMC2Initialization(::testing::benchmark::State& state) {
  Graph* g = SetUpKMC2Initialization(num_points);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_to_sample);
}
#define BENCHMARK_KMC2(p, c, d)               \
  void BM_KMC2Initialization_##p##_##c##_##d( \
      ::testing::benchmark::State& state) {   \
    BM_KMC2Initialization<p, c, d>(state);    \
  }                                           \
  BENCHMARK(BM_KMC2Initialization_##p##_##c##_##d)->UseRealTime();

#define RUN_BM_KMC2Initialization                   \
  BENCHMARK_KMC2(k10Points, k2Centers, k100Dim);    \
  BENCHMARK_KMC2(k10Points, k5Centers, k100Dim);    \
  BENCHMARK_KMC2(k10Points, k10Centers, k100Dim);   \
  BENCHMARK_KMC2(k100Points, k10Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k20Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k50Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k100Centers, k100Dim); \
  BENCHMARK_KMC2(k1kPoints, k100Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k200Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k500Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k1kCenters, k100Dim);   \
  BENCHMARK_KMC2(k10kPoints, k100Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k200Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k500Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k1kCenters, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k100Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k200Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k500Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k1kCenters, k100Dim)

RUN_BM_KMC2Initialization;
#undef RUN_BM_KMC2Initialization
#undef BENCHMARK_KMC2

Graph* SetUpNearestNeighbors(int num_dims, int num_points, int num_centers,
                             int k) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor points(DT_FLOAT, TensorShape({num_points, num_dims}));
  Tensor centers(DT_FLOAT, TensorShape({num_centers, num_dims}));
  Tensor top(DT_INT64, TensorShape({}));
  points.flat<float>().setRandom();
  centers.flat<float>().setRandom();
  top.flat<int64_t>().setConstant(k);

  TF_CHECK_OK(NodeBuilder("nearest_centers_op", "NearestNeighbors")
                  .Input(test::graph::Constant(g, points))
                  .Input(test::graph::Constant(g, centers))
                  .Input(test::graph::Constant(g, top))
                  .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_dims, int num_points, int num_centers, int k>
void BM_NearestNeighbors(::testing::benchmark::State& state) {
  Graph* g = SetUpNearestNeighbors(num_dims, num_points, num_centers, k);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_centers);
}

constexpr int kTop1 = 1;
constexpr int kTop2 = 2;
constexpr int kTop5 = 5;
constexpr int kTop10 = 10;

#define BENCHMARK_NEAREST_NEIGHBORS(d, p, c, k)  \
  void BM_NearestNeighbors##d##_##p##_##c##_##k( \
      ::testing::benchmark::State& state) {      \
    BM_NearestNeighbors<d, p, c, k>(state);      \
  }                                              \
  BENCHMARK(BM_NearestNeighbors##d##_##p##_##c##_##k)->UseRealTime();

#define RUN_BM_NearestNeighbors(k)                                 \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k100Centers, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k1kCenters, k);  \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k10kCenters, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k100Centers, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k1kCenters, k);  \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k10kCenters, k)

RUN_BM_NearestNeighbors(kTop1);
// k > 1
RUN_BM_NearestNeighbors(kTop2);
RUN_BM_NearestNeighbors(kTop5);
RUN_BM_NearestNeighbors(kTop10);

#undef RUN_BM_NearestNeighbors
#undef BENCHMARK_NEAREST_NEIGHBORS
}  // namespace
}  // namespace tensorflow
