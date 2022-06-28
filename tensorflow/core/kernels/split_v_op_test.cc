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
class MHTracer_DTPStensorflowPScorePSkernelsPSsplit_v_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_v_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsplit_v_op_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdlib.h>

#include <initializer_list>
#include <iterator>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Generate "count" random positive integers (not including zero) with sum
// "sum". Technique based on one from https://math.stackexchange.com/a/1276225
// but simplified (especially for zero-based indexing).
static std::vector<int64_t> GenerateRandomIntsWithSum(int64_t sum, int count) {
  CHECK_GE(count, 1);
  CHECK_GE(sum, count);
  std::vector<int64_t> temp(count);
  for (int i = 0; i + 1 < count; ++i) {
    temp[i] = lrand48() % (sum - count);
  }
  temp[count - 1] = sum - count;
  std::sort(temp.begin(), std::prev(temp.end()));
  std::vector<int64_t> result(count);
  std::adjacent_difference(temp.begin(), temp.end(), result.begin());
  for (int i = 0; i < count; ++i) {
    ++result[i];
  }
  CHECK(std::all_of(result.begin(), result.end(),
                    [sum](int64_t x) { return x >= 1 && x <= sum; }));
  CHECK_EQ(
      std::accumulate(result.begin(), result.end(), static_cast<int64_t>(0)),
      sum);
  CHECK_EQ(result.size(), count);
  return result;
}

static Graph* MakeGraph(int split_dim, const std::vector<int64_t>& size_splits,
                        std::initializer_list<int64_t> total_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_v_op_testDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/kernels/split_v_op_test.cc", "MakeGraph");

  Graph* g = new Graph(OpRegistry::Global());
  TensorShape in_shape(total_size);
  Tensor in(DataTypeToEnum<float>::value, in_shape);
  in.flat<float>().setRandom();
  Tensor split_dim_tensor = test::AsScalar<int32>(split_dim);
  Tensor size_splits_tensor = test::AsTensor<int64_t>(size_splits);
  Node* splitv;
  TF_CHECK_OK(NodeBuilder(g->NewName("splitv"), "SplitV")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, size_splits_tensor))
                  .Input(test::graph::Constant(g, split_dim_tensor))
                  .Attr("num_split", static_cast<int64_t>(size_splits.size()))
                  .Finalize(g, &splitv));
  return g;
}

#define BM_SPLITV_1D(num_split, total_size)                                  \
  static void BM_SplitV_1d_##num_split##_##total_size(                       \
      ::testing::benchmark::State& state) {                                  \
    auto label =                                                             \
        strings::Printf("1-D %d chunks totaling %d", num_split, total_size); \
    state.SetLabel(label);                                                   \
    auto g = MakeGraph(/* split_dim = */ 0,                                  \
                       GenerateRandomIntsWithSum(total_size, num_split),     \
                       {total_size});                                        \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);       \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *       \
                            total_size);                                     \
  }                                                                          \
  BENCHMARK(BM_SplitV_1d_##num_split##_##total_size)->UseRealTime();

#define BM_SPLITV_2D(split_dim, num_split, dim0, dim1)                         \
  static void BM_SplitV_2d_##split_dim##_##num_split##dim0##dim1(              \
      ::testing::benchmark::State& state) {                                    \
    std::vector<int64_t> total_size_vec{dim0, dim1};                           \
    auto label = strings::Printf("2-D %d chunks in dim %d totaling (%d * %d)", \
                                 num_split, split_dim, dim0, dim1);            \
    state.SetLabel(label);                                                     \
    auto g = MakeGraph(                                                        \
        split_dim,                                                             \
        GenerateRandomIntsWithSum(total_size_vec[split_dim], num_split),       \
        {dim0, dim1});                                                         \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * dim0 *  \
                            dim1);                                             \
  }                                                                            \
  BENCHMARK(BM_SplitV_2d_##split_dim##_##num_split##dim0##dim1)->UseRealTime();

#define BM_SPLITV_3D(split_dim, num_split, dim0, dim1, dim2)                  \
  static void BM_SplitV_3d_##split_dim##_##num_split##dim0##dim1##dim2(       \
      ::testing::benchmark::State& state) {                                   \
    std::vector<int64_t> total_size_vec{dim0, dim1, dim2};                    \
    auto label =                                                              \
        strings::Printf("3-D %d chunks in dim %d totaling (%d * %d * %d)",    \
                        num_split, split_dim, dim0, dim1, dim2);              \
    state.SetLabel(label);                                                    \
    auto g = MakeGraph(                                                       \
        split_dim,                                                            \
        GenerateRandomIntsWithSum(total_size_vec[split_dim], num_split),      \
        {dim0, dim1, dim2});                                                  \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);        \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * dim0 * \
                            dim1 * dim2);                                     \
  }                                                                           \
  BENCHMARK(BM_SplitV_3d_##split_dim##_##num_split##dim0##dim1##dim2)         \
      ->UseRealTime();

BM_SPLITV_1D(5, 20);
BM_SPLITV_1D(262144, 1000000);
BM_SPLITV_1D(1, 100000);
BM_SPLITV_1D(5, 100000);
BM_SPLITV_1D(5, 250000);
BM_SPLITV_1D(5, 500000);
BM_SPLITV_1D(5, 1000000);
BM_SPLITV_1D(10, 4194304);
BM_SPLITV_1D(2, 4194304);
BM_SPLITV_1D(100, 10240);
BM_SPLITV_1D(32768, 1048576);

BM_SPLITV_2D(0, 1024, 10247, 10);
BM_SPLITV_2D(0, 1024, 100000, 10);
BM_SPLITV_2D(0, 512, 1024, 256);
BM_SPLITV_2D(0, 20, 100000, 5);
BM_SPLITV_2D(0, 2, 7, 524288);
BM_SPLITV_2D(0, 100, 4096, 512);

BM_SPLITV_2D(1, 1024, 15, 10240);
BM_SPLITV_2D(1, 1024, 10, 100000);
BM_SPLITV_2D(1, 512, 1024, 2563);
BM_SPLITV_2D(1, 20, 100000, 52);
BM_SPLITV_2D(1, 2, 3, 524288);
BM_SPLITV_2D(1, 100, 4096, 512);

BM_SPLITV_3D(0, 1024, 10247, 10, 1024);
BM_SPLITV_3D(0, 987, 1000, 10, 512);
BM_SPLITV_3D(0, 512, 1024, 256, 128);
BM_SPLITV_3D(0, 20, 100000, 5, 256);
BM_SPLITV_3D(0, 2, 7, 524288, 10);
BM_SPLITV_3D(0, 100, 4096, 512, 1);

BM_SPLITV_3D(1, 1024, 15, 10240, 1024);
BM_SPLITV_3D(1, 512, 10, 1024, 512);
BM_SPLITV_3D(1, 512, 1024, 2563, 128);
BM_SPLITV_3D(1, 20, 1000, 52, 256);
BM_SPLITV_3D(1, 2, 3, 524288, 10);
BM_SPLITV_3D(1, 100, 4096, 512, 1);

BM_SPLITV_3D(2, 512, 15, 10240, 1024);
BM_SPLITV_3D(2, 128, 10, 1000, 512);
BM_SPLITV_3D(2, 63, 1024, 2563, 128);
BM_SPLITV_3D(2, 20, 1000, 52, 256);
BM_SPLITV_3D(2, 2, 3, 524288, 10);
BM_SPLITV_3D(2, 1, 4096, 512, 1);

}  // namespace tensorflow
