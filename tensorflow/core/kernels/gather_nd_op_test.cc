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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc() {
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
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace test {
namespace graph {

class Node* GatherNd(Graph* g, class Node* in0, class Node* in1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/gather_nd_op_test.cc", "GatherNd");

  class Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "GatherNd")
                  .Input(in0)
                  .Input(in1)
                  .Finalize(g, &ret));
  return ret;
}

}  // namespace graph
}  // namespace test

namespace {

class GatherNdOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType param_type, DataType index_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/kernels/gather_nd_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherNd")
                     .Input(FakeInput(param_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherNdOpTest, Simple) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected, {8, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherNdOpTest, Quantized_UINT8) {
  MakeOp(DT_QUINT8, DT_INT32);

  // Feed and run
  AddInputFromArray<quint8>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_QUINT8, TensorShape({2}));
  test::FillValues<quint8>(&expected, {8, 4});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(GatherNdOpTest, Quantized_INT8) {
  MakeOp(DT_QINT8, DT_INT32);

  AddInputFromArray<qint8>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_QINT8, TensorShape({2}));
  test::FillValues<qint8>(&expected, {8, 4});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

constexpr int kLookups = 2000;

template <typename Index>
static Graph* GatherNd(int dim) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_nd_op_testDTcc mht_2(mht_2_v, 287, "", "./tensorflow/core/kernels/gather_nd_op_test.cc", "GatherNd");

  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 512MB buffer.
  // const int kRows = ((512 << 20) / sizeof(float)) / dim;
  Tensor params(DT_FLOAT, TensorShape({dim, 8, 16, 32}));
  params.flat<float>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  Tensor indices(DataTypeToEnum<Index>::value, TensorShape({kLookups, 4}));
  auto indices_mat = indices.matrix<Index>();
  for (int i = 0; i < kLookups; i++) {
    indices_mat(i, 0) = rnd.Uniform(dim);
    indices_mat(i, 1) = rnd.Uniform(8);
    indices_mat(i, 2) = rnd.Uniform(16);
    indices_mat(i, 3) = rnd.Uniform(32);
  }

  test::graph::GatherNd(g, test::graph::Constant(g, params),
                        test::graph::Constant(g, indices));
  return g;
}

#define BM_GATHER_ND(DEVICE, INDEX)                              \
  static void BM_##DEVICE##_gather_nd_##INDEX(                   \
      ::testing::benchmark::State& state) {                      \
    const int dim = state.range(0);                              \
    test::Benchmark(#DEVICE, GatherNd<INDEX>(dim),               \
                    /*old_benchmark_api=*/false)                 \
        .Run(state);                                             \
    const int64_t tot =                                          \
        static_cast<int64_t>(state.iterations()) * kLookups * 4; \
    state.SetItemsProcessed(tot);                                \
    state.SetBytesProcessed(tot * sizeof(float));                \
  }                                                              \
  BENCHMARK(BM_##DEVICE##_gather_nd_##INDEX)                     \
      ->UseRealTime()                                            \
      ->Arg(10)                                                  \
      ->Arg(100)                                                 \
      ->Arg(1000)                                                \
      ->Arg(10000)

BM_GATHER_ND(cpu, int32);
BM_GATHER_ND(gpu, int32);
BM_GATHER_ND(cpu, int64_t);
BM_GATHER_ND(gpu, int64_t);

}  // namespace
}  // namespace tensorflow
