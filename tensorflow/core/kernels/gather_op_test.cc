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
class MHTracer_DTPStensorflowPScorePSkernelsPSgather_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgather_op_testDTcc() {
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
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class GatherOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, DataType index_type, int batch_dims = 0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/gather_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherV2")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Attr("batch_dims", batch_dims)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherOpTest, ScalarIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected, {3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, DT_INT32);

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_COMPLEX64, TensorShape({}));
  test::FillValues<std::complex<float>>(&expected,
                                        {std::complex<float>(3, 13)});
  test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32_Axis0) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32_Axis1) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 1, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 4}));
  test::FillValues<float>(&expected, {0, 1, 0, 2,  3, 4,  3,  5,  6,  7,
                                      6, 8, 9, 10, 9, 11, 12, 13, 12, 14});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, ZeroSize_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0}), {});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64_t>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int64_t>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, HighRank) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 0, 2, 3, 0});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 2, 3, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 5)"))
      << s;
}

TEST_F(GatherOpTest, Error_BatchDimsOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32, 10);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Expected batch_dims in the range [-1, 1], but got 10"))
      << s;
}

constexpr int kLookups = 2000;

template <typename Index>
static Graph* Gather(int dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgather_op_testDTcc mht_1(mht_1_v, 369, "", "./tensorflow/core/kernels/gather_op_test.cc", "Gather");

  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 512MB buffer.
  const int kRows = ((512 << 20) / sizeof(float)) / dim;
  Tensor params(DT_FLOAT, TensorShape({kRows, dim}));
  params.flat<float>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices_vec;
  indices_vec.reserve(kLookups);
  for (int i = 0; i < kLookups; i++) {
    indices_vec.push_back(rnd.Uniform(kRows));
  }
  Tensor indices(DataTypeToEnum<Index>::value, TensorShape({kLookups}));
  for (int i = 0; i < indices_vec.size(); i++) {
    indices.flat<Index>()(i) = indices_vec[i];
  }

  Tensor axis(DataTypeToEnum<Index>::value, TensorShape({}));
  axis.scalar<Index>()() = 0;

  test::graph::Gather(g, test::graph::Constant(g, params),
                      test::graph::Constant(g, indices),
                      test::graph::HostConstant(g, axis));
  return g;
}

#define BM_GATHER(DEVICE, INDEX)                                              \
  static void BM_##DEVICE##_gather_##INDEX(                                   \
      ::testing::benchmark::State& state) {                                   \
    const int dim = state.range(0);                                           \
    test::Benchmark(#DEVICE, Gather<INDEX>(dim), /*old_benchmark_api=*/false) \
        .Run(state);                                                          \
    const int64_t tot =                                                       \
        static_cast<int64_t>(state.iterations()) * kLookups * dim;            \
    state.SetItemsProcessed(tot);                                             \
    state.SetBytesProcessed(tot * sizeof(float));                             \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_gather_##INDEX)                                     \
      ->UseRealTime()                                                         \
      ->Arg(1)                                                                \
      ->Arg(10)                                                               \
      ->Arg(20)                                                               \
      ->Arg(64)                                                               \
      ->Arg(100)                                                              \
      ->Arg(200)                                                              \
      ->Arg(1000)

BM_GATHER(cpu, int32);
BM_GATHER(gpu, int32);
BM_GATHER(cpu, int64_t);
BM_GATHER(gpu, int64_t);

}  // namespace
}  // namespace tensorflow
