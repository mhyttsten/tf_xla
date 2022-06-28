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
class MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class DynamicPartitionOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/dynamic_partition_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "DynamicPartition")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("num_partitions", 4)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(DynamicPartitionOpTest, Simple_OneD) {
  MakeOp();

  // Similar to how we would use this to split embedding ids to be looked up

  // Feed and run
  AddInputFromArray<float>(TensorShape({6}), {0, 13, 2, 39, 4, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 3, 2, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&expected, {0, 13});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
    test::FillValues<float>(&expected, {17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&expected, {2, 4});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
    test::FillValues<float>(&expected, {39});
    test::ExpectTensorEqual<float>(expected, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, Simple_TwoD) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(
      TensorShape({6, 3}),
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 3, 2, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
    test::FillValues<float>(&expected, {0, 1, 2, 3, 4, 5});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3}));
    test::FillValues<float>(&expected, {15, 16, 17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
    test::FillValues<float>(&expected, {6, 7, 8, 12, 13, 14});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3}));
    test::FillValues<float>(&expected, {9, 10, 11});
    test::ExpectTensorEqual<float>(expected, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, SomeOutputsEmpty) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(TensorShape({6}), {0, 13, 2, 39, 4, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 2, 0, 2});
  TF_ASSERT_OK(RunOpKernel());

  TensorShape empty_one_dim;
  empty_one_dim.AddDim(0);
  Tensor expected_empty(allocator(), DT_FLOAT, empty_one_dim);

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected, {0, 13, 4});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    test::ExpectTensorEqual<float>(expected_empty, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected, {2, 39, 17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    test::ExpectTensorEqual<float>(expected_empty, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, Error_IndexOutOfRange) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({5}), {0, 2, 99, 2, 2});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "partitions[2] = 99 is not in [0, 4)"))
      << s;
}

Node* DynamicPartitionNode(Graph* g, Node* in0, Node* in1, int num_partitions) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc mht_1(mht_1_v, 334, "", "./tensorflow/core/kernels/dynamic_partition_op_test.cc", "DynamicPartitionNode");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "DynamicPartition")
                  .Input(in0)
                  .Input(in1)
                  .Attr("num_partitions", num_partitions)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* DynamicPartition(int num_partitions, int dim) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_op_testDTcc mht_2(mht_2_v, 348, "", "./tensorflow/core/kernels/dynamic_partition_op_test.cc", "DynamicPartition");

  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 128MB buffer.
  const int kRows = ((128 << 20) / sizeof(T)) / dim;
  Tensor data(DataTypeToEnum<T>::value, TensorShape({kRows, dim}));
  data.flat<T>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  Tensor partitions(DT_INT32, TensorShape({kRows}));
  for (int i = 0; i < kRows; i++) {
    partitions.flat<int32>()(i) = rnd.Uniform(num_partitions);
  }
  DynamicPartitionNode(g, test::graph::Constant(g, data),
                       test::graph::Constant(g, partitions), num_partitions);
  return g;
}

#define BM_DYNAMIC_PARTITION(DEVICE, T, num)                              \
  static void BM_##DEVICE##_dynpart_##T##_##num(                          \
      ::testing::benchmark::State& state) {                               \
    const int dim = state.range(0);                                       \
                                                                          \
    const int64_t items = ((128 << 20) / sizeof(T));                      \
    test::Benchmark(#DEVICE, DynamicPartition<T>(num, dim),               \
                    /*old_benchmark_api=*/false)                          \
        .Run(state);                                                      \
    const int64_t tot = static_cast<int64_t>(state.iterations()) * items; \
    state.SetItemsProcessed(tot);                                         \
  }                                                                       \
  BENCHMARK(BM_##DEVICE##_dynpart_##T##_##num)->UseRealTime()->Arg(1)->Arg(256)

BM_DYNAMIC_PARTITION(cpu, float, 2);
BM_DYNAMIC_PARTITION(cpu, float, 100);
BM_DYNAMIC_PARTITION(cpu, double, 2);
BM_DYNAMIC_PARTITION(cpu, double, 100);
BM_DYNAMIC_PARTITION(cpu, complex64, 2);
BM_DYNAMIC_PARTITION(cpu, complex64, 100);

BM_DYNAMIC_PARTITION(gpu, int32, 2);
BM_DYNAMIC_PARTITION(gpu, int32, 100);
BM_DYNAMIC_PARTITION(gpu, int64, 2);
BM_DYNAMIC_PARTITION(gpu, int64, 100);
BM_DYNAMIC_PARTITION(gpu, float, 2);
BM_DYNAMIC_PARTITION(gpu, float, 100);
BM_DYNAMIC_PARTITION(gpu, double, 2);
BM_DYNAMIC_PARTITION(gpu, double, 100);
BM_DYNAMIC_PARTITION(gpu, complex64, 2);
BM_DYNAMIC_PARTITION(gpu, complex64, 100);

}  // namespace
}  // namespace tensorflow
