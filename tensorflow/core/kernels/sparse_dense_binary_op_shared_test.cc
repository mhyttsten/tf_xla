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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

class SparseDenseCDivTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "MakeOp");

    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cdiv", "SparseDenseCwiseDiv")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class SparseDenseCMulTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "MakeOp");

    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cmul", "SparseDenseCwiseMul")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_FewerDims) {
  MakeOp<float>();
  // [1] op [2, 1]
  AddInputFromArray<int64_t>(TensorShape({1, 1}), {0});     // indices
  AddInputFromArray<float>(TensorShape({1}), {1618});       // values
  AddInputFromArray<int64_t>(TensorShape({1}), {1});        // shape
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});  // dense

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_SameDims) {
  MakeOp<float>();
  // [1, 1] op [2, 1]
  AddInputFromArray<int64_t>(TensorShape({1, 2}), {0, 0});
  AddInputFromArray<float>(TensorShape({1}), {1618});
  AddInputFromArray<int64_t>(TensorShape({2}), {1, 1});
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, SameShape) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: same shape, all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const gtl::ArraySlice<int64_t> shape(sh);

  // Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  Tensor dense(DT_FLOAT, TensorShape(shape));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape(shape), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseSameDims) {
  // No broadcast.
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [3,1], all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const gtl::ArraySlice<int64_t> shape(sh);

  Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({3, 1}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseFewerDims) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [2]]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const gtl::ArraySlice<int64_t> shape(sh);

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCMulTest, BroadcastDense) {
  MakeOp<float>();
  // [    1]
  // [2    ] (shape [3,2])  cmul  [0.5  0] (shape [2])
  // [3   4]
  //
  // Result:
  // [?   0]
  // [1   ?]  where ? remains implicitly zero.
  // [1.5 0]
  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64_t> in{0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64_t> indices(in);
  std::initializer_list<int64_t> sh{3, 2};
  const gtl::ArraySlice<int64_t> shape(sh);

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat(0) = 0.5;
  dense_flat(1) = 0;

  AddInputFromArray<int64_t>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64_t>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {0, 1, 1.5, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// Benchmarking code follows.

static Graph* SparseMatCMulDenseMat(Graph* g, Node* sp_indices, Node* sp_vals,
                                    Node* sp_shape, Node* dense) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_3(mht_3_v, 392, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "SparseMatCMulDenseMat");

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("SparseDenseCwiseMul"), "SparseDenseCwiseMul")
          .Input(sp_indices)
          .Input(sp_vals)
          .Input(sp_shape)
          .Input(dense)
          .Finalize(g, &ret));
  return g;
}

static Node* MakeTensor(Graph* g, int B, int M, int N) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_4(mht_4_v, 407, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "MakeTensor");

  Tensor data(DT_FLOAT, TensorShape({B, M, N}));
  data.flat<float>().setRandom();
  return test::graph::Constant(g, data);
}

struct ST {
  Node* indices;
  Node* vals;
  Node* shape;
};

static ST MakeSparseTensor(Graph* g, int B, int M, int N, int nnz_inner) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_shared_testDTcc mht_5(mht_5_v, 422, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared_test.cc", "MakeSparseTensor");

  const int total_nnz = B * M * nnz_inner;
  const int kNumDims = 3;

  Tensor indices(DT_INT64, TensorShape({total_nnz, kNumDims}));
  Tensor vals(DT_FLOAT, TensorShape({total_nnz}));
  Tensor shape(DT_INT64, TensorShape({kNumDims}));
  vals.flat<float>().setRandom();
  test::FillValues(&shape, gtl::ArraySlice<int64_t>({B, M, N}));
  auto indices_mat = indices.matrix<int64_t>();

  int nnz_cnt = 0;
  std::unordered_set<int> picked;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, N - 1);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < nnz_inner; ++k) {
        indices_mat(nnz_cnt, 0) = i;
        indices_mat(nnz_cnt, 1) = j;

        int inner = dist(gen);
        while (picked.count(inner) == 1) {
          inner = dist(gen);
        }
        picked.insert(inner);
        indices_mat(nnz_cnt, 2) = inner;

        ++nnz_cnt;
      }
    }
  }

  return ST{test::graph::Constant(g, indices), test::graph::Constant(g, vals),
            test::graph::Constant(g, shape)};
}

// [8, 4, N{nnz}] cmul [8, 4, N]
#define BM_SparseMatCMulDenseMatArgs(N, NNZ_INNER)                             \
  static void BM_SparseMatCMulDenseMat_##N##_##NNZ_INNER(                      \
      ::testing::benchmark::State& state) {                                    \
    Graph* g = new Graph(OpRegistry::Global());                                \
    Node* dense = MakeTensor(g, 8, 4, N);                                      \
    ST sp = MakeSparseTensor(g, 8, 4, N, NNZ_INNER);                           \
                                                                               \
    test::Benchmark(                                                           \
        "cpu", SparseMatCMulDenseMat(g, sp.indices, sp.vals, sp.shape, dense), \
        /*old_benchmark_api*/ false)                                           \
        .Run(state);                                                           \
    state.SetItemsProcessed(                                                   \
        static_cast<int64_t>(state.iterations() * 8 * 4 * N * 2));             \
  }                                                                            \
  BENCHMARK(BM_SparseMatCMulDenseMat_##N##_##NNZ_INNER)

BM_SparseMatCMulDenseMatArgs(1048576, 1);
BM_SparseMatCMulDenseMatArgs(1048576, 8);
BM_SparseMatCMulDenseMatArgs(1048576, 32);
BM_SparseMatCMulDenseMatArgs(262144, 1);
BM_SparseMatCMulDenseMatArgs(262144, 8);
BM_SparseMatCMulDenseMatArgs(262144, 32);

}  // namespace

}  // namespace tensorflow
