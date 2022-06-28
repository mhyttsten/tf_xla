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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_testDTcc() {
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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

Node* SparseTensorDenseMatMulNode(Graph* g, Node* a_indices, Node* a_values,
                                  Node* a_shape, Node* b, bool adjoint_a,
                                  bool adjoint_b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/sparse_tensor_dense_matmul_op_test.cc", "SparseTensorDenseMatMulNode");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseTensorDenseMatMul")
                  .Input(a_indices)
                  .Input(a_values)
                  .Input(a_shape)
                  .Input(b)
                  .Attr("T", DT_FLOAT)
                  .Attr("adjoint_a", adjoint_a)
                  .Attr("adjoint_b", adjoint_b)
                  .Finalize(g, &ret));
  return ret;
}

static Graph* SparseTensorDenseMatmul(int nnz, int m, int k, int n,
                                      bool adjoint_a, bool adjoint_b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_matmul_op_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/sparse_tensor_dense_matmul_op_test.cc", "SparseTensorDenseMatmul");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor a_values(DT_FLOAT, TensorShape({nnz}));
  Tensor a_indices(DT_INT64, TensorShape({nnz, 2}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  auto a_shape_t = a_shape.vec<int64_t>();
  a_shape_t(0) = adjoint_a ? k : m;
  a_shape_t(1) = adjoint_a ? m : k;
  a_values.flat<float>().setRandom();
  auto a_indices_t = a_indices.matrix<int64_t>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> a_lhs_dist(0, a_shape_t(0) - 1);
  std::uniform_int_distribution<> a_rhs_dist(0, a_shape_t(1) - 1);
  for (int32_t i = 0; i < nnz; ++i) {
    a_indices_t(i, 0) = a_lhs_dist(gen);
    a_indices_t(i, 1) = a_rhs_dist(gen);
  }
  Tensor b(DT_FLOAT, adjoint_b ? TensorShape({n, k}) : TensorShape({k, n}));
  b.flat<float>().setRandom();

  SparseTensorDenseMatMulNode(
      g, test::graph::Constant(g, a_indices),
      test::graph::Constant(g, a_values), test::graph::HostConstant(g, a_shape),
      test::graph::Constant(g, b), adjoint_a, adjoint_b);
  return g;
}

// NOLINTBEGIN
#define BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, DEVICE)                  \
  static void                                                                        \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE( \
          ::testing::benchmark::State& state) {                                      \
    int64_t items_per_iter = (static_cast<int64_t>(NNZ) * (TB ? K : N));             \
    test::Benchmark(#DEVICE, SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB),          \
                    /*old_benchmark_api*/ false)                                     \
        .Run(state);                                                                 \
    state.SetItemsProcessed(state.iterations() * items_per_iter);                    \
    state.SetBytesProcessed(state.iterations() * items_per_iter *                    \
                            sizeof(float));                                          \
  }                                                                                  \
  BENCHMARK(                                                                         \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE);
// NOLINTEND

#define BM_SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB)    \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, cpu); \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, gpu);

BM_SparseTensorDenseMatmul(128, 8, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 16, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 128, 512, 1, false, false);

BM_SparseTensorDenseMatmul(128, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1, false, false);

BM_SparseTensorDenseMatmul(128, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(1024, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, true);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, true);

}  // end namespace tensorflow
