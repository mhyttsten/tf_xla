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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc() {
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

#include "tensorflow/core/kernels/sparse_matmul_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
random::PhiloxRandom philox(1, 1);
random::SimplePhilox rnd(&philox);
using Eigen::operator==;

template <typename T>
void Sparsify(Tensor* t, float sparsity) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "Sparsify");

  const int64_t N = t->NumElements();
  CHECK_LE(sparsity, 1);
  auto flat = t->flat<T>();
  if (sparsity == 1) {
    flat.setZero();
    return;
  }
  static const uint32 K = 10000;
  for (int64_t i = 0; i < N; ++i) {
    if (rnd.Uniform(K) < sparsity * K) {
      flat(i) = T(0);
    } else if (flat(i) == T(0)) {
      flat(i) = T(1);
    }
  }
}

Node* SparseMatMulNode(Graph* g, Node* in0, Node* in1, bool transpose_a,
                       bool transpose_b, bool a_sparse, bool b_sparse) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "SparseMatMulNode");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("a_is_sparse", a_sparse)
                  .Attr("b_is_sparse", b_sparse)
                  .Finalize(g, &ret));
  return ret;
}

template <typename TA, typename TB>
static Graph* SparseMatMulHelper(Graph* g, int m, int n, int d,
                                 float sparsity_a, float sparsity_b,
                                 bool transpose_a, bool transpose_b) {
  bool a_sparse = (sparsity_a > 0);
  bool b_sparse = (sparsity_b > 0);

  auto left_shape = transpose_a ? TensorShape({d, m}) : TensorShape({m, d});
  Tensor left(DataTypeToEnum<TA>::value, left_shape);
  left.flat<TA>().setRandom();
  Sparsify<TA>(&left, sparsity_a);

  auto right_shape = transpose_b ? TensorShape({n, d}) : TensorShape({d, n});
  Tensor right(DataTypeToEnum<TB>::value, right_shape);
  right.flat<TB>().setRandom();
  Sparsify<TB>(&right, sparsity_b);

  SparseMatMulNode(g, test::graph::Constant(g, left),
                   test::graph::Constant(g, right), transpose_a, transpose_b,
                   a_sparse, b_sparse);
  return g;
}

template <typename TA, typename TB>
static Graph* SparseMatMul(int m, int n, int d, float sparsity_a,
                           float sparsity_b, bool transpose_a,
                           bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  return SparseMatMulHelper<TA, TB>(g, m, n, d, sparsity_a, sparsity_b,
                                    transpose_a, transpose_b);
}

static Graph* ReplicatedSparseMatMul(int m, int n, int d, float sparsity_1,
                                     float sparsity_2, int copies) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "ReplicatedSparseMatMul");

  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, m, n, d, sparsity_1, sparsity_2, false,
                                     false);
  }
  return g;
}

#define BM_SPARSE(M, K, N, S1, S2, TRA, TRB, TA, TB)                           \
  static void                                                                  \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB( \
          ::testing::benchmark::State& state) {                                \
    auto label = strings::Printf("tr_a: %d tr_b: %d sp_a: %0.2f sp_b: %0.2f",  \
                                 TRA, TRB, S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g = SparseMatMul<TA, TB>(M, N, K, S1 / 100.0, S2 / 100.0, TRA, TRB);  \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB) \
      ->UseRealTime();

#define BM_SPARSE_REPLICATED(M, K, N, S1, S2, Copies)                          \
  static void BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies( \
      ::testing::benchmark::State& state) {                                    \
    auto label = strings::Printf("copies: %d sp_a: %0.2f sp_b: %0.2f",         \
                                 (Copies), S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g =                                                                   \
        ReplicatedSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, (Copies));     \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
    state.SetItemsProcessed(state.iterations() * M * K * N * Copies * 2);      \
  }                                                                            \
  BENCHMARK(BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies)   \
      ->UseRealTime();

#define BM_SPARSE_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, float)
#define BM_SPARSE_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, bfloat16)
#define BM_SPARSE_FLOAT_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, bfloat16)
#define BM_SPARSE_BFLOAT16_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, float)

// Test sparse b
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 50, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 99, 0, false, false);
// Test sparse a
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 50, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, false);
// Test transposing
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, true);

// Test smaller sizes
BM_SPARSE_FLOAT(1024, 1024, 1024, 0, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 1, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(256, 256, 256, 1, 0, false, false);
BM_SPARSE_FLOAT(512, 512, 512, 1, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, true, false);

BM_SPARSE_FLOAT(400, 800, 2560, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 2560, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 1024, 256, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 256, 1, 85, 0, false, false);

BM_SPARSE_REPLICATED(400, 800, 2560, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 2560, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 1024, 256, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 256, 1, 85, 0, 6);

BM_SPARSE_FLOAT(2048, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 512, 256, 85, 0, false, false);

BM_SPARSE_FLOAT(2049, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 512, 256, 85, 0, false, false);

BM_SPARSE_REPLICATED(2048, 1792, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 1024, 768, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 768, 512, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 512, 256, 85, 0, 6);

// Test bfloat16
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);

static Graph* MultiSparseMatMul(int m, int n, int d, float sparsity_1,
                                float sparsity_2, int copies) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_3(mht_3_v, 384, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "MultiSparseMatMul");

  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, d, n, m, sparsity_1, sparsity_2, true,
                                     false);
    SparseMatMulHelper<float, float>(g, m, d, n, sparsity_2, 0, false, true);
  }
  return g;
}

// clang-format off
// NOLINTBEGIN
#define BM_SPARSE_MULTI(M, K, N, S1, S2, Copies)                              \
  static void BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies(::testing::benchmark::State& state) {                                              \
    auto label = strings::Printf("%d_%d_%d_%d_%0.2f_%0.2f", M, K, N, Copies,  \
                                 S1 / 100.0, S2 / 100.0);                     \
    state.SetLabel(label);                                                    \
    auto g = MultiSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, Copies);      \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);        \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2 * 2 * Copies); \
  }                                                                           \
  BENCHMARK(BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies)       \
      ->UseRealTime();
// NOLINTEND
// clang-format on
BM_SPARSE_MULTI(1024, 2140, 4096, 0, 82, 1);
BM_SPARSE_MULTI(1024, 4096, 2048, 83, 83, 1);
BM_SPARSE_MULTI(400, 800, 2560, 85, 85, 1);
BM_SPARSE_MULTI(400, 2560, 1024, 85, 85, 1);
BM_SPARSE_MULTI(400, 1024, 256, 85, 85, 1);
BM_SPARSE_MULTI(400, 256, 1, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 1);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 1);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 1);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 3);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 3);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 3);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 3);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 6);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 6);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 6);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 6);

}  // end namespace tensorflow

namespace Eigen {
namespace internal {

class SparseMatmulOpTest : public ::testing::Test {
 protected:
  SparseMatmulOpTest()
      : PacketSize(Eigen::internal::packet_traits<float>::size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_4(mht_4_v, 442, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "SparseMatmulOpTest");

    typedef typename NumTraits<float>::Real RealFloat;

    for (int i = 0; i < kMaxPacketSize; ++i) {
      data1[i] = internal::random<float>() / RealFloat(PacketSize);
      data2[i] = internal::random<float>() / RealFloat(PacketSize);
      data3[i] = internal::random<float>() / RealFloat(PacketSize);
    }
    for (int i = kMaxPacketSize; i < kMaxPacketSize * 2; ++i) {
      data3[i] = internal::random<float>() / RealFloat(PacketSize);
    }

    // zero out lower 16-bits of mantissa of data3 values
    // copy bfloat representation to data3_bfloat16
    for (int i = 0; i < kMaxPacketSize * 2; ++i) {
      uint16_t* data3_p = reinterpret_cast<uint16_t*>(&data3[i]);
      uint16_t* data3_bfloat16_p =
          reinterpret_cast<uint16_t*>(data3_bfloat16) + i;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      data3_p[1] = 0;
      data3_bfloat16_p[0] = data3_p[0];
#else
      data3_p[0] = 0;
      data3_bfloat16_p[0] = data3_p[1];
#endif
    }
  }

  bool areApprox(const float* a, const float* b, int size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_op_testDTcc mht_5(mht_5_v, 473, "", "./tensorflow/core/kernels/sparse_matmul_op_test.cc", "areApprox");

    for (int i = 0; i < size; ++i) {
      if (a[i] != b[i] && !internal::isApprox(a[i], b[i])) {
        auto ma = Map<const Matrix<float, 1, Dynamic> >(a, size);
        auto mb = Map<const Matrix<float, 1, Dynamic> >(b, size);
        std::cout << "[" << ma << "]"
                  << " != [" << mb << "], differences: [" << (mb - ma) << "]\n";
        return false;
      }
    }
    return true;
  }

#ifdef EIGEN_VECTORIZE_AVX512
  static const int kMaxPacketSize = 16;
#elif defined EIGEN_VECTORIZE_AVX || defined EIGEN_VECTORIZE_AVX2
  static const int kMaxPacketSize = 8;
#else
  static constexpr int kMaxPacketSize = 4;
#endif
  typedef typename Eigen::internal::packet_traits<float>::type Packet;
  const int PacketSize;
  // float values
  EIGEN_ALIGN_MAX float data1[kMaxPacketSize];
  // output of intrinsics
  EIGEN_ALIGN_MAX float data2[kMaxPacketSize];
  // float values with only 7 mantissa bits (bfloat representable)
  EIGEN_ALIGN_MAX float data3[kMaxPacketSize * 2];
  // bfloat16 representation of data3
  EIGEN_ALIGN_MAX float data3_bfloat16[kMaxPacketSize];
  EIGEN_ALIGN_MAX float ref[kMaxPacketSize];

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

TEST_F(SparseMatmulOpTest, BroadcastPacketTest) {
  for (int i = 0; i < PacketSize; ++i) ref[i] = data1[0];
  internal::pstoreu(data2, internal::pbroadcast_first<Packet>(
                               internal::ploadu<Packet>(data1)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
  if (PacketSize > 1) {
    for (int i = 0; i < PacketSize; ++i) ref[i] = data1[1];
    internal::pstoreu(data2, internal::pbroadcast_second<Packet>(
                                 internal::ploadu<Packet>(data1)));
    ASSERT_TRUE(areApprox(ref, data2, PacketSize));

    if (PacketSize > 2) {
      for (int i = 0; i < PacketSize; ++i) ref[i] = data1[2];
      internal::pstoreu(data2, internal::pbroadcast_third<Packet>(
                                   internal::ploadu<Packet>(data1)));
      ASSERT_TRUE(areApprox(ref, data2, PacketSize));

      if (PacketSize > 3) {
        for (int i = 0; i < PacketSize; ++i) ref[i] = data1[3];
        internal::pstoreu(data2, internal::pbroadcast_fourth<Packet>(
                                     internal::ploadu<Packet>(data1)));
        ASSERT_TRUE(areApprox(ref, data2, PacketSize));
      }
    }
  }
}

TEST_F(SparseMatmulOpTest, InterleavePacketTest) {
  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 4; ++i) ref[i] = data1[i];
    for (int i = PacketSize / 4; i < PacketSize / 2; ++i)
      ref[i] = data1[i + PacketSize / 4];
    for (int i = PacketSize / 2; i < 3 * PacketSize / 4; ++i)
      ref[i] = data1[i - PacketSize / 4];
    for (int i = 3 * PacketSize / 4; i < PacketSize; ++i) ref[i] = data1[i];
  } else {
    // No interleaving done for smaller packets
    for (int i = 0; i < PacketSize; ++i) ref[i] = data1[i];
  }

  internal::pstoreu(data2, internal::pinterleave4x64<Packet>(
                               internal::ploadu<Packet>(data1)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
}

TEST_F(SparseMatmulOpTest, Bfloat16ExpandTest) {
  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i] = data3[i];
    }
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i + PacketSize / 2] = data3[i + PacketSize];
    }
  } else {
    for (int i = 0; i < PacketSize; ++i) {
      ref[i] = data3[i];
    }
  }
  internal::pstoreu(data2, internal::pexpand_bf16_l<Packet>(
                               internal::ploadu<Packet>(data3_bfloat16)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));

  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i] = data3[i + PacketSize / 2];
    }
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i + PacketSize / 2] = data3[i + 3 * PacketSize / 2];
    }
  } else {
    for (int i = 0; i < PacketSize; ++i) {
      ref[i] = data3[i + PacketSize];
    }
  }

  internal::pstoreu(data2, internal::pexpand_bf16_u<Packet>(
                               internal::ploadu<Packet>(data3_bfloat16)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
}

TEST_F(SparseMatmulOpTest, Bfloat16LoadTest) {
  if (PacketSize >= 4) {
    for (int i = 0; i < 4; ++i) ref[i] = data3[i];
    internal::pstoreu(data2, internal::pload4bf16<Packet>(data3_bfloat16));
    ASSERT_TRUE(areApprox(ref, data2, 4));

    internal::pstoreu(data2, internal::pload2bf16<Packet>(data3_bfloat16));
    ASSERT_TRUE(areApprox(ref, data2, 2));
  }
}

}  // namespace internal
}  // namespace Eigen
