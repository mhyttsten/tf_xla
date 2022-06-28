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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/linalg/matrix_set_diag_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Node* SetDiag(int num_bands, Graph* g, Node* bands, Node* triangular) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/linalg/banded_triangular_solve_op_test.cc", "SetDiag");

  Node* ret;
  Tensor bandwidth(DT_INT32, TensorShape({2}));
  bandwidth.flat<int32>()(0) = -(num_bands - 1);
  bandwidth.flat<int32>()(1) = 0;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatrixSetDiagV3")
                  .Input(triangular)
                  .Input(bands)
                  .Input(test::graph::Constant(g, bandwidth))
                  .Attr("align", "RIGHT_LEFT")
                  .Finalize(g, &ret));
  return ret;
}

Node* BandedTriangularSolve(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/linalg/banded_triangular_solve_op_test.cc", "BandedTriangularSolve");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BandedTriangularSolve")
                  .Input(in0)
                  .Input(in1)
                  .Attr("lower", true)
                  .Attr("adjoint", false)
                  .Finalize(g, &ret));
  return ret;
}

Node* MatrixTriangularSolve(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/linalg/banded_triangular_solve_op_test.cc", "MatrixTriangularSolve");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatrixTriangularSolve")
                  .Input(in0)
                  .Input(in1)
                  .Attr("lower", true)
                  .Attr("adjoint", false)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* BandedTriangularSolve(int64_t num_bands, int64_t n, int64_t m,
                                    bool use_banded_solver, DataType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSbanded_triangular_solve_op_testDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/kernels/linalg/banded_triangular_solve_op_test.cc", "BandedTriangularSolve");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, TensorShape({num_bands, n}));
  // Set diagonal to nonzero to guarantee invertibility.
  in0.flat<T>().setRandom();
  in0.flat<T>() =
      in0.flat<T>().abs() + in0.flat<T>().constant(static_cast<T>(0.5));
  Tensor in1(type, TensorShape({n, m}));
  in1.flat<T>().setRandom();
  if (use_banded_solver) {
    BandedTriangularSolve(g, test::graph::Constant(g, in0),
                          test::graph::Constant(g, in1));
  } else {
    // Create a zero tensor.
    Tensor in2(type, TensorShape({n, n}));
    in2.flat<T>().setZero();
    Node* triangular_matrix =
        SetDiag(num_bands, g, test::graph::Constant(g, in0),
                test::graph::Constant(g, in2));
    MatrixTriangularSolve(g, triangular_matrix, test::graph::Constant(g, in1));
  }
  return g;
}

// Macro arguments names: --------------------------------------------------- //
//   K: Number of bands
//   N: Inner dimension of LHS, Inner dimension of RHS.
//   M: Outer dimensions of RHS
//   BS: boolean indicating whether to use the banded solver
//    T: C++ type of scalars (e.g. float, std::complex)
//   TT: TensorFlow type of scalars (e.g. DT_FLOAT, DT_COMPLEX128
#define BM_BandedTriangularSolveDev(K, N, M, BS, T, TT, D)              \
  static void BM_BandedTriangularSolve##_##K##_##N##_##M##_##BS##_##TT( \
      ::testing::benchmark::State& state) {                             \
    test::Benchmark(#D, BandedTriangularSolve<T>(K, N, M, BS, TT),      \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * K * N + N * M);        \
  }                                                                     \
  BENCHMARK(BM_BandedTriangularSolve##_##K##_##N##_##M##_##BS##_##TT)   \
      ->UseRealTime();

#define BM_BandedTriangularSolve(K, N, M, BS, D)                \
  BM_BandedTriangularSolveDev(K, N, M, BS, float, DT_FLOAT, D); \
  BM_BandedTriangularSolveDev(K, N, M, BS, double, DT_DOUBLE, D);

// Small number of bands, few rhs
BM_BandedTriangularSolve(2, 32, 1, true, cpu);
BM_BandedTriangularSolve(2, 32, 1, false, cpu);
BM_BandedTriangularSolve(4, 32, 1, true, cpu);
BM_BandedTriangularSolve(4, 32, 1, false, cpu);
BM_BandedTriangularSolve(8, 32, 1, true, cpu);
BM_BandedTriangularSolve(8, 32, 1, false, cpu);
BM_BandedTriangularSolve(16, 32, 1, true, cpu);
BM_BandedTriangularSolve(16, 32, 1, false, cpu);
BM_BandedTriangularSolve(2, 128, 1, true, cpu);
BM_BandedTriangularSolve(2, 128, 1, false, cpu);
BM_BandedTriangularSolve(4, 128, 1, true, cpu);
BM_BandedTriangularSolve(4, 128, 1, false, cpu);
BM_BandedTriangularSolve(8, 128, 1, true, cpu);
BM_BandedTriangularSolve(8, 128, 1, false, cpu);
BM_BandedTriangularSolve(16, 128, 1, true, cpu);
BM_BandedTriangularSolve(16, 128, 1, false, cpu);
BM_BandedTriangularSolve(2, 512, 1, true, cpu);
BM_BandedTriangularSolve(2, 512, 1, false, cpu);
BM_BandedTriangularSolve(4, 512, 1, true, cpu);
BM_BandedTriangularSolve(4, 512, 1, false, cpu);
BM_BandedTriangularSolve(8, 512, 1, true, cpu);
BM_BandedTriangularSolve(8, 512, 1, false, cpu);
BM_BandedTriangularSolve(16, 512, 1, true, cpu);
BM_BandedTriangularSolve(16, 512, 1, false, cpu);

// Larger # rhs
BM_BandedTriangularSolve(2, 32, 32, true, cpu);
BM_BandedTriangularSolve(2, 32, 32, false, cpu);
BM_BandedTriangularSolve(4, 32, 32, true, cpu);
BM_BandedTriangularSolve(4, 32, 32, false, cpu);
BM_BandedTriangularSolve(8, 32, 32, true, cpu);
BM_BandedTriangularSolve(8, 32, 32, false, cpu);
BM_BandedTriangularSolve(16, 32, 32, true, cpu);
BM_BandedTriangularSolve(16, 32, 32, false, cpu);
BM_BandedTriangularSolve(2, 128, 128, true, cpu);
BM_BandedTriangularSolve(2, 128, 128, false, cpu);
BM_BandedTriangularSolve(4, 128, 128, true, cpu);
BM_BandedTriangularSolve(4, 128, 128, false, cpu);
BM_BandedTriangularSolve(8, 128, 128, true, cpu);
BM_BandedTriangularSolve(8, 128, 128, false, cpu);
BM_BandedTriangularSolve(16, 128, 128, true, cpu);
BM_BandedTriangularSolve(16, 128, 128, false, cpu);
BM_BandedTriangularSolve(2, 512, 512, true, cpu);
BM_BandedTriangularSolve(2, 512, 512, false, cpu);
BM_BandedTriangularSolve(4, 512, 512, true, cpu);
BM_BandedTriangularSolve(4, 512, 512, false, cpu);
BM_BandedTriangularSolve(8, 512, 512, true, cpu);
BM_BandedTriangularSolve(8, 512, 512, false, cpu);
BM_BandedTriangularSolve(16, 512, 512, true, cpu);
BM_BandedTriangularSolve(16, 512, 512, false, cpu);

BM_BandedTriangularSolve(2, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(2, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(4, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(4, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(8, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(8, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(16, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(16, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(32, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(32, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(64, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(64, 2048, 2048, false, cpu);

}  // namespace
}  // namespace tensorflow
