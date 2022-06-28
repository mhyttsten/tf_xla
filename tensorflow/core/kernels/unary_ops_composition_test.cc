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
class MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class UnaryOpsCompositionTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunComposedOp(const std::vector<string> op_names, T input, T expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/unary_ops_composition_test.cc", "RunComposedOp");

    TF_ASSERT_OK(NodeDefBuilder("unary_op_composition", "_UnaryOpsComposition")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("op_names", op_names)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    TensorShape shape({});
    AddInputFromArray<T>(shape, {input});

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(allocator(), DataTypeToEnum<T>::value, shape);
    test::FillValues<T>(&expected_tensor, {expected});
    test::ExpectClose(expected_tensor, *GetOutput(0));
  }
};

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_F) {
  RunComposedOp<float>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_D) {
  RunComposedOp<double>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sin_F) {
  RunComposedOp<float>({"Sqrt", "Sin"}, 81.0, std::sin(9.0f));
}

TEST_F(UnaryOpsCompositionTest, Compose_Cos_Acos_F) {
  RunComposedOp<float>({"Cos", "Acos"}, 0.5, std::acos(std::cos(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_F) {
  RunComposedOp<float>({"Tanh", "Relu"}, 0.5, std::max(0.0f, std::tanh(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_D) {
  RunComposedOp<double>({"Tanh", "Relu"}, 0.5, std::max(0.0, std::tanh(0.5)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu6_F) {
  RunComposedOp<float>({"Relu6"}, 11.0f, 6.0f);
}

// Performance benchmarks below.

string Function(int i) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc mht_1(mht_1_v, 257, "", "./tensorflow/core/kernels/unary_ops_composition_test.cc", "Function");

  std::vector<string> ops = {"Tanh", "Relu", "Sigmoid", "Sqrt", "Log", "Exp"};
  return ops[i % ops.size()];
}

// Unary ops chained together as a separate graph nodes.
static Graph* UnaryOpsChain(int tensor_size, int repeat_graph,
                            int num_functions) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/kernels/unary_ops_composition_test.cc", "UnaryOpsChain");

  Graph* g = new Graph(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape({tensor_size}));
  t.flat<float>() = t.flat<float>().setRandom();

  for (int i = 0; i < repeat_graph; ++i) {
    Node* node = test::graph::Constant(g, t);
    for (int j = 0; j < num_functions; ++j) {
      TF_CHECK_OK(NodeBuilder(g->NewName("n"), Function(j))
                      .Input(node)
                      .Attr("T", DT_FLOAT)
                      .Finalize(g, &node));
    }
  }

  return g;
}

#define BM_UnaryOpsChain(N, R, F, type)                                        \
  static void BM_UnaryOpsChain##_##type##_##N##_##R##_##F(                     \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#type, UnaryOpsChain(N, R, F),                             \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * R * \
                            F);                                                \
  }                                                                            \
  BENCHMARK(BM_UnaryOpsChain##_##type##_##N##_##R##_##F);

// Unary ops fused together.
static Graph* UnaryOpsCompo(int tensor_size, int repeat_graph,
                            int num_functions) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunary_ops_composition_testDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/kernels/unary_ops_composition_test.cc", "UnaryOpsCompo");

  Graph* g = new Graph(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape({tensor_size}));
  t.flat<float>() = t.flat<float>().setRandom();

  std::vector<string> functions;
  for (int j = 0; j < num_functions; ++j) {
    functions.push_back(Function(j));
  }

  for (int i = 0; i < repeat_graph; ++i) {
    Node* node = test::graph::Constant(g, t);
    TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_UnaryOpsComposition")
                    .Input(node)
                    .Attr("T", DT_FLOAT)
                    .Attr("op_names", functions)
                    .Finalize(g, &node));
  }

  return g;
}

#define BM_UnaryOpsCompo(N, R, F, type)                                        \
  static void BM_UnaryOpsCompo##_##type##_##N##_##R##_##F(                     \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#type, UnaryOpsCompo(N, R, F),                             \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * R * \
                            F);                                                \
  }                                                                            \
  BENCHMARK(BM_UnaryOpsCompo##_##type##_##N##_##R##_##F);

// BenchmarkName(tensor_size, repeat_graph, num_ops, type)

BM_UnaryOpsChain(1000, 25, 2, cpu);
BM_UnaryOpsCompo(1000, 25, 2, cpu);

BM_UnaryOpsChain(1000, 25, 5, cpu);
BM_UnaryOpsCompo(1000, 25, 5, cpu);

BM_UnaryOpsChain(1000, 25, 10, cpu);
BM_UnaryOpsCompo(1000, 25, 10, cpu);

BM_UnaryOpsChain(100000, 25, 2, cpu);
BM_UnaryOpsCompo(100000, 25, 2, cpu);

BM_UnaryOpsChain(100000, 25, 5, cpu);
BM_UnaryOpsCompo(100000, 25, 5, cpu);

BM_UnaryOpsChain(100000, 25, 10, cpu);
BM_UnaryOpsCompo(100000, 25, 10, cpu);

BM_UnaryOpsChain(1000000, 25, 2, cpu);
BM_UnaryOpsCompo(1000000, 25, 2, cpu);

BM_UnaryOpsChain(1000000, 25, 5, cpu);
BM_UnaryOpsCompo(1000000, 25, 5, cpu);

BM_UnaryOpsChain(1000000, 25, 10, cpu);
BM_UnaryOpsCompo(1000000, 25, 10, cpu);

}  // namespace
}  // end namespace tensorflow
