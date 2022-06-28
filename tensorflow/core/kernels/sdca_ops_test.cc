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
class MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc() {
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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

const SessionOptions* GetSingleThreadedOptions() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "GetSingleThreadedOptions");

  static const SessionOptions* const kSessionOptions = []() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "lambda");

    SessionOptions* const result = new SessionOptions();
    result->config.set_intra_op_parallelism_threads(1);
    result->config.set_inter_op_parallelism_threads(1);
    result->config.add_session_inter_op_thread_pool()->set_num_threads(1);
    return result;
  }();
  return kSessionOptions;
}

const SessionOptions* GetMultiThreadedOptions() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "GetMultiThreadedOptions");

  static const SessionOptions* const kSessionOptions = []() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_3(mht_3_v, 219, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "lambda");

    SessionOptions* const result = new SessionOptions();
    result->config.set_intra_op_parallelism_threads(0);  // Auto-configured.
    result->config.set_inter_op_parallelism_threads(0);  // Auto-configured.
    result->config.add_session_inter_op_thread_pool()->set_num_threads(
        0);  // Auto-configured.
    return result;
  }();
  return kSessionOptions;
}

Node* Var(Graph* const g, const int n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "Var");

  return test::graph::Var(g, DT_FLOAT, TensorShape({n}));
}

// Returns a vector of size 'nodes' with each node being of size 'node_size'.
std::vector<Node*> VarVector(Graph* const g, const int nodes,
                             const int node_size) {
  std::vector<Node*> result;
  result.reserve(nodes);
  for (int i = 0; i < nodes; ++i) {
    result.push_back(Var(g, node_size));
  }
  return result;
}

Node* Zeros(Graph* const g, const TensorShape& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "Zeros");

  Tensor data(DT_FLOAT, shape);
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

Node* Zeros(Graph* const g, const int n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_6(mht_6_v, 260, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "Zeros");
 return Zeros(g, TensorShape({n})); }

Node* Ones(Graph* const g, const int n) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_7(mht_7_v, 265, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "Ones");

  Tensor data(DT_FLOAT, TensorShape({n}));
  test::FillFn<float>(&data, [](const int i) { return 1.0f; });
  return test::graph::Constant(g, data);
}

Node* SparseIndices(Graph* const g, const int sparse_features_per_group) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_8(mht_8_v, 274, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "SparseIndices");

  Tensor data(DT_INT64, TensorShape({sparse_features_per_group}));
  test::FillFn<int64_t>(&data, [&](const int i) { return i; });
  return test::graph::Constant(g, data);
}

Node* SparseExampleIndices(Graph* const g, const int sparse_features_per_group,
                           const int num_examples) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_9(mht_9_v, 284, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "SparseExampleIndices");

  const int x_size = num_examples * 4;
  Tensor data(DT_INT64, TensorShape({x_size}));
  test::FillFn<int64_t>(&data, [&](const int i) { return i / 4; });
  return test::graph::Constant(g, data);
}

Node* SparseFeatureIndices(Graph* const g, const int sparse_features_per_group,
                           const int num_examples) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_10(mht_10_v, 295, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "SparseFeatureIndices");

  const int x_size = num_examples * 4;
  Tensor data(DT_INT64, TensorShape({x_size}));
  test::FillFn<int64_t>(
      &data, [&](const int i) { return i % sparse_features_per_group; });
  return test::graph::Constant(g, data);
}

Node* RandomZeroOrOne(Graph* const g, const int n) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_11(mht_11_v, 306, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "RandomZeroOrOne");

  Tensor data(DT_FLOAT, TensorShape({n}));
  test::FillFn<float>(&data, [](const int i) {
    // Fill with 0.0 or 1.0 at random.
    return (random::New64() % 2) == 0 ? 0.0f : 1.0f;
  });
  return test::graph::Constant(g, data);
}

Node* RandomZeroOrOneMatrix(Graph* const g, const int n, int d) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_12(mht_12_v, 318, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "RandomZeroOrOneMatrix");

  Tensor data(DT_FLOAT, TensorShape({n, d}));
  test::FillFn<float>(&data, [](const int i) {
    // Fill with 0.0 or 1.0 at random.
    return (random::New64() % 2) == 0 ? 0.0f : 1.0f;
  });
  return test::graph::Constant(g, data);
}

void GetGraphs(const int32_t num_examples,
               const int32_t num_sparse_feature_groups,
               const int32_t sparse_features_per_group,
               const int32_t num_dense_feature_groups,
               const int32_t dense_features_per_group, Graph** const init_g,
               Graph** train_g) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_13(mht_13_v, 335, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "GetGraphs");

  {
    // Build initialization graph
    Graph* g = new Graph(OpRegistry::Global());

    // These nodes have to be created first, and in the same way as the
    // nodes in the graph below.
    std::vector<Node*> sparse_weight_nodes =
        VarVector(g, num_sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, num_dense_feature_groups, dense_features_per_group);
    Node* const multi_zero = Zeros(g, sparse_features_per_group);
    for (Node* n : sparse_weight_nodes) {
      test::graph::Assign(g, n, multi_zero);
    }
    Node* const zero = Zeros(g, dense_features_per_group);
    for (Node* n : dense_weight_nodes) {
      test::graph::Assign(g, n, zero);
    }

    *init_g = g;
  }

  {
    // Build execution graph
    Graph* g = new Graph(OpRegistry::Global());

    // These nodes have to be created first, and in the same way as the
    // nodes in the graph above.
    std::vector<Node*> sparse_weight_nodes =
        VarVector(g, num_sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, num_dense_feature_groups, dense_features_per_group);

    std::vector<NodeBuilder::NodeOut> sparse_indices;
    std::vector<NodeBuilder::NodeOut> sparse_weights;
    for (Node* n : sparse_weight_nodes) {
      sparse_indices.push_back(
          NodeBuilder::NodeOut(SparseIndices(g, sparse_features_per_group)));
      sparse_weights.push_back(NodeBuilder::NodeOut(n));
    }
    std::vector<NodeBuilder::NodeOut> dense_weights;
    dense_weights.reserve(dense_weight_nodes.size());
    for (Node* n : dense_weight_nodes) {
      dense_weights.push_back(NodeBuilder::NodeOut(n));
    }

    std::vector<NodeBuilder::NodeOut> sparse_example_indices;
    std::vector<NodeBuilder::NodeOut> sparse_feature_indices;
    std::vector<NodeBuilder::NodeOut> sparse_values;
    sparse_example_indices.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_example_indices.push_back(NodeBuilder::NodeOut(
          SparseExampleIndices(g, sparse_features_per_group, num_examples)));
    }
    sparse_feature_indices.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_feature_indices.push_back(NodeBuilder::NodeOut(
          SparseFeatureIndices(g, sparse_features_per_group, num_examples)));
    }
    sparse_values.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_values.push_back(
          NodeBuilder::NodeOut(RandomZeroOrOne(g, num_examples * 4)));
    }

    std::vector<NodeBuilder::NodeOut> dense_features;
    dense_features.reserve(num_dense_feature_groups);
    for (int i = 0; i < num_dense_feature_groups; ++i) {
      dense_features.push_back(NodeBuilder::NodeOut(
          RandomZeroOrOneMatrix(g, num_examples, dense_features_per_group)));
    }

    Node* const weights = Ones(g, num_examples);
    Node* const labels = RandomZeroOrOne(g, num_examples);
    Node* const example_state_data = Zeros(g, TensorShape({num_examples, 4}));

    Node* sdca = nullptr;
    TF_CHECK_OK(
        NodeBuilder(g->NewName("sdca"), "SdcaOptimizer")
            .Attr("loss_type", "logistic_loss")
            .Attr("num_sparse_features", num_sparse_feature_groups)
            .Attr("num_sparse_features_with_values", num_sparse_feature_groups)
            .Attr("num_dense_features", num_dense_feature_groups)
            .Attr("l1", 0.0)
            .Attr("l2", 1.0)
            .Attr("num_loss_partitions", 1)
            .Attr("num_inner_iterations", 2)
            .Input(sparse_example_indices)
            .Input(sparse_feature_indices)
            .Input(sparse_values)
            .Input(dense_features)
            .Input(weights)
            .Input(labels)
            .Input(sparse_indices)
            .Input(sparse_weights)
            .Input(dense_weights)
            .Input(example_state_data)
            .Finalize(g, &sdca));

    *train_g = g;
  }
}

void BM_SDCA(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_14(mht_14_v, 442, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "BM_SDCA");

  const int num_examples = state.range(0);
  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 20 /* sparse feature groups */,
            5 /* sparse features per group */, 1 /* dense feature groups*/,
            20 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetSingleThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}

void BM_SDCA_LARGE_DENSE(::testing::benchmark::State& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_15(mht_15_v, 457, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "BM_SDCA_LARGE_DENSE");

  const int num_examples = state.range(0);

  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 0 /* sparse feature groups */,
            0 /* sparse features per group */, 5 /* dense feature groups*/,
            200000 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetSingleThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}

void BM_SDCA_LARGE_SPARSE(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsdca_ops_testDTcc mht_16(mht_16_v, 473, "", "./tensorflow/core/kernels/sdca_ops_test.cc", "BM_SDCA_LARGE_SPARSE");

  const int num_examples = state.range(0);

  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 65 /* sparse feature groups */,
            1e6 /* sparse features per group */, 0 /* dense feature groups*/,
            0 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetMultiThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}
}  // namespace

BENCHMARK(BM_SDCA)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_SDCA_LARGE_DENSE)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_SDCA_LARGE_SPARSE)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

}  // namespace tensorflow
