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
class MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc() {
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

#include "tensorflow/core/graph/algorithm.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");
REGISTER_OP("TestUnary").Input("a: float").Output("o: float");
REGISTER_OP("TestBinary")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float");

// Compares that the order of nodes in 'inputs' respects the
// pair orders described in 'ordered_pairs'.
bool ExpectBefore(const std::vector<std::pair<string, string>>& ordered_pairs,
                  const std::vector<Node*>& inputs, string* error) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/graph/algorithm_test.cc", "ExpectBefore");

  for (const std::pair<string, string>& pair : ordered_pairs) {
    const string& before_node = pair.first;
    const string& after_node = pair.second;
    bool seen_before = false;
    bool seen_both = false;
    for (const Node* node : inputs) {
      if (!seen_before && after_node == node->name()) {
        *error = strings::StrCat("Saw ", after_node, " before ", before_node);
        return false;
      }

      if (before_node == node->name()) {
        seen_before = true;
      } else if (after_node == node->name()) {
        seen_both = seen_before;
        break;
      }
    }
    if (!seen_both) {
      *error = strings::StrCat("didn't see either ", before_node, " or ",
                               after_node);
      return false;
    }
  }

  return true;
}

TEST(AlgorithmTest, ReversePostOrder) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* w1 = SourceOp("TestParams", b.opts().WithName("W1"));
  Node* w2 = SourceOp("TestParams", b.opts().WithName("W2"));
  Node* input =
      SourceOp("TestInput", b.opts().WithName("input").WithControlInput(w1));
  Node* t1 = BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t1"));
  BinaryOp("TestMul", w1, {input, 1},
           b.opts().WithName("t2").WithControlInput(t1));
  BinaryOp("TestMul", w2, {input, 1}, b.opts().WithName("t3"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));
  std::vector<Node*> order;

  // Test reverse post order:
  GetReversePostOrder(g, &order);

  // Check that the order respects the dependencies correctly.
  std::vector<std::pair<string, string>> reverse_orders = {
      {"W1", "input"}, {"W1", "t1"},    {"W1", "t2"}, {"W1", "t3"},
      {"input", "t1"}, {"input", "t3"}, {"t1", "t2"}, {"W2", "t3"}};
  string error;
  EXPECT_TRUE(ExpectBefore(reverse_orders, order, &error)) << error;

  // A false ordering should fail the check.
  reverse_orders = {{"input", "W1"}};
  EXPECT_FALSE(ExpectBefore(reverse_orders, order, &error));

  // Test post order:
  GetPostOrder(g, &order);

  // Check that the order respects the dependencies correctly.
  std::vector<std::pair<string, string>> orders = {
      {"input", "W1"}, {"t1", "W1"},    {"t2", "W1"}, {"t3", "W1"},
      {"t1", "input"}, {"t3", "input"}, {"t2", "t1"}, {"t3", "W2"}};
  EXPECT_TRUE(ExpectBefore(orders, order, &error)) << error;

  // A false ordering should fail the check.
  orders = {{"W1", "t3"}};
  EXPECT_FALSE(ExpectBefore(orders, order, &error));
}

TEST(AlgorithmTest, ReversePostOrderStable) {
  int64_t run_count = 100;
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  for (int64_t i = 0; i < run_count; ++i) {
    // One source of nondeterminism comes from unordered set with key of a
    // pointer type, for example the order of FlatSet<Node*> depends on the
    // raw pointer value of Node. Stable post order suppose to remove this
    // nondeterminism by enforcing an ordering based on node ids.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    string error;
    Node* w1 = SourceOp("TestParams", b.opts().WithName("W1"));
    Node* input =
        SourceOp("TestInput", b.opts().WithName("input").WithControlInput(w1));
    BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t2"));
    // Insert different number of nodes between the allocation of t2 and t3,
    // this creates enough entropy in the memory distance between t2 and t3 thus
    // forces them to have randomized ordering had stable DFS was not
    // implemented correctly.
    for (int64_t j = 0; j < i; ++j) {
      BinaryOp("TestMul", w1, {input, 1},
               b.opts().WithName(strings::StrCat("internal", j)));
    }

    BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t3"));

    Graph g(OpRegistry::Global());
    TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));
    std::vector<Node*> order;

    // Test reverse post order generates expected ordering.
    GetReversePostOrder(g, &order, /*stable_comparator=*/NodeComparatorName());
    EXPECT_TRUE(ExpectBefore({{"t2", "t3"}}, order, &error));
  }
}

TEST(AlgorithmTest, PostOrderWithEdgeFilter) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Node* n0 = ops::SourceOp("TestParams", b.opts().WithName("n0"));
  Node* n1 = ops::UnaryOp("TestUnary", n0, b.opts().WithName("n1"));
  Node* n2 = ops::UnaryOp("TestUnary", n1, b.opts().WithName("n2"));
  Node* n3 = ops::BinaryOp("TestBinary", n2, n0, b.opts().WithName("n3"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  g.AddEdge(g.FindNodeId(n3->id()), 0, g.FindNodeId(n1->id()), 1);

  std::vector<Node*> post_order;
  auto edge_filter = [&](const Edge& e) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc mht_1(mht_1_v, 345, "", "./tensorflow/core/graph/algorithm_test.cc", "lambda");

    return !(e.src()->id() == n3->id() && e.dst()->id() == n1->id());
  };

  std::vector<Node*> expected_post_order = {
      g.sink_node(),          g.FindNodeId(n3->id()), g.FindNodeId(n2->id()),
      g.FindNodeId(n1->id()), g.FindNodeId(n0->id()), g.source_node()};

  std::vector<Node*> expected_reverse_post_order = expected_post_order;
  std::reverse(expected_reverse_post_order.begin(),
               expected_reverse_post_order.end());

  GetPostOrder(g, &post_order, /*stable_comparator=*/{},
               /*edge_filter=*/edge_filter);

  ASSERT_EQ(expected_post_order.size(), post_order.size());
  for (int i = 0; i < post_order.size(); i++) {
    CHECK_EQ(post_order[i], expected_post_order[i])
        << post_order[i]->name() << " vs. " << expected_post_order[i]->name();
  }

  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(g, &reverse_post_order, /*stable_comparator=*/{},
                      /*edge_filter=*/edge_filter);

  ASSERT_EQ(expected_reverse_post_order.size(), reverse_post_order.size());
  for (int i = 0; i < reverse_post_order.size(); i++) {
    CHECK_EQ(reverse_post_order[i], expected_reverse_post_order[i])
        << reverse_post_order[i]->name() << " vs. "
        << expected_reverse_post_order[i]->name();
  }
}

void BM_PruneForReverseReachability(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSalgorithm_testDTcc mht_2(mht_2_v, 381, "", "./tensorflow/core/graph/algorithm_test.cc", "BM_PruneForReverseReachability");

  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  const auto registry = OpRegistry::Global();
  GraphConstructorOptions opts;
  for (auto s : state) {
    state.PauseTiming();
    Graph graph(registry);
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
    std::unordered_set<const Node*> visited;
    visited.insert(graph.FindNodeId(graph.num_nodes() - 1));
    state.ResumeTiming();
    PruneForReverseReachability(&graph, std::move(visited));
  }
}
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 16);

}  // namespace
}  // namespace tensorflow
