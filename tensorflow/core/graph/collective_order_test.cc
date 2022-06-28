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
class MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc() {
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
#include "tensorflow/core/graph/collective_order.h"

#include <gmock/gmock.h>
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::UnorderedElementsAreArray;

REGISTER_OP("TestParams").Output("o: float");

// Verifies that the list of collective nodes in `graph` matches
// `expected_collective_nodes`, and that the list of control edges between these
// collective nodes matches `expected_collective_control_edges`.
void VerifyGraph(const Graph& graph,
                 const std::vector<string>& expected_collective_nodes,
                 const std::vector<std::pair<string, string>>&
                     expected_collective_control_edges) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/graph/collective_order_test.cc", "VerifyGraph");

  std::vector<string> actual_collective_nodes;
  std::vector<std::pair<string, string>> actual_collective_control_edges;
  for (const Node* src : graph.nodes()) {
    if (!src->IsCollective()) {
      continue;
    }
    actual_collective_nodes.push_back(src->name());
    for (const Edge* edge : src->out_edges()) {
      VLOG(2) << "collective edge " << edge->src()->name() << " -> "
              << edge->dst()->name();
      // Add all control edges found except those to `_SINK`.
      if (!edge->IsControlEdge() || edge->dst()->name() == "_SINK") {
        continue;
      }
      actual_collective_control_edges.emplace_back(src->name(),
                                                   edge->dst()->name());
    }
  }
  EXPECT_THAT(actual_collective_nodes,
              UnorderedElementsAreArray(expected_collective_nodes));
  EXPECT_THAT(actual_collective_control_edges,
              UnorderedElementsAreArray(expected_collective_control_edges));
}

// Verifies that the `wait_for` attribute on collective nodes matches
// `wait_for_map`.
void VerifyAttrs(
    const Graph& graph,
    const std::unordered_map<string, std::vector<int32>> wait_for_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/graph/collective_order_test.cc", "VerifyAttrs");

  for (const Node* node : graph.nodes()) {
    if (node->IsCollective() ||
        wait_for_map.find(node->name()) == wait_for_map.end()) {
      continue;
    }
    std::vector<int32> wait_for_actual;
    TF_EXPECT_OK(GetNodeAttr(node->attrs(), "wait_for", &wait_for_actual));
    auto wait_for_expected = wait_for_map.at(node->name());
    EXPECT_THAT(wait_for_actual, UnorderedElementsAreArray(wait_for_expected));
  }
}

Node* CollectiveReduceNode(GraphDefBuilder* builder, Node* input,
                           const string& name, const string& device,
                           int instance_key) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPScollective_order_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/graph/collective_order_test.cc", "CollectiveReduceNode");

  Node* collective_node =
      ops::UnaryOp("CollectiveReduce", input,
                   builder->opts()
                       .WithName(name)
                       .WithDevice(device)
                       .WithAttr("T", DT_FLOAT)
                       .WithAttr("group_size", 2)
                       .WithAttr("group_key", 1)
                       .WithAttr("instance_key", instance_key)
                       .WithAttr("merge_op", "Add")
                       .WithAttr("final_op", "Id")
                       .WithAttr("subdiv_offsets", {1}));
  return collective_node;
}

// Initialize the following graph:
//
//       (cpu0) (cpu1)
//         a      b
//         |      |
//         c1     c1
//         |      |
//         id     id
//        /  \   /  \
//       c2  c3 c2  c3
//
// Here ci denotes a collective node with `instance_key` i.  `a` and `b` are
// inputs, `id` is identity node.
std::unique_ptr<Graph> InitGraph() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  const string dev1 = "/job:localhost/replica:0/task:0/device:CPU:1";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* b = ops::SourceOp("TestParams",
                          builder.opts().WithName("b").WithDevice(dev1));
  Node* c1_0 = CollectiveReduceNode(&builder, a, "c1_0", dev0, 1);
  Node* c1_1 = CollectiveReduceNode(&builder, b, "c1_1", dev1, 1);
  Node* id0 = ops::UnaryOp(
      "Identity", c1_0,
      builder.opts().WithName("id0").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  Node* id1 = ops::UnaryOp(
      "Identity", c1_1,
      builder.opts().WithName("id1").WithDevice(dev1).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id0, "c2_0", dev0, 2);
  CollectiveReduceNode(&builder, id1, "c2_1", dev1, 2);
  CollectiveReduceNode(&builder, id0, "c3_0", dev0, 3);
  CollectiveReduceNode(&builder, id1, "c3_1", dev1, 3);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraph`, exactly 2 control edges are
// added after calling `OrderCollectives`: c3_0 -> c2_0 and c3_1 -> c2_1.
TEST(CollectiveOrderTest, SimpleOrder) {
  std::unique_ptr<Graph> graph = InitGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  VerifyGraph(*graph, {"c1_0", "c1_1", "c2_0", "c2_1", "c3_0", "c3_1"},
              {{"c3_0", "c2_0"}, {"c3_1", "c2_1"}});
}

TEST(CollectiveOrderTest, SimpleOrderAttr) {
  std::unique_ptr<Graph> graph = InitGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kAttrs));
  VerifyAttrs(*graph, {{"c2_0", {3}}, {"c2_1", {3}}});
}

// Initialize the following graph:
//
//         a
//         |
//         c1
//        /  \
//       c4  id
//          /  \
//         c2  c3
//
// Here ci denotes a collective node with `instance_key` i.  `a` is an input,
// `id` is identity node.
std::unique_ptr<Graph> InitGraph2() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* c1 = CollectiveReduceNode(&builder, a, "c1", dev0, 1);
  CollectiveReduceNode(&builder, c1, "c4", dev0, 4);
  Node* id = ops::UnaryOp(
      "Identity", c1,
      builder.opts().WithName("id").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id, "c2", dev0, 2);
  CollectiveReduceNode(&builder, id, "c3", dev0, 3);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraph2`, we add the following control
// edges after calling `OrderCollectives`: c4 -> c3, c3 -> c2.  c4->c2 is
// pruned because it follows from the other two edges.
TEST(CollectiveOrderTest, SimpleOrder2) {
  std::unique_ptr<Graph> graph = InitGraph2();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  VerifyGraph(*graph, {"c1", "c2", "c3", "c4"}, {{"c4", "c3"}, {"c3", "c2"}});
}

// Initialize the following graph:
//
//         w   x   y   z
//         |   |   |   |
//         c1  c2  c3  c4
//
std::unique_ptr<Graph> InitGraphForPruning() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* w = ops::SourceOp("TestParams",
                          builder.opts().WithName("w").WithDevice(dev0));
  Node* x = ops::SourceOp("TestParams",
                          builder.opts().WithName("x").WithDevice(dev0));
  Node* y = ops::SourceOp("TestParams",
                          builder.opts().WithName("y").WithDevice(dev0));
  Node* z = ops::SourceOp("TestParams",
                          builder.opts().WithName("z").WithDevice(dev0));
  CollectiveReduceNode(&builder, w, "c1", dev0, 1);
  CollectiveReduceNode(&builder, x, "c2", dev0, 2);
  CollectiveReduceNode(&builder, y, "c3", dev0, 3);
  CollectiveReduceNode(&builder, z, "c4", dev0, 4);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraphForPruning`, we only add c4 ->
// c3, c3 -> c2, c2 -> c1, and other edges are pruned away.
TEST(CollectiveOrderTest, Pruning) {
  std::unique_ptr<Graph> graph = InitGraphForPruning();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kAttrs));
  VerifyAttrs(*graph, {{"c3", {4}}, {"c2", {3}}, {"c1", {2}}});
}

}  // namespace
}  // namespace tensorflow
