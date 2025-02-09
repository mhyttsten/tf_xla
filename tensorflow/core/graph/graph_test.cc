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
class MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc() {
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

#include "tensorflow/core/graph/graph.h"

#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

REGISTER_OP("OneInput").Input("x: float");

REGISTER_OP("OneOutput").Output("y: float");

REGISTER_OP("OneInputTwoOutputs")
    .Input("x: float")
    .Output("y: float")
    .Output("z: float");

REGISTER_OP("TwoInputsOneOutput")
    .Input("x: float")
    .Input("y: float")
    .Output("z: float");

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() : graph_(OpRegistry::Global()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/graph/graph_test.cc", "GraphTest");
}
  ~GraphTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/graph/graph_test.cc", "~GraphTest");
}

  static void VerifyNodes(Node* node, const std::vector<Node*>& expected_in,
                          const std::vector<Node*>& expected_out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/graph/graph_test.cc", "VerifyNodes");

    std::vector<Node*> in;
    for (const Edge* e : node->in_edges()) {
      in.push_back(e->src());
    }
    EXPECT_EQ(Stringify(expected_in), Stringify(in));

    std::vector<Node*> out;
    for (const Edge* e : node->out_edges()) {
      out.push_back(e->dst());
    }
    EXPECT_EQ(Stringify(expected_out), Stringify(out));
  }

  void VerifyGraphStats() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/graph/graph_test.cc", "VerifyGraphStats");

    int nodes = 0;
    for (const Node* n : graph_.nodes()) {
      VLOG(1) << n->id();
      ++nodes;
    }
    EXPECT_EQ(nodes, graph_.num_nodes());
    int edges = 0;
    for (const Edge* e : graph_.edges()) {
      VLOG(1) << e->id();
      ++edges;
    }
    EXPECT_EQ(edges, graph_.num_edges());
  }

  Node* AddNodeWithName(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_4(mht_4_v, 270, "", "./tensorflow/core/graph/graph_test.cc", "AddNodeWithName");

    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "NoOp").Finalize(&graph_, &node));
    return node;
  }

  Node* FromNodeDef(const string& name, const string& node_type,
                    int num_inputs) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   mht_5_v.push_back("node_type: \"" + node_type + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/graph/graph_test.cc", "FromNodeDef");

    auto builder = NodeDefBuilder(name, node_type);
    for (int i = 0; i < num_inputs; ++i) {
      builder = builder.Input(strings::StrCat("node_", i), i, DT_FLOAT);
    }

    NodeDef node_def;
    TF_CHECK_OK(builder.Finalize(&node_def));

    Status s;
    Node* node = graph_.AddNode(node_def, &s);
    TF_CHECK_OK(s);
    return node;
  }

  void FromGraphDef(const string& gdef_ascii) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("gdef_ascii: \"" + gdef_ascii + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/graph/graph_test.cc", "FromGraphDef");

    GraphDef gdef;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef));
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef, &graph_));
  }

  Node* FindNode(const string& name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_7(mht_7_v, 312, "", "./tensorflow/core/graph/graph_test.cc", "FindNode");

    for (Node* node : graph_.nodes()) {
      if (node->name() == name) return node;
    }
    LOG(FATAL) << name;
  }

  bool ControlEdgeExistsInGraphOrNodeDef(const Node* src, const Node* dst) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_8(mht_8_v, 322, "", "./tensorflow/core/graph/graph_test.cc", "ControlEdgeExistsInGraphOrNodeDef");

    for (const Edge* e : dst->in_edges()) {
      if (e->IsControlEdge() && e->src() == src &&
          e->src_output() == Graph::kControlSlot &&
          e->dst_input() == Graph::kControlSlot) {
        return true;
      }
    }
    std::string control_edge_name = strings::StrCat("^", src->name());
    for (int i = 0; i < dst->def().input_size(); ++i) {
      if (dst->def().input(i) == control_edge_name) {
        return true;
      }
    }
    return false;
  }

  Graph graph_;

 private:
  // Convert a list of nodes to a sorted list of strings so failure messages
  // are readable.
  static std::vector<string> Stringify(const std::vector<Node*>& nodes) {
    std::vector<string> result;
    result.reserve(nodes.size());
    for (Node* n : nodes) {
      result.push_back(n->DebugString());
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

TEST_F(GraphTest, Constructor) {
  Node* source = graph_.source_node();
  EXPECT_NE(source, nullptr);
  Node* sink = graph_.sink_node();
  EXPECT_NE(sink, nullptr);
  VerifyNodes(source, {}, {sink});
  VerifyNodes(sink, {source}, {});
  EXPECT_EQ(2, graph_.num_node_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, RemoveThenAdd) {
  AddNodeWithName("A");
  Node* b = AddNodeWithName("B");
  const int b_id = b->id();
  AddNodeWithName("C");
  EXPECT_EQ(5, graph_.num_node_ids());
  graph_.RemoveNode(b);
  EXPECT_EQ(5, graph_.num_node_ids());
  Node* d = AddNodeWithName("D");
  EXPECT_NE(b_id, d->id());  // Ids should not be reused.
  EXPECT_EQ(6, graph_.num_node_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, InNodesAndOutNodes) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = AddNodeWithName("B");
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.RemoveNode(b);
  Node* d = AddNodeWithName("D");

  const Edge* source_to_a = graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(a, graph_.sink_node());
  graph_.AddEdge(a, 0, c, 0);
  graph_.AddControlEdge(c, graph_.sink_node());

  EXPECT_EQ("A", a->name());
  VerifyNodes(a, {graph_.source_node()}, {c, graph_.sink_node()});

  EXPECT_EQ("C", c->name());
  VerifyNodes(c, {a}, {graph_.sink_node()});

  EXPECT_EQ("D", d->name());
  VerifyNodes(d, {}, {});

  VerifyNodes(graph_.source_node(), {}, {a, graph_.sink_node()});
  VerifyNodes(graph_.sink_node(), {a, c, graph_.source_node()}, {});

  graph_.RemoveEdge(source_to_a);
  VerifyNodes(a, {}, {c, graph_.sink_node()});
  VerifyNodes(graph_.source_node(), {}, {graph_.sink_node()});  // no more a

  graph_.RemoveNode(c);
  VerifyNodes(a, {}, {graph_.sink_node()});                        // no more c
  VerifyNodes(graph_.sink_node(), {a, graph_.source_node()}, {});  // no more c
  EXPECT_EQ(6, graph_.num_node_ids());
  EXPECT_EQ(5, graph_.num_edge_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, NodeByIndex) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.AddEdge(a, 0, c, 0);

  // Ask for 'a' from 'c' by index.
  const Node* a_copy;
  TF_ASSERT_OK(c->input_node(0, &a_copy));
  EXPECT_EQ(a, a_copy);

  const Edge* e;
  TF_ASSERT_OK(c->input_edge(0, &e));
  EXPECT_EQ(0, e->dst_input());
  EXPECT_EQ(a, e->src());
  EXPECT_EQ(c, e->dst());
  EXPECT_EQ(0, e->src_output());

  Node* t = FromNodeDef("T", "TwoInputsOneOutput", 2);
  graph_.AddEdge(a, 0, t, 0);
  // Weird self edge
  graph_.AddEdge(t, 0, t, 1);

  const Node* t_0;
  const Node* t_1;
  TF_ASSERT_OK(t->input_node(0, &t_0));
  EXPECT_EQ(a, t_0);
  TF_ASSERT_OK(t->input_node(1, &t_1));
  EXPECT_EQ(t, t_1);

  TF_ASSERT_OK(t->input_edge(1, &e));
  EXPECT_EQ(1, e->dst_input());
  EXPECT_EQ(t, e->src());

  std::vector<const Edge*> t_input_edges;
  TF_ASSERT_OK(t->input_edges(&t_input_edges));
  ASSERT_EQ(2, t_input_edges.size());
  EXPECT_EQ(a, t_input_edges[0]->src());
  EXPECT_EQ(e, t_input_edges[1]);

  // Check out of bounds access
  EXPECT_FALSE(c->input_node(1, &a_copy).ok());
  EXPECT_FALSE(c->input_node(-1, &a_copy).ok());

  graph_.RemoveNode(a);

  // 'c's input_node entry should be invalidated.
  Status s = c->input_node(0, &a_copy);
  EXPECT_FALSE(s.ok());

  // Add two new nodes.
  Node* a_new = FromNodeDef("A_new", "OneOutput", 0);
  Node* b_new = FromNodeDef("B_new", "OneOutput", 0);

  // Connect one up to c.
  graph_.AddEdge(a_new, 0, c, 0);
  const Edge* a_new_c_edge;
  TF_ASSERT_OK(c->input_edge(0, &a_new_c_edge));

  // Connect up the second edge
  graph_.AddEdge(b_new, 0, c, 0);
  const Edge* b_new_c_edge;
  TF_ASSERT_OK(c->input_edge(0, &b_new_c_edge));

  // Now remove the old one
  graph_.RemoveEdge(a_new_c_edge);

  // Check that the second edge can still be retrieved
  TF_ASSERT_OK(c->input_edge(0, &b_new_c_edge));

  std::vector<const Edge*> c_input_edges;
  TF_ASSERT_OK(c->input_edges(&c_input_edges));
  ASSERT_EQ(1, c_input_edges.size());
  EXPECT_EQ(b_new_c_edge, c_input_edges[0]);
}

TEST_F(GraphTest, NodeIteration) {
  // Set up the graph with some holes due to removals.
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = AddNodeWithName("B");
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.RemoveNode(b);
  Node* d = AddNodeWithName("D");
  const Edge* source_to_a = graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(a, graph_.sink_node());
  graph_.AddEdge(a, 0, c, 0);
  graph_.AddControlEdge(c, graph_.sink_node());
  graph_.RemoveEdge(source_to_a);
  graph_.RemoveNode(c);

  // expected = set of all node DebugStrings we expect in the graph
  std::set<string> expected;
  expected.insert(graph_.source_node()->DebugString());
  expected.insert(a->DebugString());
  expected.insert(d->DebugString());
  expected.insert(graph_.sink_node()->DebugString());

  // Verify that iterating through ids gets the same set of nodes.
  std::set<string> actual;
  for (int id = 0; id < graph_.num_node_ids(); ++id) {
    Node* node = graph_.FindNodeId(id);
    if (node != nullptr) {
      actual.insert(node->DebugString());
    }
  }
  EXPECT_EQ(expected, actual);

  // Verify that range-based for loop gets the same set of nodes.
  actual.clear();
  for (Node* node : graph_.nodes()) {
    actual.insert(node->DebugString());
  }
  EXPECT_EQ(expected, actual);
  VerifyGraphStats();
}

static void CheckType(Node* node, bool b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_9(mht_9_v, 534, "", "./tensorflow/core/graph/graph_test.cc", "CheckType");

  EXPECT_TRUE(b) << node->DebugString();
  // Make sure none of the other IsFoo() methods return true.
  int count = 0;
  if (node->IsSource()) count++;
  if (node->IsSink()) count++;
  if (node->IsOp()) count++;
  EXPECT_EQ(1, count) << node->DebugString();
}

TEST_F(GraphTest, Type) {
  Node* op = AddNodeWithName("A");
  CheckType(graph_.source_node(), graph_.source_node()->IsSource());
  CheckType(graph_.sink_node(), graph_.sink_node()->IsSink());
  CheckType(op, op->IsOp());
  VerifyGraphStats();
}

TEST_F(GraphTest, AddAttr) {
  Node* n1 = AddNodeWithName("A");

  n1->AddAttr("_a", "new_attr");

  string attr;
  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);

  Node* n2 = graph_.CopyNode(n1);

  n1->AddAttr("_b", "new_attr_2");

  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);
  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_b", &attr));
  EXPECT_EQ("new_attr_2", attr);

  EXPECT_EQ(Status::OK(), GetNodeAttr(n2->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);
  EXPECT_NE(Status::OK(), GetNodeAttr(n2->attrs(), "_b", &attr));
}

// Convert edge iteration results into a sorted string.
static string EdgeIter(const Graph& g) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_10(mht_10_v, 579, "", "./tensorflow/core/graph/graph_test.cc", "EdgeIter");

  std::vector<std::pair<int, int> > edges;
  for (const Edge* e : g.edges()) {
    edges.push_back(std::make_pair(e->src()->id(), e->dst()->id()));
  }
  std::sort(edges.begin(), edges.end());
  string result;
  for (auto& p : edges) {
    strings::StrAppend(&result, p.first, "->", p.second, ";");
  }
  return result;
}

TEST_F(GraphTest, EdgeIteration) {
  EXPECT_EQ("0->1;", EdgeIter(graph_));

  Node* a = FromNodeDef("A", "OneInputTwoOutputs", 1);
  Node* b = FromNodeDef("B", "OneInput", 1);
  EXPECT_EQ("0->1;", EdgeIter(graph_));  // Since a,b are currently disconnected

  graph_.AddEdge(a, 0, b, 0);
  EXPECT_EQ("0->1;2->3;", EdgeIter(graph_));

  graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(b, graph_.sink_node());
  EXPECT_EQ("0->1;0->2;2->3;3->1;", EdgeIter(graph_));

  graph_.AddEdge(a, 1, a, 0);
  EXPECT_EQ("0->1;0->2;2->2;2->3;3->1;", EdgeIter(graph_));
  VerifyGraphStats();
}

TEST_F(GraphTest, NewName) {
  string a1 = graph_.NewName("A");
  string a2 = graph_.NewName("A");
  string b1 = graph_.NewName("B");
  EXPECT_NE(a1, a2);
  EXPECT_NE(a1, b1);
  EXPECT_NE(a2, b1);
  EXPECT_TRUE(absl::StartsWith(a1, "A")) << a1;
}

TEST_F(GraphTest, IsValidNode) {
  // Add 1 node to graph_
  Node* g1_node1;
  TF_CHECK_OK(NodeBuilder("g1_node1", "NoOp").Finalize(&graph_, &g1_node1));

  // Add 2 nodes to graph2
  Graph graph2(OpRegistry::Global());
  Node* g2_node1;
  Node* g2_node2;
  TF_CHECK_OK(NodeBuilder("g2_node1", "NoOp").Finalize(&graph2, &g2_node1));
  TF_CHECK_OK(NodeBuilder("g2_node2", "NoOp").Finalize(&graph2, &g2_node2));

  // nullptr
  Status s = graph_.IsValidNode(nullptr);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(string("Node is null"), s.error_message());

  // node id_ is too high
  s = graph_.IsValidNode(g2_node2);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(string("node id 3 is >= than number of nodes in graph 3"),
            s.error_message());

  // valid id_ but different ptr
  s = graph_.IsValidNode(g2_node1);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(string("Node with id 2 is different from the passed in node. "
                   "Does it belong to a different graph?"),
            s.error_message());
}

TEST_F(GraphTest, AddControlEdge) {
  FromGraphDef(
      "node { name: 'A' op: 'OneOutput' }"
      "node { name: 'B' op: 'OneInputTwoOutputs' input: [ 'A:0' ] }"
      "node { name: 'C' op: 'NoOp' } ");
  Node* a = FindNode("A");
  Node* b = FindNode("B");
  Node* c = FindNode("C");

  // Add a control edge.
  const Edge* edge = graph_.AddControlEdge(c, a);
  ASSERT_TRUE(edge != nullptr);
  // Check newly-created edge.
  EXPECT_EQ(edge->src(), c);
  EXPECT_EQ(edge->src_output(), Graph::kControlSlot);
  EXPECT_EQ(edge->dst(), a);
  EXPECT_EQ(edge->dst_input(), Graph::kControlSlot);
  // Check A's NodeDef.
  ASSERT_EQ(a->def().input_size(), 1);
  EXPECT_EQ(a->def().input(0), "^C");

  // Can add control edge redundant with data edge.
  edge = graph_.AddControlEdge(a, b);
  EXPECT_TRUE(edge != nullptr);
  ASSERT_EQ(b->def().input_size(), 2);
  EXPECT_EQ(b->def().input(0), "A:0");
  EXPECT_EQ(b->def().input(1), "^A");

  // Doesn't add edge redundant with control edge.
  edge = graph_.AddControlEdge(a, b);
  EXPECT_TRUE(edge == nullptr);
  EXPECT_EQ(b->def().input_size(), 2);

  // Can add redundant control edge with allow_duplicates.
  edge = graph_.AddControlEdge(a, b, /*allow_duplicates=*/true);
  EXPECT_TRUE(edge != nullptr);
  // create_duplicate causes the NodeDef not to be updated.
  ASSERT_EQ(b->def().input_size(), 2);
  EXPECT_EQ(b->def().input(0), "A:0");
  EXPECT_EQ(b->def().input(1), "^A");

  // Add control edge from source.
  edge = graph_.AddControlEdge(graph_.source_node(), b);
  EXPECT_TRUE(edge != nullptr);
  // Check that we don't include source input in the NodeDef.
  EXPECT_EQ(b->def().input_size(), 2);
  // Doesn't add redundant edge.
  edge = graph_.AddControlEdge(graph_.source_node(), b);
  EXPECT_TRUE(edge == nullptr);
  EXPECT_EQ(b->def().input_size(), 2);
}

TEST_F(GraphTest, RemoveControlEdge) {
  FromGraphDef(
      "node { name: 'A' op: 'OneOutput' }"
      "node { name: 'B' op: 'OneInputTwoOutputs' input: [ 'A:0' ] }"
      "node { name: 'C' op: 'NoOp' } ");
  Node* a = FindNode("A");
  Node* b = FindNode("B");
  Node* c = FindNode("C");

  // Add a control edge.
  const Edge* edge_1 = graph_.AddControlEdge(c, a);
  const Edge* edge_2 = graph_.AddControlEdge(a, b);
  ASSERT_TRUE(edge_1 != nullptr);
  ASSERT_TRUE(edge_2 != nullptr);

  ASSERT_TRUE(ControlEdgeExistsInGraphOrNodeDef(c, a));
  ASSERT_TRUE(ControlEdgeExistsInGraphOrNodeDef(a, b));

  graph_.RemoveControlEdge(edge_1);
  ASSERT_TRUE(!ControlEdgeExistsInGraphOrNodeDef(c, a));
  ASSERT_TRUE(ControlEdgeExistsInGraphOrNodeDef(a, b));

  graph_.RemoveControlEdge(edge_2);
  ASSERT_TRUE(!ControlEdgeExistsInGraphOrNodeDef(c, a));
  ASSERT_TRUE(!ControlEdgeExistsInGraphOrNodeDef(a, b));

  // Test removing a duplicate control edge.
  // Note that unless allow_duplicates is true, the duplicate edge
  // will not be added. That's why we expect edge_4 to be a null
  // pointer. We are not testing with allow_duplicates set to true,
  // as that is a highly unlikely use case that does not make much
  // sense.
  const Edge* edge_3 = graph_.AddControlEdge(c, a);
  const Edge* edge_4 = graph_.AddControlEdge(c, a);
  ASSERT_TRUE(edge_3 != nullptr);
  ASSERT_TRUE(edge_4 == nullptr);

  graph_.RemoveControlEdge(edge_3);
  ASSERT_TRUE(!ControlEdgeExistsInGraphOrNodeDef(c, a));
}

TEST_F(GraphTest, UpdateEdge) {
  // Build a little graph
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = FromNodeDef("B", "OneInputTwoOutputs", 1);
  Node* c = FromNodeDef("C", "OneInputTwoOutputs", 1);
  Node* d = FromNodeDef("D", "OneInput", 1);

  graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(a, graph_.sink_node());
  graph_.AddEdge(a, 0, c, 0);

  graph_.AddControlEdge(c, graph_.sink_node());
  graph_.AddEdge(c, 0, b, 0);
  graph_.AddEdge(c, 1, d, 0);

  // Initial edge connections
  EXPECT_EQ("0->1;0->2;2->1;2->4;4->1;4->3;4->5;", EdgeIter(graph_));

  // Update the inputs, expect that Edge a to b (2->3) is now in the graph
  // and c to b (4->3) no longer appears.
  TF_EXPECT_OK(graph_.UpdateEdge(a, 0, b, 0));
  // Check that the edge is connecting the correct nodes.
  EXPECT_EQ("0->1;0->2;2->1;2->3;2->4;4->1;4->5;", EdgeIter(graph_));

  // Update a's 0th output again.
  TF_EXPECT_OK(graph_.UpdateEdge(a, 0, d, 0));
  EXPECT_EQ("0->1;0->2;2->1;2->3;2->4;2->5;4->1;", EdgeIter(graph_));

  // Update a's 1st output which is out of range.
  Status s = graph_.UpdateEdge(a, 1, d, 0);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(
      s.error_message(),
      "Node 'A' (type: 'OneOutput', num of outputs: 1) does not have output 1");

  // Update a's 1st input which is out of range.
  s = graph_.UpdateEdge(c, 0, a, 0);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(
      s.error_message(),
      "Node 'A' (type: 'OneOutput', num of inputs: 0) does not have input 0");
}

TEST_F(GraphTest, InputEdges) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = FromNodeDef("B", "TwoInputsOneOutput", 2);
  graph_.AddEdge(a, 0, b, 0);
  std::vector<const Edge*> edges;
  EXPECT_EQ(error::INVALID_ARGUMENT, b->input_edges(&edges).code());
  graph_.AddEdge(a, 0, b, 1);
  TF_EXPECT_OK(b->input_edges(&edges));
}

TEST_F(GraphTest, AddFunctionLibrary) {
  // Basic functionality
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  *proto.add_function() = test::function::XTimesFour();
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_TRUE(graph_.flib_def().Find("XTimesTwo") != nullptr);
  EXPECT_TRUE(graph_.flib_def().Find("XTimesFour") != nullptr);

  // Duplicate functions are ignored
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_TRUE(graph_.flib_def().Find("XTimesTwo") != nullptr);
  EXPECT_TRUE(graph_.flib_def().Find("XTimesFour") != nullptr);

  // Duplicate names corresponding to different functions trigger an error
  FunctionDefLibrary error_proto = proto;
  *error_proto.mutable_function(0)->add_node_def() =
      error_proto.function(0).node_def(0);
  Status s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'XTimesTwo' because a different function with "
            "the same name already exists.");

  // Function with same name as an existing op triggers an error
  error_proto = proto;
  error_proto.mutable_function(0)->mutable_signature()->set_name("Add");
  s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'Add' because an op with the same name "
            "already exists.");

  // Adding a gradient function to an existing function is ok
  GradientDef* grad = proto.add_gradient();
  grad->set_function_name("XTimesTwo");
  grad->set_gradient_func("Undefined");  // undefined funcs in grads are ok
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_EQ(graph_.flib_def().FindGradient("XTimesTwo"), "Undefined");

  // Duplicate gradients are ignored
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_EQ(graph_.flib_def().FindGradient("XTimesTwo"), "Undefined");

  // Conflicting gradient triggers an error
  error_proto = proto;
  error_proto.mutable_gradient(0)->set_gradient_func("Undefined2");
  s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot assign gradient function 'Undefined2' to 'XTimesTwo' "
            "because it already has gradient function 'Undefined'");
}

TEST_F(GraphTest, BuildNodeNameIndex) {
  FromGraphDef(
      "node { name: 'A' op: 'OneOutput' }"
      "node { name: 'B' op: 'OneInputTwoOutputs' input: [ 'A:0' ] }"
      "node { name: 'C' op: 'NoOp' } ");

  auto node_name_index = graph_.BuildNodeNameIndex();
  EXPECT_EQ(node_name_index.size(), 5);

  std::vector<string> node_names{"_SOURCE", "_SINK", "A", "B", "C"};
  for (const string& node_name : node_names) {
    EXPECT_NE(node_name_index.find(node_name), node_name_index.end());
    EXPECT_EQ(node_name_index[node_name], FindNode(node_name));
  }
}

TEST_F(GraphTest, Clear) {
  const int num_nodes = 10;
  const int num_edges_per_node = 2;
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  const auto registry = OpRegistry::Global();
  GraphConstructorOptions opts;
  Graph graph(registry);
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  graph.Clear();
  EXPECT_EQ(graph.num_nodes(), 2);
}

void BM_InEdgeIteration(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_11(mht_11_v, 884, "", "./tensorflow/core/graph/graph_test.cc", "BM_InEdgeIteration");

  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  int64_t sum = 0;
  for (auto s : state) {
    for (const Node* node : graph.nodes()) {
      for (auto e : node->in_edges()) {
        sum += e->id();
      }
    }
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_InEdgeIteration)->ArgPair(10, 2);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 6, 2);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 9, 2);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 12, 2);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 15, 2);
BENCHMARK(BM_InEdgeIteration)->ArgPair(10, 4);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 6, 4);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 9, 4);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 12, 4);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 15, 4);
BENCHMARK(BM_InEdgeIteration)->ArgPair(10, 8);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 6, 8);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 9, 8);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 12, 8);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 15, 8);
BENCHMARK(BM_InEdgeIteration)->ArgPair(10, 16);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 6, 16);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 9, 16);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 12, 16);
BENCHMARK(BM_InEdgeIteration)->ArgPair(1 << 15, 16);

void BM_GraphCreation(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_12(mht_12_v, 927, "", "./tensorflow/core/graph/graph_test.cc", "BM_GraphCreation");

  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  const auto registry = OpRegistry::Global();
  GraphConstructorOptions opts;
  // Warmup step.
  Graph graph(registry);
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  int64_t sum = 0;
  for (auto s : state) {
    Graph graph(registry);
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
    sum += graph.num_node_ids();
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_GraphCreation)->ArgPair(10, 2);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 6, 2);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 9, 2);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 12, 2);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 15, 2);
BENCHMARK(BM_GraphCreation)->ArgPair(10, 4);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 6, 4);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 9, 4);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 12, 4);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 15, 4);
BENCHMARK(BM_GraphCreation)->ArgPair(10, 8);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 6, 8);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 9, 8);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 12, 8);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 15, 8);
BENCHMARK(BM_GraphCreation)->ArgPair(10, 16);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 6, 16);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 9, 16);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 12, 16);
BENCHMARK(BM_GraphCreation)->ArgPair(1 << 15, 16);

void BM_ToGraphDef(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_13(mht_13_v, 969, "", "./tensorflow/core/graph/graph_test.cc", "BM_ToGraphDef");

  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  const auto registry = OpRegistry::Global();
  GraphConstructorOptions opts;
  // Warmup step.
  Graph graph(registry);
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  int64_t sum = 0;
  for (auto s : state) {
    GraphDef graph_def;
    graph.ToGraphDef(&graph_def);
    sum += graph_def.node_size();
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_ToGraphDef)->ArgPair(10, 2);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 6, 2);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 9, 2);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 12, 2);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 15, 2);
BENCHMARK(BM_ToGraphDef)->ArgPair(10, 4);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 6, 4);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 9, 4);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 12, 4);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 15, 4);
BENCHMARK(BM_ToGraphDef)->ArgPair(10, 8);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 6, 8);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 9, 8);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 12, 8);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 15, 8);
BENCHMARK(BM_ToGraphDef)->ArgPair(10, 16);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 6, 16);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 9, 16);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 12, 16);
BENCHMARK(BM_ToGraphDef)->ArgPair(1 << 15, 16);

void BM_RemoveNode(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_testDTcc mht_14(mht_14_v, 1011, "", "./tensorflow/core/graph/graph_test.cc", "BM_RemoveNode");

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
    state.ResumeTiming();
    for (Node* n : graph.op_nodes()) {
      graph.RemoveNode(n);
    }
  }
}
BENCHMARK(BM_RemoveNode)->ArgPair(10, 2);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 6, 2);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 9, 2);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 12, 2);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 15, 2);
BENCHMARK(BM_RemoveNode)->ArgPair(10, 4);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 6, 4);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 9, 4);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 12, 4);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 15, 4);
BENCHMARK(BM_RemoveNode)->ArgPair(10, 8);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 6, 8);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 9, 8);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 12, 8);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 15, 8);
BENCHMARK(BM_RemoveNode)->ArgPair(10, 16);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 6, 16);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 9, 16);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 12, 16);
BENCHMARK(BM_RemoveNode)->ArgPair(1 << 15, 16);

}  // namespace
}  // namespace tensorflow
