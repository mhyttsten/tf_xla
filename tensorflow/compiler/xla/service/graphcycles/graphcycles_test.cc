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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// A test for the GraphCycles interface.

#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"

#include <optional>
#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// We emulate a GraphCycles object with a node vector and an edge vector.
// We then compare the two implementations.

typedef std::vector<int> Nodes;
struct Edge {
  int from;
  int to;
};
typedef std::vector<Edge> Edges;

// Return whether "to" is reachable from "from".
static bool IsReachable(Edges *edges, int from, int to,
                        absl::flat_hash_set<int> *seen) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "IsReachable");

  seen->insert(from);  // we are investigating "from"; don't do it again
  if (from == to) return true;
  for (int i = 0; i != edges->size(); i++) {
    Edge *edge = &(*edges)[i];
    if (edge->from == from) {
      if (edge->to == to) {  // success via edge directly
        return true;
      } else if (seen->find(edge->to) == seen->end() &&  // success via edge
                 IsReachable(edges, edge->to, to, seen)) {
        return true;
      }
    }
  }
  return false;
}

static void PrintNodes(Nodes *nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "PrintNodes");

  LOG(INFO) << "NODES (" << nodes->size() << ")";
  for (int i = 0; i != nodes->size(); i++) {
    LOG(INFO) << (*nodes)[i];
  }
}

static void PrintEdges(Edges *edges) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_2(mht_2_v, 240, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "PrintEdges");

  LOG(INFO) << "EDGES (" << edges->size() << ")";
  for (int i = 0; i != edges->size(); i++) {
    int a = (*edges)[i].from;
    int b = (*edges)[i].to;
    LOG(INFO) << a << " " << b;
  }
  LOG(INFO) << "---";
}

static void PrintGCEdges(Nodes *nodes, tensorflow::GraphCycles *gc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "PrintGCEdges");

  LOG(INFO) << "GC EDGES";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->HasEdge(a, b)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void PrintTransitiveClosure(Nodes *nodes, Edges *edges,
                                   tensorflow::GraphCycles *gc) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_4(mht_4_v, 271, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "PrintTransitiveClosure");

  LOG(INFO) << "Transitive closure";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      absl::flat_hash_set<int> seen;
      if (IsReachable(edges, a, b, &seen)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void PrintGCTransitiveClosure(Nodes *nodes,
                                     tensorflow::GraphCycles *gc) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_5(mht_5_v, 290, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "PrintGCTransitiveClosure");

  LOG(INFO) << "GC Transitive closure";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->IsReachable(a, b)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void CheckTransitiveClosure(Nodes *nodes, Edges *edges,
                                   tensorflow::GraphCycles *gc) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_6(mht_6_v, 308, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "CheckTransitiveClosure");

  absl::flat_hash_set<int> seen;
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      seen.clear();
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      bool gc_reachable = gc->IsReachable(a, b);
      CHECK_EQ(gc_reachable, gc->IsReachableNonConst(a, b));
      bool reachable = IsReachable(edges, a, b, &seen);
      if (gc_reachable != reachable) {
        PrintEdges(edges);
        PrintGCEdges(nodes, gc);
        PrintTransitiveClosure(nodes, edges, gc);
        PrintGCTransitiveClosure(nodes, gc);
        LOG(FATAL) << "gc_reachable " << gc_reachable << " reachable "
                   << reachable << " a " << a << " b " << b;
      }
    }
  }
}

static void CheckEdges(Nodes *nodes, Edges *edges,
                       tensorflow::GraphCycles *gc) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_7(mht_7_v, 334, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "CheckEdges");

  int count = 0;
  for (int i = 0; i != edges->size(); i++) {
    int a = (*edges)[i].from;
    int b = (*edges)[i].to;
    if (!gc->HasEdge(a, b)) {
      PrintEdges(edges);
      PrintGCEdges(nodes, gc);
      LOG(FATAL) << "!gc->HasEdge(" << a << ", " << b << ")";
    }
  }
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->HasEdge(a, b)) {
        count++;
      }
    }
  }
  if (count != edges->size()) {
    PrintEdges(edges);
    PrintGCEdges(nodes, gc);
    LOG(FATAL) << "edges->size() " << edges->size() << "  count " << count;
  }
}

// Returns the index of a randomly chosen node in *nodes.
// Requires *nodes be non-empty.
static int RandomNode(std::mt19937 *rnd, Nodes *nodes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_8(mht_8_v, 366, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "RandomNode");

  std::uniform_int_distribution<int> distribution(0, nodes->size() - 1);
  return distribution(*rnd);
}

// Returns the index of a randomly chosen edge in *edges.
// Requires *edges be non-empty.
static int RandomEdge(std::mt19937 *rnd, Edges *edges) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_9(mht_9_v, 376, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "RandomEdge");

  std::uniform_int_distribution<int> distribution(0, edges->size() - 1);
  return distribution(*rnd);
}

// Returns the index of edge (from, to) in *edges or -1 if it is not in *edges.
static int EdgeIndex(Edges *edges, int from, int to) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_10(mht_10_v, 385, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "EdgeIndex");

  int i = 0;
  while (i != edges->size() &&
         ((*edges)[i].from != from || (*edges)[i].to != to)) {
    i++;
  }
  return i == edges->size() ? -1 : i;
}

TEST(GraphCycles, RandomizedTest) {
  Nodes nodes;
  Edges edges;  // from, to
  tensorflow::GraphCycles graph_cycles;
  static const int kMaxNodes = 7;     // use <= 7 nodes to keep test short
  static const int kDataOffset = 17;  // an offset to the node-specific data
  int n = 100000;
  int op = 0;
  std::mt19937 rnd(tensorflow::testing::RandomSeed() + 1);

  for (int iter = 0; iter != n; iter++) {
    if ((iter % 10000) == 0) VLOG(0) << "Iter " << iter << " of " << n;

    if (VLOG_IS_ON(3)) {
      LOG(INFO) << "===============";
      LOG(INFO) << "last op " << op;
      PrintNodes(&nodes);
      PrintEdges(&edges);
      PrintGCEdges(&nodes, &graph_cycles);
    }
    for (int i = 0; i != nodes.size(); i++) {
      ASSERT_EQ(reinterpret_cast<intptr_t>(graph_cycles.GetNodeData(i)),
                i + kDataOffset)
          << " node " << i;
    }
    CheckEdges(&nodes, &edges, &graph_cycles);
    CheckTransitiveClosure(&nodes, &edges, &graph_cycles);
    std::uniform_int_distribution<int> distribution(0, 5);
    op = distribution(rnd);
    switch (op) {
      case 0:  // Add a node
        if (nodes.size() < kMaxNodes) {
          int new_node = graph_cycles.NewNode();
          ASSERT_NE(-1, new_node);
          VLOG(1) << "adding node " << new_node;
          ASSERT_EQ(nullptr, graph_cycles.GetNodeData(new_node));
          graph_cycles.SetNodeData(
              new_node, reinterpret_cast<void *>(
                            static_cast<intptr_t>(new_node + kDataOffset)));
          ASSERT_GE(new_node, 0);
          for (int i = 0; i != nodes.size(); i++) {
            ASSERT_NE(nodes[i], new_node);
          }
          nodes.push_back(new_node);
        }
        break;

      case 1:  // Remove a node
        if (!nodes.empty()) {
          int node_index = RandomNode(&rnd, &nodes);
          int node = nodes[node_index];
          nodes[node_index] = nodes.back();
          nodes.pop_back();
          VLOG(1) << "removing node " << node;
          graph_cycles.RemoveNode(node);
          int i = 0;
          while (i != edges.size()) {
            if (edges[i].from == node || edges[i].to == node) {
              edges[i] = edges.back();
              edges.pop_back();
            } else {
              i++;
            }
          }
        }
        break;

      case 2:  // Add an edge
        if (!nodes.empty()) {
          int from = RandomNode(&rnd, &nodes);
          int to = RandomNode(&rnd, &nodes);
          if (EdgeIndex(&edges, nodes[from], nodes[to]) == -1) {
            if (graph_cycles.InsertEdge(nodes[from], nodes[to])) {
              Edge new_edge;
              new_edge.from = nodes[from];
              new_edge.to = nodes[to];
              edges.push_back(new_edge);
            } else {
              absl::flat_hash_set<int> seen;
              ASSERT_TRUE(IsReachable(&edges, nodes[to], nodes[from], &seen))
                  << "Edge " << nodes[to] << "->" << nodes[from];
            }
          }
        }
        break;

      case 3:  // Remove an edge
        if (!edges.empty()) {
          int i = RandomEdge(&rnd, &edges);
          int from = edges[i].from;
          int to = edges[i].to;
          ASSERT_EQ(i, EdgeIndex(&edges, from, to));
          edges[i] = edges.back();
          edges.pop_back();
          ASSERT_EQ(-1, EdgeIndex(&edges, from, to));
          VLOG(1) << "removing edge " << from << " " << to;
          graph_cycles.RemoveEdge(from, to);
        }
        break;

      case 4:  // Check a path
        if (!nodes.empty()) {
          int from = RandomNode(&rnd, &nodes);
          int to = RandomNode(&rnd, &nodes);
          int32_t path[2 * kMaxNodes];
          int path_len = graph_cycles.FindPath(nodes[from], nodes[to],
                                               2 * kMaxNodes, path);
          absl::flat_hash_set<int> seen;
          bool reachable = IsReachable(&edges, nodes[from], nodes[to], &seen);
          bool gc_reachable = graph_cycles.IsReachable(nodes[from], nodes[to]);
          ASSERT_EQ(gc_reachable,
                    graph_cycles.IsReachableNonConst(nodes[from], nodes[to]));
          ASSERT_EQ(path_len != 0, reachable);
          ASSERT_EQ(path_len != 0, gc_reachable);
          // In the following line, we add one because a node can appear
          // twice, if the path is from that node to itself, perhaps via
          // every other node.
          ASSERT_LE(path_len, kMaxNodes + 1);
          if (path_len != 0) {
            ASSERT_EQ(nodes[from], path[0]);
            ASSERT_EQ(nodes[to], path[path_len - 1]);
            for (int i = 1; i < path_len; i++) {
              ASSERT_NE(-1, EdgeIndex(&edges, path[i - 1], path[i]));
              ASSERT_TRUE(graph_cycles.HasEdge(path[i - 1], path[i]));
            }
          }
        }
        break;

      case 5:  // Check invariants
        CHECK(graph_cycles.CheckInvariants());
        break;

      default:
        LOG(FATAL);
    }

    // Very rarely, test graph expansion by adding then removing many nodes.
    std::bernoulli_distribution rarely(1.0 / 1024.0);
    if (rarely(rnd)) {
      VLOG(3) << "Graph expansion";
      CheckEdges(&nodes, &edges, &graph_cycles);
      CheckTransitiveClosure(&nodes, &edges, &graph_cycles);
      for (int i = 0; i != 256; i++) {
        int new_node = graph_cycles.NewNode();
        ASSERT_NE(-1, new_node);
        VLOG(1) << "adding node " << new_node;
        ASSERT_GE(new_node, 0);
        ASSERT_EQ(nullptr, graph_cycles.GetNodeData(new_node));
        graph_cycles.SetNodeData(
            new_node, reinterpret_cast<void *>(
                          static_cast<intptr_t>(new_node + kDataOffset)));
        for (int j = 0; j != nodes.size(); j++) {
          ASSERT_NE(nodes[j], new_node);
        }
        nodes.push_back(new_node);
      }
      for (int i = 0; i != 256; i++) {
        ASSERT_GT(nodes.size(), 0);
        int node_index = RandomNode(&rnd, &nodes);
        int node = nodes[node_index];
        nodes[node_index] = nodes.back();
        nodes.pop_back();
        VLOG(1) << "removing node " << node;
        graph_cycles.RemoveNode(node);
        int j = 0;
        while (j != edges.size()) {
          if (edges[j].from == node || edges[j].to == node) {
            edges[j] = edges.back();
            edges.pop_back();
          } else {
            j++;
          }
        }
      }
      CHECK(graph_cycles.CheckInvariants());
    }
  }
}

class GraphCyclesTest : public ::testing::Test {
 public:
  tensorflow::GraphCycles g_;

  // Test relies on ith NewNode() call returning Node numbered i
  GraphCyclesTest() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_11(mht_11_v, 582, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "GraphCyclesTest");

    for (int i = 0; i < 100; i++) {
      CHECK_EQ(i, g_.NewNode());
    }
    CHECK(g_.CheckInvariants());
  }

  bool AddEdge(int x, int y) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_12(mht_12_v, 592, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "AddEdge");
 return g_.InsertEdge(x, y); }

  void AddMultiples() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_13(mht_13_v, 597, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "AddMultiples");

    // For every node x > 0: add edge to 2*x, 3*x
    for (int x = 1; x < 25; x++) {
      EXPECT_TRUE(AddEdge(x, 2 * x)) << x;
      EXPECT_TRUE(AddEdge(x, 3 * x)) << x;
    }
    CHECK(g_.CheckInvariants());
  }

  std::string Path(int x, int y) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_14(mht_14_v, 609, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "Path");

    static const int kPathSize = 5;
    int32_t path[kPathSize];
    int np = g_.FindPath(x, y, kPathSize, path);
    std::string result;
    for (int i = 0; i < np; i++) {
      if (i >= kPathSize) {
        result += " ...";
        break;
      }
      if (!result.empty()) result.push_back(' ');
      char buf[20];
      snprintf(buf, sizeof(buf), "%d", path[i]);
      result += buf;
    }
    return result;
  }
};

TEST_F(GraphCyclesTest, NoCycle) {
  AddMultiples();
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, SimpleCycle) {
  AddMultiples();
  EXPECT_FALSE(AddEdge(8, 4));
  EXPECT_EQ("4 8", Path(4, 8));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, IndirectCycle) {
  AddMultiples();
  EXPECT_TRUE(AddEdge(16, 9));
  CHECK(g_.CheckInvariants());
  EXPECT_FALSE(AddEdge(9, 2));
  EXPECT_EQ("2 4 8 16 9", Path(2, 9));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, LongPath) {
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(4, 6));
  ASSERT_TRUE(AddEdge(6, 8));
  ASSERT_TRUE(AddEdge(8, 10));
  ASSERT_TRUE(AddEdge(10, 12));
  ASSERT_FALSE(AddEdge(12, 2));
  EXPECT_EQ("2 4 6 8 10 ...", Path(2, 12));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, RemoveNode) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(3, 4));
  ASSERT_TRUE(AddEdge(4, 5));
  g_.RemoveNode(3);
  ASSERT_TRUE(AddEdge(5, 1));
}

TEST_F(GraphCyclesTest, ManyEdges) {
  const int N = 50;
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < N; j++) {
      ASSERT_TRUE(AddEdge(i, i + j));
    }
  }
  CHECK(g_.CheckInvariants());
  ASSERT_TRUE(AddEdge(2 * N - 1, 0));
  CHECK(g_.CheckInvariants());
  ASSERT_FALSE(AddEdge(10, 9));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, ContractEdge) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(1, 3));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(3, 4));

  EXPECT_FALSE(g_.ContractEdge(1, 3).has_value());
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(1, 3));

  // Node (2) has more edges.
  EXPECT_EQ(g_.ContractEdge(1, 2).value(), 2);
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(2, 3));
  EXPECT_TRUE(g_.HasEdge(2, 4));
  EXPECT_TRUE(g_.HasEdge(3, 4));

  // Node (2) has more edges.
  EXPECT_EQ(g_.ContractEdge(2, 3).value(), 2);
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(2, 4));
}

TEST_F(GraphCyclesTest, CanContractEdge) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(1, 3));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(3, 4));

  EXPECT_FALSE(g_.CanContractEdge(1, 3));
  EXPECT_FALSE(g_.CanContractEdge(2, 4));
  EXPECT_TRUE(g_.CanContractEdge(1, 2));
  EXPECT_TRUE(g_.CanContractEdge(2, 3));
  EXPECT_TRUE(g_.CanContractEdge(3, 4));
}

static void BM_StressTest(::testing::benchmark::State &state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_15(mht_15_v, 724, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "BM_StressTest");

  const int num_nodes = state.range(0);

  for (auto s : state) {
    tensorflow::GraphCycles g;
    int32_t *nodes = new int32_t[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
      nodes[i] = g.NewNode();
    }
    for (int i = 0; i < num_nodes; i++) {
      int end = std::min(num_nodes, i + 5);
      for (int j = i + 1; j < end; j++) {
        if (nodes[i] >= 0 && nodes[j] >= 0) {
          CHECK(g.InsertEdge(nodes[i], nodes[j]));
        }
      }
    }
    delete[] nodes;
  }
}
BENCHMARK(BM_StressTest)->Range(2048, 1048576);

static void BM_ContractEdge(::testing::benchmark::State &state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSgraphcycles_testDTcc mht_16(mht_16_v, 749, "", "./tensorflow/compiler/xla/service/graphcycles/graphcycles_test.cc", "BM_ContractEdge");

  const int num_nodes = state.range(0);

  for (auto s : state) {
    state.PauseTiming();
    tensorflow::GraphCycles g;
    std::vector<int32_t> nodes;
    nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      nodes.push_back(g.NewNode());
    }
    // All edges point toward the last one.
    for (int i = 0; i < num_nodes - 1; ++i) {
      g.InsertEdge(nodes[i], nodes[num_nodes - 1]);
    }

    state.ResumeTiming();
    int node = num_nodes - 1;
    for (int i = 0; i < num_nodes - 1; ++i) {
      node = g.ContractEdge(nodes[i], node).value();
    }
  }
}
BENCHMARK(BM_ContractEdge)->Arg(1000)->Arg(10000);
