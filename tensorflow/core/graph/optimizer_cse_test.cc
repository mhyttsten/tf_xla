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
class MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc() {
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

#include "tensorflow/core/graph/optimizer_cse.h"

#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

static void InitGraph(const string& s, Graph* graph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "InitGraph");

  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  //  parser.AllowRelaxedWhitespace(true);
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

class OptimizerCSETest : public ::testing::Test {
 public:
  OptimizerCSETest() : graph_(OpRegistry::Global()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "OptimizerCSETest");
}

  void InitGraph(const string& s) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "InitGraph");

    ::tensorflow::InitGraph(s, &graph_);
    original_ = CanonicalGraphString(&graph_);
  }

  static bool IncludeNode(const Node* n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "IncludeNode");
 return n->IsOp(); }

  static string EdgeId(const Node* n, int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "EdgeId");

    if (index == 0) {
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "CanonicalGraphString");

    std::vector<string> nodes;
    std::vector<string> edges;
    for (const Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        nodes.push_back(strings::StrCat(n->name(), "(", n->type_string(), ")"));
      }
    }
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    return strings::StrCat(absl::StrJoin(nodes, ";"), "|",
                           absl::StrJoin(edges, ";"));
  }

  string DoCSE(const std::function<bool(const Node*)>& consider_fn = nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "DoCSE");

    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before rewrites: " << before;

    OptimizeCSE(&graph_, consider_fn);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After rewrites:  " << result;
    return result;
  }

  const string& OriginalGraph() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_7(mht_7_v, 293, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "OriginalGraph");
 return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();

// Note that the "rules" in these tests are not meant to be logically correct
TEST_F(OptimizerCSETest, Simple) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul)|"
            "A->C;B->C:1");
}

TEST_F(OptimizerCSETest, Simple_ThreeEquivalent) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul)|"
            "A->C;B->C:1");
}

TEST_F(OptimizerCSETest, Simple_WithFixups) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul);E(Mul)|"
            "A->C;B->C:1;C->E;C->E:1");
}

TEST_F(OptimizerCSETest, Simple_Commutative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul)|"
            "A->C;B->C:1");
}

static bool IsNotMultiply(const Node* n) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_8(mht_8_v, 361, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "IsNotMultiply");
 return n->type_string() != "Mul"; }

// Like Simple_Commutative,
TEST_F(OptimizerCSETest, Simple_Filtered) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(IsNotMultiply), OriginalGraph());
}

TEST_F(OptimizerCSETest, Simple_NotCommutative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, NotEquivalent_Ops) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, Simple_SameOps_SameAttrs1) {
  // Should still do CSE for ops with attrs if they match.
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] attr { key: 'shape'"
      "    value { shape: { dim: { size: 37 name: 'SAME_NAME' } } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] attr { key: 'shape'"
      "    value { shape: { dim: { size: 37 name: 'SAME_NAME' } } } } }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul)|"
            "A->C;B->C:1");
}

TEST_F(OptimizerCSETest, Simple_SameOps_SameAttrs2) {
  // Should still do CSE for ops with attrs if they match, even if they
  // are not in the same order.
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 3 } } }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);C(Mul)|"
            "A->C;B->C:1");
}

TEST_F(OptimizerCSETest, SameConstants) {
  // Should still do CSE for ops with constants if the values are identical
  InitGraph(
      "node { name: 'A' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_INT32 } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Const);D(Mul)|"
            "A->D;A->D:1");
}

TEST_F(OptimizerCSETest, DifferentConstants) {
  // Should still do CSE for ops with extensions if the extensions are identical
  InitGraph(
      "node { name: 'A' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 100000 } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_INT32 } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Const);B(Const);D(Mul)|"
            "A->D;B->D:1");
}

TEST_F(OptimizerCSETest, SameOps_DifferentAttrs1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 4 } } }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, SameOps_DifferentAttrs2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 3 } } }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, NotEquivalent_Inputs) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, Constant_Dedup) {
  Tensor a(DT_FLOAT, TensorShape({1}));
  a.flat<float>()(0) = 1.0;
  Tensor b(DT_DOUBLE, TensorShape({1}));  // Different type
  b.flat<double>()(0) = 1.0;
  Tensor c(DT_FLOAT, TensorShape({1, 1}));  // Different shape
  c.flat<float>()(0) = 1.0;
  Tensor d(DT_FLOAT, TensorShape({1}));  // Different value
  d.flat<float>()(0) = 2.0;

  // A graph contains a bunch of constants.
  Graph g(OpRegistry::Global());
  for (const auto& val : {a, b, c, d, d, c, b, a}) {
    test::graph::Constant(&g, val);  // Node name is n/_0, n/_1, ...
  }
  GraphDef gdef;
  test::graph::ToGraphDef(&g, &gdef);
  InitGraph(gdef.DebugString());

  EXPECT_EQ(OriginalGraph(),
            "n/_0(Const);n/_1(Const);n/_2(Const);n/_3(Const);"
            "n/_4(Const);n/_5(Const);n/_6(Const);n/_7(Const)|");
  std::vector<string> nodes = str_util::Split(DoCSE(), ";|");
  std::set<string> node_set(nodes.begin(), nodes.end());
  // Expect exactly one of each type of node to be retained after CSE.
  EXPECT_EQ(node_set.count("n/_0(Const)") + node_set.count("n/_7(Const)"), 1);
  EXPECT_EQ(node_set.count("n/_1(Const)") + node_set.count("n/_6(Const)"), 1);
  EXPECT_EQ(node_set.count("n/_2(Const)") + node_set.count("n/_5(Const)"), 1);
  EXPECT_EQ(node_set.count("n/_3(Const)") + node_set.count("n/_4(Const)"), 1);
}

void BM_CSE(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSoptimizer_cse_testDTcc mht_9(mht_9_v, 548, "", "./tensorflow/core/graph/optimizer_cse_test.cc", "BM_CSE");

  const int op_nodes = state.range(0);
  string s;
  for (int in = 0; in < 10; in++) {
    s += strings::Printf("node { name: 'in%04d' op: 'Input'}", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < op_nodes; op++) {
    s += strings::Printf(
        "node { name: 'op%04d' op: 'Mul' attr { key: 'T' value { "
        "type: DT_FLOAT } } input: ['in%04d', 'in%04d' ] }",
        op, rnd.Uniform(10), rnd.Uniform(10));
  }

  bool first = true;
  for (auto i : state) {
    state.PauseTiming();
    Graph* graph = new Graph(OpRegistry::Global());
    InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      state.SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      state.ResumeTiming();
      OptimizeCSE(graph, nullptr);
      state.PauseTiming();
    }
    delete graph;
    state.ResumeTiming();
  }
}
BENCHMARK(BM_CSE)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow
