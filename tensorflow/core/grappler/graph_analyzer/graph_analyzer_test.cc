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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc() {
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

#include "tensorflow/core/grappler/graph_analyzer/graph_analyzer.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class GraphAnalyzerTest : public ::testing::Test, protected TestGraphs {
 protected:
  Status BuildMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "BuildMap");
 return gran_->BuildMap(); }

  void FindSubgraphs() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "FindSubgraphs");
 gran_->FindSubgraphs(); }

  void DropInvalidSubgraphs() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "DropInvalidSubgraphs");
 gran_->DropInvalidSubgraphs(); }

  Status CollateResult() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "CollateResult");
 return gran_->CollateResult(); }

  void ExtendSubgraph(Subgraph* parent) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_4(mht_4_v, 227, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "ExtendSubgraph");
 gran_->ExtendSubgraph(parent); }

  void ExtendSubgraphPortAllOrNone(Subgraph* parent, GenNode* node,
                                   GenNode::Port port) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_5(mht_5_v, 233, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "ExtendSubgraphPortAllOrNone");

    gran_->ExtendSubgraphPortAllOrNone(parent, node, port);
  }

  void ExtendSubgraphAllOrNone(Subgraph* parent, GenNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_6(mht_6_v, 240, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "ExtendSubgraphAllOrNone");

    gran_->ExtendSubgraphAllOrNone(parent, node);
  }

  std::vector<string> DumpRawSubgraphs() { return gran_->DumpRawSubgraphs(); }

  std::vector<string> DumpPartials() {
    std::vector<string> result;
    for (const auto& it : gran_->partial_) {
      result.emplace_back(it->Dump());
    }
    return result;
  }

  const GenNodeMap& GetNodes() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_7(mht_7_v, 257, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "GetNodes");
 return gran_->nodes_; }

  GenNode* GetNode(const string& name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_8(mht_8_v, 263, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "GetNode");
 return gran_->nodes_.at(name).get(); }

  SubgraphPtrSet& GetResult() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_9(mht_9_v, 268, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "GetResult");
 return gran_->result_; }
  SubgraphPtrSet& GetPartial() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_10(mht_10_v, 272, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "GetPartial");
 return gran_->partial_; }
  std::deque<Subgraph*>& GetTodo() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgraph_analyzer_testDTcc mht_11(mht_11_v, 276, "", "./tensorflow/core/grappler/graph_analyzer/graph_analyzer_test.cc", "GetTodo");
 return gran_->todo_; }

  // Gets initialized by a particular test from a suitable GraphDef.
  std::unique_ptr<GraphAnalyzer> gran_;
};

TEST_F(GraphAnalyzerTest, BuildMap) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  EXPECT_THAT(st, Eq(Status::OK()));

  auto& map = GetNodes();
  EXPECT_THAT(map.find("node1"), Ne(map.end()));
  EXPECT_THAT(map.find("node2"), Ne(map.end()));
  EXPECT_THAT(map.find("node3"), Ne(map.end()));
}

TEST_F(GraphAnalyzerTest, BuildMapError) {
  // A duplicate node.
  (*graph_3n_self_control_.add_node()) = MakeNodeConst("node1");
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  ASSERT_THAT(
      st, Eq(Status(error::INVALID_ARGUMENT, "Duplicate node name 'node1'.")));
}

TEST_F(GraphAnalyzerTest, FindSubgraphs0) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 0);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  auto& subgraphs = GetResult();
  EXPECT_THAT(subgraphs, SizeIs(0));
  EXPECT_THAT(DumpRawSubgraphs(), ElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, FindSubgraphs1) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  auto& subgraphs = GetResult();
  EXPECT_THAT(subgraphs, SizeIs(3));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: BroadcastGradientArgs(node3)",
      "1: Const(node1)",
      "1: Sub(node2)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// The required subgraphs are larger than the graph.
TEST_F(GraphAnalyzerTest, FindSubgraphsTooLarge) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  EXPECT_THAT(DumpRawSubgraphs(), ElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

//===

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node already in the graph.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsBaseIn) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node not in the graph yet.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto parent = absl::make_unique<Subgraph>(Subgraph::Identity());
  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(parent.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// where the target subgraph size is larger.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsIncomplete) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  // clang-format off
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Propagate backwards through a multi-input link, finding that the
// resulting subgraph would be too large.
TEST_F(GraphAnalyzerTest, MultiInputTooLargeBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Propagate backwards through a multi-input link, finding that nothing
// would be added to the parent subgraph.
TEST_F(GraphAnalyzerTest, MultiInputNothingAddedBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root = absl::make_unique<Subgraph>(
      Subgraph::Identity({GetNode("add2"), GetNode("const2_1"),
                          GetNode("const2_2"), GetNode("const2_3")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate forwards through a multi-input link,
// with the base (currently-extending) node not in the subgraph yet.
TEST_F(GraphAnalyzerTest, MultiInputSuccessForwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: AddN(add2), Sub(sub)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Successfully propagate forwards through a multi-input link.
TEST_F(GraphAnalyzerTest, MultiInputSuccessForwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, DropInvalidSubgraphsMulti) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  // A good one, multi-input is all-in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("const1_2"),
      GetNode("add1"),
  })));
  // A good one, multi-input is all-out
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("add1"),
      GetNode("add2"),
      GetNode("sub"),
  })));
  // A bad one, multi-input is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("add1"),
      GetNode("sub"),
  })));
  // A bad one, multi-input is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("add2"),
      GetNode("const2_1"),
      GetNode("const2_2"),
  })));

  DropInvalidSubgraphs();

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add1), AddN(add2), Sub(sub)",
      "1: AddN(add1), Const(const1_1), Const(const1_2)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

//===

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node already in the graph.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// but no control links propagate. It also tests the situation
// where the target subgraph size is larger.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwardsNoControl) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass1")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass1"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: Const(const1_1), Const(const1_2), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// The control links propagate separately as all-or-none, even on the nodes
// that are all-or-none for the normal inputs.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSeparateControl) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass1")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("pass1"),
                              GenNode::Port(true, -1));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Propagate backwards from all-or-none-input node, finding that the
// resulting subgraph would be too large.
TEST_F(GraphAnalyzerTest, AllOrNoneInputTooLargeBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Propagate backwards from all-or-none-input node, finding that nothing
// would be added to the parent subgraph.
TEST_F(GraphAnalyzerTest, AllOrNoneInputNothingAddedBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root = absl::make_unique<Subgraph>(
      Subgraph::Identity({GetNode("pass2"), GetNode("const2_1"),
                          GetNode("const2_2"), GetNode("const2_3")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate forwards to all-or-none-input node,
// with the base (currently-extending) node not in the subgraph yet.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessForwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards from all-or-none-input node.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: IdentityN(pass2), Sub(sub)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Successfully propagate forwards to all-or-none-input node. This includes
// both all-or-none-input for the normal inputs, and multi-input by the
// control path.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessForwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)",
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass1)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, DropInvalidSubgraphsAllOrNone) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  // A good one, all-or-none is all-in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("const1_2"),
      GetNode("pass1"),
  })));
  // A good one, all-or-none is all-out
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("pass1"),
      GetNode("pass2"),
      GetNode("sub"),
  })));
  // A bad one, all-or-none is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("pass1"),
      GetNode("sub"),
  })));
  // A bad one, all-or-none is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("pass2"),
      GetNode("const2_1"),
      GetNode("const2_2"),
  })));

  DropInvalidSubgraphs();

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: IdentityN(pass1), IdentityN(pass2), Sub(sub)",
      "1: Const(const1_1), Const(const1_2), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
