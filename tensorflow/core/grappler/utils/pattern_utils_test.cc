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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/pattern_utils.h"

#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {
namespace utils {
namespace {

using ::tensorflow::ops::Placeholder;

void GetMatMulBiasAddGeluGraph(GraphDef* graph,
                               bool add_external_dependent = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/grappler/utils/pattern_utils_test.cc", "GetMatMulBiasAddGeluGraph");

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto input_shape = ops::Placeholder::Shape({8, 32});
  auto weight_shape = ops::Placeholder::Shape({32, 64});
  auto bias_shape = ops::Placeholder::Shape({64});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto weight = Placeholder(s.WithOpName("weight"), DT_FLOAT, weight_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), input, weight);
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
  if (add_external_dependent) {
    auto external_dependent =
        ops::Identity(s.WithOpName("external_dependent"), bias_add);
  }
  // Gelu with smaller ops
  auto one_over_square_root_two =
      ops::Const(s.WithOpName("one_over_square_root_two"), {0.707f}, {});
  auto bias_add_times_const = ops::Mul(s.WithOpName("bias_add_times_const"),
                                       bias_add, one_over_square_root_two);
  auto erf = ops::Erf(s.WithOpName("erf"), bias_add_times_const);
  auto one = ops::Const(s.WithOpName("one"), {1.0f}, {});
  auto erf_plus_one = ops::AddV2(s.WithOpName("erf_plus_one"), erf, one);
  auto one_half = ops::Const(s.WithOpName("one_half"), {0.5f}, {});
  auto one_half_times_erf_plus_one = ops::Mul(
      s.WithOpName("one_half_times_erf_plus_one"), one_half, erf_plus_one);
  auto gelu =
      ops::Mul(s.WithOpName("gelu"), one_half_times_erf_plus_one, bias_add);
  auto fetch = ops::Identity(s.WithOpName("fetch"), gelu);

  TF_ASSERT_OK(s.ToGraphDef(graph));
}

OpTypePattern GetMatMulBiasAddGeluPattern() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/grappler/utils/pattern_utils_test.cc", "GetMatMulBiasAddGeluPattern");

  // Although labels are arbitrary, for the convenience of check they are
  // prefixed with "my_" to the orginal node names in the global graph.
  // clang-format off
  OpTypePattern pattern_syntax{"Mul", "my_gelu", NodeStatus::kReplace,
    {
      {"Mul", "my_one_half_times_erf_plus_one", NodeStatus::kRemove,
        {
          {"Const", "my_one_half", NodeStatus::kRemain},
          {"AddV2", "my_erf_plus_one", NodeStatus::kRemove,
            {
              {"Erf", "my_erf", NodeStatus::kRemove,
                {
                  {"Mul", "my_bias_add_times_const", NodeStatus::kRemove,
                    {
                      {"BiasAdd", "my_bias_add", NodeStatus::kRemove},
                      {"Const", "my_one_over_square_root_two", NodeStatus::kRemain}
                    }
                  }
                }
              },
              {"Const", "my_one", NodeStatus::kRemain}
            }
          }
        }
      },
      {"BiasAdd", "my_bias_add", NodeStatus::kRemove,
        {
          {"MatMul", "my_matmul", NodeStatus::kRemove},
          {"*", "my_bias", NodeStatus::kRemain}
        }
      }
    }
  };  // clang-format on

  return pattern_syntax;
}

class PatternMatcherTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc mht_2(mht_2_v, 285, "", "./tensorflow/core/grappler/utils/pattern_utils_test.cc", "NodeConfig");
}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSpattern_utils_testDTcc mht_3(mht_3_v, 295, "", "./tensorflow/core/grappler/utils/pattern_utils_test.cc", "CreateGraph");

    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      NodeDef node_def;
      node_def.set_name(node.name);
      node_def.set_op(node.op);
      for (const string& input : node.inputs) {
        node_def.add_input(input);
      }
      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(PatternMatcherTest, Tree) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, and E
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A                          C   D
  //         |                           \ /
  //         B                            E
  //        /
  //       C   D
  //        \ /
  //         E
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match.
  //  {"E", "my_e", NodeStatus::kReplace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove}
  //      {"D", "my_d", NodeStatus::kRemove}
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove},
      {"D", "my_d", NodeStatus::kRemove}
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, {}, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);
}

TEST_F(PatternMatcherTest, DAG) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, and E
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A
  //         |                           B
  //         B                          / \
  //        / \                        C   D
  //       C   D                        \ /
  //        \ /                          E
  //         E
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match.
  //  {"E", "my_e", NodeStatus::kReplace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      },
  //      {"D", "my_d", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      }
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {"b"}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      },
      {"D", "my_d", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      }
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::unordered_set<string> nodes_to_preserve = {"foo"};
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match =
      graph_matcher.GetMatchedNodes(pattern, nodes_to_preserve, root_node_view,
                                    &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);

  // Pattern should not be matched when a node to be removed is one of nodes to
  // be preserved.
  nodes_to_preserve.insert({"c", "d"});
  matched_nodes_map.clear();
  remove_node_indices.clear();
  found_match =
      graph_matcher.GetMatchedNodes(pattern, nodes_to_preserve, root_node_view,
                                    &matched_nodes_map, &remove_node_indices);
  EXPECT_FALSE(found_match);
  EXPECT_TRUE(matched_nodes_map.empty());
  EXPECT_TRUE(remove_node_indices.empty());
}

// Pattern should not be matched if any of candidate remove nodes has external
// dependent.
TEST_F(PatternMatcherTest, DAGExternalDependent) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, E, and F
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A
  //         |                           B
  //         B                          / \
  //        / \                        C   D
  //       C   D                        \ /
  //        \ / \                        E
  //         E   F
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match. Note D is a candidate for remove node as mentioned in the
  // syntax. So Pattern matcher should not find a match.
  //  {"E", "my_e", NodeStatus::Replace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      },
  //      {"D", "my_d", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      }
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"f", "F", {"d"}},
                                {"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {"b"}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      },
      {"D", "my_d", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      }
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, {}, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_FALSE(found_match);
  EXPECT_TRUE(matched_nodes_map.empty());
  EXPECT_TRUE(remove_node_indices.empty());
}

TEST_F(PatternMatcherTest, MatMulBiasAddGelu) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, {}, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);
}

// Pattern should not be matched if any of candidate remove nodes has external
// dependent.
TEST_F(PatternMatcherTest, MatMulBiasAddGeluExternalDependent) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph, /*add_external_dependent=*/true);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, {}, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_FALSE(found_match);
  EXPECT_TRUE(matched_nodes_map.empty());
  EXPECT_TRUE(remove_node_indices.empty());
}

TEST_F(PatternMatcherTest, MatMulBiasAddGeluMutation) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, {}, root_node_view, &matched_nodes_map, &remove_node_indices);
  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  // Before mutation number of nodes.
  int num_nodes_before = graph_view.NumNodes();
  // Before mutation node_names of the remove candidate nodes.
  std::vector<string> remove_node_names;
  for (auto const& node_idx : remove_node_indices) {
    remove_node_names.push_back(graph_view.GetNode(node_idx)->GetName());
  }

  Mutation* mutation = graph_view.GetMutationBuilder();
  // Replace with fused op.
  NodeDef fused_node;
  fused_node.set_name("gelu");
  fused_node.set_op("_FusedMatMul");
  fused_node.add_input(graph_view.GetNode("matmul")->node()->input(0));
  fused_node.add_input(graph_view.GetNode("matmul")->node()->input(1));
  fused_node.add_input(graph_view.GetNode("bias_add")->node()->input(1));
  mutation->AddNode(std::move(fused_node), &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(mutation->Apply());
  // Remove nodes that are marked as NodeStatus::kRemove.
  for (auto const& node_idx : remove_node_indices) {
    mutation->RemoveNode(graph_view.GetNode(node_idx));
  }
  TF_EXPECT_OK(mutation->Apply());

  // After mutation number of nodes.
  int num_nodes_after = graph_view.NumNodes();
  EXPECT_EQ(num_nodes_before - remove_node_indices.size(), num_nodes_after);

  bool remove_nodes_deleted = true;
  for (auto const& node_name : remove_node_names) {
    if (graph_view.GetNode(node_name) != nullptr) {
      remove_nodes_deleted = false;
      break;
    }
  }
  EXPECT_TRUE(remove_nodes_deleted);

  bool replace_node_exist = graph_view.HasNode("gelu") ? true : false;
  EXPECT_TRUE(replace_node_exist);
}

TEST_F(PatternMatcherTest, CommutativeInputs) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, and E
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A
  //         |                        B                B
  //         B                       / \              / \
  //        / \                     C   D     or     D   C
  //       C   D                     \ /              \ /
  //        \ /                       E                E
  //         E
  //
  // Here E is any of {Mul, Add, AddV2} and the root of subgraph to be matched.
  // Pattern matcher would match the following pattern syntax.
  //   {"E", "my_e", NodeStatus::kReplace,
  //     {
  //       {"C", "my_c", NodeStatus::kRemove,
  //         {
  //           {"B", "my_b", NodeStatus::kRemove}
  //         }
  //       },
  //       {"D", "my_d", NodeStatus::kRemove,
  //         {
  //           {"B", "my_b", NodeStatus::kRemove}
  //         }
  //       }
  //     }
  //   }

  ::tensorflow::Status status;
  std::vector<string> commutative_ops = {"Mul", "Add", "AddV2"};
  for (string op : commutative_ops) {
    for (bool should_swap : {false, true}) {
      std::vector<string> commutative_operands =
          (should_swap ? std::vector<string>{"d", "c"}
                       : std::vector<string>{"c", "d"});
      GraphDef graph = CreateGraph({{"e", op, commutative_operands},
                                    {"c", "C", {"b"}},
                                    {"d", "D", {"b"}},
                                    {"b", "B", {"a"}},
                                    {"a", "A", {}}});
      // clang-format off
      OpTypePattern pattern{op, "my_e", NodeStatus::kReplace,
        {
          {"C", "my_c", NodeStatus::kRemove,
            {
              {"B", "my_b", NodeStatus::kRemove}
            }
          },
          {"D", "my_d", NodeStatus::kRemove,
            {
              {"B", "my_b", NodeStatus::kRemove}
            }
          }
        }
      };  // clang-format on

      MutableGraphView graph_view(&graph, &status);
      TF_ASSERT_OK(status);
      TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
      auto root_node_view = graph_view.GetNode("e");

      SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
          &graph_view);
      std::map<string, int> matched_nodes_map;  // label to node index map
      std::set<int> remove_node_indices;
      bool found_match = graph_matcher.GetMatchedNodes(
          pattern, {}, root_node_view, &matched_nodes_map,
          &remove_node_indices);

      EXPECT_TRUE(found_match);
      EXPECT_FALSE(matched_nodes_map.empty());
      EXPECT_FALSE(remove_node_indices.empty());

      bool all_indices_matched = true;
      for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
           it++) {
        auto label = str_util::StripPrefix(it->first, "my_");
        int matched_node_idx = it->second;
        int expected_node_idx = graph_view.GetNode(label)->node_index();
        if (matched_node_idx != expected_node_idx) {
          all_indices_matched = false;
          break;
        }
      }
      EXPECT_TRUE(all_indices_matched);
    }
  }
}

}  // namespace
}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
