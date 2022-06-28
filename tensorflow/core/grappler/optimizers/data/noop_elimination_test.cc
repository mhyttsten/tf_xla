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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/noop_elimination.h"
#include <tuple>
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

std::vector<std::pair<string, AttrValue>> GetCommonAttributes() {
  AttrValue shapes_attr, types_attr;
  SetAttrValue("output_shapes", &shapes_attr);
  SetAttrValue("output_types", &types_attr);
  std::vector<std::pair<string, AttrValue>> commonAttributes = {
      {"output_shapes", shapes_attr}, {"output_types", types_attr}};

  return commonAttributes;
}

NodeDef *MakeNode(StringPiece node_type, std::vector<int> params,
                  string input_node, MutableGraphView *graph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_node: \"" + input_node + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination_test.cc", "MakeNode");

  std::vector<NodeDef *> node_params;
  for (int param : params) {
    node_params.push_back(
        graph_utils::AddScalarConstNode<int64_t>(param, graph));
  }
  std::vector<string> inputs = {input_node};
  for (int i = 0; i < node_params.size(); i++) {
    inputs.push_back(node_params[i]->name());
  }
  return graph_utils::AddNode("", node_type, inputs, GetCommonAttributes(),
                              graph);
}

NodeDef *MakeNonConstNode(StringPiece node_type,
                          std::vector<DataType> param_dtypes, string input_node,
                          MutableGraphView *graph) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input_node: \"" + input_node + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination_test.cc", "MakeNonConstNode");

  std::vector<NodeDef *> node_params;
  for (DataType dtype : param_dtypes) {
    node_params.push_back(graph_utils::AddScalarPlaceholder(dtype, graph));
  }
  std::vector<string> inputs = {input_node};
  for (int i = 0; i < node_params.size(); i++) {
    inputs.push_back(node_params[i]->name());
  }

  return graph_utils::AddNode("", node_type, inputs, GetCommonAttributes(),
                              graph);
}

NodeDef *MakeCacheNode(string input_node, MutableGraphView *graph) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input_node: \"" + input_node + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination_test.cc", "MakeCacheNode");

  NodeDef *node_filename =
      graph_utils::AddScalarConstNode<StringPiece>("", graph);
  return graph_utils::AddNode("", "CacheDataset",
                              {std::move(input_node), node_filename->name()},
                              GetCommonAttributes(), graph);
}

NodeDef *MakeRangeNode(MutableGraphView *graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_elimination_testDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination_test.cc", "MakeRangeNode");

  auto *start_node = graph_utils::AddScalarConstNode<int64_t>(0, graph);
  auto *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, graph);
  auto *step_node = graph_utils::AddScalarConstNode<int64_t>(1, graph);

  std::vector<string> range_inputs = {start_node->name(), stop_node->name(),
                                      step_node->name()};

  return graph_utils::AddNode("", "RangeDataset", range_inputs,
                              GetCommonAttributes(), graph);
}

struct NoOpLastEliminationTest
    : ::testing::TestWithParam<std::tuple<string, std::vector<int>, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations at the end of the pipeline.
TEST_P(NoOpLastEliminationTest, EliminateLastNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  const string &node_type = std::get<0>(GetParam());
  const std::vector<int> node_params = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *node = MakeNode(node_type, node_params, range_node->name(), &graph);

  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName(node->name(), output),
            should_keep_node);
}

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpLastEliminationTest,
    ::testing::Values(
        std::make_tuple("TakeDataset", std::vector<int>({-3}), false),
        std::make_tuple("TakeDataset", std::vector<int>({-1}), false),
        std::make_tuple("TakeDataset", std::vector<int>({0}), true),
        std::make_tuple("TakeDataset", std::vector<int>({3}), true),
        std::make_tuple("SkipDataset", std::vector<int>({-1}), true),
        std::make_tuple("SkipDataset", std::vector<int>({0}), false),
        std::make_tuple("SkipDataset", std::vector<int>({3}), true),
        std::make_tuple("PrefetchDataset", std::vector<int>({0}), false),
        std::make_tuple("PrefetchDataset", std::vector<int>({1}), true),
        std::make_tuple("RepeatDataset", std::vector<int>({1}), false),
        std::make_tuple("RepeatDataset", std::vector<int>({2}), true),
        std::make_tuple("ShardDataset", std::vector<int>({1, 0}), false),
        std::make_tuple("ShardDataset", std::vector<int>({2, 0}), true)));

struct NoOpMiddleEliminationTest
    : ::testing::TestWithParam<std::tuple<string, std::vector<int>, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations int the middle of the pipeline.
TEST_P(NoOpMiddleEliminationTest, EliminateMiddleNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  const string &node_type = std::get<0>(GetParam());
  const std::vector<int> node_params = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *node = MakeNode(node_type, node_params, range_node->name(), &graph);

  NodeDef *cache_node = MakeCacheNode(node->name(), &graph);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName(node->name(), output),
            should_keep_node);
  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out = output.node(
      graph_utils::FindGraphNodeWithName(cache_node->name(), output));

  EXPECT_EQ(cache_node_out.input_size(), 2);
  auto last_node_input = (should_keep_node ? node : range_node)->name();
  EXPECT_EQ(cache_node_out.input(0), last_node_input);
}

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpMiddleEliminationTest,
    ::testing::Values(
        std::make_tuple("TakeDataset", std::vector<int>({-1}), false),
        std::make_tuple("TakeDataset", std::vector<int>({-3}), false),
        std::make_tuple("TakeDataset", std::vector<int>({0}), true),
        std::make_tuple("TakeDataset", std::vector<int>({3}), true),
        std::make_tuple("SkipDataset", std::vector<int>({-1}), true),
        std::make_tuple("SkipDataset", std::vector<int>({0}), false),
        std::make_tuple("SkipDataset", std::vector<int>({3}), true),
        std::make_tuple("PrefetchDataset", std::vector<int>({0}), false),
        std::make_tuple("PrefetchDataset", std::vector<int>({1}), true),
        std::make_tuple("RepeatDataset", std::vector<int>({1}), false),
        std::make_tuple("RepeatDataset", std::vector<int>({2}), true),
        std::make_tuple("ShardDataset", std::vector<int>({1, 0}), false),
        std::make_tuple("ShardDataset", std::vector<int>({2, 0}), true)));

using NodesTypes = std::tuple<std::pair<string, std::vector<int>>,
                              std::pair<string, std::vector<int>>>;
struct NoOpMultipleEliminationTest : ::testing::TestWithParam<NodesTypes> {};

// This test checks whether the no-op elimination correctly removes
// multiple noop nodes.
TEST_P(NoOpMultipleEliminationTest, EliminateMultipleNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  static_assert(std::tuple_size<NodesTypes>::value == 2,
                "Make sure to include everything in the test");
  const std::vector<std::pair<string, std::vector<int>>> noop_nodes = {
      std::get<0>(GetParam()), std::get<1>(GetParam())};

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *previous = range_node;
  std::vector<string> nodes_to_remove;
  nodes_to_remove.reserve(noop_nodes.size());

  for (const auto &noop_node : noop_nodes) {
    NodeDef *node =
        MakeNode(noop_node.first, noop_node.second, previous->name(), &graph);
    nodes_to_remove.push_back(node->name());
    previous = node;
  }

  NodeDef *cache_node = MakeCacheNode(previous->name(), &graph);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const auto &noop_node_name : nodes_to_remove)
    EXPECT_FALSE(
        graph_utils::ContainsGraphNodeWithName(noop_node_name, output));

  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out = output.node(
      graph_utils::FindGraphNodeWithName(cache_node->name(), output));

  EXPECT_EQ(cache_node_out.input_size(), 2);
  EXPECT_EQ(cache_node_out.input(0), range_node->name());
}

const auto *const kTakeNode =
    new std::pair<string, std::vector<int>>{"TakeDataset", {-1}};
const auto *const kSkipNode =
    new std::pair<string, std::vector<int>>{"SkipDataset", {0}};
const auto *const kRepeatNode =
    new std::pair<string, std::vector<int>>{"RepeatDataset", {1}};
const auto *const kPrefetchNode =
    new std::pair<string, std::vector<int>>{"PrefetchDataset", {0}};
const auto *const kShardNode =
    new std::pair<string, std::vector<int>>{"ShardDataset", {1, 0}};

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpMultipleEliminationTest,
    ::testing::Combine(::testing::Values(*kTakeNode, *kSkipNode, *kRepeatNode,
                                         *kPrefetchNode, *kShardNode),
                       ::testing::Values(*kTakeNode, *kSkipNode, *kRepeatNode,
                                         *kPrefetchNode, *kShardNode)));

struct NoOpPlaceholdersTest
    : ::testing::TestWithParam<
          std::tuple<std::pair<string, std::vector<DataType>>,
                     std::pair<string, std::vector<DataType>>>> {};

TEST_P(NoOpPlaceholdersTest, NonConstNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  static_assert(std::tuple_size<NodesTypes>::value == 2,
                "Make sure to include everything in the test");
  const std::vector<std::pair<string, std::vector<DataType>>> noop_nodes = {
      std::get<0>(GetParam()), std::get<1>(GetParam())};
  NodeDef *range_node = MakeRangeNode(&graph);
  std::vector<string> nodes_to_keep;
  nodes_to_keep.reserve(noop_nodes.size());
  NodeDef *previous = range_node;

  for (const auto &noop_node : noop_nodes) {
    NodeDef *node = MakeNonConstNode(noop_node.first, noop_node.second,
                                     previous->name(), &graph);
    nodes_to_keep.push_back(node->name());
    previous = node;
  }

  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  for (const auto &noop_node_name : nodes_to_keep)
    EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName(noop_node_name, output));
}

const auto *const kNonConstTakeNode =
    new std::pair<string, std::vector<DataType>>{"TakeDataset", {DT_INT32}};
const auto *const kNonConstSkipNode =
    new std::pair<string, std::vector<DataType>>{"SkipDataset", {DT_INT32}};
const auto *const kNonConstRepeatNode =
    new std::pair<string, std::vector<DataType>>{"RepeatDataset", {DT_INT32}};
const auto *const kNonConstPrefetchNode =
    new std::pair<string, std::vector<DataType>>{"PrefetchDataset", {DT_INT32}};
const auto *const kNonConstShardNode =
    new std::pair<string, std::vector<DataType>>{"ShardDataset",
                                                 {DT_INT32, DT_INT32}};

INSTANTIATE_TEST_CASE_P(
    DoNotRemovePlaceholders, NoOpPlaceholdersTest,
    ::testing::Combine(::testing::Values(*kNonConstTakeNode, *kNonConstSkipNode,
                                         *kNonConstRepeatNode,
                                         *kNonConstPrefetchNode,
                                         *kNonConstShardNode),
                       ::testing::Values(*kNonConstTakeNode, *kNonConstSkipNode,
                                         *kNonConstRepeatNode,
                                         *kNonConstPrefetchNode,
                                         *kNonConstShardNode)));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
