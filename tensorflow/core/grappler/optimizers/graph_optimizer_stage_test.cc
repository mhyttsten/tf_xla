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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

class GraphOptimizerStageTest : public ::testing::Test {};

struct FakeResult {};

// NoOp optimizer stage that supports all the node types and does nothing
class FakeOptimizerStage : public GraphOptimizerStage<FakeResult> {
 public:
  explicit FakeOptimizerStage(const string& optimizer_name,
                              const string& stage_name,
                              const GraphOptimizerContext& ctx)
      : GraphOptimizerStage(optimizer_name, stage_name, ctx) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("optimizer_name: \"" + optimizer_name + "\"");
   mht_0_v.push_back("stage_name: \"" + stage_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage_test.cc", "FakeOptimizerStage");
}
  ~FakeOptimizerStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage_test.cc", "IsSupported");
 return true; }
  Status TrySimplify(NodeDef* node, FakeResult* result) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizer_stage_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/grappler/optimizers/graph_optimizer_stage_test.cc", "TrySimplify");

    return Status::OK();
  }
};

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScopeInRoot) {
  const auto scope_and_name = ParseNodeScopeAndName("Add");
  EXPECT_EQ(scope_and_name.scope, "");
  EXPECT_EQ(scope_and_name.name, "Add");
}

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScopeInScope) {
  const auto scope_and_name = ParseNodeScopeAndName("a/b/c/Add");
  EXPECT_EQ(scope_and_name.scope, "a/b/c");
  EXPECT_EQ(scope_and_name.name, "Add");
}

TEST_F(GraphOptimizerStageTest, OptimizedNodeName) {
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ nullptr,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/Add");

  // Without rewrite rule
  EXPECT_EQ(stage.OptimizedNodeName(node), "a/b/c/my_opt/my_stg_Add");
  EXPECT_EQ(stage.OptimizedNodeName(node, std::vector<string>({"Mul", "Sqrt"})),
            "a/b/c/my_opt/my_stg_Add_Mul_Sqrt");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.OptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_Add");
}

TEST_F(GraphOptimizerStageTest, UniqueOptimizedNodeName) {
  GraphDef graph =
      GDef({NDef("a/b/c/A", "NotImportant", {}),
            NDef("a/b/c/my_opt/my_stg_A", "NotImportant", {}),
            NDef("a/b/c/my_opt/my_stg_my_rewrite_A", "NotImportant", {})},
           /*funcs=*/{});

  NodeMap node_map(&graph);
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/A");

  EXPECT_EQ(stage.UniqueOptimizedNodeName(node),
            "a/b/c/my_opt/my_stg_A_unique0");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.UniqueOptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_A_unique1");
}

TEST_F(GraphOptimizerStageTest, UniqueOptimizedNodeNameWithUsedNodeNames) {
  GraphDef graph = GDef(
      {NDef("a/b/c/A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_A_unique0", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_my_rewrite_A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_my_rewrite_A_unique1", "NotImportant", {})},
      /*funcs=*/{});

  NodeMap node_map(&graph);
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/A");

  EXPECT_EQ(stage.UniqueOptimizedNodeName(node),
            "a/b/c/my_opt/my_stg_A_unique1");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.UniqueOptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_A_unique2");
}

TEST_F(GraphOptimizerStageTest, GetInputNodeAndProperties) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add = ops::Add(s.WithOpName("Add"), a, b);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(/*assume_valid_feeds*/ false));

  NodeMap node_map(&item.graph);

  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ &item.graph,
                            /*graph_properties*/ &properties,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));
  ASSERT_EQ(add_node->input_size(), 2);
  EXPECT_EQ(add_node->input(0), "a");
  EXPECT_EQ(add_node->input(1), "b");

  const OpInfo::TensorProperties* add_properties;
  TF_CHECK_OK(stage.GetTensorProperties("Add", &add_properties));
  EXPECT_EQ(add_properties->dtype(), DT_FLOAT);

  const OpInfo::TensorProperties* a_properties;
  TF_CHECK_OK(stage.GetTensorProperties("a:0", &a_properties));
  EXPECT_EQ(a_properties->dtype(), DT_FLOAT_REF);

  const OpInfo::TensorProperties* b_properties;
  TF_CHECK_OK(stage.GetTensorProperties("b:0", &b_properties));
  EXPECT_EQ(b_properties->dtype(), DT_FLOAT_REF);
}

TEST_F(GraphOptimizerStageTest, AddNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add = ops::Add(s.WithOpName("Add"), a, b);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(/*assume_valid_feeds*/ false));

  NodeMap node_map(&item.graph);

  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ &item.graph,
                            /*graph_properties*/ &properties,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));

  // Add a new copy node
  NodeDef* add_node_copy = stage.AddCopyNode("Add_1", add_node);
  EXPECT_EQ(add_node_copy->name(), "Add_1");
  EXPECT_EQ(add_node_copy->op(), "Add");
  ASSERT_EQ(add_node->input_size(), 2);
  EXPECT_EQ(add_node_copy->input(0), "a");
  EXPECT_EQ(add_node_copy->input(1), "b");

  // It must be available for by-name lookup
  NodeDef* add_node_copy_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_1", &add_node_copy_by_name));
  EXPECT_EQ(add_node_copy, add_node_copy_by_name);

  // Add new empty node
  NodeDef* empty_node = stage.AddEmptyNode("Add_2");
  EXPECT_EQ(empty_node->name(), "Add_2");
  EXPECT_EQ(empty_node->input_size(), 0);

  // It must be available for by-name lookup
  NodeDef* empty_node_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_2", &empty_node_by_name));
  EXPECT_EQ(empty_node, empty_node_by_name);

  // Check that AddEmptyNode adds a unique suffix if the node already exists.
  NodeDef* unique_empty_node = stage.AddEmptyNode("Add_2");
  EXPECT_EQ(unique_empty_node->name(), "Add_2_0");
}

}  // namespace
}  // end namespace grappler
}  // end namespace tensorflow
