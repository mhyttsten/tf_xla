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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc() {
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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const TransformFuncContext& context,
                        GraphDef* output_graph_def);

class StripUnusedNodesTest : public ::testing::Test {
 protected:
  void TestSimpleAdd() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes_test.cc", "TestSimpleAdd");

    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("a_node");
    add_node->add_input("b_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def, {{}, {"add_node"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(1, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestCommonAncestor() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc mht_1(mht_1_v, 242, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes_test.cc", "TestCommonAncestor");

    GraphDef graph_def;

    NodeDef* add_node1 = graph_def.add_node();
    add_node1->set_name("add_node1");
    add_node1->set_op("Add");
    add_node1->add_input("add_node2");
    add_node1->add_input("add_node3");

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("const_node1");
    add_node2->add_input("const_node2");

    NodeDef* add_node3 = graph_def.add_node();
    add_node3->set_name("add_node3");
    add_node3->set_op("Add");
    add_node3->add_input("const_node1");
    add_node3->add_input("const_node3");

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* const_node3 = graph_def.add_node();
    const_node3->set_name("const_node3");
    const_node3->set_op("Const");

    NodeDef* dangling_input = graph_def.add_node();
    dangling_input->set_name("dangling_input");
    dangling_input->set_op("Const");

    NodeDef* add_node4 = graph_def.add_node();
    add_node4->set_name("add_node4");
    add_node4->set_op("Add");
    add_node4->add_input("add_node2");
    add_node4->add_input("add_node3");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(
        graph_def, {{"dangling_input"}, {"add_node1"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node1"));
    EXPECT_EQ(1, node_lookup.count("add_node2"));
    EXPECT_EQ(1, node_lookup.count("add_node3"));
    EXPECT_EQ(0, node_lookup.count("add_node4"));
    EXPECT_EQ(1, node_lookup.count("const_node1"));
    EXPECT_EQ(1, node_lookup.count("const_node2"));
    EXPECT_EQ(1, node_lookup.count("const_node3"));
    EXPECT_EQ(0, node_lookup.count("const_node4"));
    EXPECT_EQ(1, node_lookup.count("dangling_input"));
  }

  void TestSimplePlaceholder() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc mht_2(mht_2_v, 305, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes_test.cc", "TestSimplePlaceholder");

    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(
        StripUnusedNodes(graph_def, {{"mul_node"}, {"add_node"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_FLOAT, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestPlaceholderDefaultArgs() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc mht_3(mht_3_v, 351, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes_test.cc", "TestPlaceholderDefaultArgs");

    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def,
                                  {{"mul_node"},
                                   {"add_node"},
                                   {{"type", {"int32"}}, {"shape", {"1,2,3"}}}},
                                  &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_INT32, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2, 3}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestPlaceholderNamedArgs() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSstrip_unused_nodes_testDTcc mht_4(mht_4_v, 400, "", "./tensorflow/tools/graph_transforms/strip_unused_nodes_test.cc", "TestPlaceholderNamedArgs");

    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def,
                                  {{"mul_node", "a_node"},
                                   {"add_node"},
                                   {{"name", {"a_node", "mul_node"}},
                                    {"type_for_name", {"int64", "quint8"}},
                                    {"shape_for_name", {"1,2", "1, 2, 3"}}}},
                                  &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_QUINT8, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2, 3}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ("Placeholder", node_lookup["a_node"]->op());
    EXPECT_EQ(DT_INT64, node_lookup["a_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2}),
              TensorShape(node_lookup["a_node"]->attr().at("shape").shape()));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }
};

TEST_F(StripUnusedNodesTest, TestSimpleAdd) { TestSimpleAdd(); }

TEST_F(StripUnusedNodesTest, TestCommonAncestor) { TestCommonAncestor(); }

TEST_F(StripUnusedNodesTest, TestSimplePlaceholder) { TestSimplePlaceholder(); }

TEST_F(StripUnusedNodesTest, TestPlaceholderDefaultArgs) {
  TestPlaceholderDefaultArgs();
}

TEST_F(StripUnusedNodesTest, TestPlaceholderNamedArgs) {
  TestPlaceholderNamedArgs();
}

}  // namespace graph_transforms
}  // namespace tensorflow
