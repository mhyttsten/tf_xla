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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc() {
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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

class SortByExecutionOrderTest : public ::testing::Test {
 protected:
  void GetOrder(const GraphDef& graph_def, std::map<string, int>* order) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/tools/graph_transforms/sort_by_execution_order_test.cc", "GetOrder");

    for (int i = 0; i < graph_def.node_size(); ++i) {
      const NodeDef& node = graph_def.node(i);
      (*order)[node.name()] = i;
    }
  }

  void TestSimpleAdd() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/tools/graph_transforms/sort_by_execution_order_test.cc", "TestSimpleAdd");

    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("a_node");
    add_node->add_input("b_node");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(2, order["add_node"]);
    EXPECT_GT(2, order["a_node"]);
    EXPECT_GT(2, order["b_node"]);
  }

  void TestSimpleLinear() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/tools/graph_transforms/sort_by_execution_order_test.cc", "TestSimpleLinear");

    GraphDef graph_def;

    NodeDef* negative_node = graph_def.add_node();
    negative_node->set_name("negative_node");
    negative_node->set_op("Negative");
    negative_node->add_input("sqrt_node");

    NodeDef* relu_node = graph_def.add_node();
    relu_node->set_name("relu_node");
    relu_node->set_op("Relu");
    relu_node->add_input("const_node");

    NodeDef* sqrt_node = graph_def.add_node();
    sqrt_node->set_name("sqrt_node");
    sqrt_node->set_op("Sqrt");
    sqrt_node->add_input("relu_node");

    NodeDef* const_node = graph_def.add_node();
    const_node->set_name("const_node");
    const_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(3, order["negative_node"]);
    EXPECT_EQ(2, order["sqrt_node"]);
    EXPECT_EQ(1, order["relu_node"]);
    EXPECT_EQ(0, order["const_node"]);
  }

  void TestSimpleTree() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/tools/graph_transforms/sort_by_execution_order_test.cc", "TestSimpleTree");

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
    add_node3->add_input("const_node3");
    add_node3->add_input("const_node4");

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* const_node3 = graph_def.add_node();
    const_node3->set_name("const_node3");
    const_node3->set_op("Const");

    NodeDef* const_node4 = graph_def.add_node();
    const_node4->set_name("const_node4");
    const_node4->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(6, order["add_node1"]);
    EXPECT_GT(6, order["add_node2"]);
    EXPECT_GT(6, order["add_node3"]);
    EXPECT_GT(5, order["const_node1"]);
    EXPECT_GT(5, order["const_node2"]);
    EXPECT_GT(5, order["const_node3"]);
    EXPECT_GT(5, order["const_node4"]);
  }

  void TestCommonAncestor() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsort_by_execution_order_testDTcc mht_4(mht_4_v, 331, "", "./tensorflow/tools/graph_transforms/sort_by_execution_order_test.cc", "TestCommonAncestor");

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

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(5, order["add_node1"]);
    EXPECT_GT(5, order["add_node2"]);
    EXPECT_GT(5, order["add_node3"]);
    EXPECT_GT(4, order["const_node2"]);
    EXPECT_GT(4, order["const_node3"]);
    EXPECT_GT(3, order["const_node1"]);
  }
};

TEST_F(SortByExecutionOrderTest, TestSimpleAdd) { TestSimpleAdd(); }

TEST_F(SortByExecutionOrderTest, TestSimpleLinear) { TestSimpleLinear(); }

TEST_F(SortByExecutionOrderTest, TestSimpleTree) { TestSimpleTree(); }

TEST_F(SortByExecutionOrderTest, TestCommonAncestor) { TestCommonAncestor(); }

}  // namespace graph_transforms
}  // namespace tensorflow
