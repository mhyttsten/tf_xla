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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc() {
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

// Declare here, so we don't need a public header.
Status InsertLogging(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def);

class InsertLoggingTest : public ::testing::Test {
 protected:
  void CheckGraphCanRun(const GraphDef& graph_def,
                        const std::vector<string>& output_names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/tools/graph_transforms/insert_logging_test.cc", "CheckGraphCanRun");

    std::unique_ptr<Session> session(NewSession(SessionOptions()));
    TF_ASSERT_OK(session->Create(graph_def));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->Run({}, output_names, {}, &outputs));
  }

  void TestInsertLogging() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/tools/graph_transforms/insert_logging_test.cc", "TestInsertLogging");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(6, print_inputs.size());
    EXPECT_EQ(1, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(1, print_inputs.count("const_node1:0"));
    EXPECT_EQ(1, print_inputs.count("const_node2:0"));
    EXPECT_EQ(1, print_inputs.count("const_node3:0"));
  }

  void TestInsertLoggingByOpType() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc mht_2(mht_2_v, 268, "", "./tensorflow/tools/graph_transforms/insert_logging_test.cc", "TestInsertLoggingByOpType");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op", {"Mul", "Add"}}));
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(3, print_inputs.size());
    EXPECT_EQ(1, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(0, print_inputs.count("const_node1:0"));
    EXPECT_EQ(0, print_inputs.count("const_node2:0"));
    EXPECT_EQ(0, print_inputs.count("const_node3:0"));
  }

  void TestInsertLoggingByPrefix() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSinsert_logging_testDTcc mht_3(mht_3_v, 320, "", "./tensorflow/tools/graph_transforms/insert_logging_test.cc", "TestInsertLoggingByPrefix");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"prefix", {"add_node"}}));
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(2, print_inputs.size());
    EXPECT_EQ(0, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(0, print_inputs.count("const_node1:0"));
    EXPECT_EQ(0, print_inputs.count("const_node2:0"));
    EXPECT_EQ(0, print_inputs.count("const_node3:0"));
  }
};

TEST_F(InsertLoggingTest, TestInsertLogging) { TestInsertLogging(); }

TEST_F(InsertLoggingTest, TestInsertLoggingByOpType) {
  TestInsertLoggingByOpType();
}

TEST_F(InsertLoggingTest, TestInsertLoggingByPrefix) {
  TestInsertLoggingByPrefix();
}

}  // namespace graph_transforms
}  // namespace tensorflow
