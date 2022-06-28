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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_elimination_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_elimination_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_elimination_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

void VerifyGraphsMatch(const GraphDef& original_graph,
                       const GraphDef& optimized_graph, int line) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScommon_subgraph_elimination_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/grappler/optimizers/common_subgraph_elimination_test.cc", "VerifyGraphsMatch");

  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << line;
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = optimized_graph.node(i);
    EXPECT_EQ(original.name(), optimized.name()) << line;
    EXPECT_EQ(original.op(), optimized.op()) << line;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << line;
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j)) << line;
    }
  }
}
}  // namespace

class CommonSubgraphEliminationTest : public ArithmeticOptimizerTest {};

TEST_F(CommonSubgraphEliminationTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  CommonSubgraphElimination optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsMatch(item.graph, output, __LINE__);
}

TEST_F(CommonSubgraphEliminationTest, OpDedupping) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {3.14, 2.7}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.14, 2.7}, {1, 2});
  Output div = ops::Div(s.WithOpName("div"), c1, c2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);
  EXPECT_EQ(output.node_size(), 2);
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);

  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  ASSERT_EQ(new_div->input_size(), 2);
  EXPECT_EQ(new_div->input(0), "c1");
  EXPECT_EQ(new_div->input(1), "c1");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(CommonSubgraphEliminationTest, OpDeduppingAssertAndCheckNumerics) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output p = ops::Placeholder(s, DT_BOOL, ops::Placeholder::Shape({}));
  Output c = ops::Const(s.WithOpName("c"), {3.14, 2.7}, {1, 2});
  auto check1 = ops::CheckNumerics(s.WithOpName("check1"), c, "foo");
  auto check2 = ops::CheckNumerics(s.WithOpName("check2"), c, "foo");
  auto assert1 = ops::Assert(s.WithOpName("assert1"), p, {c});
  auto assert2 = ops::Assert(s.WithOpName("assert2"), p, {c});
  Output div = ops::Div(s.WithOpName("div").WithControlDependencies(
                            {assert1.operation, assert2.operation}),
                        check1, check2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div"};
  Tensor bool_t(DT_BOOL, TensorShape({}));
  bool_t.scalar<bool>().setConstant(true);
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", bool_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;

  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 6);
  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  ASSERT_EQ(new_div->input_size(), 3);
  EXPECT_EQ(new_div->input(0), "check1");
  EXPECT_EQ(new_div->input(1), "check2");
  EXPECT_EQ(new_div->input(2), "^assert1");

  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", bool_t}});
  EXPECT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(CommonSubgraphEliminationTest, OpDedupCommutative) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {1.0f, 2.0f}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.0f, 4.0f}, {1, 2});
  Output mul1 = ops::Mul(s.WithOpName("mul1"), c1, c2);
  Output mul2 = ops::Mul(s.WithOpName("mul2"), c2, c1);
  Output div1 = ops::Div(s.WithOpName("div1"), mul1, mul2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div1"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 4);
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);
  const NodeDef* new_c2 = node_map.GetNode("c2");
  ASSERT_NE(new_c2, nullptr);
  const NodeDef* new_mul1 = node_map.GetNode("mul1");
  ASSERT_NE(new_mul1, nullptr);
  ASSERT_EQ(new_mul1->input_size(), 2);
  EXPECT_EQ(new_mul1->input(0), "c1");
  EXPECT_EQ(new_mul1->input(1), "c2");
  const NodeDef* new_div1 = node_map.GetNode("div1");
  ASSERT_NE(new_div1, nullptr);
  ASSERT_EQ(new_div1->input_size(), 2);
  EXPECT_EQ(new_div1->input(0), "mul1");
  EXPECT_EQ(new_div1->input(1), "mul1");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

}  // namespace grappler
}  // namespace tensorflow
