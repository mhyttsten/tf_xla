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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/quantize_training.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class QuantizeTrainingTest : public ::testing::Test {
 protected:
  QuantizeTrainingTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/common_runtime/quantize_training_test.cc", "QuantizeTrainingTest");
 Reset(); }
  void Reset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/common_runtime/quantize_training_test.cc", "Reset");
 g_.reset(new Graph(OpRegistry::Global())); }

  template <typename T>
  Node* Constant(gtl::ArraySlice<T> values, TensorShape shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/common_runtime/quantize_training_test.cc", "Constant");

    return test::graph::Constant(g_.get(), test::AsTensor(values, shape));
  }

  Status Placeholder(Graph* g, const string& name, TensorShape shape,
                     Node** out) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/common_runtime/quantize_training_test.cc", "Placeholder");

    TF_RETURN_IF_ERROR(NodeBuilder(name, "Placeholder")
                           .Attr("dtype", DT_FLOAT)
                           .Attr("shape", shape)
                           .Finalize(g, out));
    return Status::OK();
  }

  Status FindNode(Graph* g, const string& name, Node** out) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSquantize_training_testDTcc mht_4(mht_4_v, 246, "", "./tensorflow/core/common_runtime/quantize_training_test.cc", "FindNode");

    for (Node* node : g->nodes()) {
      if (node->name() == name) {
        *out = node;
        return Status::OK();
      }
    }
    return errors::Unimplemented("Node ", name, " not found.");
  }

  std::unique_ptr<Graph> g_;
};

TEST_F(QuantizeTrainingTest, SignedInput) {
  // Test that Quantization ops are created with the correct signed_input value.
  // Construct the following graph
  /*
           m1
        /      \
      Relu   Identity
        |       |
        a       b
  */
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  g->AddControlEdge(m1, g->sink_node());

  /*
         m1
      /      \
    EMA_Q   EMA_Q  <- these are subgraphs that estimate moving average.
      |       |
    Relu   Identity
      |       |
      a       b
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(63, g->num_nodes());

  // Quantize_and_dequantize node for identity should have signed_input==true.
  Node* identity_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/QuantizeAndDequantizeV2"),
               &identity_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*identity_q_node->attrs().Find("signed_input")));
  // Quantize_and_dequantize node for relu should have signed_input==false.
  Node* relu_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &relu_q_node));
  ASSERT_EQ("false",
            SummarizeAttrValue(*relu_q_node->attrs().Find("signed_input")));
}

TEST_F(QuantizeTrainingTest, RangeGivenTrue) {
  // Test that Quantization ops are created with the correct range_given value.
  // Construct the following graph
  /*
           m1
        /      \
      Relu   Relu6
        |       |
        a       b
  */
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, b);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  /*
         m1
      /      \
    EMA_Q     Q
      |       |
    Relu   Relu6
      |       |
      a       b
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(38, g->num_nodes());

  // Quantize_and_dequantize node for relu6 should have range_given==true.
  Node* relu6_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu6->name(), "/QuantizeAndDequantizeV2"),
               &relu6_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*relu6_q_node->attrs().Find("range_given")));
  // Quantize_and_dequantize node for relu should have range_given==true.
  Node* relu_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &relu_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*relu_q_node->attrs().Find("range_given")));
}

TEST_F(QuantizeTrainingTest, WithBackwardNodes_QuantizeAndDequantize) {
  // Construct a graph with an additional backward Matmul.
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  // We will use node d as input to the backwards matmul to ensure that it
  // isn't quantized.
  Node* d = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  g->AddControlEdge(g->source_node(), d);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // Add a Matmul node with name starting with "gradients". We will check that
  // its input d was not quantized.
  Node* backward_m;
  TF_ASSERT_OK(NodeBuilder(g->NewName("gradients/n"), "MatMul")
                   .Input(d)
                   .Input(m2)
                   .Attr("transpose_a", true)
                   .Attr("transpose_b", false)
                   .Finalize(g, &backward_m));
  g->AddControlEdge(backward_m, g->sink_node());

  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(95, g->num_nodes());

  // Ensure that the backwards matmul input was not quantized.
  Node* found_node;
  Status s = FindNode(g, strings::StrCat(d->name(), "/QuantizeAndDequantizeV2"),
                      &found_node);
  EXPECT_TRUE(absl::StrContains(s.ToString(), "not found")) << s;

  // Ensure that m1 and m2's inputs were quantized.
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &found_node));
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/QuantizeAndDequantizeV2"),
               &found_node));
  TF_ASSERT_OK(FindNode(
      g, strings::StrCat(c->name(), "/QuantizeAndDequantizeV2"), &found_node));
}

TEST_F(QuantizeTrainingTest, WithBackwardNodes_FakeQuant) {
  // Construct a graph with an additional backward Matmul.
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  // We will use node d as input to the backwards matmul to ensure that it
  // isn't quantized.
  Node* d = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  g->AddControlEdge(g->source_node(), d);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // Add a Matmul node with name starting with "gradients". We will check that
  // its input d was not quantized.
  Node* backward_m;
  TF_ASSERT_OK(NodeBuilder(g->NewName("gradients/n"), "MatMul")
                   .Input(d)
                   .Input(m2)
                   .Attr("transpose_a", true)
                   .Attr("transpose_b", false)
                   .Finalize(g, &backward_m));
  g->AddControlEdge(backward_m, g->sink_node());

  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "FakeQuantWithMinMaxVars", g));

  EXPECT_EQ(95, g->num_nodes());

  // Ensure that the backwards matmul input was not quantized.
  Node* found_node;
  Status s = FindNode(g, strings::StrCat(d->name(), "/FakeQuantWithMinMaxVars"),
                      &found_node);
  EXPECT_TRUE(absl::StrContains(s.ToString(), "not found")) << s;

  // Ensure that m1 and m2's inputs were quantized.
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/FakeQuantWithMinMaxVars"),
               &found_node));
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/FakeQuantWithMinMaxVars"),
               &found_node));
  TF_ASSERT_OK(FindNode(
      g, strings::StrCat(c->name(), "/FakeQuantWithMinMaxVars"), &found_node));
}

TEST_F(QuantizeTrainingTest, QuantizeSerializedGraphDef) {
  // Construct a simple graph with 5 nodes.
  Reset();
  Graph* graph = g_.get();
  Node* const_a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* const_b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  graph->AddControlEdge(graph->source_node(), const_a);
  graph->AddControlEdge(graph->source_node(), const_b);
  Node* relu = test::graph::Relu(graph, const_a);
  Node* identity = test::graph::Identity(graph, const_b);
  Node* matmul = test::graph::Matmul(graph, relu, identity, false, false);
  graph->AddControlEdge(matmul, graph->sink_node());

  int num_bits = 8;

  // Convert the graph to the graphdef string.
  GraphDef input_graph;
  graph->ToGraphDef(&input_graph);
  string input_string;
  input_graph.SerializeToString(&input_string);

  string result_string;
  TF_ASSERT_OK(DoQuantizeTrainingOnSerializedGraphDef(
      input_string, num_bits, "QuantizeAndDequantizeV2", &result_string));

  GraphDef result_graphdef;
  EXPECT_TRUE(ParseProtoUnlimited(&result_graphdef, result_string));

  // Ensure that quantizing the serialized graph_def results in a graph with the
  // same number of nodes as quantizing the graph.
  GraphConstructorOptions opts;
  Graph result_graph(OpRegistry::Global());
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, result_graphdef, &result_graph));
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", graph));
  EXPECT_EQ(graph->num_nodes(), result_graph.num_nodes());
}

TEST_F(QuantizeTrainingTest, QuantizeGraphDef) {
  // Construct a simple graph with 5 nodes.
  Reset();
  Graph* graph = g_.get();
  Node* const_a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* const_b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  graph->AddControlEdge(graph->source_node(), const_a);
  graph->AddControlEdge(graph->source_node(), const_b);
  Node* relu = test::graph::Relu(graph, const_a);
  Node* identity = test::graph::Identity(graph, const_b);
  Node* matmul = test::graph::Matmul(graph, relu, identity, false, false);
  graph->AddControlEdge(matmul, graph->sink_node());

  int num_bits = 8;

  // Convert the graph to the graphdef string.
  GraphDef input_graphdef;
  graph->ToGraphDef(&input_graphdef);

  GraphDef result_graphdef;
  TF_ASSERT_OK(DoQuantizeTrainingOnGraphDef(
      input_graphdef, num_bits, "QuantizeAndDequantizeV2", &result_graphdef));

  // Ensure that quantizing the graph_def results in a graph with the same
  // number of nodes as the graph_def.
  GraphConstructorOptions opts;
  Graph result_graph(OpRegistry::Global());
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, result_graphdef, &result_graph));
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", graph));
  EXPECT_EQ(graph->num_nodes(), result_graph.num_nodes());
}

TEST_F(QuantizeTrainingTest, FixedRangeAndEMARange_QuantizeAndDequantize) {
  // Construct the following graph
  // Relu has an unknown range, so we will check if the EMA correctly estimates
  // the range.
  /*
           m1
        /      \
      Relu    Relu6
        |       |
        a       c
  */
  Reset();
  Graph* g = g_.get();
  Node* a;
  TF_ASSERT_OK(Placeholder(g, "a", {2, 2}, &a));
  Node* c = Constant<float>({2.0, 3.0, 4.0, 5.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, c);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  // This is rewritten into the following subgraph, where Q_a and Q_c are
  // quantize and dequantize subgraphs.
  // Since relu's range is unknown, we check that the exponential moving average
  // works correctly.
  /*
         m1
      /      \
     Q_a     Q_c
      |       |
    Relu     Relu6
      |       |
      a       c
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  SessionOptions options;
  Session* sess;
  TF_ASSERT_OK(NewSession(options, &sess));
  GraphDef gdef;
  g->ToGraphDef(&gdef);
  TF_ASSERT_OK(sess->Create(gdef));

  // The min and max values of the relu6 quantization should be constant values
  // of 0 and 6.
  string min_const_name = strings::StrCat(relu6->name(), "/InputMin");
  string max_const_name = strings::StrCat(relu6->name(), "/InputMax");
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  Tensor a1(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a1, {0.0, 1.0, 2.0, 3.0});
  Tensor a2(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a2, {1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(sess->Run({{"a", a1}}, {m1->name()}, {}, &outputs));

  // The value of the min and max should be set to the min and max of a1 since
  // this is the first run that initializes the EMA variables.
  string min_var_name = strings::StrCat(relu->name(), "/Min/Variable");
  string max_var_name = strings::StrCat(relu->name(), "/Max/Variable");
  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 3.0);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  // Now when we run with new inputs, we should get a moving average for the min
  // and max variables. They should be equal to:
  // min_var = old_min_var * decay + min(a2) * (1 - decay)
  // max_var = old_max_var * decay + max(a2) * (1 - decay)
  TF_ASSERT_OK(sess->Run({{"a", a2}}, {m1->name()}, {}, &outputs));

  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  const float decay = 0.999;
  const float expected_min = 0.0 * decay + 1.0 * (1.0 - decay);
  const float expected_max = 3.0 * decay + 4.0 * (1.0 - decay);
  EXPECT_NEAR(outputs[0].flat<float>()(0), expected_min, 1e-4);
  EXPECT_NEAR(outputs[1].flat<float>()(0), expected_max, 1e-4);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);
}

TEST_F(QuantizeTrainingTest, FixedRangeAndEMARange_FakeQuant) {
  // Construct the following graph
  // Relu has an unknown range, so we will check if the EMA correctly estimates
  // the range.
  /*
           m1
        /      \
      Relu    Relu6
        |       |
        a       c
  */
  Reset();
  Graph* g = g_.get();
  Node* a;
  TF_ASSERT_OK(Placeholder(g, "a", {2, 2}, &a));
  Node* c = Constant<float>({2.0, 3.0, 4.0, 5.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, c);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  // This is rewritten into the following subgraph, where Q_a and Q_c are
  // quantize and dequantize subgraphs.
  // Since relu's range is unknown, we check that the exponential moving average
  // works correctly.
  /*
         m1
      /      \
     Q_a     Q_c
      |       |
    Relu     Relu6
      |       |
      a       c
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "FakeQuantWithMinMaxVars", g));

  SessionOptions options;
  Session* sess;
  TF_ASSERT_OK(NewSession(options, &sess));
  GraphDef gdef;
  g->ToGraphDef(&gdef);
  TF_ASSERT_OK(sess->Create(gdef));

  // The min and max values of the relu6 quantization should be constant values
  // of 0 and 6.
  string min_const_name = strings::StrCat(relu6->name(), "/InputMin");
  string max_const_name = strings::StrCat(relu6->name(), "/InputMax");
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  Tensor a1(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a1, {0.0, 1.0, 2.0, 3.0});
  Tensor a2(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a2, {1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(sess->Run({{"a", a1}}, {m1->name()}, {}, &outputs));

  // The value of the min and max should be set to the min and max of a1 since
  // this is the first run that initializes the EMA variables.
  string min_var_name = strings::StrCat(relu->name(), "/Min/Variable");
  string max_var_name = strings::StrCat(relu->name(), "/Max/Variable");
  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 3.0);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  // Now when we run with new inputs, we should get a moving average for the min
  // and max variables. They should be equal to:
  // min_var = old_min_var * decay + min(a2) * (1 - decay)
  // max_var = old_max_var * decay + max(a2) * (1 - decay)
  TF_ASSERT_OK(sess->Run({{"a", a2}}, {m1->name()}, {}, &outputs));

  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  const float decay = 0.999;
  const float expected_min = 0.0 * decay + 1.0 * (1.0 - decay);
  const float expected_max = 3.0 * decay + 4.0 * (1.0 - decay);
  EXPECT_NEAR(outputs[0].flat<float>()(0), expected_min, 1e-4);
  EXPECT_NEAR(outputs[1].flat<float>()(0), expected_max, 1e-4);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);
}

}  // namespace
}  // namespace tensorflow
