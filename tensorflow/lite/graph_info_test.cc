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
class MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc() {
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

#include "tensorflow/lite/graph_info.h"

#include <stddef.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

// Makes a TfLiteIntArray* from std::vector, must free with TfLiteIntFree().
TfLiteIntArray* ConvertVector(const std::vector<int>& x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/graph_info_test.cc", "ConvertVector");

  TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
  for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
  return lite;
}

// A very simple test graph that supports setting in/out tensors on nodes.
class SimpleTestGraph : public GraphInfo {
 public:
  explicit SimpleTestGraph(int node_index_offset = 0)
      : node_index_offset_(node_index_offset) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/graph_info_test.cc", "SimpleTestGraph");

    // 'node_index_offset' number of nodes are not present in the execution
    // plan. (and hence not considered for partitioning)
    for (int i = 0; i < node_index_offset; ++i) AddNode({}, {});
  }

  ~SimpleTestGraph() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/graph_info_test.cc", "~SimpleTestGraph");

    for (auto& node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
    }
  }

  size_t num_total_nodes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_3(mht_3_v, 231, "", "./tensorflow/lite/graph_info_test.cc", "num_total_nodes");
 return nodes_.size(); }
  size_t num_execution_nodes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_4(mht_4_v, 235, "", "./tensorflow/lite/graph_info_test.cc", "num_execution_nodes");

    return nodes_.size() - node_index_offset_;
  }
  const TfLiteNode& node(size_t index) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_5(mht_5_v, 241, "", "./tensorflow/lite/graph_info_test.cc", "node");

    return nodes_[index + node_index_offset_];
  }
  size_t node_index(size_t index) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_6(mht_6_v, 247, "", "./tensorflow/lite/graph_info_test.cc", "node_index");

    return index + node_index_offset_;
  }
  size_t num_tensors() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_7(mht_7_v, 253, "", "./tensorflow/lite/graph_info_test.cc", "num_tensors");
 return tensors_.size(); }
  TfLiteTensor* tensor(size_t index) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_8(mht_8_v, 257, "", "./tensorflow/lite/graph_info_test.cc", "tensor");
 return &tensors_[index]; }
  const std::vector<int>& inputs() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_9(mht_9_v, 261, "", "./tensorflow/lite/graph_info_test.cc", "inputs");
 return inputs_; }
  const std::vector<int>& outputs() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_10(mht_10_v, 265, "", "./tensorflow/lite/graph_info_test.cc", "outputs");
 return outputs_; }
  const std::vector<int>& variables() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_11(mht_11_v, 269, "", "./tensorflow/lite/graph_info_test.cc", "variables");
 return variables_; }

  void AddNode(const std::vector<int>& inputs, const std::vector<int>& outputs,
               bool might_have_side_effect = false) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_12(mht_12_v, 275, "", "./tensorflow/lite/graph_info_test.cc", "AddNode");

    nodes_.push_back(TfLiteNode());
    TfLiteNode& node = nodes_.back();
    node.inputs = ConvertVector(inputs);
    node.outputs = ConvertVector(outputs);
    node.might_have_side_effect = might_have_side_effect;
  }

  void AddTensors(int count) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_13(mht_13_v, 286, "", "./tensorflow/lite/graph_info_test.cc", "AddTensors");
 tensors_.resize(count + tensors_.size()); }

  void SetInputsAndOutputs(const std::vector<int>& inputs,
                           const std::vector<int>& outputs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_14(mht_14_v, 292, "", "./tensorflow/lite/graph_info_test.cc", "SetInputsAndOutputs");

    inputs_ = inputs;
    outputs_ = outputs;
  }

 private:
  size_t node_index_offset_;
  std::vector<TfLiteNode> nodes_;
  std::vector<TfLiteTensor> tensors_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
};

// Partition a graph to generate a list of subgraphs. This wraps the API call
// we are testing and handles memory management and conversion to
// TfLiteIntArray. Populates `subgraphs` with resulting generated subgraphs.
void PartitionGraph(const SimpleTestGraph& graph,
                    const std::vector<int>& nodes_to_partition,
                    std::vector<NodeSubset>* subgraphs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_15(mht_15_v, 314, "", "./tensorflow/lite/graph_info_test.cc", "PartitionGraph");

  TfLiteIntArray* nodes_to_partition_int_array =
      ConvertVector(nodes_to_partition);
  PartitionGraphIntoIndependentNodeSubsets(&graph, nodes_to_partition_int_array,
                                           subgraphs);
  TfLiteIntArrayFree(nodes_to_partition_int_array);
}

// Check a generated list of subgraphs against the expected list of subgraphs.
void CheckPartitionSubgraphs(
    const std::vector<NodeSubset>& generated_subgraphs,
    const std::vector<NodeSubset>& expected_subgraphs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSgraph_info_testDTcc mht_16(mht_16_v, 328, "", "./tensorflow/lite/graph_info_test.cc", "CheckPartitionSubgraphs");

  ASSERT_EQ(generated_subgraphs.size(), expected_subgraphs.size());
  for (size_t subgraph_index = 0; subgraph_index < generated_subgraphs.size();
       subgraph_index++) {
    EXPECT_EQ(generated_subgraphs[subgraph_index].nodes,
              expected_subgraphs[subgraph_index].nodes);
    EXPECT_EQ(generated_subgraphs[subgraph_index].input_tensors,
              expected_subgraphs[subgraph_index].input_tensors);
    EXPECT_EQ(generated_subgraphs[subgraph_index].output_tensors,
              expected_subgraphs[subgraph_index].output_tensors);
  }
}

// Test an empty trivial graph with no partitions.
TEST(PartitionTest, Nodes0PartitionNodes0) {
  SimpleTestGraph graph;
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);
  CheckPartitionSubgraphs(generated_subgraphs, {});
}

// Test a trivial graph with no node and only 1 tensor.
// The tensor is input & output of the graph at the same time.
// Note: This is a regression test to ensure the partitioning logic
// handles this case without crashing.
TEST(PartitionTest, Nodes0PartitionNodes0Tensors1) {
  SimpleTestGraph graph;
  graph.AddTensors(1);
  graph.SetInputsAndOutputs({0}, {0});
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);
  CheckPartitionSubgraphs(generated_subgraphs, {});
}

// Test a 1 node graph with no partitions.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[]
// Output: [kTfNoPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0) {
  SimpleTestGraph graph;
  graph.AddTensors(2);
  graph.AddNode({0}, {1});
  graph.SetInputsAndOutputs({0}, {1});
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfNonPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {0};
  expected_subgraph.output_tensors = {1};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

TEST(PartitionTest, Nodes1PartitionNodes0_WithOffset) {
  constexpr int node_index_offset = 17;
  SimpleTestGraph graph(node_index_offset);
  graph.AddTensors(2);
  graph.AddNode({0}, {1});
  graph.SetInputsAndOutputs({0}, {1});
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfNonPartition;
  expected_subgraph.nodes = {node_index_offset};
  expected_subgraph.input_tensors = {0};
  expected_subgraph.output_tensors = {1};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 1 node graph with no inputs that is fully partitioned.
// Input: node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0Inputs0) {
  SimpleTestGraph graph;
  graph.AddTensors(1);
  graph.AddNode({}, {0});
  graph.SetInputsAndOutputs({}, {0});
  std::vector<NodeSubset> generated_subgraphs;
  std::vector<int> nodes_to_partition = {0};
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {};
  expected_subgraph.output_tensors = {0};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 1 node graph that is partitioned completely.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes1) {
  SimpleTestGraph graph;
  graph.AddTensors(2);
  graph.AddNode({0}, {1});
  graph.SetInputsAndOutputs({0}, {1});
  std::vector<int> nodes_to_partition = {0};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {0};
  expected_subgraph.output_tensors = {1};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 2 node graph where 1 node is partitioned and the other is not.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [1]
// Output: [kTfNonPartition, tensor(0) -> node(0) -> tensor(1),
//          kTfPartition, tensor(1) -> node(1), tensor(2)]
TEST(PartitionTest, Nodes2PartitionNodes1) {
  SimpleTestGraph graph;
  graph.AddTensors(3);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.SetInputsAndOutputs({0}, {2});
  std::vector<int> nodes_to_partition = {1};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfPartition;
  expected_subgraph1.nodes = {1};
  expected_subgraph1.input_tensors = {1};
  expected_subgraph1.output_tensors = {2};
  CheckPartitionSubgraphs(generated_subgraphs,
                          {expected_subgraph0, expected_subgraph1});
}

// Same as above, but with node offset to ensure correct handling of original vs
// execution plan indices.
TEST(PartitionTest, Nodes2PartitionNodes1_WithOffset) {
  constexpr int node_index_offset = 17;
  SimpleTestGraph graph(node_index_offset);
  graph.AddTensors(3);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.SetInputsAndOutputs({0}, {2});
  std::vector<int> nodes_to_partition = {node_index_offset + 1};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {node_index_offset + 0};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfPartition;
  expected_subgraph1.nodes = {node_index_offset + 1};
  expected_subgraph1.input_tensors = {1};
  expected_subgraph1.output_tensors = {2};
  CheckPartitionSubgraphs(generated_subgraphs,
                          {expected_subgraph0, expected_subgraph1});
}

// Test a 2 node graph where both nodes are fully partitioned.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [0, 1]
// Output: [kTfPartition, tensor(0) -> node(0) -> node(1) -> tensor(1)]
TEST(PartitionTest, Nodes2PartitionNodes2) {
  SimpleTestGraph graph;
  graph.AddTensors(3);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.SetInputsAndOutputs({0}, {2});
  std::vector<int> nodes_to_partition = {0, 1};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0, 1};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {2};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph0});
}

// Test a three node model where we want to partition node 0 and node
// 2, but node 0 and node 2 cannot be in the same subgraph since node 2
// depends on node 1 which depends on node 0. Thus, we need to produce three
// subgraphs.
//
// Input: tensor(0) -> node(0) -> tensor(1)
//        tensor(1) -> node(1) -> tensor(2)
//        [tensor(2), tensor(1)] -> node(2) -> tensor(3)
//    nodes_to_partition = [0, 2]
// Output: [[kTfPartition, tensor(0) -> node(0) -> tensor(1),
//          [kTfNonPartition, tensor(1) -> node(1) -> tensor(2)],
//          [kTfPartition, [tensor(2), tensor(1)] -> node(2) -> node(3)]
TEST(PartitionTest, Nodes3PartitionNodes2) {
  SimpleTestGraph graph;
  graph.AddTensors(4);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.AddNode({1, 2}, {3});
  graph.SetInputsAndOutputs({0}, {3});
  std::vector<int> nodes_to_partition = {0, 2};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfNonPartition;
  expected_subgraph1.nodes = {1};
  expected_subgraph1.input_tensors = {1};
  expected_subgraph1.output_tensors = {2};
  NodeSubset expected_subgraph2;
  expected_subgraph2.type = NodeSubset::kTfPartition;
  expected_subgraph2.nodes = {2};
  expected_subgraph2.input_tensors = {1, 2};
  expected_subgraph2.output_tensors = {3};
  CheckPartitionSubgraphs(
      generated_subgraphs,
      {expected_subgraph0, expected_subgraph1, expected_subgraph2});
}

// Test correct partition for graph with control dependency.
// Graph for test is like
// varhandleOp -> ReadVariableOp -> Add -> AssignVariableOp
//             |_________________________^    ^^
//             |------------------------->ReadVariableOp -> (Output)
// ^^ is control dependency, in this case we don't want to invoke the
// last ReadVariableOp before AssignVariableOp finishes executing.
// '>' and '^' represents data dependency.
TEST(PartitionTest, Nodes4PartitionNodes3_WithControlDependency) {
  SimpleTestGraph graph;
  // Construct graph.
  {
    graph.AddTensors(5);
    graph.AddNode({0}, {1}, true);
    graph.AddNode({1}, {2}, true);
    graph.AddNode({2}, {3}, false);
    graph.AddNode({1, 3}, {}, true);
    graph.AddNode({1}, {4}, true);
  }
  graph.SetInputsAndOutputs({0}, {4});
  std::vector<int> nodes_to_partition = {0, 1, 3, 4};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0, 1};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1, 2};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfNonPartition;
  expected_subgraph1.nodes = {2};
  expected_subgraph1.input_tensors = {2};
  expected_subgraph1.output_tensors = {3};
  NodeSubset expected_subgraph2;
  expected_subgraph2.type = NodeSubset::kTfPartition;
  expected_subgraph2.nodes = {3, 4};
  expected_subgraph2.input_tensors = {1, 3};
  expected_subgraph2.output_tensors = {4};
  CheckPartitionSubgraphs(
      generated_subgraphs,
      {expected_subgraph0, expected_subgraph1, expected_subgraph2});
}

}  // namespace
}  // namespace tflite
