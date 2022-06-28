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
class MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/simple_planner.h"

#include <cstdarg>
#include <initializer_list>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

// A simple op to be used in tests, as syntactic sugar.
class TestOp {
 public:
  TestOp(std::initializer_list<int> inputs, std::initializer_list<int> outputs,
         std::initializer_list<int> temporaries)
      : inputs_(inputs), outputs_(outputs), temporaries_(temporaries) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/simple_planner_test.cc", "TestOp");
}

  const std::vector<int>& inputs() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/simple_planner_test.cc", "inputs");
 return inputs_; }
  const std::vector<int>& outputs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/lite/simple_planner_test.cc", "outputs");
 return outputs_; }
  const std::vector<int>& temporaries() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_3(mht_3_v, 218, "", "./tensorflow/lite/simple_planner_test.cc", "temporaries");
 return temporaries_; }

 private:
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> temporaries_;
};

// A test graph where inputs are processed by the given nodes to produce
// outputs.
class TestGraph {
 public:
  TestGraph(std::initializer_list<int> inputs,
            std::initializer_list<TestOp> nodes,
            std::initializer_list<int> outputs)
      : inputs_(inputs), outputs_(outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_4(mht_4_v, 236, "", "./tensorflow/lite/simple_planner_test.cc", "TestGraph");

    int max_tensor_index = 0;

    for (int t : inputs) {
      max_tensor_index = std::max(max_tensor_index, t);
    }
    for (int t : outputs) {
      max_tensor_index = std::max(max_tensor_index, t);
    }
    for (const auto& node : nodes) {
      auto int_array = [](const std::vector<int>& x) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_5(mht_5_v, 249, "", "./tensorflow/lite/simple_planner_test.cc", "lambda");

        TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
        for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
        return lite;
      };

      nodes_.push_back(TfLiteNode());
      nodes_.back().inputs = int_array(node.inputs());
      for (int t : node.inputs()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
      nodes_.back().outputs = int_array(node.outputs());
      for (int t : node.outputs()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
      nodes_.back().temporaries = int_array(node.temporaries());
      for (int t : node.temporaries()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
    }

    for (int i = 0; i <= max_tensor_index; ++i) {
      tensors_.push_back(TfLiteTensor());
      // Set some default values for allocation_type and bytes, which are the
      // only fields used by the arena planner.
      tensors_.back().allocation_type = kTfLiteArenaRw;
      tensors_.back().bytes = (i + 1) * 3;
    }
  }

  ~TestGraph() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_6(mht_6_v, 282, "", "./tensorflow/lite/simple_planner_test.cc", "~TestGraph");

    for (auto node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
      TfLiteIntArrayFree(node.temporaries);
    }
  }

  const std::vector<TfLiteNode>& nodes() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_7(mht_7_v, 293, "", "./tensorflow/lite/simple_planner_test.cc", "nodes");
 return nodes_; }
  std::vector<TfLiteTensor>* tensors() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_8(mht_8_v, 297, "", "./tensorflow/lite/simple_planner_test.cc", "tensors");
 return &tensors_; }
  const std::vector<int>& inputs() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_9(mht_9_v, 301, "", "./tensorflow/lite/simple_planner_test.cc", "inputs");
 return inputs_; }
  const std::vector<int>& outputs() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_10(mht_10_v, 305, "", "./tensorflow/lite/simple_planner_test.cc", "outputs");
 return outputs_; }
  const std::vector<int>& variables() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_11(mht_11_v, 309, "", "./tensorflow/lite/simple_planner_test.cc", "variables");
 return variables_; }

  void SetVariables(const std::vector<int>& variables) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_12(mht_12_v, 314, "", "./tensorflow/lite/simple_planner_test.cc", "SetVariables");

    variables_ = variables;
  }

  void Swap(TestGraph* other) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_13(mht_13_v, 321, "", "./tensorflow/lite/simple_planner_test.cc", "Swap");

    std::swap(nodes_, other->nodes_);
    std::swap(tensors_, other->tensors_);
    std::swap(inputs_, other->inputs_);
    std::swap(outputs_, other->outputs_);
    std::swap(variables_, other->variables_);
  }

 private:
  std::vector<TfLiteNode> nodes_;
  std::vector<TfLiteTensor> tensors_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
};

// The GraphInfo for a TestGraph.
class TestGraphInfo : public GraphInfo {
 public:
  explicit TestGraphInfo(TestGraph* graph) : graph_(graph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_14(mht_14_v, 343, "", "./tensorflow/lite/simple_planner_test.cc", "TestGraphInfo");
}

  size_t num_tensors() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_15(mht_15_v, 348, "", "./tensorflow/lite/simple_planner_test.cc", "num_tensors");
 return graph_->tensors()->size(); }
  TfLiteTensor* tensor(size_t index) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_16(mht_16_v, 352, "", "./tensorflow/lite/simple_planner_test.cc", "tensor");

    return &graph_->tensors()->at(index);
  }
  size_t num_execution_nodes() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_17(mht_17_v, 358, "", "./tensorflow/lite/simple_planner_test.cc", "num_execution_nodes");
 return graph_->nodes().size(); }
  size_t num_total_nodes() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_18(mht_18_v, 362, "", "./tensorflow/lite/simple_planner_test.cc", "num_total_nodes");
 return graph_->nodes().size(); }
  const TfLiteNode& node(size_t index) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_19(mht_19_v, 366, "", "./tensorflow/lite/simple_planner_test.cc", "node");

    return graph_->nodes()[index];
  }
  size_t node_index(size_t index) const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_20(mht_20_v, 372, "", "./tensorflow/lite/simple_planner_test.cc", "node_index");
 return index; }
  const std::vector<int>& inputs() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_21(mht_21_v, 376, "", "./tensorflow/lite/simple_planner_test.cc", "inputs");
 return graph_->inputs(); }
  const std::vector<int>& outputs() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_22(mht_22_v, 380, "", "./tensorflow/lite/simple_planner_test.cc", "outputs");
 return graph_->outputs(); }
  const std::vector<int>& variables() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_23(mht_23_v, 384, "", "./tensorflow/lite/simple_planner_test.cc", "variables");

    return graph_->variables();
  }

 private:
  TestGraph* graph_;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_24(mht_24_v, 396, "", "./tensorflow/lite/simple_planner_test.cc", "ReportError");

  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  LOG(INFO) << temp_buffer;
}

class SimplePlannerTest : public ::testing::Test {
 protected:
  void SetGraph(TestGraph* graph, bool preserve_all_tensors = false) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_25(mht_25_v, 413, "", "./tensorflow/lite/simple_planner_test.cc", "SetGraph");

    graph_ = graph;
    context_.ReportError = ReportError;
    planner_.reset(new SimplePlanner(
        &context_, std::unique_ptr<GraphInfo>(new TestGraphInfo(graph))));
    CHECK(planner_->ResetAllocations() == kTfLiteOk);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void SwapGraph(TestGraph* graph) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_26(mht_26_v, 425, "", "./tensorflow/lite/simple_planner_test.cc", "SwapGraph");

    graph_->Swap(graph);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void Execute(int start, int end) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_27(mht_27_v, 433, "", "./tensorflow/lite/simple_planner_test.cc", "Execute");

    CHECK(planner_->ExecuteAllocations(start, end) == kTfLiteOk);
  }

  void ReleaseNonPersistentMemory() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_28(mht_28_v, 440, "", "./tensorflow/lite/simple_planner_test.cc", "ReleaseNonPersistentMemory");

    CHECK(planner_->ReleaseNonPersistentMemory() == kTfLiteOk);
  }

  void AcquireNonPersistentMemory() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_29(mht_29_v, 447, "", "./tensorflow/lite/simple_planner_test.cc", "AcquireNonPersistentMemory");

    CHECK(planner_->AcquireNonPersistentMemory() == kTfLiteOk);
  }

  void ResetAllocationsAfter(int node) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_30(mht_30_v, 454, "", "./tensorflow/lite/simple_planner_test.cc", "ResetAllocationsAfter");

    CHECK(planner_->ResetAllocationsAfter(node) == kTfLiteOk);
  }

  bool HasNonPersistentMemory() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_31(mht_31_v, 461, "", "./tensorflow/lite/simple_planner_test.cc", "HasNonPersistentMemory");

    return planner_ && planner_->HasNonPersistentMemory();
  }

  // Returns if the given tensor is allocated or not.
  bool IsAllocated(int tensor_index) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSsimple_planner_testDTcc mht_32(mht_32_v, 469, "", "./tensorflow/lite/simple_planner_test.cc", "IsAllocated");

    return (*graph_->tensors())[tensor_index].data.raw != nullptr;
  }

  TfLiteContext context_;
  TestGraph* graph_;
  std::unique_ptr<SimplePlanner> planner_;
};

TEST_F(SimplePlannerTest, EmptyGraph) {
  TestGraph graph({}, {}, {});
  SetGraph(&graph);
  Execute(0, 10);
}

TEST_F(SimplePlannerTest, GraphWithNoOps) {
  TestGraph graph({0, 10}, {}, {5, 11});
  SetGraph(&graph);
  Execute(0, 10);
  // The outputs are never allocated because they are not connected to any
  // inputs.
  EXPECT_FALSE(IsAllocated(5));
  EXPECT_FALSE(IsAllocated(11));
}

TEST_F(SimplePlannerTest, ZeroSizedTensors) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  (*graph.tensors())[1].bytes = 0;
  SetGraph(&graph);
  ASSERT_EQ(planner_->ExecuteAllocations(0, 10), kTfLiteOk);
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
}

TEST_F(SimplePlannerTest, SimpleGraph) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphInputsPreserved) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithTemporary) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithResetAllocationsAfter) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_FALSE(IsAllocated(3));
  EXPECT_FALSE(IsAllocated(4));
  EXPECT_FALSE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithPersistentResetAllocationsAfter) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  // Make the tensor #5 persistent.
  (*graph.tensors())[5].allocation_type = kTfLiteArenaRwPersistent;
  SetGraph(&graph);
  Execute(0, 10);

  // Save the pointer of the persistent temporary tensor #5.
  void* tensor5_ptr = (*graph.tensors())[5].data.raw;

  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_FALSE(IsAllocated(3));
  EXPECT_FALSE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));

  // Second run
  Execute(0, 10);

  // Check if the persistent pointer isn't changed.
  EXPECT_TRUE(tensor5_ptr == (*graph.tensors())[5].data.raw);
}

}  // namespace
}  // namespace tflite
