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
class MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc() {
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
#include "tensorflow/lite/arena_planner.h"

#include <stdio.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

constexpr const int kTensorAlignment = 4;

// A simple op to be used in tests, as syntactic sugar.
class TestOp {
 public:
  TestOp(std::initializer_list<int> inputs, std::initializer_list<int> outputs,
         std::initializer_list<int> temporaries)
      : inputs_(inputs), outputs_(outputs), temporaries_(temporaries) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/arena_planner_test.cc", "TestOp");
}

  const std::vector<int>& inputs() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/arena_planner_test.cc", "inputs");
 return inputs_; }
  const std::vector<int>& outputs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/arena_planner_test.cc", "outputs");
 return outputs_; }
  const std::vector<int>& temporaries() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_3(mht_3_v, 227, "", "./tensorflow/lite/arena_planner_test.cc", "temporaries");
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
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_4(mht_4_v, 245, "", "./tensorflow/lite/arena_planner_test.cc", "TestGraph");

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
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_5(mht_5_v, 258, "", "./tensorflow/lite/arena_planner_test.cc", "lambda");

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
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_6(mht_6_v, 291, "", "./tensorflow/lite/arena_planner_test.cc", "~TestGraph");

    for (auto node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
      TfLiteIntArrayFree(node.temporaries);
    }
  }

  const std::vector<TfLiteNode>& nodes() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_7(mht_7_v, 302, "", "./tensorflow/lite/arena_planner_test.cc", "nodes");
 return nodes_; }
  std::vector<TfLiteTensor>* tensors() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_8(mht_8_v, 306, "", "./tensorflow/lite/arena_planner_test.cc", "tensors");
 return &tensors_; }
  const std::vector<int>& inputs() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_9(mht_9_v, 310, "", "./tensorflow/lite/arena_planner_test.cc", "inputs");
 return inputs_; }
  const std::vector<int>& outputs() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_10(mht_10_v, 314, "", "./tensorflow/lite/arena_planner_test.cc", "outputs");
 return outputs_; }
  const std::vector<int>& variables() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_11(mht_11_v, 318, "", "./tensorflow/lite/arena_planner_test.cc", "variables");
 return variables_; }

  void SetVariables(const std::vector<int>& variables) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_12(mht_12_v, 323, "", "./tensorflow/lite/arena_planner_test.cc", "SetVariables");

    variables_ = variables;
  }

  void Swap(TestGraph* other) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_13(mht_13_v, 330, "", "./tensorflow/lite/arena_planner_test.cc", "Swap");

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
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_14(mht_14_v, 352, "", "./tensorflow/lite/arena_planner_test.cc", "TestGraphInfo");
}

  size_t num_tensors() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_15(mht_15_v, 357, "", "./tensorflow/lite/arena_planner_test.cc", "num_tensors");
 return graph_->tensors()->size(); }
  TfLiteTensor* tensor(size_t index) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_16(mht_16_v, 361, "", "./tensorflow/lite/arena_planner_test.cc", "tensor");

    return &graph_->tensors()->at(index);
  }
  size_t num_execution_nodes() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_17(mht_17_v, 367, "", "./tensorflow/lite/arena_planner_test.cc", "num_execution_nodes");
 return graph_->nodes().size(); }
  size_t num_total_nodes() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_18(mht_18_v, 371, "", "./tensorflow/lite/arena_planner_test.cc", "num_total_nodes");
 return graph_->nodes().size(); }
  const TfLiteNode& node(size_t index) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_19(mht_19_v, 375, "", "./tensorflow/lite/arena_planner_test.cc", "node");

    return graph_->nodes()[index];
  }
  size_t node_index(size_t index) const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_20(mht_20_v, 381, "", "./tensorflow/lite/arena_planner_test.cc", "node_index");
 return index; }
  const std::vector<int>& inputs() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_21(mht_21_v, 385, "", "./tensorflow/lite/arena_planner_test.cc", "inputs");
 return graph_->inputs(); }
  const std::vector<int>& outputs() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_22(mht_22_v, 389, "", "./tensorflow/lite/arena_planner_test.cc", "outputs");
 return graph_->outputs(); }
  const std::vector<int>& variables() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_23(mht_23_v, 393, "", "./tensorflow/lite/arena_planner_test.cc", "variables");

    return graph_->variables();
  }

 private:
  TestGraph* graph_;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_24(mht_24_v, 405, "", "./tensorflow/lite/arena_planner_test.cc", "ReportError");

  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  LOG(INFO) << temp_buffer;
}

class ArenaPlannerTest : public ::testing::Test {
 protected:
  void SetGraph(TestGraph* graph, bool preserve_all_tensors = false) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_25(mht_25_v, 422, "", "./tensorflow/lite/arena_planner_test.cc", "SetGraph");

    graph_ = graph;
    context_.ReportError = ReportError;
    planner_.reset(new ArenaPlanner(
        &context_, std::unique_ptr<GraphInfo>(new TestGraphInfo(graph)),
        preserve_all_tensors, kTensorAlignment));
    CHECK(planner_->ResetAllocations() == kTfLiteOk);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void SwapGraph(TestGraph* graph) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_26(mht_26_v, 435, "", "./tensorflow/lite/arena_planner_test.cc", "SwapGraph");

    graph_->Swap(graph);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void Execute(int start, int end) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_27(mht_27_v, 443, "", "./tensorflow/lite/arena_planner_test.cc", "Execute");

    CHECK(planner_->ExecuteAllocations(start, end) == kTfLiteOk);
  }

  void ReleaseNonPersistentMemory() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_28(mht_28_v, 450, "", "./tensorflow/lite/arena_planner_test.cc", "ReleaseNonPersistentMemory");

    CHECK(planner_->ReleaseNonPersistentMemory() == kTfLiteOk);
  }

  void AcquireNonPersistentMemory() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_29(mht_29_v, 457, "", "./tensorflow/lite/arena_planner_test.cc", "AcquireNonPersistentMemory");

    CHECK(planner_->AcquireNonPersistentMemory() == kTfLiteOk);
  }

  void ResetAllocationsAfter(int node) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_30(mht_30_v, 464, "", "./tensorflow/lite/arena_planner_test.cc", "ResetAllocationsAfter");

    CHECK(planner_->ResetAllocationsAfter(node) == kTfLiteOk);
  }

  bool HasNonPersistentMemory() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_31(mht_31_v, 471, "", "./tensorflow/lite/arena_planner_test.cc", "HasNonPersistentMemory");

    return planner_ && planner_->HasNonPersistentMemory();
  }

  // Returns the actual offset of a given tensor, relative to the start of its
  // arena.
  std::ptrdiff_t GetOffset(int tensor_index) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_32(mht_32_v, 480, "", "./tensorflow/lite/arena_planner_test.cc", "GetOffset");

    const TfLiteTensor& tensor = (*graph_->tensors())[tensor_index];
    return reinterpret_cast<std::intptr_t>(tensor.data.raw) -
           planner_->BasePointer(tensor.allocation_type);
  }

  // Returns the first aligned offset after a given tensor.
  std::ptrdiff_t GetOffsetAfter(int tensor_index) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_33(mht_33_v, 490, "", "./tensorflow/lite/arena_planner_test.cc", "GetOffsetAfter");

    const TfLiteTensor& tensor = (*graph_->tensors())[tensor_index];
    std::ptrdiff_t offset = GetOffset(tensor_index) + tensor.bytes;
    // We must make sure the offset is aligned to kDefaultArenaAlignment.
    if (offset % kTensorAlignment != 0) {
      offset += kTensorAlignment - offset % kTensorAlignment;
    }
    return offset;
  }

  // Returns if the given tensor is unallocated or not.
  bool IsUnallocated(int tensor_index) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSarena_planner_testDTcc mht_34(mht_34_v, 504, "", "./tensorflow/lite/arena_planner_test.cc", "IsUnallocated");

    return (*graph_->tensors())[tensor_index].data.raw == nullptr;
  }

  TfLiteContext context_;
  TestGraph* graph_;
  std::unique_ptr<ArenaPlanner> planner_;
};

TEST_F(ArenaPlannerTest, EmptyGraph) {
  TestGraph graph({}, {}, {});
  SetGraph(&graph);
  Execute(0, 10);
}

TEST_F(ArenaPlannerTest, GraphWithNoOps) {
  TestGraph graph({0, 10}, {}, {5, 11});
  SetGraph(&graph);
  Execute(0, 10);
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(10), GetOffsetAfter(0));
  // The outputs are never allocated because they are not connected to any
  // inputs.
  EXPECT_TRUE((*graph.tensors())[5].data.raw == nullptr);
  EXPECT_TRUE((*graph.tensors())[11].data.raw == nullptr);
}

TEST_F(ArenaPlannerTest, GraphWithOneOp) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  SetGraph(&graph);
  Execute(0, 10);
  EXPECT_EQ(GetOffset(2), 8);
  EXPECT_EQ(GetOffsetAfter(2), 20);
}

TEST_F(ArenaPlannerTest, ZeroSizedTensors) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  (*graph.tensors())[1].bytes = 0;
  SetGraph(&graph);
  ASSERT_EQ(planner_->ExecuteAllocations(0, 10), kTfLiteOk);
  EXPECT_EQ((*graph_->tensors())[1].data.raw, nullptr);
}

TEST_F(ArenaPlannerTest, SimpleGraph) {
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

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);
}

TEST_F(ArenaPlannerTest, SimpleGraphInputsPreserved) {
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

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithTemporary) {
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

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithResetAllocationsAfter) {
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

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));

  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_FALSE(IsUnallocated(0));
  EXPECT_FALSE(IsUnallocated(1));
  EXPECT_FALSE(IsUnallocated(2));
  EXPECT_TRUE(IsUnallocated(3));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_TRUE(IsUnallocated(5));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithPersistentResetAllocationsAfter) {
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

  EXPECT_FALSE(IsUnallocated(0));
  EXPECT_FALSE(IsUnallocated(1));
  EXPECT_FALSE(IsUnallocated(2));
  EXPECT_TRUE(IsUnallocated(3));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_FALSE(IsUnallocated(5));

  // Second run
  Execute(0, 10);

  // Check if the persistent pointer isn't changed.
  EXPECT_TRUE(tensor5_ptr == (*graph.tensors())[5].data.raw);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithOptionals) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, -1, 5}, {3}, {}}  // Third op, with optional
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithLargeTensor) {
  TestGraph graph({0, -1},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},      // First op
                      {{1}, {2}, {}},      // Second op
                      {{2, 0}, {4}, {5}},  // Third op, with temporary
                      {{4, -1}, {3}, {}}   // Fourth op, with optional
                  },
                  {3});

  // Make #1 very large so its vacancy can be filled with #5 and #4.
  (*graph.tensors())[1].bytes = 40;

  SetGraph(&graph);
  Execute(0, 10);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 -1 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(1), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithPersistentTensor) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with persistent
                      {{4, -1}, {3}, {}}   // Third op, with optional
                  },
                  {3});

  // Make #1 persistent so it goes into its own arena.
  (*graph.tensors())[1].allocation_type = kTfLiteArenaRwPersistent;
  // The only use case for kTfLiteArenaRwPersistent is variable tensor now.
  graph.SetVariables({1});

  SetGraph(&graph);
  Execute(0, 10);

  // Make sure #0 and #1 were given different memory locations (because they
  // will both have offset=0, in different arenas.)
  EXPECT_NE((*graph.tensors())[0].data.raw, (*graph.tensors())[1].data.raw);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), 0);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithDynamicTensor) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4, -1}, {3}, {}}   // Third op, with optional
                  },
                  {3});

  // Make #1 dynamic so it does not get allocated.
  (*graph.tensors())[1].allocation_type = kTfLiteDynamic;

  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_EQ((*graph.tensors())[1].data.raw, nullptr);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, LargerGraphAndStepwiseAllocation) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2, 3}, {}},
                      {{2, 0}, {4, 5}, {6}},
                      {{1, -1}, {7}, {}},
                      {{7, 3}, {8}, {9}},
                      {{4, 5, 8}, {10}, {}},
                  },
                  {10});
  SetGraph(&graph);

  // The allocation plan is made at the beginning and is independent of
  // the execution steps. Here's the allocation order:
  //   Op0: +0 +1 +2 +3
  //   Op1: +6 +4 +5 -6 -2
  //   Op2: +7
  //   Op3: +9 +8 -9 -3 -7
  //   Op4: +10 -4 -5 -8

  Execute(0, 0);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_TRUE(IsUnallocated(6));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_TRUE(IsUnallocated(5));
  EXPECT_TRUE(IsUnallocated(7));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(1, 1);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_TRUE(IsUnallocated(7));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(2, 2);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(3, 3);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(4, 4);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_EQ(GetOffset(10), 12);
}

TEST_F(ArenaPlannerTest, ModifiedGraph) {
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

  // Now update the graph data used by the existing allocator. It should behave
  // as if it had been recreated with the new graph.
  TestGraph pruned_graph({0, 1},
                         {
                             /* in, out, tmp */
                             {{0, 1}, {3}, {}},  // First op
                         },
                         {3});
  SwapGraph(&pruned_graph);
  Execute(0, 10);

  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));
}

TEST_F(ArenaPlannerTest, ModifiedGraph_DeallocateNonPersistentArena) {
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

  // Should be no-ops, since ReleaseNonPersistentMemory() hasn't been called.
  AcquireNonPersistentMemory();
  AcquireNonPersistentMemory();

  EXPECT_TRUE(HasNonPersistentMemory());

  // Release non-persistent arena.
  ReleaseNonPersistentMemory();
  EXPECT_FALSE(HasNonPersistentMemory());
  // Offsets should be zero.
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), 0);
  EXPECT_EQ(GetOffset(3), 0);

  // Now update the graph data used by the existing allocator. It should behave
  // as if it had been recreated with the new graph.
  TestGraph pruned_graph({0, 1},
                         {
                             /* in, out, tmp */
                             {{0, 1}, {3}, {}},  // First op
                         },
                         {3});
  SwapGraph(&pruned_graph);
  Execute(0, 10);

  // Should be a no-op.
  AcquireNonPersistentMemory();
  EXPECT_TRUE(HasNonPersistentMemory());

  // Release & acquire non-persistent memory.
  ReleaseNonPersistentMemory();
  AcquireNonPersistentMemory();
  // Offset checks from previous test should still apply.
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));
}

TEST_F(ArenaPlannerTest, ComplexGraph) {
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},
                      {{1}, {2}, {}},
                      {{1}, {3}, {}},
                      {{1}, {4}, {}},
                      {{2, 3, 4}, {5}, {}},
                      {{5}, {6}, {}},
                      {{5}, {7}, {}},
                      {{6, 7}, {8}, {}},
                  },
                  {8});
  (*graph.tensors())[0].bytes = 32;
  (*graph.tensors())[1].bytes = 28;
  (*graph.tensors())[2].bytes = 36;
  (*graph.tensors())[3].bytes = 16;
  (*graph.tensors())[4].bytes = 8;
  (*graph.tensors())[5].bytes = 64;
  (*graph.tensors())[6].bytes = 10;
  (*graph.tensors())[7].bytes = 40;
  SetGraph(&graph);
  Execute(0, 10);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +3 +4 -1 +5 -2 -3 -4 +6 +7 -5 +8
  EXPECT_EQ(GetOffset(5), 32);
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(7));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(8), 32);
}

TEST_F(ArenaPlannerTest, GraphWithIntermediates) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0}, {2}, {3}},
                      {{1, 2}, {4, 5}, {}},
                      {{5}, {6, 7}, {8, 9, 10}},
                      {{4, 6}, {11}, {12}},
                      {{11}, {13}, {}},
                      {{7, 13}, {14}, {15}},
                  },
                  {11, 14});
  SetGraph(&graph);
  Execute(0, 10);

  // Alloc(+) and dealloc(-) order by operation:
  // Op0: +0 +1 +2 +3 -3
  // Op1: +4 +5 -2 -4
  // Op2: +6 +7 +8 +9 +10 -8 -9 -10 -5
  // Op3: +11 +12 -12 -4 -6
  // Op4: +13
  // Op5: +14 +15 -7 -13 -15
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(15), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(14), GetOffsetAfter(15));
  EXPECT_EQ(GetOffset(13), GetOffsetAfter(14));
  EXPECT_EQ(GetOffset(12), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(11), GetOffsetAfter(13));
  EXPECT_EQ(GetOffset(10), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(10));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(11));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(8));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(7));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));

  // 2 is allocated in the smallest suitable gap, which is not equal to the
  // first available one.
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(5));
}

TEST_F(ArenaPlannerTest, DebugTensors) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {5}},  // First op, with temporary
                      {{2, 0}, {4}, {6}},  // Second op, with temporary
                      {{4}, {3}, {7}}      // Third op, with temporary
                  },
                  {3});
  SetGraph(&graph, /*preserve_all_tensors=*/false);
  Execute(0, 10);

  // Memory of temporary tensors are shared by default.
  EXPECT_EQ(GetOffset(5), GetOffset(6));
  EXPECT_EQ(GetOffset(6), GetOffset(7));

  SetGraph(&graph, /*preserve_all_tensors=*/true);
  Execute(0, 10);

  std::set<std::ptrdiff_t> tensorOffsets;
  for (int i = 0; i < 8; i++) {
    tensorOffsets.insert(GetOffset(i));
  }
  // Every tensor should have unique memory allocation with
  // preserve_all_tensors.
  EXPECT_EQ(tensorOffsets.size(), 8);
}

}  // namespace
}  // namespace tflite
