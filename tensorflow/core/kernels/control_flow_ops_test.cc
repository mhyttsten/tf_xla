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
class MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc() {
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

#include <string>

#include "tensorflow/core/common_runtime/forward_type_inference.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests for the switch op
class SwitchOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/control_flow_ops_test.cc", "Initialize");

    TF_ASSERT_OK(NodeDefBuilder("op", "Switch")
                     .Input(FakeInput(dt))
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SwitchOpTest, Int32Success_6_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, Int32Success_6_s1) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

TEST_F(SwitchOpTest, Int32Success_2_3_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, StringSuccess_s1) {
  Initialize(DT_STRING);
  AddInputFromArray<tstring>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<tstring>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

class AbortOpTest : public OpsTestBase {
 protected:
};

#ifdef PLATFORM_WINDOWS
#define SIGABRT 3

class KilledBySignal {
 public:
  explicit KilledBySignal(int signum) : signum_(signum) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc mht_1(mht_1_v, 269, "", "./tensorflow/core/kernels/control_flow_ops_test.cc", "KilledBySignal");
}
  bool operator()(int exit_status) const { return exit_status == signum_; }

 private:
  const int signum_;
};
#else
#define KilledBySignal ::testing::KilledBySignal
#endif

// Pass an error message to the op.
TEST_F(AbortOpTest, pass_error_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("error_msg", "abort_op_test")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; abort_op_test");
}

// Use the default error message.
TEST_F(AbortOpTest, default_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort").Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; ");
}

// Exit normally.
TEST_F(AbortOpTest, exit_normally) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("exit_without_error", true)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), ::testing::ExitedWithCode(0), "");
}

// Adds identity notes to all outputs of this node
static void add_identity_nodes(Node* node, Graph& graph,
                               std::vector<Node*>& identity_nodes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc mht_2(mht_2_v, 311, "", "./tensorflow/core/kernels/control_flow_ops_test.cc", "add_identity_nodes");

  for (int i = 0; i < node->num_outputs(); i++) {
    Node* new_node;
    std::string name = absl::StrCat("Identity", i);
    TF_EXPECT_OK(NodeBuilder(name, "Identity")
                     .Attr("T", node->output_type(i))
                     .Input(node, i)
                     .Finalize(&graph, &new_node));
    identity_nodes.push_back(new_node);
  }
}

// Runs type inference pass on graph
static Status type_inference(Graph& graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_ops_testDTcc mht_3(mht_3_v, 327, "", "./tensorflow/core/kernels/control_flow_ops_test.cc", "type_inference");

  GraphOptimizationPassOptions opt_options;
  std::unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  graph_ptr->Copy(graph);
  opt_options.graph = &graph_ptr;
  opt_options.flib_def = graph.mutable_flib_def();
  ForwardTypeInferencePass pass;
  return pass.Run(opt_options);
}

TEST(MergeOpTest, TypeInference) {
  GTEST_SKIP() << "TODO(b/222556864) fix \"Merge\" forward type inference "
               << "to support \"value_index\" special case";
  Graph graph(OpRegistry::Global());  // NOLINT(*-unreachable-code)
  protobuf::TextFormat::Parser parser;

  FullTypeDef input_dataset_t;
  Node* input_dataset1;
  Node* input_dataset2;
  Node* merge;
  CHECK(parser.ParseFromString(
      R"pb(type_id: TFT_PRODUCT
           args {
             type_id: TFT_DATASET
             args {
               type_id: TFT_PRODUCT
               args {
                 type_id: TFT_RAGGED
                 args { type_id: TFT_STRING }
               }
             }
           })pb",
      &input_dataset_t));
  TensorProto tensor_proto;
  TF_EXPECT_OK(NodeBuilder("input_dataset1", "Const")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_VARIANT)
                   .Finalize(&graph, &input_dataset1));
  (*input_dataset1->mutable_def()->mutable_experimental_type()) =
      input_dataset_t;
  TF_EXPECT_OK(NodeBuilder("input_dataset2", "Const")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_VARIANT)
                   .Finalize(&graph, &input_dataset2));
  (*input_dataset1->mutable_def()->mutable_experimental_type()) =
      input_dataset_t;

  TF_EXPECT_OK(NodeBuilder("Merge", "Merge")
                   .Attr("T", DT_VARIANT)
                   .Attr("N", 2)
                   .Input({input_dataset1, input_dataset2})
                   .Finalize(&graph, &merge));
  std::vector<Node*> identity_nodes;
  add_identity_nodes(merge, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset1->def().experimental_type()))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset1->def().experimental_type().DebugString();
}

}  // namespace
}  // namespace tensorflow
