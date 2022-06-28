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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_while_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_while_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_while_op_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/match.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

SessionOptions SessionOptionsWithInlining() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_while_op_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/common_runtime/lower_while_op_test.cc", "SessionOptionsWithInlining");

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  return session_options;
}

Status Rewrite(std::unique_ptr<Graph>* graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_while_op_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/common_runtime/lower_while_op_test.cc", "Rewrite");

  FunctionLibraryDefinition flib_def((*graph)->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options = SessionOptionsWithInlining();
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  LowerFunctionalOpsPass pass;
  return pass.Run(opt_options);
}

TEST(LowerWhileOpTest, Simple) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::LessThanOrEqualToN(8);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwo");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("T", {DT_INT32})
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr("parallel_iterations", 100)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(root.graph(), &while_node));
  auto c = ops::Identity(
      root.WithOpName("C").WithControlDependencies(Output(while_node)),
      Output(while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no lower level control flow primitives.
  int node_called_while_count = 0;
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsEnter());
    ASSERT_FALSE(op->IsExit());
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    ASSERT_FALSE(op->IsNextIteration());
    ASSERT_FALSE(op->IsLoopCond());
    if (op->name() == "while") {
      node_called_while_count++;
    }
  }
  ASSERT_EQ(node_called_while_count, 1);

  TF_ASSERT_OK(Rewrite(&graph));

  int enter_count = 0;
  int exit_count = 0;
  int switch_count = 0;
  int merge_count = 0;
  int next_iteration_count = 0;
  node_called_while_count = 0;
  int less_than_or_equan_to_n_count = 0;
  int x_times_two_count = 0;

  for (const auto* op : graph->op_nodes()) {
    if (op->IsEnter()) {
      ++enter_count;
      ASSERT_EQ(op->attrs().Find("parallel_iterations")->i(), 100);
    }
    if (op->IsExit()) {
      ++exit_count;
    }
    if (op->IsSwitch()) {
      ++switch_count;
    }
    if (op->IsMerge()) {
      ++merge_count;
    }
    if (op->IsNextIteration()) {
      ++next_iteration_count;
    }
    if (op->name() == "while") {
      node_called_while_count++;
    }
    if (op->type_string() == "LessThanOrEqualToN") {
      less_than_or_equan_to_n_count++;
    }
    if (op->type_string() == "XTimesTwo") {
      x_times_two_count++;
    }
    if (op->name() == "C") {
      ASSERT_EQ(op->in_edges().size(), 2);
    }
    ASSERT_NE(op->type_string(), "While");
  }
  // One node per loop input.
  ASSERT_EQ(enter_count, 1);
  ASSERT_EQ(exit_count, 1);
  ASSERT_EQ(switch_count, 1);
  ASSERT_EQ(merge_count, 1);
  ASSERT_EQ(next_iteration_count, 1);
  ASSERT_EQ(node_called_while_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 16);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(3));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 12);
  }
}

TEST(LowerWhileOpTest, ForwardAssignedInputDevice) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::LessThanOrEqualToN(8);

  TF_ASSERT_OK(graph->AddFunctionLibrary(f_lib_proto));
  auto type = DT_FLOAT;
  Node* placeholder;
  TF_CHECK_OK(NodeBuilder("placed_node", "Placeholder")
                  .Attr("dtype", type)
                  .Finalize(graph.get(), &placeholder));
  const string assigned_device_name = "/job:localhost/replica:0/task:0/gpu:0";
  placeholder->set_assigned_device_name(assigned_device_name);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(placeholder)});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwo");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &graph->flib_def())
          .Input(inputs)
          .Attr("T", {type})
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr("parallel_iterations", 100)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(graph.get(), &while_node));
  TF_ASSERT_OK(Rewrite(&graph));

  const Node* placeholder_node = nullptr;
  for (const auto* op : graph->op_nodes()) {
    if (op->name() == "placed_node") {
      placeholder_node = op;
    }
  }
  ASSERT_NE(placeholder_node, nullptr);
  // Verify the assigned device of the Enter node.
  int enter_consumers = 0;
  const Node* enter_node = nullptr;
  for (const Node* consumer : placeholder_node->out_nodes()) {
    if (consumer->type_string() == "Enter") {
      enter_consumers += 1;
      enter_node = consumer;
      ASSERT_EQ(consumer->assigned_device_name(), assigned_device_name);
    }
  }
  ASSERT_EQ(enter_consumers, 1);
  // Verify the assigned device of the Merge node.
  int merge_consumers = 0;
  const Node* merge_node = nullptr;
  for (const Node* consumer : enter_node->out_nodes()) {
    if (consumer->type_string() == "Merge") {
      merge_consumers += 1;
      merge_node = consumer;
      ASSERT_EQ(consumer->assigned_device_name(), assigned_device_name);
    }
  }
  ASSERT_EQ(merge_consumers, 1);
  // Verify the assigned device of the NextIteration node.
  int next_iteration_consumers = 0;
  for (const Node* consumer : merge_node->in_nodes()) {
    if (consumer->type_string() == "NextIteration") {
      next_iteration_consumers += 1;
      ASSERT_EQ(consumer->assigned_device_name(), assigned_device_name);
    }
  }
  ASSERT_EQ(next_iteration_consumers, 1);
  // Verify the assigned device of the Switch node.
  int switch_consumers = 0;
  const Node* switch_node = nullptr;
  for (const Node* consumer : merge_node->out_nodes()) {
    if (consumer->type_string() == "Switch") {
      switch_consumers += 1;
      switch_node = consumer;
      ASSERT_EQ(consumer->assigned_device_name(), assigned_device_name);
    }
  }
  ASSERT_EQ(switch_consumers, 1);
  // Verify the assigned device of the Exit node.
  int exit_consumers = 0;
  for (const Node* consumer : switch_node->out_nodes()) {
    if (consumer->type_string() == "Exit") {
      exit_consumers += 1;
      ASSERT_EQ(consumer->assigned_device_name(), assigned_device_name);
    }
  }
  ASSERT_EQ(exit_consumers, 1);
}

TEST(LowerWhileOpTest, ForwardRequestedInputDevice) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::LessThanOrEqualToN(8);

  TF_ASSERT_OK(graph->AddFunctionLibrary(f_lib_proto));
  auto type = DT_FLOAT;
  // We will place the loop var on the gpu:0.
  const string gpu_0_device = "/job:localhost/replica:0/task:0/gpu:0";
  // We will place loop's control input on the gpu:1.
  const string gpu_1_device = "/job:localhost/replica:0/task:0/gpu:1";
  // We will place While op on gpu:2.
  const string gpu_2_device = "/job:localhost/replica:0/task:0/gpu:2";
  Node* gpu_0_ph;
  TF_CHECK_OK(NodeBuilder("placed_node", "Placeholder")
                  .Attr("dtype", type)
                  .Device(gpu_0_device)
                  .Finalize(graph.get(), &gpu_0_ph));
  Node* control_in;
  // Add a control input to the While op to trigger the creation of a
  // LoopExecuted node.
  TF_CHECK_OK(NodeBuilder("control_in", "Placeholder")
                  .Attr("dtype", type)
                  .Device(gpu_1_device)
                  .Finalize(graph.get(), &control_in));
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(gpu_0_ph)});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwo");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &graph->flib_def())
          .Input(inputs)
          .ControlInput(control_in)
          .Device(gpu_2_device)
          .Attr("T", {type})
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr("parallel_iterations", 100)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(graph.get(), &while_node));

  // Create an empty Const node with control dep from the While op.
  // This triggers the creation of a LoopExecuted node.
  Node* control_out;
  TensorProto proto;
  proto.set_dtype(DT_FLOAT);
  TensorShape empty_shape({0});
  empty_shape.AsProto(proto.mutable_tensor_shape());
  TF_ASSERT_OK(NodeBuilder("control_out", "Const")
                   .ControlInput(while_node)
                   .Attr("dtype", DT_FLOAT)
                   .Attr("value", proto)
                   .Finalize(graph.get(), &control_out));

  TF_ASSERT_OK(Rewrite(&graph));

  const Node* placeholder_node = nullptr;
  for (const auto* op : graph->op_nodes()) {
    if (op->name() == "placed_node") {
      placeholder_node = op;
    }
  }
  ASSERT_NE(placeholder_node, nullptr);
  // Verify the requested device of the Enter node.
  int enter_consumers = 0;
  const Node* enter_node = nullptr;
  for (const Node* consumer : placeholder_node->out_nodes()) {
    if (consumer->type_string() == "Enter") {
      enter_consumers += 1;
      enter_node = consumer;
      ASSERT_EQ(consumer->requested_device(), gpu_0_device);
    }
  }
  ASSERT_EQ(enter_consumers, 1);
  // Verify the requested device of the Merge node.
  int merge_consumers = 0;
  const Node* merge_node = nullptr;
  for (const Node* consumer : enter_node->out_nodes()) {
    if (consumer->type_string() == "Merge") {
      merge_consumers += 1;
      merge_node = consumer;
      ASSERT_EQ(consumer->requested_device(), gpu_0_device);
    }
  }
  ASSERT_EQ(merge_consumers, 1);
  // Verify the requested device of the NextIteration node.
  int next_iteration_consumers = 0;
  for (const Node* consumer : merge_node->in_nodes()) {
    if (consumer->type_string() == "NextIteration") {
      next_iteration_consumers += 1;
      ASSERT_EQ(consumer->requested_device(), gpu_0_device);
    }
  }
  ASSERT_EQ(next_iteration_consumers, 1);
  // Verify the requested device of the Switch node.
  int switch_consumers = 0;
  const Node* switch_node = nullptr;
  for (const Node* consumer : merge_node->out_nodes()) {
    if (consumer->type_string() == "Switch") {
      switch_consumers += 1;
      switch_node = consumer;
      ASSERT_EQ(consumer->requested_device(), gpu_0_device);
    }
  }
  ASSERT_EQ(switch_consumers, 1);
  // Verify the requested device of the Exit node.
  int exit_consumers = 0;
  for (const Node* consumer : switch_node->out_nodes()) {
    if (consumer->type_string() == "Exit") {
      exit_consumers += 1;
      ASSERT_EQ(consumer->requested_device(), gpu_0_device);
    }
  }
  ASSERT_EQ(exit_consumers, 1);
  // Verify the requested device of LoopControlInputs.
  const Node* loop_control_inputs_node = nullptr;
  for (const auto* op : graph->op_nodes()) {
    if (absl::StrContains(op->name(), "LoopControlInputs")) {
      loop_control_inputs_node = op;
    }
  }
  ASSERT_NE(loop_control_inputs_node, nullptr);
  ASSERT_EQ(loop_control_inputs_node->requested_device(), gpu_2_device);
  // Verify the requested device of LoopExecuted.
  const Node* loop_executed_node = nullptr;
  for (const auto* op : graph->op_nodes()) {
    if (absl::StrContains(op->name(), "LoopExecuted")) {
      loop_executed_node = op;
    }
  }
  ASSERT_NE(loop_executed_node, nullptr);
  ASSERT_EQ(loop_executed_node->requested_device(), gpu_2_device);
}

TEST(LowerWhileOpTest, MultipleInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XPlusOneXTimesY();
  *(f_lib_proto.add_function()) = test::function::XYXLessThanOrEqualToN(4);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  auto b = ops::Placeholder(root.WithOpName("B"), DT_INT32);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs(
      {NodeBuilder::NodeOut(a.node()), NodeBuilder::NodeOut(b.node())});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("XYXLessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XPlusOneXTimesY");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("T", {DT_INT32, DT_INT32})
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(root.graph(), &while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no lower level control flow primitives.
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsEnter());
    ASSERT_FALSE(op->IsExit());
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    ASSERT_FALSE(op->IsNextIteration());
    ASSERT_FALSE(op->IsLoopCond());
  }

  TF_ASSERT_OK(Rewrite(&graph));

  int enter_count = 0;
  int exit_count = 0;
  int switch_count = 0;
  int merge_count = 0;
  int next_iteration_count = 0;
  int x_plus_one_x_times_y_count = 0;
  int x_y_x_less_than_equal_to_n_count = 0;

  for (const auto* op : graph->op_nodes()) {
    if (op->IsEnter()) {
      ++enter_count;
    }
    if (op->IsExit()) {
      ++exit_count;
    }
    if (op->IsSwitch()) {
      ++switch_count;
    }
    if (op->IsMerge()) {
      ++merge_count;
    }
    if (op->IsNextIteration()) {
      ++next_iteration_count;
    }
    if (op->type_string() == "XPlusOneXTimesY") {
      x_plus_one_x_times_y_count++;
    }
    if (op->type_string() == "XYXLessThanOrEqualToN") {
      x_y_x_less_than_equal_to_n_count++;
    }
    ASSERT_NE(op->type_string(), "While");
  }
  // Two nodes per loop input.
  ASSERT_EQ(enter_count, 2);
  ASSERT_EQ(exit_count, 2);
  ASSERT_EQ(switch_count, 2);
  ASSERT_EQ(merge_count, 2);
  ASSERT_EQ(next_iteration_count, 2);
  ASSERT_EQ(x_plus_one_x_times_y_count, 0);
  ASSERT_EQ(x_y_x_less_than_equal_to_n_count, 0);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    feeds.emplace(Output(b.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(
        feeds, {Output(while_node, 0), Output(while_node, 1)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 5);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 24);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(3));
    feeds.emplace(Output(b.node()), Input::Initializer(5));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(
        feeds, {Output(while_node, 0), Output(while_node, 1)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 5);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 60);
  }
}

TEST(LowerWhileOpTest, DoNotInlineLoweredFunctions) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDef x_times_two = test::function::XTimesTwo();
  FunctionDef less_than_or_eq = test::function::LessThanOrEqualToN(8);

  // While loop `cond` and `body` nodes can't be inlined.
  (*x_times_two.mutable_attr())["_noinline"].set_b(true);
  (*less_than_or_eq.mutable_attr())["_noinline"].set_b(true);

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = x_times_two;
  *f_lib_proto.add_function() = less_than_or_eq;

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwo");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("T", {DT_INT32})
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr("parallel_iterations", 100)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(root.graph(), &while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(Rewrite(&graph));

  // Verify that while node was lowered but functions were not inlined.
  int x_times_two_count = 0;
  int less_than_or_eq_count = 0;

  for (const auto* op : graph->op_nodes()) {
    if (op->type_string() == x_times_two.signature().name()) {
      x_times_two_count++;
    }
    if (op->type_string() == less_than_or_eq.signature().name()) {
      less_than_or_eq_count++;
    }
    ASSERT_NE(op->type_string(), "While");
  }

  ASSERT_EQ(x_times_two_count, 1);
  ASSERT_EQ(less_than_or_eq_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 16);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(3));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 12);
  }
}

}  // namespace
}  // namespace tensorflow
