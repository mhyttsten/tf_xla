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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc() {
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

AttrValue FuncListAttr(const absl::Span<const char* const> names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/common_runtime/lower_case_op_test.cc", "FuncListAttr");

  AttrValue attr;
  for (const char* name : names) {
    attr.mutable_list()->add_func()->set_name(name);
  }
  return attr;
}

SessionOptions SessionOptionsWithInlining() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/common_runtime/lower_case_op_test.cc", "SessionOptionsWithInlining");

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  return session_options;
}

Status Rewrite(std::unique_ptr<Graph>* graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/common_runtime/lower_case_op_test.cc", "Rewrite");

  FunctionLibraryDefinition flib_def((*graph)->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options = SessionOptionsWithInlining();
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  LowerFunctionalOpsPass pass;
  return pass.Run(opt_options);
}

TEST(LowerCaseOpTest, Simple) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for then and else branch.
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();
  *(f_lib_proto.add_function()) = test::function::XTimesFour();
  *(f_lib_proto.add_function()) = test::function::XTimes16();

  // Construct simple conditional that switches on `pred` and operates only on
  // single input `A`.
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  auto branch_index =
      ops::Placeholder(root.WithOpName("branch_index"), DT_INT32);
  Node* written_if;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  TF_ASSERT_OK(
      NodeBuilder("case", "Case", &root.graph()->flib_def())
          .Input(branch_index.node())
          .Input(inputs)
          .Attr("branches",
                FuncListAttr({"XTimesTwo", "XTimesFour", "XTimes16"}))
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Attr("Tout", {DT_INT32})
          .Finalize(root.graph(), &written_if));
  TF_ASSERT_OK(root.DoShapeInference(written_if));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no switch or merge nodes.
  int node_called_case_count = 0;
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    if (op->name() == "case") {
      ++node_called_case_count;
    }
  }
  ASSERT_EQ(node_called_case_count, 1);

  TF_ASSERT_OK(Rewrite(&graph));

  // Verify the resultant graph has switch and merge nodes, and a node called
  // `if` (but not If nodes).
  int switch_count = 0;
  int merge_count = 0;
  node_called_case_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsSwitch()) {
      ++switch_count;
    }
    if (op->IsMerge()) {
      ++merge_count;
    }
    ASSERT_NE(op->type_string(), "Case");
    if (op->name() == "case") {
      ++node_called_case_count;
    }
  }
  // One switch for predicate and one for input (A).
  ASSERT_EQ(switch_count, 2);
  // One merge for the single output value of then and else, and one more merge
  // to enforce then and else function call execution (`branch_executed` node).
  ASSERT_EQ(merge_count, 2);
  ASSERT_EQ(node_called_case_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(-1));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_if)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(0));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_if)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(1));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_if)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 40);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(2));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_if)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(20));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_if)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
}

TEST(LowerCaseOpTest, BranchFunctionsWithoutOutputs) {
  using ::tensorflow::test::function::GDef;
  using ::tensorflow::test::function::NDef;
  using FDH = ::tensorflow::FunctionDefHelper;

  // Wrap AssignAddVariable + Const into a function.
  const auto assign_add = [](const string& fn_name, int v) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fn_name: \"" + fn_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_op_testDTcc mht_3(mht_3_v, 366, "", "./tensorflow/core/common_runtime/lower_case_op_test.cc", "lambda");

    const Tensor tensor = test::AsScalar<int32>(v);
    return FDH::Create(
        fn_name, {"v: resource"}, {}, {},
        {
            {{"c"}, "Const", {}, {{"value", tensor}, {"dtype", DT_INT32}}},
            {{"upd"},
             "AssignAddVariableOp",
             {"v", "c:output"},
             {{"dtype", DT_INT32}}},
        },
        /*ret_def=*/{},
        /*control_ret_def=*/{{"side_effects", "upd"}});
  };

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  // Add test functions for then and else branch.
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = assign_add("AddOne", 1);
  *(f_lib_proto.add_function()) = assign_add("AddTwo", 2);
  *(f_lib_proto.add_function()) = assign_add("AddTen", 10);

  // Construct a graph to represent following program:
  //
  //  (branch_index: int32, initial_val: int32) -> (int32) {
  //    var = Variable(initial_value)
  //    switch (branch_index) {
  //      case 0:
  //        var += 1; break;  # AddOne function call
  //      case 1:
  //        var += 2; break;  # AddTwo function call
  //      default:
  //        var += 10; break;  # AddTen function call
  //    }
  //    return var
  //  }
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));

  auto branch_index =
      ops::Placeholder(root.WithOpName("branch_index"), DT_INT32);
  auto initial_val = ops::Placeholder(root.WithOpName("initial_val"), DT_INT32);

  auto var = ops::VarHandleOp(root.WithOpName("var"), DT_INT32, {});
  auto init = ops::AssignVariableOp(root.WithOpName("init"), var, initial_val);

  Node* case_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(var.node())});
  TF_ASSERT_OK(
      NodeBuilder("case", "Case", &root.graph()->flib_def())
          .Input(branch_index.node())
          .Input(inputs)
          .ControlInput(init.operation.node())
          .Attr("branches", FuncListAttr({"AddOne", "AddTwo", "AddTen"}))
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Attr("Tout", DataTypeSlice{})
          .Finalize(root.graph(), &case_node));

  auto read = ops::ReadVariableOp(
      root.WithOpName("read").WithControlDependencies(Output(case_node)), var,
      DT_INT32);

  TF_ASSERT_OK(root.DoShapeInference(case_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(Rewrite(&graph));

  // Verify the resultant graph has switch, merge and function call nodes.
  int switch_count = 0;
  int merge_count = 0;
  int node_called_case_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsSwitch()) ++switch_count;
    if (op->IsMerge()) ++merge_count;
    if (op->name() == "case") ++node_called_case_count;
    ASSERT_NE(op->type_string(), "Case");
  }
  // One switch for predicate and one for input (A).
  ASSERT_EQ(switch_count, 2);
  // One merge for the else/then branch (`branch_executed` node).
  ASSERT_EQ(merge_count, 1);
  // We keep a NoOp with the same name as original If node.
  ASSERT_EQ(node_called_case_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(-5));
    feeds.emplace(Output(initial_val.node()), Input::Initializer(10));

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(read)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(0));
    feeds.emplace(Output(initial_val.node()), Input::Initializer(10));

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(read)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 11);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(1));
    feeds.emplace(Output(initial_val.node()), Input::Initializer(10));

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(read)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 12);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(2));
    feeds.emplace(Output(initial_val.node()), Input::Initializer(10));

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(read)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(31));
    feeds.emplace(Output(initial_val.node()), Input::Initializer(10));

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(read)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
  }
}

TEST(LowerCaseOpTest, DoNotInlineLoweredFunction) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDef x_times_two = test::function::XTimesTwo();
  FunctionDef x_times_four = test::function::XTimesFour();
  FunctionDef x_times_16 = test::function::XTimes16();

  // Case branches can't be inlined.
  (*x_times_two.mutable_attr())["_noinline"].set_b(true);
  (*x_times_four.mutable_attr())["_noinline"].set_b(true);
  (*x_times_16.mutable_attr())["_noinline"].set_b(true);

  // Add test functions for the branches.
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = x_times_two;
  *(f_lib_proto.add_function()) = x_times_four;
  *(f_lib_proto.add_function()) = x_times_16;

  // Construct simple conditional that switches on `pred` and operates only on
  // single input `A`.
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  auto branch_index =
      ops::Placeholder(root.WithOpName("branch_index"), DT_INT32);
  Node* written_case;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  TF_ASSERT_OK(
      NodeBuilder("case", "Case", &root.graph()->flib_def())
          .Input(branch_index.node())
          .Input(inputs)
          .Attr("branches",
                FuncListAttr({"XTimesTwo", "XTimesFour", "XTimes16"}))
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Attr("Tout", {DT_INT32})
          .Finalize(root.graph(), &written_case));
  TF_ASSERT_OK(root.DoShapeInference(written_case));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(Rewrite(&graph));

  // Verify that Case node was lowered but branch functions were not inlined.
  int x_times_two_count = 0;
  int x_times_four_count = 0;
  int x_times_16_count = 0;

  for (const auto* op : graph->op_nodes()) {
    if (op->type_string() == x_times_two.signature().name()) {
      x_times_two_count++;
    }
    if (op->type_string() == x_times_four.signature().name()) {
      x_times_four_count++;
    }
    if (op->type_string() == x_times_16.signature().name()) {
      x_times_16_count++;
    }
    ASSERT_NE(op->type_string(), "Case");
  }

  // One function for each branch.
  ASSERT_EQ(x_times_two_count, 1);
  ASSERT_EQ(x_times_four_count, 1);
  ASSERT_EQ(x_times_16_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(-2));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_case)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(0));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_case)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(1));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_case)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 40);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(2));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_case)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(branch_index.node()), Input::Initializer(31));
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(written_case)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 160);
  }
}

}  // namespace
}  // namespace tensorflow
