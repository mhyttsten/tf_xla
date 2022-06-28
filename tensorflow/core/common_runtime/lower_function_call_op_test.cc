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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

AttrValue FuncAttr(const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/common_runtime/lower_function_call_op_test.cc", "FuncAttr");

  AttrValue attr;
  attr.mutable_func()->set_name(name);
  return attr;
}

AttrValue FuncAttr(const string& name, const DataType type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/common_runtime/lower_function_call_op_test.cc", "FuncAttr");

  AttrValue attr;
  attr.mutable_func()->set_name(name);
  (*attr.mutable_func()->mutable_attr())["T"].set_type(type);
  return attr;
}

SessionOptions SessionOptionsWithInlining() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/common_runtime/lower_function_call_op_test.cc", "SessionOptionsWithInlining");

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  return session_options;
}

Status Rewrite(std::unique_ptr<Graph>* graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_function_call_op_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/lower_function_call_op_test.cc", "Rewrite");

  FunctionLibraryDefinition flib_def((*graph)->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options = SessionOptionsWithInlining();
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  LowerFunctionalOpsPass pass;
  return pass.Run(opt_options);
}

TEST(LowerFunctionCallTest, InlineFunctionCall) {
  using FDH = FunctionDefHelper;

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDefLibrary f_lib_proto;

  // `add` node is not required to compute regular output `o`, but it must
  // execute because it is in `control_ret`.
  *(f_lib_proto.add_function()) =
      FDH::Create("AddAndMul", {"i: int32"}, {"o: int32"}, {},
                  {{{"add"}, "Add", {"i", "i"}, {{"T", DT_INT32}}},
                   {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_INT32}}}},
                  /*ret_def=*/{{"o", "ret:z:0"}},
                  /*control_ret_def=*/{{"must_execute", "add"}});

  // Construct a graph:
  //   A = Placeholder[dtype=int32]
  //   F = PartitionedCall[f=AddAndMul](a)
  //   B = Identity(func, ^func)
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  Node* function_call;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  TF_ASSERT_OK(NodeBuilder("F", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("AddAndMul"))
                   .Finalize(root.graph(), &function_call));
  TF_ASSERT_OK(root.DoShapeInference(function_call));

  auto b = ops::Identity(root.WithOpName("B"), Output(function_call, 0));
  root.graph()->AddControlEdge(function_call, b.node());

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Verify the resultant graph has no PartitionedCall ops and function body was
  // inlined into the main graph.
  int partitioned_call_count = 0;
  int add_count = 0;
  int mul_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsPartitionedCall()) partitioned_call_count++;
    if (op->type_string() == "Add") add_count++;
    if (op->type_string() == "Mul") mul_count++;
  }

  ASSERT_EQ(partitioned_call_count, 0);
  ASSERT_EQ(add_count, 1);
  ASSERT_EQ(mul_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(b)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 100);
  }
}

TEST(LowerFunctionCallTest, DoNotInlineTpuOrXlaFunctions) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDef tpu_func = test::function::XTimesTwo();
  tpu_func.mutable_signature()->set_name("TpuXTimesTwo");
  (*tpu_func.mutable_attr())["_tpu_replicate"].set_b(true);

  FunctionDef xla_func = test::function::XTimesTwo();
  xla_func.mutable_signature()->set_name("XlaXTimesTwo");
  (*xla_func.mutable_attr())["_xla_compile_id"].set_s("cluster_0");

  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();

  // Construct a graph:
  //   A = Placeholder[dtype=int32]
  //   B = XTimesTwo[_tpu_replicate="cluster"](A)
  //   C = XTimesTwo[_xla_compile_id="cluster"](A)
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* tpu_call;
  TF_ASSERT_OK(NodeBuilder("B", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("XTimesTwo", DT_INT32))
                   .Attr("_tpu_replicate", "cluster")
                   .Finalize(root.graph(), &tpu_call));

  Node* xla_call;
  TF_ASSERT_OK(NodeBuilder("C", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("XTimesTwo", DT_INT32))
                   .Attr("_xla_compile_id", "cluster")
                   .Finalize(root.graph(), &xla_call));

  TF_ASSERT_OK(root.DoShapeInference(tpu_call));
  TF_ASSERT_OK(root.DoShapeInference(xla_call));
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Verify that we do not inline any of the special function call nodes.
  int partitioned_call_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsPartitionedCall()) partitioned_call_count++;
  }
  ASSERT_EQ(partitioned_call_count, 2);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(
        session.Run(feeds, {Output(tpu_call), Output(xla_call)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 20);
  }
}

}  // namespace
}  // namespace tensorflow
