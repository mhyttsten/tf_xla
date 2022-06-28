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
class MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc() {
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

#include "tensorflow/compiler/jit/compilability_check_util.h"

#include "absl/memory/memory.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

AttrValue FuncListAttr(const absl::Span<const char* const> names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/jit/compilability_check_util_test.cc", "FuncListAttr");

  AttrValue attr;
  for (const char* name : names) {
    attr.mutable_list()->add_func()->set_name(name);
  }
  return attr;
}

constexpr char kFunctionalIfNodeName[] = "If";
constexpr char kFunctionalCaseNodeName[] = "Case";
constexpr char kFunctionalWhileNodeName[] = "While";
constexpr char kCompilableFunctionName[] = "CompilableFn";
constexpr char kCompilableFunctionNodeName[] = "n_c";
constexpr char kUncompilableFunctionName[] = "UncompilableFn";
constexpr char kUncompilableFunctionNodeName[] = "n_c_uncompilable";
constexpr char kUncompilableFunctionTwoName[] = "UncompilableFnTwo";
constexpr char kUncompilableFunctionNodeTwoName[] = "n_d_uncompilable";

// A dummy OpKernel for testing.
class DummyCompilableOp : public XlaOpKernel {
 public:
  explicit DummyCompilableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/jit/compilability_check_util_test.cc", "DummyCompilableOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/jit/compilability_check_util_test.cc", "Compile");

    ctx->SetOutput(0, ctx->Input(0));
  }
};

// Register the DummyCompilableOp kernel for CPU.
REGISTER_OP("InputFloatOp").Output("o: float");
REGISTER_OP("CompilableOp").Input("i: float").Output("o: float");
REGISTER_XLA_OP(Name("CompilableOp").Device(DEVICE_CPU_XLA_JIT),
                DummyCompilableOp);

// Dummy op that is uncompilable in CPU.
REGISTER_OP("MissingKernel").Input("i: float").Output("o: float");

class CompilabilityCheckUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/jit/compilability_check_util_test.cc", "SetUp");

    XlaOpRegistry::RegisterCompilationKernels();

    op_filter_.allow_resource_ops_in_called_functions = false;
    op_filter_.allow_stack_ops = false;
    op_filter_.allow_tensor_array_ops = false;
    op_filter_.allow_stateful_rng_ops = false;
    op_filter_.allow_control_trigger = false;
    op_filter_.allow_eliding_assert_and_checknumerics_ops = false;
    op_filter_.allow_ops_producing_or_consuming_variant = false;
    op_filter_.allow_inaccurate_ops = false;
    op_filter_.allow_slow_ops = false;
    op_filter_.allow_outside_compiled = false;

    checker_ = CreateCompilabilityChecker();
  }

  std::unique_ptr<RecursiveCompilabilityChecker> CreateCompilabilityChecker() {
    return absl::make_unique<RecursiveCompilabilityChecker>(op_filter_,
                                                            device_type_);
  }

  FunctionLibraryRuntime* GetFunctionLibraryRuntime() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_util_testDTcc mht_4(mht_4_v, 278, "", "./tensorflow/compiler/jit/compilability_check_util_test.cc", "GetFunctionLibraryRuntime");

    OptimizerOptions opts;
    pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        nullptr, Env::Default(), /*config=*/nullptr, TF_GRAPH_DEF_VERSION,
        flib_def_.get(), opts);

    return pflr_->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  }

  RecursiveCompilabilityChecker::OperationFilter op_filter_;
  DeviceType device_type_ = DeviceType(DEVICE_CPU_XLA_JIT);
  std::unique_ptr<FunctionDefLibrary> func_library_ =
      absl::make_unique<FunctionDefLibrary>();
  std::unique_ptr<FunctionLibraryDefinition> flib_def_ =
      absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
                                                   *func_library_);
  std::unique_ptr<RecursiveCompilabilityChecker> checker_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(CompilabilityCheckUtilTest, CheckNonFunctionalNodes) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  auto opts = builder.opts();
  Node* const0 = ops::SourceOp("InputFloatOp", opts);
  Node* compilable_op = ops::UnaryOp("CompilableOp", const0, opts);
  Node* uncompilable_op = ops::UnaryOp("MissingKernel", compilable_op, opts);
  GraphDef graph_def;
  TF_EXPECT_OK(builder.ToGraphDef(&graph_def));

  auto* flib_runtime = GetFunctionLibraryRuntime();
  // Source node is not compilable.
  EXPECT_FALSE(checker_->IsCompilableNode(*const0, flib_runtime));

  EXPECT_TRUE(checker_->IsCompilableNode(*compilable_op, flib_runtime));

  // Uncompilable as we are only checking compilability in CPU device type.
  EXPECT_FALSE(checker_->IsCompilableNode(*uncompilable_op, flib_runtime));

  const auto uncompilable_nodes =
      checker_->FindUncompilableNodes(*uncompilable_op, flib_runtime);
  ASSERT_EQ(1, uncompilable_nodes.size());
  auto node_info_it =
      uncompilable_nodes.find(NameAttrList().ShortDebugString());
  ASSERT_NE(uncompilable_nodes.end(), node_info_it);
  const auto& uncompilable_nodes_inside_function = node_info_it->second.second;
  ASSERT_EQ(1, uncompilable_nodes_inside_function.size());
  const auto& uncompilable_node_info = uncompilable_nodes_inside_function.at(0);
  EXPECT_TRUE(absl::StrContains(uncompilable_node_info.uncompilable_reason,
                                "unsupported op"));
  ASSERT_EQ(1, uncompilable_node_info.stack_trace.size());
  ASSERT_EQ("", uncompilable_node_info.stack_trace.at(0).function_name);
}

TEST_F(CompilabilityCheckUtilTest, CheckOutsideCompiledNode) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  auto opts = builder.opts();
  Node* const0 = ops::SourceOp("InputFloatOp", opts);
  Node* uncompilable_op = ops::UnaryOp("MissingKernel", const0, opts);
  uncompilable_op->AddAttr("_xla_outside_compilation", "0");
  GraphDef graph_def;
  TF_EXPECT_OK(builder.ToGraphDef(&graph_def));

  auto* flib_runtime = GetFunctionLibraryRuntime();

  // Outside compiled ops are considered by default..
  EXPECT_FALSE(checker_->IsCompilableNode(*uncompilable_op, flib_runtime));

  const auto uncompilable_nodes =
      checker_->FindUncompilableNodes(*uncompilable_op, flib_runtime);
  ASSERT_EQ(1, uncompilable_nodes.size());

  op_filter_.allow_outside_compiled = true;
  checker_ = CreateCompilabilityChecker();
  // With filter option outside compiled ops are ignored and considered
  // compilable.
  EXPECT_TRUE(checker_->IsCompilableNode(*uncompilable_op, flib_runtime));

  const auto uncompilable_nodes2 =
      checker_->FindUncompilableNodes(*uncompilable_op, flib_runtime);
  ASSERT_EQ(0, uncompilable_nodes2.size());
}

TEST_F(CompilabilityCheckUtilTest, CheckSimpleFunctionNode) {
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_c_uncompilable:float"},
      /*Attributes*/ {},
      // Node info
      {{{kUncompilableFunctionNodeName}, "MissingKernel", {"n_a"}}});
  flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, flib_def_.get());
  std::unique_ptr<Graph> graph(new Graph(flib_def_.get()));
  Node* const0 = ops::SourceOp("InputFloatOp", builder.opts());
  Node* functional_node = ops::UnaryOp(kUncompilableFunctionName, const0,
                                       builder.opts().WithName("D"));
  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));

  auto* flib_runtime = GetFunctionLibraryRuntime();
  EXPECT_FALSE(checker_->IsCompilableNode(*functional_node, flib_runtime));
  const auto uncompilable_nodes =
      checker_->FindUncompilableNodes(*functional_node, flib_runtime);

  EXPECT_EQ(1, uncompilable_nodes.size());
  NameAttrList function;
  function.set_name(kUncompilableFunctionName);
  const auto node_info_it =
      uncompilable_nodes.find(function.ShortDebugString());
  ASSERT_NE(uncompilable_nodes.end(), node_info_it);
  const auto& uncompilable_node_list = node_info_it->second.second;
  ASSERT_EQ(1, uncompilable_node_list.size());
  const auto& node_info = uncompilable_node_list.at(0);
  const auto& node_stack = node_info.stack_trace;
  ASSERT_EQ(2, node_stack.size());
  EXPECT_EQ("D", node_stack.at(0).name);
  EXPECT_EQ(kUncompilableFunctionNodeName, node_stack.at(1).name);
  EXPECT_EQ(kUncompilableFunctionNodeName, node_info.name);
  EXPECT_TRUE(
      absl::StrContains(node_info.uncompilable_reason, "unsupported op"));
}

TEST_F(CompilabilityCheckUtilTest, CheckFunctionalWhileNode) {
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kCompilableFunctionName,
      /*Inputs*/ {"n_a:float", "n_b:float"},
      /*Outputs*/ {"n_c:float"},
      /*Attribute*/ {},
      // Node info
      {{{kCompilableFunctionNodeName},
        "Add",
        {"n_a", "n_b"},
        {{"T", DT_FLOAT}}}});
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_c_uncompilable:float"},
      /*Attributes*/ {},
      // Node info
      {{{kUncompilableFunctionNodeName}, "MissingKernel", {"n_a"}}});

  flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, flib_def_.get());

  Node* const0 = ops::SourceOp("InputFloatOp", builder.opts());
  Node* input_node = ops::UnaryOp("CompilableOp", const0, builder.opts());

  NameAttrList compilable;
  compilable.set_name(kCompilableFunctionName);
  NameAttrList uncompilable;
  uncompilable.set_name(kUncompilableFunctionName);

  NodeBuilder while_builder(kFunctionalWhileNodeName, "While",
                            builder.opts().op_registry());
  while_builder.Input({input_node, input_node})
      .Attr("cond", compilable)
      .Attr("body", uncompilable);
  builder.opts().FinalizeBuilder(&while_builder);

  GraphDef graph_def;
  TF_EXPECT_OK(builder.ToGraphDef(&graph_def));
  std::unique_ptr<Graph> graph(new Graph(flib_def_.get()));
  TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));

  auto while_node_it = std::find_if(
      graph->nodes().begin(), graph->nodes().end(),
      [&](const Node* n) { return n->name() == kFunctionalWhileNodeName; });
  EXPECT_NE(while_node_it, graph->nodes().end());

  auto* flib_runtime = GetFunctionLibraryRuntime();

  EXPECT_FALSE(checker_->IsCompilableNode(**while_node_it, flib_runtime));
  const auto uncompilable_nodes =
      checker_->FindUncompilableNodes(**while_node_it, flib_runtime);
  ASSERT_EQ(1, uncompilable_nodes.size());

  NameAttrList function;
  function.set_name(kUncompilableFunctionName);
  const auto node_info_it =
      uncompilable_nodes.find(function.ShortDebugString());
  ASSERT_NE(uncompilable_nodes.end(), node_info_it);
  const auto& uncompilable_node_list = node_info_it->second.second;
  ASSERT_EQ(1, uncompilable_node_list.size());
  const auto& node_info = uncompilable_node_list.at(0);

  const auto& node_stack = node_info.stack_trace;
  ASSERT_EQ(2, node_stack.size());
  const auto& stacktrace_first_node_info = node_stack.at(0);
  EXPECT_EQ(kFunctionalWhileNodeName, stacktrace_first_node_info.name);
  EXPECT_EQ("", stacktrace_first_node_info.function_name);

  const auto& stacktrace_second_node_info = node_stack.at(1);
  EXPECT_EQ(kUncompilableFunctionNodeName, stacktrace_second_node_info.name);
  EXPECT_EQ(kUncompilableFunctionName,
            stacktrace_second_node_info.function_name);

  EXPECT_EQ(kUncompilableFunctionNodeName, node_info.name);
  EXPECT_TRUE(
      absl::StrContains(node_info.uncompilable_reason, "unsupported op"));
}

TEST_F(CompilabilityCheckUtilTest, CheckFunctionalIfNode) {
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_c_uncompilable:float"},
      /*Attributes*/ {},
      // Node info
      {{{kUncompilableFunctionNodeName}, "MissingKernel", {"n_a"}}});
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionTwoName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_d_uncompilable:float"},
      /*Attribute*/ {},
      // Node info
      {{{kUncompilableFunctionNodeTwoName}, "MissingKernel", {"n_a"}}});
  NameAttrList uncompilable_fn1_attr;
  uncompilable_fn1_attr.set_name(kUncompilableFunctionName);
  NameAttrList uncompilable_fn2_attr;
  uncompilable_fn2_attr.set_name(kUncompilableFunctionTwoName);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib));
  auto predicate = ops::Placeholder(root.WithOpName("pred"), DT_BOOL);
  auto placeholder = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> if_inputs(
      {NodeBuilder::NodeOut(placeholder.node())});
  Node* if_node;
  TF_ASSERT_OK(
      NodeBuilder(kFunctionalIfNodeName, "If", &root.graph()->flib_def())
          .Input(predicate.node())
          .Input(if_inputs)
          .Attr("then_branch", uncompilable_fn1_attr)
          .Attr("else_branch", uncompilable_fn2_attr)
          .Attr("Tout", {DT_INT32})
          .Finalize(root.graph(), &if_node));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  auto if_node_it = std::find_if(
      graph->nodes().begin(), graph->nodes().end(),
      [&](const Node* n) { return n->name() == kFunctionalIfNodeName; });
  EXPECT_NE(if_node_it, graph->nodes().end());
  auto* flib_runtime = GetFunctionLibraryRuntime();

  EXPECT_FALSE(checker_->IsCompilableNode(**if_node_it, flib_runtime));
  const auto uncompilable_nodes =
      checker_->FindUncompilableNodes(**if_node_it, flib_runtime);
  ASSERT_EQ(2, uncompilable_nodes.size());

  NameAttrList function_one;
  function_one.set_name(kUncompilableFunctionName);
  auto it = uncompilable_nodes.find(function_one.ShortDebugString());
  ASSERT_NE(uncompilable_nodes.end(), it);

  const auto& uncompilable_node_list = it->second.second;
  ASSERT_EQ(1, uncompilable_node_list.size());
  const auto& uncompilable_node_one = uncompilable_node_list.at(0);
  const auto& node_one_stack = uncompilable_node_one.stack_trace;

  ASSERT_EQ(2, node_one_stack.size());
  const auto& node_one_stacktrace_first_node = node_one_stack.at(0);
  EXPECT_EQ(kFunctionalIfNodeName, node_one_stacktrace_first_node.name);
  EXPECT_EQ("", node_one_stacktrace_first_node.function_name);

  const auto& stacktrace_second_node_info = node_one_stack.at(1);
  EXPECT_EQ(kUncompilableFunctionNodeName, stacktrace_second_node_info.name);
  EXPECT_EQ(kUncompilableFunctionName,
            stacktrace_second_node_info.function_name);

  EXPECT_EQ(kUncompilableFunctionNodeName, uncompilable_node_one.name);
  EXPECT_TRUE(absl::StrContains(uncompilable_node_one.uncompilable_reason,
                                "unsupported op"));

  NameAttrList function_two;
  function_two.set_name(kUncompilableFunctionTwoName);
  it = uncompilable_nodes.find(function_two.ShortDebugString());
  ASSERT_NE(uncompilable_nodes.end(), it);

  const auto& uncompilable_node_two_list = it->second.second;
  ASSERT_EQ(1, uncompilable_node_two_list.size());
  const auto& uncompilable_node_two = uncompilable_node_two_list.at(0);
  const auto& node_two_stack = uncompilable_node_two.stack_trace;
  ASSERT_EQ(2, node_two_stack.size());
  const auto& node_two_stacktrace_first_node = node_two_stack.at(0);
  EXPECT_EQ(kFunctionalIfNodeName, node_two_stacktrace_first_node.name);
  EXPECT_EQ("", node_two_stacktrace_first_node.function_name);

  const auto& node_two_stacktrace_second_node = node_two_stack.at(1);
  EXPECT_EQ(kUncompilableFunctionNodeTwoName,
            node_two_stacktrace_second_node.name);
  EXPECT_EQ(kUncompilableFunctionTwoName,
            node_two_stacktrace_second_node.function_name);

  EXPECT_EQ(kUncompilableFunctionNodeTwoName, uncompilable_node_two.name);
  EXPECT_TRUE(absl::StrContains(uncompilable_node_one.uncompilable_reason,
                                "unsupported op"));
}

TEST_F(CompilabilityCheckUtilTest, CheckFunctionalCaseNode) {
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_c_uncompilable:float"},
      /*Attributes*/ {},
      // Node info
      {{{kUncompilableFunctionNodeName}, "MissingKernel", {"n_a"}}});
  *flib.add_function() = FunctionDefHelper::Define(
      /*Function*/ kUncompilableFunctionTwoName,
      /*Inputs*/ {"n_a:float"},
      /*Outputs*/ {"n_d_uncompilable:float"},
      /*Attribute*/ {},
      // Node info
      {{{kUncompilableFunctionNodeTwoName}, "MissingKernel", {"n_a"}}});

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib));
  auto branch_index = ops::Placeholder(root.WithOpName("pred"), DT_INT32);
  auto placeholder = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> inputes(
      {NodeBuilder::NodeOut(placeholder.node())});
  Node* case_node;
  TF_ASSERT_OK(
      NodeBuilder(kFunctionalCaseNodeName, "Case", &root.graph()->flib_def())
          .Input(branch_index.node())
          .Input(inputes)
          .Attr("branches", FuncListAttr({kUncompilableFunctionName,
                                          kUncompilableFunctionTwoName}))
          .Attr("Tout", {DT_INT32})
          .Finalize(root.graph(), &case_node));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  auto case_node_it = std::find_if(
      graph->nodes().begin(), graph->nodes().end(),
      [&](const Node* n) { return n->name() == kFunctionalCaseNodeName; });
  EXPECT_NE(case_node_it, graph->nodes().end());
  auto* flib_runtime = GetFunctionLibraryRuntime();

  op_filter_.require_always_compilable = false;
  checker_ = CreateCompilabilityChecker();
  EXPECT_TRUE(checker_->IsCompilableNode(**case_node_it, flib_runtime));
  op_filter_.require_always_compilable = true;
  checker_ = CreateCompilabilityChecker();
  EXPECT_FALSE(checker_->IsCompilableNode(**case_node_it, flib_runtime));
}

TEST_F(CompilabilityCheckUtilTest, TestCanNotTriggerXlaCompilation) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;

  FunctionDef identity_func = FunctionDefHelper::Create(
      "IdentityFunc",
      /*in_def=*/{"x:float"},
      /*out_def=*/{"res:float"},
      /*attr_def=*/{},
      /*node_def=*/{{{"t0"}, "Identity", {"x"}, {{"T", DT_FLOAT}}}},
      /*ret_def*/ {{"res", "t0:output"}});

  *library.add_function() = identity_func;

  Output in = ops::Placeholder(root, DT_FLOAT);
  NameAttrList b_name_attr;
  b_name_attr.set_name("IdentityFunc");
  ops::PartitionedCall call(root.WithOpName("call"), {in}, {DT_FLOAT},
                            b_name_attr);

  GraphDef graph_def;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  EXPECT_FALSE(CanTriggerXlaCompilation(graph_def));
}

TEST_F(CompilabilityCheckUtilTest, TestXlaOpsCanTriggerXlaCompilation) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;

  FunctionDef sort_func = FunctionDefHelper::Create(
      "SortFunc",
      /*in_def=*/{"x:float"},
      /*out_def=*/{"res:float"},
      /*attr_def=*/{},
      /*node_def=*/{{{"t0"}, "XlaSort", {"x"}, {{"T", DT_FLOAT}}}},
      /*ret_def*/ {{"res", "t0:output"}});

  *library.add_function() = sort_func;

  Output in = ops::Placeholder(root, DT_FLOAT);
  NameAttrList b_name_attr;
  b_name_attr.set_name("SortFunc");
  ops::PartitionedCall call(root.WithOpName("call"), {in}, {DT_FLOAT},
                            b_name_attr);

  GraphDef graph_def;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  EXPECT_TRUE(CanTriggerXlaCompilation(graph_def));
}

TEST_F(CompilabilityCheckUtilTest, TestCanTriggerXlaCompilation) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;

  AttrValue true_attribute;
  true_attribute.set_b(true);

  FunctionDef identity_func = FunctionDefHelper::Create(
      "IdentityFunc",
      /*in_def=*/{"x:float"},
      /*out_def=*/{"res:float"},
      /*attr_def=*/{},
      /*node_def=*/{{{"t0"}, "Identity", {"x"}, {{"T", DT_FLOAT}}}},
      /*ret_def*/ {{"res", "t0:output"}});

  (*identity_func.mutable_attr())[kXlaMustCompileAttr] = true_attribute;

  FunctionDef call_identity = FunctionDefHelper::Create(
      "CallIdentity",
      /*in_def=*/{"x:float"},
      /*out_def=*/{"z:float"}, /*attr_def=*/{},
      /*node_def=*/
      {{{"func_call"},
        "PartitionedCall",
        {"x"},
        {{"Tin", DataTypeSlice({DT_FLOAT})},
         {"Tout", DataTypeSlice({DT_FLOAT})},
         {"f",
          FunctionDefHelper::FunctionRef("IdentityRef", {{"T", DT_FLOAT}})},
         {kXlaMustCompileAttr, true}}}},
      /*ret_def=*/{{"z", "func_call:output:0"}});

  *library.add_function() = identity_func;
  *library.add_function() = call_identity;

  Output in = ops::Placeholder(root, DT_FLOAT);
  NameAttrList b_name_attr;
  b_name_attr.set_name("CallIdentity");
  ops::PartitionedCall call(root.WithOpName("call"), {in}, {DT_FLOAT},
                            b_name_attr);

  GraphDef graph_def;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  EXPECT_TRUE(CanTriggerXlaCompilation(graph_def));
}

}  // namespace
}  // namespace tensorflow
