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
class MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_state_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_state_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_state_testDTcc() {
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
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::proto::IgnoringFieldPaths;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

class PruneGraphDefTest : public grappler::GrapplerTest {};

TEST_F(PruneGraphDefTest, ConstFeedWithInput) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithControlDependencies(a).WithOpName("b"),
                          0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("b");
  callable_options.add_fetch("c");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  CompareGraphs(expected, graphdef);
}

Status LessThanTenCond(const Scope& scope, const std::vector<Output>& inputs,
                       Output* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_state_testDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state_test.cc", "LessThanTenCond");

  *output = ops::Less(scope, inputs[0], 10);
  return scope.status();
}

Status AddOneBody(const Scope& scope, const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPStfrt_graph_execution_state_testDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/tfrt/utils/tfrt_graph_execution_state_test.cc", "AddOneBody");

  outputs->push_back(ops::AddN(scope, {inputs[0], 1}));
  return scope.status();
}

TEST_F(PruneGraphDefTest, InsertIdentityForLoopExitFeed) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  callable_options.add_fetch("while/Exit");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  for (const auto& node : graphdef.node()) {
    if (node.op() == "Exit") {
      EXPECT_EQ(node.name(), "while/Exit/tfrt_renamed");
    }
    if (node.name() == "while/Exit") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "while/Exit/tfrt_renamed");
    }
  }
}

TEST_F(PruneGraphDefTest, EliminateRefEntersFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs1;
    std::vector<Output> outputs2;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs1));
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop2",
                                     &outputs2));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Enter with RefEnter. Note this is not valid graph though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Enter") {
        node.set_op("RefEnter");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_enter_op = 0;
  int num_ref_enter_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefEnter") {
      num_ref_enter_op++;
    } else if (node.op() == "Enter") {
      // Identity op should be placed before Enter.
      EXPECT_EQ(num_identity_op, 1);
      num_enter_op++;
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_enter_op, 2);
  EXPECT_EQ(num_ref_enter_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefSwitchesFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond_a = ops::Placeholder(scope.WithOpName("cond_a"), DT_BOOL);
    Output cond_b = ops::Placeholder(scope.WithOpName("cond_b"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_a(scope.WithOpName("switch_a"), input, cond_a);
    ops::Switch switch_b(scope.WithOpName("switch_b"), input, cond_b);

    Output switch_a_true =
        ops::Identity(scope.WithOpName("switch_a_true"), switch_a.output_true);
    Output switch_b_true =
        ops::Identity(scope.WithOpName("switch_b_true"), switch_b.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_switch_op = 0;
  int num_ref_switch_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.name() == "switch_a_true" || node.name() == "switch_b_true") {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    } else if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefSwitch") {
      num_ref_switch_op++;
    } else if (node.op() == "Switch") {
      // Identity op should be placed before Switch.
      EXPECT_EQ(num_identity_op, 1);
      num_switch_op++;
      ASSERT_EQ(node.input().size(), 2);
      EXPECT_TRUE(node.input(0) == "input/identity" ||
                  node.input(1) == "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_switch_op, 2);
  EXPECT_EQ(num_ref_switch_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefVariablesFromV1ControlFlowFailed) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond = ops::Placeholder(scope.WithOpName("cond"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_op(scope.WithOpName("switch"), input, cond);
    Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
    Output assign =
        ops::Assign(scope.WithOpName("assign"), var, switch_op.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  const auto status = EliminateRefVariablesFromV1ControlFlow(graphdef);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("requires its input to be refs"));
}

TEST_F(PruneGraphDefTest, KeepLoopStructureComplete) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  // Sets the fetch node such that traversing from there will miss part of the
  // while loop structure.
  callable_options.add_fetch("while/LoopCond");

  GraphDef original_graphdef = graphdef;
  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));
  EXPECT_THAT(graphdef,
              IgnoringRepeatedFieldOrdering(EqualsProto(original_graphdef)));
}

class OptimizeGraphTest : public grappler::GrapplerTest {};

TEST_F(OptimizeGraphTest, OptimizeFunctions) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graphdef, *fallback_state, true));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    // After optimization, "x^3" will be transformed to "(x^2)*x".
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y_retval: float"}, {},
        {{{"ArithmeticOptimizer/ConvertPow__inner_pow3"},
          "Square",
          {"x"},
          {{"dtype", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"},
         {{"pow3"},
          "Mul",
          {"ArithmeticOptimizer/ConvertPow__inner_pow3:y:0", "x"},
          {{"T", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"}},
        {{"y_retval", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  CompareGraphs(expected, optimized_graph_def);
  CompareFunctions(expected.library().function(0),
                   optimized_graph_def.library().function(0));
}

TEST_F(OptimizeGraphTest, OptimizeFunctionsUsedByFunctionNodes) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    const Tensor kOne = test::AsScalar<float>(1.0);
    auto base2pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Add1Pow3", {"x: float"}, {"y: float"}, {},
        {{{"one"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}},
         {{"add"}, "Add", {"x", "one:output:0"}, {{"T", DT_FLOAT}}},
         {{"pcall"},
          "PartitionedCall",
          {"add:z:0"},
          {{"Tin", DataTypeSlice({DT_FLOAT})},
           {"Tout", DataTypeSlice({DT_FLOAT})},
           {"f", tensorflow::FunctionDefHelper::FunctionRef(
                     "Pow3", {{"T", DT_FLOAT}})}}}},
        {{"y", "pcall:output:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = pow3_fdef;
    *fdef_lib.add_function() = base2pow3_fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 1.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        base2pow3_fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(base2pow3_fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graphdef, *fallback_state, true));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    // After optimization, "x^3" will be transformed to "(x^2)*x".
    auto pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y_retval: float"}, {},
        {{{"ArithmeticOptimizer/ConvertPow__inner_pow3"},
          "Square",
          {"x"},
          {{"dtype", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"},
         {{"pow3"},
          "Mul",
          {"ArithmeticOptimizer/ConvertPow__inner_pow3:y:0", "x"},
          {{"T", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"}},
        {{"y_retval", "pow3:z:0"}});

    const Tensor kOne = test::AsScalar<float>(1.0);
    auto base2pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Add1Pow3", {"x: float"}, {"y: float"}, {},
        {{{"one"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}},
         {{"add"}, "Add", {"x", "one:output:0"}, {{"T", DT_FLOAT}}},
         {{"pcall"},
          "PartitionedCall",
          {"add:z:0"},
          {{"Tin", DataTypeSlice({DT_FLOAT})},
           {"Tout", DataTypeSlice({DT_FLOAT})},
           {"f", tensorflow::FunctionDefHelper::FunctionRef(
                     "Pow3", {{"T", DT_FLOAT}})}}}},
        {{"y", "pcall:output:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = pow3_fdef;
    *fdef_lib.add_function() = base2pow3_fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 1.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        base2pow3_fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(base2pow3_fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  // Since `Pow3` is called by `Add1Pow3`, it is optimized.
  CompareFunctions(expected.library().function(1),
                   optimized_graph_def.library().function(1));
  ASSERT_EQ("Pow3",
            optimized_graph_def.library().function(1).signature().name());
}

TEST_F(OptimizeGraphTest, DontOptimizeUnsafeFunction) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    Output cond = ops::Const(scope.WithOpName("cond"), true, {1, 1});
    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto if_op =
        ops::If(scope, cond, inputs, output_dtypes, func_attr, func_attr);
    Output b = if_op.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graphdef, *fallback_state, true));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  // The optimized graph remains the same as the original one, because the
  // function used by `If` op is not optimized.
  CompareGraphs(graphdef, optimized_graph_def);
  CompareFunctions(graphdef.library().function(0),
                   optimized_graph_def.library().function(0));
}

TEST_F(OptimizeGraphTest, FunctionBecomeUnsafeIfAnyOpIsUnsafe) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    Output cond = ops::Const(scope.WithOpName("cond"), true, {1, 1});
    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto if_op =
        ops::If(scope, cond, inputs, output_dtypes, func_attr, func_attr);
    Output b = if_op.output.front();

    inputs = {b};
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output c = pcall.output.front();

    Output d = ops::Identity(scope.WithOpName("d"), c);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graphdef, *fallback_state, true));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"d"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  // Both `If` and `PartitionedCall` ops use the function, so the function
  // remains unoptimized.
  CompareFunctions(graphdef.library().function(0),
                   optimized_graph_def.library().function(0));
}

class ExtendGraphTest : public grappler::GrapplerTest {};

TEST_F(ExtendGraphTest, ExtendGraph) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          tensorflow::tfrt_stub::FallbackState::Create({}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graphdef, *fallback_state, false));

  GraphDef extension;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&extension));
  }

  TF_ASSERT_OK(graph_execution_state->Extend(extension));

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  ASSERT_NE(graph_execution_state->original_graph_def(), nullptr);
  CompareGraphs(expected, *graph_execution_state->original_graph_def());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
