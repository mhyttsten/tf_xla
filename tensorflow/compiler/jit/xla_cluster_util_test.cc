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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc() {
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

#include "tensorflow/compiler/jit/xla_cluster_util.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(CreateCycleDetectionGraph, ConnectivityThroughEnterExitRegion) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter =
      ops::internal::Enter(root.WithOpName("enter"), a, "only_frame");
  Output exit = ops::internal::Exit(root.WithOpName("exit"), enter);
  Output b = ops::Add(root.WithOpName("b"), a, exit);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.CanContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ConnectivityThroughMultipleEnterExitRegions) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);
  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), a, "frame_1");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);
  Output b = ops::Add(root.WithOpName("b"), a, exit_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.CanContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ReachingEnterExit) {
  // TODO(b/127521408): We can lift this limitation with some work.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);

  Output add = ops::Add(root.WithOpName("add"), exit_0, exit_0);

  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), add, "frame_0");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK_AND_ASSIGN(bool ok,
                          CreateCycleDetectionGraph(root.graph(), &cycles));
  EXPECT_FALSE(ok);
}

const char* kCPU0 = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:localhost/replica:0/task:0/device:GPU:0";
const char* kGPU1 = "/job:localhost/replica:0/task:0/device:GPU:1";

TEST(IsSingleGpuGraph, ReturnsTrue) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_TRUE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForCpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kCPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForMultiGpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b").WithAssignedDevice(kGPU1), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}

StatusOr<std::vector<string>> GetNodesRelatedToRefVarsSorted(
    const Scope& scope, FunctionLibraryDefinition* flib_def = nullptr) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def_local(OpRegistry::Global(), flib);
  if (flib_def == nullptr) {
    flib_def = &flib_def_local;
  }

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));

  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          nullptr, Env::Default(), /*config=*/nullptr, TF_GRAPH_DEF_VERSION,
          flib_def, OptimizerOptions{}));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> nodes_related_to_ref_vars,
                      GetNodesRelatedToRefVariables(*graph, lib_runtime));

  std::vector<string> names;
  absl::c_transform(nodes_related_to_ref_vars, std::back_inserter(names),
                    [](Node* n) { return n->name(); });
  absl::c_sort(names);
  return names;
}

void CreateSubgraphTouchingRefVar(const Scope& s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_0(mht_0_v, 336, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "CreateSubgraphTouchingRefVar");

  Output variable =
      ops::Variable(s.WithOpName("variable"), PartialTensorShape{}, DT_FLOAT);
  Output read = ops::Identity(s.WithOpName("read_ref_var"), variable);
  Output neg = ops::Negate(s.WithOpName("negate_ref"), read);
  Output add = ops::Add(s.WithOpName("add_ref"), neg, neg);

  Output constant =
      ops::Const(s.WithOpName("constant_ref"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(constant.node(), variable.node());
}

void CreateSubgraphNotTouchingRefVar(const Scope& s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_1(mht_1_v, 351, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "CreateSubgraphNotTouchingRefVar");

  Output constant =
      ops::Const(s.WithOpName("constant_normal"), Input::Initializer(0.0));
  Output neg = ops::Negate(s.WithOpName("negate_normal"), constant);
  Output add = ops::Add(s.WithOpName("add_normal"), neg, neg);
}

void CreateSubgraphCallingFunctionWithRefVar(const Scope& s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_2(mht_2_v, 361, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "CreateSubgraphCallingFunctionWithRefVar");

  NameAttrList ref_float_function;
  ref_float_function.set_name("RefFloatFn");
  ops::PartitionedCall call(s.WithOpName("RefFloat"), {absl::Span<Input>{}},
                            {DT_FLOAT}, ref_float_function);
  Output constant =
      ops::Const(s.WithOpName("constant_ref_pco"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(call.operation.node(), constant.node());
}

void CreateSubgraphCallingFunctionWithoutRefVar(const Scope& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_3(mht_3_v, 374, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "CreateSubgraphCallingFunctionWithoutRefVar");

  NameAttrList regular_float_function;
  regular_float_function.set_name("RegularFloatFn");
  ops::PartitionedCall call(s.WithOpName("RegularFloat"), {absl::Span<Input>{}},
                            {DT_FLOAT}, regular_float_function);
  Output constant =
      ops::Const(s.WithOpName("constant_normal_pco"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(call.operation.node(), constant.node());
}

void AddRefFunctionFunctionDef(FunctionDefLibrary* fdef_lib) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_4(mht_4_v, 387, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "AddRefFunctionFunctionDef");

  FunctionDef make_ref_float = FunctionDefHelper::Define(
      "RefFloatFn", {}, {"r:float"}, {},
      {{{"var"},
        "VariableV2",
        {},
        {{"dtype", DT_FLOAT}, {"shape", TensorShape({})}}},
       {{"r"}, "Identity", {"var"}, {{"T", DT_FLOAT}}}});
  *fdef_lib->add_function() = make_ref_float;
}

void AddRegularFunctionFunctionDef(FunctionDefLibrary* fdef_lib) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_5(mht_5_v, 401, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "AddRegularFunctionFunctionDef");

  Tensor seven(DT_FLOAT, {});
  seven.scalar<float>()() = 7;
  FunctionDef make_regular_float = FunctionDefHelper::Define(
      "RegularFloatFn", {}, {"r:float"}, {},
      {{{"r"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", seven}}}});
  *fdef_lib->add_function() = make_regular_float;
}

TEST(NodesRelatedToRefVariables, Basic) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary fdef_lib;

  CreateSubgraphTouchingRefVar(root);
  CreateSubgraphNotTouchingRefVar(root);

  AddRefFunctionFunctionDef(&fdef_lib);
  CreateSubgraphCallingFunctionWithRefVar(root);

  AddRegularFunctionFunctionDef(&fdef_lib);
  CreateSubgraphCallingFunctionWithoutRefVar(root);

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<string> names,
                          GetNodesRelatedToRefVarsSorted(root, &flib_def));

  std::vector<string> expected({
      "RefFloat",
      "add_ref",
      "constant_ref",
      "constant_ref_pco",
      "negate_ref",
      "read_ref_var",
      "variable",
  });

  EXPECT_EQ(names, expected);
}

Status MakeLoop(Scope s, Output init_value, absl::string_view loop_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cluster_util_testDTcc mht_6(mht_6_v, 446, "", "./tensorflow/compiler/jit/xla_cluster_util_test.cc", "MakeLoop");

  s = s.NewSubScope(std::string(loop_name));
  ops::internal::Enter enter(s.WithOpName("init_value"), init_value, loop_name);
  ops::Merge merge(s.WithOpName("merge"), {init_value, init_value});
  Output next_iteration =
      ops::NextIteration(s.WithOpName("next_itr"), merge.output);
  return s.graph()->UpdateEdge(next_iteration.node(), 0, merge.output.node(),
                               1);
}

TEST(NodesRelatedToRefVariables, Cycles) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output variable = ops::Variable(root.WithOpName("variable"),
                                  PartialTensorShape{}, DT_FLOAT);
  TF_ASSERT_OK(
      MakeLoop(root, ops::Identity(root.WithOpName("read_ref_var"), variable),
               "ref_loop"));
  TF_ASSERT_OK(MakeLoop(
      root, ops::Const(root.WithOpName("constant"), Input::Initializer(0.0)),
      "normal_loop"));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<string> names,
                          GetNodesRelatedToRefVarsSorted(root));
  std::vector<string> expected({"read_ref_var", "ref_loop/init_value",
                                "ref_loop/merge", "ref_loop/next_itr",
                                "variable"});

  EXPECT_EQ(names, expected);
}
}  // namespace
}  // namespace tensorflow
