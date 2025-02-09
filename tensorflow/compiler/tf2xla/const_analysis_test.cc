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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysis_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysis_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysis_testDTcc() {
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

// Tests for the backward const analysis.

#include "tensorflow/compiler/tf2xla/const_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(ConstAnalysisTest, Basics) {
  Scope root = Scope::NewRootScope();

  auto arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
  auto arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
  auto arg2 = ops::_Arg(root.WithOpName("Arg2"), DT_INT32, 2);
  auto arg3 = ops::_Arg(root.WithOpName("Arg3"), DT_INT32, 3);
  auto a = ops::Shape(root, arg0);
  auto b = ops::Add(root, a, arg1);
  auto c = ops::Reshape(root, arg2, b);
  auto d = ops::Mul(root, c, ops::Sum(root, arg3, arg3));

  FixupSourceAndSinkEdges(root.graph());

  std::vector<bool> const_args(4, false);
  std::vector<bool> const_nodes(root.graph()->num_node_ids(), false);
  TF_ASSERT_OK(BackwardsConstAnalysis(*root.graph(), &const_args, &const_nodes,
                                      /*flib_runtime=*/nullptr));

  // Arg 0 doesn't need to be constant since the graph only uses its shape.
  // Arg 1 must be constant because it flows to the shape argument of a Reshape.
  // Arg 2 is used only as the value input to a Reshape and need not be const.
  // Arg 3 is used as the reduction-indices argument to Sum and must be const.
  EXPECT_EQ(const_args, std::vector<bool>({false, true, false, true}));

  EXPECT_FALSE(const_nodes[arg0.node()->id()]);
  EXPECT_TRUE(const_nodes[arg1.node()->id()]);
  EXPECT_FALSE(const_nodes[arg2.node()->id()]);
  EXPECT_TRUE(const_nodes[arg3.node()->id()]);
}

// Regression test for a case where the backward const analysis did
// not visit nodes in topological order.
TEST(ConstAnalysisTest, TopologicalOrder) {
  for (bool order : {false, true}) {
    Scope root = Scope::NewRootScope();

    auto arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(root.WithOpName("Arg2"), DT_INT32, 2);
    auto a = ops::Reshape(root, arg0, arg1);
    auto b = ops::Reshape(root, arg2, a);
    if (order) {
      // Consider both orders for arguments to the Sum so we aren't sensitive
      // to the DFS traversal order.
      std::swap(a, b);
    }
    auto c = ops::Add(root, a, b);

    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(root.ToGraph(&graph));

    std::vector<bool> const_args(3, false);
    TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args,
                                        /*compile_time_const_nodes=*/nullptr,
                                        /*flib_runtime=*/nullptr));

    EXPECT_EQ(const_args, std::vector<bool>({true, true, false}));
  }
}

void TestFunctionCall(bool is_stateful_partitioned_call) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSconst_analysis_testDTcc mht_0(mht_0_v, 265, "", "./tensorflow/compiler/tf2xla/const_analysis_test.cc", "TestFunctionCall");

  FunctionDef callee = FunctionDefHelper::Define(
      "Callee", {"t:float", "shape:int32"}, {"result:float"}, {},
      {{{"result"}, "Reshape", {"t", "shape"}, {{"T", DT_FLOAT}}}});

  FunctionDefLibrary flib;
  *flib.add_function() = callee;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  Scope root = Scope::NewRootScope().ExitOnError();

  auto arg0 = ops::_Arg(root.WithOpName("tensor"), DT_FLOAT, 0);
  auto arg1 = ops::_Arg(root.WithOpName("shape"), DT_INT32, 1);

  NameAttrList call_attrs;
  call_attrs.set_name("Callee");
  if (is_stateful_partitioned_call) {
    ops::StatefulPartitionedCall b(root.WithOpName("Call"),
                                   {Output(arg0), Output(arg1)}, {DT_FLOAT},
                                   call_attrs);
  } else {
    ops::PartitionedCall b(root.WithOpName("Call"),
                           {Output(arg0), Output(arg1)}, {DT_FLOAT},
                           call_attrs);
  }

  Graph graph(&flib_def);
  TF_ASSERT_OK(root.ToGraph(&graph));

  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, Env::Default(),
                                        /*config=*/nullptr,
                                        TF_GRAPH_DEF_VERSION, &flib_def, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  std::vector<bool> const_args(2, false);
  TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args,
                                      /*compile_time_const_nodes=*/nullptr,
                                      lib_runtime));

  EXPECT_EQ(const_args, std::vector<bool>({false, true}));
}

TEST(ConstAnalysisTest, PartitionedCall) {
  TestFunctionCall(/*is_stateful_partitioned_call=*/false);
}

TEST(ConstAnalysisTest, StatefulPartitionedCall) {
  TestFunctionCall(/*is_stateful_partitioned_call=*/true);
}

TEST(ConstAnalysisTest, DontFollowControlDependencies) {
  Scope root = Scope::NewRootScope();

  Output arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
  Output arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
  Output c1 =
      ops::Const(root.WithOpName("c1").WithControlDependencies(arg0), 1, {1});
  Output add = ops::Add(root, arg1, c1);
  Output reshape = ops::Reshape(root, arg1, add);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));

  std::vector<bool> const_args(2, false);
  TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args,
                                      /*compile_time_const_nodes=*/nullptr,
                                      /*flib_runtime=*/nullptr));

  EXPECT_EQ(const_args, std::vector<bool>({false, true}));
}

TEST(ConstAnalysisTest, RespectExplicitAttr_0) {
  Scope root = Scope::NewRootScope();

  Output arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
  Output arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
  Output c1 =
      ops::Const(root.WithOpName("c1").WithControlDependencies(arg0), 1, {1});
  Output add = ops::Add(root, arg1, c1);

  // Force const analysis to pretend that the shape argument to `reshape` does
  // not need to be a constant.
  Output reshape = ops::Reshape(root, arg1, add);
  reshape.node()->AddAttr(kXlaCompileTimeConstantInputsAttr,
                          std::vector<string>());

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));

  std::vector<bool> const_args(2, false);
  TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args,
                                      /*compile_time_const_nodes=*/nullptr,
                                      /*flib_runtime=*/nullptr));

  EXPECT_EQ(const_args, std::vector<bool>({false, false}));
}

TEST(ConstAnalysisTest, RespectExplicitAttr_1) {
  Scope root = Scope::NewRootScope();

  Output arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
  Output c1 =
      ops::Const(root.WithOpName("c1").WithControlDependencies(arg0), 1, {1});
  Output add = ops::Add(root, arg0, c1);

  // Force const analysis to pretend that the first argument to `add` needs to
  // be a constant.
  std::vector<string> add_constant_inputs;
  add_constant_inputs.push_back("x");
  add.node()->AddAttr(kXlaCompileTimeConstantInputsAttr, add_constant_inputs);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));

  std::vector<bool> const_args(1, false);
  TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args,
                                      /*compile_time_const_nodes=*/nullptr,
                                      /*flib_runtime=*/nullptr));

  EXPECT_EQ(const_args, std::vector<bool>({true}));
}

static bool Initialized = [] {
  tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

}  // namespace
}  // namespace tensorflow
