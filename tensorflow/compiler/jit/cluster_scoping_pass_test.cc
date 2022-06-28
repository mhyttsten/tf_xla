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
class MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_pass_testDTcc() {
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

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

Status ClusterScoping(std::unique_ptr<Graph>* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_pass_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/jit/cluster_scoping_pass_test.cc", "ClusterScoping");

  FixupSourceAndSinkEdges(graph->get());

  GraphOptimizationPassWrapper wrapper;
  wrapper.session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  GraphOptimizationPassOptions opt_options =
      wrapper.CreateGraphOptimizationPassOptions(graph);

  ClusterScopingPass pass;
  return pass.Run(opt_options);
}

absl::flat_hash_map<string, string> GetXlaInternalScopes(const Graph& graph) {
  absl::flat_hash_map<string, string> scopes;
  for (Node* node : graph.nodes()) {
    string scope;
    if (GetNodeAttr(node->attrs(), kXlaInternalScopeAttr, &scope).ok()) {
      scopes[node->name()] = scope;
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "_XlaInternalScopes:";
    for (const auto& p : scopes) {
      VLOG(2) << " " << p.first << " -> " << p.second;
    }
  }
  return scopes;
}

Node* BuildStageNode(GraphDefBuilder& builder, string name,
                     std::initializer_list<DataType> dtypes,
                     absl::Span<const ops::NodeOut> values) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScluster_scoping_pass_testDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/jit/cluster_scoping_pass_test.cc", "BuildStageNode");

  auto opts = builder.opts()
                  .WithName(std::move(name))
                  .WithAttr("dtypes", std::move(dtypes));
  if (opts.HaveError()) {
    return nullptr;
  }

  NodeBuilder node_builder(name, "Stage", opts.op_registry());
  node_builder.Input(values);
  return opts.FinalizeBuilder(&node_builder);
}

TEST(XlaCompilationTest, StagePipelinePreserved) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    // Graph:
    //       b
    //       |
    //       v
    // a -> add0 (ClusterX) -> relu0 (ClusterX) -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 (ClusterY) -> relu1 (ClusterY)
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("a")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::SourceOp("Const", builder.opts()
                                         .WithName("b")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* unstage = ops::SourceOp(
        "Unstage",
        builder.opts().WithName("unstage").WithAttr("dtypes", {DT_FLOAT}));

    Node* add0 = ops::BinaryOp("Add", a, b, builder.opts().WithName("add0"));
    Node* add1 =
        ops::BinaryOp("Add", unstage, b, builder.opts().WithName("add1"));
    Node* relu0 = ops::UnaryOp("Relu", add0, builder.opts().WithName("relu0"));
    ops::UnaryOp("Relu", add1, builder.opts().WithName("relu1"));
    BuildStageNode(builder, "stage", {DT_FLOAT}, {relu0});

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ClusterScoping(&graph));

  auto scopes = GetXlaInternalScopes(*graph);
  EXPECT_NE(scopes["add0"], scopes["add1"]);
  EXPECT_EQ(scopes["add0"], scopes["relu0"]);
  EXPECT_EQ(scopes["add1"], scopes["relu1"]);
}

TEST(XlaCompilationTest, StagePipelinePreservedAndInitialScopesRespected) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    // Graph:
    //       b
    //       |
    //       v
    // a -> add0 (ClusterA) -> relu0 (ClusterB) -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 (ClusterC) -> relu1 (ClusterD)
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("a")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::SourceOp("Const", builder.opts()
                                         .WithName("b")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* unstage = ops::SourceOp(
        "Unstage",
        builder.opts().WithName("unstage").WithAttr("dtypes", {DT_FLOAT}));

    // Intentionally give add0 and add1 the same initial scope but they should
    // be separated by the ClusterScopingPass.
    Node* add0 = ops::BinaryOp("Add", a, b,
                               builder.opts().WithName("add0").WithAttr(
                                   kXlaInternalScopeAttr, "ClusterA"));
    Node* add1 = ops::BinaryOp("Add", unstage, b,
                               builder.opts().WithName("add1").WithAttr(
                                   kXlaInternalScopeAttr, "ClusterA"));
    Node* relu0 = ops::UnaryOp("Relu", add0,
                               builder.opts().WithName("relu0").WithAttr(
                                   kXlaInternalScopeAttr, "ClusterB"));
    ops::UnaryOp("Relu", add1,
                 builder.opts().WithName("relu1").WithAttr(
                     kXlaInternalScopeAttr, "ClusterD"));
    BuildStageNode(builder, "stage", {DT_FLOAT}, {relu0});

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ClusterScoping(&graph));

  auto scopes = GetXlaInternalScopes(*graph);
  EXPECT_NE(scopes["add0"], scopes["add1"]);
  EXPECT_NE(scopes["add0"], scopes["relu0"]);
  EXPECT_NE(scopes["add1"], scopes["relu1"]);
}

}  // namespace
}  // namespace tensorflow
