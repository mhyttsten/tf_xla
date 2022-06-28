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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inference_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inference_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inference_testDTcc() {
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

#include "tensorflow/core/common_runtime/forward_type_inference.h"

#include <functional>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status Rewrite(std::unique_ptr<Graph>* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inference_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/common_runtime/forward_type_inference_test.cc", "Rewrite");

  FunctionLibraryDefinition flib_def((*graph)->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options;
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  ForwardTypeInferencePass pass;
  return pass.Run(opt_options);
}

TEST(ForwardTypeInferenceTest, BasicStraightline) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  auto start = ops::Placeholder(root.WithOpName("start"), DT_INT64);
  auto stop = ops::Placeholder(root.WithOpName("stop"), DT_INT64);
  auto step = ops::Placeholder(root.WithOpName("step"), DT_INT64);

  Node* ds;
  TensorShapeProto shape;
  shape.mutable_dim();
  shape.set_unknown_rank(false);
  TF_ASSERT_OK(NodeBuilder("ds", "RangeDataset", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(start.node())})
                   .Input({NodeBuilder::NodeOut(stop.node())})
                   .Input({NodeBuilder::NodeOut(step.node())})
                   .Attr("output_types", {DT_INT32})
                   .Attr("output_shapes", {shape})
                   .Finalize(root.graph(), &ds));

  Node* id;
  TF_ASSERT_OK(NodeBuilder("id", "Identity", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(ds)})
                   .Attr("T", DT_VARIANT)
                   .Finalize(root.graph(), &id));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  for (const auto& node : graph->nodes()) {
    if ((node->name() == "ds") || ((node->name() == "id"))) {
      const auto& t = node->def().experimental_type();
      EXPECT_EQ(t.type_id(), TFT_PRODUCT) << node->def().DebugString();
      EXPECT_EQ(t.args(0).type_id(), TFT_DATASET) << node->def().DebugString();
    }
  }
}

TEST(ForwardTypeInferenceTest, CyclicGraphWithV1ControlFlow) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  auto start = ops::Placeholder(root.WithOpName("start"), DT_INT64);
  auto stop = ops::Placeholder(root.WithOpName("stop"), DT_INT64);
  auto step = ops::Placeholder(root.WithOpName("step"), DT_INT64);
  auto cond = ops::Placeholder(root.WithOpName("cond"), DT_BOOL);

  Node* ds;
  TensorShapeProto shape;
  shape.mutable_dim();
  shape.set_unknown_rank(false);
  TF_ASSERT_OK(NodeBuilder("ds", "RangeDataset", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(start.node())})
                   .Input({NodeBuilder::NodeOut(stop.node())})
                   .Input({NodeBuilder::NodeOut(step.node())})
                   .Attr("output_types", {DT_INT32})
                   .Attr("output_shapes", {shape})
                   .Finalize(root.graph(), &ds));

  Node* enter;
  TF_ASSERT_OK(NodeBuilder("enter", "Enter", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(ds)})
                   .Attr("frame_name", "loop")
                   .Finalize(root.graph(), &enter));

  Node* loop_cond;
  TF_ASSERT_OK(NodeBuilder("loop_cond", "Enter", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(cond.node())})
                   .Attr("frame_name", "loop")
                   .Finalize(root.graph(), &loop_cond));

  Node* merge;
  TF_ASSERT_OK(
      NodeBuilder("merge", "Merge", &root.graph()->flib_def())
          .Input({NodeBuilder::NodeOut(enter), NodeBuilder::NodeOut(enter)})
          .Finalize(root.graph(), &merge));

  Node* sw;
  TF_ASSERT_OK(NodeBuilder("sw", "Switch", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(merge)})
                   .Input({NodeBuilder::NodeOut(loop_cond)})
                   .Finalize(root.graph(), &sw));

  Node* id;
  TF_ASSERT_OK(NodeBuilder("id", "Identity", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(sw)})
                   .Finalize(root.graph(), &id));

  Node* next;
  TF_ASSERT_OK(NodeBuilder("next", "NextIteration", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(id)})
                   .Finalize(root.graph(), &next));

  TF_ASSERT_OK(root.graph()->UpdateEdge(next, 0, merge, 1));

  Node* exit;
  TF_ASSERT_OK(NodeBuilder("exit", "Exit", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(sw)})
                   .Finalize(root.graph(), &exit));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  for (const auto& node : graph->nodes()) {
    if ((node->name() == "ds") || (node->name() == "id") ||
        (node->name() == "enter") || (node->name() == "exit") ||
        (node->name() == "sw") || (node->name() == "merge") ||
        (node->name() == "next")) {
      const auto& t = node->def().experimental_type();
      ASSERT_EQ(t.type_id(), TFT_PRODUCT) << node->def().DebugString();
      EXPECT_EQ(t.args(0).type_id(), TFT_DATASET) << node->def().DebugString();
    }
  }
}

REGISTER_OP("TestSourceOp").Output("o: variant");

REGISTER_OP("TestTensorUnaryOp")
    .Input("i: variant")
    .Output("o: variant")
    .SetForwardTypeFn([](const TypeRefVector& input_types,
                         const TypeRefMap& type_vars) {
      FullTypeDef t;
      t.set_type_id(TFT_PRODUCT);
      t.add_args()->set_type_id(TFT_TENSOR);
      return t;
    });

REGISTER_OP("TestArrayUnaryOp")
    .Input("i: variant")
    .Output("o: variant")
    .SetForwardTypeFn([](const TypeRefVector& input_types,
                         const TypeRefMap& type_vars) {
      FullTypeDef t;
      t.set_type_id(TFT_PRODUCT);
      t.add_args()->set_type_id(TFT_ARRAY);
      return t;
    });

REGISTER_OP("TestMergeOp")
    .Input("i1: variant")
    .Input("i2: variant")
    .Output("o: variant")
    .SetForwardTypeFn([](const TypeRefVector& input_types,
                         const TypeRefMap& type_vars) {
      EXPECT_EQ(input_types.size(), 2);
      FullTypeDef t;
      t.set_type_id(TFT_PRODUCT);
      if ((input_types[0].get().type_id() == TFT_TENSOR) &&
          (input_types[1].get().type_id() == TFT_ARRAY)) {
        t.add_args()->set_type_id(TFT_ARRAY);
      } else {
        t.add_args()->set_type_id(TFT_TENSOR);
      }
      return t;
    });

TEST(ForwardTypeInferenceTest, TernaryNodeWithIgnoredInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  Node* s;
  TF_ASSERT_OK(NodeBuilder("s", "TestSourceOp", &root.graph()->flib_def())
                   .Finalize(root.graph(), &s));

  Node* tn;
  TF_ASSERT_OK(NodeBuilder("tn", "TestTensorUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &tn));

  Node* id;
  TF_ASSERT_OK(NodeBuilder("id", "Identity", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &id));

  Node* an;
  TF_ASSERT_OK(NodeBuilder("an", "TestArrayUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(id)})
                   .Finalize(root.graph(), &an));

  // This node has an unbalanced path from s, and its type inference can produce
  // different results if the ancestors have incomplete type information.
  // This test is designed that the more complete type inference still takes
  // place, even though the node would be first visited with incomplete type
  // info under a naive BFS walk.
  Node* m;
  TF_ASSERT_OK(NodeBuilder("m", "TestMergeOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(tn)})
                   .Input({NodeBuilder::NodeOut(an)})
                   .Finalize(root.graph(), &m));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  for (const auto& node : graph->nodes()) {
    if (node->name() == "m") {
      const auto& t = node->def().experimental_type();
      ASSERT_EQ(t.type_id(), TFT_PRODUCT) << node->def().DebugString();
      // We want complete type info (ARRAY), not incomplete one (TENSOR).
      EXPECT_EQ(t.args(0).type_id(), TFT_ARRAY) << node->def().DebugString();
    }
  }
}

TEST(ForwardTypeInferenceTest, BinaryNodeWithUnorderedInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  Node* s;
  TF_ASSERT_OK(NodeBuilder("s", "TestSourceOp", &root.graph()->flib_def())
                   .Finalize(root.graph(), &s));

  Node* tn;
  TF_ASSERT_OK(NodeBuilder("tn", "TestTensorUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &tn));

  Node* an;
  TF_ASSERT_OK(NodeBuilder("an", "TestArrayUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &an));

  Node* m;
  // These edges are temporary, and will be updated below.
  TF_ASSERT_OK(NodeBuilder("m", "TestMergeOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &m));

  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // Rewire the inputs of "m", in a way that causes the second edge to appear
  // first when iterating over the node's in_edges.
  // Warning: this is highly implementation-specific. Changes to infra code
  // may break this logic.
  Node* m_copy = nullptr;
  Node* tn_copy = nullptr;
  Node* an_copy = nullptr;
  for (const auto& node : graph->nodes()) {
    if (node->name() == "m") {
      m_copy = node;
    } else if (node->name() == "tn") {
      tn_copy = node;
    } else if (node->name() == "an") {
      an_copy = node;
    }
  }
  TF_ASSERT_OK(graph->UpdateEdge(an_copy, 0, m_copy, 1));
  TF_ASSERT_OK(graph->UpdateEdge(tn_copy, 0, m_copy, 0));

  TF_ASSERT_OK(Rewrite(&graph));

  for (const auto& node : graph->nodes()) {
    if (node->name() == "m") {
      const auto& t = node->def().experimental_type();
      ASSERT_EQ(t.type_id(), TFT_PRODUCT) << node->def().DebugString();
      // We want complete type info (ARRAY), not incomplete one (TENSOR).
      EXPECT_EQ(t.args(0).type_id(), TFT_ARRAY) << node->def().DebugString();
    }
  }
}

TEST(ForwardTypeInferenceTest, BinaryNodeWithCycleInput) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  auto cond = ops::Placeholder(root.WithOpName("cond"), DT_BOOL);

  Node* s;
  TF_ASSERT_OK(NodeBuilder("s", "TestSourceOp", &root.graph()->flib_def())
                   .Finalize(root.graph(), &s));

  Node* an;
  TF_ASSERT_OK(NodeBuilder("an", "TestArrayUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &an));

  Node* enter;
  TF_ASSERT_OK(NodeBuilder("enter", "Enter", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(an)})
                   .Attr("frame_name", "loop")
                   .Finalize(root.graph(), &enter));

  Node* loop_cond;
  TF_ASSERT_OK(NodeBuilder("loop_cond", "Enter", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(cond.node())})
                   .Attr("frame_name", "loop")
                   .Finalize(root.graph(), &loop_cond));

  // TODO(mdan): Is there any way to create a cycle without using a Merge node?
  Node* merge;
  TF_ASSERT_OK(
      NodeBuilder("merge", "Merge", &root.graph()->flib_def())
          .Input({NodeBuilder::NodeOut(enter), NodeBuilder::NodeOut(enter)})
          .Finalize(root.graph(), &merge));

  Node* sw;
  TF_ASSERT_OK(NodeBuilder("sw", "Switch", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(merge)})
                   .Input({NodeBuilder::NodeOut(loop_cond)})
                   .Finalize(root.graph(), &sw));

  Node* tn;
  TF_ASSERT_OK(NodeBuilder("tn", "TestTensorUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(sw)})
                   .Finalize(root.graph(), &tn));

  Node* next;
  TF_ASSERT_OK(NodeBuilder("next", "NextIteration", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(tn)})
                   .Finalize(root.graph(), &next));

  TF_ASSERT_OK(root.graph()->UpdateEdge(next, 0, merge, 1));

  Node* exit;
  TF_ASSERT_OK(NodeBuilder("exit", "Exit", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(sw)})
                   .Finalize(root.graph(), &exit));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  const auto& status = Rewrite(&graph);
  ASSERT_FALSE(status.ok());

  // We always expect the merge node to raise a type inference error, because
  // the type in the loop (TENSOR, from TestTensorUnaryOp) doesn't match the
  // type upon loop entry (ARRAY, from TestArrayUnaryOp).
  // This error is only raised when both types are resolved, otherwise Merge
  // will deduct the type from its partial inputs.
  // In effect, the assertion verifies that the merge node is always visited
  // at least once after both its inputs have been resolved, so the graph always
  // has complete type information.
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("expected compatible input types"));
}

TEST(WeakForwardTypeInferenceTest, AlwaysSucceeds) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope root = Scope::NewRootScope().ExitOnError();

  auto cond = ops::Placeholder(root.WithOpName("cond"), DT_BOOL);

  Node* s;
  TF_ASSERT_OK(NodeBuilder("s", "TestSourceOp", &root.graph()->flib_def())
                   .Finalize(root.graph(), &s));

  Node* an;
  TF_ASSERT_OK(NodeBuilder("an", "TestArrayUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &an));

  Node* tn;
  TF_ASSERT_OK(NodeBuilder("tn", "TestTensorUnaryOp", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(s)})
                   .Finalize(root.graph(), &tn));

  Node* merge;
  TF_ASSERT_OK(NodeBuilder("merge", "Merge", &root.graph()->flib_def())
                   .Input({NodeBuilder::NodeOut(an), NodeBuilder::NodeOut(tn)})
                   .Finalize(root.graph(), &merge));

  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FunctionLibraryDefinition flib_def(graph->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options;
  opt_options.session_options = &session_options;
  opt_options.graph = &graph;
  opt_options.flib_def = &flib_def;
  WeakForwardTypeInferencePass pass;

  TF_ASSERT_OK(pass.Run(opt_options));
}

}  // namespace
}  // namespace tensorflow
