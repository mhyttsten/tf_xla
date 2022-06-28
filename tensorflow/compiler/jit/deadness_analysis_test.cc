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
class MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc() {
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

#include "tensorflow/compiler/jit/deadness_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/deadness_analysis_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

se::port::StatusOr<bool> HasInputsWithMismatchingDeadness(
    const DeadnessAnalysis& deadness_analysis, const Node& n) {
  absl::optional<DeadnessAnalysis::DeadnessPredicate> pred;
  for (const Edge* edge : n.in_edges()) {
    TF_ASSIGN_OR_RETURN(
        DeadnessAnalysis::DeadnessPredicate this_pred,
        deadness_analysis.GetPredicateFor(edge->src(), edge->src_output()));
    if (pred && *pred != this_pred) {
      return true;
    }
    pred = this_pred;
  }

  return false;
}

using deadness_analysis_internal::ComputePredicates;
using deadness_analysis_internal::PredicateMapTy;

Status AnalyzeDeadness(Graph* graph,
                       std::unique_ptr<DeadnessAnalysis>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "AnalyzeDeadness");

  FixupSourceAndSinkEdges(graph);
  return DeadnessAnalysis::Run(*graph, result);
}

ops::Switch CreateSwitch(const Scope& root, const string& prefix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateSwitch");

  Output value = ops::Placeholder(root.WithOpName(prefix + "/value"), DT_FLOAT);
  Output predicate =
      ops::Placeholder(root.WithOpName(prefix + "/pred"), DT_BOOL);
  return ops::Switch(root.WithOpName(prefix + "/switch"), value, predicate);
}

TensorId ControlOutputFor(const Output& o) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "ControlOutputFor");

  return {o.node()->name(), Graph::kControlSlot};
}

void VLogGraphIfAsked(const Graph& graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "VLogGraphIfAsked");

  if (VLOG_IS_ON(3)) {
    GraphDef graph_def;
    graph.ToGraphDef(&graph_def);
    string serialized;
    ::tensorflow::protobuf::TextFormat::PrintToString(graph_def, &serialized);
    LOG(INFO) << serialized;
  }
}

struct InductionVarInfo {
  Output induction_var;
  Output loop_cond;
};

// Creates an induction variable with the following structure (simplified for
// brevity):
//
//            +---------------+
//            | initial_value |
//            +---------------+
//              |
//              |
//              v
//            +---------------+
//            |     Enter     |
//            +---------------+
//              |
//              |
//              v
//            +---------------+
//         +> |     Merge     | -+
//         |  +---------------+  |
//         |    |                |
//         |    |                |
//         |    v                |
//         |  +---------------+  |
//         |  |  LessThan10   |  |
//         |  +---------------+  |
//         |    |                |
//         |    |                |
//         |    v                |
//         |  +---------------+  |
//    +----+- |    Switch     | <+
//    |    |  +---------------+
//    |    |    |
//    |    |    |
//    |    |    v
//    |    |  +---------------+
//    |    +- |    AddOne     |
//    |       +---------------+
//    |       +---------------+
//    +-----> |     Exit      |
//            +---------------+
InductionVarInfo CreateInductionVariable(const Scope& root,
                                         const string& prefix,
                                         const string& frame_name,
                                         const Output& initial_value) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("prefix: \"" + prefix + "\"");
   mht_4_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_4(mht_4_v, 316, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateInductionVariable");

  Output enter_initial_value = ops::internal::Enter(
      root.WithOpName(prefix + "/enter"), initial_value, frame_name);

  ops::Merge iv(root.WithOpName(prefix + "/iv"),
                {enter_initial_value, enter_initial_value});
  Output increment_by = ops::Const(root.WithOpName(prefix + "/incr"), 1);
  Output final_value = ops::Const(root.WithOpName(prefix + "/final"), 10);
  Output loop_cond_expr =
      ops::Less(root.WithOpName(prefix + "/cond"), iv.output, final_value);
  ops::Switch latch(root.WithOpName(prefix + "/latch"), iv.output,
                    loop_cond_expr);
  ops::internal::Exit exit(root.WithOpName(prefix + "/exit"),
                           latch.output_false);
  Output iv_next = ops::Add(root.WithOpName(prefix + "/ivnext"),
                            latch.output_true, increment_by);
  Output next_iteration =
      ops::NextIteration(root.WithOpName(prefix + "/next_iteration"), iv_next);

  CHECK(root.graph()
            ->UpdateEdge(next_iteration.node(), 0, iv.output.node(), 1)
            .ok());
  root.graph()->AddControlEdge(iv.output.node(), increment_by.node());
  root.graph()->AddControlEdge(iv.output.node(), final_value.node());

  return {iv.output, loop_cond_expr};
}

InductionVarInfo CreateInductionVariable(const Scope& root,
                                         const string& prefix,
                                         const string& frame_name,
                                         int32_t init) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("prefix: \"" + prefix + "\"");
   mht_5_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_5(mht_5_v, 352, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateInductionVariable");

  return CreateInductionVariable(
      root, prefix, frame_name,
      ops::Const(root.WithOpName(prefix + "/init"), init));
}

// Creates an induction variable with the following structure:
//
//                           +---------------+
//                           | initial_value |
//                           +---------------+
//                             |
//                             |
//                             v
//                           +---------------+
//                           |     Enter     |
//                           +---------------+
//                             |
//                             |
//                             v
//                           +---------------+
//                           |     Merge     | <+
//                           +---------------+  |
//                             |                |
//                             |                |
//                             v                |
//         +-----------+     +---------------+  |
//         | loop_cond | --> |    Switch     | -+
//         +-----------+     +---------------+
//                             |
//                             |
//                             v
//                           +---------------+
//                           |     Exit      |
//                           +---------------+
struct DependentInductionVar {
  Output induction_var;
  ops::Switch latch;
};

DependentInductionVar CreateDependentLoopInvariantValue(
    const Scope& root, const string& prefix, const string& frame_name,
    const Output& loop_cond, const Output& value) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("prefix: \"" + prefix + "\"");
   mht_6_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_6(mht_6_v, 399, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateDependentLoopInvariantValue");

  Output enter_value = ops::internal::Enter(root.WithOpName(prefix + "/enter"),
                                            value, frame_name);
  ops::Merge iv(root.WithOpName(prefix + "/iv"), {enter_value, enter_value});
  ops::Switch latch(root.WithOpName(prefix + "/latch"), iv.output, loop_cond);
  ops::internal::Exit exit(root.WithOpName(prefix + "/exit"),
                           latch.output_false);
  Output next_iteration = ops::NextIteration(
      root.WithOpName(prefix + "/next_iteration"), latch.output_true);
  CHECK(root.graph()
            ->UpdateEdge(next_iteration.node(), 0, iv.output.node(), 1)
            .ok());
  return {iv.output, latch};
}

DependentInductionVar CreateDependentLoopInvariantValue(
    const Scope& root, const string& prefix, const string& frame_name,
    const Output& loop_cond, int32_t value) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("prefix: \"" + prefix + "\"");
   mht_7_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_7(mht_7_v, 421, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateDependentLoopInvariantValue");

  return CreateDependentLoopInvariantValue(
      root, prefix, frame_name, loop_cond,
      ops::Const(root.WithOpName(prefix + "/init"), value));
}

TEST(DeadnessAnalysisTest, BasicPositive) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw = CreateSwitch(root, "0");
  Output add =
      ops::Add(root.WithOpName("add"), sw.output_true, sw.output_false);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, BasicNegative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("b"), DT_FLOAT);
  Output add = ops::Add(root.WithOpName("add"), a, b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, AndIsCommutative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");

  Output a0 =
      ops::Add(root.WithOpName("a0"), sw_0.output_false, sw_1.output_false);
  Output a1 =
      ops::Add(root.WithOpName("a1"), sw_1.output_false, sw_0.output_false);

  Output b0 =
      ops::Add(root.WithOpName("b0"), sw_0.output_false, sw_1.output_true);
  Output b1 =
      ops::Add(root.WithOpName("b1"), sw_1.output_true, sw_0.output_false);

  Output live0 = ops::Add(root.WithOpName("live0"), a0, a1);
  Output live1 = ops::Add(root.WithOpName("live1"), b0, b1);

  Output halfdead0 = ops::Add(root.WithOpName("halfdead0"), a0, b0);
  Output halfdead1 = ops::Add(root.WithOpName("halfdead1"), a1, b1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  bool has_inputs_with_mismatching_deadness;

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *live0.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *live1.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *halfdead0.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *halfdead1.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, AndIsAssociative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  Output a0 =
      ops::Add(root.WithOpName("a0"), sw_0.output_false, sw_1.output_false);
  Output a1 = ops::Add(root.WithOpName("a1"), a0, sw_2.output_false);

  Output b0 =
      ops::Add(root.WithOpName("b0"), sw_1.output_false, sw_2.output_false);
  Output b1 = ops::Add(root.WithOpName("b1"), sw_0.output_false, b0);

  Output add = ops::Add(root.WithOpName("add"), a1, b1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, OrIsCommutative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {sw_1.output_false, sw_0.output_false});
  ops::Merge m2(root.WithOpName("m2"), {sw_0.output_false, sw_1.output_true});
  ops::Merge m3(root.WithOpName("m3"), {sw_1.output_true, sw_0.output_false});

  Output live0 = ops::Add(root.WithOpName("live0"), m0.output, m1.output);
  Output live1 = ops::Add(root.WithOpName("live1"), m2.output, m3.output);

  Output halfdead0 =
      ops::Add(root.WithOpName("halfdead0"), m0.output, m2.output);
  Output halfdead1 =
      ops::Add(root.WithOpName("halfdead1"), m1.output, m3.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  bool has_inputs_with_mismatching_deadness;

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *live0.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *live1.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *halfdead0.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);

  TF_ASSERT_OK_AND_ASSIGN(
      has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *halfdead1.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, OrIsAssociative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {m0.output, sw_2.output_false});
  ops::Merge m2(root.WithOpName("m2"), {sw_1.output_false, sw_2.output_false});
  ops::Merge m3(root.WithOpName("m3"), {sw_0.output_false, m2.output});

  Output add = ops::Add(root.WithOpName("add"), m1.output, m3.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, AndOfOr) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");
  ops::Switch sw_3 = CreateSwitch(root, "3");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {sw_2.output_false, sw_3.output_false});

  Output add0 = ops::Add(root.WithOpName("add0"), m0.output, m1.output);
  Output add1 = ops::Add(root.WithOpName("add1"), m0.output, m1.output);

  Output add2 = ops::Add(root.WithOpName("add2"), add0, add1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add2.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, OrOfAnd) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");
  ops::Switch sw_3 = CreateSwitch(root, "3");

  Output add0 =
      ops::Add(root.WithOpName("add0"), sw_0.output_false, sw_1.output_false);
  Output add1 =
      ops::Add(root.WithOpName("add1"), sw_2.output_false, sw_3.output_false);

  ops::Merge m0(root.WithOpName("m0"), {add0, add1});
  ops::Merge m1(root.WithOpName("m1"), {add0, add1});

  Output add2 = ops::Add(root.WithOpName("add2"), m0.output, m1.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add2.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, AndOrDistributiveSimplified) {
  // (*A | (~*A & ((~*B & ~*A) | (~*A & *B)))) == #true
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "A");
  ops::Switch sw_1 = CreateSwitch(root, "B");
  Output add0 =
      ops::Add(root.WithOpName("and0"), sw_0.output_false, sw_1.output_true);
  Output add1 =
      ops::Add(root.WithOpName("and1"), sw_0.output_false, sw_1.output_false);
  ops::Merge or2(root.WithOpName("or2"), {add0, add1});
  Output add3 =
      ops::Add(root.WithOpName("and3"), or2.output, sw_0.output_false);
  ops::Merge or4(root.WithOpName("or4"), {add3, sw_0.output_true});

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));
  EXPECT_EQ(predicate_map[ControlOutputFor(or4.output)], "#true");
}

TEST(DeadnessAnalysisTest, AndOrDistributive) {
  // (A|B)&C == (A&C)|(B&C)
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  Output add0 = ops::Add(root.WithOpName("add0"), m0.output, sw_2.output_false);

  Output add1 =
      ops::Add(root.WithOpName("add1"), sw_0.output_false, sw_2.output_false);
  Output add2 =
      ops::Add(root.WithOpName("add2"), sw_1.output_false, sw_2.output_false);
  ops::Merge m1(root.WithOpName("m1"), {add1, add2});

  Output add3 = ops::Add(root.WithOpName("add3"), add0, m1.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add3.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, Ternary) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output predicate = ops::Placeholder(root.WithOpName("predicate"), DT_BOOL);
  Output true_value = ops::Placeholder(root.WithOpName("true_value"), DT_FLOAT);
  Output false_value =
      ops::Placeholder(root.WithOpName("false_value"), DT_FLOAT);

  ops::Switch predicated_true(root.WithOpName("predicated_true"), true_value,
                              predicate);

  ops::Switch predicated_false(root.WithOpName("predicated_false"), true_value,
                               predicate);
  ops::Merge merge(root.WithOpName("ternary"), {predicated_true.output_true,
                                                predicated_false.output_false});
  Output addend = ops::Placeholder(root.WithOpName("addend"), DT_FLOAT);
  Output add = ops::Add(root.WithOpName("add"), merge.output, addend);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, Recv) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv_a = ops::_Recv(root.WithOpName("recv_a"), DT_FLOAT, "tensor_a",
                             "sender", 0, "receiver");
  Output recv_b = ops::_Recv(root.WithOpName("recv_b"), DT_FLOAT, "tensor_b",
                             "sender", 0, "receiver");
  Output add = ops::Add(root.WithOpName("add"), recv_a, recv_b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, HostRecv) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv_a = ops::_HostRecv(root.WithOpName("recv_a"), DT_FLOAT,
                                 "tensor_a", "sender", 0, "receiver");
  Output recv_b = ops::_HostRecv(root.WithOpName("recv_b"), DT_FLOAT,
                                 "tensor_b", "sender", 0, "receiver");
  Output add = ops::Add(root.WithOpName("add"), recv_a, recv_b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, Loop) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output iv0 = CreateInductionVariable(root, "iv0", "fr0", 0).induction_var;
  Output iv1 = CreateInductionVariable(root, "iv1", "fr0", 0).induction_var;
  Output iv2 = CreateInductionVariable(root, "iv2", "fr0", 1).induction_var;
  Output add0 = ops::Add(root.WithOpName("add0"), iv0, iv1);
  Output add1 = ops::Add(root.WithOpName("add1"), iv1, iv2);

  // NB!  iv0 and iv1 are equivalent and a smarter deadness analysis would have
  // noticed that.  Today we are pessimistic here because we assign an
  // uninterpreted symbol to merges with backedges.

  VLogGraphIfAsked(*root.graph());

  {
    std::unique_ptr<DeadnessAnalysis> result;
    TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

    bool has_inputs_with_mismatching_deadness;

    TF_ASSERT_OK_AND_ASSIGN(
        has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add0.node()));
    EXPECT_TRUE(has_inputs_with_mismatching_deadness);

    TF_ASSERT_OK_AND_ASSIGN(
        has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add1.node()));
    EXPECT_TRUE(has_inputs_with_mismatching_deadness);
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

    // In theory we should be able to tell that iv0/cond:0 and iv1/cond:0
    // produce the same deadness.  But we're not that smart today.
    EXPECT_EQ(predicate_map[ControlOutputFor(iv0)],
              "{#true,&,*iv0/cond:0}<fr0>");
    EXPECT_EQ(predicate_map[ControlOutputFor(iv1)],
              "{#true,&,*iv1/cond:0}<fr0>");
    EXPECT_EQ(predicate_map[ControlOutputFor(iv2)],
              "{#true,&,*iv2/cond:0}<fr0>");
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              "({#true,&,*iv0/cond:0}<fr0> & {#true,&,*iv1/cond:0}<fr0>)");
    EXPECT_EQ(predicate_map[ControlOutputFor(add1)],
              "({#true,&,*iv1/cond:0}<fr0> & {#true,&,*iv2/cond:0}<fr0>)");
  }
}

TEST(DeadnessAnalysisTest, ControlEquivalentLoopBodies) {
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv = CreateInductionVariable(root, "iv0", "loop", 0);
  Output dependent_iv0 =
      CreateDependentLoopInvariantValue(root, "div0", "loop", iv.loop_cond, 0)
          .induction_var;
  Output dependent_iv1 =
      CreateDependentLoopInvariantValue(root, "div1", "loop", iv.loop_cond, 0)
          .induction_var;
  Output add0 = ops::Add(root.WithOpName("add0"), dependent_iv0, dependent_iv1);

  VLogGraphIfAsked(*root.graph());

  {
    std::unique_ptr<DeadnessAnalysis> result;
    TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

    TF_ASSERT_OK_AND_ASSIGN(
        bool has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add0.node()));
    EXPECT_FALSE(has_inputs_with_mismatching_deadness);
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/true));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv0)],
              predicate_map[ControlOutputFor(iv.induction_var)]);
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv1)],
              predicate_map[ControlOutputFor(iv.induction_var)]);
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              predicate_map[ControlOutputFor(iv.induction_var)]);
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/false));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv0)],
              "{#true,&,(iv0/iv:0 & *iv0/cond:0)}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv1)],
              "{#true,&,(iv0/iv:0 & *iv0/cond:0)}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              "{#true,&,(iv0/iv:0 & *iv0/cond:0)}<loop>");
  }
}

TEST(DeadnessAnalysisTest, LoopInvariantPredicateOnBackedge) {
  // Create a merge that "looks like" a loop but isn't really.  It has a value
  // that does not depend on the merge on its backedge.
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv = CreateInductionVariable(root, "iv0", "frame", 0);
  DependentInductionVar dependent_iv =
      CreateDependentLoopInvariantValue(root, "div0", "frame", iv.loop_cond, 0);
  FixupSourceAndSinkEdges(root.graph());

  TF_ASSERT_OK(root.graph()->UpdateEdge(
      iv.induction_var.node(), 0, dependent_iv.latch.output_true.node(), 0));

  VLogGraphIfAsked(*root.graph());

  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/true));

    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv.induction_var)],
              "{#true,&,*iv0/cond:0}<frame>");
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/false));

    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_iv.induction_var)],
              "div0/iv:0");
  }
}

TEST(DeadnessAnalysisTest, ControlEquivalentNestedLoopBodies) {
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv_outer =
      CreateInductionVariable(root, "iv_outer", "outer_loop", 0);
  Output enter_constant_outer_loop = ops::internal::Enter(
      root.WithOpName("constant_enter_outer_loop"),
      ops::Const(root.WithOpName("constant"), 5), "outer_loop",
      ops::internal::Enter::Attrs().IsConstant(true));
  ops::Switch inner_value(root.WithOpName("outer_is_live"),
                          enter_constant_outer_loop, iv_outer.loop_cond);
  InductionVarInfo iv_inner = CreateInductionVariable(
      root, "iv_inner", "inner_loop", inner_value.output_true);

  Output dependent_outer_iv0 =
      CreateDependentLoopInvariantValue(root, "dependent_outer_iv0",
                                        "outer_loop", iv_outer.loop_cond, 0)
          .induction_var;
  Output dependent_outer_iv1 =
      CreateDependentLoopInvariantValue(root, "dependent_outer_iv1",
                                        "outer_loop", iv_outer.loop_cond, 0)
          .induction_var;

  Output dependent_inner_iv0 = CreateDependentLoopInvariantValue(
                                   root, "dependent_inner_iv0", "inner_loop",
                                   iv_inner.loop_cond, dependent_outer_iv0)
                                   .induction_var;
  Output dependent_inner_iv1 = CreateDependentLoopInvariantValue(
                                   root, "dependent_inner_iv1", "inner_loop",
                                   iv_inner.loop_cond, dependent_outer_iv1)
                                   .induction_var;

  Output add0 = ops::Add(root.WithOpName("add0"), dependent_inner_iv0,
                         dependent_inner_iv1);

  VLogGraphIfAsked(*root.graph());

  {
    std::unique_ptr<DeadnessAnalysis> result;
    TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

    TF_ASSERT_OK_AND_ASSIGN(
        bool has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add0.node()));
    EXPECT_FALSE(has_inputs_with_mismatching_deadness);
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/true));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv_outer.induction_var)],
              "{#true,&,*iv_outer/cond:0}<outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(iv_inner.induction_var)],
              "{(*iv_outer/cond:0 & "
              "{#true,&,*iv_outer/cond:0}<outer_loop>),&,*iv_inner/"
              "cond:0}<inner_loop;outer_loop>");

    // enable_optimistic = true or not should produce the same results because
    // of fallback.  However, note that the order of iv_inner/cond:0 and
    // iv_inner/iv:0 is different because the optimistic approach does not
    // create predicates for all merges and it can change the predicate id and
    // hence the symbol order.
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_inner_iv0)],
              "{{#true,&,(iv_outer/iv:0 & "
              "*iv_outer/cond:0)}<outer_loop>,&,(*iv_inner/cond:0 & "
              "iv_inner/iv:0)}<inner_loop;outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_inner_iv1)],
              predicate_map[ControlOutputFor(dependent_inner_iv0)]);
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              predicate_map[ControlOutputFor(dependent_inner_iv0)]);
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/false));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv_outer.induction_var)],
              "{#true,&,*iv_outer/cond:0}<outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(iv_inner.induction_var)],
              "{(*iv_outer/cond:0 & "
              "{#true,&,*iv_outer/cond:0}<outer_loop>),&,*iv_inner/"
              "cond:0}<inner_loop;outer_loop>");

    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_inner_iv0)],
              "{{#true,&,(iv_outer/iv:0 & "
              "*iv_outer/cond:0)}<outer_loop>,&,(iv_inner/iv:0 & "
              "*iv_inner/cond:0)}<inner_loop;outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(dependent_inner_iv1)],
              predicate_map[ControlOutputFor(dependent_inner_iv0)]);
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              predicate_map[ControlOutputFor(dependent_inner_iv0)]);
  }
}

TEST(DeadnessAnalysisTest, ControlNonEquivalentNestedLoopBodies) {
  Scope root = Scope::NewRootScope().ExitOnError();

  std::array<Output, 2> outer_iv;
  std::array<Output, 2> inner_iv;

  for (int i : {0, 1}) {
    InductionVarInfo iv_outer =
        CreateInductionVariable(root, "iv_outer", "outer_loop", 0);
    Output enter_constant_outer_loop = ops::internal::Enter(
        root.WithOpName("constant_enter_outer_loop"),
        ops::Const(root.WithOpName("constant"), 5), "outer_loop",
        ops::internal::Enter::Attrs().IsConstant(true));
    ops::Switch inner_value(root.WithOpName("outer_is_live"),
                            enter_constant_outer_loop, iv_outer.loop_cond);
    InductionVarInfo iv_inner = CreateInductionVariable(
        root, "iv_inner", "inner_loop", inner_value.output_true);

    outer_iv[i] = iv_outer.induction_var;
    inner_iv[i] = iv_inner.induction_var;
  }

  Output add0 = ops::Add(root.WithOpName("add0"), inner_iv[0], inner_iv[1]);

  VLogGraphIfAsked(*root.graph());

  {
    std::unique_ptr<DeadnessAnalysis> result;
    TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

    TF_ASSERT_OK_AND_ASSIGN(
        bool has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add0.node()));
    EXPECT_TRUE(has_inputs_with_mismatching_deadness);
  }

  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

    EXPECT_EQ(predicate_map[ControlOutputFor(outer_iv[0])],
              "{#true,&,*iv_outer/cond:0}<outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(inner_iv[0])],
              "{(*iv_outer/cond:0 & "
              "{#true,&,*iv_outer/cond:0}<outer_loop>),&,*iv_inner/"
              "cond:0}<inner_loop;outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(outer_iv[1])],
              "{#true,&,*iv_outer/cond_1:0}<outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(inner_iv[1])],
              "{(*iv_outer/cond_1:0 & "
              "{#true,&,*iv_outer/cond_1:0}<outer_loop>),&,*iv_inner/"
              "cond_1:0}<inner_loop;outer_loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(add0)],
              "({(*iv_outer/cond:0 & "
              "{#true,&,*iv_outer/cond:0}<outer_loop>),&,*iv_inner/"
              "cond:0}<inner_loop;outer_loop> & {(*iv_outer/cond_1:0 & "
              "{#true,&,*iv_outer/cond_1:0}<outer_loop>),&,*iv_inner/"
              "cond_1:0}<inner_loop;outer_loop>)");
  }
}

TEST(DeadnessAnalysisTest, NestedLoopBodiesWithACapture) {
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv_outer =
      CreateInductionVariable(root, "iv_outer", "outer_loop", 0);
  Output enter_constant_outer_loop = ops::internal::Enter(
      root.WithOpName("constant_enter_outer_loop"),
      ops::Const(root.WithOpName("constant"), 5), "outer_loop",
      ops::internal::Enter::Attrs().IsConstant(true));
  ops::Switch inner_value(root.WithOpName("outer_is_live"),
                          enter_constant_outer_loop, iv_outer.loop_cond);
  InductionVarInfo iv_inner = CreateInductionVariable(
      root, "iv_inner", "inner_loop", inner_value.output_true);

  DependentInductionVar div0_outer = CreateDependentLoopInvariantValue(
      root, "div0_outer", "outer_loop", iv_outer.loop_cond, 0);
  DependentInductionVar div1_outer = CreateDependentLoopInvariantValue(
      root, "div1_outer", "outer_loop", iv_outer.loop_cond, 0);

  DependentInductionVar div0_inner = CreateDependentLoopInvariantValue(
      root, "div0_inner", "inner_loop", iv_inner.loop_cond,
      div0_outer.induction_var);
  DependentInductionVar div1_inner = CreateDependentLoopInvariantValue(
      root, "div1_inner", "inner_loop", iv_inner.loop_cond,
      div1_outer.induction_var);

  Output captured = ops::_Recv(root.WithOpName("captured"), DT_INT32,
                               "tensor_a", "sender", 0, "receiver");
  Output capture_enter_outer = ops::internal::Enter(
      root.WithOpName("capture_enter_outer"), captured, "outer_loop",
      ops::internal::Enter::Attrs().IsConstant(true));
  Output capture_enter_inner = ops::internal::Enter(
      root.WithOpName("capture_enter_inner"), capture_enter_outer, "inner_loop",
      ops::internal::Enter::Attrs().IsConstant(true));
  Output mul0 = ops::Mul(root.WithOpName("mul0"), div1_inner.induction_var,
                         capture_enter_inner);
  TF_ASSERT_OK(root.graph()->UpdateEdge(
      mul0.node(), 0, div1_inner.latch.output_true.node(), 0));

  Output add0 = ops::Add(root.WithOpName("add0"), div0_inner.induction_var,
                         div1_inner.induction_var);

  VLogGraphIfAsked(*root.graph());

  {
    std::unique_ptr<DeadnessAnalysis> result;
    TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

    TF_ASSERT_OK_AND_ASSIGN(
        bool has_inputs_with_mismatching_deadness,
        HasInputsWithMismatchingDeadness(*result, *add0.node()));
    EXPECT_TRUE(has_inputs_with_mismatching_deadness);
  }
}

TEST(DeadnessAnalysisTest, CyclicRecurrence) {
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv = CreateInductionVariable(root, "iv0", "loop", 0);
  DependentInductionVar div0 =
      CreateDependentLoopInvariantValue(root, "div0", "loop", iv.loop_cond, 0);
  DependentInductionVar div1 =
      CreateDependentLoopInvariantValue(root, "div1", "loop", iv.loop_cond, 0);
  FixupSourceAndSinkEdges(root.graph());
  TF_ASSERT_OK(root.graph()->UpdateEdge(div1.induction_var.node(), 0,
                                        div0.latch.output_true.node(), 0));
  TF_ASSERT_OK(root.graph()->UpdateEdge(div0.induction_var.node(), 0,
                                        div1.latch.output_true.node(), 0));

  VLogGraphIfAsked(*root.graph());

  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/true));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(div0.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(div1.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");

    // This tests the rule {S,&,X} & ~X => S.
    TensorId switch_false_out = {div1.latch.output_false.node()->name(),
                                 div1.latch.output_false.index()};
    EXPECT_EQ(predicate_map[switch_false_out], "(#true)");
  }
  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map,
                                   /*enable_optimistic=*/false));

    EXPECT_EQ(predicate_map[ControlOutputFor(iv.induction_var)],
              "{#true,&,*iv0/cond:0}<loop>");
    EXPECT_EQ(predicate_map[ControlOutputFor(div0.induction_var)], "div0/iv:0");
    EXPECT_EQ(predicate_map[ControlOutputFor(div1.induction_var)], "div1/iv:0");
  }
}

TEST(DeadnessAnalysisTest, AndRecurrenceNeedsFrameName) {
  Scope root = Scope::NewRootScope().ExitOnError();
  InductionVarInfo iv_0 = CreateInductionVariable(root, "iv_0", "frame_0", 10);
  InductionVarInfo iv_1 = CreateInductionVariable(root, "iv_1", "frame_1", 9);

  Output init = CreateSwitch(root, "init").output_true;
  Output step = CreateSwitch(root, "step").output_true;

  std::array<Output, 2> exits;
  std::array<Output, 2> next_iterations;

  for (int i : {0, 1}) {
    Output init_enter = ops::internal::Enter(
        root.WithOpName(absl::StrCat("init_enter_frame_", i)), init,
        absl::StrCat("frame_", i),
        ops::internal::Enter::Attrs().IsConstant(true));
    Output step_enter = ops::internal::Enter(
        root.WithOpName(absl::StrCat("step_enter_frame_", i)), step,
        absl::StrCat("frame_", i),
        ops::internal::Enter::Attrs().IsConstant(true));

    ops::Merge iv(root.WithOpName(absl::StrCat("expr_", i)),
                  {init_enter, init_enter});
    Output add = ops::Add(root.WithOpName(absl::StrCat("add_", i)), iv.output,
                          step_enter);
    next_iterations[i] = ops::NextIteration(
        root.WithOpName(absl::StrCat("expr_", i, "_next_iteration")), add);
    EXPECT_TRUE(
        root.graph()
            ->UpdateEdge(next_iterations[i].node(), 0, iv.output.node(), 1)
            .ok());
    exits[i] = ops::internal::Exit(root.WithOpName(absl::StrCat("exit_", i)),
                                   iv.output);
  }

  FixupSourceAndSinkEdges(root.graph());

  {
    PredicateMapTy predicate_map;
    TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

    EXPECT_NE(predicate_map[ControlOutputFor(exits[0])],
              predicate_map[ControlOutputFor(exits[1])]);
    EXPECT_NE(predicate_map[ControlOutputFor(exits[0])], "");
    EXPECT_NE(predicate_map[ControlOutputFor(exits[1])], "");

    EXPECT_NE(predicate_map[ControlOutputFor(next_iterations[0])],
              predicate_map[ControlOutputFor(next_iterations[1])]);
    EXPECT_NE(predicate_map[ControlOutputFor(next_iterations[0])], "");
    EXPECT_NE(predicate_map[ControlOutputFor(next_iterations[1])], "");
  }
}

TEST(DeadnessAnalysisTest, ControlInputs) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  Output const0 = ops::Const(root.WithOpName("const0"), 1);
  Output const1 = ops::Const(root.WithOpName("const1"), 2);

  Output add = ops::Add(root.WithOpName("add"), const0, const1);

  root.graph()->AddControlEdge(id0.node(), const0.node());
  root.graph()->AddControlEdge(id1.node(), const1.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, ControlTrigger) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  ops::ControlTrigger ctrl_trigger0(root.WithOpName("ctrl_trigger0"));
  ops::ControlTrigger ctrl_trigger1(root.WithOpName("ctrl_trigger1"));

  Output const0 = ops::Const(root.WithOpName("const0"), 1);
  Output const1 = ops::Const(root.WithOpName("const1"), 2);

  Output add = ops::Add(root.WithOpName("add"), const0, const1);

  root.graph()->AddControlEdge(id0.node(), ctrl_trigger0.operation.node());
  root.graph()->AddControlEdge(ctrl_trigger0.operation.node(), const0.node());

  root.graph()->AddControlEdge(id1.node(), ctrl_trigger1.operation.node());
  root.graph()->AddControlEdge(ctrl_trigger1.operation.node(), const1.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, ControlInputsToMerge) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  Output constant = ops::Const(root.WithOpName("constant"), 5);
  ops::Merge m0(root.WithOpName("m0"), {constant});
  ops::Merge m1(root.WithOpName("m0"), {constant});
  Output add = ops::Add(root.WithOpName("add"), m0.output, m1.output);

  root.graph()->AddControlEdge(id0.node(), m0.output.node());
  root.graph()->AddControlEdge(id1.node(), m1.output.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *add.node()));
  EXPECT_FALSE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, RecvVsSwitch) {
  // Demonstrates why we need the must_be_true bit on SymbolP.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv = ops::_Recv(root.WithOpName("recv"), DT_BOOL, "tensor", "sender",
                           0, "receiver");
  Output value = ops::Placeholder(root.WithOpName("value"), DT_BOOL);
  ops::Switch sw(root.WithOpName("switch"), value, recv);
  Output logical_and =
      ops::LogicalAnd(root.WithOpName("and"), recv, sw.output_true);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  TF_ASSERT_OK_AND_ASSIGN(
      bool has_inputs_with_mismatching_deadness,
      HasInputsWithMismatchingDeadness(*result, *logical_and.node()));
  EXPECT_TRUE(has_inputs_with_mismatching_deadness);
}

TEST(DeadnessAnalysisTest, RecvVsSwitchText) {
  // Demonstrates why we need the must_be_true bit on SymbolP.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv = ops::_Recv(root.WithOpName("recv"), DT_BOOL, "tensor", "sender",
                           0, "receiver");
  Output value = ops::Placeholder(root.WithOpName("value"), DT_BOOL);
  ops::Switch sw(root.WithOpName("switch"), value, recv);
  Output logical_and =
      ops::LogicalAnd(root.WithOpName("and"), recv, sw.output_true);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  TensorId logical_and_output_0 = {logical_and.node()->name(),
                                   Graph::kControlSlot};
  EXPECT_EQ(predicate_map[logical_and_output_0], "(recv:0 & *recv:0)");
}

TEST(DeadnessAnalysisTest, DeMorgan) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_0 = ops::Placeholder(root.WithOpName("cond_0"), DT_BOOL);
  Output cond_1 = ops::Placeholder(root.WithOpName("cond_1"), DT_BOOL);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch sw_0(root.WithOpName("switch_0"), value, cond_0);
  ops::Switch sw_1(root.WithOpName("switch_1"), value, cond_1);

  Output and_0_1 =
      ops::Add(root.WithOpName("and_0_1"), sw_0.output_true, sw_1.output_true);

  Output or_not0_not1 = ops::Merge(root.WithOpName("or_not0_not1"),
                                   {sw_0.output_false, sw_1.output_false})
                            .output;

  // Predicate(should_always_be_dead) =
  // (A & B) & (~A | ~B) = (A & B) & ~(A & B) = False
  Output should_always_be_dead =
      ops::Add(root.WithOpName("should_always_be_dead"), and_0_1, or_not0_not1);

  // Predicate(should_always_be_dead) =
  // (A & B) | (~A | ~B) = (A & B) | ~(A & B) = True
  Output should_always_be_alive =
      ops::Merge(root.WithOpName("should_always_be_alive"),
                 {and_0_1, or_not0_not1})
          .output;

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(should_always_be_dead)], "#false");
  EXPECT_EQ(predicate_map[ControlOutputFor(should_always_be_alive)], "#true");
}

TEST(DeadnessAnalysisTest, ConstantTrueSwitchCondition) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output constant_true = ops::Const(root.WithOpName("const_true"), true);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  ops::Switch sw(root.WithOpName("switch"), value, constant_true);

  Output id_false = ops::Identity(root.WithOpName("id_false"), sw.output_false);
  Output id_true = ops::Identity(root.WithOpName("id_true"), sw.output_true);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_false)], "#false");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_true)], "#true");
}

TEST(DeadnessAnalysisTest, ConstantFalseSwitchCondition) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output constant_false = ops::Const(root.WithOpName("const_false"), false);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  ops::Switch sw(root.WithOpName("switch"), value, constant_false);

  Output id_false = ops::Identity(root.WithOpName("id_false"), sw.output_false);
  Output id_true = ops::Identity(root.WithOpName("id_true"), sw.output_true);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_false)], "#true");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_true)], "#false");
}

TEST(DeadnessAnalysisTest, RefBoolSwitchCondition) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output condition_ref_var =
      ops::Variable(root.WithOpName("cond_ref"), TensorShape({}), DT_BOOL);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  ops::Switch sw(root.WithOpName("switch"), value, condition_ref_var);

  Output id_false = ops::Identity(root.WithOpName("id_false"), sw.output_false);
  Output id_true = ops::Identity(root.WithOpName("id_true"), sw.output_true);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_false)], "~*cond_ref:0");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_true)], "*cond_ref:0");
}

void CreateSwitchN(const Scope& scope, Input data, Input output_index,
                   int64_t num_outs, OutputList* outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdeadness_analysis_testDTcc mht_8(mht_8_v, 1427, "", "./tensorflow/compiler/jit/deadness_analysis_test.cc", "CreateSwitchN");

  if (!scope.ok()) return;
  auto _data = ops::AsNodeOut(scope, data);
  if (!scope.ok()) return;
  auto _output_index = ops::AsNodeOut(scope, output_index);
  if (!scope.ok()) return;
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("_SwitchN");
  auto builder = NodeBuilder(unique_name, "_SwitchN")
                     .Input(_data)
                     .Input(_output_index)
                     .Attr("num_outs", num_outs);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  for (int32_t i = 0; i < ret->num_outputs(); ++i) {
    outputs->push_back(Output(ret, i));
  }
}

TEST(DeadnessAnalysisTest, Constant1_SwitchN_2Branches_DoesNotFail) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output constant_1 = ops::Const(root.WithOpName("const_1"), 1);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  OutputList outputs;
  CreateSwitchN(root.WithOpName("switchn"), value, constant_1, 2, &outputs);

  Output id_0 = ops::Identity(root.WithOpName("id_0"), outputs[0]);
  Output id_1 = ops::Identity(root.WithOpName("id_1"), outputs[1]);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_0)], "#false");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_1)], "#true");
}

TEST(DeadnessAnalysisTest, Constant7_SwitchN_3Branches) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output constant_7 = ops::Const(root.WithOpName("const_7"), 7);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  OutputList outputs;
  CreateSwitchN(root.WithOpName("switchn"), value, constant_7, 3, &outputs);

  Output id_0 = ops::Identity(root.WithOpName("id_0"), outputs[0]);
  Output id_1 = ops::Identity(root.WithOpName("id_1"), outputs[1]);
  Output id_2 = ops::Identity(root.WithOpName("id_2"), outputs[2]);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_0)], "#false");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_1)], "#false");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_2)], "#true");
}

TEST(DeadnessAnalysisTest, RefInt_SwitchN_3Branches) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output condition_ref_var =
      ops::Variable(root.WithOpName("bidx"), TensorShape({}), DT_INT32);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  OutputList outputs;
  CreateSwitchN(root.WithOpName("switchn"), value, condition_ref_var, 3,
                &outputs);

  Output id_0 = ops::Identity(root.WithOpName("id_0"), outputs[0]);
  Output id_1 = ops::Identity(root.WithOpName("id_1"), outputs[1]);
  Output id_2 = ops::Identity(root.WithOpName("id_2"), outputs[2]);

  FixupSourceAndSinkEdges(root.graph());

  PredicateMapTy predicate_map;
  TF_ASSERT_OK(ComputePredicates(*root.graph(), &predicate_map));

  EXPECT_EQ(predicate_map[ControlOutputFor(id_0)], "bidx:0=0");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_1)], "(~bidx:0=0 & bidx:0=1)");
  EXPECT_EQ(predicate_map[ControlOutputFor(id_2)], "(~bidx:0=0 & ~bidx:0=1)");
}

}  // namespace
}  // namespace tensorflow
