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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graph_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graph_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graph_testDTcc() {
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

#include "tensorflow/compiler/xla/service/call_graph.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;

class CallGraphTest : public HloTestBase {
 protected:
  // Build and return a trivial computation taking and returning a scalar.
  std::unique_ptr<HloComputation> MakeScalarComputation(
      HloOpcode opcode = HloOpcode::kNegate) {
    HloComputation::Builder builder(TestName() + ".ScalarComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(kScalarShape, opcode, param0));
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and maps (kMap) the
  // given computation to the value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeMappingComputation(
      HloComputation* map_computation, int64_t callsites) {
    HloComputation::Builder builder(TestName() + ".MappingComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateMap(
          kScalarShape, {last_value}, map_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and calls (kCall) the
  // given computation with value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeCallingComputation(
      HloComputation* callee_computation, int64_t callsites,
      const std::string& suffix = ".CallingComputation") {
    HloComputation::Builder builder(TestName() + suffix);
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateCall(
          kScalarShape, {last_value}, callee_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and returns a PRED
  // value.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    HloComputation::Builder builder(TestName() + ".ConditionComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* zero = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
    builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                      zero, ComparisonDirection::kGt));
    return builder.Build();
  }

  const Shape kScalarShape = ShapeUtil::MakeShape(F32, {});
};

TEST_F(CallGraphTest, SingletonComputation) {
  // Test the call graph of a module with a single computation.
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeScalarComputation());
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(1, call_graph->nodes().size());
  EXPECT_TRUE(call_graph->IsFlattened());

  const CallGraphNode& node = call_graph->GetNode(computation);
  EXPECT_EQ(computation, node.computation());
  EXPECT_EQ(node.depth(), 0);
  EXPECT_TRUE(node.callsites().empty());
  EXPECT_TRUE(node.callees().empty());
  EXPECT_TRUE(node.caller_callsites().empty());
  EXPECT_TRUE(node.callers().empty());
  EXPECT_EQ(CallContext::kControlFlow, node.context());
}

TEST_F(CallGraphTest, UnreachableComputation) {
  // Test the call graph of a module with an entry computation and an
  // unreachable computation.
  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeScalarComputation());
  HloComputation* unreachable_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(2, call_graph->nodes().size());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  EXPECT_EQ(entry_node.depth(), 0);
  EXPECT_EQ(entry_computation, entry_node.computation());
  EXPECT_EQ(CallContext::kControlFlow, entry_node.context());

  const CallGraphNode& unreachable_node =
      call_graph->GetNode(unreachable_computation);
  EXPECT_EQ(unreachable_node.depth(), 0);
  EXPECT_EQ(unreachable_computation, unreachable_node.computation());
  EXPECT_EQ(CallContext::kControlFlow, unreachable_node.context());
}

TEST_F(CallGraphTest, ParallelComputation) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in a parallel context via kMap.
  auto module = CreateNewVerifiedModule();
  HloComputation* map_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* entry_computation = module->AddEntryComputation(
      MakeMappingComputation(map_computation, /*callsites=*/5));

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(2, call_graph->nodes().size());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  EXPECT_EQ(entry_computation, entry_node.computation());
  EXPECT_EQ(entry_node.depth(), 0);
  EXPECT_EQ(CallContext::kControlFlow, entry_node.context());
  EXPECT_EQ(5, entry_node.callsites().size());
  EXPECT_EQ(1, entry_node.callees().size());
  EXPECT_TRUE(entry_node.caller_callsites().empty());
  EXPECT_TRUE(entry_node.callers().empty());

  const CallGraphNode& map_node = call_graph->GetNode(map_computation);
  EXPECT_EQ(map_computation, map_node.computation());
  EXPECT_EQ(map_node.depth(), 1);
  EXPECT_EQ(CallContext::kEmbedded, map_node.context());
  EXPECT_TRUE(map_node.callsites().empty());
  EXPECT_TRUE(map_node.callees().empty());
  EXPECT_EQ(5, map_node.caller_callsites().size());
  EXPECT_EQ(1, map_node.callers().size());
}

TEST_F(CallGraphTest, SequentialComputations) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in a sequential context via kCall.
  auto module = CreateNewVerifiedModule();
  HloComputation* called_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* entry_computation = module->AddEntryComputation(
      MakeCallingComputation(called_computation, /*callsites=*/3));

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(2, call_graph->nodes().size());

  // The called computation is only called from one other computation, but there
  // are multiple callsites.
  EXPECT_FALSE(call_graph->IsFlattened());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  EXPECT_EQ(entry_computation, entry_node.computation());
  EXPECT_EQ(CallContext::kControlFlow, entry_node.context());
  EXPECT_EQ(3, entry_node.callsites().size());
  EXPECT_EQ(1, entry_node.callees().size());
  EXPECT_TRUE(entry_node.caller_callsites().empty());
  EXPECT_TRUE(entry_node.callers().empty());

  const CallGraphNode& called_node = call_graph->GetNode(called_computation);
  EXPECT_EQ(called_computation, called_node.computation());
  EXPECT_EQ(CallContext::kControlFlow, called_node.context());
  EXPECT_TRUE(called_node.callsites().empty());
  EXPECT_TRUE(called_node.callees().empty());
  EXPECT_EQ(3, called_node.caller_callsites().size());
  EXPECT_EQ(1, called_node.callers().size());
}

TEST_F(CallGraphTest, ContextBothComputations) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in both a parallel and sequential context.
  auto module = CreateNewVerifiedModule();
  HloComputation* subcomputation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, kScalarShape, "param0"));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(kScalarShape, {param0}, subcomputation));
  HloInstruction* map = builder.AddInstruction(
      HloInstruction::CreateMap(kScalarShape, {call}, subcomputation));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(2, call_graph->nodes().size());

  EXPECT_FALSE(call_graph->IsFlattened());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  EXPECT_EQ(entry_computation, entry_node.computation());
  EXPECT_EQ(2, entry_node.callsites().size());

  const CallSite& call_callsite = entry_node.callsites()[0];
  EXPECT_EQ(call, call_callsite.instruction());
  EXPECT_THAT(call_callsite.called_computations(),
              UnorderedElementsAre(subcomputation));
  EXPECT_EQ(CallContext::kControlFlow, call_callsite.context());
  EXPECT_EQ(entry_node.GetCallSite(call), &call_callsite);

  const CallSite& map_callsite = entry_node.callsites()[1];
  EXPECT_EQ(map, map_callsite.instruction());
  EXPECT_THAT(map_callsite.called_computations(),
              UnorderedElementsAre(subcomputation));
  EXPECT_EQ(CallContext::kEmbedded, map_callsite.context());
  EXPECT_EQ(entry_node.GetCallSite(map), &map_callsite);

  const CallGraphNode& sub_node = call_graph->GetNode(subcomputation);
  EXPECT_EQ(sub_node.depth(), 1);
  EXPECT_EQ(CallContext::kBoth, sub_node.context());
}

TEST_F(CallGraphTest, ComputationWithConditional) {
  // Test a call graph of a module with a conditional.
  auto module = CreateNewVerifiedModule();
  HloComputation* true_computation =
      module->AddEmbeddedComputation(MakeScalarComputation(HloOpcode::kCeil));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(MakeScalarComputation(HloOpcode::kFloor));

  HloComputation::Builder builder(TestName());
  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloInstruction* const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(56.4f)));
  HloInstruction* const2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12.6f)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          kScalarShape, pred, const1, true_computation, const2,
          false_computation));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());

  EXPECT_EQ(3, call_graph->nodes().size());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  EXPECT_EQ(entry_node.depth(), 0);
  EXPECT_EQ(entry_computation, entry_node.computation());
  EXPECT_EQ(1, entry_node.callsites().size());

  const CallSite& conditional_callsite = entry_node.callsites()[0];
  EXPECT_EQ(conditional, conditional_callsite.instruction());
  EXPECT_THAT(conditional_callsite.called_computations(),
              UnorderedElementsAre(true_computation, false_computation));
  EXPECT_EQ(CallContext::kControlFlow, conditional_callsite.context());
  EXPECT_EQ(entry_node.GetCallSite(conditional), &conditional_callsite);

  const CallGraphNode& true_node = call_graph->GetNode(true_computation);
  EXPECT_EQ(true_node.depth(), 1);
  EXPECT_TRUE(true_node.callees().empty());
  EXPECT_EQ(1, true_node.callers().size());
  EXPECT_EQ(entry_computation, true_node.callers()[0]);

  const CallGraphNode& false_node = call_graph->GetNode(false_computation);
  EXPECT_EQ(false_node.depth(), 1);
  EXPECT_TRUE(false_node.callees().empty());
  EXPECT_EQ(1, false_node.callers().size());
  EXPECT_EQ(entry_computation, false_node.callers()[0]);
}

TEST_F(CallGraphTest, ComplexGraph) {
  // Test a call graph of a module with several computation called in various
  // contexts. The call graph looks like:
  //
  //      entry
  //      /  |
  //     a   |
  //   / | \ |
  //  b  |  cond
  //   \ |
  //    c
  //
  // Calls are made via kCall, kWhile, and kMap instructions.
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(MakeConditionComputation());
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeMappingComputation(c_computation, /*callsites=*/1));

  HloComputation* a_computation;
  {
    HloComputation::Builder builder(TestName() + ".a");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* call = builder.AddInstruction(
        HloInstruction::CreateCall(kScalarShape, {param0}, c_computation));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, b_computation, call));
    a_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, a_computation, param0));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(5, call_graph->nodes().size());
  EXPECT_FALSE(call_graph->IsFlattened());

  const CallGraphNode& entry_node = call_graph->GetNode(entry_computation);
  const CallGraphNode& a_node = call_graph->GetNode(a_computation);
  const CallGraphNode& b_node = call_graph->GetNode(b_computation);
  const CallGraphNode& c_node = call_graph->GetNode(c_computation);
  const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);

  // Verify depths.
  EXPECT_EQ(entry_node.depth(), 0);
  EXPECT_EQ(a_node.depth(), 1);
  EXPECT_EQ(b_node.depth(), 2);
  EXPECT_EQ(c_node.depth(), 3);
  EXPECT_EQ(cond_node.depth(), 2);

  // Entry computation has one while instruction calling two computations
  // (cond_computation and a_computation).
  ASSERT_EQ(1, entry_node.callsites().size());
  auto called_computations = entry_node.callsites()[0].called_computations();
  EXPECT_THAT(called_computations,
              UnorderedElementsAre(cond_computation, a_computation));
  EXPECT_EQ(CallContext::kControlFlow, entry_node.context());

  EXPECT_TRUE(c_node.callsites().empty());
  EXPECT_THAT(c_node.callers(),
              UnorderedElementsAre(a_computation, b_computation));
  EXPECT_EQ(CallContext::kBoth, c_node.context());

  // Visit the graph and verify nodes were visited in callee-before-caller
  // order.
  std::vector<const HloComputation*> visited;
  TF_ASSERT_OK(call_graph->VisitNodes([&visited](const CallGraphNode& node) {
    visited.push_back(node.computation());
    return Status::OK();
  }));
  EXPECT_EQ(visited.size(), 5);
  // All values in visited should be unique.
  EXPECT_EQ(
      absl::flat_hash_set<const HloComputation*>(visited.begin(), visited.end())
          .size(),
      5);

  // Verify visitation order of some computations in the graph.
  auto index_of = [&visited](const HloComputation* comp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graph_testDTcc mht_0(mht_0_v, 556, "", "./tensorflow/compiler/xla/service/call_graph_test.cc", "lambda");

    auto it = absl::c_find(visited, comp);
    EXPECT_NE(it, visited.end());
    return std::distance(visited.begin(), it);
  };
  EXPECT_EQ(4, index_of(entry_computation));
  EXPECT_LT(index_of(cond_computation), index_of(a_computation));
  EXPECT_LT(index_of(c_computation), index_of(b_computation));
  EXPECT_LT(index_of(b_computation), index_of(a_computation));

  // Verify dominance relations between computation in the graph.

  // Entry dominates everybody, and is dominated by no one except itself.
  EXPECT_TRUE(call_graph->Dominates(entry_computation, entry_computation));
  EXPECT_TRUE(call_graph->Dominates(entry_computation, a_computation));
  EXPECT_TRUE(call_graph->Dominates(entry_computation, b_computation));
  EXPECT_TRUE(call_graph->Dominates(entry_computation, c_computation));
  EXPECT_TRUE(call_graph->Dominates(entry_computation, cond_computation));
  EXPECT_FALSE(call_graph->Dominates(a_computation, entry_computation));
  EXPECT_FALSE(call_graph->Dominates(b_computation, entry_computation));
  EXPECT_FALSE(call_graph->Dominates(c_computation, entry_computation));
  EXPECT_FALSE(call_graph->Dominates(cond_computation, entry_computation));

  // 'a' only dominates 'b' and 'c'.
  EXPECT_TRUE(call_graph->Dominates(a_computation, a_computation));
  EXPECT_TRUE(call_graph->Dominates(a_computation, b_computation));
  EXPECT_TRUE(call_graph->Dominates(a_computation, c_computation));
  EXPECT_FALSE(call_graph->Dominates(b_computation, a_computation));
  EXPECT_FALSE(call_graph->Dominates(c_computation, a_computation));
  EXPECT_FALSE(call_graph->Dominates(a_computation, cond_computation));

  EXPECT_TRUE(call_graph->Dominates(b_computation, b_computation));
  EXPECT_FALSE(call_graph->Dominates(b_computation, c_computation));
  EXPECT_FALSE(call_graph->Dominates(b_computation, cond_computation));

  EXPECT_TRUE(call_graph->Dominates(c_computation, c_computation));
  EXPECT_FALSE(call_graph->Dominates(c_computation, cond_computation));
  EXPECT_FALSE(call_graph->Dominates(cond_computation, c_computation));

  EXPECT_TRUE(call_graph->Dominates(cond_computation, cond_computation));
}

TEST_F(CallGraphTest, ComplexGraphNearestAncestors) {
  // Test NearestAncestorsInSameComputation on a call graph of a module with
  // several computation called in various contexts. The call graph looks like:
  //
  //      entry
  //      /  |
  //     a   |
  //   / | \ |
  //  b  |  cond
  //   \ |
  //    c
  //
  // Calls are made via kCall, kWhile, and kMap instructions.
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(MakeConditionComputation());
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeMappingComputation(c_computation, /*callsites=*/1));
  HloInstruction* b_map = b_computation->root_instruction();

  HloComputation* a_computation;
  HloInstruction* a_call;
  HloInstruction* a_while;
  {
    HloComputation::Builder builder(TestName() + ".a");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    a_call = builder.AddInstruction(
        HloInstruction::CreateCall(kScalarShape, {param0}, c_computation));
    a_while = builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, b_computation, a_call));
    a_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  HloInstruction* entry_while;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    entry_while = builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, a_computation, param0));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(5, call_graph->nodes().size());

  // Verify NearestAncestorsInSameComputation for various instructions in the
  // module.
  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(a_call, a_call),
            std::make_pair(a_call, a_call));

  // c_computation is called from more than one site, so
  // NearestAncestorsInSameComputation bails and returns nullptrs.
  std::pair<HloInstruction*, HloInstruction*> null_pair = {nullptr, nullptr};
  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(
                b_map, c_computation->root_instruction()),
            null_pair);

  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(b_map, entry_while),
            std::make_pair(entry_while, entry_while));
  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(b_map, a_call),
            std::make_pair(a_while, a_call));
  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(a_while, a_call),
            std::make_pair(a_while, a_call));
  EXPECT_EQ(call_graph->NearestAncestorsInSameComputation(a_while, b_map),
            std::make_pair(a_while, a_while));
}

TEST_F(CallGraphTest, VisitSingletonComputation) {
  // Test the call graph visitor with a call graph with a single node.
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeScalarComputation());
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());

  std::vector<HloComputation*> visited;
  TF_ASSERT_OK(call_graph->VisitNodes([&visited](const CallGraphNode& node) {
    visited.push_back(node.computation());
    return Status::OK();
  }));
  EXPECT_THAT(visited, UnorderedElementsAre(computation));
}

TEST_F(CallGraphTest, VisitUnreachableComputation) {
  // Test the call graph visitor with a call graph with an unreachable node.
  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeScalarComputation());
  HloComputation* unreachable_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());

  // Test visitation of only reachable nodes.
  {
    std::vector<const HloComputation*> visited;
    TF_ASSERT_OK(call_graph->VisitNodes(
        [&visited](const CallGraphNode& node) {
          visited.push_back(node.computation());
          return Status::OK();
        },
        /*visit_unreachable_nodes=*/false));
    EXPECT_EQ(visited.size(), 1);
    EXPECT_EQ(visited[0], entry_computation);
  }

  // Test visitation of all nodes (reachable and unreachable).
  {
    std::vector<HloComputation*> visited;
    TF_ASSERT_OK(call_graph->VisitNodes(
        [&visited](const CallGraphNode& node) {
          visited.push_back(node.computation());
          return Status::OK();
        },
        /*visit_unreachable_nodes=*/true));
    EXPECT_EQ(visited.size(), 2);
    EXPECT_THAT(visited, UnorderedElementsAre(entry_computation,
                                              unreachable_computation));
  }
}

TEST_F(CallGraphTest, VisitWithError) {
  // Test that the call graph visitor properly propagates errors.
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(MakeScalarComputation());
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());

  Status status = call_graph->VisitNodes(
      [](const CallGraphNode&) { return InternalError("Visitation failed"); });

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), tensorflow::error::INTERNAL);
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("Visitation failed"));
}

}  // namespace
}  // namespace xla
