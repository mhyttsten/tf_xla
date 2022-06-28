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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_graph_dumper_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_graph_dumper_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_graph_dumper_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace {

using absl::StrCat;
using ::testing::HasSubstr;

using HloGraphDumperTest = HloTestBase;

std::string TestName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_graph_dumper_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/hlo_graph_dumper_test.cc", "TestName");

  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST_F(HloGraphDumperTest, NestedFusion) {
  HloComputation::Builder b("b");

  // Build param0 + param1 + param2 + param3 + param4.
  auto shape = ShapeUtil::MakeShape(F32, {10, 100});
  std::vector<HloInstruction*> params;
  for (int i = 0; i <= 4; ++i) {
    params.push_back(b.AddInstruction(
        HloInstruction::CreateParameter(i, shape, StrCat("param", i))));
  }
  std::vector<HloInstruction*> sums;
  sums.push_back(b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, params[0], params[1])));
  for (int i = 0; i <= 2; ++i) {
    sums.push_back(b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, sums[i], params[i + 2])));
  }
  HloModuleConfig config;
  HloModule m(TestName(), config);
  m.AddEntryComputation(b.Build());
  HloComputation* root_computation = m.entry_computation();

  // Fuse into fusion(param0 + param1 + param2 + param3 + param4).
  auto* outer_fusion = root_computation->CreateFusionInstruction(
      {sums[3], sums[2], sums[1], sums[0]}, HloInstruction::FusionKind::kLoop);

  // Fusing invalidates the pointers in sums -- the instructions are cloned when
  // they're moved to the new computation.  Get the updated pointers to sums.
  std::vector<HloInstruction*> fused_sums;
  for (auto* instr : outer_fusion->fused_instructions_computation()
                         ->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kAdd) {
      fused_sums.push_back(instr);
    }
  }

  // Fuse into fusion(fusion(param0 + param1 + param2) + param3 + param4).
  auto* inner_fusion =
      outer_fusion->fused_instructions_computation()->CreateFusionInstruction(
          {fused_sums[1], fused_sums[0]}, HloInstruction::FusionKind::kLoop);

  // Generate the graph; all nodes should be present.
  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*root_computation, /*label=*/"", DebugOptions(),
                  RenderedGraphFormat::kDot));
  for (const HloComputation* computation :
       {root_computation,  //
        inner_fusion->fused_instructions_computation(),
        outer_fusion->fused_instructions_computation()}) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_THAT(graph, HasSubstr(instruction->name()));
    }
  }

  // Dump a neighborhood around one of the inner sum nodes.  We don't really
  // care that the outer nodes are omitted -- whether they are or not is based
  // fiddly heuristics -- but we do care that the node we asked for is printed.
  const HloInstruction* inner_sum = nullptr;
  for (const HloInstruction* instruction :
       inner_fusion->fused_instructions_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kAdd) {
      inner_sum = instruction;
      break;
    }
  }
  ASSERT_NE(inner_sum, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::string neighborhood_graph,
                          RenderNeighborhoodAround(*inner_sum, /*radius=*/1,
                                                   RenderedGraphFormat::kDot));
  EXPECT_THAT(neighborhood_graph, HasSubstr(inner_sum->name()));
}

TEST_F(HloGraphDumperTest, Constant) {
  HloComputation::Builder b("b");
  auto instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-42)));
  instruction->SetAndSanitizeName("i_am_a_constant_root_instruction");
  HloModuleConfig config;
  HloModule m(TestName(), config);
  HloComputation* root_computation = m.AddEntryComputation(b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*root_computation, /*label=*/"an_empty_graph", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("an_empty_graph"));
}

TEST_F(HloGraphDumperTest, TupleConstant) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(S32, {4, 5})});
  HloComputation::Builder b("b");
  auto constant = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(tuple_shape)));
  auto gte = b.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::MakeShape(F32, {3, 2}), constant, 0));

  HloModuleConfig config;
  HloModule m(TestName(), config);
  HloComputation* root_computation = m.AddEntryComputation(b.Build(gte));
  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*root_computation, /*label=*/"tuple_constant", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("tuple_constant"));
  EXPECT_THAT(graph, HasSubstr("constant (f32[3,2], s32[4,5])"));
}

TEST_F(HloGraphDumperTest, Compare) {
  const char* hlo_string = R"(
    HloModule comp

    ENTRY comp {
      param.0 = f32[10] parameter(0)
      param.1 = f32[10] parameter(1)
      ROOT lt = pred[10] compare(param.0, param.1), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("direction=LT"));
}

TEST_F(HloGraphDumperTest, RootIsConstant) {
  const char* hlo_string = R"(
HloModule indexed_conditional

%then_branch (empty: ()) -> f32[] {
  %empty = () parameter(0)
  ROOT %then = f32[] constant(1)
}

%else_branch (empty.1: ()) -> f32[] {
  %empty.1 = () parameter(0)
  ROOT %else = f32[] constant(2)
}

ENTRY %conditional_select (constant: pred[]) -> (f32[]) {
  %constant = pred[] parameter(0)
  %emptytuple = () tuple()
  %conditional = f32[] conditional(pred[] %constant, () %emptytuple, () %emptytuple), true_computation=%then_branch, false_computation=%else_branch
  ROOT %t = (f32[]) tuple(f32[] %conditional)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Just check that it doesn't crash.
  TF_ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot));
}

}  // anonymous namespace
}  // namespace xla
