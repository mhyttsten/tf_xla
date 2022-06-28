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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_layout_assignment_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_layout_assignment_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_layout_assignment_testDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class CpuLayoutAssignmentTest : public HloTestBase {
 protected:
  void AssignLayouts(HloModule* module,
                     ComputationLayout* entry_computation_layout) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_layout_assignment_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/service/cpu/cpu_layout_assignment_test.cc", "AssignLayouts");

    cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features(
        [](int64_t shape_size) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        });
    cpu::CpuLayoutAssignment layout_assignment(entry_computation_layout,
                                               &target_machine_features);
    EXPECT_IS_OK(layout_assignment.Run(module).status());
  }
};

TEST_F(CpuLayoutAssignmentTest, DotWithConstantRhsTensor) {
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12}, {0});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {24}, {0});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0}),
                                dot_lhs->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                dot_rhs->shape().layout()));
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({0}), result->shape().layout()));
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, MultipleDotsWithSameConstantRhsTensor0) {
  // Two dot products have the same constant as the RHS, and both those dot
  // products can be optimized if the constant has a column-major layout.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12}, {0});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {24}, {0});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_b_lhs, dot_rhs));
  builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kAdd, dot_a_result, dot_b_result));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                dot_rhs->shape().layout()));
  for (HloInstruction* instruction :
       {dot_a_lhs, dot_b_lhs, dot_a_result, dot_b_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, MultipleDotsWithSameConstantRhsTensor1) {
  // Two dot products have the same constant as the RHS, but only one of the two
  // dot products can be optimized if the constant has a column-major layout.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_a_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape lhs_b_shape = ShapeUtil::MakeShapeWithLayout(F32, {2, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape result_a_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  Shape result_b_shape = ShapeUtil::MakeShapeWithLayout(F32, {2, 24}, {0, 1});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_a_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_b_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(
      CreateCanonicalDot(result_a_shape, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(
      CreateCanonicalDot(result_b_shape, dot_b_lhs, dot_rhs));
  auto tuple_result = builder.AddInstruction(
      HloInstruction::CreateTuple({dot_a_result, dot_b_result}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_a_shape));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_b_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(tuple_result->shape()));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction :
       {dot_rhs, dot_a_lhs, dot_b_lhs, dot_a_result, dot_b_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, DotWithConstantLhsTensor) {
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(lhs_shape)));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs_shape, "param0"));
  auto dot_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(rhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction : {dot_lhs, dot_rhs, dot_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, DotWithConstantRhsTensorThroughGTE) {
  // This is a case we could theoretically optimize at some point, but today we
  // don't.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape other_shape = ShapeUtil::MakeShapeWithLayout(F32, {100, 24}, {0, 1});

  auto constant_shape = ShapeUtil::MakeTupleShape({other_shape, rhs_shape});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(constant_shape)));

  Shape result_shape = ShapeUtil::MakeShape(F32, {1, 24});

  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, constant, 1));
  auto dot_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction : {dot_lhs, dot_rhs, dot_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

struct DotOutputFusionLayoutAssignmentResult {
  bool layout_assignment_changed_something;
  const HloInstruction* dot_lhs_fusion_param;
  const HloInstruction* dot_rhs_fusion_param;
  const HloInstruction* addend_fusion_param;
};

static StatusOr<DotOutputFusionLayoutAssignmentResult> RunDotOutputFusion(
    HloModule* module, const std::string& test_name, int m, int k, int n,
    const int64_t dot_operand_idx_in_add) {
  DotOutputFusionLayoutAssignmentResult result;

  CHECK(dot_operand_idx_in_add == 0 || dot_operand_idx_in_add == 1);

  auto builder = HloComputation::Builder(test_name);

  Shape dot_lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape dot_rhs_shape = ShapeUtil::MakeShape(F32, {k, n});
  Shape dot_shape = ShapeUtil::MakeShape(F32, {m, n});
  if (m == 1) {
    dot_lhs_shape = ShapeUtil::MakeShape(F32, {k});
    dot_shape = ShapeUtil::MakeShape(F32, {n});
  } else if (n == 1) {
    dot_rhs_shape = ShapeUtil::MakeShape(F32, {k});
    dot_shape = ShapeUtil::MakeShape(F32, {m});
  }

  HloInstruction* dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dot_lhs_shape, "param0"));
  HloInstruction* addend = builder.AddInstruction(
      HloInstruction::CreateParameter(1, dot_shape, "param1"));
  HloInstruction* dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(dot_rhs_shape)));
  HloInstruction* dot_result =
      builder.AddInstruction(CreateCanonicalDot(dot_shape, dot_lhs, dot_rhs));
  HloInstruction* add_result;
  if (dot_operand_idx_in_add == 0) {
    add_result = builder.AddInstruction(HloInstruction::CreateBinary(
        dot_shape, HloOpcode::kAdd, dot_result, addend));
  } else {
    add_result = builder.AddInstruction(HloInstruction::CreateBinary(
        dot_shape, HloOpcode::kAdd, addend, dot_result));
  }

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloInstruction* fusion_instruction =
      module->entry_computation()->AddInstruction(HloInstruction::CreateFusion(
          dot_shape, HloInstruction::FusionKind::kOutput, add_result));
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(add_result, fusion_instruction));

  HloInstruction* fused_add =
      fusion_instruction->fused_instructions_computation()->root_instruction();
  HloInstruction* fused_dot = fusion_instruction->FuseInstruction(dot_result);

  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(dot_result));

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_lhs_shape));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_shape));

  result.dot_lhs_fusion_param =
      fusion_instruction->operand(fused_dot->operand(0)->parameter_number());
  result.dot_rhs_fusion_param =
      fusion_instruction->operand(fused_dot->operand(1)->parameter_number());
  result.addend_fusion_param = fusion_instruction->operand(
      fused_add->operand(1 - dot_operand_idx_in_add)->parameter_number());

  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  cpu::CpuLayoutAssignment layout_assignment(&computation_layout,
                                             &target_machine_features);
  TF_ASSIGN_OR_RETURN(result.layout_assignment_changed_something,
                      layout_assignment.Run(module));

  return result;
}

static void AssertCorrectLayoutForDotOutputFusion(
    const HloComputation* computation,
    const DotOutputFusionLayoutAssignmentResult& layout_assignment_result,
    bool expect_col_major_dot_rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_layout_assignment_testDTcc mht_1(mht_1_v, 514, "", "./tensorflow/compiler/xla/service/cpu/cpu_layout_assignment_test.cc", "AssertCorrectLayoutForDotOutputFusion");

  Layout expected_dot_rhs_layout = expect_col_major_dot_rhs
                                       ? LayoutUtil::MakeLayout({0, 1})
                                       : LayoutUtil::MakeLayout({1, 0});
  if (layout_assignment_result.dot_rhs_fusion_param->shape().rank() == 1) {
    expected_dot_rhs_layout = LayoutUtil::MakeLayout({0});
  }
  EXPECT_TRUE(LayoutUtil::Equal(
      expected_dot_rhs_layout,
      layout_assignment_result.dot_rhs_fusion_param->shape().layout()));

  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::MakeDescendingLayout(
          layout_assignment_result.dot_lhs_fusion_param->shape().rank()),
      layout_assignment_result.dot_lhs_fusion_param->shape().layout()));

  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::MakeDescendingLayout(
          layout_assignment_result.addend_fusion_param->shape().rank()),
      layout_assignment_result.addend_fusion_param->shape().layout()));
  EXPECT_THAT(computation->instructions(), Each(Not(op::Copy())));
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_1x50x19_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/1, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/true);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_1x50x19_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/1, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/true);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x1_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/1,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x1_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/1,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x19_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x19_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  TF_ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, BatchDotLayoutMustBeRowMajor) {
  const char* hlo_string = R"(
HloModule BatchDotLayoutMustBeRowMajor

ENTRY BatchDotLayoutMustBeRowMajor {
  p0 = f32[10,1,10] parameter(0)
  p1 = f32[10,10,1] parameter(1)
  ROOT dot = f32[10,1,1] dot(p0, p1), lhs_batch_dims={0},
                                      lhs_contracting_dims={2},
                                      rhs_batch_dims={0},
                                      rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* computation = module->entry_computation();

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithLayout(F32, {10, 1, 10}, {2, 1, 0}));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(ShapeUtil::MakeShapeWithLayout(F32, {10, 10, 1}, {2, 1, 0}));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeShapeWithLayout(F32, {10, 1, 1}, {1, 2, 0}));
  AssignLayouts(module.get(), &computation_layout);

  Shape expected_shape =
      ShapeUtil::MakeShapeWithLayout(F32, {10, 1, 1}, {2, 1, 0});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::ShapeWithLayout(expected_shape)));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::Dot(
          op::ShapeWithLayout(computation_layout.parameter_layout(0).shape()),
          op::ShapeWithLayout(
              computation_layout.parameter_layout(1).shape()))));
}
}  // namespace
}  // namespace xla
