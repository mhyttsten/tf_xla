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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

using FusionNodeIndexingEvaluationTest = HloTestBase;

// Subclass of InstructionFusion exposing the protected methods Fuse and
// FuseInstruction for testing. Also adds the FusionNodeIndexingEvaluation to
// track the code duplication due to indexing HloInstructions with
// different index values.
class InstructionFusionForTesting : public InstructionFusion {
 public:
  explicit InstructionFusionForTesting()
      : InstructionFusion(InstructionFusion::IsExpensive) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation_test.cc", "InstructionFusionForTesting");
}

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation_test.cc", "FuseInstruction");

    auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
    if (evaluation == fusion_node_evaluations_.end()) {
      evaluation =
          fusion_node_evaluations_
              .emplace(fusion_instruction,
                       FusionNodeIndexingEvaluation(fusion_instruction))
              .first;
    }
    auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
    HloInstruction* new_producer =
        InstructionFusion::FuseInstruction(fusion_instruction, producer);
    evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
    return new_producer;
  }

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       HloComputation* computation) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation_test.cc", "Fuse");

    return InstructionFusion::Fuse(producer, consumer, computation);
  }

  int64_t EvaluateEmittedInstructions(const HloInstruction* producer,
                                      const HloInstruction* consumer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluation_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation_test.cc", "EvaluateEmittedInstructions");

    if (consumer->opcode() != HloOpcode::kFusion) {
      return 0;
    }
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    return fusion_node_evaluations_.at(consumer).EvaluateEmittedInstructions(
        producer);
  }

 private:
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;
};

TEST_F(FusionNodeIndexingEvaluationTest, FuseTwoInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    ROOT sub = f32[4,3]{1,0} subtract(add, p0)
  })")
                    .ValueOrDie();
  HloInstruction* sub = module->entry_computation()->root_instruction();
  HloInstruction* add = sub->mutable_operand(0);
  InstructionFusionForTesting().Fuse(add, sub, module->entry_computation());
}

TEST_F(FusionNodeIndexingEvaluationTest, FuseThreeInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4]{0} parameter(0)
    slice1 = f32[3]{0} slice(p0), slice={[0:3]}
    slice2 = f32[3]{0} slice(p0), slice={[0:3]}
    ROOT sub = f32[3]{0} subtract(slice1, slice2)
  })")
                    .ValueOrDie();
  HloInstruction* sub = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion;
  HloInstruction* slice1 = sub->mutable_operand(0);
  HloInstruction* slice2 = sub->mutable_operand(1);
  auto fusion =
      instruction_fusion.Fuse(slice1, sub, module->entry_computation());
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice2, fusion), 1);
  instruction_fusion.Fuse(slice2, fusion, module->entry_computation());
}

TEST_F(FusionNodeIndexingEvaluationTest, ExponentialDuplicationPattern) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4]{0} parameter(0)
    p1 = f32[4]{0} parameter(1)
    add0 = f32[4]{0} add(p0, p1)
    slice1.0 = f32[3]{0} slice(add0), slice={[0:3]}
    slice1.1 = f32[3]{0} slice(add0), slice={[1:4]}
    add1 = f32[3]{0} add(slice1.0, slice1.1)
    slice2.0 = f32[2]{0} slice(add1), slice={[0:2]}
    slice2.1 = f32[2]{0} slice(add1), slice={[1:3]}
    ROOT add2 = f32[2]{0} add(slice2.0, slice2.1)
  })")
                    .ValueOrDie();
  // This corresponds to the following graph:
  //              add0
  //            /      \
  //       slice1.0  slice1.1
  //            \      /
  //              add1
  //            /      \
  //       slice2.0  slice2.1
  //            \      /
  //              add2
  // This pattern can be arbitrarily extended. In this example, add2, slice2.0,
  // slice2.1 each get emitted once because they can be indexed with the same
  // index vector. Since add1 has a different shape than its two users, it
  // needs to be emitted twice. slice1.0 and slice1.1 each also get emitted
  // twice because they get passed both different index vectors from add1. add0
  // then gets emitted 4 times.
  HloInstruction* add2 = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion;
  HloInstruction* slice2_0 = add2->mutable_operand(0);
  HloInstruction* slice2_1 = add2->mutable_operand(1);
  auto fusion =
      instruction_fusion.Fuse(slice2_0, add2, module->entry_computation());
  // So far we have fused add2 and slice2.0. So when we also fuse slice2.1, we
  // expect to emit it 1 time.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice2_1, fusion),
            1);
  instruction_fusion.Fuse(slice2_1, fusion, module->entry_computation());
  HloInstruction* add1 = fusion->mutable_operand(0);
  EXPECT_EQ(add1->opcode(), HloOpcode::kAdd);
  // If we fuse add1 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add1, fusion), 2);
  instruction_fusion.Fuse(add1, fusion, module->entry_computation());
  HloInstruction* slice1_0 = fusion->mutable_operand(0);
  EXPECT_EQ(slice1_0->opcode(), HloOpcode::kSlice);
  // If we fuse slice1.0 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice1_0, fusion),
            2);
  instruction_fusion.Fuse(slice1_0, fusion, module->entry_computation());
  HloInstruction* slice1_1 = fusion->mutable_operand(0);
  EXPECT_EQ(slice1_1->opcode(), HloOpcode::kSlice);
  // If we fuse slice1.1 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice1_1, fusion),
            2);
  instruction_fusion.Fuse(slice1_1, fusion, module->entry_computation());
  HloInstruction* add0 = fusion->mutable_operand(0);
  EXPECT_EQ(add0->opcode(), HloOpcode::kAdd);
  // If we fuse add0 into 'fusion', it needs to be emitted four times.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add0, fusion), 4);
  instruction_fusion.Fuse(add0, fusion, module->entry_computation());
}

TEST_F(FusionNodeIndexingEvaluationTest, RecomputeCache) {
  // This is the same HloModule as in ExponentialDuplicationPattern above, but
  // starting with the fusion node as it is before 'add0' is fused in.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module
%fused_computation (param_0.5: f32[4]) -> f32[2] {
  %param_0.5 = f32[4]{0} parameter(0)
  %slice1.2 = f32[3]{0} slice(f32[4]{0} %param_0.5), slice={[0:3]}
  %slice1.3 = f32[3]{0} slice(f32[4]{0} %param_0.5), slice={[1:4]}
  %add1.1 = f32[3]{0} add(f32[3]{0} %slice1.2, f32[3]{0} %slice1.3)
  %slice2.2 = f32[2]{0} slice(f32[3]{0} %add1.1), slice={[0:2]}
  %slice2.3 = f32[2]{0} slice(f32[3]{0} %add1.1), slice={[1:3]}
  ROOT %add2.1 = f32[2]{0} add(f32[2]{0} %slice2.2, f32[2]{0} %slice2.3)
}

ENTRY entry_computation {
  p0 = f32[4]{0} parameter(0)
  p1 = f32[4]{0} parameter(1)
  add0 = f32[4]{0} add(p0, p1)
  ROOT %fusion = f32[2]{0} fusion(add0), kind=kLoop, calls=%fused_computation
})")
                    .ValueOrDie();
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion;
  HloInstruction* add0 = fusion->mutable_operand(0);
  EXPECT_EQ(add0->opcode(), HloOpcode::kAdd);
  // Here, the cache for the fusion node needs to be recomputed. Make sure we
  // still get the same evaluation as before when we incrementally build the
  // cache.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add0, fusion), 4);
}

}  // namespace xla
