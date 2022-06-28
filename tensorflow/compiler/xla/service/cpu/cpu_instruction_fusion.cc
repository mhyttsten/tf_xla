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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"

#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

namespace xla {
namespace cpu {

namespace {

bool CanBeLoopFused(const HloInstruction& hlo) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CanBeLoopFused");

  // These are the only ones we fuse since we rely on effective elemental IR
  // generation.
  return hlo.IsElementwise() ||  //
         hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kGather ||
         hlo.opcode() == HloOpcode::kIota || hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kReverse ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}

bool IsNonComplexNonBatchedMatrixVectorDot(const HloInstruction* hlo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "IsNonComplexNonBatchedMatrixVectorDot");

  const Shape& hlo_shape = hlo->shape();
  return !ShapeUtil::ElementIsComplex(hlo_shape) &&
         hlo->opcode() == HloOpcode::kDot && hlo_shape.dimensions_size() <= 1 &&
         hlo->dot_dimension_numbers().lhs_batch_dimensions_size() == 0;
}

bool HasExactlyOneUse(const HloInstruction& hlo_instr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "HasExactlyOneUse");

  return hlo_instr.user_count() == 1 &&
         absl::c_count(hlo_instr.users().front()->operands(), &hlo_instr) == 1;
}

bool CanBeOutputFused(const HloInstruction* producer,
                      const HloInstruction* consumer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CanBeOutputFused");

  return consumer->opcode() == HloOpcode::kAdd &&
         IsNonComplexNonBatchedMatrixVectorDot(producer) &&
         HasExactlyOneUse(*producer) == 1;
}

bool CanBeOutputFusedIntoSomeOperand(const HloInstruction* consumer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CanBeOutputFusedIntoSomeOperand");

  return consumer->opcode() == HloOpcode::kAdd &&
         (CanBeOutputFused(consumer->operand(0), consumer) ||
          CanBeOutputFused(consumer->operand(1), consumer));
}
}  // namespace

FusionDecision CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_5(mht_5_v, 256, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CpuInstructionFusion::ShouldFuse");

  HloInstruction* producer = consumer->mutable_operand(operand_index);
  VLOG(2) << "Considering for fusion: operand " << operand_index << " of "
          << consumer->ToString();

  constexpr int kFusionThresholdBytes = 16 * 1024;

  if (CanBeOutputFused(producer, consumer)) {
    VLOG(2) << "Fusion OK: Can create output fusion.";
    return {};
  }

  if (CanBeOutputFusedIntoSomeOperand(producer)) {
    return "Bailing because producer can be output-fused into some operand.";
  }

  if (!CanBeLoopFused(*producer)) {
    return "Producer is not loop-fusible.";
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return "Fusion is not profitable.";
  }

  if (NoFusionPossible should_fuse =
          !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    return !should_fuse;
  }

  // Fuse constants in general but avoid creating 2-instruction fusions with
  // just a constant and another node.
  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() != HloOpcode::kFusion) {
    return "Not fusing: insufficient non-constant nodes.";
  }

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return "Not fusing: producer is itself a fusion node.";
  }

  // Don't fuse if fusing would cause too much code duplication because of
  // inefficiencies in the fusion emitter.
  // TODO(b/119692968): Remove this once the fusion emitter can handle
  // arbitrary fusion nodes.
  if (consumer->opcode() == HloOpcode::kFusion) {
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      // We have no cached results for this fusion node yet. This can happen
      // when we run the InstructionFusion pass more than once. We can only
      // cache the results within one run.
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(
            producer)) {
      return "Code duplication too high";
    }
  }

  if (consumer->opcode() == HloOpcode::kDot) {
    // In the general case we call out to optimized "black box" GEMM routines
    // for Dot, which precludes fusion.  However, in very specific cases, we try
    // to fuse Dot operations by generating an elemental dot implementation.
    //
    // We need to be careful and conservative here since any benefit we get from
    // fusion can easily be overshadowed by the overhead of a naive GEMM
    // algorithm in the IR.
    const Shape& output_shape = consumer->shape();
    if (output_shape.dimensions_size() <= 1) {
      // We fuse in cases where we have a matrix*vector or vector*matrix dot and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (consumer->operand(0)->shape().rank() == 1 && operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return {};
      } else if (consumer->operand(1)->shape().rank() == 1 &&
                 operand_index == 0 &&
                 ShapeUtil::ByteSizeOfElements(consumer->operand(1)->shape()) <
                     kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return {};
      }
    }
  }

  // Don't fuse reductions over the major dimensions. These have an efficient
  // lowering that's only implemented for the unfused case.
  if (consumer->opcode() == HloOpcode::kReduce &&
      !absl::c_linear_search(
          consumer->dimensions(),
          LayoutUtil::Minor(consumer->operand(0)->shape().layout(), 0))) {
    return "Not fusing reductions over major dimensions";
  }
  if (producer->opcode() == HloOpcode::kReduce &&
      !absl::c_linear_search(
          producer->dimensions(),
          LayoutUtil::Minor(producer->operand(0)->shape().layout(), 0))) {
    return "Not fusing reductions over major dimensions";
  }

  if (consumer->IsLoopFusion()) {
    VLOG(2) << "Fusing: consumer is a fusion node.";
    return {};
  }

  if (CanBeLoopFused(*consumer)) {
    VLOG(2) << "Fusing: consumer is elementwise or fusible.";
    return {};
  }

  return "Not fusing: not found a fusible case";
}

HloInstruction::FusionKind CpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CpuInstructionFusion::ChooseKind");

  return CanBeOutputFused(producer, consumer)
             ? HloInstruction::FusionKind::kOutput
             : HloInstruction::FusionKind::kLoop;
}

HloInstruction* CpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_instruction_fusionDTcc mht_7(mht_7_v, 391, "", "./tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.cc", "CpuInstructionFusion::FuseInstruction");

  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
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
}  // namespace cpu
}  // namespace xla
