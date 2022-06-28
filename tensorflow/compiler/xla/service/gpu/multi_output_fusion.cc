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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"

#include <stdint.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

bool IsProfitableOperand(HloInstruction* instr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "IsProfitableOperand");

  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  return true;
}

FusionDecision LegalToFuse(HloInstruction* instr1, HloInstruction* instr2,
                           FusionInfoCache* fusion_info_cache) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "LegalToFuse");

  // If we're fusing fusions only do it if the fusion kind matches. Loop fusions
  // merge into bigger loop fusions and input (reduce) fusions become fusions
  // with multiple reduce outputs. We could fuse reduce and loop fusions
  // together too (the result being an input fusion) if we find cases where this
  // improves things. Also disable fusing standalone input-fusible reduces into
  // loop fusions.
  CHECK(instr1->opcode() == HloOpcode::kFusion);
  if ((instr2->opcode() == HloOpcode::kFusion &&
       instr1->fusion_kind() != instr2->fusion_kind()) ||
      (IsReductionFromOrToContiguousDimensions(*instr2) &&
       instr1->IsLoopFusion())) {
    return "Can't merge fusions of two different types";
  }
  // The emitter only supports in-place DUS for fusions with a single DUS at the
  // root. Don't sibling fuse DUS for now.
  // TODO(b/119178699): Multi-output fusing DUS can improve performance if we
  // share the input and output buffers and add support to the emitter.
  if (instr1->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice ||
      (instr2->opcode() == HloOpcode::kFusion &&
       instr2->fused_expression_root()->opcode() ==
           HloOpcode::kDynamicUpdateSlice)) {
    return "Can't fuse multiple DUSs";
  }

  // Do this check last, as it may be expensive.
  return FusionFitsInBudget(*instr1, *instr2,
                            /*is_consumer_producer_fusion=*/false,
                            fusion_info_cache);
}

// We prefer multi-output fusions over other fusions over unfused ops, because
// we want to preserve fusion opportunities if possible.
int FusionPriority(const HloInstruction* instr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "FusionPriority");

  if (instr->IsMultiOutputFusion()) {
    return 2;
  }
  if (instr->opcode() == HloOpcode::kFusion) {
    return 1;
  }
  return 0;
}

HloInstruction* SelectPreferredFusionCandidate(
    const std::vector<HloInstruction*> candidates) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "SelectPreferredFusionCandidate");

  if (candidates.empty()) {
    return nullptr;
  }
  return *std::max_element(
      candidates.begin(), candidates.end(),
      [](const HloInstruction* a, const HloInstruction* b) {
        return FusionPriority(a) < FusionPriority(b);
      });
}

std::vector<HloInstruction*> GetProducerConsumerMultiOutputFusionCandidates(
    const HloInstruction* producer, const HloReachabilityMap& reachability,
    FusionInfoCache* fusion_info_cache) {
  std::vector<HloInstruction*> fusion_candidates;
  // If there is only one user, and it is not a multi-output fusion node, this
  // fusion possibility was already considered and rejected by the FusionMerger
  // pass. No need to try again!
  if (producer->user_count() == 1 &&
      !producer->users()[0]->IsMultiOutputFusion()) {
    return fusion_candidates;
  }
  for (HloInstruction* consumer : producer->users()) {
    VLOG(3) << "Looking at producer " << producer->name()
            << " and its consumer " << consumer->name();
    if (!IsFusibleAsMultiOutputFusionRoot(*consumer)) {
      VLOG(3) << "Consumer " << consumer->name()
              << " is not eligible as multi-output fusion root.";
      continue;
    }
    if (!IsProducerConsumerMultiOutputFusible(*producer, *consumer)) {
      VLOG(3) << producer->name() << " and " << consumer->name()
              << " are not fusible.";
      continue;
    }
    // Do not fuse a producer if the other operands of the fusion are
    // reachable from the producer, this would create a cycle.
    auto operand_reachable_from_producer = [&](const HloInstruction* operand) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_4(mht_4_v, 318, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "lambda");

      // If a get-tuple-element instruction is not in the reachability
      // map, it has been created by fusion in this pass. Simply move
      // on to its operand, which is in the reachability map.
      if (!reachability.IsPresent(operand) &&
          operand->opcode() == HloOpcode::kGetTupleElement) {
        operand = operand->operand(0);
      }
      CHECK(reachability.IsPresent(operand) && reachability.IsPresent(producer))
          << "Reachability map is incomplete. This should never "
             "happen.";
      return producer != operand && reachability.IsReachable(producer, operand);
    };
    if (absl::c_any_of(consumer->operands(), operand_reachable_from_producer)) {
      VLOG(3) << producer->name() << " would introduce a cycle when fused.";
      continue;
    }
    if (!FusionFitsInBudget(*producer, *consumer,
                            /*is_consumer_producer_fusion=*/false,
                            fusion_info_cache)) {
      VLOG(3) << producer->name() << " and " << consumer->name()
              << " would be too large of a fusion.";
      continue;
    }
    // Make sure the emitter can codegen the fusion op efficiently. We currently
    // can have exponential time/memory requirements for emitting certain fusion
    // ops, in which case we don't want to fuse.
    // TODO(b/119692968): Remove this once fixed in the emitter.
    if (FusedIrEmitter::IsFusedIrEmitterInefficient(*consumer, *producer)) {
      VLOG(3) << "Fusion of " << producer->name() << " into "
              << consumer->name()
              << " would result in overly large code duplication.";
      continue;
    }
    fusion_candidates.push_back(consumer);
  }
  return fusion_candidates;
}

bool IsSiblingFusionCandidate(const HloInstruction* instr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_5(mht_5_v, 360, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "IsSiblingFusionCandidate");

  if (instr->IsDead()) {
    return false;
  }
  if (!IsFusibleAsMultiOutputFusionRoot(*instr)) {
    return false;
  }
  // Check if the users of multioutput fusion is not a get-tuple-element.
  // If this is the case, we bail out because the transformation assumes
  // the users are get-tuple-element.
  if (instr->IsMultiOutputFusion()) {
    for (auto user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

void GpuMultiOutputFusion::RecomputeReachability() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_6(mht_6_v, 385, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "GpuMultiOutputFusion::RecomputeReachability");

  reachability_ = HloReachabilityMap::Build(computation_);
}

bool GpuMultiOutputFusion::FuseSiblings(HloInstruction* parent,
                                        FusionInfoCache* fusion_info_cache) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_7(mht_7_v, 393, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "GpuMultiOutputFusion::FuseSiblings");

  if (!IsProfitableOperand(parent)) {
    VLOG(3) << "Operand " << parent->ToShortString() << " is not profitable";
    return false;
  }
  bool changed = false;
  std::vector<HloInstruction*> siblings = parent->users();
  // Sort the siblings such that multi-output fusion ops occur first, followed
  // by fusion ops, followed by unfused ops.
  absl::c_stable_sort(siblings,
                      [](const HloInstruction* a, const HloInstruction* b) {
                        return FusionPriority(a) > FusionPriority(b);
                      });
  for (auto i = siblings.begin(); i != siblings.end(); ++i) {
    VLOG(3) << "Considering " << (*i)->name();
    if ((*i)->opcode() != HloOpcode::kFusion || !IsSiblingFusionCandidate(*i)) {
      continue;
    }
    for (auto j = i + 1; j != siblings.end();) {
      VLOG(3) << "Considering " << (*i)->name() << " and " << (*j)->name();
      if (!IsSiblingFusionCandidate(*j) || reachability_->IsConnected(*i, *j) ||
          !ShapesCompatibleForMultiOutputFusion(*(*i), *(*j)) ||
          !LegalToFuse(*i, *j, fusion_info_cache)) {
        ++j;
        continue;
      }
      if (!ConsumeFuel(name(), [&] {
            return absl::StrFormat("Not fusing siblings %s and %s.",
                                   (*i)->name(), (*j)->name());
          })) {
        ++j;
        continue;
      }
      VLOG(2) << "Fuse siblings " << (*i)->name() << " and " << (*j)->name();
      fusion_info_cache->Invalidate(*i);
      fusion_info_cache->Invalidate(*j);
      HloInstruction* remaining = *i;
      HloInstruction* fused = *j;

      DumpFusionState(*remaining,
                      absl::StrCat("About to fuse producer |", fused->name(),
                                   "| into consumer |", remaining->name(),
                                   "| inside GPU multi-output fusion"),
                      /*producer=*/fused);

      if (fused->opcode() == HloOpcode::kFusion) {
        remaining->MergeFusionInstructionIntoMultiOutput(fused);
      } else {
        remaining->FuseInstructionIntoMultiOutput(fused);
        CHECK_EQ(0, fused->user_count());
        TF_CHECK_OK(computation_->RemoveInstruction(fused));
      }
      DumpFusionState(*remaining,
                      absl::StrCat("Fused into consumer |", remaining->name(),
                                   "| inside GPU multi-output fusion"));
      changed = true;
      siblings.erase(j);
      RecomputeReachability();
    }
  }
  return changed;
}

StatusOr<bool> GpuMultiOutputFusion::DoMultiOutputFusion() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_8(mht_8_v, 459, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "GpuMultiOutputFusion::DoMultiOutputFusion");

  bool changed = false;
  RecomputeReachability();
  std::vector<HloInstruction*> defs_before_uses =
      computation_->MakeInstructionPostOrder();

  FusionInfoCache fusion_info_cache;
  while (!defs_before_uses.empty()) {
    // Traverse the HLO in uses-before-defs order by removing instruction from
    // the back of the vector.
    HloInstruction* producer = defs_before_uses.back();

    // Copy on purpose: to use after removing the producer.
    std::string producer_name = producer->name();
    defs_before_uses.pop_back();
    // Never multi-output fuse constants.  To the extent that we want to fuse
    // constants, that should be handled by the regular fusion pass.
    if (producer->opcode() == HloOpcode::kConstant) {
      VLOG(3) << producer->name() << " is a constant.";
      continue;
    }
    // First, fuse the consumer ops of the current op, which are siblings.
    if (FuseSiblings(/*parent=*/producer, &fusion_info_cache)) {
      changed = true;
    }
    // Second, perform producer-consumer multi-output fusion. This order will
    // ensure that all get-tuple-element ops inserted as a by-product of
    // multi-output fusion will occur before the current op in the order of
    // traversal, and hence, not get into the way of subsequent fusion attempts.
    const auto candidates = GetProducerConsumerMultiOutputFusionCandidates(
        producer, *reachability_, &fusion_info_cache);
    auto* consumer_for_fusion = SelectPreferredFusionCandidate(candidates);
    if (consumer_for_fusion == nullptr) {
      continue;
    }
    if (!ConsumeFuel(name(), [&] {
          return absl::StrFormat("Not fusing %s and %s.", producer->name(),
                                 consumer_for_fusion->name());
        })) {
      continue;
    }
    changed = true;
    fusion_info_cache.Invalidate(producer);
    fusion_info_cache.Invalidate(consumer_for_fusion);

    if (consumer_for_fusion->opcode() == HloOpcode::kFusion) {
      VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
              << consumer_for_fusion->name();
      DumpFusionState(
          *consumer_for_fusion,
          absl::StrCat("About to fuse producer |", producer_name,
                       "| into consumer |", consumer_for_fusion->name(),
                       "| inside GPU multi-output fusion"),
          /*producer=*/producer);
      if (producer->opcode() == HloOpcode::kFusion) {
        consumer_for_fusion->MergeFusionInstructionIntoMultiOutput(producer);
      } else {
        consumer_for_fusion->FuseInstructionIntoMultiOutput(producer);
        CHECK_EQ(0, producer->user_count());
        TF_CHECK_OK(computation_->RemoveInstruction(producer));
      }

      DumpFusionState(
          *consumer_for_fusion,
          absl::StrCat("Fusing producer |", producer_name, "| into consumer |",
                       consumer_for_fusion->name(),
                       "| inside GPU multi-output fusion"));
      RecomputeReachability();
      continue;
    }
    HloInstruction* input_fusion =
        computation_->AddInstruction(HloInstruction::CreateFusion(
            consumer_for_fusion->shape(),
            ChooseFusionKind(*producer, *consumer_for_fusion),
            consumer_for_fusion));
    VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
            << consumer_for_fusion->name() << " into " << input_fusion->name();
    DumpFusionState(
        *input_fusion,
        absl::StrCat("About to fuse |", producer_name, "| into consumer |",
                     input_fusion->name(), "| inside GPU multi-output fusion"),
        /*producer=*/input_fusion);
    TF_CHECK_OK(
        computation_->ReplaceInstruction(consumer_for_fusion, input_fusion));
    if (producer->opcode() == HloOpcode::kFusion) {
      input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      input_fusion->FuseInstructionIntoMultiOutput(producer);
      CHECK_EQ(0, producer->user_count());
      TF_CHECK_OK(computation_->RemoveInstruction(producer));
    }

    DumpFusionState(
        *input_fusion,
        absl::StrCat("Fusing producer |", producer_name, "| into consumer |",
                     input_fusion->name(), "| inside GPU multi-output fusion"));
    RecomputeReachability();
  }
  return changed;
}

void GpuMultiOutputFusion::DumpFusionState(const HloInstruction& consumer,
                                           absl::string_view label,
                                           const HloInstruction* producer) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("label: \"" + std::string(label.data(), label.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_9(mht_9_v, 566, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "GpuMultiOutputFusion::DumpFusionState");

  if (consumer.GetModule()
          ->config()
          .debug_options()
          .xla_dump_fusion_visualization()) {
    RegisterFusionState(*computation_, label, consumer, producer);
  }
}

StatusOr<bool> GpuMultiOutputFusion::Run(HloModule* module) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSmulti_output_fusionDTcc mht_10(mht_10_v, 578, "", "./tensorflow/compiler/xla/service/gpu/multi_output_fusion.cc", "GpuMultiOutputFusion::Run");

  bool changed = false;
  for (auto* computation : module->MakeNonfusionComputations()) {
    computation_ = computation;
    TF_ASSIGN_OR_RETURN(bool fusion_changed, DoMultiOutputFusion());
    if (fusion_changed) {
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
