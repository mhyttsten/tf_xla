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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc() {
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

#include "tensorflow/compiler/xla/service/multi_output_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

StatusOr<bool> MultiOutputFusion::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::Run");

  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    computation_ = computation;
    candidates_.clear();
    candidates_index_.clear();
    all_fusion_candidates_.clear();
    RecomputeReachability();

    int64_t index = 0;
    for (auto it : computation_->MakeInstructionPostOrder()) {
      candidates_.emplace_back(it);
      InsertOrDie(&candidates_index_, it, index++);
    }

    // Create the initial candidate list for each Node.
    for (auto& node : candidates_) {
      HloInstruction* instruction = node.hlo;
      int64_t instruction_id = get_candidate_id(instruction);
      FusionCandidate& instr_node = candidates_[instruction_id];
      if (!IsFusible(instruction)) {
        continue;
      }
      all_fusion_candidates_.emplace_back(instruction,
                                          reachability_->GetIndex(instruction));

      std::vector<HloInstruction*> candidates;
      absl::flat_hash_set<HloInstruction*> candidates_set;
      VLOG(10) << "Looking at instruction: " << instruction->name();
      for (auto operand : instruction->operands()) {
        // Filter out the non-interesting instructions -- they
        // will not generate the savings.
        if (!IsProfitableOperand(operand)) {
          VLOG(10) << "Operand not profitable: " << operand->name();
          continue;
        }
        VLOG(10) << "Operand profitable: " << operand->name();
        // We don't look at all users of operands as it's quadratic. Only look
        // at one slice of users.
        const int64_t kUserSliceSize = 128;

        const int64_t user_slice_begin =
            RoundDownTo(operand->UserId(instruction), kUserSliceSize);

        const int64_t user_slice_end =
            std::min(static_cast<int64_t>(operand->users().size()),
                     user_slice_begin + kUserSliceSize);

        for (int64_t i = user_slice_begin; i < user_slice_end; ++i) {
          HloInstruction* user = operand->users()[i];
          VLOG(10) << "User: " << user->name();
          if (user == instruction || !IsFusible(user)) {
            VLOG(10) << "User is not fusible, or is the instruction itself: "
                     << user->name();
            continue;
          }
          int64_t user_id = get_candidate_id(user);
          if (is_connected(instruction, user)) {
            VLOG(10) << "User is connected: " << user->name();
            continue;
          }
          if (instruction_id < user_id &&
              user->opcode() == HloOpcode::kFusion) {
            VLOG(10) << "User ID for user: " << user->name() << " is "
                     << user_id << " which is higher than " << instruction_id;
            continue;
          }
          if (!LegalToFuse(instruction, user)) {
            VLOG(10) << "User not legal to fuse: " << user->name();
            continue;
          }
          if (candidates_set.insert(user).second) {
            VLOG(10) << "User added to candidate list: " << user->name();
            candidates.push_back(user);
          }
        }
      }

      // Iterate over candidates rather than candidates_set to avoid
      // nondeterminism.
      for (auto candidate : candidates) {
        int64_t profit = GetProfit(instruction, candidate);
        if (profit > 0) {
          FusionCandidate& candidate_node =
              candidates_[get_candidate_id(candidate)];
          instr_node.fusibles.emplace_back(candidate, profit);
          candidate_node.fusibles.emplace_back(instruction, profit);
          worklist_.emplace(instruction, candidate, profit);
        }
      }
    }
    if (Perform()) {
      changed = true;
    }
  }
  // Clean up state in case this pass is wrapped in an HloPassPipeline.
  candidates_.clear();
  candidates_index_.clear();
  all_fusion_candidates_.clear();
  reachability_.reset();
  if (changed) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  return changed;
}

HloInstruction* MultiOutputFusion::Fuse(HloInstruction* instr1,
                                        HloInstruction* instr2) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_1(mht_1_v, 311, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::Fuse");

  HloInstruction* remaining = instr1;
  HloInstruction* fused = instr2;
  // Make sure that if only one of the instructions is a fusion, or if only one
  // of the instructions is a multi-output fusion, it's what will be fused into.
  if (fused->opcode() == HloOpcode::kFusion) {
    std::swap(remaining, fused);
  }
  if (fused->IsMultiOutputFusion()) {
    std::swap(remaining, fused);
  }
  if (fused->opcode() == HloOpcode::kFusion) {
    remaining->MergeFusionInstructionIntoMultiOutput(fused);
  } else {
    remaining->FuseInstructionIntoMultiOutput(fused);
    CHECK_EQ(0, fused->user_count());
    TF_CHECK_OK(computation()->RemoveInstruction(fused));
  }
  return remaining;
}

HloInstruction* MultiOutputFusion::CreateFusion(HloInstruction* base,
                                                HloInstruction* to_fuse) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_2(mht_2_v, 336, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::CreateFusion");

  HloInstruction* input_fusion =
      computation()->AddInstruction(HloInstruction::CreateFusion(
          base->shape(), HloInstruction::FusionKind::kLoop, base));

  // Update candidate_ and all_fusion_candidates_.
  int64_t index = candidates_.size();
  InsertOrDie(&candidates_index_, input_fusion, index);
  candidates_.emplace_back(input_fusion);
  reachability_->Replace(base, input_fusion);
  all_fusion_candidates_.emplace_back(input_fusion,
                                      reachability_->GetIndex(input_fusion));
  TF_CHECK_OK(computation()->ReplaceInstruction(base, input_fusion));
  return input_fusion;
}

bool MultiOutputFusion::IsProfitableOperand(HloInstruction* instr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_3(mht_3_v, 355, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::IsProfitableOperand");

  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  // We don't target to fuse producer/consumer instructions -- this should
  // be taken care of by the instruction_fusion pass. If instr has only
  // one user, it will not have sibling instructions. We won't consider it.
  if (instr->user_count() < 2) {
    return false;
  }
  return true;
}

std::vector<std::pair<HloInstruction*, int64_t>>
MultiOutputFusion::GetNewFusibles(HloInstruction* instr1,
                                  HloInstruction* instr2) {
  HloInstruction* fusion = instr1;
  HloInstruction* fused = instr2;
  if (is_fused(instr1)) {
    fusion = instr2;
    fused = instr1;
  }

  FusionCandidate& fusion_node = candidates_[get_candidate_id(fusion)];
  FusionCandidate& fused_node = candidates_[get_candidate_id(fused)];

  // The second entry of the pair is an old profit value.
  std::vector<std::pair<HloInstruction*, int64_t>> new_fusibles;
  absl::flat_hash_set<HloInstruction*> in_list;
  auto it = fusion_node.fusibles.begin();
  while (it != fusion_node.fusibles.end()) {
    HloInstruction* instr = it->first;
    if (is_fused(instr) || is_connected(fusion, instr)) {
      it = fusion_node.fusibles.erase(it);
      continue;
    }
    in_list.insert(instr);
    new_fusibles.emplace_back(instr, it->second);
    ++it;
  }

  // Fused_node has been fused into fusion_node. Take the fusion candidates
  // (fusibles) from fused_nodes and add them to the fusion_node's. Filter
  // out those fusibles that no longer valid (or already in the list).
  for (const auto& it : fused_node.fusibles) {
    HloInstruction* instr = it.first;
    if (instr == fusion || is_fused(instr) || is_connected(fusion, instr)) {
      continue;
    }
    if (in_list.contains(instr)) {
      continue;
    }
    // Set old profit to zero because instr is not originally fusible to
    // fusion_node.
    new_fusibles.emplace_back(instr, 0);
  }
  fused_node.fusibles.clear();

  return new_fusibles;
}

void MultiOutputFusion::UpdateBeforeFuse(HloInstruction* instr1,
                                         HloInstruction* instr2) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_4(mht_4_v, 423, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::UpdateBeforeFuse");

  HloInstruction* fusion = instr1;
  HloInstruction* fused = instr2;
  if (is_fused(instr1)) {
    fusion = instr2;
    fused = instr1;
  }

  // Insert the newly created instruction (if any), to candidates_.
  for (auto use : fusion->users()) {
    if (candidates_index_.find(use) == candidates_index_.end()) {
      int64_t index = candidates_.size();
      candidates_.emplace_back(use);
      InsertOrDie(&candidates_index_, use, index++);
    }
  }

  // Update the reachability graph.
  UpdateReachability(fusion, fused, all_fusion_candidates_,
                     [this](HloInstruction* instr) { return is_fused(instr); });
}

void MultiOutputFusion::UpdateAfterFuse(
    HloInstruction* fusion,
    const std::vector<std::pair<HloInstruction*, int64_t>>& new_fusibles,
    bool new_fusion_node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_5(mht_5_v, 451, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::UpdateAfterFuse");

  FusionCandidate& candidate_node = candidates_[candidates_index_[fusion]];
  for (auto it : new_fusibles) {
    int64_t profit = GetProfit(it.first, fusion);
    if (new_fusion_node) {
      // If `fusion' is a new fusion node, then add all fusibles.
      if (profit > 0) {
        candidate_node.fusibles.emplace_back(it.first, profit);
        worklist_.emplace(fusion, it.first, profit);
      }
    } else {
      if (profit > it.second) {
        // If the new profit is higher than the old profit, add the fusible
        // into worklist.
        worklist_.emplace(fusion, it.first, profit);
      }
      if (it.second == 0) {
        // If the old profit is zero, that means `it.first' is not
        // originally fusible to the base op of `fusion', so we must add it
        // to candidate_node.fusibles.
        candidate_node.fusibles.emplace_back(it.first, profit);
      }
    }
  }
}

bool MultiOutputFusion::LegalToFuse(HloInstruction* instr1,
                                    HloInstruction* instr2) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_6(mht_6_v, 481, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::LegalToFuse");

  if (instr1->opcode() != HloOpcode::kFusion) {
    return false;
  }
  return LegalToFuseMainConstraints(instr1, instr2);
}

bool MultiOutputFusion::LegalToFuseMainConstraints(HloInstruction* instr1,
                                                   HloInstruction* instr2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_7(mht_7_v, 492, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::LegalToFuseMainConstraints");

  if (instr1 == instr2) {
    return false;
  }

  // Fusing nodes with 0 users makes no sense and the rest of the implementation
  // doesn't support it either.
  if (instr1->IsDead() || instr2->IsDead()) {
    return false;
  }

  // Check if the users of multioutput fusion is not a get-tuple-element.
  // If this is the case, we bail out because the transformation assumes
  // the users are get-tuple-element.
  auto multioutput_user_is_not_gte = [](HloInstruction* instr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_8(mht_8_v, 509, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "lambda");

    if (!instr->IsMultiOutputFusion()) {
      return false;
    }
    for (auto user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return true;
      }
    }
    return false;
  };
  if (multioutput_user_is_not_gte(instr1) ||
      multioutput_user_is_not_gte(instr2)) {
    return false;
  }
  if (is_connected(instr1, instr2)) {
    return false;
  }
  if (!ShapesCompatibleForFusion(instr1, instr2)) {
    return false;
  }

  // If both nodes are in-place operations and they use a common in-place
  // operand, we can't fuse these two.
  for (const auto& operand_and_output_index1 :
       HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr1)) {
    const HloInstruction* operand =
        instr1->operand(operand_and_output_index1.first.operand_number);
    for (const auto& operand_and_output_index2 :
         HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr2)) {
      if (operand ==
          instr2->operand(operand_and_output_index2.first.operand_number)) {
        return false;
      }
    }
  }
  return true;
}

void MultiOutputFusion::RecomputeReachability() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_9(mht_9_v, 551, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::RecomputeReachability");

  // Free the memory used for the reachability map before computing a new one.
  reachability_.reset();
  reachability_ = HloReachabilityMap::Build(computation_);
}

void MultiOutputFusion::UpdateReachability(
    HloInstruction* instr1, HloInstruction* instr2,
    absl::Span<const std::pair<HloInstruction*, HloReachabilityMap::Index>>
        instrs_to_update,
    const std::function<bool(HloInstruction*)>& skip) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_10(mht_10_v, 564, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::UpdateReachability");

  auto instr1_i = reachability_->GetIndex(instr1);
  auto instr2_i = reachability_->GetIndex(instr2);
  for (auto& instr_and_index : instrs_to_update) {
    HloInstruction* instr = instr_and_index.first;
    if (skip != nullptr && skip(instr)) {
      continue;
    }
    auto instr_i = instr_and_index.second;
    bool instr2_instr = reachability_->IsReachable(instr2_i, instr_i);
    bool instr1_instr = reachability_->IsReachable(instr1_i, instr_i);
    if (instr2_instr && instr1_instr) {
      // If a candidate was already reachable by both, no update needed.
      continue;
    }
    if (instr2_instr) {
      reachability_->FastSetReachabilityToUnion({instr_i, instr1_i}, instr_i);
    }
    if (reachability_->IsReachable(instr1_i, instr_i)) {
      reachability_->FastSetReachabilityToUnion({instr_i, instr2_i}, instr_i);
    }
  }
}

bool MultiOutputFusion::Perform() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_11(mht_11_v, 591, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::Perform");

  int changed = false;
  // Pick the top candidate from queue and try to merge.
  while (!worklist_.empty()) {
    ToBeFused candidate = worklist_.pop();

    HloInstruction* instr1 = candidate.instr1;
    HloInstruction* instr2 = candidate.instr2;

    // Candidates are already fused.
    if (is_fused(instr1) || is_fused(instr2)) {
      continue;
    }

    VLOG(1) << "Considering candidate profit_score=" << candidate.score
            << "\n\t\tinstr1 = " << instr1->ToString()
            << "\n\t\tinstr2 = " << instr2->ToString();

    if (LegalToFuse(instr1, instr2)) {
      if (!ConsumeFuel(name(), [&] {
            return absl::StrFormat("Not fusing %s and %s.", instr1->ToString(),
                                   instr2->ToString());
          })) {
        break;
      }
      VLOG(1) << "Fuse!";
      VLOG(2) << "Before multi_output_fusion:";
      VLOG(2) << "instr1: " << instr1->ToString();
      if (instr1->opcode() == HloOpcode::kFusion) {
        VLOG(2) << "\n"
                << instr1->fused_instructions_computation()->ToString(
                       HloPrintOptions().set_indent_amount(1));
      }
      VLOG(2) << "instr2: " << instr2->ToString();
      if (instr2->opcode() == HloOpcode::kFusion) {
        VLOG(2) << "\n"
                << instr2->fused_instructions_computation()->ToString(
                       HloPrintOptions().set_indent_amount(1));
      }
      UpdateBeforeFuse(instr1, instr2);
      std::vector<std::pair<HloInstruction*, int64_t>> new_fusibles =
          GetNewFusibles(instr1, instr2);
      HloInstruction* fusion = Fuse(instr1, instr2);
      if (fusion != instr1) {
        set_is_fused(instr1);
      }
      if (fusion != instr2) {
        set_is_fused(instr2);
      }
      UpdateAfterFuse(
          fusion, new_fusibles,
          /*new_fusion_node=*/(fusion != instr1) && (fusion != instr2));

      changed = true;
      VLOG(2) << "After fusion, \t this: " << fusion->name() << "\n"
              << fusion->fused_instructions_computation()->ToString(
                     HloPrintOptions().set_indent_amount(1));
    }
  }
  if (DoProducerConsumerMultiOutputFusion()) {
    changed = true;
  }
  return changed;
}

bool MultiOutputFusion::DoProducerConsumerMultiOutputFusion() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTcc mht_12(mht_12_v, 659, "", "./tensorflow/compiler/xla/service/multi_output_fusion.cc", "MultiOutputFusion::DoProducerConsumerMultiOutputFusion");
 return false; }

}  // namespace xla
