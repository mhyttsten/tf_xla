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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc() {
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
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

FusionNodeIndexingEvaluation::FusionNodeIndexingEvaluation(
    const HloInstruction* fusion, int64_t root_usage_count)
    : fusion_(fusion) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::FusionNodeIndexingEvaluation");

  HloInstruction* root = fusion->fused_expression_root();
  indexing_users_[root].insert(fusion);
  index_usage_count_[fusion] = root_usage_count;
  RecomputeCache();
}

// This constant is arbitrarily chosen. Essentially we don't want to have too
// much code duplication, because it slows down the compilation time. There is
// a tradeoff between compilation time and runtime here.
const int64_t FusionNodeIndexingEvaluation::kAllowedCodeDuplication = 15;

namespace {

// Returns which ops invalidate the cache of emitted instructions by creating a
// new BasicBlock and setting the insertion point to the newly created
// BasicBlock. We can only reuse cached values if they were emitted in the same
// BasicBlock as the current BasicBlock.
bool OpInvalidatesCache(const HloInstruction* hlo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "OpInvalidatesCache");

  switch (hlo->opcode()) {
    // This list of ops was created by inspecting the code. There is no
    // guarantee that it is complete.
    case HloOpcode::kConcatenate:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return true;
    default:
      return false;
  }
}

// Counts the number of "real" users of 'hlo'. When 'hlo' has a fusion node as
// user, we consider the users of the fusion parameter corresponding to 'hlo' as
// the real users.
int64_t UserCount(const HloInstruction* hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "UserCount");

  int64_t cnt = 0;
  for (HloInstruction* user : hlo->users()) {
    if (user->opcode() == HloOpcode::kFusion) {
      // Count the number of users of the parameter corresponding to the fusion
      // operand.
      int64_t operand_index = user->operand_index(hlo);
      cnt += user->fused_parameter(operand_index)->user_count();
    } else {
      ++cnt;
    }
  }
  return cnt;
}
}  // namespace

bool FusionNodeIndexingEvaluation::CodeDuplicationTooHigh(
    const HloInstruction* producer) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::CodeDuplicationTooHigh");

  int64_t emitted_instructions = EvaluateEmittedInstructions(producer);
  return emitted_instructions > kAllowedCodeDuplication ||
         (OpInvalidatesCache(producer) &&
          (emitted_instructions > 1 || UserCount(producer) > 1));
}

bool FusionNodeIndexingEvaluation::MaxCodeDuplicationTooHigh() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_4(mht_4_v, 271, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::MaxCodeDuplicationTooHigh");

  for (const auto& entry : index_usage_count_) {
    if (entry.second > kAllowedCodeDuplication ||
        (OpInvalidatesCache(entry.first) &&
         (entry.second > 1 || UserCount(entry.first) > 1))) {
      return true;
    }
  }
  return false;
}

int64_t FusionNodeIndexingEvaluation::EvaluateEmittedInstructions(
    const HloInstruction* producer) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_5(mht_5_v, 286, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::EvaluateEmittedInstructions");

  int64_t total = 0;
  for (const auto* user : indexing_users_.at(producer)) {
    total += index_usage_count_.at(user);
  }
  return total;
}

void FusionNodeIndexingEvaluation::UpdateEvaluationCache(
    const HloInstruction* producer,
    absl::flat_hash_set<const HloInstruction*> indexing_users_of_producer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::UpdateEvaluationCache");

  CHECK(!indexing_users_.contains(producer));
  indexing_users_[producer] = std::move(indexing_users_of_producer);
  UpdateIndexUsageCount(producer);
  UpdateIndexingUsersOfOperands(producer);
}

absl::flat_hash_set<const HloInstruction*>
FusionNodeIndexingEvaluation::RemoveFusionOperand(
    HloInstruction* fusion_operand) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_7(mht_7_v, 311, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::RemoveFusionOperand");

  auto indexing_users_of_operand =
      std::move(indexing_users_.at(fusion_operand));
  indexing_users_.erase(fusion_operand);
  CHECK(!index_usage_count_.contains(fusion_operand));
  return indexing_users_of_operand;
}

void FusionNodeIndexingEvaluation::RecomputeCache() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_8(mht_8_v, 322, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::RecomputeCache");

  auto postorder =
      fusion_->fused_instructions_computation()->MakeInstructionPostOrder();
  std::reverse(postorder.begin(), postorder.end());
  for (const auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    UpdateIndexUsageCount(instruction);
    UpdateIndexingUsersOfOperands(instruction);
  }
}

void FusionNodeIndexingEvaluation::UpdateIndexUsageCount(
    const HloInstruction* instruction) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_9(mht_9_v, 339, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::UpdateIndexUsageCount");

  int64_t total = 0;
  for (const auto* user : indexing_users_[instruction]) {
    total += index_usage_count_.at(user);
  }
  CHECK(index_usage_count_.emplace(instruction, total).second);
}

void FusionNodeIndexingEvaluation::UpdateIndexingUsersOfOperands(
    const HloInstruction* instruction) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSfusion_node_indexing_evaluationDTcc mht_10(mht_10_v, 351, "", "./tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.cc", "FusionNodeIndexingEvaluation::UpdateIndexingUsersOfOperands");

  for (const auto* operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kParameter) {
      // Although actually the parameter gets indexed, we store it as indexing
      // of the corresponding fusion operand instead because parameter
      // instruction pointers can be invalidated when we fuse another
      // instruction into 'fusion_'.
      operand = fusion_->operand(operand->parameter_number());
    }
    // For simplicity we assume that all shape and layout changing
    // operations except Transposes invalidate index reuse. Transposes are
    // special: although they are shape changing, we can reuse the
    // multi-dimensional index for the operand by permuting it.
    if (instruction->opcode() == HloOpcode::kTranspose ||
        Shape::Equal().IgnoreElementType()(operand->shape(),
                                           instruction->shape())) {
      // If the index is reused, it means the operand gets index values
      // from the same set of (indirect) users as 'instruction' itself.
      indexing_users_[operand].insert(indexing_users_[instruction].begin(),
                                      indexing_users_[instruction].end());
    } else {
      // If the index is not reused, it means 'instruction' computes a
      // new index derived from the index it gets.
      indexing_users_[operand].insert(instruction);
    }
  }
}

}  // namespace xla
