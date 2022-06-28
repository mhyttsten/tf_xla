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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc() {
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

#include "tensorflow/compiler/xla/service/ar_crs_combiner.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

// In SPMD mode, if there's a cross-replica all-reduce that produces the same
// value for all partitions, replaces it with a global all-reduce and then
// divide by the number of partitions. Depending on the topology and the
// implementation of the all-reduce for the backend, this may give a better
// performance.
StatusOr<bool> ReplaceReplicatedAllReduce(HloModule* module,
                                          int64_t partition_count) {
  TF_ASSIGN_OR_RETURN(
      auto replication_analysis,
      HloReplicationAnalysis::Run(module, /*cross_partition_spmd=*/true));

  bool changed = false;
  int64_t next_channel = hlo_query::NextChannelId(*module);
  for (auto computation : module->computations()) {
    for (auto instruction : computation->instructions()) {
      if (auto ar = DynCast<HloAllReduceInstruction>(instruction)) {
        const Shape& shape = ar->shape();
        if (ar->channel_id()) {
          continue;
        }
        if (ar->replica_groups().size() > 1) {
          continue;
        }
        if (shape.IsTuple() || shape.element_type() != F32) {
          continue;
        }
        // We would need a cost model for the target, but in general we want to
        // rewrite only if the replica count in the original op was large.
        if (module->config().replica_count() < 8 * partition_count) {
          continue;
        }
        if (replication_analysis->HloInstructionIsReplicatedAt(ar, {})) {
          VLOG(2) << "Replaced replicated all-reduce:" << ar->ToString();
          ar->set_channel_id(next_channel++);
          auto divisor =
              computation->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<float>(partition_count)));
          auto bcast = computation->AddInstruction(
              HloInstruction::CreateBroadcast(shape, divisor, {}));
          auto div = computation->AddInstruction(HloInstruction::CreateBinary(
              ar->shape(), HloOpcode::kDivide, ar, bcast));
          TF_RETURN_IF_ERROR(ar->ReplaceAllUsesWith(div));
          changed = true;
        }
      }
    }
  }
  return changed;
}

// Returns true if the given instruction (must be a cross-partition all-reduce)
// has a ReplicaGroup config that can be combined with cross-replica all-reduce.
// We currently restrict to those groups where all partitions in each replica
// belong to the same group.
bool HasCombinableReplicaGroup(HloInstruction* hlo, int64_t num_partitions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_0(mht_0_v, 263, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "HasCombinableReplicaGroup");

  auto all_reduce = Cast<HloAllReduceInstruction>(hlo);
  auto replica_groups = all_reduce->replica_groups();
  const int64_t replica_count = hlo->GetModule()->config().replica_count();
  CHECK(all_reduce->IsCrossModuleAllReduce());

  if (all_reduce->use_global_device_ids()) {
    if (replica_groups.size() != replica_count) {
      return false;
    }
    for (const auto& group : replica_groups) {
      if (group.replica_ids_size() != num_partitions) {
        return false;
      }
      absl::flat_hash_set<int64_t> partition_ids;
      int64_t replica_id = group.replica_ids(0) / num_partitions;
      for (int64_t i = 0; i < num_partitions; ++i) {
        if (group.replica_ids(i) / num_partitions != replica_id) {
          return false;
        }
        partition_ids.insert(group.replica_ids(i) % num_partitions);
      }
      if (partition_ids.size() != num_partitions) {
        return false;
      }
    }
    return true;
  }

  return replica_groups.size() == replica_count;
}

}  // namespace

namespace m = match;

// Checks if the argument instruction is an AllReduce, followed by a certain
// sequence of instructions and then a CRS. It must be possible to move
// the AR past each instruction in the sequence.
absl::optional<ArCrsCombiner::ArCrsPair> ArCrsCombiner::MatchesArCrsPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_1(mht_1_v, 306, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::MatchesArCrsPattern");

  auto can_ar_move_past_instruction = [](HloInstruction* instruction) -> bool {
    if (instruction->user_count() != 1) {
      return false;
    }
    switch (instruction->opcode()) {
      case HloOpcode::kBitcast:
      case HloOpcode::kTranspose:
      case HloOpcode::kReshape:
        return true;
      case HloOpcode::kConvert:
        // Can be moved across if both input and output is either float or
        // integer (e.g. S32<->U32 or F32<->BF16)
        return ShapeUtil::ElementIsFloating(instruction->shape()) ==
               ShapeUtil::ElementIsFloating(instruction->operand(0)->shape());
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
        // Only supported for floating point operands.
        return ShapeUtil::ElementIsFloating(instruction->shape());
      default:
        return false;
    }
  };

  auto computation_is_addition = [](HloComputation* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_2(mht_2_v, 334, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "lambda");

    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), m::Add(m::Parameter(), m::Parameter()));
  };

  // We only support combining cross-partition all-reduce where each replica
  // belongs to its own group, since the later cross-replica all-reduce combines
  // along the replica dimension.
  if (instruction->IsCrossModuleAllReduce() &&
      HasCombinableReplicaGroup(instruction, num_spatial_partitions_) &&
      computation_is_addition(instruction->called_computations()[0]) &&
      instruction->user_count() == 1) {
    auto next = instruction->users()[0];
    int64_t distance = 1;
    while (!next->IsCrossReplicaAllReduce()) {
      if (can_ar_move_past_instruction(next)) {
        next = next->users()[0];
      } else {
        return absl::nullopt;
      }
      ++distance;
    }
    if (!Cast<HloAllReduceInstruction>(next)->IsNoop() &&
        computation_is_addition(next->called_computations()[0])) {
      ArCrsPair pair(instruction, next, distance);
      VLOG(2) << "ArCrsPair matching pattern: " << pair.ToString();
      return pair;
    }
  }
  return absl::nullopt;
}

absl::optional<HloInstruction*> ArCrsCombiner::WhileFromBodyParameter(
    HloInstruction* instruction) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_3(mht_3_v, 370, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::WhileFromBodyParameter");

  CHECK_EQ(HloOpcode::kParameter, instruction->opcode());
  HloComputation* computation = instruction->parent();
  auto caller_instructions = call_graph_->GetComputationCallers(computation);
  if (caller_instructions.size() == 1) {
    auto caller_instruction = caller_instructions[0];
    if (caller_instruction->opcode() == HloOpcode::kWhile) {
      return caller_instruction;
    }
  }
  return absl::nullopt;
}

absl::optional<HloInstruction*> ArCrsCombiner::ConditionalFromBodyParameter(
    HloInstruction* instruction) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_4(mht_4_v, 387, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::ConditionalFromBodyParameter");

  CHECK_EQ(HloOpcode::kParameter, instruction->opcode());
  HloComputation* computation = instruction->parent();
  auto caller_instructions = call_graph_->GetComputationCallers(computation);
  if (caller_instructions.size() == 1) {
    auto caller_instruction = caller_instructions[0];
    if (caller_instruction->opcode() == HloOpcode::kConditional) {
      return caller_instruction;
    }
  }
  return absl::nullopt;
}

absl::optional<std::vector<HloInstruction*>> ArCrsCombiner::GetAllTuples(
    HloInstruction* instruction,
    absl::flat_hash_set<HloInstruction*>* visited) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_5(mht_5_v, 405, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::GetAllTuples");

  if (visited->find(instruction) != visited->end()) {
    return std::vector<HloInstruction*>();
  }
  visited->insert(instruction);

  switch (instruction->opcode()) {
    case HloOpcode::kTuple: {
      return std::vector<HloInstruction*>({instruction});
    }
    case HloOpcode::kDomain: {
      return GetAllTuples(instruction->operands()[0], visited);
    }
    case HloOpcode::kParameter: {
      auto maybe_while = WhileFromBodyParameter(instruction);
      if (maybe_while) {
        auto while_instr = *maybe_while;
        auto init_tuples = GetAllTuples(while_instr->while_init(), visited);
        auto body_tuples = GetAllTuples(
            while_instr->while_body()->root_instruction(), visited);
        if (!init_tuples || !body_tuples) {
          return absl::nullopt;
        }
        auto result = *init_tuples;
        result.insert(result.end(), body_tuples->begin(), body_tuples->end());
        return result;
      }
      auto maybe_conditional = ConditionalFromBodyParameter(instruction);
      if (maybe_conditional) {
        auto cond_instr = *maybe_conditional;
        std::vector<HloInstruction*> tuples;
        for (int64_t i = 0; i < cond_instr->branch_computations().size(); ++i) {
          if (cond_instr->branch_computation(i)->parameter_instruction(0) ==
              instruction) {
            // If the same computation is used for more than one branch of the
            // conditional, we collect the arguments that flow to the
            // computation from all branches.
            auto branch_tuples =
                GetAllTuples(cond_instr->mutable_operand(i + 1), visited);
            if (!branch_tuples) {
              return absl::nullopt;
            }
            tuples.insert(tuples.end(), branch_tuples->begin(),
                          branch_tuples->end());
          }
        }
        return tuples;
      }
      return absl::nullopt;
    }
    case HloOpcode::kGetTupleElement: {
      std::vector<HloInstruction*> result_tuples;
      auto tuples = GetAllTuples(instruction->operands()[0], visited);
      if (!tuples) {
        return absl::nullopt;
      }
      for (auto tuple : *tuples) {
        auto tmp_tuples = GetAllTuples(
            tuple->mutable_operand(instruction->tuple_index()), visited);
        if (!tmp_tuples) {
          return absl::nullopt;
        }
        result_tuples.insert(result_tuples.end(), tmp_tuples->begin(),
                             tmp_tuples->end());
      }
      return result_tuples;
    }
    case HloOpcode::kConditional: {
      std::vector<HloInstruction*> result_tuples;
      const auto& branch_computations = instruction->branch_computations();
      result_tuples.reserve(branch_computations.size());
      for (HloComputation* body : branch_computations) {
        if (body->root_instruction()->opcode() != HloOpcode::kTuple) {
          return absl::nullopt;
        }
        result_tuples.push_back(body->root_instruction());
      }
      return result_tuples;
    }
    case HloOpcode::kWhile: {
      auto init_tuples = GetAllTuples(instruction->while_init(), visited);
      auto body_tuples =
          GetAllTuples(instruction->while_body()->root_instruction(), visited);
      if (!init_tuples || !body_tuples) {
        return absl::nullopt;
      }
      auto result = *init_tuples;
      result.insert(result.end(), body_tuples->begin(), body_tuples->end());
      return result;
    }
    default:
      return absl::nullopt;
  }
}

bool ArCrsCombiner::TupleElementsComputeSameValue(
    HloInstruction* tuple_shaped_instruction, int64_t i1, int64_t i2,
    absl::flat_hash_map<int64_t, int64_t>* visited_pairs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_6(mht_6_v, 505, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::TupleElementsComputeSameValue");

  absl::flat_hash_set<HloInstruction*> visited;
  auto tuples = GetAllTuples(tuple_shaped_instruction, &visited);
  if (!tuples) {
    return false;
  }
  for (auto tuple : *tuples) {
    CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
    if (!InstructionsComputeSameValue(tuple->mutable_operand(i1),
                                      tuple->mutable_operand(i2),
                                      visited_pairs)) {
      return false;
    }
  }
  return true;
}

/* static */
bool ArCrsCombiner::TestInstructionsComputeSameValue(HloInstruction* i1,
                                                     HloInstruction* i2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_7(mht_7_v, 527, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::TestInstructionsComputeSameValue");

  ArCrsCombiner combiner(/*num_spatial_partitions=*/2,
                         /*spmd_partition=*/false);
  auto module = i1->parent()->parent();
  CHECK_EQ(module, i2->parent()->parent());
  combiner.call_graph_ = CallGraph::Build(module);
  absl::flat_hash_map<int64_t, int64_t> visited_pairs;
  return combiner.InstructionsComputeSameValue(i1, i2, &visited_pairs);
}

bool ArCrsCombiner::InstructionsComputeSameValue(
    HloInstruction* i1, HloInstruction* i2,
    absl::flat_hash_map<int64_t, int64_t>* visited_pairs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_8(mht_8_v, 542, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::InstructionsComputeSameValue");

  if (i1 == i2) {
    return true;
  }
  auto uid1 = i1->unique_id();
  auto uid2 = i2->unique_id();
  auto min_uid = std::min(uid1, uid2);
  auto max_uid = std::max(uid1, uid2);
  auto it = visited_pairs->find(min_uid);
  if (it != visited_pairs->end() && max_uid == it->second) {
    return true;
  }
  auto opcode1 = i1->opcode();
  auto operands1 = i1->operands();
  if (opcode1 != i2->opcode() || operands1.size() != i2->operands().size()) {
    return false;
  }
  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_9(mht_9_v, 562, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "lambda");

    return *a == *b;
  };
  // Two MPMD AllReduces are identical if they have the same channel_id. Their
  // operands don't have to be identical.
  auto eq_operands = [](const HloInstruction*, const HloInstruction*) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_10(mht_10_v, 570, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "lambda");

    return true;
  };
  if (i1->IsCrossModuleAllReduce()) {
    return i1->Identical(*i2, eq_operands, eq_computations,
                         /*layout_sensitive=*/false);
  }
  visited_pairs->emplace(min_uid, max_uid);
  for (int i = 0; i < operands1.size(); ++i) {
    auto operand1 = operands1[i];
    auto operand2 = i2->operands()[i];
    if (!InstructionsComputeSameValue(operand1, operand2, visited_pairs)) {
      return false;
    }
  }
  if (opcode1 == HloOpcode::kParameter) {
    // In the general case, we don't try to prove equality of parameters.
    // We only try in the context of get-tuple-element
    // (see TupleElementsComputeSameValue).
    return false;
  }
  if (opcode1 == HloOpcode::kGetTupleElement) {
    return i1->tuple_index() == i2->tuple_index() ||
           TupleElementsComputeSameValue(operands1[0], i1->tuple_index(),
                                         i2->tuple_index(), visited_pairs);
  }
  // Don't check that the operands are identical, because Identical can
  // return false for instructions that compute the same value but are not
  // identical, which we don't want. We have checked the arguments with
  // InstructionsComputeSameValue earlier.
  auto eq_instructions = [](const HloInstruction* i1,
                            const HloInstruction* i2) -> bool { return true; };
  return i1->Identical(*i2, eq_instructions, eq_computations,
                       /*layout_sensitive=*/false);
}

void ArCrsCombiner::GroupAllReducesById(HloModule* module) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_11(mht_11_v, 609, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::GroupAllReducesById");

  // Say that two or more ARs lead to the same CRS: (AR1, CRS), (AR2, CRS),
  // ... , (ARn, CRS).
  // If as we traverse the HLO graph we start tracking the pair (AR2, CRS),
  // and later find that AR1's distance from the CRS is longer, we discard
  // AR2 and start tracking AR1. We put the discarded ids in this set, in order
  // to skip processing of short paths when we encounter the other ARs that
  // have the same id as AR2.
  absl::flat_hash_set<int64_t> discarded_ar_ids;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      auto maybe_pair = MatchesArCrsPattern(instruction);
      if (maybe_pair) {
        auto pair = *maybe_pair;
        int64_t ar_id = *(instruction->channel_id());
        if (discarded_ar_ids.find(ar_id) != discarded_ar_ids.end()) {
          continue;
        }
        auto it = crs_reserved_map_.find(pair.crs);
        if (it != crs_reserved_map_.end()) {
          auto prev_ar_id = it->second;
          // Since there is another AR paired with CRS,
          // all_reduce_map_[prev_ar_id] should exist, but
          // all_reduce_map_[ar_id] shouldn't.
          CHECK(all_reduce_map_.find(ar_id) == all_reduce_map_.end());
          CHECK_NE(prev_ar_id, ar_id);
          auto prev_pair = all_reduce_map_[prev_ar_id].back();
          int64_t prev_distance = prev_pair.distance;
          if (prev_distance < pair.distance) {
            // The current AR's distance to CRS is longer than the previously
            // tracked AR, so we discard the previous AR.
            VLOG(2) << "Replacing ArCrsPair: " << prev_pair.ToString()
                    << " with ArCrsPair: " << pair.ToString();
            all_reduce_map_.erase(prev_ar_id);
            discarded_ar_ids.insert(prev_ar_id);
            all_reduce_map_[ar_id].push_back(pair);
            crs_reserved_map_[pair.crs] = ar_id;
          } else {
            // Discard the current AR id because we are keeping the previously
            // tracked AR.
            discarded_ar_ids.insert(ar_id);
          }
        } else {
          if (all_reduce_map_.find(ar_id) != all_reduce_map_.end()) {
            int64_t prev_distance = all_reduce_map_[ar_id].back().distance;
            CHECK_EQ(prev_distance, pair.distance)
                << "All ARs with the same AR ID must have the same distance "
                   "from the corresponding CRSs. Found: "
                << prev_distance << " and " << pair.distance;
          }
          all_reduce_map_[ar_id].push_back(pair);
          crs_reserved_map_[pair.crs] = ar_id;
        }
      }
    }
  }
}

Status ArCrsCombiner::KeepProvablyEqualInstructionGroupsMPMD() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_12(mht_12_v, 670, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::KeepProvablyEqualInstructionGroupsMPMD");

  for (auto it = all_reduce_map_.begin(); it != all_reduce_map_.end();) {
    auto copy_it = it++;  // Advance `it` before invalidation from erase.
    auto channel_id = copy_it->first;
    VLOG(2)
        << "KeepProvablyEqualInstructionGroups. Checking AllReduce channel id: "
        << channel_id << "\n";
    auto pairs_vec = copy_it->second;
    TF_RET_CHECK(pairs_vec.size() == num_spatial_partitions_);
    auto instr_0 = pairs_vec[0].ar;
    for (int i = 1; i < pairs_vec.size(); ++i) {
      auto instr_i = pairs_vec[i].ar;
      auto next_0 = instr_0->users()[0];
      auto next_i = instr_i->users()[0];
      absl::flat_hash_map<int64_t, int64_t> visited_pairs;
      while (true) {
        if (!InstructionsComputeSameValue(next_0, next_i, &visited_pairs)) {
          all_reduce_map_.erase(copy_it);
          VLOG(2) << "KeepProvablyEqualInstructionGroups. Erased AllReduce "
                     "channel id: "
                  << channel_id << "\n";
          break;
        }
        if (next_0->IsCrossReplicaAllReduce()) {
          break;
        }
        next_0 = next_0->users()[0];
        next_i = next_i->users()[0];
      }
    }
  }
  return Status::OK();
}

Status ArCrsCombiner::KeepProvablyEqualInstructionGroupsSPMD(
    HloModule* module) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_13(mht_13_v, 708, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::KeepProvablyEqualInstructionGroupsSPMD");

  // For SPMD mode, use HloReplicationAnalysis to figure out HLO value
  // equivalence across partitions.
  TF_ASSIGN_OR_RETURN(
      auto replication_analysis,
      HloReplicationAnalysis::Run(module, /*cross_partition_spmd=*/true));

  for (auto it = all_reduce_map_.begin(); it != all_reduce_map_.end();) {
    auto copy_it = it++;  // Advance `it` before invalidation from erase.
    auto channel_id = copy_it->first;
    VLOG(2)
        << "KeepProvablyEqualInstructionGroups. Checking AllReduce channel id: "
        << channel_id << "\n";
    auto pairs_vec = copy_it->second;
    TF_RET_CHECK(pairs_vec.size() == 1);
    auto instr = pairs_vec[0].ar;
    auto next = instr->users()[0];
    while (true) {
      // The patterns we detect in ArCrsCombiner::MatchesArCrsPattern()
      // guarantee that the HLO produces an array.
      TF_RET_CHECK(next->shape().IsArray());
      if (!replication_analysis->HloInstructionIsReplicatedAt(next, {})) {
        all_reduce_map_.erase(copy_it);
        VLOG(2) << "KeepProvablyEqualInstructionGroups. Erased AllReduce "
                   "channel id: "
                << channel_id << "\n";
        break;
      }
      if (next->IsCrossReplicaAllReduce()) {
        break;
      }
      next = next->users()[0];
    }
  }
  return Status::OK();
}

StatusOr<bool> ArCrsCombiner::RewriteGraph() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_14(mht_14_v, 748, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::RewriteGraph");

  if (all_reduce_map_.empty()) {
    return false;
  }
  for (const auto& it : all_reduce_map_) {
    auto pairs_vec = it.second;
    for (auto pair : pairs_vec) {
      auto all_reduce = pair.ar;
      auto parent_computation = all_reduce->parent();
      auto channel_id = all_reduce->channel_id();
      auto prev = all_reduce->mutable_operand(0);
      auto next = all_reduce->users()[0];
      TF_CHECK_OK(all_reduce->ReplaceUseWith(next, prev));
      TF_CHECK_OK(parent_computation->RemoveInstruction(all_reduce));
      while (!next->IsCrossReplicaAllReduce()) {
        switch (next->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kTranspose:
          case HloOpcode::kReshape:
          case HloOpcode::kConvert:
          case HloOpcode::kMultiply:
            break;
          case HloOpcode::kAdd:
          case HloOpcode::kSubtract: {
            auto other_operand = (next->operands()[0] == prev)
                                     ? next->operands()[1]
                                     : next->operands()[0];
            // To move the AR past the addition/subtraction, we need to divide
            // other_operand by the number of spatial partitions, except if
            // other_operand is a cross-module AR, which can be eliminated.
            if (other_operand->IsCrossModuleAllReduce() &&
                other_operand->user_count() == 1) {
              TF_CHECK_OK(other_operand->ReplaceAllUsesWith(
                  other_operand->mutable_operand(0)));
            } else {
              auto shape = other_operand->shape();
              Literal lit(shape);
              lit.PopulateWithValue<float>(num_spatial_partitions_);
              auto divisor = parent_computation->AddInstruction(
                  HloInstruction::CreateConstant(lit.Clone()));
              auto division = parent_computation->AddInstruction(
                  HloInstruction::CreateBinary(shape, HloOpcode::kDivide,
                                               other_operand, divisor));
              TF_CHECK_OK(other_operand->ReplaceUseWith(next, division));
            }
            break;
          }
          default:
            LOG(FATAL) << "Unexpected instruction: " << next->ToShortString();
        }
        prev = next;
        next = next->users()[0];
      }
      // The AllReduce and the CRS are combined to an all-core AllReduce.
      //
      // Note that we can just reuse the ReplicaGroup config of cross-replica
      // all-reduce since we already checked that cross-partition all-reduce
      // is always across all partitions (HasCombinableReplicaGroup). We need to
      // combine ReplicaGroup configs using global ids here if we relax that
      // restriction.
      next->set_channel_id(channel_id);
    }
  }
  return true;
}

StatusOr<bool> ArCrsCombiner::Run(HloModule* module) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTcc mht_15(mht_15_v, 817, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.cc", "ArCrsCombiner::Run");

  call_graph_ = CallGraph::Build(module);

  GroupAllReducesById(module);

  if (spmd_partition_) {
    TF_RETURN_IF_ERROR(KeepProvablyEqualInstructionGroupsSPMD(module));
  } else {
    TF_RETURN_IF_ERROR(KeepProvablyEqualInstructionGroupsMPMD());
  }

  TF_ASSIGN_OR_RETURN(auto changed, RewriteGraph());

  if (module->config().replica_count() > 1 && spmd_partition_) {
    TF_ASSIGN_OR_RETURN(auto replaced, ReplaceReplicatedAllReduce(
                                           module, num_spatial_partitions_));
    changed |= replaced;
  }

  return changed;
}

}  // namespace xla
