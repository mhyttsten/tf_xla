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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creatorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creatorDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_collective_creator.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

StatusOr<bool> AsyncCollectiveCreator::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creatorDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/service/async_collective_creator.cc", "AsyncCollectiveCreator::Run");

  bool changed = false;
  struct ReplacedAsync {
    HloInstruction* start;
    HloInstruction* done;
  };
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Find all all-reduce ops first as we can't modify the instructions while
    // iterating through them.
    std::vector<HloInstruction*> supported_collectives;
    for (HloInstruction* instruction : computation->instructions()) {
      if ((instruction->opcode() == HloOpcode::kAllReduce &&
           convert_all_reduce_(instruction)) ||
          (instruction->opcode() == HloOpcode::kAllGather &&
           convert_all_gather_(instruction)) ||
          (instruction->opcode() == HloOpcode::kCollectivePermute &&
           convert_collective_permute_(instruction))) {
        supported_collectives.push_back(instruction);
      }
    }
    if (supported_collectives.empty()) {
      continue;
    }

    absl::flat_hash_map<HloInstruction*, ReplacedAsync> replaced_pairs;
    bool should_update_schedule =
        module->has_schedule() &&
        module->schedule().is_computation_scheduled(computation);
    for (HloInstruction* instruction : supported_collectives) {
      if (HloAllReduceInstruction* ar =
              DynCast<HloAllReduceInstruction>(instruction)) {
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllReduceStart(
                ar->shape(), ar->operands(), ar->to_apply(),
                ar->replica_groups(), ar->constrain_layout(), ar->channel_id(),
                ar->use_global_device_ids()));
        std::unique_ptr<HloInstruction> done = HloInstruction::CreateUnary(
            ar->shape(), HloOpcode::kAllReduceDone, start);
        start->set_metadata(ar->metadata());
        start->CopyBackendConfigFrom(ar);
        if (should_update_schedule) {
          replaced_pairs[ar] = ReplacedAsync{start, done.get()};
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(ar, std::move(done)),
            "replacing ", ar->ToShortString());
        changed = true;
        continue;
      }
      if (HloAllGatherInstruction* ag =
              DynCast<HloAllGatherInstruction>(instruction)) {
        std::vector<Shape> operand_shapes;
        operand_shapes.reserve(ag->operand_count());
        for (const HloInstruction* op : ag->operands()) {
          operand_shapes.push_back(op->shape());
        }
        Shape shape = ShapeUtil::MakeTupleShape(
            {ag->operand_count() > 1 ? ShapeUtil::MakeTupleShape(operand_shapes)
                                     : operand_shapes[0],
             ag->shape()});
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllGatherStart(
                shape, ag->operands(), ag->all_gather_dimension(),
                ag->replica_groups(), ag->constrain_layout(), ag->channel_id(),
                ag->use_global_device_ids()));
        std::unique_ptr<HloInstruction> done = HloInstruction::CreateUnary(
            ag->shape(), HloOpcode::kAllGatherDone, start);
        start->set_metadata(ag->metadata());
        start->CopyBackendConfigFrom(ag);
        if (should_update_schedule) {
          replaced_pairs[ag] = ReplacedAsync{start, done.get()};
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(ag, std::move(done)),
            "replacing ", ag->ToShortString());
        changed = true;
        continue;
      }
      if (HloCollectivePermuteInstruction* cp =
              DynCast<HloCollectivePermuteInstruction>(instruction)) {
        HloInstruction* collective_permute_start;
        HloInstruction* operand = cp->mutable_operand(0);
        if (cp->operand_count() == 1) {
          collective_permute_start = computation->AddInstruction(
              HloInstruction::CreateCollectivePermuteStart(
                  ShapeUtil::MakeTupleShape(
                      {operand->shape(), cp->shape(),
                       ShapeUtil::MakeShape(U32, {}, {}),
                       ShapeUtil::MakeShape(U32, {}, {})}),
                  operand, cp->source_target_pairs(), cp->channel_id()));
        } else {
          CHECK_EQ(cp->operand_count(), 4);
          std::vector<const Shape*> operand_shapes;
          absl::c_transform(cp->operands(), std::back_inserter(operand_shapes),
                            [](const HloInstruction* operand) {
                              return &(operand->shape());
                            });
          collective_permute_start = computation->AddInstruction(
              HloInstruction::CreateCollectivePermuteStart(
                  ShapeInference::InferCollectivePermuteStartShape(
                      operand_shapes)
                      .ValueOrDie(),
                  operand, cp->mutable_operand(1), cp->mutable_operand(2),
                  cp->mutable_operand(3), cp->source_target_pairs(),
                  cp->dynamic_slice_sizes_list(), cp->channel_id()));
        }
        collective_permute_start->set_metadata(cp->metadata());
        collective_permute_start->CopyBackendConfigFrom(cp);
        HloInstruction* collective_permute_done =
            computation->AddInstruction(HloInstruction::CreateUnary(
                cp->shape(), HloOpcode::kCollectivePermuteDone,
                collective_permute_start));
        if (should_update_schedule) {
          replaced_pairs[cp] =
              ReplacedAsync{collective_permute_start, collective_permute_done};
        }
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstruction(cp, collective_permute_done));
        changed = true;
        continue;
      }
    }
    if (should_update_schedule) {
      std::vector<HloInstruction*> new_sequence;
      const HloInstructionSequence& sequence =
          module->schedule().sequence(computation);
      new_sequence.reserve(sequence.size() + replaced_pairs.size());
      for (HloInstruction* instr : sequence.instructions()) {
        auto it = replaced_pairs.find(instr);
        if (it != replaced_pairs.end()) {
          new_sequence.push_back(it->second.start);
          new_sequence.push_back(it->second.done);
          continue;
        }
        new_sequence.push_back(instr);
      }
      module->schedule().set_sequence(computation, new_sequence);
    }
  }
  return changed;
}

}  // namespace xla
