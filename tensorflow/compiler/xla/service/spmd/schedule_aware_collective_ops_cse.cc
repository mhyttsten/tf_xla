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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc() {
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

#include "tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace {

// Returns if an instructions adds only degenerate dimensions to the shape of
// the input, like going from [X,Y] to [1,X,Y,1].
bool IsAddingOnlyDegenerateDimensions(const HloInstruction* inst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "IsAddingOnlyDegenerateDimensions");

  if (inst->opcode() != HloOpcode::kBitcast &&
      inst->opcode() != HloOpcode::kReshape) {
    return false;
  }
  const Shape& in_shape = inst->operand(0)->shape();
  const Shape& out_shape = inst->shape();
  return ShapeUtil::ElementsIn(in_shape) == ShapeUtil::ElementsIn(out_shape) &&
         ShapeUtil::DimensionsUnmodifiedByReshape(in_shape, out_shape).size() ==
             in_shape.rank();
}

// Passthrough reshapes or bitcasts adding only degenerate hdimensions to some
// shape.
const HloInstruction* PassthroughDegenerateAddingReshapes(
    const HloInstruction* inst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "PassthroughDegenerateAddingReshapes");

  while (IsAddingOnlyDegenerateDimensions(inst)) {
    inst = inst->operand(0);
  }
  return inst;
}

bool ShouldConsiderSchedule(HloInstruction* hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "ShouldConsiderSchedule");

  return hlo->opcode() != HloOpcode::kCollectivePermute;
}

HloInstruction* MayConsiderCollective(HloInstruction* hlo, bool for_replicas) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "MayConsiderCollective");

  auto chan_instr = DynCast<HloChannelInstruction>(hlo);
  if (!chan_instr) {
    return nullptr;
  }
  if (for_replicas == chan_instr->channel_id().has_value()) {
    return nullptr;
  }
  if (hlo->opcode() == HloOpcode::kCollectivePermute) {
    return hlo;
  }
  auto coll = DynCast<HloCollectiveInstruction>(hlo);
  if (!coll) {
    return nullptr;
  }
  if (coll->constrain_layout()) {
    return nullptr;
  }
  if (coll->opcode() == HloOpcode::kAllGather) {
    return coll;
  }
  // Consider broadcast -> dynamic-update-slice -> all-reduce as all-gather.
  if (coll->opcode() == HloOpcode::kAllReduce && coll->shape().IsArray()) {
    auto operand = coll->operand(0);
    return operand->opcode() == HloOpcode::kDynamicUpdateSlice &&
                   operand->operand(0)->opcode() == HloOpcode::kBroadcast
               ? coll
               : nullptr;
  }
  return nullptr;
}

StatusOr<bool> RunOnComputation(HloComputation* comp, bool for_replicas,
                                int64_t distance_threshold) {
  // We consider estimate the live ranges of all-gathers by comparing their
  // users' distance to the root, e.g., height.
  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, int64_t> height;
  auto ordered_hlos = comp->MakeInstructionPostOrder();
  int64_t max_height = 0;
  for (auto it = ordered_hlos.rbegin(); it != ordered_hlos.rend(); ++it) {
    auto hlo = *it;
    int64_t h = 0;
    for (auto user : hlo->users()) {
      h = std::max(h, height[user]) + 1;
    }
    max_height = std::max(max_height, h);
    height[hlo] = h;
  }

  auto lowest_user_height = [&](const HloInstruction* hlo) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_4(mht_4_v, 287, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "lambda");

    int64_t lowest = height[hlo];
    for (auto user : hlo->users()) {
      lowest = std::min(lowest, height[user]);
    }
    return lowest;
  };

  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      operand_to_collective;
  for (HloInstruction* hlo : ordered_hlos) {
    HloInstruction* coll = MayConsiderCollective(hlo, for_replicas);
    if (!coll) {
      continue;
    }
    auto& earlier_colls =
        operand_to_collective[PassthroughDegenerateAddingReshapes(
            coll->operand(0))];
    bool found = false;
    int64_t coll_height = height[coll];
    for (HloInstruction* earlier_coll : earlier_colls) {
      if (!ShapeUtil::Equal(earlier_coll->shape(), coll->shape())) {
        continue;
      }
      HloInstruction* coll_operand = coll->mutable_operand(0);
      TF_RETURN_IF_ERROR(
          coll->ReplaceOperandWith(0, earlier_coll->mutable_operand(0)));
      if (!earlier_coll->IdenticalIgnoringChannelIdValues(*coll)) {
        TF_RETURN_IF_ERROR(coll->ReplaceOperandWith(0, coll_operand));
        continue;
      }
      found = true;
      if (ShouldConsiderSchedule(coll) &&
          lowest_user_height(earlier_coll) > coll_height + distance_threshold) {
        TF_RETURN_IF_ERROR(coll->ReplaceOperandWith(0, coll_operand));
        earlier_coll = coll;
        continue;
      }
      changed = true;
      VLOG(1) << "Replacing " << coll->ToString() << " with "
              << earlier_coll->ToString();
      TF_RETURN_IF_ERROR(coll->ReplaceAllUsesWith(earlier_coll));
      break;
    }
    if (!found) {
      earlier_colls.push_back(coll);
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> ScheduleAwareCollectiveOpsCSE::Run(HloModule* module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPSschedule_aware_collective_ops_cseDTcc mht_5(mht_5_v, 343, "", "./tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.cc", "ScheduleAwareCollectiveOpsCSE::Run");

  bool changed = false;
  for (auto comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(
        auto comp_changed,
        RunOnComputation(comp, for_replicas_, distance_threshold_));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
