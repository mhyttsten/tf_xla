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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc() {
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

#include "tensorflow/compiler/xla/service/all_gather_combiner.h"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/collective_combiner_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single AllGather op. All entries
// in to_combine must be AllGather ops with exactly one operand and the same
// all_gather_dimension.
Status CombineAllGathers(absl::Span<HloInstruction* const> to_combine) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/all_gather_combiner.cc", "CombineAllGathers");

  if (to_combine.size() < 2) {
    return Status::OK();
  }
  VLOG(1) << "Combined " << to_combine.size() << " AllGather ops";

  HloComputation& computation = *to_combine.back()->parent();
  int64_t all_gather_dimension =
      Cast<HloAllGatherInstruction>(to_combine.front())->all_gather_dimension();

  // Create a single bigger AllGather of the operands of the smaller AllGather.
  std::vector<HloInstruction*> operands;
  std::vector<Shape> output_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllGather);
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(Cast<HloAllGatherInstruction>(hlo)->all_gather_dimension() ==
                 all_gather_dimension);
    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      output_shapes.push_back(hlo->shape());
    }
  }

  HloInstruction* combined;
  // AllGather ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateAllGather(
      ShapeUtil::MakeTupleShape(output_shapes), operands, all_gather_dimension,
      to_combine.front()->replica_groups(),
      /*constrain_layout=*/false, to_combine.front()->channel_id(),
      Cast<HloAllGatherInstruction>(to_combine.front())
          ->use_global_device_ids()));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  if (to_combine.front()->has_sharding()) {
    combined->set_sharding(to_combine.front()->sharding());
  }
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller AllGathers with elements of the tuple output
  // of the single bigger AllGather.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return Status::OK();
}

// The group key encapsulates all of the properties which must match for it to
// be possible to combine the instructions.
using GroupKey =
    std::tuple<int64_t, int64_t, bool, bool, std::vector<std::vector<int64_t>>>;

// Returns a key that will be equal for instructions that might be combined, or
// different if not.
absl::optional<GroupKey> CombineKey(const HloInstruction* instruction,
                                    const HloDomainMap& domain_map) {
  if (instruction->opcode() != HloOpcode::kAllGather) {
    return absl::nullopt;
  }

  const auto* ag = Cast<HloAllGatherInstruction>(instruction);

  std::vector<std::vector<int64_t>> replica_groups;
  replica_groups.reserve(ag->replica_groups().size());
  for (const ReplicaGroup& replica_group : ag->replica_groups()) {
    replica_groups.push_back(
        std::vector<int64_t>(replica_group.replica_ids().begin(),
                             replica_group.replica_ids().end()));
  }

  return GroupKey{ag->all_gather_dimension(),
                  domain_map.GetDomainMetadataId(ag),
                  ag->channel_id().has_value(), ag->use_global_device_ids(),
                  replica_groups};
}

}  // namespace

AllGatherCombiner::AllGatherCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc mht_1(mht_1_v, 310, "", "./tensorflow/compiler/xla/service/all_gather_combiner.cc", "AllGatherCombiner::AllGatherCombiner");
}

StatusOr<bool> AllGatherCombiner::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc mht_2(mht_2_v, 315, "", "./tensorflow/compiler/xla/service/all_gather_combiner.cc", "AllGatherCombiner::Run");

  VLOG(1) << "Running AllGatherCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip AllGatherCombiner because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip AllGatherCombiner because the module contains "
               "all-gather with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&domain_map](const HloInstruction* instruction) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combinerDTcc mht_3(mht_3_v, 338, "", "./tensorflow/compiler/xla/service/all_gather_combiner.cc", "lambda");

      return CombineKey(instruction, *domain_map);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<GroupKey>(
            computation, key_fn, &CombineAllGathers,
            combine_threshold_in_bytes_, combine_threshold_count_));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
