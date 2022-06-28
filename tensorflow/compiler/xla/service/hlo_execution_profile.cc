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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile_data.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
HloProfileIndexMap::HloProfileIndexMap(
    const HloModule& module, absl::Span<const std::string> extra_metrics) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloProfileIndexMap::HloProfileIndexMap");

  size_t current_profile_index = 0;
  for (xla::HloComputation* computation : module.MakeComputationPostOrder()) {
    InsertOrDie(&computation_to_profile_idx_, computation,
                current_profile_index++);
    for (const HloInstruction* instruction : computation->instructions()) {
      // For simplicity we track all instructions here, but we could skip
      // non-executing instructions like constants and parameters.
      InsertOrDie(&instruction_to_profile_idx_, instruction,
                  current_profile_index++);
    }
  }
  for (const std::string& key : extra_metrics) {
    InsertOrDie(&extra_metric_to_profile_idx_, key, current_profile_index++);
  }
}

std::unique_ptr<HloProfilePrinterData> CreateHloProfilePrinterData(
    const HloProfileIndexMap& hlo_profile_index_map,
    const HloCostAnalysis& cost_analysis,
    const std::string& entry_computation_name) {
  using HloComputationInfo = HloProfilePrinterData::HloComputationInfo;
  using HloInstructionInfo = HloProfilePrinterData::HloInstructionInfo;

  size_t profile_counters_size = hlo_profile_index_map.total_count();

  std::unique_ptr<HloProfilePrinterData> profile_printer_data =
      absl::make_unique<HloProfilePrinterData>();
  profile_printer_data->set_profile_counters_size(profile_counters_size);
  profile_printer_data->mutable_computation_infos()->Reserve(
      hlo_profile_index_map.computation_count());

  const auto& computation_to_profile_idx_map =
      hlo_profile_index_map.computation_to_profile_idx();

  // computation_to_profile_idx_map's order is not deterministic so create a
  // deterministic computation_and_profile_idx_list so that we end up with a
  // deterministic HloProfilePrinterData protobuf.

  std::vector<std::pair<const HloComputation*, int64_t>>
      computation_and_profile_idx_list(computation_to_profile_idx_map.begin(),
                                       computation_to_profile_idx_map.end());

  // The profile indices were computed deterministically in
  // HloProfileIndexMap::HloProfileIndexMap.
  absl::c_sort(computation_and_profile_idx_list,
               [](const std::pair<const HloComputation*, int64_t>& left,
                  const std::pair<const HloComputation*, int64_t>& right) {
                 return left.second < right.second;
               });

  for (const auto& pair : computation_and_profile_idx_list) {
    CHECK_LT(pair.second, profile_counters_size);
    const HloComputation* computation = pair.first;
    HloComputationInfo* computation_info =
        profile_printer_data->add_computation_infos();

    computation_info->set_name(computation->name());
    computation_info->set_profile_index(pair.second);
    computation_info->mutable_instruction_infos()->Reserve(
        computation->instruction_count());

    for (const HloInstruction* hlo : computation->instructions()) {
      HloInstructionInfo* instruction_info =
          computation_info->add_instruction_infos();
      instruction_info->set_long_name(hlo->ToString());
      instruction_info->set_short_name(hlo->ToString(
          HloPrintOptions().set_compact_operands(true).set_print_operand_names(
              false)));
      instruction_info->set_category(hlo->ToCategory());
      instruction_info->set_flop_count(cost_analysis.flop_count(*hlo));
      instruction_info->set_transcendental_count(
          cost_analysis.transcendental_count(*hlo));
      instruction_info->set_bytes_accessed(cost_analysis.bytes_accessed(*hlo));
      instruction_info->set_optimal_seconds(
          cost_analysis.optimal_seconds(*hlo));
      instruction_info->set_profile_index(
          hlo_profile_index_map.GetProfileIndexFor(*hlo));
    }
  }

  // Add extra metrics if any.
  for (const auto& pair : hlo_profile_index_map.extra_metric_to_profile_idx()) {
    profile_printer_data->mutable_extra_metrics()->insert(
        {pair.first, pair.second});
  }

  profile_printer_data->set_entry_computation(entry_computation_name);

  return profile_printer_data;
}

HloExecutionProfile::HloExecutionProfile(
    const HloProfilePrinterData* hlo_profile_printer_data,
    const HloProfileIndexMap* hlo_profile_index_map)
    : hlo_profile_printer_data_(*hlo_profile_printer_data),
      hlo_profile_index_map_(*hlo_profile_index_map),
      profile_counters_(
          /*count=*/hlo_profile_index_map_.total_count(),
          /*value=*/0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_1(mht_1_v, 304, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::HloExecutionProfile");
}

void HloExecutionProfile::SetCyclesTakenBy(const HloInstruction* hlo,
                                           uint64_t cycles_taken) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_2(mht_2_v, 310, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::SetCyclesTakenBy");

  SetCyclesTakenBy(hlo_profile_index_map_.GetProfileIndexFor(*hlo),
                   cycles_taken);
}

void HloExecutionProfile::SetCyclesTakenBy(size_t index,
                                           uint64_t cycles_taken) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_3(mht_3_v, 319, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::SetCyclesTakenBy");

  profile_counters_[index] = cycles_taken;
}

uint64_t HloExecutionProfile::GetCyclesTakenBy(
    const HloInstruction& hlo) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_4(mht_4_v, 327, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::GetCyclesTakenBy");

  return GetCyclesTakenBy(hlo_profile_index_map_.GetProfileIndexFor(hlo));
}

uint64_t HloExecutionProfile::GetCyclesTakenBy(size_t index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::GetCyclesTakenBy");

  return profile_counters_[index];
}

HloExecutionProfileData HloExecutionProfile::ToProto() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTcc mht_6(mht_6_v, 341, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.cc", "HloExecutionProfile::ToProto");

  HloExecutionProfileData hlo_execution_profile_data;
  for (const auto& counter : profile_counters_) {
    hlo_execution_profile_data.add_profile_counters(counter);
  }
  *(hlo_execution_profile_data.mutable_printer_data()) =
      hlo_profile_printer_data_;
  return hlo_execution_profile_data;
}

}  // namespace xla
