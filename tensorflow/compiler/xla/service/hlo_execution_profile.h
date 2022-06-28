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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh() {
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


#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile_data.pb.h"
#include "tensorflow/compiler/xla/service/hlo_profile_printer.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloInstruction;

// Maps all HloInstructions and HloComputations in an HloModule to integers.
// These integers form the contiguous range [0, total_count()).
class HloProfileIndexMap {
 public:
  // Scans `module` to populate this instance of HloProfileIndexMap.
  explicit HloProfileIndexMap(const HloModule& module)
      : HloProfileIndexMap(module, {}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "HloProfileIndexMap");
}
  explicit HloProfileIndexMap(const HloModule& module,
                              absl::Span<const std::string> extra_metrics);

  HloProfileIndexMap(const HloProfileIndexMap&) = default;
  HloProfileIndexMap(HloProfileIndexMap&&) = default;

  HloProfileIndexMap& operator=(const HloProfileIndexMap&) = default;
  HloProfileIndexMap& operator=(HloProfileIndexMap&&) = default;

  size_t GetProfileIndexFor(const HloInstruction& instruction) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "GetProfileIndexFor");

    return FindOrDie(instruction_to_profile_idx(), &instruction);
  }

  size_t GetProfileIndexFor(const HloComputation& computation) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "GetProfileIndexFor");

    return FindOrDie(computation_to_profile_idx(), &computation);
  }

  size_t GetProfileIndexFor(const std::string& key) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_3(mht_3_v, 235, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "GetProfileIndexFor");

    return xla::FindOrDie(extra_metric_to_profile_idx(), key);
  }

  size_t instruction_count() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "instruction_count");

    return instruction_to_profile_idx().size();
  }

  size_t computation_count() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_5(mht_5_v, 249, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "computation_count");

    return computation_to_profile_idx().size();
  }

  size_t extra_metrics_count() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_6(mht_6_v, 256, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "extra_metrics_count");

    return extra_metric_to_profile_idx().size();
  }

  size_t total_count() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_7(mht_7_v, 263, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "total_count");

    return instruction_count() + computation_count() + extra_metrics_count();
  }

  const absl::flat_hash_map<const HloInstruction*, int64_t>&
  instruction_to_profile_idx() const {
    return instruction_to_profile_idx_;
  }

  const absl::flat_hash_map<const HloComputation*, int64_t>&
  computation_to_profile_idx() const {
    return computation_to_profile_idx_;
  }

  const absl::flat_hash_map<std::string, int64_t>& extra_metric_to_profile_idx()
      const {
    return extra_metric_to_profile_idx_;
  }

 private:
  absl::flat_hash_map<const HloInstruction*, int64_t>
      instruction_to_profile_idx_;
  absl::flat_hash_map<const HloComputation*, int64_t>
      computation_to_profile_idx_;
  absl::flat_hash_map<std::string, int64_t> extra_metric_to_profile_idx_;
};

// Create an instance of `HloProfilePrinterData`.
std::unique_ptr<HloProfilePrinterData> CreateHloProfilePrinterData(
    const HloProfileIndexMap& hlo_profile_index_map,
    const HloCostAnalysis& cost_analysis,
    const std::string& entry_computation_name);

// Describes how much time each HLO operation took.
//
// Each HloComputation takes a certain number of cycles.  This class helps break
// down how much time each HLO took.
class HloExecutionProfile {
 public:
  HloExecutionProfile(const HloProfilePrinterData* hlo_profile_printer_data,
                      const HloProfileIndexMap* hlo_profile_index_map);

  // Record how many cycles this HLO took to execute.
  void SetCyclesTakenBy(const HloInstruction* hlo, uint64_t cycles_taken);

  // Record how many cycles this HLO took to execute.
  void SetCyclesTakenBy(size_t index, uint64_t cycles_taken);

  // Returns how many cycles this HLO took to execute.  Profiling information
  // may not be available for some instructions in which case zero is returned.
  uint64_t GetCyclesTakenBy(const HloInstruction& hlo) const;

  // Returns how many cycles this HLO took to execute.  Profiling information
  // may not be available for some instructions in which case zero is returned.
  uint64_t GetCyclesTakenBy(size_t index) const;

  // Return the number of cycles this computation took to execute.
  uint64_t total_cycles_executed(const HloComputation& computation) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_8(mht_8_v, 323, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "total_cycles_executed");

    return profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(
        computation)];
  }

  // Record how many cycles a computation took to execute.
  void set_total_cycles_executed(const HloComputation& computation,
                                 uint64_t total_cycles_executed) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_9(mht_9_v, 333, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "set_total_cycles_executed");

    profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(computation)] =
        total_cycles_executed;
  }

  // Record extra metric.
  void set_extra_metrics(const std::string& metric, uint64_t value) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("metric: \"" + metric + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_10(mht_10_v, 343, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "set_extra_metrics");

    profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(metric)] =
        value;
  }

  // Returns a version of the execution profile suitable for performance
  // debugging; e.g. emits cycle counts, execution time at the nominal device
  // frequency, and the effective throughput given the provided cost_analysis
  // for the operations in a given computation. Returns an empty string if it
  // wasn't possible to generate a printable version.
  std::string ToString(float clock_rate_ghz) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_11(mht_11_v, 356, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "ToString");

    return PrintHloProfile(hlo_profile_printer_data_, profile_counters_.data(),
                           clock_rate_ghz);
  }

  std::vector<int64_t>* mutable_profile_counters() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_12(mht_12_v, 364, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "mutable_profile_counters");

    return &profile_counters_;
  }
  const std::vector<int64_t>& profile_counters() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_execution_profileDTh mht_13(mht_13_v, 370, "", "./tensorflow/compiler/xla/service/hlo_execution_profile.h", "profile_counters");

    return profile_counters_;
  }

  HloExecutionProfileData ToProto() const;

 private:
  const HloProfilePrinterData& hlo_profile_printer_data_;
  const HloProfileIndexMap& hlo_profile_index_map_;

  // Stores per-Hlo profile counters.  This is the only thing that changes when
  // we execute an XLA computation.
  std::vector<int64_t> profile_counters_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_
