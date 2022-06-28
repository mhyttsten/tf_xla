/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations under
the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh() {
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


#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Class which computes live range of the output buffers of HLOs and their
// interference by flattening all computations. The live range is only available
// when all global computations (while, if, call, etc) have total order
// sequential orders.
class HloLiveRange {
 public:
  // Constructs a hlo live range object for the given module and computation
  // assuming the given HLO instruction ordering.
  static StatusOr<std::unique_ptr<HloLiveRange>> Run(
      const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
      const HloComputation* computation, bool module_scoped_analysis = true);

  // LogicalTime represents the time in a virtual clock. Each instruction has
  // one monotonically increasing logical time assigned according to the
  // schedule.
  using LogicalTime = int64_t;

  struct TimeBound {
    LogicalTime start;
    LogicalTime end;
    // The buffer can hold multiple instructions during its life time (each
    // tenant exclusively owns the buffer at any given time). `end_instruction`
    // represents the last instruction that the buffer holds.
    HloPosition end_position;

    bool friend operator==(const TimeBound& a, const TimeBound& b) {
      return a.start == b.start && a.end == b.end;
    }
    bool friend operator!=(const TimeBound& a, const TimeBound& b) {
      return !(a == b);
    }
  };

  std::string ToString() const;

  const HloInstructionSequence& flattened_instruction_sequence() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh mht_0(mht_0_v, 242, "", "./tensorflow/compiler/xla/service/hlo_live_range.h", "flattened_instruction_sequence");

    return flattened_instruction_sequence_;
  }

  // Returns the map from instruction to the end time of that instruction.
  const absl::flat_hash_map<const HloInstruction*, LogicalTime>&
  instruction_schedule() const {
    return instruction_schedule_;
  }

  // Returns the map from a hlo value to the definition time of that hlo value.
  const absl::flat_hash_map<const HloValue*, TimeBound>& buffer_live_ranges()
      const {
    return buffer_live_ranges_;
  }

  absl::flat_hash_map<const HloValue*, TimeBound>& buffer_live_ranges() {
    return buffer_live_ranges_;
  }

  // Returns the map from a computation and its time span in the schedule.
  const absl::flat_hash_map<const HloComputation*, TimeBound>&
  computation_span_times() const {
    return computation_span_times_;
  }

  // Returns the time stamp of the end of the program.
  LogicalTime schedule_end_time() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh mht_1(mht_1_v, 272, "", "./tensorflow/compiler/xla/service/hlo_live_range.h", "schedule_end_time");

    return flattened_instruction_sequence_.size();
  }

  // Returns whether hlo live range is available on this entire module. Hlo live
  // range is not available if the module is partially ordered.
  bool total_order_scheduled() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh mht_2(mht_2_v, 281, "", "./tensorflow/compiler/xla/service/hlo_live_range.h", "total_order_scheduled");
 return total_order_scheduled_; }

 private:
  explicit HloLiveRange(const HloSchedule& schedule,
                        const HloAliasAnalysis& alias_analysis,
                        bool module_scoped_analysis)
      : schedule_(schedule),
        alias_analysis_(alias_analysis),
        module_scoped_analysis_(module_scoped_analysis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_live_rangeDTh mht_3(mht_3_v, 292, "", "./tensorflow/compiler/xla/service/hlo_live_range.h", "HloLiveRange");
}

  // FlattenSchedule walks through the instructions in `computation`, and
  // recurse into each called computations in module_scoped_analysis mode. As it
  // walks it also tracks down the ordinal number of each instruction in the
  // schedule and store it in the `instruction_schedule` and
  // 'flattened_instruction_sequence`.
  void FlattenSchedule(const HloComputation& computation);

  // Returns the last position of a value.
  TimeBound GetLastPosition(const HloValue& value,
                            LogicalTime definition_end_time) const;

  // Returns the time of the last use of a value.
  LogicalTime GetLastUsageTime(const HloValue& value) const;

  // Based on the flattened schedule, calculate the start and end of each
  // buffer.
  void CalculateBufferStartEndMap();

  // The aliased buffers could have overlapping live ranges.
  // NormalizeAliasedBuffers normalizes the buffer such that each alias buffer
  // has disjoint live range while keeping the live range union the same. This
  // avoid double counting aliased buffer sizes.
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----+          live range of buffer1
  //   +------------------+    live range of buffer2
  //
  // After:
  //
  //           +----------+    live range of buffer1
  //   +------+                live range of buffer2
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----------+    live range of buffer1
  //   +------------+          live range of buffer2
  //
  // After:
  //
  //           +----------+    live range of buffer1
  //   +------+                live range of buffer2
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----------+    live range of buffer1
  //   +---+                   live range of buffer2
  //
  // After(unchanged):
  //
  //           +----------+    live range of buffer1
  //   +---+                   live range of buffer2
  //
  // As another example, imagine we have the following code sequence with live
  // ranges of each while-aliased buffers:
  //
  //                     a      p1    p2    e     b
  // a = ...             +
  //                     |
  // {                   |
  //   p1 = param        |       +
  //   ROOT true         |       |
  // }                   |       +
  // { // body           |
  //   p2 = param        +             +
  //   c = p2 + 1                      +
  //   d = c + 1
  //   ROOT e = d + 1                       +
  // }                                      |
  //                                        |
  // b = while (a)                          +     +
  //                                              |
  // f = b + 1                                    +
  //
  // After normalization it becomes:
  //
  //                     a      p1    p2    e     b
  // a = ...             +
  //                     |
  // {                   +
  //   p1 = param                +
  //   ROOT true                 |
  // }                           +
  // { // body
  //   p2 = param                      +
  //   c = p2 + 1                      +
  //   d = c + 1
  //   ROOT e = d + 1                       +
  // }                                      |
  //                                        |
  // b = while (a)                          +
  //                                              +
  // f = b + 1                                    +
  //
  // Note there is no overlap of live ranges after normalization.
  void NormalizeAliasedBuffers();

  LogicalTime ComputePeakMemoryMoment() const;

  const HloSchedule& schedule_;
  const HloAliasAnalysis& alias_analysis_;
  bool module_scoped_analysis_;
  bool total_order_scheduled_ = true;

  HloInstructionSequence flattened_instruction_sequence_;
  absl::flat_hash_map<const HloInstruction*, LogicalTime> instruction_schedule_;
  absl::flat_hash_map<const HloComputation*, TimeBound> computation_span_times_;
  absl::flat_hash_map<const HloValue*, TimeBound> buffer_live_ranges_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_
