/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh() {
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
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Postprocessor of the HloInstructionSequence. This is an opt-in postprocessing
// function to MemorySchedulerAlgorithm to enforce certain hlo schedule
// constraints desired for custom-calls.
using MemorySchedulerPostprocessor =
    std::function<HloInstructionSequence(const HloInstructionSequence&)>;

// A memory scheduler computes an execution sequence for the HLO instructions in
// 'computation' that minimizes peak memory, given a points-to analysis result
// that describes buffer aliasing, together with a target-specific size function
// that maps a tensor's logical size to its padded size. peak_memory (may be
// nullptr) is set to the peak memory of the resulting schedule according to the
// HeapSimulator.
//
// TODO(yunxing): Cleanup usage of TuplePointsToAnalysis.
typedef std::function<StatusOr<HloInstructionSequence>(
    HloComputation*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
    const LogicalBuffer::SizeFunction&,
    const absl::flat_hash_map<const HloComputation*, int64_t>&,
    const MemorySchedulerPostprocessor&,
    /*peak_memory*/ int64_t*)>
    MemorySchedulerAlgorithm;

// Scheduler for the entire module.
typedef std::function<StatusOr<HloSchedule>(
    const HloModule*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
    const LogicalBuffer::SizeFunction&,
    /*peak_memory*/ int64_t*)>
    ModuleSchedulerAlgorithm;

// Lift a computation scheduler into a module scheduler by calling the
// computation scheduler on all computations in a module.
ModuleSchedulerAlgorithm ComputationSchedulerToModuleScheduler(
    const MemorySchedulerAlgorithm&, const MemorySchedulerPostprocessor& = {});

// List scheduler
StatusOr<HloInstructionSequence> ListMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// DFS-order scheduler
StatusOr<HloInstructionSequence> DFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// Naive Post Order scheduler
StatusOr<HloInstructionSequence> PostOrderMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// The default scheduling algorithm. Runs the list scheduler, the DFS scheduler,
// and the post-order scheduler and chooses whichever returns a lower min-
// memory, not accounting for fragmentation. peak_memory (may be nullptr) is set
// to the peak memory of the resulting schedule according to the HeapSimulator.
StatusOr<HloInstructionSequence> DefaultMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

StatusOr<HloSchedule> DefaultModuleScheduler(
    const HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function, int64_t* peak_memory);

// Returns an HloSchedule which seeks to minimize the memory required for the
// module. size_function is the function returning the number of bytes required
// for a LogicalBuffer. peak_memory (if not nullptr) is set to the largest peak
// memory (according to the HeapSimulator) of all computations in the module.
StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const LogicalBuffer::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm = {},
    int64_t* peak_memory = nullptr);

// Computes the schedule for a single computation.
// Currently only used by the GPU backend.
StatusOr<HloInstructionSequence> ScheduleComputation(
    HloComputation* computation,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor);

// A pass which schedules the HLO instructions in a module. The HloModule's
// schedule field is set to the resulting HloSchedule using
// HloModule::set_schedule.
class HloMemoryScheduler : public HloModulePass {
 public:
  // size_function is the function returning the number of bytes required for a
  // LogicalBuffer. algorithm is the memory scheduling algorithm to use. If not
  // specified, then DefaultMemoryScheduler is used.
  HloMemoryScheduler(const LogicalBuffer::SizeFunction& size_function,
                     const ModuleSchedulerAlgorithm& algorithm = {});

  ~HloMemoryScheduler() override = default;

  absl::string_view name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh mht_0(mht_0_v, 315, "", "./tensorflow/compiler/xla/service/hlo_memory_scheduler.h", "name");
 return "hlo-memory-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  LogicalBuffer::SizeFunction size_function_;

  ModuleSchedulerAlgorithm algorithm_;
};

// A pass which produces a naive, but correct schedule. The schedule is produced
// using a DFS traversal of the graph with no attempt to minimize memory use.
class HloTrivialScheduler : public HloModulePass {
 public:
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh mht_1(mht_1_v, 332, "", "./tensorflow/compiler/xla/service/hlo_memory_scheduler.h", "name");
 return "hlo-trivial-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;
};

// A trivial pass which clears the schedule currently set on the
// HloModule. After this pass runs HloModule::has_schedule will return false.
class HloDescheduler : public HloModulePass {
 public:
  HloDescheduler() = default;
  ~HloDescheduler() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_memory_schedulerDTh mht_2(mht_2_v, 346, "", "./tensorflow/compiler/xla/service/hlo_memory_scheduler.h", "name");
 return "hlo-descheduler"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
