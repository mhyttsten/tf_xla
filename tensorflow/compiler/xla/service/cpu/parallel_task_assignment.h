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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace cpu {

// Simple interface for different parallel cost model implementations.
class ParallelCostModel {
 public:
  virtual ~ParallelCostModel() = default;
  virtual int64_t GetParallelTaskCount(HloInstruction* instruction) = 0;
};

// ParallelTaskAssignment computes parallel task counts for HLOs in 'module'.
class ParallelTaskAssignment {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  // 'module': the containing HloModule.
  ParallelTaskAssignment(const int64_t max_parallelism,
                         const HloCostAnalysis::ShapeSizeFunction& shape_size,
                         HloModule* module,
                         const TargetMachineFeatures* target_machine_features);
  ~ParallelTaskAssignment() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h", "~ParallelTaskAssignment");
}

  // Computes and returns the target parallel task count for 'instruction'.
  int64_t GetTargetParallelTaskCount(HloInstruction* instruction);

 private:
  std::unique_ptr<ParallelCostModel> cost_model_;
  const TargetMachineFeatures& target_machine_features_;
};

// ParallelTaskAssigner computes target parallel task counts for all HLOs
// in the module, then assigns parallel task counts to HLOs in the entry
// computation, or to HLOs in embedded computations invoked by (potentially
// nested) kWhile or kCall instructions.
// Each HLO which is assigned parallel task counts is outlined into its
// own embedded computation, which is compiled as a parallel compute function,
// and which is invoked from a kCall instruction that is lowered in codegen to
// a runtime parallel fork/join call.
class ParallelTaskAssigner : public HloModulePass {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  ParallelTaskAssigner(const int64_t max_parallelism,
                       const HloCostAnalysis::ShapeSizeFunction& shape_size,
                       const TargetMachineFeatures* target_machine_features)
      : max_parallelism_(max_parallelism),
        shape_size_function_(shape_size),
        target_machine_features_(*target_machine_features) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h", "ParallelTaskAssigner");
}
  ~ParallelTaskAssigner() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h", "~ParallelTaskAssigner");
}

  absl::string_view name() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTh mht_3(mht_3_v, 255, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h", "name");

    return "cpu-parallel-task-assigner";
  }

  // Run parallel task assigner on 'module'.
  // Returns true if the computation was changed, false otherwise.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  using HloToParallelTasks =
      absl::flat_hash_map<const HloInstruction*, int64_t>;

  // Assigns target parallel tasks from 'hlo_to_parallel_tasks' to HLOs in
  // 'module'.
  // Returns true if the computation was changed, false otherwise.
  bool AssignParallelTasks(HloModule* module,
                           const HloToParallelTasks& hlo_to_parallel_tasks);
  bool AssignParallelTasksHelper(
      HloModule* module, HloComputation* computation,
      const HloToParallelTasks& hlo_to_parallel_tasks);

  // Computes target parallel task counts (returned in 'parallel_task_counts')
  // for parallelizable instructions in 'module'.
  void ComputeTargetParallelTasks(HloModule* module,
                                  HloToParallelTasks* hlo_to_parallel_tasks);

  int64_t max_parallelism_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
  const TargetMachineFeatures& target_machine_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
