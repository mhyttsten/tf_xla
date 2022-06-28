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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"

namespace xla {
namespace cpu {

class SimpleCostModel : public ParallelCostModel {
 public:
  SimpleCostModel(const int64_t max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "SimpleCostModel");
}
  ~SimpleCostModel() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "~SimpleCostModel");
}

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_2(mht_2_v, 213, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "GetParallelTaskCount");

    // Simple cost model based on hlo size and typical L2 cache size.
    const int64_t instruction_cost = shape_size_(instruction->shape());
    const int64_t min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism_,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(const int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        shape_size_(shape_size),
        cost_analysis_(std::move(cost_analysis)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "DefaultCostModel");
}
  ~DefaultCostModel() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "~DefaultCostModel");
}

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_5(mht_5_v, 247, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "GetParallelTaskCount");

    // Parameters for parallel task count computation.
    int64_t instruction_cost;
    int64_t min_cost_per_thread;
    int64_t max_parallelism;
    // Calculate flops-to-bytes-ratio for 'instruction'.
    const int64_t bytes_accessed =
        std::max(int64_t{1}, cost_analysis_->bytes_accessed(*instruction));
    const float flops_to_bytes_ratio =
        cost_analysis_->flop_count(*instruction) /
        static_cast<float>(bytes_accessed);
    // Check for I/O bound instructions.
    if (flops_to_bytes_ratio <= 1.0) {
      // Limit max parallelism for I/O bound instructions by assuming a
      // sub-linear scaling function (fit based on empirical benchmark results).
      // TODO(b/29630486) Develop system bandwidth model.
      max_parallelism = std::min<int64_t>(
          max_parallelism_,
          std::ceil(std::sqrt(tensorflow::port::MaxParallelism())));
      // Use shape size instruction cost and L2 cache size min per-thread cost.
      instruction_cost = shape_size_(instruction->shape());
      min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    } else {
      // Use max parallelism for compute bound instructions.
      max_parallelism = max_parallelism_;
      // Calculate the instruction cost in cycles.
      // TODO(b/29630486) Improve on this linear cost model.
      // Consider making 'min_cost_per_thread' be a function of the target
      // bandwidth limit for instructions with low arithmetic complexity.
      instruction_cost =
          1 * cost_analysis_->flop_count(*instruction) +
          2 * cost_analysis_->transcendental_count(*instruction) +
          10 * cost_analysis_->bytes_accessed(*instruction);
      // Minimum per-thread cost is 100us of work on a 2GHz core.
      min_cost_per_thread = 100000;
    }
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};

ParallelTaskAssignment::ParallelTaskAssignment(
    const int64_t max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module,
    const TargetMachineFeatures* target_machine_features)
    : target_machine_features_(*target_machine_features) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_6(mht_6_v, 302, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssignment::ParallelTaskAssignment");

  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
  // Run cost analysis on 'module'.
  auto cost_analysis = absl::make_unique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  Status status = computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    // Set default cost model based on 'cost_analysis'.
    cost_model_.reset(new DefaultCostModel(max_parallelism, shape_size,
                                           std::move(cost_analysis)));
  } else {
    // Fall back to a simple cost model based on hlo size and L2 cache size.
    // Note that HloCostAnalysis can returns an error status (likely because
    // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    cost_model_.reset(new SimpleCostModel(max_parallelism, shape_size));
  }
}

int64_t ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_7(mht_7_v, 324, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssignment::GetTargetParallelTaskCount");

  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kConv, kDot, kFft, kCustomCall).
  // *) Emit custom loops (kSelectAndScatter).
  // *) Operations that are not thread safe (like infeed and rng).
  // *) Tuple-shaped.
  // *) Operations that might be implemented as an in-place
  //    dynamic-update-slice, because we can't know how many output elements
  //    they will write (out-of-place will touch the whole output buffer, while
  //    in-place will only touch the updated elements).
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  auto opcode = instruction->opcode();
  if (llvm_ir::MayBeImplementedAsInPlaceDynamicUpdateSlice(instruction) ||
      instruction->shape().IsTuple() || opcode == HloOpcode::kRng ||
      opcode == HloOpcode::kConstant) {
    return 1;
  }

  // Only allow instructions that can be trivially parallelized (where all
  // outputs can be computed independently of each other).
  if (instruction->IsElementwise() || instruction->IsLoopFusion() ||
      opcode == HloOpcode::kBroadcast || opcode == HloOpcode::kConcatenate ||
      opcode == HloOpcode::kDynamicSlice ||
      opcode == HloOpcode::kDynamicUpdateSlice ||
      opcode == HloOpcode::kGather || opcode == HloOpcode::kIota ||
      opcode == HloOpcode::kPad || opcode == HloOpcode::kReduce ||
      opcode == HloOpcode::kReduceWindow || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kReverse || opcode == HloOpcode::kSlice ||
      opcode == HloOpcode::kTranspose ||
      (opcode == HloOpcode::kConvolution &&
       !PotentiallyImplementedAsEigenConvolution(*instruction,
                                                 target_machine_features_))) {
    // Consult 'cost_model_' to compute target parallel task count.
    return cost_model_->GetParallelTaskCount(instruction);
  }

  return 1;
}

StatusOr<bool> ParallelTaskAssigner::Run(HloModule* module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_8(mht_8_v, 367, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssigner::Run");

  XLA_VLOG_LINES(2, "ParallelTaskAssigner ENTRY");
  XLA_VLOG_LINES(3, module->ToString());
  // Compute target parallel task counts for all instructions in 'module'.
  HloToParallelTasks hlo_to_parallel_tasks;
  ComputeTargetParallelTasks(module, &hlo_to_parallel_tasks);

  // Assign parallel tasks to target specific instructions in 'module'.
  // TODO(b/27458679) Support inter-op parallelism.
  bool changed = AssignParallelTasks(module, hlo_to_parallel_tasks);

  XLA_VLOG_LINES(2, "ParallelTaskAssigner EXIT");
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

bool ParallelTaskAssigner::AssignParallelTasks(
    HloModule* module, const HloToParallelTasks& hlo_to_parallel_tasks) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_9(mht_9_v, 387, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssigner::AssignParallelTasks");

  return AssignParallelTasksHelper(module, module->entry_computation(),
                                   hlo_to_parallel_tasks);
}

bool ParallelTaskAssigner::AssignParallelTasksHelper(
    HloModule* module, HloComputation* computation,
    const HloToParallelTasks& hlo_to_parallel_tasks) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_10(mht_10_v, 397, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssigner::AssignParallelTasksHelper");

  bool changed = false;
  // Snapshot set of instructions because outlining modifies the set below.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  for (auto* instruction : instructions) {
    // Assign parallel tasks to sub-computations for While and Call HLOs.
    // TODO(b/27458679) Evaluate alternative intra-op parallelism placement,
    // and support other callable computations like reduce.
    if (instruction->opcode() == HloOpcode::kWhile) {
      changed |= AssignParallelTasksHelper(module, instruction->while_body(),
                                           hlo_to_parallel_tasks);
      continue;
    } else if (instruction->opcode() == HloOpcode::kCall) {
      changed |= AssignParallelTasksHelper(module, instruction->to_apply(),
                                           hlo_to_parallel_tasks);
      continue;
    }
    // Skip if no parallel tasks were computed in first pass.
    auto it = hlo_to_parallel_tasks.find(instruction);
    if (it == hlo_to_parallel_tasks.end()) {
      continue;
    }
    // Get target parallel task count computed for 'instruction'.
    const int64_t target_parallel_task_count = (*it).second;
    // Assign feasible dimension partitions (based on actual dimension sizes).
    auto dim_partition_counts = ShapePartitionAssigner(instruction->shape())
                                    .Run(target_parallel_task_count);
    const int64_t total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    if (total_partition_count <= 1) {
      // Feasible partition calculation resulting in no partitioning, so skip.
      continue;
    }

    // Outline 'instruction' in 'computation' for parallel task assignment.
    auto* call = module->OutlineExpressionFromComputation(
        {instruction}, absl::StrCat("parallel_", instruction->name()),
        computation);

    // Set assigned dimension partitioning to 'instruction'.
    auto* new_root = call->to_apply()->root_instruction();
    new_root->set_outer_dimension_partitions(dim_partition_counts);

    VLOG(2) << "Assigned parallel task count: " << total_partition_count
            << " to instruction: " << new_root->name()
            << " parent: " << new_root->parent()->name();
    changed = true;
  }
  return changed;
}

void ParallelTaskAssigner::ComputeTargetParallelTasks(
    HloModule* module, HloToParallelTasks* hlo_to_parallel_tasks) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignmentDTcc mht_11(mht_11_v, 453, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment.cc", "ParallelTaskAssigner::ComputeTargetParallelTasks");

  ParallelTaskAssignment parallel_task_assignment(max_parallelism_,
                                                  shape_size_function_, module,
                                                  &target_machine_features_);

  // Compute parallel task counts for all instructions in 'module'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      // Query ParallelTaskAssignment for target parallel task count.
      const int64_t target_parallel_task_count =
          parallel_task_assignment.GetTargetParallelTaskCount(instruction);
      if (target_parallel_task_count > 1) {
        hlo_to_parallel_tasks->insert(
            {instruction, target_parallel_task_count});
      }
    }
  }
}

}  // namespace cpu
}  // namespace xla
