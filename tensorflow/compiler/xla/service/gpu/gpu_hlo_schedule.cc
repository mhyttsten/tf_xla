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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include <deque>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

namespace {

// An HLO partial ordering based on the actual stream assignment and thunk
// launch order.
class GpuHloOrdering : public PredecessorHloOrdering {
 public:
  GpuHloOrdering(const HloModule* module,
                 const StreamAssignment& stream_assignment,
                 const std::vector<HloInstruction*>& thunk_launch_order);
  ~GpuHloOrdering() override = default;

  // Only the entry computation can possibly be sequentially ordered, and only
  // if we've assigned all instructions to a single stream.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "SequentialOrder");

    return &computation == module_->entry_computation() ? entry_sequence_.get()
                                                        : nullptr;
  }

  std::string ToString() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "ToString");

    return ToStringHelper("GpuHloOrdering");
  }

 private:
  std::unique_ptr<HloInstructionSequence> entry_sequence_;
};

GpuHloOrdering::GpuHloOrdering(
    const HloModule* module, const StreamAssignment& stream_assignment,
    const std::vector<HloInstruction*>& thunk_launch_order)
    : PredecessorHloOrdering(module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "GpuHloOrdering::GpuHloOrdering");

  // The entry computation has a total order when there's only one stream.
  if (stream_assignment.StreamCount() == 1) {
    entry_sequence_ =
        absl::make_unique<HloInstructionSequence>(thunk_launch_order);
  }

  // The ordering of instructions for the entry computation is determined by the
  // total order of thunk launches, and stream assignment. Instructions are
  // sequential within a stream and concurrent across streams. In addition, the
  // GpuExecutable adds cross-stream dependency edges to ensure each instruction
  // waits for its operands before executing.
  //
  // The predecessor map is built incrementally, in thunk launch order. We
  // record the most-recently seen instructions per stream in
  // 'last_instruction_per_stream'. This lets us quickly determine the
  // same-stream predecessors of each instruction.

  // Compute the set of all instructions we will want to set reachability on.
  auto predecessor_map = absl::make_unique<HloReachabilityMap>(
      module->entry_computation()->MakeInstructionPostOrder());

  // The most recently visited instruction per stream.
  std::vector<const HloInstruction*> last_instruction_per_stream(
      stream_assignment.StreamCount(), nullptr);

  for (const HloInstruction* hlo : thunk_launch_order) {
    predecessor_map->SetReachable(hlo, hlo);
    if (stream_assignment.HasStreamAssigned(*hlo)) {
      // Gather all instruction which are immediate predecessors of 'hlo' in the
      // reachability graph.
      std::vector<const HloInstruction*> immediate_preds;
      immediate_preds.insert(immediate_preds.end(), hlo->operands().begin(),
                             hlo->operands().end());
      immediate_preds.insert(immediate_preds.end(),
                             hlo->control_predecessors().begin(),
                             hlo->control_predecessors().end());

      // All ops already queued on the same instruction stream, and their
      // transitive predecessors, are predecessors.
      const int stream_no = stream_assignment.StreamNumberForHlo(*hlo);
      if (last_instruction_per_stream[stream_no] != nullptr) {
        immediate_preds.push_back(last_instruction_per_stream[stream_no]);
      }
      predecessor_map->FastSetReachabilityToUnion(immediate_preds, hlo);
      last_instruction_per_stream[stream_no] = hlo;
    } else {
      // Only parameters and constants don't have an assigned stream, since they
      // don't require a thunk. These ops don't have any predecessors.
      CHECK(hlo->opcode() == HloOpcode::kParameter ||
            hlo->opcode() == HloOpcode::kConstant);
      CHECK_EQ(hlo->operand_count(), 0);
    }
  }
  predecessors_.emplace(module->entry_computation(),
                        std::move(predecessor_map));

  // The ordering of instructions in subcomputations is based solely on control
  // and data dependencies.
  //
  // TODO(toddw): Each subcomputation is actually emitted as a function in DFS
  // postorder, so we can do better and establish the total order here. We don't
  // do that yet since it's hard to ensure that the order here is the order used
  // by IrEmitterNested. And mismatched ordering bugs would be hard to find.
  for (auto* computation : module->computations()) {
    if (computation != module->entry_computation() &&
        !computation->IsFusionComputation()) {
      predecessors_.emplace(computation,
                            HloReachabilityMap::Build(computation));
    }
  }
}

// Computes a topological launch_order that is close to a breadth-first
// order. This heuristic works well for graphs where concurrent kernels are
// located at the same layer. It can often reduce dependency between concurrent
// GEMMs due to intra-stream total orders.  E.g. consider the following HLO
// graph where the numbers in the parens indicate the stream assigned to each
// HLO.
//
//   A(0) -> D(0) -> E(1)
//    |
//    v
//   B(0)
//    |
//    v
//   C(0)
//
// If the total order is A,B,C,D,E, then C and E would be sequentialized
// because C completes before D starts in stream 0, and E depends on D.
// However, if the total order is A,B,D,C,E, then C and E can run
// concurrently.
void BFSLaunchOrder(const HloComputation* computation,
                    std::vector<HloInstruction*>* launch_order) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_3(mht_3_v, 334, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "BFSLaunchOrder");

  // This topological sort uses two data structures:
  // 1. `incoming_edge_count` which keeps track of the number of incoming
  // edges to each HLO;
  // 2. `queue` which contains all HLOs with no incoming edges.
  //
  // The sorting algorithm repeatedly pops the top from the queue and deletes
  // that HLO from the graph, making more HLOs incoming-edge free.
  std::deque<HloInstruction*> queue;
  absl::flat_hash_map<const HloInstruction*, int64_t> incoming_edge_count;
  for (auto* hlo : computation->instructions()) {
    if (hlo->operand_count() == 0) {
      queue.push_back(hlo);
    } else {
      incoming_edge_count[hlo] =
          std::set<HloInstruction*>(hlo->operands().begin(),
                                    hlo->operands().end())
              .size();
    }
  }

  while (!queue.empty()) {
    HloInstruction* x = queue.front();
    queue.pop_front();
    launch_order->push_back(x);
    for (HloInstruction* y : x->users()) {
      --incoming_edge_count[y];
      if (incoming_edge_count[y] == 0) {
        queue.push_back(y);
      }
    }
  }
}

bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_4(mht_4_v, 371, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "ShouldScheduleAsEarlyAsPossible");

  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() ==
             CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return false;
  }
}

bool ShouldScheduleSuccessor(
    const HloInstruction& sussessor,
    const std::function<bool(const HloInstruction*)>& is_scheduled) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_5(mht_5_v, 389, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "ShouldScheduleSuccessor");

  return ShouldScheduleAsEarlyAsPossible(sussessor) &&
         absl::c_all_of(sussessor.operands(), is_scheduled) &&
         absl::c_all_of(sussessor.control_predecessors(), is_scheduled);
}

bool ShouldScheduleAsLateAsPossible(const HloInstruction& instr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_6(mht_6_v, 398, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "ShouldScheduleAsLateAsPossible");

  switch (instr.opcode()) {
    case HloOpcode::kAllReduceDone:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() == CustomCallSchedule::SCHEDULE_LATEST;
    default:
      return false;
  }
}

bool ShouldSchedulePredecessor(
    const HloInstruction& predecessor,
    const std::function<bool(const HloInstruction*)>& is_scheduled) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_7(mht_7_v, 415, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "ShouldSchedulePredecessor");

  return ShouldScheduleAsLateAsPossible(predecessor) &&
         absl::c_all_of(predecessor.users(), is_scheduled) &&
         absl::c_all_of(predecessor.control_successors(), is_scheduled);
}

// Schedules certain ops as early or late as possible. This supports a
// custom-call use case, where a logical operation is lowered into two HLOs
// (e.g., PerformX and PerformXDone). We utilize this mechanism to either hide
// host latencies between the pair of the custom-calls or more accurately
// identify the def-use relationship of the two calls (typically PerformX is
// scheduled right after all of its producers have been scheduled and
// PerformXDone is scheduled right before its first consumer.)
HloInstructionSequence PostprocessorToScheduleAsEarlyOrLateAsPossible(
    const HloInstructionSequence& input) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_8(mht_8_v, 432, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "PostprocessorToScheduleAsEarlyOrLateAsPossible");

  std::vector<HloInstruction*> earliest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_9(mht_9_v, 442, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "lambda");

      earliest_scheduled.push_back(instr);
      scheduled.insert(instr);
    };
    for (HloInstruction* instr : input.instructions()) {
      if (is_scheduled(instr)) {
        continue;
      }

      add_to_schedule(instr);

      // Schedule any successor that should be scheduled as early as possible if
      // all of its producers and control_predecessors have been scheduled.
      for (HloInstruction* user : instr->users()) {
        if (ShouldScheduleSuccessor(*user, is_scheduled)) {
          add_to_schedule(user);
        }
      }
      for (HloInstruction* successor : instr->control_successors()) {
        if (ShouldScheduleSuccessor(*successor, is_scheduled)) {
          add_to_schedule(successor);
        }
      }
    }
  }

  std::deque<HloInstruction*> latest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_10(mht_10_v, 477, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "lambda");

      latest_scheduled.push_front(instr);
      scheduled.insert(instr);
    };
    for (auto it = earliest_scheduled.rbegin(); it != earliest_scheduled.rend();
         it++) {
      if (is_scheduled(*it)) {
        continue;
      }

      add_to_schedule(*it);

      // Schedule any predecessor that should be scheduled as late as possible
      // if all of its users and control_successors have been scheduled.
      for (HloInstruction* operand : (*it)->operands()) {
        if (ShouldSchedulePredecessor(*operand, is_scheduled)) {
          add_to_schedule(operand);
        }
      }
      for (HloInstruction* predecessor : (*it)->control_predecessors()) {
        if (ShouldSchedulePredecessor(*predecessor, is_scheduled)) {
          add_to_schedule(predecessor);
        }
      }
    }
  }

  HloInstructionSequence result;
  absl::c_for_each(latest_scheduled,
                   [&](HloInstruction* i) { result.push_back(i); });
  return result;
}

}  // end namespace

GpuHloSchedule::GpuHloSchedule() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_11(mht_11_v, 515, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "GpuHloSchedule::GpuHloSchedule");
}

/* static */
StatusOr<std::unique_ptr<GpuHloSchedule>> GpuHloSchedule::Build(
    const HloModule* module, const StreamAssignment& stream_assignment,
    int64_t pointer_size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_hlo_scheduleDTcc mht_12(mht_12_v, 523, "", "./tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc", "GpuHloSchedule::Build");

  std::unique_ptr<GpuHloSchedule> schedule(new GpuHloSchedule);

  // Initialize thunk_launch_order_, the total order of thunk launches.
  HloComputation* entry_computation = module->entry_computation();
  if (stream_assignment.StreamCount() == 1) {
    // All kernels are launched on a single stream, so there's no loss of
    // concurrency by optimizing for minimal memory usage.
    TF_ASSIGN_OR_RETURN(
        HloSchedule sequences,
        ScheduleModule(
            module,
            [pointer_size](const BufferValue& buffer) {
              return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
            },
            ComputationSchedulerToModuleScheduler(
                DefaultMemoryScheduler,
                PostprocessorToScheduleAsEarlyOrLateAsPossible)));
    schedule->thunk_launch_order_ =
        sequences.sequence(entry_computation).instructions();
    schedule->hlo_ordering_ =
        absl::make_unique<SequentialHloOrdering>(sequences);
  } else {
    // BFS tends to increase concurrency, but also increases memory usage.
    BFSLaunchOrder(entry_computation, &schedule->thunk_launch_order_);
    schedule->hlo_ordering_ = absl::make_unique<GpuHloOrdering>(
        module, stream_assignment, schedule->thunk_launch_order_);
  }

  return std::move(schedule);
}

}  // namespace gpu
}  // namespace xla
