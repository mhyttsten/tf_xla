/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh() {
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
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

class HloModule;

// Class representing a sequence of HLO instructions such as the sequential
// execution order of an HLO computation.
class HloInstructionSequence {
 public:
  HloInstructionSequence() = default;
  explicit HloInstructionSequence(
      absl::Span<HloInstruction* const> instructions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "HloInstructionSequence");

    for (HloInstruction* instruction : instructions) {
      push_back(instruction);
    }
  }

  // Adds the instruction to the end of the sequence.
  void push_back(HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "push_back");

    instruction_sequence_.push_back(instruction);
    id_sequence_.push_back(instruction->unique_id());
  }

  // Removes the instruction from the sequence.
  void remove_instruction(HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "remove_instruction");

    auto instruction_it = std::find(instruction_sequence_.begin(),
                                    instruction_sequence_.end(), instruction);
    if (instruction_it != instruction_sequence_.end()) {
      auto id_it = std::find(id_sequence_.begin(), id_sequence_.end(),
                             instruction->unique_id());
      instruction_sequence_.erase(instruction_it);
      id_sequence_.erase(id_it);
    }
  }

  // Replaces the old instruction with the new instruction in the sequence.
  void replace_instruction(HloInstruction* old_instruction,
                           HloInstruction* new_instruction) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_3(mht_3_v, 241, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "replace_instruction");

    auto instruction_it =
        std::find(instruction_sequence_.begin(), instruction_sequence_.end(),
                  old_instruction);
    auto id_it = std::find(id_sequence_.begin(), id_sequence_.end(),
                           old_instruction->unique_id());
    CHECK(instruction_it != instruction_sequence_.end())
        << "Do not find instruction id " << old_instruction->unique_id();
    CHECK(id_it != id_sequence_.end());
    *instruction_it = new_instruction;
    *id_it = new_instruction->unique_id();
  }

  // Clears the sequence of all instructions.
  void clear() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_4(mht_4_v, 258, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "clear");

    instruction_sequence_.clear();
    id_sequence_.clear();
  }

  int64_t size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_5(mht_5_v, 266, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "size");
 return instruction_sequence_.size(); }

  // Returns the sequence of HLO instructions.
  const std::vector<HloInstruction*>& instructions() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_6(mht_6_v, 272, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "instructions");

    return instruction_sequence_;
  }

  // Returns the unique IDs of the instructions in the sequence (in order).
  const std::vector<int>& ids() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_7(mht_7_v, 280, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "ids");
 return id_sequence_; }

 private:
  // The sequence as HloInstructions.
  std::vector<HloInstruction*> instruction_sequence_;

  // The sequence of HLO instructions, represented by their unique IDs. The
  // sequence is stored as both HloInstructions and unique IDs because the
  // sequence may be referenced after transformations to the HLO graph and HLO
  // pointers can be invalidated or recycled in this process (see
  // HloSchedule::Update).
  std::vector<int> id_sequence_;
};

// A class representing a sequential schedule of instructions for an HLO
// module. A complete HLO schedule contains an instruction sequence for every
// non-fusion computation in the HLO module.
class HloSchedule {
 public:
  explicit HloSchedule(const HloModule* module) : module_(module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_8(mht_8_v, 302, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "HloSchedule");
}

  // (De)Serialize an HloSchedule to/from a HloScheduleProto.
  static StatusOr<HloSchedule> CreateFromProto(const HloModule* module,
                                               const HloScheduleProto& proto);
  StatusOr<HloScheduleProto> ToProto() const;

  // Returns a reference to the sequence for the given computation.
  const HloInstructionSequence& sequence(
      const HloComputation* computation) const;

  // Returns the sequence for the given computation. An empty sequence is
  // created if none exists for the computation.
  HloInstructionSequence& GetOrCreateSequence(
      const HloComputation* computation);

  // Sets the sequence for the given computation to the given sequence.
  void set_sequence(const HloComputation* computation,
                    absl::Span<HloInstruction* const> sequence);
  void set_sequence(const HloComputation* computation,
                    HloInstructionSequence sequence);

  // Returns a map from HloComputation unique ID to instruction sequence. The
  // map contains all sequences in the schedule.
  const absl::flat_hash_map<int64_t, HloInstructionSequence>& sequences()
      const {
    return sequences_;
  }

  // Returns true if the schedule has a sequence for the given computation.
  bool is_computation_scheduled(const HloComputation* computation) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_9(mht_9_v, 335, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "is_computation_scheduled");

    return sequences_.contains(computation->unique_id());
  }

  // Removes the computation from the sequences.
  void remove_computation(const HloComputation* computation) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_10(mht_10_v, 343, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "remove_computation");

    auto it = sequences_.find(computation->unique_id());
    CHECK(it != sequences_.end());
    sequences_.erase(it);
  }

  // Removes the instruction from the computation's sequence.
  void remove_instruction(const HloComputation* computation,
                          HloInstruction* instruction) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_11(mht_11_v, 354, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "remove_instruction");

    sequences_[computation->unique_id()].remove_instruction(instruction);
  }

  // Replaces the old instruction with the new instruction in the computation's
  // sequence.
  void replace_instruction(const HloComputation* computation,
                           HloInstruction* old_instruction,
                           HloInstruction* new_instruction) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_12(mht_12_v, 365, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "replace_instruction");

    sequences_[computation->unique_id()].replace_instruction(old_instruction,
                                                             new_instruction);
  }

  // Updates the schedule such that it is (again) a valid schedule for the
  // module. This is used to update a schedule after the HLO module has been
  // transformed in some way. In general, the only transformations to the module
  // for which a schedule can be updated is the addition or removal of
  // instructions and removal of computations. Updating the schedule after new
  // dependencies between existing instructions in the module is not supported
  // and may result in an error status returned.
  //
  // Instructions in the module which also exist in the given schedule will
  // remain in the same order in the updated schedule. Instructions which exist
  // in the module but not in the given schedule will be placed as early as
  // possible in the updated schedule.
  Status Update();

  // Verifies that the given schedule is valid for the given module.
  // Specifically, the schedule contains exactly the instructions in the
  // non-fusion computations in the module and every dependency in the module is
  // satisfied in the schedule.
  Status Verify() const;

  std::string ToString() const;

  bool empty() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_13(mht_13_v, 395, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "empty");
 return sequences_.empty(); }

  const HloModule* module() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_scheduleDTh mht_14(mht_14_v, 400, "", "./tensorflow/compiler/xla/service/hlo_schedule.h", "module");
 return module_; }

 private:
  // Updates the instruction sequence for the given computation.
  Status UpdateComputationSchedule(const HloComputation* computation);

  const HloModule* module_;

  // A map from computation unique ID to instruction sequence. Unique IDs are
  // used rather than HloComputation pointers because HLO pointers are not
  // unique across HLO transformations because pointers may be recycled.
  absl::flat_hash_map<int64_t, HloInstructionSequence> sequences_;
};

std::ostream& operator<<(std::ostream& out, const HloSchedule& schedule);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_
