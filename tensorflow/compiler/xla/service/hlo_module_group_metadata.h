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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh() {
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
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Class for bookkeeping the information on the given modules, in particular on
// the interaction between computations.
//
// Companion instructions are one piece of information collected as we build the
// metadata. For example, for each While instruction, companion instructions
// refer to a set of While instructions in other computations that communicate
// with each other.
// In the example below with 3 modules, {While_0, While_2, While_5}, {While_1,
// While_4}, {While_3, While_6} are companion sets.
//
// <Module 0>               <Module 1>                 <Module 2>
// While_0() {              While_2() {                While_5() {
//   While_1() { Send(0) }    While_3() { Send(1) }      While_6() { Recv(1) }
// }                          While_4() { Recv(0) }
//                          }
//
// Each instruction can belong to at most one companion set: While_0 and While_5
// are in the same set even though they don't communicate with each other,
// because they both communicate with While_2.
//
// A send and the matching recv must both have the same level of nesting of
// companion instructions.
//
// Companion instructions are used to detect cycles in the graph and also for
// global scheduling.
class HloModuleGroupMetadata {
 public:
  // The kind of companion computation a given instruction can be within.
  enum class ComputationKind {
    kInvalid,
    kWhileCondition,
    kWhileBody,
    kConditionalBranch,
    kCallFunction,
  };

  // Tracks the instruction mapped to a given computation, and the computation
  // kind.
  // For example, a body computation of a while instruction, will generate a
  // TrackedInstruction with instruction being the while instruction, and
  // kind being ComputationKind::kWhileBody.
  class TrackedInstruction {
   public:
    TrackedInstruction() = default;
    TrackedInstruction(HloInstruction* instruction, ComputationKind kind,
                       int index = -1)
        : instruction_(instruction), kind_(kind), index_(index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_0(mht_0_v, 251, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "TrackedInstruction");
}

    bool operator==(const TrackedInstruction& rhs) const {
      return instruction_->opcode() == rhs.instruction_->opcode() &&
             kind_ == rhs.kind_ && index_ == rhs.index_;
    }
    bool operator!=(const TrackedInstruction& rhs) const {
      return !operator==(rhs);
    }

    HloInstruction* instruction() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_1(mht_1_v, 264, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "instruction");
 return instruction_; }

    std::string ToString() const;

   private:
    HloInstruction* instruction_ = nullptr;
    ComputationKind kind_ = ComputationKind::kInvalid;
    int index_ = -1;
  };

  // Represents a channel and the instructions that form the channel.
  struct Channel {
    int64_t id = -1;
    HloInstruction* send = nullptr;
    HloInstruction* recv = nullptr;
    HloInstruction* send_done = nullptr;
    HloInstruction* recv_done = nullptr;
  };

  explicit HloModuleGroupMetadata(absl::Span<HloModule* const> modules)
      : modules_(modules.begin(), modules.end()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_2(mht_2_v, 287, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "HloModuleGroupMetadata");
}

  ~HloModuleGroupMetadata() = default;

  // Build and return the metadata for the given modules.
  static StatusOr<std::unique_ptr<HloModuleGroupMetadata>> Build(
      absl::Span<HloModule* const> modules);

  // Returns true if the instruction is one of the 4 channel instructions (Send,
  // Recv, SendDone, RecvDone).
  bool IsChannelInstruction(const HloInstruction* instruction) const;

  // Returns true if the instruction is a companion instruction. See the class
  // comment above on companion instructions.
  bool IsCompanionInstruction(HloInstruction* hlo) const;

  // Returns true if the instruction is either a channel instruction, a
  // cross-module all-reduce instruction, or a companion instruction.
  bool InstructionCommunicates(HloInstruction* hlo) const;

  // Returns the Channel instance for the given channel id.
  const Channel& GetChannel(int64_t channel_id) const;

  // Returns if the given channel id exists in metadata.
  bool HasChannel(int64_t channel_id) const;

  // Returns the all-reduce instructions with the same channel_id.
  const std::vector<HloInstruction*>& GetAllReduceGroup(
      int64_t channel_id) const;

  // Returns the computation that contains the peer channel instructions for
  // the given instruction.
  //
  // Precondition: IsChannelInstruction(instruction) is true.
  HloComputation* PeerComputation(const HloInstruction* instruction) const;

  // Returns the path of the nested companion instructions, in terms of HLO
  // instructions. The path goes from inner to outer companions.
  // The returned path does not include the input hlo instruction, in case it
  // is a companion instruction.
  std::vector<TrackedInstruction> GetCompanionsPath(
      const HloInstruction* hlo) const;

  // Checks whether two companion paths (as returned by the GetCompanionsPath()
  // API) are compatible. The two paths are compatible if the sequence of
  // opcodes, and the companion kinds, of the two paths matches.
  bool CheckCompanionPathsCompatibility(
      const std::vector<TrackedInstruction>& path0,
      const std::vector<TrackedInstruction>& path1) const;

  // Returns the unique integer for each module. The returned id is the index of
  // the module in the module vector.
  int64_t GetModuleId(const HloModule* module) const;

  // Retrieves the device an instruction is assigned to. Either from the
  // sharding information, or from the ordinal of the module the instruction
  // is in.
  absl::optional<int64_t> GetInstructionDevice(
      const HloInstruction& instruction) const;

  // Returns the number of modules for devices (excluding the host module).
  int64_t GetDeviceModulesCount() const;

  // Returns the companion set for the given instruction, including the
  // instruction itself.
  //
  // Precondition: IsCompanionWhile(instruction) is true.
  const std::vector<HloInstruction*>& Companions(
      const HloInstruction* instruction) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_3(mht_3_v, 358, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "Companions");

    CHECK(companion_set_index_.contains(instruction));
    return companion_set(companion_set_index_.at(instruction));
  }

  // Returns the companion set at the given index.
  const std::vector<HloInstruction*>& companion_set(int64_t index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_4(mht_4_v, 367, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "companion_set");

    CHECK_LT(index, companion_sets_.size());
    return *companion_sets_[index];
  }

  // Returns the companion set index of the given instruction.
  int64_t companion_set_index(HloInstruction* instruction) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_5(mht_5_v, 376, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "companion_set_index");

    return companion_set_index_.at(instruction);
  }

  // Returns the list of all companion sets in the HLO module group. Each
  // returned set contains at least one HloInstruction.
  const std::vector<std::unique_ptr<std::vector<HloInstruction*>>>&
  companion_sets() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_6(mht_6_v, 386, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "companion_sets");

    return companion_sets_;
  }

  // Returns all channels in the module group.
  const std::vector<Channel>& channels() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_7(mht_7_v, 394, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "channels");
 return channels_; }

  // Returns the maximum channel id used in the module group.
  int64_t max_channel_id() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_8(mht_8_v, 400, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "max_channel_id");
 return max_channel_id_; }

  HloAliasAnalysis* alias_analysis(HloModule* module) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_9(mht_9_v, 405, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "alias_analysis");

    return alias_analyses_.at(module).get();
  }

 private:
  Status Build();

  // Record all channel instructions, cross-module AllReduce instructions, and
  // While/Conditional/Call instructions.
  Status RecordInstructions();

  // Verifies the given HloModules are well-formed and follow the specification,
  // in particular with respect to using channel instructions.
  //
  // * Each channel has all 4 instructions (Send, Recv, SendDone, RecvDone).
  // * The shape of channel instructions match.
  // * The nest level of channel instructions match.
  // * Channel instructions are used in allowed computations, i.e., in the
  //   entry computation of the module or condition/body of While computations.
  Status VerifyChannelInstructions();

  // Adds metadata that the given two instructions are companions.
  Status AddCompanion(HloInstruction* instruction1,
                      HloInstruction* instruction2);

  // Checks whether a communicating instruction is placed in a valid position
  // within the graph.
  Status CheckCommunicatingInstruction(HloInstruction* instruction) const;

  // Performs a consistency check on the companion sets built for the input
  // modules. Checks that each instruction in a companion set is in a different
  // module/device.
  Status VerifyCompanionSets() const;

  // Retrieves a pointer to the stored TrackedInstruction associated with a
  // tracked computation, or nullptr in case such computation is not tracked.
  const TrackedInstruction* GetTrackedInstruction(
      const HloComputation* computation) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_metadataDTh mht_10(mht_10_v, 445, "", "./tensorflow/compiler/xla/service/hlo_module_group_metadata.h", "GetTrackedInstruction");

    auto it = tracked_instructions_.find(computation);
    return it != tracked_instructions_.end() ? &it->second : nullptr;
  }

  // Dump all the collected module group statistics to the logs.
  void DumpCollectedStats() const;

  // List of all companion instructions sets in the module.
  std::vector<std::unique_ptr<std::vector<HloInstruction*>>> companion_sets_;

  // Map from each companion while instruction to the index into companion_set_.
  absl::flat_hash_map<const HloInstruction*, int64_t> companion_set_index_;

  // Map from computation to the instruction using it (a kWhile, kConditional).
  absl::flat_hash_map<const HloComputation*, TrackedInstruction>
      tracked_instructions_;

  // Maps tracked instructions (kWhile, kConditional, kCall, ...) to the set of
  // communicating instructions within the proper called computation(s).
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      tracked_instructions_comms_;

  // All channels in the module.
  std::vector<Channel> channels_;

  // Map from channel ids to the index in channels_.
  absl::flat_hash_map<int64_t, int64_t> channel_id_map_;

  // Map from all-reduce ids to the all reduce instructions.
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>> all_reduce_map_;

  // The maximum channel id used in the module group.
  int64_t max_channel_id_ = -1;

  // The modules that this metadata was built from.
  const std::vector<HloModule*> modules_;

  absl::flat_hash_map<HloModule*, std::unique_ptr<HloAliasAnalysis>>
      alias_analyses_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_
