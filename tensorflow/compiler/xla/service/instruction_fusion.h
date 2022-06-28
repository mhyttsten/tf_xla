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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh() {
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


#include <functional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

struct NoFusionPossible;

// Propagating explanation of fusion decisions: if something could not be fused,
// explain the reason.
class FusionDecision {
 public:
  // Can not be fused: explain why. Implicit conversion due to optional-like
  // semantics: waiver granted in cl/419938611.
  FusionDecision(absl::string_view explanation)  // NOLINT
      : explanation_(explanation) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("explanation: \"" + std::string(explanation.data(), explanation.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "FusionDecision");
}

  // Same constructor as string_view, to allow implicit string conversion (can't
  // implicitly convert both char* to string_view and string_view to
  // FusionDecision).
  FusionDecision(const char* explanation)  // NOLINT
      : explanation_(explanation) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("explanation: \"" + (explanation == nullptr ? std::string("nullptr") : std::string((char*)explanation)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "FusionDecision");
}

  // If condition is `true` means that we CAN fuse. In that case, explanation is
  // discarded.
  FusionDecision(bool condition, absl::string_view explanation) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("explanation: \"" + std::string(explanation.data(), explanation.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_2(mht_2_v, 231, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "FusionDecision");

    if (!condition) {
      explanation_ = std::string(explanation);
    }
  }

  // Can be fused.
  FusionDecision() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_3(mht_3_v, 241, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "FusionDecision");
}

  // A trick to declare and test fusion decision in a single statement (as TF
  // is still on C++14 and can't use if statement with explicit initializer).
  //
  // Cf. NoFusionPossible definition for sample usage.
  // TODO(b/157309856): Use conditional initializer instead.
  NoFusionPossible operator!();

  // Returns whether it can be fused.
  explicit operator bool() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_4(mht_4_v, 254, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "bool");
 return CanFuse(); }

  // Whether the fusion decision is positive.
  bool CanFuse() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_5(mht_5_v, 260, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "CanFuse");
 return !explanation_.has_value(); }

  // Connects two decisions with a disjunction. This is different than just
  // picking one, as we also have to propagate both explanations if only one of
  // them is false to show why fusion wasn't performed.
  FusionDecision Or(const FusionDecision& decision) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_6(mht_6_v, 268, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "Or");

    if (CanFuse() || decision.CanFuse()) {
      return {};
    }
    return {absl::StrCat(explanation_.value_or(""), " ; ", decision.Explain())};
  }

  // Connects two fusion decision with a conjunction. Unlike disjunction,
  // propagates only one explanation (as it is enough to show that fusion could
  // not be done).
  FusionDecision And(const FusionDecision& decision) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_7(mht_7_v, 281, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "And");

    if (CanFuse()) {
      return decision;
    }
    if (decision.CanFuse()) {
      return *this;
    }
    // Both conditions were violated: returning either is valid.
    return *this;
  }

  // Appends to explanation, or turns the decision negative.
  FusionDecision operator<<(absl::string_view explanation) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("explanation: \"" + std::string(explanation.data(), explanation.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_8(mht_8_v, 297, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "operator<<");

    return {absl::StrCat(explanation_.value_or(""), explanation)};
  }

  // Appends to explanation, or turns the decision negative.
  FusionDecision operator<<(int64_t explanation) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_9(mht_9_v, 305, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "operator<<");

    return {absl::StrCat(explanation_.value_or(""), explanation)};
  }

  // Explains why the fusion could not be performed.
  std::string Explain() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_10(mht_10_v, 313, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "Explain");
 return *explanation_; }

 private:
  // Empty IFF fusion is possible (explanation provided for negative cases).
  absl::optional<std::string> explanation_;
};

// Helper class: contextually convertible to "no fusion possible" unlike
// FusionDecision. This is a trick to declare and test fusion decision in a
// single statement (as we are still on C++14).
//
// Sample usage:
//
// if (NoFusionPossible fusible = !FusabilityRestriction(producer, consume)) {
//   return !fusible; // Note that negation converts it back to FusionDecision.
// }
struct NoFusionPossible {
  // Inverts the test value (true <=> not fusible) on wrapped FusionDecision.
  explicit operator bool() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_11(mht_11_v, 334, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "bool");
 return !static_cast<bool>(fusion_decision); }

  // Returns wrapped fusion decision.
  FusionDecision operator!() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_12(mht_12_v, 340, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "!");
 return fusion_decision; }

  FusionDecision fusion_decision;
};

inline NoFusionPossible FusionDecision::operator!() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_13(mht_13_v, 348, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "!");
 return {*this}; }

// HLO pass which performs instruction fusion. Instructions are fused
// "vertically", meaning producing instructions are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation. Derived classes define ShouldFuse method to select which
// instructions to fuse.
class InstructionFusion : public HloModulePass {
 public:
  explicit InstructionFusion(
      std::function<bool(const HloInstruction& instruction)> is_expensive,
      bool may_duplicate = true,
      FusionConfigCollection config_collection_mode =
          FusionConfigCollection::kOff)
      : is_expensive_(is_expensive),
        may_duplicate_(may_duplicate),
        config_collection_mode_(config_collection_mode) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_14(mht_14_v, 367, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "InstructionFusion");
}
  ~InstructionFusion() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_15(mht_15_v, 372, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "name");
 return "fusion"; }

  // Run instruction fusion on the given computation. Returns whether the
  // computation was changed (instructions were fused).
  StatusOr<bool> Run(HloModule* module) override;

  // Returns true if the computation of the given instruction is significantly
  // more expensive than just writing all the values of the instructions' result
  // array. Expensive operations will not be duplicated.
  static bool IsExpensive(const HloInstruction& instruction);

  // Returns true if it's legal to fuse the producer instruction into consumer
  // with regard to in-place semantics of the consumer. For example, it is
  // illegal to fuse a slice into a dynamic-update-slice if the slice output is
  // used as the update and if slice and dynamic-update-slice indices cannot be
  // proven to be the same.
  static FusionDecision ShouldFuseInPlaceOp(const HloInstruction* producer,
                                            const HloInstruction* consumer);

 protected:
  // Returns a list of computations on which Fusion is performed.
  virtual std::vector<HloComputation*> GetFusionComputations(HloModule* module);

  // Returns a FusionQueue that implements custom order of instructions being
  // fused. The default implementation processes consumers in reverse post
  // order.
  virtual std::unique_ptr<FusionQueue> GetFusionQueue(
      HloComputation* computation);

  // Returns whether the given producer instruction should be fused into the
  // given consumer instruction. producer is necessarily an operand of consumer.
  // Derived classes should define this method to specify which instructions
  // should be fused. `operand_index` is which operand of the consumer the
  // producer is.
  //
  // Instructions are traversed in reverse post order (computation root to
  // leaves). This method is called for each operand of the instruction (where
  // the operand is 'producer' and the instruction is 'consumer')
  //
  // Subtypes can override this with target-specific heuristics.
  virtual FusionDecision ShouldFuse(HloInstruction* consumer,
                                    int64_t operand_index);

  // Returns whether multi-output fusion can be applied to fuse `producer` into
  // `consumer`. In contrast to "regular" fusion, the `producer` is not
  // duplicated by multi-output fusion.
  virtual FusionDecision ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                   int64_t operand_index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_16(mht_16_v, 422, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "ShouldFuseIntoMultiOutput");

    return "multi-output fusion not supported by this pass";
  }

  // Chooses a fusion kind for `producer` and `consumer`.
  // Default method chooses `kLoop`.
  virtual HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                                const HloInstruction* consumer);

  // Fuses 'producer' into 'fusion_instruction'. 'fusion_instruction' needs to
  // be a fusion instruction. Returns the newly created clone of 'producer'
  // which is part of the fusion computation.
  virtual HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                          HloInstruction* producer);

  // Fuses producer into consumer. Returns the fusion instruction.
  virtual HloInstruction* Fuse(HloInstruction* producer,
                               HloInstruction* consumer,
                               HloComputation* computation);

  // Creates a new fusion instruction containing `producer` and `consumer`. A
  // tuple is added as the fusion instruction's root, which consumes from both,
  // `producer` and `consumer`. This style of fusion is referred to as
  // multi-output fusion.
  virtual HloInstruction* FuseIntoMultiOutput(HloInstruction* producer,
                                              HloInstruction* consumer,
                                              HloComputation* computation);

  // An "effectively unary" operation is one that has at most one "large"
  // input with the others being negligible in terms of memory usage.
  // We use "has a smaller true rank than the output" as a heuristic
  // for "negligible" memory usage.
  bool EffectivelyAtMostUnary(HloInstruction* hlo);

  // Returns true if fusing producer into consumer would cause producer to be
  // duplicated. This is the case if producer has uses other than consumer.
  bool FusionWouldDuplicate(const HloInstruction& producer,
                            const HloInstruction& consumer) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_17(mht_17_v, 462, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "FusionWouldDuplicate");

    return !(producer.users().size() == 1 && consumer.IsUserOf(&producer));
  }

  bool is_expensive(const HloInstruction& instruction) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_18(mht_18_v, 469, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "is_expensive");

    return is_expensive_(instruction);
  }

  // Overwrites the originally initialized is_expensive function.
  void set_is_expensive(
      std::function<bool(const HloInstruction& instruction)> is_expensive) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_19(mht_19_v, 478, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "set_is_expensive");

    is_expensive_ = is_expensive;
  }

  // Whether multi-output fusion would introduce a cycle into the HLO graph.
  bool MultiOutputFusionCreatesCycle(HloInstruction* producer,
                                     HloInstruction* consumer,
                                     const HloReachabilityMap& reachability);

  FusionConfigCollection config_collection_mode() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinstruction_fusionDTh mht_20(mht_20_v, 490, "", "./tensorflow/compiler/xla/service/instruction_fusion.h", "config_collection_mode");

    return config_collection_mode_;
  }

  // Returns whether 'consumer' may reuse elements of its `operand_index`th
  // operand.
  bool ReusesOperandElements(const HloInstruction* consumer,
                             int64_t operand_index);

  // The set of producers whose consumers we cannot fuse into.
  using HloInstructionSet = absl::flat_hash_set<HloInstruction*>;

  // Computes the set of nodes that we do not want to fuse into any of their
  // consumers based on a global analysis of the HLO graph.
  virtual HloInstructionSet ComputeGloballyUnfusible(
      absl::Span<HloInstruction* const> post_order,
      const HloReachabilityMap& reachability);

 private:
  // Returns the reused operands of `instruction` from reused_fusion_operands_,
  // computing them if they have not previously been computed for that
  // instruction.
  // The returned value has pointer stability, assuming entries are not deleted
  // from reused_fusion_operands_.
  absl::flat_hash_set<const HloInstruction*>& ReusedOperandsOf(
      const HloInstruction* instruction);

  // Updates reused_fusion_operands_ for a fusion when we are about to fuse
  // `producer` into `fusion_instruction`.
  void UpdateReusedOperandsForFusion(HloInstruction* producer,
                                     HloInstruction* fusion_instruction);

  HloInstruction* AddFusionInstruction(HloInstruction* producer,
                                       HloInstruction* consumer,
                                       HloComputation* computation);

  // Whether or not we can fuse producer into consumer on all paths
  // from the producer to the consumer where nodes are HLOs and edges are uses.
  //
  // A map from <producer, consumer> to a bool is required as the result cache
  // to store and query the results of calls to this function, in order to avoid
  // repeated computations.
  bool CanFuseOnAllPaths(
      HloInstruction* producer, HloInstruction* consumer,
      const HloInstructionSet& do_not_fuse,
      const HloReachabilityMap& reachability,
      absl::flat_hash_map<std::pair<HloInstruction*, HloInstruction*>, bool>*
          result_cache);

  // Used to determine if an HLO is expensive. Expensive operations will not be
  // duplicated.
  std::function<bool(const HloInstruction& instruction)> is_expensive_;

  // Returns whether we may duplicate an instruction if we want to fuse it.
  bool may_duplicate_;

  // Configuration mode.
  FusionConfigCollection config_collection_mode_;

  // Caches which operands are reused inside fusion computations.
  absl::flat_hash_map<
      const HloInstruction*,
      std::unique_ptr<absl::flat_hash_set<const HloInstruction*>>>
      reused_fusion_operands_;

  InstructionFusion(const InstructionFusion&) = delete;
  InstructionFusion& operator=(const InstructionFusion&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
