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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh() {
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


#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// This class implements the fusing of sibling fusion instructions that sharing
// common operands.
// It constructs the following associated data structures.
//  (1) candidates_: stores the instruction and the set of instructions it can
//      fuse to.
//  (2) candidates_index_: maps instruction to id.
//  (3) reachability_: reachability map in this computation.
//  (4) all_fusion_candidates_: the vector of candidate instructions.
//  (5) worklist_: a priority queue that contains pairs of instructions to be
//      fused and their fusion profit scores.
//
//  Function Perform() applies the optimization. It picks up the most profitable
//  pair in the worklist_, checks if it's legal to fuse and fuses the pair.
//  After fusion, it updates the associated structures such as reachability_,
//  candidates_ and worklist_.
//  Note that the reachability map is updated based on the original computation.
//  This works because the reachability is monotonically increasing with
//  instruction fusion.
class MultiOutputFusion : public HloModulePass {
 public:
  MultiOutputFusion() = default;

  absl::string_view name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_0(mht_0_v, 222, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "name");
 return "multi_output_fusion"; }

  // Run multi-output fusion on the given module. Returns whether the module
  // was changed.
  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Main entry for the optimization. Returns true if the optimization happens.
  bool Perform();

  // Test if instr1 and instr2 have the compatible shapes that can be legally
  // fused.
  virtual bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                         HloInstruction* instr2) = 0;

  // Whether the instruction is a candidate for fusion.
  virtual bool IsFusible(HloInstruction* instr) = 0;

  // This function estimates the savings by merging instr1 and instr2 into one
  // multi-output fusion instruction. It returns a result in kib. (The result
  // is intentionally not granules, because this method is not TPU-specific.)
  virtual int64_t GetProfit(HloInstruction* instr1, HloInstruction* instr2) = 0;

  // Whether fusing the instruction can reduce memory reads.
  virtual bool IsProfitableOperand(HloInstruction* instr);

  // Test if it's legal to fuse instr1 and instr2 into one fusion instruction.
  virtual bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2);

  // Test if it's legal to fuse instr1 and instr2 into one fusion instruction
  // using main constraints.
  bool LegalToFuseMainConstraints(HloInstruction* instr1,
                                  HloInstruction* instr2);

  // Fuse HloInstruction instr1 and instr2 and return the fused instruction.
  // The other instruction is removed from its parent computation.
  virtual HloInstruction* Fuse(HloInstruction* instr1, HloInstruction* instr2);

  // Recompute reachability for the current computation.
  void RecomputeReachability();

  // Returns the reachability map for the current computation.
  HloReachabilityMap* reachability() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_1(mht_1_v, 267, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "reachability");
 return reachability_.get(); }

  // Returns the computation for the pass.
  HloComputation* computation() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_2(mht_2_v, 273, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "computation");
 return computation_; }

  // Update the reachability map after fusing instr1 and instr2.
  void UpdateReachability(
      HloInstruction* instr1, HloInstruction* instr2,
      absl::Span<const std::pair<HloInstruction*, HloReachabilityMap::Index>>
          instrs_to_update,
      const std::function<bool(HloInstruction*)>& skip = nullptr);

  // Hook for multi-output fusion along producer-consumer edges.
  // Returns whether any instructions were fused.
  //
  // TODO(b/80420762): Perform producer-consumer multi-output fusion in
  // InstructionFusion instead.
  virtual bool DoProducerConsumerMultiOutputFusion();

  // Return a list of fusible instructions that can be fused into the fusion of
  // instr1 and instr2. The second entry in the vector is an old profit value
  // from fusing the corresponding instruction and the base op of the new
  // fusion.
  std::vector<std::pair<HloInstruction*, int64_t>> GetNewFusibles(
      HloInstruction* instr1, HloInstruction* instr2);

  // Create a new fusion instruction and add `base' into it.
  // Prepare for fusing `to_fuse' into the created fusion by updating
  // reachability, worklist, and fusion candidates.
  HloInstruction* CreateFusion(HloInstruction* base, HloInstruction* to_fuse);

 private:
  // An internal data structure for each instruction in current computation.
  // When an instruction is removed, member 'hlo' is set to nullptr.
  struct FusionCandidate {
    HloInstruction* hlo;
    std::list<std::pair<HloInstruction*, int64_t>> fusibles;
    explicit FusionCandidate(HloInstruction* hlo) : hlo(hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_3(mht_3_v, 310, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "FusionCandidate");
}
  };

  // The pair of candidates to be fused and the profit score.
  struct ToBeFused {
    HloInstruction* instr1;
    HloInstruction* instr2;
    int64_t score;
    int64_t timestamp;
    ToBeFused(HloInstruction* instr1, HloInstruction* instr2, int64_t score,
              int64_t timestamp)
        : instr1(instr1), instr2(instr2), score(score), timestamp(timestamp) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_4(mht_4_v, 324, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "ToBeFused");
}
    bool operator<(const ToBeFused& rhs) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_5(mht_5_v, 328, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "operator<");

      return std::pair<int64_t, int64_t>(score, timestamp) <
             std::pair<int64_t, int64_t>(rhs.score, rhs.timestamp);
    }
  };

  // Stable priority queue where each insertion has a timestamp for
  // deterministic popping.
  class WorkList {
   public:
    bool empty() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_6(mht_6_v, 341, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "empty");
 return worklist_.empty(); }
    ToBeFused pop() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_7(mht_7_v, 345, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "pop");

      ToBeFused tmp = worklist_.top();
      worklist_.pop();
      return tmp;
    }
    template <class... Args>
    void emplace(Args&&... args) {
      worklist_.emplace(std::forward<Args>(args)..., timestamp_++);
    }

   private:
    std::priority_queue<ToBeFused> worklist_;
    int64_t timestamp_ = 0;
  };

  // Update the internal data structures before instr1 and instr2 are fused into
  // one fusion instruction.
  void UpdateBeforeFuse(HloInstruction* instr1, HloInstruction* instr2);

  // Update the internal data structures after instructions are fused into
  // one fusion instruction.
  void UpdateAfterFuse(
      HloInstruction* fusion,
      const std::vector<std::pair<HloInstruction*, int64_t>>& new_fusibles,
      bool new_fusion_node);

  int64_t get_candidate_id(HloInstruction* instr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_8(mht_8_v, 374, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "get_candidate_id");

    return FindOrDie(candidates_index_, instr);
  }

  bool is_fused(HloInstruction* instr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_9(mht_9_v, 381, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "is_fused");

    return candidates_[get_candidate_id(instr)].hlo == nullptr;
  }

  void set_is_fused(HloInstruction* instr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_10(mht_10_v, 388, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "set_is_fused");

    candidates_[get_candidate_id(instr)].hlo = nullptr;
  }

  bool is_connected(HloInstruction* instr1, HloInstruction* instr2) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmulti_output_fusionDTh mht_11(mht_11_v, 395, "", "./tensorflow/compiler/xla/service/multi_output_fusion.h", "is_connected");

    return reachability_->IsConnected(instr1, instr2);
  }

  std::vector<FusionCandidate> candidates_;
  WorkList worklist_;

  // A map that maps an instruction to the index_.
  absl::flat_hash_map<HloInstruction*, int> candidates_index_;

  // The reachability map of current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  // This stores all the candidate instructions and their indices within
  // reachability_ in current computation.
  std::vector<std::pair<HloInstruction*, HloReachabilityMap::Index>>
      all_fusion_candidates_;

  // Computation for the pass.
  HloComputation* computation_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_
