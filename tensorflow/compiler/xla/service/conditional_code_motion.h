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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh() {
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


#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace conditional_opt {
// At the conceptual level, a boundary can be thought of as representing a
// single virtual operation, except this virtual operation is conditionally
// instantiated into different concrete operations at each conditional branch.
// So a boundary is mapped to a single concrete operation if it is outside of
// conditional branches, and is mapped to a list of instructions if inside the
// branches. This data structure therefore allows a common data structure
// representation of the instructions to be moved, whether  they are inside or
// outside of the branches. Subsequently, it allows a common implementation
// basis to be used for both moving instructions out of and for moving them
// inside branches.
class Boundary {
 public:
  enum class Position { kInsideBranch, kOutsideBranch, kUndefined };
  Boundary() : position_(Position::kUndefined) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "Boundary");
}
  explicit Boundary(Position p) : position_(p) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "Boundary");
}
  std::vector<HloInstruction*>& mutable_operands() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_2(mht_2_v, 219, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "mutable_operands");
 return operands_; }
  const std::vector<HloInstruction*>& operands() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_3(mht_3_v, 223, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "operands");
 return operands_; }
  bool IsInsideBranch() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_4(mht_4_v, 227, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "IsInsideBranch");
 return position_ == Position::kInsideBranch; }
  bool IsOutsideBranch() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_5(mht_5_v, 231, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "IsOutsideBranch");
 return position_ == Position::kOutsideBranch; }
  Position GetPosition() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_6(mht_6_v, 235, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "GetPosition");
 return position_; }
  bool IsEmpty() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_7(mht_7_v, 239, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "IsEmpty");
 return operands_.empty(); }
  std::string ToString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_8(mht_8_v, 243, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "ToString");

    std::string res;
    for (HloInstruction* op : operands_) {
      res += op->ToString() + ";";
    }
    return res;
  }
  bool operator==(const Boundary& that) const {
    return absl::c_equal(operands_, that.operands_);
  }
  template <typename H>
  friend H AbslHashValue(H h, const Boundary& boundary) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_9(mht_9_v, 257, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "AbslHashValue");

    return H::combine(std::move(h), boundary.operands_);
  }

 private:
  // Boundary instructions in the conditional branches, one from each branch
  // of the conditional; or a single operand from outside the conditional.
  std::vector<HloInstruction*> operands_;
  Position position_;
};

// HLO pass that moves identical ops in/out of conditional.
// - The definition of identical are the shape of the operands are identical
// and their properties are identical.
// - Only the identical ops that won't share operands with other ops will
// be moved out of conditional.
// The cost model of the code motion optimization includes two components:
// represented by the move_config_ and reuse_config_ arrays of the optimization.
// The move_config_ array uses 1 vs 0 to dictate whether each Hlo Opcode, when
// used with its first operand being another given Hlo Opcode, is allowed to
// move across any conditional boundary; the reuse_config_ array uses an integer
// to represent the force between each pair of HloOpcode regarding how
// attractive it is to place these instructions together (both inside or outside
// of a conditional). Both arrays use Hlo Opcode only to drive the
// configuration, regardless of where the operations are located in the
// module.
class ConditionalCodeMotion : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the hoist process preserves layout
  // during identical comparison. Otherwise, layout is ignored.
  // The search configuration is a single integer but is split into four parts:
  // (sign, n, m, p), where n,m,p each occupy 8 bits and together make the 24
  // bits at the end of the int32_t. For the sign part, if search_config is <0,
  // the reuse_config_ cost model is modified (tuned); if search_config is >0,
  // the move_config_ cost model is modified (tuned); if search_config == 0,
  // the default cost model is used with no tuning. When tuning, the entries in
  // the designated configuration array (move_config_ or reuse_config_) are
  // flipped between 0 and another default integer, starting from the pth entry
  // being queried by the optimization and repeated every nth time a new entry
  // is visited, until a maximal of m entries have been changed. The tuning
  // start over when optimizing a new model.
  explicit ConditionalCodeMotion(bool is_layout_sensitive,
                                 bool pursue_full_conditional_code_motion,
                                 int64_t search_config = 0)
      : is_layout_sensitive_(is_layout_sensitive),
        pursue_full_conditional_code_motion_(
            /*turn off special case if tuning*/
            pursue_full_conditional_code_motion && search_config == 0),
        search_config_index_(0) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_10(mht_10_v, 308, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "ConditionalCodeMotion");

    search_config_.push_back(search_config);
    if (search_config != 0) {
      search_config_map_[0] = search_config_;
    }
  }
  explicit ConditionalCodeMotion(bool is_layout_sensitive,
                                 bool pursue_full_conditional_code_motion,
                                 std::string search_config)
      : is_layout_sensitive_(is_layout_sensitive),
        pursue_full_conditional_code_motion_(
            /*turn off special case if tuning*/
            pursue_full_conditional_code_motion && search_config.empty()),
        search_config_index_(-1) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("search_config: \"" + search_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_11(mht_11_v, 325, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "ConditionalCodeMotion");

    ParseSearchConfiguration(search_config);
  }
  // Parse a given string in the format of a sequence of i,s,m,t into a
  // list of transformation search configurations, each configuration generated
  // by invoking MakeSearchConfig(s,m,t) and will be used for the ith
  // conditional encountered when optimizing a given module.
  void ParseSearchConfiguration(const std::string& search_config);
  // Make a single search configuration for changing transformation decisions:
  // flip the decisions at position n = flip_start + flip_stride * m, and
  // m = 0..max_flip.
  // The following defines how the int64_t search configuration is composed, as
  // flip_start + (flip_max << kMaxPos) + (flip_stride << kStridePos).
  // Position (digit) for maximum number of flips.
  static constexpr int kMaxPos = 16;
  // Position (digit) for the count-down to the first flip.
  static constexpr int kStartPos = 0;
  // Position (digit) for the count-down to the next flip.
  static constexpr int kStridePos = 32;
  // Bit mask for extracting the last digits of value.
  static constexpr int kValueMask = 0xffff;
  static int64_t MakeSearchConfig(int64_t start, int64_t max, int64_t stride) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_12(mht_12_v, 349, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "MakeSearchConfig");

    const int64_t config =
        (max << kMaxPos) + (start << kStartPos) + (stride << kStridePos);
    VLOG(2) << "flip stride = " << flip_stride(config) << "\n";
    VLOG(2) << "flig config = " << config << "\n";
    return config;
  }

  static int16_t flip_start(int64_t search_config) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_13(mht_13_v, 360, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "flip_start");

    return (search_config >> kStartPos) & kValueMask;
  }

  static int16_t flip_stride(int64_t search_config) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_14(mht_14_v, 367, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "flip_stride");

    return (search_config >> kStridePos) & kValueMask;
  }

  static int16_t DecrementMaxFlip(int64_t* search_config) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_15(mht_15_v, 374, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "DecrementMaxFlip");

    const int16_t max_flip = ((*search_config) >> kMaxPos) & kValueMask;
    // Decrement flip count so we can stop if it reaches 0.
    if (max_flip > 0) {
      *search_config -= (1 << kMaxPos);
    }
    return max_flip;
  }

  absl::string_view name() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_16(mht_16_v, 386, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "name");
 return "conditional-code-motion"; }
  StatusOr<bool> Run(HloModule* module) override;

  // Optimization decision for each boundary of the conditional instruction.
  class Decision {
   public:
    enum class Direction : uint8_t {
      kMoveOutOfBranch,
      kMoveIntoBranch,
      kNoChange
    };

   public:
    Decision(Direction direction, int benefit)
        : direction_(direction), benefit_(benefit) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_17(mht_17_v, 403, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "Decision");
}
    Direction GetDirection() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_18(mht_18_v, 407, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "GetDirection");
 return direction_; }
    int GetBenefit() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSconditional_code_motionDTh mht_19(mht_19_v, 411, "", "./tensorflow/compiler/xla/service/conditional_code_motion.h", "GetBenefit");
 return benefit_; }

   private:
    Direction direction_;
    int benefit_;
  };
  // If the optimization decision is NO_CHANGE, new_boundary is set to nullptr;
  // otherwise, it is set to the new boundary after proposed optimization.
  virtual Decision ConsiderCodeMotion(
      HloInstruction* conditional, const Boundary& cur_boundary,
      std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries,
      absl::flat_hash_map<HloInstruction*, int>& visited_count);

 private:
  const bool is_layout_sensitive_;
  const bool pursue_full_conditional_code_motion_;
  // The following parameterizes the transformation decisions and cost model.
  std::vector<int64_t> search_config_;
  int64_t search_config_index_;
  // Map each conditional to a vector of its search configurations. The key of
  // the map is the index number of the conditional in a module when traversed
  // in post order, and the value of the map is the sequence of search
  // configurations specified with the same index number for the conditional.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> search_config_map_;
  std::vector<std::vector<int64_t>> move_config_, reuse_config_;

  StatusOr<bool> MoveInstructionOut(HloInstruction* conditional,
                                    std::vector<Boundary>& to_move_out,
                                    std::vector<Boundary>& new_boundaries);
  StatusOr<bool> MoveInstructionIn(HloInstruction* conditional,
                                   std::vector<Boundary>& to_move_in,
                                   std::vector<Boundary>& new_boundaries);
  void SetDefaultMoveConfig();
};
}  // namespace conditional_opt

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
