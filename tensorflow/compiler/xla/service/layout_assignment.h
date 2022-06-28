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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh() {
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


#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Abstract base class for layout constraints. These constraint objects are
// gathered together in LayoutConstraints object.
class LayoutConstraint {
 public:
  LayoutConstraint(bool mandatory, bool dfs, int64_t priority)
      : mandatory_(mandatory), dfs_(dfs), priority_(priority) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_0(mht_0_v, 222, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "LayoutConstraint");
}
  virtual ~LayoutConstraint() = default;

  virtual std::string ToString() const = 0;

  // True if this constraint cannot be overwritten by a different constraint.
  bool mandatory() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "mandatory");
 return mandatory_; }

  // When true, propagate in DFS. When false, constraint will propagate in BFS.
  bool dfs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_2(mht_2_v, 237, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "dfs");
 return dfs_; }

  // Return the priority of the current constraint. When conflicting constraints
  // are encountered, the higher priority one should win.
  int64_t priority() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_3(mht_3_v, 244, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "priority");
 return priority_; }
  bool IsDefaultLayout() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_4(mht_4_v, 248, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "IsDefaultLayout");
 return priority_ == kDefaultPriority; }

  // The priority of all default layouts when not set explicitly.
  static constexpr int64_t kDefaultPriority = -2;
  // The beginning priority of layout assignment.
  static constexpr int64_t kBeginningPriority = 0;
  // The priority of layout assignment given by the user for entry computation.
  static constexpr int64_t kGivenPriority = 3;

 protected:
  bool mandatory_;
  bool dfs_;
  int64_t priority_;
};

std::ostream& operator<<(std::ostream& out, const LayoutConstraint& constraint);

// Layout constraint on a single LogicalBuffer. This constrains the layout of an
// array produced by a particular instruction.
class BufferLayoutConstraint : public LayoutConstraint {
 public:
  BufferLayoutConstraint(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory, bool dfs, int64_t priority);

  const LogicalBuffer& buffer() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_5(mht_5_v, 275, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "buffer");
 return *buffer_; }
  const Layout& layout() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_6(mht_6_v, 279, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "layout");
 return layout_; }
  bool UpdateLayout(int64_t priority, const Layout& layout, bool mandatory,
                    bool dfs);

  std::string ToString() const override;

 private:
  Layout layout_;
  const LogicalBuffer* buffer_;
};

// Constraint on the layout of the operand of an instruction. The constrained
// shape can be arbitrarily shaped (array or tuple). This is a constraint on the
// use of a shaped value and is not a hard constraint on the instruction(s)
// which define the value as copies may be inserted between the definition and
// use.
class OperandLayoutConstraint : public LayoutConstraint {
 public:
  OperandLayoutConstraint(const ShapeLayout& shape_layout,
                          const HloInstruction* instruction, int64_t operand_no,
                          bool mandatory, bool dfs, int64_t priority);

  const ShapeLayout& shape_layout() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_7(mht_7_v, 304, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "shape_layout");
 return shape_layout_; }
  const HloInstruction* instruction() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_8(mht_8_v, 308, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "instruction");
 return instruction_; }
  const int64_t operand_no() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_9(mht_9_v, 312, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "operand_no");
 return operand_no_; }
  const HloInstruction* operand() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_10(mht_10_v, 316, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "operand");

    return instruction_->operand(operand_no_);
  }

  std::string ToString() const override;

 private:
  ShapeLayout shape_layout_;
  const HloInstruction* instruction_;
  int64_t operand_no_;
};

// Constraint on the layout of a computation interface.
class ComputationLayoutConstraint : public LayoutConstraint {
 public:
  static constexpr int64_t kDefaultLayoutIsUsed = 0;
  static constexpr int64_t kResultLayoutIsSet = 1;
  static constexpr int64_t kParameterLayoutIsSet = 2;
  static constexpr int64_t kComputationLayoutIsSet = 3;
  explicit ComputationLayoutConstraint(const HloComputation* computation,
                                       ComputationLayout* computation_layout,
                                       int64_t priority)
      : LayoutConstraint(/*mandatory=*/true, /*dfs=*/true, priority),
        layout_state_((computation_layout == nullptr)
                          ? kDefaultLayoutIsUsed
                          : kComputationLayoutIsSet),
        computation_layout_(
            (computation_layout == nullptr)
                ? ComputationLayout(computation->ComputeProgramShape(),
                                    /*ignore_layouts=*/false)
                : *computation_layout) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_11(mht_11_v, 349, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ComputationLayoutConstraint");
}

  const ComputationLayout& computation_layout() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_12(mht_12_v, 354, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "computation_layout");

    return computation_layout_;
  }
  void ResetComputationLayout(const ComputationLayout& layout, int64_t priority,
                              bool prop_result_layout,
                              bool prop_parameter_layout) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_13(mht_13_v, 362, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ResetComputationLayout");

    computation_layout_ = layout;
    priority_ = priority;
    if (prop_result_layout) {
      layout_state_ |= kResultLayoutIsSet;
    }
    if (prop_parameter_layout) {
      layout_state_ |= kParameterLayoutIsSet;
    }
  }
  void ResetResultLayout(const ShapeLayout& shape_layout, int64_t priority) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_14(mht_14_v, 375, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ResetResultLayout");

    *computation_layout_.mutable_result_layout() = shape_layout;
    layout_state_ |= kResultLayoutIsSet;
    priority_ = priority;
  }
  bool parameter_layout_is_set() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_15(mht_15_v, 383, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "parameter_layout_is_set");

    return layout_state_ & kParameterLayoutIsSet;
  }
  bool result_layout_is_set() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_16(mht_16_v, 389, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "result_layout_is_set");

    return layout_state_ & kResultLayoutIsSet;
  }
  bool default_layout_is_used() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_17(mht_17_v, 395, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "default_layout_is_used");

    return layout_state_ == kDefaultLayoutIsUsed;
  }
  std::string ToString() const override;

 private:
  // The layout_state_ variable is used to remember whether the layout for
  // the overall computation is explicitly set, whether its result layout is
  // explicitly set, or whether it only stores the default layout of the
  // computation.
  int64_t layout_state_;
  ComputationLayout computation_layout_;
};

// Contains constraints on the layout of channels; sends and recvs.
class ChannelLayoutConstraints {
 public:
  // Construct an empty constraint set.
  ChannelLayoutConstraints() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_18(mht_18_v, 416, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ChannelLayoutConstraints");
}

  // Returns true if channel_id has a layout constraint.
  bool IsChannelConstrained(int64_t channel_id) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_19(mht_19_v, 422, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "IsChannelConstrained");

    return constraints_.contains(channel_id);
  }

  // Given `shape`, apply the layout for `channel_id`. `channel_id` must already
  // be constrained.
  Shape LayoutShapeForChannel(Shape shape, int64_t channel_id) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_20(mht_20_v, 431, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "LayoutShapeForChannel");

    auto it = constraints_.find(channel_id);
    CHECK(it != constraints_.end()) << "Channel " << channel_id;
    *shape.mutable_layout() = it->second;
    return shape;
  }

  // Returns the layout constraint for `channel_id`, which must already be
  // constrained.
  const Layout& LayoutForChannel(int64_t channel_id) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_21(mht_21_v, 443, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "LayoutForChannel");

    auto it = constraints_.find(channel_id);
    CHECK(it != constraints_.end()) << "Channel " << channel_id;
    return it->second;
  }

  // Adds a new layout constraint for `channel_id`. If a constraint for
  // `channel_id` has been added, this API returns nullptr, otherwise returns
  // the layout which has already been set for the channel.
  const Layout* ConstrainChannel(int64_t channel_id, const Layout& layout) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_22(mht_22_v, 455, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ConstrainChannel");

    auto it = constraints_.emplace(std::make_pair(channel_id, layout));
    if (it.second) {
      return nullptr;
    }
    return LayoutUtil::Equal(layout, it.first->second) ? nullptr
                                                       : &it.first->second;
  }

 private:
  absl::flat_hash_map<int64_t, Layout> constraints_;
};

// HLO pass which assigns layouts to all instructions in the HLO module while
// satisfying all necessary invariants and minimizing cost.
class LayoutAssignment : public HloModulePass {
 public:
  // entry_computation_layout is modified to populate a layout for the result in
  // the case that no particular layout is requested.
  //
  // channel_constraints is both an input and output. Any sends or recvs that
  // are present in channel_constraints will be laid out as constrained. Any
  // unconstrained sends or recvs will be laid out as locally optimal and their
  // layout will be added as a constraint to channel_constraints.
  //
  // If channel_constraints is nullptr, no kSend or kRecvs must be contained
  // within any module passed to `Run`.
  explicit LayoutAssignment(
      ComputationLayout* entry_computation_layout,
      ChannelLayoutConstraints* channel_constraints = nullptr,
      bool reverse_computation_order = false);
  ~LayoutAssignment() override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_23(mht_23_v, 489, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "~LayoutAssignment");
}
  const TuplePointsToAnalysis& points_to_analysis() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_24(mht_24_v, 493, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "points_to_analysis");

    return *points_to_analysis_;
  }
  absl::string_view name() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_25(mht_25_v, 499, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "name");
 return "layout-assignment"; }

  // Assign layouts to the given module. Returns whether the module was changed
  // (any layouts were changed).
  StatusOr<bool> Run(HloModule* module) override;

  // Class encapsulating the layout constraints of the values in a HLO
  // computation.
  class LayoutConstraints {
   public:
    explicit LayoutConstraints(HloComputation* computation,
                               ComputationLayout* computation_layout,
                               int64_t priority);
    ~LayoutConstraints() = default;

    const HloComputation* computation() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_26(mht_26_v, 517, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "computation");
 return computation_; }
    HloComputation* computation() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_27(mht_27_v, 521, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "computation");
 return computation_; }
    void ResetOperandConstraints() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_28(mht_28_v, 525, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ResetOperandConstraints");
 operand_constraints_.clear(); }
    const ShapeLayout* OperandLayout(const HloInstruction* instruction,
                                     int64_t operand_no) const;
    const OperandLayoutConstraint* GetOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no) const;
    const ShapeLayout* ResultLayout() const;
    OperandLayoutConstraint* InsertOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no,
        const OperandLayoutConstraint& constraint);
    Status SetResultLayout(LayoutAssignment* assignment,
                           const Shape& shape_with_layout, int64_t priority);

    const ComputationLayout& computation_layout() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_29(mht_29_v, 540, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "computation_layout");

      return computation_constraint_.computation_layout();
    }
    const ComputationLayoutConstraint& computation_constraint() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_30(mht_30_v, 546, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "computation_constraint");

      return computation_constraint_;
    }
    ComputationLayoutConstraint* mutable_computation_constraint() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_31(mht_31_v, 552, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "mutable_computation_constraint");

      return &computation_constraint_;
    }

   private:
    // The set of OperandLayoutConstraints applied to the computation.
    using OperandConstraintKey = std::pair<const HloInstruction*, int64_t>;
    std::map<OperandConstraintKey, OperandLayoutConstraint>
        operand_constraints_;

    HloComputation* computation_;
    ComputationLayoutConstraint computation_constraint_;
  };

  // Determines whether an instruction can change layouts. An instruction not
  // being able to change layout means that it requires operands with the same
  // rank as the output to have the same layout as the output.
  static bool InstructionCanChangeLayout(const HloInstruction* instruction);

  LayoutConstraints* mutable_computation_constraints(
      HloComputation* computation) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_32(mht_32_v, 575, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "mutable_computation_constraints");

    auto it = computation_layouts_.find(computation);
    LayoutConstraints* constraints = nullptr;
    if (it == computation_layouts_.end()) {
      computation_layouts_.emplace(
          computation,
          constraints = new LayoutConstraints(
              computation, nullptr, LayoutConstraint::kDefaultPriority));
    } else {
      constraints = (*it).second.get();
    }
    return constraints;
  }
  void PushAddedConstraints(const LayoutConstraint* constraint);

  // In case of an array shape returns true iff it is at most rank 1. In case of
  // a tuple shape returns true iff all leaf shapes are at most rank 1.
  static bool IsAtMostRank1(const Shape& shape);
  // Convenience wrapper around SetOperandLayout for setting the layout of a
  // operand using a Layout object. The operand must be array-shaped.
  Status SetArrayOperandLayout(const Layout& layout,
                               const HloInstruction* instruction,
                               int64_t operand_no, bool mandatory = true,
                               bool dfs = true) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_33(mht_33_v, 601, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "SetArrayOperandLayout");

    return SetArrayOperandLayout(layout, instruction, operand_no, mandatory,
                                 dfs, current_priority_);
  }
  Status SetArrayOperandLayout(const Layout& layout,
                               const HloInstruction* instruction,
                               int64_t operand_no, bool mandatory, bool dfs,
                               int64_t priority);
  // Convenience wrapper around SetBufferLayout. Sets the layouts of all buffers
  // created by the instruction to the layouts in the given shape. The
  // instruction must define every logical buffer in its output.
  // If `allow_alias` is false, the function will check that all output buffers
  // are defined by `instruction`, not aliased to an instruction elsewhere.
  Status SetInstructionLayout(const Shape& shape_with_layout,
                              const HloInstruction* instruction,
                              bool mandatory = true, bool dfs = true,
                              bool allow_alias = false) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_34(mht_34_v, 620, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "SetInstructionLayout");

    return SetInstructionLayout(shape_with_layout, instruction, mandatory, dfs,
                                allow_alias, current_priority_);
  }
  Status SetInstructionLayout(const Shape& shape_with_layout,
                              const HloInstruction* instruction, bool mandatory,
                              bool dfs, bool allow_alias, int64_t priority);
  // Set the same given layout across all components of the instruction output.
  // It works the same as the API above if the output is a single array.
  Status SetInstructionLayout(const Layout& layout,
                              const HloInstruction* instruction,
                              bool mandatory = true, bool dfs = true,
                              bool allow_alias = false, int64_t priority = -1);
  // Add a constraint on the layout of a LogicalBuffer, the layout of the
  // operand of the instruction, or the layout of the result of the computation,
  // respectively.
  Status SetBufferLayout(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory = true, bool dfs = true) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_35(mht_35_v, 640, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "SetBufferLayout");

    return SetBufferLayout(layout, buffer, mandatory, dfs, current_priority_);
  }
  Status SetBufferLayout(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory, bool dfs, int64_t priority);
  Status SetOperandLayout(const Shape& shape_with_layout,
                          const HloInstruction* instruction, int64_t operand_no,
                          bool mandatory = true, bool dfs = true) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_36(mht_36_v, 650, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "SetOperandLayout");

    return SetOperandLayout(shape_with_layout, instruction, operand_no,
                            mandatory, dfs, current_priority_);
  }
  Status SetOperandLayout(const Shape& shape_with_layout,
                          const HloInstruction* instruction, int64_t operand_no,
                          bool mandatory, bool dfs, int64_t priority);
  bool reverse_computation_order() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_37(mht_37_v, 660, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "reverse_computation_order");
 return reverse_computation_order_; }

 protected:
  // These methods, invoked by PropagateConstraints, propagate a layout
  // constraint to its neighbors (i.e. operands and users) in order to minimize
  // the cost of the instructions being constrainted on. New constraints are
  // added to the given constraint set.
  //
  // Backends can override these methods with backend-specific propagation
  // rules.
  virtual Status PropagateBufferConstraint(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);
  virtual Status PropagateOperandConstraint(
      const OperandLayoutConstraint& operand_constraint,
      LayoutConstraints* constraints);
  virtual Status PropagateResultConstraint(
      const ComputationLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);

  virtual Layout GetUnconstrainedLayout(const LogicalBuffer& buffer) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_38(mht_38_v, 683, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "GetUnconstrainedLayout");

    return LayoutUtil::GetDefaultLayoutForShape(buffer.shape());
  }
  // Called after layouts of an instruction have been finalized to allow
  // subclasses to check for platform specific assumptions.
  virtual Status Verify(const HloInstruction* instruction) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_39(mht_39_v, 691, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "Verify");

    return Status::OK();
  }

  Status PropagateUnconstraintedBuffers(LayoutConstraints* constraints);
  const BufferLayoutConstraint* GetBufferLayoutConstraint(
      const LogicalBuffer& buffer) const;
  // Find a bufferset in the bufferset cache. This is useful since we can
  // currently create the flattened buffer set for the same instruction many
  // times, which is often slow.
  PointsToSet::BufferSet* GetBufferSet(const HloInstruction* instruction) const;
  // Similar to above, but returns true only if all buffers associated with that
  // operand are forwarded.
  bool AllOperandBuffersForwarded(const HloInstruction* instruction,
                                  int64_t operand_no) const;
  // Returns true if any buffer in the given operand is forwarded to the output
  // of the given instruction. For example, the Tuple instruction forwards the
  // buffers of its operands and would return true for each of its operands.
  bool AnyOperandBufferForwarded(const HloInstruction* instruction,
                                 int64_t operand_no) const;
  StatusOr<Layout> InferArrayLayout(const HloInstruction* instruction,
                                    const ShapeIndex& index);

  // Propagates a buffer layout constraint into the operands that use it.
  Status PropagateBufferConstraintToUses(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Propagates a layout constraint on the use of the result of the given
  // instruction to the definitions of the LogicalBuffers which make up the
  // result.
  Status PropagateUseConstraintToDefs(const ShapeLayout& shape_layout,
                                      const HloInstruction* instruction,
                                      LayoutConstraints* constraints,
                                      int64_t priority);

  // Propagates the memory space defined in the entry computation to the called
  // computations.
  Status PropagateMemorySpace(HloModule* module);

  // Chooses a layout of operand `operand_no` of `instruction` that minimizes
  // the cost of `instruction`. `output_layout` is the layout of `instruction`.
  // Returns null if it can't decide the best layout.
  // Precondition: `instruction` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOperandLayoutFromOutputLayout(
      const Layout& output_layout, const HloInstruction* instruction,
      int64_t operand_no);
  // Given the layout of `user`'s `operand_no`-th operand, chooses a layout of
  // `user` that minimizes its cost on that operand.  Returns null if it can't
  // decide the best layout.
  // Precondition: `user` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOutputLayoutFromOperandLayout(
      const Layout& operand_layout, const HloInstruction* user,
      int64_t operand_no);

  // Convenient wrapper for InstructionCanChangeLayout which can be overridden
  // in subclasses.
  virtual bool InstructionCanChangeLayoutInstance(
      const HloInstruction* instruction);

 private:
  // Initializes the layout assignment object for a new Run() call.
  Status Init(HloModule* module);

  // Adds constraints which must be satisfied for correctness on all
  // backends. Called once prior to propagating constraints.
  Status AddMandatoryConstraints(ChannelLayoutConstraints* channel_constraints,
                                 LayoutConstraints* constraints);

  // Return a vector containing the constraints which have been added to the
  // LayoutConstraints object since the construction of the object or since the
  // last time ConsumeAddedConstraints() has been called. This is used to
  // identify newly added constraints when propagating layouts.
  std::vector<const LayoutConstraint*> ConsumeAddedConstraints() {
    std::vector<const LayoutConstraint*> ret_vec(std::move(added_constraints_));
    added_constraints_.clear();
    return ret_vec;
  }
  void ClearAddedConstraints() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_40(mht_40_v, 772, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ClearAddedConstraints");
 added_constraints_.clear(); }

  // This method can be overridden to add backend-specific constraints to the
  // layout of the instructions of a computation. This method is called after
  // all mandatory constraints have been added via AddMandatoryConstraints
  // and before propagating constraints.
  virtual Status AddBackendConstraints(LayoutConstraints* constraints) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_41(mht_41_v, 781, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "AddBackendConstraints");

    return Status::OK();
  }

  // Construct constraints and assign layouts to all instructions in the
  // computation satisfying the given ComputationLayout, if not nullptr.
  // Otherwise the ComputationLayout will be calculated by propagating the
  // computation instruction constraints.
  // Layouts constraints are added, then propagated until all LogicalBuffers in
  // the computation are constrained.
  Status RunOnComputation(LayoutConstraints* constraints,
                          ChannelLayoutConstraints* channel_constraints);

  // Assign layouts to the instructions of a computation which satisfy the given
  // layout constraints. Copies may be added to satisfy the constraints. The
  // given LayoutConstraints must have layout constraints every logical buffer
  // in the computation.
  Status AssignLayouts(LayoutConstraints& constraints);

  // Propagates layout constraints from a set of initial constraints in order to
  // minimize the local cost of the computation. This propagation is *not*
  // required for correctness.
  Status PropagateConstraints(LayoutConstraints* constraints);

  Status PropagateBufferConstraintToOperands(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Check that all layouts in the module have been set and satisfy all
  // necessary conditions.
  Status CheckLayouts(HloModule* module);

  // Computes the ComputationLayout of the given constraints based of the
  // layouts assigned to parameters and root instruction. Also propagate
  // constraints to computation nested inside.
  Status CalculateComputationLayout(LayoutConstraints* constraints);

  // Clears all the layouts which can be cleared within a computation.
  Status ClearComputationLayouts(HloComputation* computation);

  // Clears the side effects of a previous pass, like added copy instructions.
  Status ClearPreviousPassSideEffects(HloModule* module);

  // Propagates the layouts computed by the layout assignment pass on the given
  // computation, to the computation layout passed in to this API.
  // This API propagates missing layout, and also checks that the caller
  // specified have been respected, by comparing those with the parameters and
  // root computation instruction.
  Status PropagateComputationLayouts(HloComputation* computation,
                                     ComputationLayout* computation_layout);

  // The pointer to the ComputationLayout passed as constructor parameter.
  ComputationLayout* entry_computation_layout_;

  // A copy of entry_computation_layout_ used to reset it to the initial values
  // during the multiple passes done by the layout assignment operation.
  ComputationLayout saved_entry_computation_layout_;
  // If set true, reverse the computation traversal order when assigning layout.
  bool reverse_computation_order_;

 protected:
  static constexpr int64_t kNumberOfPropagationRounds = 2;
  // Sets up the copy instruction according to the characteristic (sharding,
  // metadata, ...) of the reference instruction. The index argument is used
  // when the instruction is a tuple, and in such case the index represents
  // the location from where the copy instruction was created from.
  // If the index is empty, the whole sharding will be propagated, even in case
  // the instruction has a tuple sharding.
  static void SetupCopiedInstruction(const HloInstruction& instruction,
                                     HloInstruction* copy,
                                     const ShapeIndex& index);

  // Creates and returns a copy of the given instruction with a different
  // layout. Tuple-shaped instructions will be deep-copied, and the last Tuple
  // instruction producing the copy is returned.
  StatusOr<HloInstruction*> CreateCopyWithNewLayout(
      const Shape& shape_with_layout, HloInstruction* instruction);

  // Creates a copy of the given operand if the operand's layout does not match
  // the given layout. This copy replaces the use in the given instruction.
  // Tuple operands will be deep-copied.
  virtual Status CopyOperandIfLayoutsDiffer(const ShapeLayout& operand_layout,
                                            HloInstruction* instruction,
                                            int64_t operand_no);

  // Registers a copy instruction added by the layout assignment pass.
  void RegisterAddedCopy(HloInstruction* copy) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_42(mht_42_v, 870, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "RegisterAddedCopy");

    CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
    added_copies_.insert(copy);
  }

  // Adds a copy for the operand of an instruction, unless such operand is
  // already a copy, and has a single user (which is forcibly the instruction
  // itself).
  Status AddCopyForOperand(HloInstruction* instruction, int64_t operand_number);

  // Apply the channel layout constraints by populating the channel_constraints
  // data structure passed in at constructor time. Eventually adds copies in
  // case two ends of a channel ended up with a different leyout.
  Status ConstrainChannelLayouts(HloComputation* computation,
                                 ChannelLayoutConstraints* channel_constraints);

  // Resets the input ChannelLayoutConstraints to the original copy received
  // from the constructor input.
  void ResetChannelConstraints() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlayout_assignmentDTh mht_43(mht_43_v, 891, "", "./tensorflow/compiler/xla/service/layout_assignment.h", "ResetChannelConstraints");

    if (channel_layout_constraints_ != nullptr) {
      *channel_layout_constraints_ = channel_constraints_;
    }
  }

  // Adds constraints related to host Send/Recv instructions.
  Status BuildHostChannelConstraints(HloComputation* computation);

  // Module points to analysis that can be updated for cloned computations.
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  // The set of HLO instructions which lacked any layout constraint, thus
  // receiving propagated default layouts.
  absl::flat_hash_set<const HloInstruction*> unconstrained_layout_instructions_;

  std::function<bool(const HloInstruction*)>
      instruction_can_change_layout_func_;

  // CallGraph of the module, used to track callsites of each computation.
  std::unique_ptr<CallGraph> call_graph_;

  std::string ToString(const LayoutConstraints& constraints) const;

 private:
  // Map containing the layouts of all computations assigned so
  // far. Computations are handled in a topological sort where computations are
  // handled before their caller instructions so the layouts of caller
  // instructions can be set to match the computation.
  absl::flat_hash_map<const HloComputation*, std::unique_ptr<LayoutConstraints>>
      computation_layouts_;

  // Map from branch computations to the result layout they should apply.
  absl::flat_hash_map<HloComputation*, ComputationLayout> conditional_mismatch_;

  // Every copy added to the module by the layout assignment pass is registered
  // here.
  absl::flat_hash_set<HloInstruction*> added_copies_;

  // The pointer to the channel layout constraints passed in with the
  // constructor. If not nullptr, this is an input/output argument.
  ChannelLayoutConstraints* channel_layout_constraints_ = nullptr;

  // A copy of the input layout constraints used to reset the above pointer in
  // case we have to undo operations due to the multiple passes over the
  // computations/instructions.
  ChannelLayoutConstraints channel_constraints_;

  // Layout constraints for send/recv instructions which communicate with the
  // host.
  ChannelLayoutConstraints host_channel_constraints_;

  // Array-shaped buffers which have not yet been constrained.
  std::set<LogicalBuffer::Id> unconstrained_buffer_ids_;

  mutable absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<PointsToSet::BufferSet>>
      buffer_sets_cache_;

  // The set of BufferLayoutConstraints applied to the computation.
  absl::node_hash_map<const LogicalBuffer*, BufferLayoutConstraint>
      buffer_constraints_;

  // A vector which holds constraints as they are added. Can be cleared with
  // ClearAddedConstraints.
  std::vector<const LayoutConstraint*> added_constraints_;
  int64_t current_priority_ = LayoutConstraint::kBeginningPriority;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
