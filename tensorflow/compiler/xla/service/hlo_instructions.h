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

// All HloInstruction subclasses are put in this file.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh() {
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

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Base class for instructions with a dimensions vector.
class HloDimensionsInstruction : public HloInstruction {
 public:
  HloDimensionsInstruction(HloOpcode opcode, const Shape& shape,
                           absl::Span<const int64_t> dimensions)
      : HloInstruction(opcode, shape),
        dimensions_(dimensions.begin(), dimensions.end()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "HloDimensionsInstruction");
}

  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t>* mutable_dimensions() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_dimensions");
 return &dimensions_; }

  HloInstructionProto ToProto() const override;

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  std::vector<int64_t> dimensions_;
};

class HloBatchNormInstruction : public HloInstruction {
 public:
  // Returns feature_index field associated with the instruction. The index
  // represents the index of the feature dimension.
  int64_t feature_index() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_2(mht_2_v, 239, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "feature_index");
 return feature_index_; }

  // Returns a epsilon value associated with the instruction. The is a small
  // number added to the variance to avoid divide-by-zero error.
  float epsilon() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_3(mht_3_v, 246, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "epsilon");
 return epsilon_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  explicit HloBatchNormInstruction(HloOpcode opcode, const Shape& shape,
                                   HloInstruction* operand,
                                   HloInstruction* scale, float epsilon,
                                   int64_t feature_index);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // A small float number added to the variance to avoid divide-by-zero error.
  float epsilon_ = 0.0f;

  // An integer value representing the index of the feature dimension.
  int64_t feature_index_ = -1;
};

class HloBatchNormTrainingInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormTrainingInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, float epsilon, int64_t feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormInferenceInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormInferenceInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
      float epsilon, int64_t feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormGradInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormGradInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* mean, HloInstruction* variance,
      HloInstruction* grad_output, float epsilon, int64_t feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloFftInstruction : public HloInstruction {
 public:
  explicit HloFftInstruction(const Shape& shape, HloInstruction* operand,
                             FftType fft_type,
                             absl::Span<const int64_t> fft_length);
  FftType fft_type() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_4(mht_4_v, 320, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "fft_type");
 return fft_type_; }

  const std::vector<int64_t>& fft_length() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_5(mht_5_v, 325, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "fft_length");
 return fft_length_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes FFT type for an FFT instruction.
  FftType fft_type_ = FftType::FFT;

  // Indicates the FFT length for an FFT instruction.
  std::vector<int64_t> fft_length_;
};

class HloAsyncInstruction : public HloInstruction {
 public:
  HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                      absl::Span<HloInstruction* const> operands,
                      HloComputation* async_computation);
  HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                      HloInstruction* operand,
                      HloComputation* async_computation);
  HloInstruction* async_wrapped_instruction() const;
  HloOpcode async_wrapped_opcode() const;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCopyStartInstruction : public HloInstruction {
 public:
  explicit HloCopyStartInstruction(const Shape& shape, HloInstruction* operand,
                                   bool is_cross_program_prefetch);

  bool is_cross_program_prefetch() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_6(mht_6_v, 381, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "is_cross_program_prefetch");
 return is_cross_program_prefetch_; }
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  bool is_cross_program_prefetch_;
};

class HloCompareInstruction : public HloInstruction {
 public:
  explicit HloCompareInstruction(const Shape& shape, HloInstruction* lhs,
                                 HloInstruction* rhs,
                                 ComparisonDirection direction,
                                 absl::optional<Comparison::Type> type);
  ComparisonDirection direction() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_7(mht_7_v, 407, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "direction");
 return compare_.GetDirection(); }
  Comparison::Type type() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_8(mht_8_v, 411, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "type");
 return compare_.GetType(); }
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  Comparison compare_;
};

class HloTriangularSolveInstruction : public HloInstruction {
 public:
  explicit HloTriangularSolveInstruction(const Shape& shape, HloInstruction* a,
                                         HloInstruction* b,
                                         const TriangularSolveOptions& options);
  const TriangularSolveOptions& triangular_solve_options() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_9(mht_9_v, 436, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "triangular_solve_options");

    return triangular_solve_options_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  TriangularSolveOptions triangular_solve_options_;
};

class HloCholeskyInstruction : public HloInstruction {
 public:
  explicit HloCholeskyInstruction(const Shape& shape, HloInstruction* a,
                                  const CholeskyOptions& options);
  const CholeskyOptions& cholesky_options() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_10(mht_10_v, 466, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "cholesky_options");
 return cholesky_options_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  CholeskyOptions cholesky_options_;
};

// Class that represents instructions that synchronize and transfer data between
// partitioned devices. Send/Recv and collective instructions (AllReduce,
// AllToAll, CollectivePermute) belong to this instruction type. A group of
// instructions (of the same opcode) with the same channel_id communicate during
// execution.
class HloChannelInstruction : public HloInstruction {
 public:
  // Returns the channel id associated with the instruction. The id is
  // shared between each Send/Recv pair or a group of collective instructions
  // and is globally unique to identify each channel.
  absl::optional<int64_t> channel_id() const { return channel_id_; }
  void set_channel_id(const absl::optional<int64_t>& channel_id);

  // Whether this instruction is identical to `other` except for the values of
  // channel IDs, as long as both have channel IDs or neither has a channel ID.
  virtual bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_11(mht_11_v, 508, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "IdenticalSlowPathIgnoringChannelIdValues");

    return channel_id_.has_value() == other.channel_id().has_value();
  }

 protected:
  explicit HloChannelInstruction(HloOpcode opcode, const Shape& shape,
                                 const absl::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  // Do not override IdenticalSlowPath(). Override
  // IdenticalSlowPathIgnoringChannelIdValues() instead.
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const final;

  absl::optional<int64_t> channel_id_;
};

class HloSendRecvInstruction : public HloChannelInstruction {
 public:
  // Returns whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_12(mht_12_v, 537, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "is_host_transfer");
 return is_host_transfer_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  explicit HloSendRecvInstruction(HloOpcode opcode, const Shape& shape,
                                  int64_t channel_id, bool is_host_transfer);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer_;
};

class HloSendInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendInstruction(HloInstruction* operand, HloInstruction* token,
                              int64_t channel_id, bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSendDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendDoneInstruction(HloSendInstruction* operand,
                                  bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvInstruction(const Shape& shape, HloInstruction* token,
                              int64_t channel_id, bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvDoneInstruction(HloRecvInstruction* operand,
                                  bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectiveInstruction : public HloChannelInstruction {
 public:
  const std::vector<ReplicaGroup>& replica_groups() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_13(mht_13_v, 610, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "replica_groups");

    return replica_groups_;
  }

  // Returns true if the layout of the AllReduce is enforced by XLA client (as
  // the layout set in the shape). The only reason for the client to set the
  // layout is to separately compile computations that communicate with
  // AllReduce. Since this field is only set `true` by the client, the compiler
  // only needs to propagate existing values (e.g., Clone, X64Rewriter) or set
  // `false` for all other cases.
  //
  // When this is `true`, there may be communication endpoints outside the
  // current compilation unit, so the compiler considers this AllReduce as
  // side-effecting to disable compiler transformations. The compiler is free to
  // transform unconstrained AllReduces differently across compilation units.
  // It is an error for an HloModule to have a mix of constrained and
  // unconstrained AllReduce instructions (checked by HloVerifier).
  bool constrain_layout() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_14(mht_14_v, 630, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "constrain_layout");
 return constrain_layout_; }

 protected:
  explicit HloCollectiveInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const absl::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  std::vector<ReplicaGroup> replica_groups_;
  bool constrain_layout_;
};

class HloAllGatherInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllGatherInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const absl::optional<int64_t>& channel_id, bool use_global_device_ids);
  // Same as HloAllReduceInstruction::use_global_device_ids.
  bool use_global_device_ids() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_15(mht_15_v, 663, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "use_global_device_ids");
 return use_global_device_ids_; }

  // The dimension on which data from different participants are concatenated.
  int64_t all_gather_dimension() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_16(mht_16_v, 669, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "all_gather_dimension");
 return all_gather_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&all_gather_dimension_, 1);
  }

  void set_all_gather_dimension(int64_t dim) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_17(mht_17_v, 677, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_all_gather_dimension");
 all_gather_dimension_ = dim; }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t all_gather_dimension_;
  bool use_global_device_ids_;
};

// Base class for all-reduce and all-reduce scatter instructions.
class HloAllReduceInstructionBase : public HloCollectiveInstruction {
 public:
  explicit HloAllReduceInstructionBase(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const absl::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Returns true if the ids in the ReplicaGroup config represent a global id of
  // (replica_id * partition_count + partition_id) instead of a replica id.
  // This enables more flexible grouping of devices if this all-reduce is both
  // cross-partition and cross-replica.
  //
  // For example with 2 replicas and 4 partitions,
  // replica_groups={{0,1,4,5},{2,3,6,7}}, use_global_device_ids=true means that
  // group[0] = (0,0), (0,1), (1,0), (1,1)
  // group[1] = (0,2), (0,3), (1,2), (1,3)
  // where each pair is (replica_id, partition_id).
  bool use_global_device_ids() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_18(mht_18_v, 722, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "use_global_device_ids");
 return use_global_device_ids_; }
  void set_use_global_device_ids(bool value) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_19(mht_19_v, 726, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_use_global_device_ids");
 use_global_device_ids_ = value; }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

 private:
  bool use_global_device_ids_;
};

class HloAllReduceInstruction : public HloAllReduceInstructionBase {
 public:
  using HloAllReduceInstructionBase::HloAllReduceInstructionBase;

  // Returns true if the AllReduce does no communication, so it's equivalent
  // to a mem copy.
  bool IsNoop() const;

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloReduceScatterInstruction : public HloAllReduceInstructionBase {
 public:
  explicit HloReduceScatterInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const absl::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  // The dimension on which reduced data is scattered to different participants.
  int64_t scatter_dimension() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_20(mht_20_v, 770, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "scatter_dimension");
 return scatter_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&scatter_dimension_, 1);
  }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t scatter_dimension_;
};

class HloAllToAllInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const absl::optional<int64_t>& channel_id,
      const absl::optional<int64_t>& split_dimension);

  // AllToAll can optionally take a split dimension, which means that this
  // AllToAll takes a single (flattened) array operand and produces an array
  // output (instead of taking a list of operands and producing a tuple).
  //
  // split_dimension specifies which dimension in the operand is split across
  // devices in each replica_group, and also means the concatenated dimension
  // on the output (i.e., input and the output shapes are the same).
  absl::optional<int64_t> split_dimension() const { return split_dimension_; }
  void set_split_dimension(int64_t dim) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_21(mht_21_v, 813, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_split_dimension");
 split_dimension_ = dim; }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  absl::optional<int64_t> split_dimension_;
};

class HloCollectivePermuteInstruction : public HloChannelInstruction {
 public:
  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const absl::optional<int64_t>& channel_id);

  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* input,
      HloInstruction* output, HloInstruction* input_start_indices,
      HloInstruction* output_start_indices,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
      absl::Span<const std::vector<int64_t>> slice_sizes,
      const absl::optional<int64_t>& channel_id);

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

  const std::vector<std::vector<int64_t>>& dynamic_slice_sizes_list() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_22(mht_22_v, 856, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dynamic_slice_sizes_list");

    return slice_sizes_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
  const std::vector<std::vector<int64_t>> slice_sizes_;
};

class HloReverseInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReverseInstruction(const Shape& shape, HloInstruction* operand,
                                 absl::Span<const int64_t> dimensions);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloConcatenateInstruction : public HloDimensionsInstruction {
 public:
  explicit HloConcatenateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     int64_t dimension);
  // Accessor for the dimension in which a concatenate HLO should occur.
  int64_t concatenate_dimension() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_23(mht_23_v, 901, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "concatenate_dimension");

    return HloInstruction::dimensions(0);
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloReduceInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReduceInstruction(const Shape& shape,
                                absl::Span<HloInstruction* const> args,
                                absl::Span<const int64_t> dimensions_to_reduce,
                                HloComputation* reduce_computation);

  // Returns the number of input arrays (and, consequentially, the number of
  // init values) this reduce has.
  int64_t input_count() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_24(mht_24_v, 924, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "input_count");
 return operand_count() / 2; }

  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction* const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }

  // Returns the init values of the reduction.
  absl::Span<HloInstruction* const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSortInstruction : public HloDimensionsInstruction {
 public:
  explicit HloSortInstruction(const Shape& shape, int64_t dimension,
                              absl::Span<HloInstruction* const> operands,
                              HloComputation* compare, bool is_stable);
  // Returns the sort dimension for this instruction
  int64_t sort_dimension() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_25(mht_25_v, 956, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "sort_dimension");
 return HloInstruction::dimensions(0); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the key operand to this instruction.
  const HloInstruction* keys() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_26(mht_26_v, 963, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "keys");
 return operand(0); }
  HloInstruction* mutable_keys() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_27(mht_27_v, 967, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_keys");
 return mutable_operand(0); }
  // Returns the number of value operands.
  int64_t values_count() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_28(mht_28_v, 972, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "values_count");
 return operand_count() - 1; }
  bool is_stable() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_29(mht_29_v, 976, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "is_stable");
 return is_stable_; }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  bool is_stable_;
};

class HloTransposeInstruction : public HloDimensionsInstruction {
 public:
  explicit HloTransposeInstruction(const Shape& shape, HloInstruction* operand,
                                   absl::Span<const int64_t> dimensions);
  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBroadcastInstruction : public HloDimensionsInstruction {
 public:
  explicit HloBroadcastInstruction(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> broadcast_dimension);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloDynamicReshapeInstruction : public HloInstruction {
 public:
  explicit HloDynamicReshapeInstruction(
      const Shape& shape, HloInstruction* data_operand,
      absl::Span<HloInstruction* const> dim_sizes);

  // Returns the input dim sizes dimensions, which is operands[1:]
  absl::Span<HloInstruction* const> dim_sizes() const {
    return absl::MakeSpan(operands()).subspan(1, operand_count());
  }

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Returns the input dim size dimension, which is operands[1+i]
  HloInstruction* dim_sizes(int64_t i) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_30(mht_30_v, 1039, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dim_sizes");
 return operands()[i + 1]; }
};

class HloReshapeInstruction : public HloInstruction {
 public:
  explicit HloReshapeInstruction(const Shape& shape, HloInstruction* operand,
                                 int64_t inferred_dimension);
  int64_t inferred_dimension() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_31(mht_31_v, 1049, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "inferred_dimension");
 return inferred_dimension_; }
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  int64_t inferred_dimension_;
};

class HloMapInstruction : public HloInstruction {
 public:
  explicit HloMapInstruction(const Shape& shape,
                             absl::Span<HloInstruction* const> operands,
                             HloComputation* map_computation);
  // Returns the dimension sizes or numbers associated with this instruction.
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t>* mutable_dimensions() override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_32(mht_32_v, 1077, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_dimensions");
 return &dimensions_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64_t> dimensions_;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape& shape, HloInstruction* operand,
                               absl::Span<const int64_t> start_indices,
                               absl::Span<const int64_t> limit_indices,
                               absl::Span<const int64_t> strides);

  HloInstructionProto ToProto() const override;

  // Returns the start index in the given dimension for a slice node.
  int64_t slice_starts(int64_t dimension) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_33(mht_33_v, 1111, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_starts");

    return slice_starts_[dimension];
  }
  const std::vector<int64_t>& slice_starts() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_34(mht_34_v, 1117, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_starts");
 return slice_starts_; }
  std::vector<int64_t>* mutable_slice_starts() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_35(mht_35_v, 1121, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_slice_starts");
 return &slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  int64_t slice_limits(int64_t dimension) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_36(mht_36_v, 1128, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_limits");

    return slice_limits_[dimension];
  }
  const std::vector<int64_t>& slice_limits() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_37(mht_37_v, 1134, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_limits");
 return slice_limits_; }
  std::vector<int64_t>* mutable_slice_limits() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_38(mht_38_v, 1138, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_slice_limits");
 return &slice_limits_; }

  // Returns the stride in the given dimension for a slice node.
  int64_t slice_strides(int64_t dimension) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_39(mht_39_v, 1144, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_strides");

    return slice_strides_[dimension];
  }
  const std::vector<int64_t>& slice_strides() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_40(mht_40_v, 1150, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_strides");
 return slice_strides_; }
  std::vector<int64_t>* mutable_slice_strides() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_41(mht_41_v, 1154, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_slice_strides");
 return &slice_strides_; }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64_t> slice_starts_;
  std::vector<int64_t> slice_limits_;
  std::vector<int64_t> slice_strides_;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(Literal literal);
  explicit HloConstantInstruction(Literal literal, const Shape& shape);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape& shape);
  // Returns the literal associated with this instruction.
  const Literal& literal() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_42(mht_42_v, 1184, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "literal");
 return *literal_; }
  // Returns the (mutable) literal associated with this instruction.
  Literal* mutable_literal() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_43(mht_43_v, 1189, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_literal");
 return &literal_.value(); }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_44(mht_44_v, 1194, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "HasLiteral");
 return literal_.has_value(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Change the layout for an Constant Hlo instruction to match new_layout.  For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64_t>& operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  absl::optional<Literal> literal_;
};

class HloTraceInstruction : public HloInstruction {
 public:
  explicit HloTraceInstruction(const std::string& tag, HloInstruction* operand);
  // Returns a tag to be used in tracing.
  std::string TracingTag() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_45(mht_45_v, 1228, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "TracingTag");
 return literal_.GetR1U8AsString(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Literal literal_;
};

class HloFusionInstruction : public HloInstruction {
 public:
  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                HloInstruction* fused_root);

  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                absl::Span<HloInstruction* const> operands,
                                HloComputation* fusion_computation);

  ~HloFusionInstruction() override;

  void ClearCalledComputations() override;

  // When a fusion instruction is being destructed, clear the back pointer of
  // its fusion computation, to avoid referencing freed memory.
  void ClearFusionComputationInstruction();

  std::string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Adds a new operand the fusion instruction.
  HloInstruction* AddFusionOperand(HloInstruction* new_operand);

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Precondition: 'instruction_to_merge' must be an operand of 'this'.
  void MergeFusionInstruction(HloFusionInstruction* instruction_to_merge);

  // Merges the fused instructions from instruction_to_merge into the fused
  // instruction set of 'this' and generates multioutput fusion instructions.
  // All the users of instruction_to_merge will be redirected to 'this'
  // instruction. instruction_to_merge will be removed from its parent
  // computation.
  void MergeFusionInstructionIntoMultiOutput(
      HloFusionInstruction* instruction_to_merge);

  // Fuses the given instruction in this fusion instruction. instruction_to_fuse
  // is cloned and the clone is placed in the fusion
  // instruction. instruction_to_fuse is unchanged. Instruction is cloned rather
  // than moved to cleanly handle the case where the instruction has a use
  // outside the fusion instruction. Moving such an instruction into a fusion
  // instruction would violate the single-result invariant of HLO instructions
  // and significantly complicate code generation.
  HloInstruction* FuseInstruction(HloInstruction* instruction_to_fuse) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_46(mht_46_v, 1292, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "FuseInstruction");

    return FuseInstructionInternal(instruction_to_fuse);
  }

  // Fuses the given instruction in this fusion instruction and generates a
  // multioutput fusion instruction. A clone of the instruction_to_fuse will
  // be part of the output of fusion instructions. The users of
  // instruction_to_fuse will be redirected to this fusion instructions.
  // instruction_to_fuse is unchanged otherwise.
  HloInstruction* FuseInstructionIntoMultiOutput(
      HloInstruction* instruction_to_fuse) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_47(mht_47_v, 1305, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "FuseInstructionIntoMultiOutput");

    return FuseInstructionInternal(instruction_to_fuse, /* add_output */ true);
  }

  // Returns the computation for this fused instruction.
  HloComputation* fused_instructions_computation() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  HloInstruction* fused_expression_root() const;

  // Returns the list of fused instructions inside this fusion instruction.  The
  // returned type is a range of HloInstruction*s.
  const tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Gets the number of instructions inside this fusion instruction.
  int64_t fused_instruction_count() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  HloInstruction* fused_parameter(int64_t parameter_number) const;

  // Returns the vector of fused parameters inside this fusion instruction.
  const std::vector<HloInstruction*>& fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  const bool IsMultiOutputFusion() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_48(mht_48_v, 1341, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "IsMultiOutputFusion");

    return fused_expression_root()->opcode() == HloOpcode::kTuple;
  }

  FusionKind fusion_kind() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_49(mht_49_v, 1348, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "fusion_kind");
 return fusion_kind_; }

  void set_fusion_kind(FusionKind kind) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_50(mht_50_v, 1353, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_fusion_kind");
 fusion_kind_ = kind; }

  // If multiple operands are the same instruction, keeps only one of them.
  Status DeduplicateFusionOperands();

 private:
  // Fuses the given instruction into this fusion instruction.
  // instruction_to_fuse is cloned and the clone is placed in the fusion
  // instruction.  The users of instruction_to_fuse will be redirected to this
  // fusion instruction. instruction_to_fuse is unchanged otherwise. When
  // add_output is true, a clone of the instruction_to_fuse will be added as
  // additional output resulting in a multi-output fusion.
  HloInstruction* FuseInstructionInternal(HloInstruction* instruction_to_fuse,
                                          bool add_output = false);
  // Clones the given instruction_to_fuse and insert the clone into this fusion
  // instruction. If add_output is true, a clone of instruction_to_fuse will
  // be in the output of the this fusion instruction (part of the tuple of the
  // fusion root).
  HloInstruction* CloneAndFuseInternal(HloInstruction* instruction_to_fuse,
                                       bool add_output = false);

  bool IsElementwiseImpl(
      const absl::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The type of the fusion. Used by kFusion only.
  FusionKind fusion_kind_;
};

class HloRngInstruction : public HloInstruction {
 public:
  explicit HloRngInstruction(const Shape& shape,
                             RandomDistribution distribution,
                             absl::Span<HloInstruction* const> parameters);
  // Returns the random distribution for this rng node.
  RandomDistribution random_distribution() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_51(mht_51_v, 1401, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "random_distribution");
 return distribution_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The distribution requested for random number generation.
  RandomDistribution distribution_;
};

class HloParameterInstruction : public HloInstruction {
 public:
  explicit HloParameterInstruction(int64_t parameter_number, const Shape& shape,
                                   const std::string& name);
  int64_t parameter_number() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_52(mht_52_v, 1430, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "parameter_number");
 return parameter_number_; }

  // Sets and gets the whether all replicas will receive the same parameter data
  // for each leaf buffer in data parallelism.
  void set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_53(mht_53_v, 1438, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_parameter_replicated_at_leaf_buffers");

    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_.emplace(
        parameter_replicated_at_leaf_buffers.begin(),
        parameter_replicated_at_leaf_buffers.end());
  }
  void set_parameter_replicated_at_leaf_buffers(
      const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_54(mht_54_v, 1449, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_parameter_replicated_at_leaf_buffers");

    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_ =
        parameter_replicated_at_leaf_buffers;
  }
  const absl::optional<std::vector<bool>>&
  parameter_replicated_at_leaf_buffers() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_55(mht_55_v, 1459, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "parameter_replicated_at_leaf_buffers");

    return parameter_replicated_at_leaf_buffers_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t parameter_number_ = 0;

  // Specifies whether each buffer has the same parameter value on all replicas
  // in data parallelism.
  absl::optional<std::vector<bool>> parameter_replicated_at_leaf_buffers_;
};

class HloGetTupleElementInstruction : public HloInstruction {
 public:
  explicit HloGetTupleElementInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         int64_t index);
  // Returns the tuple index associated with this instruction.
  int64_t tuple_index() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_56(mht_56_v, 1497, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "tuple_index");
 return tuple_index_; }
  // Sets the tuple index associated with this instruction.
  void set_tuple_index(int64_t new_tuple_index) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_57(mht_57_v, 1502, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_tuple_index");

    tuple_index_ = new_tuple_index;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t tuple_index_ = -1;
};

class HloReducePrecisionInstruction : public HloInstruction {
 public:
  explicit HloReducePrecisionInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         const int exponent_bits,
                                         const int mantissa_bits);
  // Returns the number of exponent bits for a reduce-precision node.
  int32_t exponent_bits() const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_58(mht_58_v, 1533, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "exponent_bits");
 return exponent_bits_; }
  // Returns the number of mantissa bits for a reduce-precision node.
  int32_t mantissa_bits() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_59(mht_59_v, 1538, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mantissa_bits");
 return mantissa_bits_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The bit sizes for a reduce-precision operation.
  int32_t exponent_bits_ = 0;
  int32_t mantissa_bits_ = 0;
};

class HloInfeedInstruction : public HloInstruction {
 public:
  explicit HloInfeedInstruction(const Shape& infeed_shape,
                                HloInstruction* token_operand,
                                const std::string& config);
  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  std::string infeed_config() const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_60(mht_60_v, 1570, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "infeed_config");
 return infeed_config_; }
  void set_infeed_config(const std::string& config) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_61(mht_61_v, 1575, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_infeed_config");
 infeed_config_ = config; }
  // Returns the shape of the data received by the infeed. This is not the same
  // as the shape of the infeed instruction which produces a tuple containing
  // the infeed data shape and a TOKEN.
  const Shape& infeed_shape() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_62(mht_62_v, 1582, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "infeed_shape");

    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape()));
    return ShapeUtil::GetSubshape(shape(), {0});
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the infeed configuration.
  std::string infeed_config_;
};

class HloOutfeedInstruction : public HloInstruction {
 public:
  explicit HloOutfeedInstruction(const Shape& outfeed_shape,
                                 HloInstruction* operand,
                                 HloInstruction* token_operand,
                                 absl::string_view outfeed_config);
  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_63(mht_63_v, 1615, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "outfeed_shape");
 return outfeed_shape_; }
  // Returns the mutable shape for the Outfeed instruction.
  Shape* mutable_outfeed_shape() {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_64(mht_64_v, 1620, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_outfeed_shape");
 return &outfeed_shape_; }
  // Returns the config for the Outfeed instruction.
  const std::string& outfeed_config() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_65(mht_65_v, 1625, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "outfeed_config");
 return outfeed_config_; }
  void set_outfeed_config(const std::string& config) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_66(mht_66_v, 1630, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_outfeed_config");

    outfeed_config_ = config;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Shape of outfeed request.
  Shape outfeed_shape_;
  // Outfeed configuration information, only present for kOutfeed.
  std::string outfeed_config_;
};

class HloConvolutionInstruction : public HloInstruction {
 public:
  explicit HloConvolutionInstruction(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      int64_t feature_group_count, int64_t batch_group_count,
      const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);
  const Window& window() const override {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_67(mht_67_v, 1665, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "window");
 return window_; }
  void set_window(const Window& window) override {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_68(mht_68_v, 1669, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_window");
 window_ = window; }
  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_69(mht_69_v, 1673, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "convolution_dimension_numbers");

    return convolution_dimension_numbers_;
  }
  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_70(mht_70_v, 1680, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_convolution_dimension_numbers");

    convolution_dimension_numbers_ = dnums;
  }
  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64_t feature_group_count() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_71(mht_71_v, 1688, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "feature_group_count");
 return feature_group_count_; }
  void set_feature_group_count(int64_t num_feature_groups) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_72(mht_72_v, 1692, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_feature_group_count");

    feature_group_count_ = num_feature_groups;
  }
  // The number of batch groups. Must be a divisor of the input batch dimension.
  int64_t batch_group_count() const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_73(mht_73_v, 1699, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "batch_group_count");
 return batch_group_count_; }
  void set_batch_group_count(int64_t num_batch_groups) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_74(mht_74_v, 1703, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_batch_group_count");

    batch_group_count_ = num_batch_groups;
  }

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  const PrecisionConfig& precision_config() const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_75(mht_75_v, 1717, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "precision_config");
 return precision_config_; }
  PrecisionConfig* mutable_precision_config() {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_76(mht_76_v, 1721, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_precision_config");
 return &precision_config_; }

  std::string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64_t feature_group_count_;
  // The number of batch groups. Must be a divisor of the input batch dimension.
  int64_t batch_group_count_;
  // Describes the window used for a convolution.
  Window window_;
  // Describes the dimension numbers used for a convolution.
  ConvolutionDimensionNumbers convolution_dimension_numbers_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class HloReduceWindowInstruction : public HloInstruction {
 public:
  explicit HloReduceWindowInstruction(const Shape& shape,
                                      HloInstruction* operand,
                                      HloInstruction* init_value,
                                      const Window& window,
                                      HloComputation* reduce_computation);
  explicit HloReduceWindowInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloInstruction* const> init_values, const Window& window,
      HloComputation* reduce_computation);
  const Window& window() const override {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_77(mht_77_v, 1766, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "window");
 return window_; }
  void set_window(const Window& window) override {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_78(mht_78_v, 1770, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_window");
 window_ = window; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the number of input arrays (and, consequentially, the number of
  // init values) this reduce has.
  int64_t input_count() const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_79(mht_79_v, 1778, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "input_count");
 return operand_count() / 2; }
  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction* const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }
  // Returns the init values of the reduction.
  absl::Span<HloInstruction* const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }
  // Returns the shapes of input tensors to be reduced.
  absl::InlinedVector<const Shape*, 2> input_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    for (const auto* op : inputs()) {
      VLOG(2) << "Pushing input array shape for: " << op->ToString() << "\n";
      shapes.push_back(&op->shape());
      VLOG(2) << "Pushed shape: " << shapes.back()->ToString() << "\n";
    }
    return shapes;
  }
  // Returns the init values of the reduction.
  absl::InlinedVector<const Shape*, 2> init_value_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    for (const auto* op : init_values()) {
      shapes.push_back(&op->shape());
    }
    return shapes;
  }
  // Returns the shapes of the reduced output tensors.
  absl::InlinedVector<const Shape*, 2> output_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    if (shape().IsArray()) {
      shapes.push_back(&shape());
    } else {
      for (const Shape& tuple_element_shape : shape().tuple_shapes()) {
        shapes.push_back(&tuple_element_shape);
      }
    }
    return shapes;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  Window window_;
};

class HloSelectAndScatterInstruction : public HloInstruction {
 public:
  explicit HloSelectAndScatterInstruction(
      const Shape& shape, HloInstruction* operand, HloComputation* select,
      const Window& window, HloInstruction* source, HloInstruction* init_value,
      HloComputation* scatter);
  const Window& window() const override {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_80(mht_80_v, 1842, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "window");
 return window_; }
  void set_window(const Window& window) override {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_81(mht_81_v, 1846, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_window");
 window_ = window; }
  // Gets/sets the select or scatter HloComputation for SelectAndScatter. The
  // setters should only be called by HloModule or HloComputation methods.
  HloComputation* select() const {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_82(mht_82_v, 1852, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "select");

    return called_computations()[kSelectComputationIndex];
  }

  HloComputation* scatter() const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_83(mht_83_v, 1859, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "scatter");

    return called_computations()[kScatterComputationIndex];
  }

  void set_select(HloComputation* computation) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_84(mht_84_v, 1866, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_select");

    // Don't allow changing the computation for fused instructions so we don't
    // have to recompute called_instructions for the entire fusion instruction.
    CHECK(!IsFused());
    set_called_computation(kSelectComputationIndex, computation);
  }

  void set_scatter(HloComputation* computation) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_85(mht_85_v, 1876, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_scatter");

    // Don't allow changing the computation for fused instructions so we don't
    // have to recompute called_instructions for the entire fusion instruction.
    CHECK(!IsFused());
    set_called_computation(kScatterComputationIndex, computation);
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Window window_;
};

class HloCustomCallInstruction : public HloInstruction {
 public:
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with constrained layout. 'shape' and
  // 'operands_with_layout' must all have layouts.
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           absl::Span<const Shape> operand_shapes_with_layout,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with a to_apply computation.
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           HloComputation* to_apply,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with multiple computations.
  HloCustomCallInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloComputation* const> called_computations,
      absl::string_view custom_call_target, std::string opaque,
      CustomCallApiVersion api_version);

  const Window& window() const override {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_86(mht_86_v, 1934, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "window");

    CHECK(window_ != nullptr);
    return *window_;
  }

  void set_window(const Window& window) override {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_87(mht_87_v, 1942, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_window");

    window_ = absl::make_unique<Window>(window);
  }

  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_88(mht_88_v, 1949, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "convolution_dimension_numbers");

    CHECK(convolution_dimension_numbers_ != nullptr);
    return *convolution_dimension_numbers_;
  }

  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_89(mht_89_v, 1958, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_convolution_dimension_numbers");

    convolution_dimension_numbers_ =
        absl::make_unique<ConvolutionDimensionNumbers>(dnums);
  }
  // TODO(jpienaar): Remove this accessor in the follow up.
  const std::string& opaque() const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_90(mht_90_v, 1966, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "opaque");
 return raw_backend_config_string(); }
  const std::string& custom_call_target() const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_91(mht_91_v, 1970, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "custom_call_target");
 return custom_call_target_; }
  void set_custom_call_target(absl::string_view target) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_92(mht_92_v, 1975, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_custom_call_target");

    custom_call_target_ = std::string(target);
  }
  void set_feature_group_count(int64_t feature_group_count) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_93(mht_93_v, 1981, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_feature_group_count");

    feature_group_count_ = feature_group_count;
  }
  void set_batch_group_count(int64_t batch_group_count) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_94(mht_94_v, 1987, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_batch_group_count");

    batch_group_count_ = batch_group_count;
  }
  // Sets whether this custom call has a side-effect - by default a custom call
  // has no side-effects.
  void set_custom_call_has_side_effect(bool custom_call_has_side_effect) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_95(mht_95_v, 1995, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_custom_call_has_side_effect");

    custom_call_has_side_effect_ = custom_call_has_side_effect;
  }
  int64_t feature_group_count() const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_96(mht_96_v, 2001, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "feature_group_count");
 return feature_group_count_; }
  int64_t batch_group_count() const {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_97(mht_97_v, 2005, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "batch_group_count");
 return batch_group_count_; }
  bool custom_call_has_side_effect() const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_98(mht_98_v, 2009, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "custom_call_has_side_effect");

    return custom_call_has_side_effect_;
  }
  // Returns padding type used for ops like convolution.
  PaddingType padding_type() const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_99(mht_99_v, 2016, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "padding_type");
 return padding_type_; }

  void set_padding_type(PaddingType padding_type) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_100(mht_100_v, 2021, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_padding_type");

    padding_type_ = padding_type;
  }

  // Returns the literal associated with this instruction.
  const Literal& literal() const {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_101(mht_101_v, 2029, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "literal");
 return *literal_; }
  // Set the value of literal to a new one.
  void set_literal(Literal&& literal) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_102(mht_102_v, 2034, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_literal");
 literal_.emplace(std::move(literal)); }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_103(mht_103_v, 2039, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "HasLiteral");
 return literal_.has_value(); }

  const PrecisionConfig& precision_config() const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_104(mht_104_v, 2044, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "precision_config");
 return precision_config_; }
  PrecisionConfig* mutable_precision_config() {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_105(mht_105_v, 2048, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_precision_config");
 return &precision_config_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Returns whether the result and operand layouts are constrained.
  bool layout_constrained() const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_106(mht_106_v, 2057, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "layout_constrained");
 return layout_constrained_; }

  // Returns the shapes (with layout) of the operands. CHECKs if this custom
  // call does not have constrained layouts.
  const std::vector<Shape>& operand_shapes_with_layout() const {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_107(mht_107_v, 2064, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "operand_shapes_with_layout");

    CHECK(layout_constrained());
    return operand_shapes_with_layout_;
  }
  // Gets a list of output/operand buffer pairs that alias each other, where the
  // output buffer is represented as a ShapeIndex, and the operand buffer is
  // represented as the operand index and the ShapeIndex. By default this list
  // is empty.
  const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>&
  output_to_operand_aliasing() const {
    return output_to_operand_aliasing_;
  }
  // Sets the list of output/operand buffer pairs that alias each other.
  void set_output_to_operand_aliasing(
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          aliasing) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_108(mht_108_v, 2082, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_output_to_operand_aliasing");

    output_to_operand_aliasing_ = std::move(aliasing);
  }
  void set_custom_call_schedule(CustomCallSchedule custom_call_schedule) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_109(mht_109_v, 2088, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_custom_call_schedule");

    custom_call_schedule_ = custom_call_schedule;
  }
  CustomCallSchedule custom_call_schedule() const {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_110(mht_110_v, 2094, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "custom_call_schedule");

    return custom_call_schedule_;
  }
  void set_api_version(CustomCallApiVersion api_version) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_111(mht_111_v, 2100, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_api_version");

    api_version_ = api_version;
  }
  CustomCallApiVersion api_version() const {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_112(mht_112_v, 2106, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "api_version");
 return api_version_; }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // Name of a global symbol to call.
  std::string custom_call_target_;
  // Describes the window in a windowed operation such as convolution.
  std::unique_ptr<Window> window_;
  // Describes the dimension numbers used for a convolution.
  std::unique_ptr<ConvolutionDimensionNumbers> convolution_dimension_numbers_;
  // The number of feature groups. This is used for grouped convolutions.
  int64_t feature_group_count_;
  int64_t batch_group_count_;
  // Whether the result and operand layouts are constrained.
  bool layout_constrained_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results for convolution instructions.
  PrecisionConfig precision_config_;
  // Describes the padding type for convolution instructions.
  PaddingType padding_type_;
  // For layout-constrained custom calls, this vector holds the shape with
  // layout for each operand.
  std::vector<Shape> operand_shapes_with_layout_;
  // Whether this custom call has a side-effect.
  bool custom_call_has_side_effect_;
  // A list of output/operand buffer pairs that alias each other. See comment of
  // output_to_operand_aliasing().
  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
      output_to_operand_aliasing_;
  absl::optional<Literal> literal_;
  // A custom-call schedule hint.
  CustomCallSchedule custom_call_schedule_;
  // The version of the API used by the custom call function.
  // TODO(b/189822916): Remove this field when all clients are migrated to the
  // status-returning API.
  CustomCallApiVersion api_version_;
};

class HloPadInstruction : public HloInstruction {
 public:
  explicit HloPadInstruction(const Shape& shape, HloInstruction* operand,
                             HloInstruction* padding_value,
                             const PaddingConfig& padding_config);
  // Returns the padding configuration for a pad node.
  const PaddingConfig& padding_config() const {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_113(mht_113_v, 2162, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "padding_config");
 return padding_config_; }
  PaddingConfig* mutable_padding_config() {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_114(mht_114_v, 2166, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_padding_config");
 return &padding_config_; }
  // Returns the operand being padded.
  const HloInstruction* padded_operand() const {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_115(mht_115_v, 2171, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "padded_operand");
 return operand(0); }
  HloInstruction* mutable_padded_operand() {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_116(mht_116_v, 2175, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_padded_operand");
 return mutable_operand(0); }
  // Returns the padding value.
  const HloInstruction* padding_value() const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_117(mht_117_v, 2180, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "padding_value");
 return operand(1); }
  HloInstruction* mutable_padding_value() {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_118(mht_118_v, 2184, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_padding_value");
 return mutable_operand(1); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The padding configuration that describes the edge padding and interior
  // padding of this pad instruction.
  PaddingConfig padding_config_;
};

class HloDynamicIndexInstruction : public HloInstruction {
 public:
  explicit HloDynamicIndexInstruction(HloOpcode opcode, const Shape& shape)
      : HloInstruction(opcode, shape) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_119(mht_119_v, 2211, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "HloDynamicIndexInstruction");
}
  virtual int64_t first_index_operand_number() const = 0;

  // Returns a subspan of operands which represent the start indices.
  absl::Span<HloInstruction* const> index_operands() const {
    return absl::MakeSpan(operands()).subspan(first_index_operand_number());
  }

  // Returns the shapes of the index operands.
  std::vector<Shape> index_shapes() const {
    std::vector<Shape> shapes;
    auto indices = index_operands();
    for (const HloInstruction* index : indices) {
      shapes.push_back(index->shape());
    }
    return shapes;
  }
};

class HloDynamicSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicSliceInstruction(const Shape& shape,
                                      HloInstruction* operand,
                                      HloInstruction* start_indices,
                                      absl::Span<const int64_t> slice_sizes);
  explicit HloDynamicSliceInstruction(
      const Shape& shape, HloInstruction* operand,
      absl::Span<HloInstruction* const> start_indices,
      absl::Span<const int64_t> slice_sizes);
  // Old methods kept for smooth subclassing transition END.
  // Returns the size of the slice in the given dimension for a dynamic
  // slice node.
  int64_t slice_sizes(int64_t dimension) const {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_120(mht_120_v, 2246, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "slice_sizes");

    return dynamic_slice_sizes_[dimension];
  }
  const std::vector<int64_t>& dynamic_slice_sizes() const {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_121(mht_121_v, 2252, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dynamic_slice_sizes");

    return dynamic_slice_sizes_;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  int64_t first_index_operand_number() const override {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_122(mht_122_v, 2261, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "first_index_operand_number");
 return 1; }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64_t> dynamic_slice_sizes_;
};

class HloDynamicUpdateSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicUpdateSliceInstruction(const Shape& shape,
                                            HloInstruction* operand,
                                            HloInstruction* update,
                                            HloInstruction* start_indices);
  explicit HloDynamicUpdateSliceInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* update,
      absl::Span<HloInstruction* const> start_indices);

  int64_t first_index_operand_number() const override {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_123(mht_123_v, 2293, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "first_index_operand_number");
 return 2; }
};

class HloGatherInstruction : public HloInstruction {
 public:
  explicit HloGatherInstruction(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted);
  const GatherDimensionNumbers& gather_dimension_numbers() const {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_124(mht_124_v, 2306, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "gather_dimension_numbers");

    CHECK(gather_dimension_numbers_ != nullptr);
    return *gather_dimension_numbers_;
  }
  absl::Span<const int64_t> gather_slice_sizes() const {
    return gather_slice_sizes_;
  }
  bool indices_are_sorted() const {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_125(mht_125_v, 2316, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "indices_are_sorted");
 return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_126(mht_126_v, 2320, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_indices_are_sorted");

    indices_are_sorted_ = indices_are_sorted;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of GatherDimensionNumbers.
  static GatherDimensionNumbers MakeGatherDimNumbers(
      absl::Span<const int64_t> offset_dims,
      absl::Span<const int64_t> collapsed_slice_dims,
      absl::Span<const int64_t> start_index_map, int64_t index_vector_dim);
  // Returns the dump string of the given gather dimension numbers.
  static std::string GatherDimensionNumbersToString(
      const GatherDimensionNumbers& gather_dimension_numbers);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<GatherDimensionNumbers> gather_dimension_numbers_;
  std::vector<int64_t> gather_slice_sizes_;
  bool indices_are_sorted_;
};

class HloScatterInstruction : public HloInstruction {
 public:
  explicit HloScatterInstruction(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* scatter_indices, HloInstruction* updates,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers,
      bool indices_are_sorted, bool unique_indices);
  const ScatterDimensionNumbers& scatter_dimension_numbers() const {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_127(mht_127_v, 2362, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "scatter_dimension_numbers");

    CHECK(scatter_dimension_numbers_ != nullptr);
    return *scatter_dimension_numbers_;
  }
  bool indices_are_sorted() const {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_128(mht_128_v, 2369, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "indices_are_sorted");
 return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_129(mht_129_v, 2373, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_indices_are_sorted");

    indices_are_sorted_ = indices_are_sorted;
  }
  bool unique_indices() const override {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_130(mht_130_v, 2379, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "unique_indices");
 return unique_indices_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of ScatterDimensionNumbers.
  static ScatterDimensionNumbers MakeScatterDimNumbers(
      absl::Span<const int64_t> update_window_dims,
      absl::Span<const int64_t> inserted_window_dims,
      absl::Span<const int64_t> scatter_dims_to_operand_dims,
      int64_t index_vector_dim);
  // Returns the dump string of the given scatter dimension numbers.
  static std::string ScatterDimensionNumbersToString(
      const ScatterDimensionNumbers& scatter_dimension_numbers);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<ScatterDimensionNumbers> scatter_dimension_numbers_;
  bool indices_are_sorted_;
  bool unique_indices_;
};

class HloIotaInstruction : public HloInstruction {
 public:
  explicit HloIotaInstruction(const Shape& shape, int64_t iota_dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t iota_dimension() const {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_131(mht_131_v, 2418, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "iota_dimension");
 return iota_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&iota_dimension_, 1);
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t iota_dimension_;
};

class HloDotInstruction : public HloInstruction {
 public:
  // Creates a dot op with operands 'lhs' and 'rhs' with contracting and batch
  // dimensions specified in 'dimension_numbers'.
  explicit HloDotInstruction(const Shape& shape, HloInstruction* lhs,
                             HloInstruction* rhs,
                             const DotDimensionNumbers& dimension_numbers,
                             const PrecisionConfig& precision_config);

  // Returns data on the dimension numbers used for a dot operation.
  const DotDimensionNumbers& dot_dimension_numbers() const {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_132(mht_132_v, 2453, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dot_dimension_numbers");

    return dot_dimension_numbers_;
  }

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  const PrecisionConfig& precision_config() const {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_133(mht_133_v, 2467, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "precision_config");
 return precision_config_; }
  PrecisionConfig* mutable_precision_config() {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_134(mht_134_v, 2471, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "mutable_precision_config");
 return &precision_config_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the dimension numbers used for a dot.
  DotDimensionNumbers dot_dimension_numbers_;

  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class HloDomainInstruction : public HloInstruction {
 public:
  explicit HloDomainInstruction(
      const Shape& shape, HloInstruction* operand,
      std::unique_ptr<DomainMetadata> operand_side_metadata,
      std::unique_ptr<DomainMetadata> user_side_metadata);

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Retrieves the operand side metadata of a kDomain instruction.
  const DomainMetadata& operand_side_metadata() const {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_135(mht_135_v, 2510, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "operand_side_metadata");

    return *operand_side_metadata_;
  }
  // Retrieves the user side metadata of a kDomain instruction.
  const DomainMetadata& user_side_metadata() const {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_136(mht_136_v, 2517, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "user_side_metadata");

    return *user_side_metadata_;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<DomainMetadata> operand_side_metadata_;
  std::unique_ptr<DomainMetadata> user_side_metadata_;
};

class HloGetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloGetDimensionSizeInstruction(const Shape& shape,
                                          HloInstruction* operand,
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_137(mht_137_v, 2547, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dimension");
 return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t dimension_;
};

class HloSetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloSetDimensionSizeInstruction(const Shape& shape,
                                          HloInstruction* operand,
                                          HloInstruction* val,
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_138(mht_138_v, 2577, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "dimension");
 return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t dimension_;
};

class HloRngGetAndUpdateStateInstruction : public HloInstruction {
 public:
  explicit HloRngGetAndUpdateStateInstruction(const Shape& shape,
                                              int64_t delta);

  // Returns the delta value.
  int64_t delta() const {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_139(mht_139_v, 2605, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "delta");
 return delta_; }
  void set_delta(int64_t delta) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_140(mht_140_v, 2609, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "set_delta");
 delta_ = delta; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t delta_;
};

class HloRngBitGeneratorInstruction : public HloInstruction {
 public:
  HloRngBitGeneratorInstruction(const Shape& shape, HloInstruction* state,
                                RandomAlgorithm algorithm);

  RandomAlgorithm algorithm() const {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTh mht_141(mht_141_v, 2636, "", "./tensorflow/compiler/xla/service/hlo_instructions.h", "algorithm");
 return algorithm_; }
  HloInstructionProto ToProto() const override;

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  RandomAlgorithm algorithm_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
