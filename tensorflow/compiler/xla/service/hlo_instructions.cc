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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

using absl::CEscape;
using absl::StrAppend;
using absl::StrCat;
using absl::StrJoin;

bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "IsInstructionElementwiseOnOperand");

  const auto operand_indices = instruction->OperandIndices(operand);
  return absl::c_all_of(operand_indices, [instruction](int64_t operand_index) {
    return instruction->IsElementwiseOnOperand(operand_index);
  });
}

std::string PrecisionConfigToString(const PrecisionConfig& precision_config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_1(mht_1_v, 233, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "PrecisionConfigToString");

  if (absl::c_all_of(
          precision_config.operand_precision(), [](int32_t precision) {
            return static_cast<PrecisionConfig::Precision>(precision) ==
                   PrecisionConfig::DEFAULT;
          })) {
    return "";
  }

  return StrCat(
      "operand_precision={",
      StrJoin(
          precision_config.operand_precision(), ",",
          [](std::string* out, int32_t precision) {
            CHECK(PrecisionConfig::Precision_IsValid(precision)) << precision;
            StrAppend(out,
                      PrecisionToString(
                          static_cast<PrecisionConfig::Precision>(precision)));
          }),
      "}");
}
}  // namespace

HloBatchNormInstruction::HloBatchNormInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* operand,
    HloInstruction* scale, float epsilon, int64_t feature_index)
    : HloInstruction(opcode, shape),
      epsilon_(epsilon),
      feature_index_(feature_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInstruction::HloBatchNormInstruction");

  AppendOperand(operand);
  AppendOperand(scale);
}

bool HloBatchNormInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_3(mht_3_v, 275, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloBatchNormInstruction&>(other);
  return feature_index() == casted_other.feature_index() &&
         epsilon() == casted_other.epsilon();
}

HloInstructionProto HloBatchNormInstruction::ToProto() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_4(mht_4_v, 284, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_epsilon(epsilon_);
  proto.set_feature_index(feature_index_);
  return proto;
}

std::vector<std::string> HloBatchNormInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_5(mht_5_v, 295, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInstruction::ExtraAttributesToStringImpl");

  return {StrCat("epsilon=", epsilon()),
          StrCat("feature_index=", feature_index())};
}

HloBatchNormTrainingInstruction::HloBatchNormTrainingInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormTraining, shape, operand,
                              scale, epsilon, feature_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_6(mht_6_v, 307, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormTrainingInstruction::HloBatchNormTrainingInstruction");

  AppendOperand(offset);
}

std::unique_ptr<HloInstruction>
HloBatchNormTrainingInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_7(mht_7_v, 317, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormTrainingInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloBatchNormTrainingInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], epsilon(),
      feature_index());
}

HloBatchNormInferenceInstruction::HloBatchNormInferenceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormInference, shape, operand,
                              scale, epsilon, feature_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_8(mht_8_v, 332, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInferenceInstruction::HloBatchNormInferenceInstruction");

  AppendOperand(offset);
  AppendOperand(mean);
  AppendOperand(variance);
}

std::unique_ptr<HloInstruction>
HloBatchNormInferenceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_9(mht_9_v, 344, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormInferenceInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 5);
  return absl::make_unique<HloBatchNormInferenceInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloBatchNormGradInstruction::HloBatchNormGradInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* mean, HloInstruction* variance, HloInstruction* grad_output,
    float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormGrad, shape, operand, scale,
                              epsilon, feature_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_10(mht_10_v, 359, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormGradInstruction::HloBatchNormGradInstruction");

  AppendOperand(mean);
  AppendOperand(variance);
  AppendOperand(grad_output);
}

std::unique_ptr<HloInstruction>
HloBatchNormGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_11(mht_11_v, 371, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBatchNormGradInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 5);
  return absl::make_unique<HloBatchNormGradInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloFftInstruction::HloFftInstruction(const Shape& shape,
                                     HloInstruction* operand, FftType fft_type,
                                     absl::Span<const int64_t> fft_length)
    : HloInstruction(HloOpcode::kFft, shape), fft_type_(fft_type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_12(mht_12_v, 384, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFftInstruction::HloFftInstruction");

  fft_length_.assign(fft_length.begin(), fft_length.end());
  AppendOperand(operand);
}

HloInstructionProto HloFftInstruction::ToProto() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_13(mht_13_v, 392, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFftInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fft_type(fft_type_);
  for (int64_t fft_len : fft_length_) {
    proto.add_fft_length(fft_len);
  }
  return proto;
}

std::vector<std::string> HloFftInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_14(mht_14_v, 405, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFftInstruction::ExtraAttributesToStringImpl");

  return {StrCat("fft_type=", FftType_Name(fft_type())),
          StrCat("fft_length={", StrJoin(fft_length(), ","), "}")};
}

bool HloFftInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_15(mht_15_v, 416, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFftInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloFftInstruction&>(other);
  return fft_type() == casted_other.fft_type() &&
         fft_length() == casted_other.fft_length();
}

std::unique_ptr<HloInstruction> HloFftInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_16(mht_16_v, 427, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFftInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloFftInstruction>(shape, new_operands[0], fft_type_,
                                              fft_length_);
}

HloAsyncInstruction::HloAsyncInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* async_computation)
    : HloInstruction(opcode, shape) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_17(mht_17_v, 440, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::HloAsyncInstruction");

  CHECK(opcode == HloOpcode::kAsyncStart || operands.size() == 1);
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(async_computation);
  CHECK(!async_computation->IsCustomCallComputation());
  CHECK(!async_computation->IsFusionComputation());
  async_computation->SetAsyncInstruction(this);
}

HloAsyncInstruction::HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                                         HloInstruction* operand,
                                         HloComputation* async_computation)
    : HloInstruction(opcode, shape) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_18(mht_18_v, 457, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::HloAsyncInstruction");

  AppendOperand(operand);
  AppendComputation(async_computation);
  CHECK(!async_computation->IsCustomCallComputation());
  CHECK(!async_computation->IsFusionComputation());
  async_computation->SetAsyncInstruction(this);
}

HloInstruction* HloAsyncInstruction::async_wrapped_instruction() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_19(mht_19_v, 468, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::async_wrapped_instruction");

  CHECK(!called_computations().empty());
  return called_computations()[0]->root_instruction();
}

HloOpcode HloAsyncInstruction::async_wrapped_opcode() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_20(mht_20_v, 476, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::async_wrapped_opcode");

  return async_wrapped_instruction()->opcode();
}

std::vector<std::string> HloAsyncInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_21(mht_21_v, 484, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::ExtraAttributesToStringImpl");

  if (options.syntax_sugar_async_ops()) {
    return async_wrapped_instruction()->ExtraAttributesToString(options);
  }
  return {};
}

bool HloAsyncInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_22(mht_22_v, 497, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::IdenticalSlowPath");

  return opcode() == other.opcode() &&
         eq_computations(async_wrapped_computation(),
                         other.async_wrapped_computation());
}

std::unique_ptr<HloInstruction> HloAsyncInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_23(mht_23_v, 508, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAsyncInstruction::CloneWithNewOperandsImpl");

  HloModule* module = context != nullptr ? context->module() : GetModule();
  HloComputation* new_wrapped_computation = nullptr;
  if (context != nullptr) {
    new_wrapped_computation =
        context->FindComputation(async_wrapped_computation());
  }
  if (new_wrapped_computation == nullptr) {
    new_wrapped_computation = module->AddEmbeddedComputation(
        async_wrapped_computation()->Clone("clone", context));
  }
  return absl::make_unique<HloAsyncInstruction>(opcode(), shape, new_operands,
                                                new_wrapped_computation);
}

HloCopyStartInstruction::HloCopyStartInstruction(const Shape& shape,
                                                 HloInstruction* operand,
                                                 bool is_cross_program_prefetch)
    : HloInstruction(HloOpcode::kCopyStart, shape),
      is_cross_program_prefetch_(is_cross_program_prefetch) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_24(mht_24_v, 530, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCopyStartInstruction::HloCopyStartInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloCopyStartInstruction::ToProto() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_25(mht_25_v, 537, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCopyStartInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_is_cross_program_prefetch(is_cross_program_prefetch_);
  return proto;
}

std::vector<std::string> HloCopyStartInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_26(mht_26_v, 547, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCopyStartInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result;
  if (is_cross_program_prefetch()) {
    result.push_back("is_cross_program_prefetch=true");
  }
  return result;
}

bool HloCopyStartInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_27(mht_27_v, 561, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCopyStartInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloCopyStartInstruction&>(other);
  return is_cross_program_prefetch() ==
         casted_other.is_cross_program_prefetch();
}

std::unique_ptr<HloInstruction>
HloCopyStartInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_28(mht_28_v, 573, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCopyStartInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloCopyStartInstruction>(
      shape, new_operands[0], is_cross_program_prefetch());
}

HloCompareInstruction::HloCompareInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    ComparisonDirection direction, absl::optional<Comparison::Type> type)
    : HloInstruction(HloOpcode::kCompare, shape),
      compare_(direction, type ? (*type)
                               : Comparison::DefaultComparisonType(
                                     lhs->shape().element_type())) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_29(mht_29_v, 588, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCompareInstruction::HloCompareInstruction");

  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloCompareInstruction::ToProto() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_30(mht_30_v, 596, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCompareInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_comparison_direction(
      ComparisonDirectionToString(compare_.GetDirection()));
  proto.set_comparison_type(ComparisonTypeToString(compare_.GetType()));
  return proto;
}

std::vector<std::string> HloCompareInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_31(mht_31_v, 608, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCompareInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result;
  result.push_back(
      StrCat("direction=", ComparisonDirectionToString(direction())));
  if (compare_.GetType() !=
      Comparison::DefaultComparisonType(operand(0)->shape().element_type())) {
    result.push_back(
        StrCat("type=", ComparisonTypeToString(compare_.GetType())));
  }
  return result;
}

bool HloCompareInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_32(mht_32_v, 626, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCompareInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloCompareInstruction&>(other);
  return direction() == casted_other.direction();
}

std::unique_ptr<HloInstruction> HloCompareInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_33(mht_33_v, 636, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCompareInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloCompareInstruction>(
      shape, new_operands[0], new_operands[1], direction(), type());
}

namespace {

// Converts a protocol buffer message (e.g., TriangularSolveOptions) to a vector
// of "key=value" attribute strings generically, using protocol buffer
// reflection.
//
// Currently implements a small subset of cases; feel free to add more as
// needed.
std::vector<std::string> AttributeProtoToStringVector(
    const tensorflow::protobuf::Message& message) {
  const tensorflow::protobuf::Reflection* reflection = message.GetReflection();
  std::vector<const tensorflow::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);

  std::vector<std::string> output;
  for (const tensorflow::protobuf::FieldDescriptor* field : fields) {
    std::string s = absl::StrCat(field->name(), "=");
    CHECK(!field->is_repeated()) << "Repeated fields aren't implemented";
    switch (field->type()) {
      case tensorflow::protobuf::FieldDescriptor::TYPE_BOOL: {
        bool val = reflection->GetBool(message, field);
        absl::StrAppend(&s, val ? "true" : "false");
        break;
      }
      case tensorflow::protobuf::FieldDescriptor::TYPE_ENUM: {
        const tensorflow::protobuf::EnumValueDescriptor* evd =
            reflection->GetEnum(message, field);
        absl::StrAppend(&s, evd->name());
        break;
      }
      default:
        LOG(FATAL) << "Unimplemented field type: " << field->DebugString();
    }
    output.push_back(std::move(s));
  }
  return output;
}

}  // namespace

HloTriangularSolveInstruction::HloTriangularSolveInstruction(
    const Shape& shape, HloInstruction* a, HloInstruction* b,
    const TriangularSolveOptions& options)
    : HloInstruction(HloOpcode::kTriangularSolve, shape),
      triangular_solve_options_(options) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_34(mht_34_v, 689, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTriangularSolveInstruction::HloTriangularSolveInstruction");

  AppendOperand(a);
  AppendOperand(b);
}

HloInstructionProto HloTriangularSolveInstruction::ToProto() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_35(mht_35_v, 697, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTriangularSolveInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_triangular_solve_options() = triangular_solve_options_;
  return proto;
}

std::vector<std::string>
HloTriangularSolveInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_36(mht_36_v, 708, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTriangularSolveInstruction::ExtraAttributesToStringImpl");

  return AttributeProtoToStringVector(triangular_solve_options_);
}

bool HloTriangularSolveInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_37(mht_37_v, 718, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTriangularSolveInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloTriangularSolveInstruction&>(other);
  const auto& options = triangular_solve_options();
  const auto& other_options = casted_other.triangular_solve_options();

  return options.left_side() == other_options.left_side() &&
         options.lower() == other_options.lower() &&
         options.unit_diagonal() == other_options.unit_diagonal() &&
         options.transpose_a() == other_options.transpose_a();
}

std::unique_ptr<HloInstruction>
HloTriangularSolveInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_38(mht_38_v, 736, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTriangularSolveInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloTriangularSolveInstruction>(
      shape, new_operands[0], new_operands[1], triangular_solve_options());
}

HloCholeskyInstruction::HloCholeskyInstruction(const Shape& shape,
                                               HloInstruction* a,
                                               const CholeskyOptions& options)
    : HloInstruction(HloOpcode::kCholesky, shape), cholesky_options_(options) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_39(mht_39_v, 748, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCholeskyInstruction::HloCholeskyInstruction");

  AppendOperand(a);
}

HloInstructionProto HloCholeskyInstruction::ToProto() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_40(mht_40_v, 755, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCholeskyInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_cholesky_options() = cholesky_options_;
  return proto;
}

std::vector<std::string> HloCholeskyInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_41(mht_41_v, 765, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCholeskyInstruction::ExtraAttributesToStringImpl");

  return AttributeProtoToStringVector(cholesky_options_);
}

bool HloCholeskyInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_42(mht_42_v, 775, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCholeskyInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloCholeskyInstruction&>(other);
  const auto& options = cholesky_options();
  const auto& other_options = casted_other.cholesky_options();

  return options.lower() == other_options.lower();
}

std::unique_ptr<HloInstruction>
HloCholeskyInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_43(mht_43_v, 789, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCholeskyInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloCholeskyInstruction>(shape, new_operands[0],
                                                   cholesky_options());
}

HloChannelInstruction::HloChannelInstruction(
    HloOpcode opcode, const Shape& shape,
    const absl::optional<int64_t>& channel_id)
    : HloInstruction(opcode, shape), channel_id_(channel_id) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_44(mht_44_v, 801, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloChannelInstruction::HloChannelInstruction");
}

void HloChannelInstruction::set_channel_id(
    const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_45(mht_45_v, 807, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloChannelInstruction::set_channel_id");

  channel_id_ = channel_id;
}

HloInstructionProto HloChannelInstruction::ToProto() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_46(mht_46_v, 814, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloChannelInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  if (channel_id_) {
    CHECK_GT(channel_id_.value(), 0)
        << "Non-positive channel id is equivalent to no channel id";
    proto.set_channel_id(*channel_id_);
  }
  return proto;
}

std::vector<std::string> HloChannelInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_47(mht_47_v, 828, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloChannelInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result;
  if (channel_id_) {
    result.push_back(StrCat("channel_id=", *channel_id_));
  }
  return result;
}

bool HloChannelInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_48(mht_48_v, 842, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloChannelInstruction::IdenticalSlowPath");

  if (!IdenticalSlowPathIgnoringChannelIdValues(other, eq_computations)) {
    return false;
  }
  const auto& casted_other = static_cast<const HloChannelInstruction&>(other);
  return channel_id() == casted_other.channel_id();
}

HloSendRecvInstruction::HloSendRecvInstruction(HloOpcode opcode,
                                               const Shape& shape,
                                               int64_t channel_id,
                                               bool is_host_transfer)
    : HloChannelInstruction(opcode, shape, channel_id),
      is_host_transfer_(is_host_transfer) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_49(mht_49_v, 858, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendRecvInstruction::HloSendRecvInstruction");
}

HloInstructionProto HloSendRecvInstruction::ToProto() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_50(mht_50_v, 863, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendRecvInstruction::ToProto");

  HloInstructionProto proto = HloChannelInstruction::ToProto();
  proto.set_is_host_transfer(is_host_transfer_);
  return proto;
}

std::vector<std::string> HloSendRecvInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_51(mht_51_v, 873, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendRecvInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> attrs =
      HloChannelInstruction::ExtraAttributesToStringImpl(options);
  if (is_host_transfer()) {
    attrs.push_back("is_host_transfer=true");
  }
  return attrs;
}

bool HloSendRecvInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_52(mht_52_v, 888, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendRecvInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  // Not yet supported.
  return false;
}

// Send instruction produces a tuple of {aliased operand, U32 context}.
HloSendInstruction::HloSendInstruction(HloInstruction* operand,
                                       HloInstruction* token,
                                       int64_t channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kSend,
          ShapeUtil::MakeTupleShape({CHECK_NOTNULL(operand)->shape(),
                                     ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_53(mht_53_v, 906, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendInstruction::HloSendInstruction");

  AppendOperand(operand);
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloSendInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_54(mht_54_v, 916, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloSendInstruction>(
      new_operands[0], new_operands[1], *channel_id(), is_host_transfer());
}

HloSendDoneInstruction::HloSendDoneInstruction(HloSendInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             CHECK_NOTNULL(operand)->channel_id().value(),
                             is_host_transfer) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_55(mht_55_v, 929, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendDoneInstruction::HloSendDoneInstruction");

  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloSendDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_56(mht_56_v, 939, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSendDoneInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloSendDoneInstruction>(
      Cast<HloSendInstruction>(new_operands[0]), is_host_transfer());
}

// Recv instruction produces a tuple of {receive buffer, U32 context}.
HloRecvInstruction::HloRecvInstruction(const Shape& shape,
                                       HloInstruction* token,
                                       int64_t channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecv,
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_57(mht_57_v, 957, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRecvInstruction::HloRecvInstruction");

  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloRecvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_58(mht_58_v, 966, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRecvInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRecvInstruction>(
      ShapeUtil::GetTupleElementShape(shape, 0), new_operands[0], *channel_id(),
      is_host_transfer());
}

HloRecvDoneInstruction::HloRecvDoneInstruction(HloRecvInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          CHECK_NOTNULL(operand)->channel_id().value(), is_host_transfer) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_59(mht_59_v, 983, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRecvDoneInstruction::HloRecvDoneInstruction");

  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloRecvDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_60(mht_60_v, 993, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRecvDoneInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRecvDoneInstruction>(
      Cast<HloRecvInstruction>(new_operands[0]), is_host_transfer());
}

HloCollectiveInstruction::HloCollectiveInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      replica_groups_(SpanToVector(replica_groups)),
      constrain_layout_(constrain_layout) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_61(mht_61_v, 1009, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectiveInstruction::HloCollectiveInstruction");

  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCollectiveInstruction::ToProto() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_62(mht_62_v, 1018, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectiveInstruction::ToProto");

  HloInstructionProto proto = HloChannelInstruction::ToProto();
  *proto.mutable_replica_groups() = {replica_groups_.begin(),
                                     replica_groups_.end()};
  proto.set_constrain_layout(constrain_layout_);
  return proto;
}

std::vector<std::string> HloCollectiveInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_63(mht_63_v, 1030, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectiveInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloChannelInstruction::ExtraAttributesToStringImpl(options);
  result.push_back(
      StrCat("replica_groups=", ReplicaGroupsToString(replica_groups())));
  if (constrain_layout_) {
    result.push_back("constrain_layout=true");
  }
  return result;
}

bool HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_64(mht_64_v, 1047, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  const auto& casted_other =
      static_cast<const HloCollectiveInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         absl::c_equal(replica_groups(), casted_other.replica_groups(),
                       [](const ReplicaGroup& a, const ReplicaGroup& b) {
                         return absl::c_equal(a.replica_ids(), b.replica_ids());
                       });
}

HloAllGatherInstruction::HloAllGatherInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, replica_groups,
                               constrain_layout, channel_id),
      all_gather_dimension_(all_gather_dimension),
      use_global_device_ids_(use_global_device_ids) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_65(mht_65_v, 1070, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllGatherInstruction::HloAllGatherInstruction");
}

std::vector<std::string> HloAllGatherInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_66(mht_66_v, 1076, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllGatherInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloCollectiveInstruction::ExtraAttributesToStringImpl(options);
  result.push_back(StrCat("dimensions={", all_gather_dimension_, "}"));
  if (use_global_device_ids_) {
    result.push_back("use_global_device_ids=true");
  }
  return result;
}

std::unique_ptr<HloInstruction>
HloAllGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_67(mht_67_v, 1092, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllGatherInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloAllGatherInstruction>(
      opcode(), shape, new_operands, all_gather_dimension(), replica_groups(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloInstructionProto HloAllGatherInstruction::ToProto() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_68(mht_68_v, 1101, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllGatherInstruction::ToProto");

  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.add_dimensions(all_gather_dimension_);
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

bool HloAllGatherInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_69(mht_69_v, 1114, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllGatherInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  const auto& casted_other = static_cast<const HloAllGatherInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         all_gather_dimension_ == casted_other.all_gather_dimension() &&
         use_global_device_ids() == casted_other.use_global_device_ids();
}

HloAllReduceInstructionBase::HloAllReduceInstructionBase(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, replica_groups,
                               constrain_layout, channel_id),
      use_global_device_ids_(use_global_device_ids) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_70(mht_70_v, 1133, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstructionBase::HloAllReduceInstructionBase");

  AppendComputation(reduce_computation);
}

HloInstructionProto HloAllReduceInstructionBase::ToProto() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_71(mht_71_v, 1140, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstructionBase::ToProto");

  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

std::vector<std::string>
HloAllReduceInstructionBase::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_72(mht_72_v, 1151, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstructionBase::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloCollectiveInstruction::ExtraAttributesToStringImpl(options);
  if (use_global_device_ids_) {
    result.push_back("use_global_device_ids=true");
  }
  return result;
}

bool HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_73(mht_73_v, 1166, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues");

  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloAllReduceInstructionBase&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         use_global_device_ids() == casted_other.use_global_device_ids() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

bool HloAllReduceInstruction::IsNoop() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_74(mht_74_v, 1182, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstruction::IsNoop");

  for (const auto& replica_group : replica_groups()) {
    if (replica_group.replica_ids().size() != 1) {
      return false;
    }
  }
  return !channel_id();
}

std::unique_ptr<HloInstruction>
HloAllReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_75(mht_75_v, 1197, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllReduceInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloAllReduceInstruction>(
      opcode(), shape, new_operands, to_apply(), replica_groups(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids,
    int64_t scatter_dimension)
    : HloAllReduceInstructionBase(
          HloOpcode::kReduceScatter, shape, operands, reduce_computation,
          replica_groups, constrain_layout, channel_id, use_global_device_ids),
      scatter_dimension_(scatter_dimension) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_76(mht_76_v, 1215, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceScatterInstruction::HloReduceScatterInstruction");
}

std::vector<std::string>
HloReduceScatterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_77(mht_77_v, 1222, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceScatterInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloAllReduceInstructionBase::ExtraAttributesToStringImpl(options);
  result.push_back(StrCat("dimensions={", scatter_dimension_, "}"));
  return result;
}

HloInstructionProto HloReduceScatterInstruction::ToProto() const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_78(mht_78_v, 1232, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceScatterInstruction::ToProto");

  HloInstructionProto proto = HloAllReduceInstructionBase::ToProto();
  proto.add_dimensions(scatter_dimension_);
  return proto;
}

bool HloReduceScatterInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_79(mht_79_v, 1244, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceScatterInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  const auto& casted_other =
      static_cast<const HloReduceScatterInstruction&>(other);
  return HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         scatter_dimension_ == casted_other.scatter_dimension();
}

std::unique_ptr<HloInstruction>
HloReduceScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_80(mht_80_v, 1258, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceScatterInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloReduceScatterInstruction>(
      shape, new_operands, to_apply(), replica_groups(), constrain_layout(),
      channel_id(), use_global_device_ids(), scatter_dimension());
}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id,
    const absl::optional<int64_t>& split_dimension)
    : HloCollectiveInstruction(HloOpcode::kAllToAll, shape, operands,
                               replica_groups, constrain_layout, channel_id),
      split_dimension_(split_dimension) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_81(mht_81_v, 1274, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllToAllInstruction::HloAllToAllInstruction");
}

std::unique_ptr<HloInstruction>
HloAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_82(mht_82_v, 1282, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllToAllInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloAllToAllInstruction>(
      shape, new_operands, replica_groups(), constrain_layout(), channel_id(),
      split_dimension());
}

HloInstructionProto HloAllToAllInstruction::ToProto() const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_83(mht_83_v, 1291, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllToAllInstruction::ToProto");

  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  if (split_dimension_) {
    proto.add_dimensions(*split_dimension_);
  }
  return proto;
}

std::vector<std::string> HloAllToAllInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_84(mht_84_v, 1303, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllToAllInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloCollectiveInstruction::ExtraAttributesToStringImpl(options);
  if (split_dimension_) {
    result.push_back(StrCat("dimensions={", *split_dimension_, "}"));
  }
  return result;
}

bool HloAllToAllInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_85(mht_85_v, 1318, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloAllToAllInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  const auto& casted_other = static_cast<const HloAllToAllInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         split_dimension_ == casted_other.split_dimension();
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const absl::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_86(mht_86_v, 1333, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::HloCollectivePermuteInstruction");

  AppendOperand(operand);
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* input,
    HloInstruction* output, HloInstruction* input_start_indices,
    HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const absl::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs.begin(),
                           source_target_pairs.end()),
      slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_87(mht_87_v, 1350, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::HloCollectivePermuteInstruction");

  AppendOperand(input);
  AppendOperand(output);
  AppendOperand(input_start_indices);
  AppendOperand(output_start_indices);
}

HloInstructionProto HloCollectivePermuteInstruction::ToProto() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_88(mht_88_v, 1360, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::ToProto");

  HloInstructionProto proto = HloChannelInstruction::ToProto();
  for (const auto& pair : source_target_pairs()) {
    auto* proto_pair = proto.add_source_target_pairs();
    proto_pair->set_source(pair.first);
    proto_pair->set_target(pair.second);
  }
  for (const auto& slice_size : dynamic_slice_sizes_list()) {
    for (const auto& dimension_slice_size : slice_size) {
      proto.add_dynamic_slice_sizes(dimension_slice_size);
    }
  }
  return proto;
}

std::vector<std::string>
HloCollectivePermuteInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_89(mht_89_v, 1380, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result =
      HloChannelInstruction::ExtraAttributesToStringImpl(options);
  {
    std::vector<std::string> strs;
    const auto& pairs = source_target_pairs();
    strs.reserve(pairs.size());
    for (const auto& pair : pairs) {
      strs.push_back(StrCat("{", pair.first, ",", pair.second, "}"));
    }
    result.push_back(StrCat("source_target_pairs={", StrJoin(strs, ","), "}"));
  }
  if (!dynamic_slice_sizes_list().empty()) {
    std::vector<std::string> strs;
    const auto& sizes_list = dynamic_slice_sizes_list();
    strs.reserve(sizes_list.size());
    for (const auto& slice_sizes : dynamic_slice_sizes_list()) {
      strs.push_back(StrCat("{", StrJoin(slice_sizes, ","), "}"));
    }
    result.push_back(StrCat("slice_sizes={", StrJoin(strs, ","), "}"));
  }
  return result;
}

bool HloCollectivePermuteInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_90(mht_90_v, 1410, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::IdenticalSlowPathIgnoringChannelIdValues");

  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloCollectivePermuteInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         absl::c_equal(
             source_target_pairs(), casted_other.source_target_pairs(),
             [](const std::pair<int64_t, int64_t>& a,
                const std::pair<int64_t, int64_t>& b) { return a == b; }) &&
         absl::c_equal(
             dynamic_slice_sizes_list(),
             casted_other.dynamic_slice_sizes_list(),
             [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
               return absl::c_equal(a, b);
             });
}

std::unique_ptr<HloInstruction>
HloCollectivePermuteInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_91(mht_91_v, 1436, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCollectivePermuteInstruction::CloneWithNewOperandsImpl");

  if (dynamic_slice_sizes_list().empty()) {
    return absl::make_unique<HloCollectivePermuteInstruction>(
        opcode(), shape, new_operands[0], source_target_pairs(), channel_id());
  } else {
    return absl::make_unique<HloCollectivePermuteInstruction>(
        opcode(), shape, new_operands[0], new_operands[1], new_operands[2],
        new_operands[3], source_target_pairs(), dynamic_slice_sizes_list(),
        channel_id());
  }
}

HloReverseInstruction::HloReverseInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kReverse, shape, dimensions) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_92(mht_92_v, 1454, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReverseInstruction::HloReverseInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloDimensionsInstruction::ToProto() const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_93(mht_93_v, 1461, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDimensionsInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<std::string> HloDimensionsInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_94(mht_94_v, 1473, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDimensionsInstruction::ExtraAttributesToStringImpl");

  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
}

bool HloDimensionsInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_95(mht_95_v, 1483, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDimensionsInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloDimensionsInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_96(mht_96_v, 1494, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReverseInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloReverseInstruction>(shape, new_operands[0],
                                                  dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension)
    : HloDimensionsInstruction(HloOpcode::kConcatenate, shape, {dimension}) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_97(mht_97_v, 1506, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConcatenateInstruction::HloConcatenateInstruction");

  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloConcatenateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_98(mht_98_v, 1518, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConcatenateInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloConcatenateInstruction>(shape, new_operands,
                                                      concatenate_dimension());
}

HloReduceInstruction::HloReduceInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> args,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation)
    : HloDimensionsInstruction(HloOpcode::kReduce, shape,
                               dimensions_to_reduce) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_99(mht_99_v, 1531, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceInstruction::HloReduceInstruction");

  for (HloInstruction* arg : args) {
    AppendOperand(arg);
  }
  AppendComputation(reduce_computation);
}

bool HloReduceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_100(mht_100_v, 1544, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloReduceInstruction&>(other);
  // Reduction results are determined by the reduction dimension and the
  // reduction computation.
  return dimensions() == casted_other.dimensions() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

std::unique_ptr<HloInstruction> HloReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_101(mht_101_v, 1557, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size() % 2, 0);
  return absl::make_unique<HloReduceInstruction>(shape, new_operands,
                                                 dimensions(), to_apply());
}

HloSortInstruction::HloSortInstruction(
    const Shape& shape, int64_t dimension,
    absl::Span<HloInstruction* const> operands, HloComputation* compare,
    bool is_stable)
    : HloDimensionsInstruction(HloOpcode::kSort, shape, {dimension}),
      is_stable_(is_stable) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_102(mht_102_v, 1571, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSortInstruction::HloSortInstruction");

  for (auto* value : operands) {
    AppendOperand(value);
  }
  AppendComputation(compare);
}

HloInstructionProto HloSortInstruction::ToProto() const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_103(mht_103_v, 1581, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSortInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  proto.set_is_stable(is_stable());
  return proto;
}

std::vector<std::string> HloSortInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_104(mht_104_v, 1594, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSortInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> attrs;
  attrs.push_back(StrCat("dimensions={", StrJoin(dimensions(), ","), "}"));
  if (is_stable()) {
    attrs.push_back("is_stable=true");
  }
  return attrs;
}

bool HloSortInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_105(mht_105_v, 1609, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSortInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloSortInstruction&>(other);
  if (dimensions() != casted_other.dimensions()) {
    return false;
  }
  if (is_stable() != casted_other.is_stable()) {
    return false;
  }
  return eq_computations(to_apply(), other.to_apply());
}

std::unique_ptr<HloInstruction> HloSortInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_106(mht_106_v, 1625, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSortInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloSortInstruction>(
      shape, dimensions_[0], new_operands, to_apply(), is_stable());
}

HloTransposeInstruction::HloTransposeInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kTranspose, shape, dimensions) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_107(mht_107_v, 1636, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTransposeInstruction::HloTransposeInstruction");

  AppendOperand(operand);
}

bool HloTransposeInstruction::IsRank2Transpose() const {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_108(mht_108_v, 1643, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTransposeInstruction::IsRank2Transpose");

  return dimensions() == std::vector<int64_t>({1, 0}) &&
         shape().dimensions_size() == 2 &&
         std::equal(shape().dimensions().begin(), shape().dimensions().end(),
                    operand(0)->shape().dimensions().rbegin());
}

std::unique_ptr<HloInstruction>
HloTransposeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_109(mht_109_v, 1656, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTransposeInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloTransposeInstruction>(shape, new_operands[0],
                                                    dimensions());
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> broadcast_dimension)
    : HloDimensionsInstruction(HloOpcode::kBroadcast, shape,
                               broadcast_dimension) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_110(mht_110_v, 1669, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBroadcastInstruction::HloBroadcastInstruction");

  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_111(mht_111_v, 1679, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloBroadcastInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloBroadcastInstruction>(shape, new_operands[0],
                                                    dimensions());
}

HloDynamicReshapeInstruction::HloDynamicReshapeInstruction(
    const Shape& shape, HloInstruction* data_operand,
    absl::Span<HloInstruction* const> dim_sizes)
    : HloInstruction(HloOpcode::kDynamicReshape, shape) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_112(mht_112_v, 1691, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicReshapeInstruction::HloDynamicReshapeInstruction");

  AppendOperand(data_operand);
  for (auto operand : dim_sizes) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloDynamicReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_113(mht_113_v, 1704, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicReshapeInstruction::CloneWithNewOperandsImpl");

  CHECK_GE(new_operands.size(), 1);
  return absl::make_unique<HloDynamicReshapeInstruction>(
      shape, new_operands[0], new_operands.subspan(1));
}

HloReshapeInstruction::HloReshapeInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             int64_t inferred_dimension)
    : HloInstruction(HloOpcode::kReshape, shape),
      inferred_dimension_(inferred_dimension) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_114(mht_114_v, 1717, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReshapeInstruction::HloReshapeInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloReshapeInstruction::ToProto() const {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_115(mht_115_v, 1724, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReshapeInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  if (inferred_dimension_ != -1) {
    proto.add_dimensions(inferred_dimension_);
  }
  return proto;
}

std::vector<std::string> HloReshapeInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_116(mht_116_v, 1736, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReshapeInstruction::ExtraAttributesToStringImpl");

  if (inferred_dimension() == -1) {
    return {};
  }
  return {StrCat("inferred_dimension=", inferred_dimension())};
}

bool HloReshapeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_117(mht_117_v, 1749, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReshapeInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloReshapeInstruction&>(other);
  return inferred_dimension() == casted_other.inferred_dimension();
}

std::unique_ptr<HloInstruction> HloReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_118(mht_118_v, 1759, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReshapeInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloReshapeInstruction>(shape, new_operands[0],
                                                  inferred_dimension());
}

HloMapInstruction::HloMapInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     HloComputation* map_computation)
    : HloInstruction(HloOpcode::kMap, shape) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_119(mht_119_v, 1771, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::HloMapInstruction");

  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(map_computation);
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  dimensions_.resize(shape.rank());
  std::iota(dimensions_.begin(), dimensions_.end(), 0);
}

HloInstructionProto HloMapInstruction::ToProto() const {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_120(mht_120_v, 1785, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

bool HloMapInstruction::IsElementwiseImpl(
    const absl::optional<int64_t>& operand_idx) const {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_121(mht_121_v, 1797, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::IsElementwiseImpl");

  if (!dimensions().empty()) {
    // Check that the map is executed in elementwise compatible dimensions.
    if (dimensions().size() != shape().dimensions_size()) {
      return false;
    }
    for (int i = 0; i < dimensions().size(); ++i) {
      if (dimensions()[i] != i) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::string> HloMapInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_122(mht_122_v, 1816, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::ExtraAttributesToStringImpl");

  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
}

bool HloMapInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_123(mht_123_v, 1826, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloMapInstruction&>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloMapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_124(mht_124_v, 1837, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloMapInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloMapInstruction>(shape, new_operands, to_apply());
}

HloSliceInstruction::HloSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides)
    : HloInstruction(HloOpcode::kSlice, shape),
      slice_starts_(start_indices.begin(), start_indices.end()),
      slice_limits_(limit_indices.begin(), limit_indices.end()),
      slice_strides_(strides.begin(), strides.end()) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_125(mht_125_v, 1851, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSliceInstruction::HloSliceInstruction");

  AppendOperand(operand);
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (slice_strides_.empty()) {
    slice_strides_ = std::vector<int64_t>(start_indices.size(), 1LL);
  }
}

HloInstructionProto HloSliceInstruction::ToProto() const {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_126(mht_126_v, 1864, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSliceInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto* slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  return proto;
}

std::vector<std::string> HloSliceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_127(mht_127_v, 1879, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSliceInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> bounds;
  bounds.reserve(slice_starts_.size());
  const bool omit_stride = absl::c_all_of(
      slice_strides_, [](int64_t stride) { return stride == 1; });
  for (int i = 0; i < slice_starts_.size(); ++i) {
    std::string stride_str = omit_stride ? "" : StrCat(":", slice_strides_[i]);
    bounds.push_back(
        StrCat("[", slice_starts_[i], ":", slice_limits_[i], stride_str, "]"));
  }
  return {StrCat("slice={", StrJoin(bounds, ", "), "}")};
}

bool HloSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_128(mht_128_v, 1898, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSliceInstruction::IdenticalSlowPath");

  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return slice_starts_ == other_slice.slice_starts_ &&
         slice_limits_ == other_slice.slice_limits_ &&
         slice_strides_ == other_slice.slice_strides_;
}

std::unique_ptr<HloInstruction> HloSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_129(mht_129_v, 1910, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSliceInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloSliceInstruction>(
      shape, new_operands[0], slice_starts_, slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(Literal literal)
    : HloInstruction(HloOpcode::kConstant, literal.shape()),
      literal_(std::move(literal)) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_130(mht_130_v, 1921, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::HloConstantInstruction");
}

HloConstantInstruction::HloConstantInstruction(Literal literal,
                                               const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape),
      literal_(std::move(literal)) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_131(mht_131_v, 1929, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::HloConstantInstruction");
}

HloConstantInstruction::HloConstantInstruction(const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_132(mht_132_v, 1935, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::HloConstantInstruction");
}

HloInstructionProto HloConstantInstruction::ToProto() const {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_133(mht_133_v, 1940, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_.has_value()) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const absl::optional<int64_t>& operand_idx) const {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_134(mht_134_v, 1952, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::IsElementwiseImpl");

  return true;
}

void HloConstantInstruction::RelayoutConstant(const Layout& new_layout,
                                              const ShapeIndex& shape_index) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_135(mht_135_v, 1960, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::RelayoutConstant");

  Shape* mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(mutable_array_subshape->IsArray());

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    *literal_ = literal_->Relayout(new_layout, shape_index);
    *mutable_array_subshape->mutable_layout() = new_layout;
  }
}

bool HloConstantInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_136(mht_136_v, 1981, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::IdenticalSlowPath");

  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return literal() == other_slice.literal();
}

std::unique_ptr<HloInstruction>
HloConstantInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_137(mht_137_v, 1992, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::CloneWithNewOperandsImpl");

  if (!literal_.has_value()) {
    return absl::make_unique<HloConstantInstruction>(this->shape());
  }
  CHECK(literal_.has_value());
  // Literal's shape may have no/different tiling info. Use this instruction's
  // shape instead.
  CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(literal_->shape(),
                                                  this->shape()));
  return absl::make_unique<HloConstantInstruction>(literal_->Clone(),
                                                   this->shape());
}

std::string HloConstantInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_138(mht_138_v, 2010, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConstantInstruction::OperandsToStringWithCanonicalNameMap");

  if (options.print_only_essential_constants()) {
    if (!literal_.has_value()) {
      return "{...}";
    }
    if (literal().IsAll(0)) {
      return "0";
    }
    if (literal().IsAll(1)) {
      return "1";
    }
    if (shape().IsInteger()) {
      return literal_->ToStringWithoutShapeOneline();
    }
    return "{...}";
  }

  // For constants, show the actual value in place of an empty operand list.
  if (literal_.has_value() &&
      ((shape().IsArray() && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    return literal_->ToStringWithoutShapeOneline();
  } else {
    // Do not show large constants or tuples.
    return "{...}";
  }
}

HloTraceInstruction::HloTraceInstruction(const std::string& tag,
                                         HloInstruction* operand)
    : HloInstruction(HloOpcode::kTrace, ShapeUtil::MakeNil()),
      literal_(LiteralUtil::CreateR1U8(tag)) {
   std::vector<std::string> mht_139_v;
   mht_139_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_139(mht_139_v, 2047, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTraceInstruction::HloTraceInstruction");

  AppendOperand(operand);
  operand->set_tracing(this);
}

HloInstructionProto HloTraceInstruction::ToProto() const {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_140(mht_140_v, 2055, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTraceInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_literal() = literal_.ToProto();
  return proto;
}

bool HloTraceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_141(mht_141_v, 2067, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTraceInstruction::IdenticalSlowPath");

  return false;
}

std::unique_ptr<HloInstruction> HloTraceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_142(mht_142_v, 2076, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloTraceInstruction::CloneWithNewOperandsImpl");

  LOG(FATAL) << "Not yet implemented, clone: " << ToString();
}

HloFusionInstruction::HloFusionInstruction(const Shape& shape,
                                           FusionKind fusion_kind,
                                           HloInstruction* fused_root)
    : HloInstruction(HloOpcode::kFusion, shape), fusion_kind_(fusion_kind) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_143(mht_143_v, 2086, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::HloFusionInstruction");

  CHECK(fused_root != nullptr);
  std::string fusion_name = [&] {
    if (fusion_kind == FusionKind::kInput) {
      return absl::StrCat("input_fusion_",
                          HloOpcodeString(fused_root->opcode()));

    } else {
      return std::string("fusion");
    }
  }();
  SetAndSanitizeName(fusion_name);
  set_parent(fused_root->parent());
  set_metadata(fused_root->metadata());
  CloneAndFuseInternal(fused_root);
}

HloFusionInstruction::HloFusionInstruction(
    const Shape& shape, FusionKind fusion_kind,
    absl::Span<HloInstruction* const> operands,
    HloComputation* fusion_computation)
    : HloInstruction(HloOpcode::kFusion, shape), fusion_kind_(fusion_kind) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_144(mht_144_v, 2110, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::HloFusionInstruction");

  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName("fusion");
  AppendComputation(fusion_computation);
  fusion_computation->SetFusionInstruction(this);
}

HloFusionInstruction::~HloFusionInstruction() {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_145(mht_145_v, 2122, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::~HloFusionInstruction");

  ClearFusionComputationInstruction();
}

void HloFusionInstruction::ClearFusionComputationInstruction() {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_146(mht_146_v, 2129, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::ClearFusionComputationInstruction");

  // Each fusion calls a single computation, but we use called_computations()
  // instead of fused_instructions_computation(), because the order in which
  // things get destructed can vary; the fusion computation's back-pointer may
  // already be null, which violates a check in fused_instructions_computation.
  for (HloComputation* computation : called_computations()) {
    // Some passes that rewrite fusions may reassign a fusion computation to a
    // different fusion instruction as this instruction gets destructed.
    if (computation->FusionInstruction() == this) {
      computation->SetFusionInstruction(nullptr);
    }
  }
}

void HloFusionInstruction::ClearCalledComputations() {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_147(mht_147_v, 2146, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::ClearCalledComputations");

  ClearFusionComputationInstruction();
  HloInstruction::ClearCalledComputations();
}

std::string HloFusionInstruction::ToCategory() const {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_148(mht_148_v, 2154, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::ToCategory");

  switch (fusion_kind()) {
    case FusionKind::kLoop:
      return "loop fusion";
    case FusionKind::kInput:
      return "input fusion";
    case FusionKind::kOutput:
      return "output fusion";
    case FusionKind::kCustom:
      return "custom fusion";
  }
}

HloInstructionProto HloFusionInstruction::ToProto() const {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_149(mht_149_v, 2170, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fusion_kind(xla::ToString(fusion_kind()));
  proto.add_called_computation_ids(
      fused_instructions_computation()->unique_id());
  return proto;
}

bool HloFusionInstruction::IsElementwiseImpl(
    const absl::optional<int64_t>& operand_idx) const {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_150(mht_150_v, 2182, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::IsElementwiseImpl");

  if (!operand_idx.has_value()) {
    for (auto* fused : fused_instructions()) {
      if (fused->opcode() != HloOpcode::kParameter && !fused->IsElementwise()) {
        return false;
      }
    }
    return true;
  }
  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction*> worklist;
  absl::flat_hash_set<const HloInstruction*> visited;
  worklist.push_back(fused_parameter(operand_idx.value()));
  visited.insert(fused_parameter(operand_idx.value()));
  while (!worklist.empty()) {
    HloInstruction* operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction* user : operand->users()) {
      CHECK_GE(user->unique_id(), 0);
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

HloInstruction* HloFusionInstruction::AddFusionOperand(
    HloInstruction* new_operand) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_151(mht_151_v, 2221, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::AddFusionOperand");

  CHECK_EQ(operand_count(),
           fused_instructions_computation()->parameter_instructions().size());
  const int64_t param_no = operand_count();
  std::string param_name = StrCat("param_", param_no);
  HloInstruction* fused_parameter =
      fused_instructions_computation()->AddParameter(
          HloInstruction::CreateParameter(param_no, new_operand->shape(),
                                          param_name));
  AppendOperand(new_operand);
  return fused_parameter;
}

void HloFusionInstruction::MergeFusionInstruction(
    HloFusionInstruction* instruction_to_merge) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_152(mht_152_v, 2238, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::MergeFusionInstruction");

  CHECK(absl::c_linear_search(operands(), instruction_to_merge));
  // Clone the instruction from which to merge fused instructions.
  std::unique_ptr<HloInstruction> cloned = instruction_to_merge->Clone();
  HloFusionInstruction* cloned_fusion =
      static_cast<HloFusionInstruction*>(cloned.get());
  // Replace uses of fused parameters with the corresponding operand of the
  // fusion.  Add all non-parameter fused instructions to
  // 'unfused_instructions' to be merged into 'this'.  This is done in reverse
  // post order.
  std::vector<HloInstruction*> unfused_instructions;
  auto fused_instructions = cloned_fusion->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
  for (auto fused_it = fused_instructions.rbegin();
       fused_it != fused_instructions.rend(); ++fused_it) {
    auto fused_instruction = *fused_it;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      TF_CHECK_OK(
          fused_instruction->ReplaceAllUsesWith(cloned_fusion->mutable_operand(
              fused_instruction->parameter_number())));
    } else {
      unfused_instructions.push_back(fused_instruction);
    }
  }

  // If there are no unfused instructions, the fused computation must consist
  // only of kParameter instructions. Make the operand of the corresponding
  // parameter number the new root.
  HloInstruction* unfused_root =
      unfused_instructions.empty()
          ? instruction_to_merge->mutable_operand(
                instruction_to_merge->fused_instructions_computation()
                    ->root_instruction()
                    ->parameter_number())
          : unfused_instructions.front();
  CHECK(unfused_root == cloned_fusion->fused_expression_root() ||
        unfused_instructions.empty());
  // Replace instruction_to_merge use of 'this' with unfused_root.
  TF_CHECK_OK(instruction_to_merge->ReplaceUseWith(this, unfused_root));

  // Build a dummy root for the cloned fusion as we may remove the original root
  // in the fusion process.
  if (!unfused_instructions.empty()) {
    HloComputation* computation = unfused_root->parent();
    auto* dummy_root = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
    computation->set_root_instruction(dummy_root,
                                      /*accept_different_shape=*/true);
  }

  // Fuse 'unfused_instructions' into 'this'. Everytime we fuse an instruction
  // we remove it from the closed fusion node. This is so that we don't add
  // extra users to the producer of that instruction (we use user count to
  // decide if a side-effectful instruction is fusible).
  for (auto& instruction : unfused_instructions) {
    auto* fused = FuseInstruction(instruction);
    TF_CHECK_OK(instruction->ReplaceAllUsesWith(fused));
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
  CHECK_EQ(0, cloned_fusion->user_count());
  TF_CHECK_OK(parent()->parent()->RemoveEmbeddedComputation(
      cloned_fusion->fused_instructions_computation()));
}

void HloFusionInstruction::MergeFusionInstructionIntoMultiOutput(
    HloFusionInstruction* instruction_to_merge) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_153(mht_153_v, 2306, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::MergeFusionInstructionIntoMultiOutput");

  // Add all non-parameter fused instructions to 'unfused_instructions' to be
  // merged into 'this'. `old_to_new' maps the instructions in the fused node
  // to the disassembled fusion instructions.
  // Note that we add the unfused instructions to this->parent_ computation.
  // This is necessary because the unique_id needs for an instruction and
  // it's only added when inserting to the computation.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> unfused_instructions;
  auto computation_to_merge =
      instruction_to_merge->fused_instructions_computation();
  auto post_order = computation_to_merge->MakeInstructionPostOrder();
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto fused_instruction = *rit;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&old_to_new, fused_instruction,
                  instruction_to_merge->mutable_operand(
                      fused_instruction->parameter_number()));
      continue;
    }

    // Here we clone the insertion and call FuseInstructionIntoMultiOutput()
    // which clones again. This can be improved.
    auto cloned_instruction =
        parent()->AddInstruction(fused_instruction->Clone());
    unfused_instructions.push_back(cloned_instruction);
    InsertOrDie(&old_to_new, fused_instruction, cloned_instruction);
  }
  for (auto unfused_instruction : unfused_instructions) {
    for (int64_t index = 0; index < unfused_instruction->operand_count();
         index++) {
      auto new_operand =
          FindOrDie(old_to_new, unfused_instruction->mutable_operand(index));
      TF_CHECK_OK(unfused_instruction->ReplaceOperandWith(index, new_operand));
    }
  }

  // If there are no unfused instructions, the fused computation must consist
  // only of kParameter instructions. Make the operand of the corresponding
  // parameter number the new root.
  HloInstruction* unfused_root =
      unfused_instructions.empty()
          ? instruction_to_merge->mutable_operand(
                instruction_to_merge->fused_instructions_computation()
                    ->root_instruction()
                    ->parameter_number())
          : unfused_instructions.front();
  TF_CHECK_OK(instruction_to_merge->ReplaceAllUsesWith(unfused_root));

  TF_CHECK_OK(
      instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge));
  if (GetModule()) {
    TF_CHECK_OK(GetModule()->RemoveEmbeddedComputation(computation_to_merge));
  }

  // Fuse the root instruction and generate multiple outputs.
  if (unfused_instructions.empty()) {
    return;
  }
  FuseInstructionIntoMultiOutput(unfused_root);
  TF_CHECK_OK(unfused_root->parent()->RemoveInstruction(unfused_root));
  // The rest instructions are of normal fusing.
  for (int64_t i = 1; i < unfused_instructions.size(); i++) {
    auto instruction = unfused_instructions[i];
    FuseInstruction(instruction);
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
}

HloComputation* HloFusionInstruction::fused_instructions_computation() const {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_154(mht_154_v, 2378, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_instructions_computation");

  CHECK(!called_computations().empty());
  auto* fused_instructions_computation = called_computations().front();
  CHECK(fused_instructions_computation->IsFusionComputation())
      << "Computation " << fused_instructions_computation->name()
      << " is not a fusion kind";
  return fused_instructions_computation;
}

HloInstruction* HloFusionInstruction::fused_expression_root() const {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_155(mht_155_v, 2390, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_expression_root");

  return fused_instructions_computation()->root_instruction();
}

HloInstruction* HloFusionInstruction::fused_parameter(
    int64_t parameter_number) const {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_156(mht_156_v, 2398, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_parameter");

  return fused_instructions_computation()->parameter_instruction(
      parameter_number);
}

const std::vector<HloInstruction*>& HloFusionInstruction::fused_parameters()
    const {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_157(mht_157_v, 2407, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_parameters");

  return fused_instructions_computation()->parameter_instructions();
}

const tensorflow::gtl::iterator_range<UnwrappingIterator<
    std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
HloFusionInstruction::fused_instructions() const {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_158(mht_158_v, 2416, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_instructions");

  const HloComputation* subcomp = fused_instructions_computation();
  return subcomp->instructions();
}

const tensorflow::gtl::iterator_range<
    UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
HloFusionInstruction::fused_instructions() {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_159(mht_159_v, 2426, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_instructions");

  return fused_instructions_computation()->instructions();
}

int64_t HloFusionInstruction::fused_instruction_count() const {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_160(mht_160_v, 2433, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::fused_instruction_count");

  return fused_instructions_computation()->instruction_count();
}

HloInstruction* HloFusionInstruction::FuseInstructionInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_161(mht_161_v, 2441, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::FuseInstructionInternal");

  // When add_output is false, this fusion instruction must be a user of
  // instruction_to_fuse.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_fuse));
  }
  HloInstruction* fused_instruction =
      CloneAndFuseInternal(instruction_to_fuse, add_output);
  return fused_instruction;
}

HloInstruction* HloFusionInstruction::CloneAndFuseInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_162(mht_162_v, 2456, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::CloneAndFuseInternal");

  CHECK(instruction_to_fuse->IsFusible()) << instruction_to_fuse->ToString();
  VLOG(3) << "CloneAndFuseInternal:\n" << instruction_to_fuse->ToString();
  HloInstruction* clone = nullptr;
  if (called_computations().empty()) {
    // New fusion instruction. It should not be a multioutput instruction.
    CHECK(!add_output);
    std::string computation_name = [&] {
      if (fusion_kind_ == FusionKind::kInput) {
        return absl::StrCat("input_fused_computation_",
                            HloOpcodeString(instruction_to_fuse->opcode()));

      } else {
        return std::string("fused_computation");
      }
    }();
    auto builder = HloComputation::Builder(computation_name, this);
    builder.AddInstruction(instruction_to_fuse->Clone(/*suffix=*/""));
    AppendComputation(
        CHECK_NOTNULL(GetModule())->AddEmbeddedComputation(builder.Build()));
    clone = fused_expression_root();
  } else {
    // When add_output is false, instruction_to_fuse is necessarily an operand
    // of the fusion instruction. After fusion this will no longer be the
    // case. Remove the operand from the operand list and remove its
    // corresponding fused parameter instruction. Renumber parameters as
    // necessary to make parameter numbers consistent with their index in the
    // fused_parameter_ vector.
    bool in_operand_list =
        absl::c_linear_search(operands(), instruction_to_fuse);
    CHECK(add_output || in_operand_list);
    if (instruction_to_fuse->opcode() == HloOpcode::kTuple) {
      // We assume all uses of a kTuple operation are GTE ops, not another
      // fusion node. In this case, we don't need to clone
      // 'instruction_to_fuse'.
      CHECK(!in_operand_list);
      clone = instruction_to_fuse;
    } else {
      clone = fused_instructions_computation()->AddInstruction(
          instruction_to_fuse->Clone(/*suffix=*/""));
    }
    const std::vector<HloInstruction*>& fused_parameters =
        fused_instructions_computation()->parameter_instructions();
    for (int64_t operand_num = 0; operand_num < operand_count();
         ++operand_num) {
      if (instruction_to_fuse == operand(operand_num)) {
        // replace the fused parameter instruction's uses with the clone.
        HloInstruction* fused_parameter = fused_parameters[operand_num];
        TF_CHECK_OK(fused_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding fused parameter and operand from their
        // respective vectors.
        TF_CHECK_OK(
            fused_instructions_computation()->RemoveParameter(operand_num));
        RemoveOperandAt(operand_num);
        break;
      }
    }
    // We've cloned instruction_to_fuse into this fusion instruction, so this
    // fusion instruction is no longer a use of instruction_to_fuse.
    if (in_operand_list) {
      DetachFrom(instruction_to_fuse);
      // When the instruction_to_fuse does not have other users, we don't need
      // to generate a multioutput fusion instruction.
      if (instruction_to_fuse->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const std::vector<HloInstruction*>& fused_parameters =
      fused_instructions_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the fusion instruction. A
  // complication is that some clone operands may already be operands of the
  // fusion instruction.
  for (int64_t operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the fusion node.
    CHECK_EQ(operands().size(), fused_parameters.size());
    HloInstruction* fused_param = nullptr;
    for (int64_t i = 0; i < operands().size(); ++i) {
      if (this->operand(i) == operand) {
        fused_param = fused_parameters[i];
        break;
      }
    }

    if (fused_param == nullptr) {
      // Clone's operand was not already an operand of the fusion
      // instruction. Add it as an operand and add a corresponding fused
      // parameter instruction.
      fused_param = AddFusionOperand(operand);
    }
    TF_CHECK_OK(clone->ReplaceOperandWith(operand_num, fused_param));
  }

  if (add_output) {
    CHECK_GT(instruction_to_fuse->user_count(), 0);
    // If this is already a multioutput fusion instruction, expand the root
    // tuple by 1.
    HloInstruction* fused_root = fused_expression_root();
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (fused_root->opcode() == HloOpcode::kTuple) {
      tuple_elements = fused_root->operands();
    } else {
      tuple_elements.push_back(fused_root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction* new_root = fused_instructions_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_instructions_computation()->set_root_instruction(new_root);
    *mutable_shape() = new_root->shape();
    if (fused_root->opcode() == HloOpcode::kTuple) {
      TF_CHECK_OK(
          fused_instructions_computation()->RemoveInstruction(fused_root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original fusion instruction.
    if (newly_created_tuple_instr) {
      HloInstruction* new_instr = parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(fused_root->shape(), this, 0));
      TF_CHECK_OK(ReplaceAllUsesWithDifferentShape(new_instr));
    }
    int64_t index = tuple_elements.size();
    if (instruction_to_fuse->opcode() == HloOpcode::kTuple) {
      CHECK_EQ(clone, instruction_to_fuse);
      index -= clone->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      const auto& users = clone->users();
      to_be_removed.reserve(users.size());
      for (auto old_gte : users) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64_t old_tuple_index = old_gte->tuple_index();
        HloInstruction* new_gte =
            parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        TF_CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        TF_CHECK_OK(parent()->RemoveInstruction(old_gte));
      }
    } else {
      HloInstruction* new_gte =
          parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      TF_CHECK_OK(instruction_to_fuse->ReplaceAllUsesWith(new_gte));
    }
  }

  if (clone != instruction_to_fuse) {
    VLOG(2) << "New clone:\n" << clone->ToString();
  }
  return clone;
}

std::vector<std::string> HloFusionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_163(mht_163_v, 2629, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::ExtraAttributesToStringImpl");

  return {StrCat("kind=", xla::ToString(fusion_kind()))};
}

bool HloFusionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_164(mht_164_v, 2639, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::IdenticalSlowPath");

  return fusion_kind() == other.fusion_kind() &&
         eq_computations(fused_instructions_computation(),
                         other.fused_instructions_computation());
}

std::unique_ptr<HloInstruction> HloFusionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_165(mht_165_v, 2650, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::CloneWithNewOperandsImpl");

  HloModule* module = context != nullptr ? context->module() : GetModule();
  HloComputation* new_fused_computation = nullptr;
  if (context != nullptr) {
    new_fused_computation =
        context->FindComputation(fused_instructions_computation());
  }
  if (new_fused_computation == nullptr) {
    new_fused_computation = module->AddEmbeddedComputation(
        fused_instructions_computation()->Clone("clone", context));
  }
  return absl::make_unique<HloFusionInstruction>(
      shape, fusion_kind(), new_operands, new_fused_computation);
}

Status HloFusionInstruction::DeduplicateFusionOperands() {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_166(mht_166_v, 2668, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloFusionInstruction::DeduplicateFusionOperands");

  if (IsCustomFusion()) {
    return Status::OK();
  }
  absl::flat_hash_map<const HloInstruction*, int> operand_indices;
  std::vector<int> operands_to_remove;
  const int count = operand_count();
  operands_to_remove.reserve(count);
  for (int i = 0; i < count; ++i) {
    auto emplace_result = operand_indices.emplace(operand(i), i);
    if (!emplace_result.second) {
      TF_RETURN_IF_ERROR(fused_parameter(i)->ReplaceAllUsesWith(
          fused_parameter(emplace_result.first->second)));
      operands_to_remove.push_back(i);
    }
  }
  if (operands_to_remove.empty()) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(fused_instructions_computation()
                         ->RemoveUnusedParametersFromFusedComputation());
  RemoveOperandsAtAscendingIndices(operands_to_remove);
  return Status::OK();
}

HloRngInstruction::HloRngInstruction(
    const Shape& shape, RandomDistribution distribution,
    absl::Span<HloInstruction* const> parameters)
    : HloInstruction(HloOpcode::kRng, shape), distribution_(distribution) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_167(mht_167_v, 2699, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::HloRngInstruction");

  for (HloInstruction* param : parameters) {
    AppendOperand(param);
  }
}

HloInstructionProto HloRngInstruction::ToProto() const {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_168(mht_168_v, 2708, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_distribution(distribution_);
  return proto;
}

std::vector<std::string> HloRngInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_169(mht_169_v, 2718, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::ExtraAttributesToStringImpl");

  return {StrCat("distribution=", RandomDistributionToString(distribution_))};
}

bool HloRngInstruction::IsElementwiseImpl(
    const absl::optional<int64_t>& operand_idx) const {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_170(mht_170_v, 2726, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::IsElementwiseImpl");

  return true;
}

bool HloRngInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_171(mht_171_v, 2736, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloRngInstruction&>(other);
  return distribution_ == casted_other.distribution_;
}

std::unique_ptr<HloInstruction> HloRngInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_172(mht_172_v, 2746, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloRngInstruction>(shape, distribution_,
                                              new_operands);
}

HloParameterInstruction::HloParameterInstruction(int64_t parameter_number,
                                                 const Shape& shape,
                                                 const std::string& name)
    : HloInstruction(HloOpcode::kParameter, shape),
      parameter_number_(parameter_number) {
   std::vector<std::string> mht_173_v;
   mht_173_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_173(mht_173_v, 2759, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::HloParameterInstruction");

  SetAndSanitizeName(name);
}

HloInstructionProto HloParameterInstruction::ToProto() const {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_174(mht_174_v, 2766, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_parameter_number(parameter_number_);
  if (parameter_replicated_at_leaf_buffers_) {
    for (bool replicated : *parameter_replicated_at_leaf_buffers_) {
      proto.mutable_parameter_replication()->add_replicated_at_leaf_buffers(
          replicated);
    }
  }
  return proto;
}

std::vector<std::string> HloParameterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_175(mht_175_v, 2782, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> result;
  if (!parameter_replicated_at_leaf_buffers_) {
    return result;
  }
  std::vector<std::string> buffers_replicated_strs;
  buffers_replicated_strs.reserve(
      parameter_replicated_at_leaf_buffers_->size());
  for (bool replicated : *parameter_replicated_at_leaf_buffers_) {
    buffers_replicated_strs.push_back(replicated ? "true" : "false");
  }
  if (options.print_ids()) {
    result.push_back(StrCat("parameter_replication={",
                            StrJoin(buffers_replicated_strs, ","), "}"));
  }
  return result;
}

std::string HloParameterInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_176(mht_176_v, 2805, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::OperandsToStringWithCanonicalNameMap");

  return StrCat(parameter_number_);
}

bool HloParameterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_177(mht_177_v, 2815, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloParameterInstruction&>(other);
  return parameter_number() == casted_other.parameter_number();
}

std::unique_ptr<HloInstruction>
HloParameterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_178(mht_178_v, 2826, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloParameterInstruction::CloneWithNewOperandsImpl");

  auto clone = absl::make_unique<HloParameterInstruction>(parameter_number_,
                                                          shape, name());
  if (parameter_replicated_at_leaf_buffers_ &&
      ShapeUtil::Equal(shape, this->shape())) {
    clone->set_parameter_replicated_at_leaf_buffers(
        *parameter_replicated_at_leaf_buffers_);
  }
  return clone;
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape& shape, HloInstruction* operand, int64_t index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_179(mht_179_v, 2842, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetTupleElementInstruction::HloGetTupleElementInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloGetTupleElementInstruction::ToProto() const {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_180(mht_180_v, 2849, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetTupleElementInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_tuple_index(tuple_index_);
  return proto;
}

std::vector<std::string>
HloGetTupleElementInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_181(mht_181_v, 2860, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetTupleElementInstruction::ExtraAttributesToStringImpl");

  return {StrCat("index=", tuple_index())};
}

bool HloGetTupleElementInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_182(mht_182_v, 2870, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetTupleElementInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloGetTupleElementInstruction&>(other);
  return tuple_index() == casted_other.tuple_index();
}

std::unique_ptr<HloInstruction>
HloGetTupleElementInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_183(mht_183_v, 2882, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetTupleElementInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloGetTupleElementInstruction>(
      shape, new_operands[0], tuple_index());
}

HloReducePrecisionInstruction::HloReducePrecisionInstruction(
    const Shape& shape, HloInstruction* operand, const int exponent_bits,
    const int mantissa_bits)
    : HloInstruction(HloOpcode::kReducePrecision, shape),
      exponent_bits_(exponent_bits),
      mantissa_bits_(mantissa_bits) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_184(mht_184_v, 2896, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReducePrecisionInstruction::HloReducePrecisionInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloReducePrecisionInstruction::ToProto() const {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_185(mht_185_v, 2903, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReducePrecisionInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_exponent_bits(exponent_bits_);
  proto.set_mantissa_bits(mantissa_bits_);
  return proto;
}

std::vector<std::string>
HloReducePrecisionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_186(mht_186_v, 2915, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReducePrecisionInstruction::ExtraAttributesToStringImpl");

  return {StrCat("exponent_bits=", exponent_bits_),
          StrCat("mantissa_bits=", mantissa_bits_)};
}

bool HloReducePrecisionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_187(mht_187_v, 2926, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReducePrecisionInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloReducePrecisionInstruction&>(other);
  // A reduce-precision operation is determined by the bit sizes.
  return exponent_bits() == casted_other.exponent_bits() &&
         mantissa_bits() == casted_other.mantissa_bits();
}

std::unique_ptr<HloInstruction>
HloReducePrecisionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_188(mht_188_v, 2940, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReducePrecisionInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloReducePrecisionInstruction>(
      shape, new_operands[0], exponent_bits(), mantissa_bits());
}

HloInfeedInstruction::HloInfeedInstruction(const Shape& infeed_shape,
                                           HloInstruction* token_operand,
                                           const std::string& config)
    : HloInstruction(HloOpcode::kInfeed,
                     ShapeUtil::MakeTupleShape(
                         {infeed_shape, ShapeUtil::MakeTokenShape()})),
      infeed_config_(config) {
   std::vector<std::string> mht_189_v;
   mht_189_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_189(mht_189_v, 2956, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloInfeedInstruction::HloInfeedInstruction");

  AppendOperand(token_operand);
}

HloInstructionProto HloInfeedInstruction::ToProto() const {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_190(mht_190_v, 2963, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloInfeedInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_infeed_config(infeed_config_);
  return proto;
}

std::vector<std::string> HloInfeedInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_191(mht_191_v, 2973, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloInfeedInstruction::ExtraAttributesToStringImpl");

  if (!options.print_infeed_outfeed_config() || infeed_config_.empty()) {
    return {};
  }
  return {StrCat("infeed_config=\"", CEscape(infeed_config_), "\"")};
}

bool HloInfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_192(mht_192_v, 2986, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloInfeedInstruction::IdenticalSlowPath");

  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloInfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_193(mht_193_v, 2996, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloInfeedInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloInfeedInstruction>(
      infeed_shape(), new_operands[0], infeed_config());
}

HloOutfeedInstruction::HloOutfeedInstruction(const Shape& outfeed_shape,
                                             HloInstruction* operand,
                                             HloInstruction* token_operand,
                                             absl::string_view outfeed_config)
    : HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeTokenShape()),
      outfeed_shape_(outfeed_shape),
      outfeed_config_(outfeed_config) {
   std::vector<std::string> mht_194_v;
   mht_194_v.push_back("outfeed_config: \"" + std::string(outfeed_config.data(), outfeed_config.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_194(mht_194_v, 3012, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloOutfeedInstruction::HloOutfeedInstruction");

  AppendOperand(operand);
  AppendOperand(token_operand);
}

HloInstructionProto HloOutfeedInstruction::ToProto() const {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_195(mht_195_v, 3020, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloOutfeedInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_outfeed_config(outfeed_config());
  *proto.mutable_outfeed_shape() = outfeed_shape().ToProto();
  return proto;
}

std::vector<std::string> HloOutfeedInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_196(mht_196_v, 3031, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloOutfeedInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra;
  extra.push_back(StrCat("outfeed_shape=",
                         ShapeUtil::HumanStringWithLayout(outfeed_shape_)));
  if (options.print_infeed_outfeed_config() && !outfeed_config_.empty()) {
    extra.push_back(
        StrCat("outfeed_config=\"", CEscape(outfeed_config_), "\""));
  }
  return extra;
}

bool HloOutfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_197(mht_197_v, 3048, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloOutfeedInstruction::IdenticalSlowPath");

  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloOutfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_198(mht_198_v, 3058, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloOutfeedInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloOutfeedInstruction>(
      outfeed_shape(), new_operands[0], new_operands[1], outfeed_config());
}

HloConvolutionInstruction::HloConvolutionInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64_t feature_group_count, int64_t batch_group_count,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kConvolution, shape),
      feature_group_count_(feature_group_count),
      batch_group_count_(batch_group_count),
      window_(window),
      convolution_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_199(mht_199_v, 3077, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::HloConvolutionInstruction");

  if (window_util::HasBaseDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-base-dilated"));
  }
  if (window_util::HasWindowDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-window-dilated"));
  }
  AppendOperand(lhs);
  AppendOperand(rhs);
}

std::string HloConvolutionInstruction::ToCategory() const {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_200(mht_200_v, 3091, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::ToCategory");

  std::string category = "convolution";
  if (window_util::HasBaseDilation(window())) {
    category += " base-dilated";
  }
  if (window_util::HasWindowDilation(window())) {
    category += " window-dilated";
  }
  return category;
}

HloInstructionProto HloConvolutionInstruction::ToProto() const {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_201(mht_201_v, 3105, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  *proto.mutable_convolution_dimension_numbers() =
      convolution_dimension_numbers_;
  proto.set_feature_group_count(feature_group_count_);
  proto.set_batch_group_count(batch_group_count_);
  *proto.mutable_precision_config() = precision_config_;
  return proto;
}

std::vector<std::string> HloConvolutionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_202(mht_202_v, 3120, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  extra.push_back(StrCat("dim_labels=", ConvolutionDimensionNumbersToString(
                                            convolution_dimension_numbers_)));
  if (feature_group_count_ != 1) {
    extra.push_back(StrCat("feature_group_count=", feature_group_count_));
  }

  if (batch_group_count_ != 1) {
    extra.push_back(StrCat("batch_group_count=", batch_group_count_));
  }

  std::string precision_config_string =
      PrecisionConfigToString(precision_config_);
  if (!precision_config_string.empty()) {
    extra.push_back(precision_config_string);
  }
  return extra;
}

bool HloConvolutionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_203(mht_203_v, 3149, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloConvolutionInstruction&>(other);
  if (feature_group_count_ != other.feature_group_count()) {
    return false;
  }
  if (batch_group_count_ != other.batch_group_count()) {
    return false;
  }
  return protobuf_util::ProtobufEquals(window(), casted_other.window()) &&
         protobuf_util::ProtobufEquals(
             convolution_dimension_numbers(),
             casted_other.convolution_dimension_numbers()) &&
         protobuf_util::ProtobufEquals(precision_config(),
                                       casted_other.precision_config());
}

std::unique_ptr<HloInstruction>
HloConvolutionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_204(mht_204_v, 3172, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloConvolutionInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloConvolutionInstruction>(
      shape, new_operands[0], new_operands[1], feature_group_count_,
      batch_group_count_, window(), convolution_dimension_numbers_,
      precision_config_);
}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation)
    : HloReduceWindowInstruction(shape, absl::MakeSpan(&operand, 1),
                                 absl::MakeSpan(&init_value, 1), window,
                                 reduce_computation) {
   std::vector<std::string> mht_205_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_205(mht_205_v, 3188, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::HloReduceWindowInstruction");
}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values, const Window& window,
    HloComputation* reduce_computation)
    : HloInstruction(HloOpcode::kReduceWindow, shape), window_(window) {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_206(mht_206_v, 3197, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::HloReduceWindowInstruction");

  for (auto* operand : operands) {
    AppendOperand(operand);
  }
  for (auto* init_value : init_values) {
    AppendOperand(init_value);
  }
  AppendComputation(reduce_computation);
}

HloInstructionProto HloReduceWindowInstruction::ToProto() const {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_207(mht_207_v, 3210, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

std::vector<std::string>
HloReduceWindowInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_208(mht_208_v, 3221, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  return extra;
}

bool HloReduceWindowInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_209(mht_209_v, 3235, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloReduceWindowInstruction&>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         protobuf_util::ProtobufEquals(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloReduceWindowInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_210(mht_210_v, 3248, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloReduceWindowInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size() % 2, 0);
  int64_t num_operands = new_operands.size() / 2;
  return absl::make_unique<HloReduceWindowInstruction>(
      shape, absl::MakeSpan(new_operands).subspan(0, num_operands),
      absl::MakeSpan(new_operands)
          .subspan(num_operands, new_operands.size() / 2),
      window(), to_apply());
}

HloSelectAndScatterInstruction::HloSelectAndScatterInstruction(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter)
    : HloInstruction(HloOpcode::kSelectAndScatter, shape), window_(window) {
   std::vector<std::string> mht_211_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_211(mht_211_v, 3265, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSelectAndScatterInstruction::HloSelectAndScatterInstruction");

  AppendOperand(operand);
  AppendOperand(source);
  AppendOperand(init_value);
  // Select comes before scatter in the vector.
  AppendComputation(select);
  AppendComputation(scatter);
}

HloInstructionProto HloSelectAndScatterInstruction::ToProto() const {
   std::vector<std::string> mht_212_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_212(mht_212_v, 3277, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSelectAndScatterInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

std::vector<std::string>
HloSelectAndScatterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_213_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_213(mht_213_v, 3288, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSelectAndScatterInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  return extra;
}

bool HloSelectAndScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_214_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_214(mht_214_v, 3302, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSelectAndScatterInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloSelectAndScatterInstruction&>(other);
  return eq_computations(select(), casted_other.select()) &&
         eq_computations(scatter(), casted_other.scatter()) &&
         protobuf_util::ProtobufEquals(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloSelectAndScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_215_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_215(mht_215_v, 3316, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSelectAndScatterInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloSelectAndScatterInstruction>(
      shape, new_operands[0], select(), window(), new_operands[1],
      new_operands[2], scatter());
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
   std::vector<std::string> mht_216_v;
   mht_216_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_216_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_216(mht_216_v, 3340, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::HloCustomCallInstruction");

  set_raw_backend_config_string(std::move(opaque));
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* to_apply, absl::string_view custom_call_target,
    std::string opaque, CustomCallApiVersion api_version)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
   std::vector<std::string> mht_217_v;
   mht_217_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_217_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_217(mht_217_v, 3364, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::HloCustomCallInstruction");

  set_raw_backend_config_string(std::move(opaque));
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(to_apply);
  to_apply->SetCustomCallInstruction(this);
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
   std::vector<std::string> mht_218_v;
   mht_218_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_218_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_218(mht_218_v, 3391, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::HloCustomCallInstruction");

  set_raw_backend_config_string(std::move(opaque));
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  for (auto comp : called_computations) {
    AppendComputation(comp);
  }
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, std::string opaque,
    absl::Span<const Shape> operand_shapes_with_layout,
    CustomCallApiVersion api_version)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(true),
      padding_type_(PaddingType::PADDING_INVALID),
      operand_shapes_with_layout_(operand_shapes_with_layout.begin(),
                                  operand_shapes_with_layout.end()),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
   std::vector<std::string> mht_219_v;
   mht_219_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_219_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_219(mht_219_v, 3421, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::HloCustomCallInstruction");

  set_raw_backend_config_string(std::move(opaque));
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCustomCallInstruction::ToProto() const {
   std::vector<std::string> mht_220_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_220(mht_220_v, 3431, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  if (window_ != nullptr) {
    *proto.mutable_window() = *window_;
  }
  if (convolution_dimension_numbers_ != nullptr) {
    *proto.mutable_convolution_dimension_numbers() =
        *convolution_dimension_numbers_;
  }
  proto.set_custom_call_target(custom_call_target_);
  proto.set_feature_group_count(feature_group_count_);
  proto.set_batch_group_count(batch_group_count_);
  *proto.mutable_precision_config() = precision_config_;
  proto.set_padding_type(padding_type_);
  if (layout_constrained()) {
    proto.set_constrain_layout(true);
    for (const Shape& shape : operand_shapes_with_layout_) {
      *proto.add_operand_shapes_with_layout() = shape.ToProto();
    }
  }
  proto.set_custom_call_has_side_effect(custom_call_has_side_effect_);
  if (literal_.has_value()) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  for (const auto& pair : output_to_operand_aliasing_) {
    auto aliasing = proto.add_custom_call_output_operand_aliasing();
    aliasing->set_operand_index(pair.second.first);
    for (int64_t index : pair.first) {
      aliasing->add_output_shape_index(index);
    }
    for (int64_t index : pair.second.second) {
      aliasing->add_operand_shape_index(index);
    }
  }
  proto.set_custom_call_schedule(custom_call_schedule_);
  proto.set_custom_call_api_version(api_version_);
  return proto;
}

std::vector<std::string> HloCustomCallInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_221_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_221(mht_221_v, 3474, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra;
  if (window_ != nullptr) {
    extra.push_back(StrCat("window={", window_util::ToString(*window_), "}"));
  }
  if (convolution_dimension_numbers_ != nullptr) {
    extra.push_back(StrCat(
        "dim_labels=",
        ConvolutionDimensionNumbersToString(*convolution_dimension_numbers_)));
  }
  if (feature_group_count_ != 1) {
    extra.push_back(StrCat("feature_group_count=", feature_group_count_));
  }
  if (batch_group_count_ != 1) {
    extra.push_back(StrCat("batch_group_count=", batch_group_count_));
  }
  std::string precision_config_string =
      PrecisionConfigToString(precision_config_);
  if (!precision_config_string.empty()) {
    extra.push_back(precision_config_string);
  }
  if (padding_type_ != PaddingType::PADDING_INVALID) {
    extra.push_back(StrCat("padding_type=", PaddingType_Name(padding_type())));
  }
  // By contract, we print the custom call target even if
  // options.print_subcomputation_mode() == kOff, because the call target is not
  // an HloComputation.
  extra.push_back(
      StrCat("custom_call_target=\"", CEscape(custom_call_target_), "\""));

  if (layout_constrained()) {
    std::vector<std::string> shape_strings;
    shape_strings.reserve(operand_shapes_with_layout_.size());
    for (const Shape& shape : operand_shapes_with_layout_) {
      shape_strings.push_back(ShapeUtil::HumanStringWithLayout(shape));
    }
    extra.push_back(StrCat("operand_layout_constraints={",
                           StrJoin(shape_strings, ", "), "}"));
  }
  if (custom_call_has_side_effect_) {
    extra.push_back("custom_call_has_side_effect=true");
  }
  if (literal_.has_value()) {
    extra.push_back(StrCat("literal=", literal_->ToStringWithLayoutOneline()));
  }
  if (!output_to_operand_aliasing_.empty()) {
    std::vector<std::string> pair_strings;
    pair_strings.reserve(output_to_operand_aliasing_.size());
    for (const auto& pair : output_to_operand_aliasing_) {
      pair_strings.push_back(StrCat(pair.first.ToString(), ": (",
                                    pair.second.first, ", ",
                                    pair.second.second.ToString(), ")"));
    }
    extra.push_back(StrCat("output_to_operand_aliasing={",
                           StrJoin(pair_strings, ", "), "}"));
  }
  if (custom_call_schedule_ != CustomCallSchedule::SCHEDULE_NONE) {
    extra.push_back(
        StrCat("schedule=", CustomCallSchedule_Name(custom_call_schedule_)));
  }
  if (api_version_ != CustomCallApiVersion::API_VERSION_ORIGINAL) {
    extra.push_back(
        StrCat("api_version=", CustomCallApiVersion_Name(api_version_)));
  }
  return extra;
}

bool HloCustomCallInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_222_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_222(mht_222_v, 3547, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloCustomCallInstruction&>(other);
  if ((window_ == nullptr) != (casted_other.window_ == nullptr) ||
      (window_ != nullptr &&
       !protobuf_util::ProtobufEquals(*window_, *casted_other.window_))) {
    return false;
  }
  if ((convolution_dimension_numbers_ == nullptr) !=
          (casted_other.convolution_dimension_numbers_ == nullptr) ||
      (convolution_dimension_numbers_ != nullptr &&
       !protobuf_util::ProtobufEquals(
           convolution_dimension_numbers(),
           casted_other.convolution_dimension_numbers()))) {
    return false;
  }
  if (feature_group_count_ != casted_other.feature_group_count_) {
    return false;
  }
  if (batch_group_count_ != casted_other.batch_group_count_) {
    return false;
  }

  if (padding_type_ != casted_other.padding_type()) {
    return false;
  }

  if (layout_constrained() != casted_other.layout_constrained()) {
    return false;
  }
  if (layout_constrained()) {
    for (int64_t i = 0; i < operand_shapes_with_layout_.size(); ++i) {
      if (!ShapeUtil::Equal(operand_shapes_with_layout_[i],
                            casted_other.operand_shapes_with_layout_[i])) {
        return false;
      }
    }
  }
  if (custom_call_has_side_effect_ !=
      casted_other.custom_call_has_side_effect()) {
    return false;
  }
  if (output_to_operand_aliasing_ !=
      casted_other.output_to_operand_aliasing()) {
    return false;
  }
  if (!protobuf_util::ProtobufEquals(precision_config(),
                                     casted_other.precision_config())) {
    return false;
  }

  if (called_computations().size() != other.called_computations().size()) {
    return false;
  }
  for (int64_t i = 0; i < called_computations().size(); ++i) {
    if (!eq_computations(called_computations()[i],
                         other.called_computations()[i])) {
      return false;
    }
  }
  if (custom_call_schedule_ != casted_other.custom_call_schedule()) {
    return false;
  }
  if (HasLiteral() == casted_other.HasLiteral()) {
    if (HasLiteral() && literal() == casted_other.literal()) {
      return false;
    }
  } else {
    return true;
  }
  // Note: backend_config comparison is done in Identical, which is the
  // intended/exposed way to compare computations, and so not repeated here.
  return custom_call_target_ == casted_other.custom_call_target_;
}

std::unique_ptr<HloInstruction>
HloCustomCallInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_223_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_223(mht_223_v, 3628, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloCustomCallInstruction::CloneWithNewOperandsImpl");

  auto cloned = absl::make_unique<HloCustomCallInstruction>(
      shape, new_operands, called_computations(), custom_call_target(),
      opaque(), api_version_);
  if (layout_constrained()) {
    cloned->layout_constrained_ = true;
    cloned->operand_shapes_with_layout_ = operand_shapes_with_layout();
  }
  if (window_ != nullptr) {
    cloned->set_window(*window_);
  }
  if (convolution_dimension_numbers_ != nullptr) {
    cloned->set_convolution_dimension_numbers(*convolution_dimension_numbers_);
  }
  if (HasLiteral()) {
    cloned->set_literal(literal().Clone());
  }
  cloned->set_feature_group_count(feature_group_count_);
  cloned->set_batch_group_count(batch_group_count_);
  cloned->set_custom_call_has_side_effect(custom_call_has_side_effect_);
  cloned->set_output_to_operand_aliasing(output_to_operand_aliasing_);
  cloned->set_padding_type(padding_type_);
  *cloned->mutable_precision_config() = precision_config();
  cloned->set_custom_call_schedule(custom_call_schedule_);
  return std::move(cloned);
}

HloPadInstruction::HloPadInstruction(const Shape& shape,
                                     HloInstruction* operand,
                                     HloInstruction* padding_value,
                                     const PaddingConfig& padding_config)
    : HloInstruction(HloOpcode::kPad, shape), padding_config_(padding_config) {
   std::vector<std::string> mht_224_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_224(mht_224_v, 3662, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloPadInstruction::HloPadInstruction");

  AppendOperand(operand);
  AppendOperand(padding_value);
}

HloInstructionProto HloPadInstruction::ToProto() const {
   std::vector<std::string> mht_225_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_225(mht_225_v, 3670, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloPadInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_padding_config() = padding_config_;
  return proto;
}

std::vector<std::string> HloPadInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_226_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_226(mht_226_v, 3680, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloPadInstruction::ExtraAttributesToStringImpl");

  return {StrCat("padding=", xla::PaddingConfigToString(padding_config_))};
}

bool HloPadInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_227_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_227(mht_227_v, 3690, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloPadInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloPadInstruction&>(other);
  return protobuf_util::ProtobufEquals(padding_config(),
                                       casted_other.padding_config());
}

std::unique_ptr<HloInstruction> HloPadInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_228_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_228(mht_228_v, 3701, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloPadInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloPadInstruction>(shape, new_operands[0],
                                              new_operands[1], padding_config_);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
   std::vector<std::string> mht_229_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_229(mht_229_v, 3714, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::HloDynamicSliceInstruction");

  AppendOperand(operand);
  AppendOperand(start_indices);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
   std::vector<std::string> mht_230_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_230(mht_230_v, 3727, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::HloDynamicSliceInstruction");

  AppendOperand(operand);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
   std::vector<std::string> mht_231_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_231(mht_231_v, 3740, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction");

  AppendOperand(operand);
  AppendOperand(update);
  AppendOperand(start_indices);
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
   std::vector<std::string> mht_232_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_232(mht_232_v, 3752, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction");

  AppendOperand(operand);
  AppendOperand(update);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
}

HloInstructionProto HloDynamicSliceInstruction::ToProto() const {
   std::vector<std::string> mht_233_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_233(mht_233_v, 3763, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t slice_size : dynamic_slice_sizes_) {
    proto.add_dynamic_slice_sizes(slice_size);
  }
  return proto;
}

std::vector<std::string>
HloDynamicSliceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_234_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_234(mht_234_v, 3776, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::ExtraAttributesToStringImpl");

  return {StrCat("dynamic_slice_sizes={", StrJoin(dynamic_slice_sizes(), ","),
                 "}")};
}

bool HloDynamicSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_235_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_235(mht_235_v, 3787, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloMapInstruction&>(other);
  return dynamic_slice_sizes() == casted_other.dynamic_slice_sizes();
}

std::unique_ptr<HloInstruction>
HloDynamicSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_236_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_236(mht_236_v, 3798, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDynamicSliceInstruction::CloneWithNewOperandsImpl");

  if (new_operands.size() == 2 && new_operands[1]->shape().rank() == 1) {
    // TODO(b/118437727): Old form, remove this path.
    return absl::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands[1], dynamic_slice_sizes_);
  } else {
    return absl::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands.subspan(1), dynamic_slice_sizes_);
  }
}

HloGatherInstruction::HloGatherInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted)
    : HloInstruction(HloOpcode::kGather, shape),
      indices_are_sorted_(indices_are_sorted) {
   std::vector<std::string> mht_237_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_237(mht_237_v, 3817, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::HloGatherInstruction");

  AppendOperand(operand);
  AppendOperand(start_indices);
  gather_dimension_numbers_ =
      absl::make_unique<GatherDimensionNumbers>(gather_dim_numbers);
  absl::c_copy(slice_sizes, std::back_inserter(gather_slice_sizes_));
}

/*static*/ std::string HloGatherInstruction::GatherDimensionNumbersToString(
    const GatherDimensionNumbers& gather_dimension_numbers) {
   std::vector<std::string> mht_238_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_238(mht_238_v, 3829, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::GatherDimensionNumbersToString");

  std::string offset_dims =
      StrCat("offset_dims={",
             StrJoin(gather_dimension_numbers.offset_dims(), ","), "}");
  std::string collapsed_slice_dims = StrCat(
      "collapsed_slice_dims={",
      StrJoin(gather_dimension_numbers.collapsed_slice_dims(), ","), "}");
  std::string start_index_map =
      StrCat("start_index_map={",
             StrJoin(gather_dimension_numbers.start_index_map(), ","), "}");
  std::string index_vector_dim =
      StrCat("index_vector_dim=", gather_dimension_numbers.index_vector_dim());

  return StrJoin<std::initializer_list<std::string>>(
      {offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim},
      ", ");
}

/* static */ GatherDimensionNumbers HloGatherInstruction::MakeGatherDimNumbers(
    absl::Span<const int64_t> offset_dims,
    absl::Span<const int64_t> collapsed_slice_dims,
    absl::Span<const int64_t> start_index_map, int64_t index_vector_dim) {
   std::vector<std::string> mht_239_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_239(mht_239_v, 3853, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::MakeGatherDimNumbers");

  GatherDimensionNumbers gather_dim_numbers;
  for (int64_t output_window_dim : offset_dims) {
    gather_dim_numbers.add_offset_dims(output_window_dim);
  }
  for (int64_t elided_window_dim : collapsed_slice_dims) {
    gather_dim_numbers.add_collapsed_slice_dims(elided_window_dim);
  }
  for (int64_t gather_dim_to_input_dim : start_index_map) {
    gather_dim_numbers.add_start_index_map(gather_dim_to_input_dim);
  }

  gather_dim_numbers.set_index_vector_dim(index_vector_dim);
  return gather_dim_numbers;
}

HloInstructionProto HloGatherInstruction::ToProto() const {
   std::vector<std::string> mht_240_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_240(mht_240_v, 3872, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_gather_dimension_numbers() = gather_dimension_numbers();
  for (int64_t bound : gather_slice_sizes()) {
    proto.add_gather_slice_sizes(bound);
  }
  proto.set_indices_are_sorted(indices_are_sorted());
  return proto;
}

std::vector<std::string> HloGatherInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_241_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_241(mht_241_v, 3886, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> attrs{
      GatherDimensionNumbersToString(gather_dimension_numbers()),
      StrCat("slice_sizes={", StrJoin(gather_slice_sizes(), ","), "}")};
  if (indices_are_sorted()) {
    attrs.push_back("indices_are_sorted=true");
  }
  return attrs;
}

bool HloGatherInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_242_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_242(mht_242_v, 3902, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloGatherInstruction&>(other);
  return protobuf_util::ProtobufEquals(
             gather_dimension_numbers(),
             casted_other.gather_dimension_numbers()) &&
         gather_slice_sizes() == casted_other.gather_slice_sizes() &&
         indices_are_sorted() == casted_other.indices_are_sorted();
}

std::unique_ptr<HloInstruction> HloGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_243_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_243(mht_243_v, 3916, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGatherInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloGatherInstruction>(
      shape, new_operands[0], new_operands[1], gather_dimension_numbers(),
      gather_slice_sizes(), indices_are_sorted());
}

HloScatterInstruction::HloScatterInstruction(
    const Shape& shape, HloInstruction* operand,
    HloInstruction* scatter_indices, HloInstruction* updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices)
    : HloInstruction(HloOpcode::kScatter, shape),
      indices_are_sorted_(indices_are_sorted),
      unique_indices_(unique_indices) {
   std::vector<std::string> mht_244_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_244(mht_244_v, 3934, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::HloScatterInstruction");

  AppendOperand(operand);
  AppendOperand(scatter_indices);
  AppendOperand(updates);
  AppendComputation(update_computation);
  scatter_dimension_numbers_ =
      absl::make_unique<ScatterDimensionNumbers>(scatter_dim_numbers);
}

/*static*/ std::string HloScatterInstruction::ScatterDimensionNumbersToString(
    const ScatterDimensionNumbers& scatter_dimension_numbers) {
   std::vector<std::string> mht_245_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_245(mht_245_v, 3947, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::ScatterDimensionNumbersToString");

  std::string update_window_dims =
      StrCat("update_window_dims={",
             StrJoin(scatter_dimension_numbers.update_window_dims(), ","), "}");
  std::string inserted_window_dims = StrCat(
      "inserted_window_dims={",
      StrJoin(scatter_dimension_numbers.inserted_window_dims(), ","), "}");
  std::string scatter_dims_to_operand_dims = StrCat(
      "scatter_dims_to_operand_dims={",
      StrJoin(scatter_dimension_numbers.scatter_dims_to_operand_dims(), ","),
      "}");
  std::string index_vector_dim =
      StrCat("index_vector_dim=", scatter_dimension_numbers.index_vector_dim());

  return StrJoin<std::initializer_list<std::string>>(
      {update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims,
       index_vector_dim},
      ", ");
}

/* static */ ScatterDimensionNumbers
HloScatterInstruction::MakeScatterDimNumbers(
    absl::Span<const int64_t> update_window_dims,
    absl::Span<const int64_t> inserted_window_dims,
    absl::Span<const int64_t> scatter_dims_to_operand_dims,
    int64_t index_vector_dim) {
   std::vector<std::string> mht_246_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_246(mht_246_v, 3975, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::MakeScatterDimNumbers");

  ScatterDimensionNumbers scatter_dim_numbers;
  for (int64_t update_window_dim : update_window_dims) {
    scatter_dim_numbers.add_update_window_dims(update_window_dim);
  }
  for (int64_t inserted_window_dim : inserted_window_dims) {
    scatter_dim_numbers.add_inserted_window_dims(inserted_window_dim);
  }
  for (int64_t scatter_dim_to_operand_dim : scatter_dims_to_operand_dims) {
    scatter_dim_numbers.add_scatter_dims_to_operand_dims(
        scatter_dim_to_operand_dim);
  }
  scatter_dim_numbers.set_index_vector_dim(index_vector_dim);
  return scatter_dim_numbers;
}

HloInstructionProto HloScatterInstruction::ToProto() const {
   std::vector<std::string> mht_247_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_247(mht_247_v, 3994, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_scatter_dimension_numbers() = scatter_dimension_numbers();
  proto.set_indices_are_sorted(indices_are_sorted());
  proto.set_unique_indices(unique_indices());
  return proto;
}

std::vector<std::string> HloScatterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_248_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_248(mht_248_v, 4006, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> attrs{
      ScatterDimensionNumbersToString(scatter_dimension_numbers())};
  if (indices_are_sorted()) {
    attrs.push_back("indices_are_sorted=true");
  }
  if (unique_indices()) {
    attrs.push_back("unique_indices=true");
  }
  return attrs;
}

bool HloScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_249_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_249(mht_249_v, 4024, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloScatterInstruction&>(other);
  return protobuf_util::ProtobufEquals(
             scatter_dimension_numbers(),
             casted_other.scatter_dimension_numbers()) &&
         eq_computations(to_apply(), casted_other.to_apply()) &&
         indices_are_sorted() == casted_other.indices_are_sorted() &&
         unique_indices() == casted_other.unique_indices();
}

std::unique_ptr<HloInstruction> HloScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_250_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_250(mht_250_v, 4039, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloScatterInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloScatterInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], to_apply(),
      scatter_dimension_numbers(), indices_are_sorted(), unique_indices());
}

HloIotaInstruction::HloIotaInstruction(const Shape& shape,
                                       int64_t iota_dimension)
    : HloInstruction(HloOpcode::kIota, shape),
      iota_dimension_(iota_dimension) {
   std::vector<std::string> mht_251_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_251(mht_251_v, 4052, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloIotaInstruction::HloIotaInstruction");
}

HloInstructionProto HloIotaInstruction::ToProto() const {
   std::vector<std::string> mht_252_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_252(mht_252_v, 4057, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloIotaInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(iota_dimension());
  return proto;
}

std::vector<std::string> HloIotaInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_253_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_253(mht_253_v, 4067, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloIotaInstruction::ExtraAttributesToStringImpl");

  return {StrCat("iota_dimension=", iota_dimension())};
}

bool HloIotaInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_254_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_254(mht_254_v, 4077, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloIotaInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloIotaInstruction&>(other);
  return iota_dimension() == casted_other.iota_dimension();
}

std::unique_ptr<HloInstruction> HloIotaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_255_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_255(mht_255_v, 4087, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloIotaInstruction::CloneWithNewOperandsImpl");

  return absl::make_unique<HloIotaInstruction>(shape, iota_dimension());
}

HloDotInstruction::HloDotInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kDot, shape),
      dot_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
   std::vector<std::string> mht_256_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_256(mht_256_v, 4100, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDotInstruction::HloDotInstruction");

  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloDotInstruction::ToProto() const {
   std::vector<std::string> mht_257_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_257(mht_257_v, 4108, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDotInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_dot_dimension_numbers() = dot_dimension_numbers_;
  *proto.mutable_precision_config() = precision_config_;
  return proto;
}

std::vector<std::string> HloDotInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_258_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_258(mht_258_v, 4119, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDotInstruction::ExtraAttributesToStringImpl");

  std::vector<std::string> extra = {
      DotDimensionNumbersToString(dot_dimension_numbers_)};

  std::string precision_config_string =
      PrecisionConfigToString(precision_config_);
  if (!precision_config_string.empty()) {
    extra.push_back(precision_config_string);
  }
  return extra;
}

bool HloDotInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_259_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_259(mht_259_v, 4137, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDotInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloDotInstruction&>(other);
  return protobuf_util::ProtobufEquals(dot_dimension_numbers(),
                                       casted_other.dot_dimension_numbers()) &&
         protobuf_util::ProtobufEquals(precision_config(),
                                       casted_other.precision_config());
}

std::unique_ptr<HloInstruction> HloDotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_260_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_260(mht_260_v, 4150, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDotInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloDotInstruction>(
      shape, new_operands[0], new_operands[1], dot_dimension_numbers_,
      precision_config_);
}

HloDomainInstruction::HloDomainInstruction(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata)
    : HloInstruction(HloOpcode::kDomain, shape),
      operand_side_metadata_(std::move(operand_side_metadata)),
      user_side_metadata_(std::move(user_side_metadata)) {
   std::vector<std::string> mht_261_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_261(mht_261_v, 4166, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDomainInstruction::HloDomainInstruction");

  AppendOperand(operand);
}

std::vector<std::string> HloDomainInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_262_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_262(mht_262_v, 4174, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDomainInstruction::ExtraAttributesToStringImpl");

  if (operand_side_metadata_ != nullptr && user_side_metadata_ != nullptr) {
    return {StrCat("domain={kind=\"", operand_side_metadata_->Kind(),
                   "\", entry=", user_side_metadata_->ToString(),
                   ", exit=", operand_side_metadata_->ToString(), "}")};
  }
  return {};
}

bool HloDomainInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_263_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_263(mht_263_v, 4189, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDomainInstruction::IdenticalSlowPath");

  const auto& casted_other = static_cast<const HloDomainInstruction&>(other);
  return operand_side_metadata().Matches(
             casted_other.operand_side_metadata()) &&
         user_side_metadata().Matches(casted_other.user_side_metadata());
}

std::unique_ptr<HloInstruction> HloDomainInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_264_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_264(mht_264_v, 4201, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDomainInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloDomainInstruction>(
      shape, new_operands[0], operand_side_metadata_->Clone(),
      user_side_metadata_->Clone());
}

HloInstructionProto HloDomainInstruction::ToProto() const {
   std::vector<std::string> mht_265_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_265(mht_265_v, 4211, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloDomainInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  auto operand_side_sharding =
      dynamic_cast<const ShardingMetadata*>(operand_side_metadata_.get());
  if (operand_side_sharding && operand_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_entry_sharding() =
        operand_side_sharding->sharding()->ToProto();
  }

  auto user_side_sharding =
      dynamic_cast<const ShardingMetadata*>(user_side_metadata_.get());
  if (user_side_sharding && user_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_exit_sharding() =
        user_side_sharding->sharding()->ToProto();
  }

  return proto;
}

HloGetDimensionSizeInstruction::HloGetDimensionSizeInstruction(
    const Shape& shape, HloInstruction* operand, int64_t dimension)
    : HloInstruction(HloOpcode::kGetDimensionSize, shape),
      dimension_(dimension) {
   std::vector<std::string> mht_266_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_266(mht_266_v, 4236, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetDimensionSizeInstruction::HloGetDimensionSizeInstruction");

  AppendOperand(operand);
}

HloInstructionProto HloGetDimensionSizeInstruction::ToProto() const {
   std::vector<std::string> mht_267_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_267(mht_267_v, 4243, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetDimensionSizeInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

std::vector<std::string>
HloGetDimensionSizeInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
   std::vector<std::string> mht_268_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_268(mht_268_v, 4254, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetDimensionSizeInstruction::ExtraAttributesToStringImpl");

  return {StrCat("dimensions={", dimension(), "}")};
}

bool HloGetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
   std::vector<std::string> mht_269_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_269(mht_269_v, 4264, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetDimensionSizeInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloGetDimensionSizeInstruction&>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloGetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_270_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_270(mht_270_v, 4276, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloGetDimensionSizeInstruction::CloneWithNewOperandsImpl");

  if (new_operands.size() != 1) {
    LOG(FATAL) << "expects 1 operand";
  }
  return absl::make_unique<HloGetDimensionSizeInstruction>(
      shape, new_operands[0], dimension());
}

HloSetDimensionSizeInstruction::HloSetDimensionSizeInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* val,
    int64_t dimension)
    : HloInstruction(HloOpcode::kSetDimensionSize, shape),
      dimension_(dimension) {
   std::vector<std::string> mht_271_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_271(mht_271_v, 4291, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSetDimensionSizeInstruction::HloSetDimensionSizeInstruction");

  AppendOperand(operand);
  AppendOperand(val);
}

std::vector<std::string>
HloSetDimensionSizeInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
   std::vector<std::string> mht_272_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_272(mht_272_v, 4301, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSetDimensionSizeInstruction::ExtraAttributesToStringImpl");

  return {StrCat("dimensions={", dimension(), "}")};
}

HloInstructionProto HloSetDimensionSizeInstruction::ToProto() const {
   std::vector<std::string> mht_273_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_273(mht_273_v, 4308, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSetDimensionSizeInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

bool HloSetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
   std::vector<std::string> mht_274_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_274(mht_274_v, 4320, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSetDimensionSizeInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloSetDimensionSizeInstruction&>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloSetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_275_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_275(mht_275_v, 4332, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloSetDimensionSizeInstruction::CloneWithNewOperandsImpl");

  if (new_operands.size() != 2) {
    LOG(FATAL) << "expects 2 operand";
  }
  return absl::make_unique<HloSetDimensionSizeInstruction>(
      shape, new_operands[0], new_operands[1], dimension());
}

HloRngGetAndUpdateStateInstruction::HloRngGetAndUpdateStateInstruction(
    const Shape& shape, int64_t delta)
    : HloInstruction(HloOpcode::kRngGetAndUpdateState, shape), delta_(delta) {
   std::vector<std::string> mht_276_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_276(mht_276_v, 4345, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngGetAndUpdateStateInstruction::HloRngGetAndUpdateStateInstruction");
}

HloInstructionProto HloRngGetAndUpdateStateInstruction::ToProto() const {
   std::vector<std::string> mht_277_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_277(mht_277_v, 4350, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngGetAndUpdateStateInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_delta(delta_);
  return proto;
}

std::vector<std::string>
HloRngGetAndUpdateStateInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
   std::vector<std::string> mht_278_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_278(mht_278_v, 4361, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngGetAndUpdateStateInstruction::ExtraAttributesToStringImpl");

  return {StrCat("delta=", delta())};
}

bool HloRngGetAndUpdateStateInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
   std::vector<std::string> mht_279_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_279(mht_279_v, 4371, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngGetAndUpdateStateInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloRngGetAndUpdateStateInstruction&>(other);
  return delta() == casted_other.delta();
}

std::unique_ptr<HloInstruction>
HloRngGetAndUpdateStateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_280_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_280(mht_280_v, 4383, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngGetAndUpdateStateInstruction::CloneWithNewOperandsImpl");

  if (!new_operands.empty()) {
    LOG(FATAL) << "expects 0 operand";
  }
  return absl::make_unique<HloRngGetAndUpdateStateInstruction>(shape, delta());
}

HloRngBitGeneratorInstruction::HloRngBitGeneratorInstruction(
    const Shape& shape, HloInstruction* state, RandomAlgorithm algorithm)
    : HloInstruction(HloOpcode::kRngBitGenerator, shape),
      algorithm_(algorithm) {
   std::vector<std::string> mht_281_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_281(mht_281_v, 4396, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngBitGeneratorInstruction::HloRngBitGeneratorInstruction");

  AppendOperand(state);
}

HloInstructionProto HloRngBitGeneratorInstruction::ToProto() const {
   std::vector<std::string> mht_282_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_282(mht_282_v, 4403, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngBitGeneratorInstruction::ToProto");

  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_rng_algorithm(algorithm_);
  return proto;
}

std::vector<std::string>
HloRngBitGeneratorInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_283_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_283(mht_283_v, 4414, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngBitGeneratorInstruction::ExtraAttributesToStringImpl");

  return {StrCat("algorithm=", RandomAlgorithmToString(algorithm_))};
}

bool HloRngBitGeneratorInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_284_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_284(mht_284_v, 4424, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngBitGeneratorInstruction::IdenticalSlowPath");

  const auto& casted_other =
      static_cast<const HloRngBitGeneratorInstruction&>(other);
  return algorithm() == casted_other.algorithm();
}

std::unique_ptr<HloInstruction>
HloRngBitGeneratorInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
   std::vector<std::string> mht_285_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionsDTcc mht_285(mht_285_v, 4436, "", "./tensorflow/compiler/xla/service/hlo_instructions.cc", "HloRngBitGeneratorInstruction::CloneWithNewOperandsImpl");

  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRngBitGeneratorInstruction>(
      shape, new_operands[0], algorithm());
}

}  // namespace xla
