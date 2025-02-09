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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc() {
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

#include "tensorflow/compiler/xla/service/bfloat16_conversion_folding.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

class BFloat16ConversionFoldingVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit BFloat16ConversionFoldingVisitor(
      HloComputation* computation, const BFloat16Support* bfloat16_support,
      BFloat16ConversionFolding* bfloat16_conversion_folding)
      : computation_(computation),
        bfloat16_support_(bfloat16_support),
        bfloat16_conversion_folding_(bfloat16_conversion_folding) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor");
}

  Status DefaultAction(HloInstruction* hlo) override;

  // Special handling for all-reduce which can have a tuple output.
  Status HandleAllReduce(HloInstruction* crs) override;

  static bool Run(HloComputation* computation,
                  const BFloat16Support* bfloat16_support,
                  BFloat16ConversionFolding* bfloat16_conversion_folding) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "Run");

    BFloat16ConversionFoldingVisitor visitor(computation, bfloat16_support,
                                             bfloat16_conversion_folding);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

 private:
  // Checks if the HLO has a BF16 -> F32 conversion as input, or a F32 -> BF16
  // conversion as output, and folds them to the HLO itself if feasible.
  Status TryFoldBF16Conversions(HloInstruction* hlo);

  // Folds the F32 -> BF16 conversions from the HLO's output.
  //
  // Precondition: all of the HLO's users are F32 -> BF16 conversions.
  Status FoldOutputConversions(HloInstruction* hlo);

  // Folds the BF16 -> F32 conversion operand to the HLO.
  //
  // Precondition: the operand is a BF16 -> F32 conversion.
  Status FoldOperandConversion(HloInstruction* hlo, int64_t operand_index);

  HloComputation* computation_;
  const BFloat16Support* bfloat16_support_;
  BFloat16ConversionFolding* bfloat16_conversion_folding_;
  bool changed_ = false;
};

Status BFloat16ConversionFoldingVisitor::FoldOutputConversions(
    HloInstruction* hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor::FoldOutputConversions");

  std::vector<HloInstruction*> materialized_users = hlo->users();
  hlo->mutable_shape()->set_element_type(BF16);
  bfloat16_conversion_folding_->UpdateLayout(hlo->mutable_shape());
  for (auto user : materialized_users) {
    CHECK_EQ(user->opcode(), HloOpcode::kConvert);
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(hlo));
    changed_ = true;
  }
  return Status::OK();
}

Status BFloat16ConversionFoldingVisitor::FoldOperandConversion(
    HloInstruction* hlo, int64_t operand_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor::FoldOperandConversion");

  // The operand is a convert from BF16 to F32.
  auto operand = hlo->mutable_operand(operand_index);
  CHECK_EQ(operand->opcode(), HloOpcode::kConvert);
  TF_RETURN_IF_ERROR(
      hlo->ReplaceOperandWith(operand_index, operand->mutable_operand(0)));
  changed_ = true;
  return Status::OK();
}

namespace {

// Returns whether hlo has users and all users are conversions from F32 to BF16.
bool AllUsersAreF32ToBF16Converts(const HloInstruction* hlo) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_4(mht_4_v, 282, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "AllUsersAreF32ToBF16Converts");

  if (hlo->user_count() == 0 || hlo->shape().element_type() != F32) {
    return false;
  }
  for (const auto user : hlo->users()) {
    if (user->opcode() == HloOpcode::kConvert &&
        user->shape().element_type() == BF16) {
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

Status BFloat16ConversionFoldingVisitor::TryFoldBF16Conversions(
    HloInstruction* hlo) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_5(mht_5_v, 302, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor::TryFoldBF16Conversions");

  std::vector<int64_t> bf16_to_f32_operands;
  bool has_other_f32_operands = false;
  for (int64_t i = 0; i < hlo->operands().size(); ++i) {
    auto operand = hlo->operand(i);
    if (operand->shape().element_type() == F32) {
      if (operand->opcode() == HloOpcode::kConvert &&
          operand->operand(0)->shape().element_type() == BF16 &&
          bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
        // Operand is a convert from BF16 to F32 and we support BF16 input
        // directly in the current HLO at the operand index.
        bf16_to_f32_operands.push_back(i);
      } else {
        has_other_f32_operands = true;
      }
      continue;
    }
  }

  const bool fold_output_conversion =
      AllUsersAreF32ToBF16Converts(hlo) &&
      bfloat16_support_->SupportsBF16Output(*hlo);

  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    if (has_other_f32_operands ||
        (!fold_output_conversion && hlo->shape().element_type() == F32)) {
      // Some of the operands/output will remain F32, but we cannot use mixed
      // precisions, so we cannot do anything here.
      return Status::OK();
    }
  }

  if (fold_output_conversion) {
    TF_RETURN_IF_ERROR(FoldOutputConversions(hlo));
  }

  for (int64_t i : bf16_to_f32_operands) {
    TF_RETURN_IF_ERROR(FoldOperandConversion(hlo, i));
  }
  return Status::OK();
}

Status BFloat16ConversionFoldingVisitor::DefaultAction(HloInstruction* hlo) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_6(mht_6_v, 347, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor::DefaultAction");

  // Do not fold BF16 conversions for instructions related to tuples, entry and
  // exit of a computation, fusion, convert, side-effecting instructions,
  // in-place operations and control flow.
  if (hlo->opcode() == HloOpcode::kTuple ||                      //
      hlo->opcode() == HloOpcode::kGetTupleElement ||            //
      hlo->opcode() == HloOpcode::kConstant ||                   //
      hlo->opcode() == HloOpcode::kParameter ||                  //
      hlo->opcode() == HloOpcode::kFusion ||                     //
      hlo->opcode() == HloOpcode::kBitcastConvert ||             //
      hlo->opcode() == HloOpcode::kConvert ||                    //
      hlo->opcode() == HloOpcode::kCall ||                       //
      hlo->opcode() == HloOpcode::kCustomCall ||                 //
      hlo->opcode() == HloOpcode::kWhile ||                      //
      hlo->opcode() == HloOpcode::kConditional ||                //
      HloDataflowAnalysis::IsInPlaceOperation(hlo->opcode()) ||  //
      hlo->HasSideEffectNoRecurse()) {
    return Status::OK();
  }
  if (hlo == computation_->root_instruction() &&
      !bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    // If hlo is the root instruction, we cannot change its output, so folding
    // can only happen when it supports mixed precision so that we can change
    // its operands.
    return Status::OK();
  }
  return TryFoldBF16Conversions(hlo);
}

Status BFloat16ConversionFoldingVisitor::HandleAllReduce(HloInstruction* crs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_7(mht_7_v, 379, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFoldingVisitor::HandleAllReduce");

  if (crs->HasSideEffectNoRecurse()) {
    // Do not perform optimization on side-effected AllReduce.
    return Status::OK();
  }
  // First use DefaultAction() to handle the operands. It can't handle
  // tuple-shaped output.
  TF_RETURN_IF_ERROR(DefaultAction(crs));

  if (!bfloat16_support_->SupportsMixedPrecisions(*crs)) {
    return Status::OK();
  }

  // If the output is not a tuple, we don't need special handling.
  if (!crs->shape().IsTuple()) {
    return Status::OK();
  }

  // If crs is the root instruction, we should keep its original output type.
  // The root instruction implicitly has a use from being the result of the
  // computation, and the code below does not take this use into account.
  if (crs == computation_->root_instruction()) {
    return Status::OK();
  }

  // Then do per-tuple-element handling on the output.
  std::vector<std::vector<HloInstruction*>> per_tuple_element_gtes(
      crs->operand_count());
  for (auto user : crs->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return Status::OK();
    }
    per_tuple_element_gtes[user->tuple_index()].push_back(user);
  }

  for (int64_t i = 0; i < crs->operand_count(); ++i) {
    // Fold conversions only when all the get-tuple-elements' users are
    // conversions from F32 to BF16.
    auto all_gte_users_are_bf16_convert = [&per_tuple_element_gtes, i]() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_8(mht_8_v, 420, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "lambda");

      // If no uses then return false. (As no uses are bf16 converts).
      if (per_tuple_element_gtes[i].empty()) {
        return false;
      }
      for (auto gte : per_tuple_element_gtes[i]) {
        if (!AllUsersAreF32ToBF16Converts(gte)) {
          return false;
        }
      }
      return true;
    };
    if (!all_gte_users_are_bf16_convert()) {
      continue;
    }

    ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i})
        ->set_element_type(BF16);
    bfloat16_conversion_folding_->UpdateLayout(
        ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i}));
    for (auto gte : per_tuple_element_gtes[i]) {
      TF_RETURN_IF_ERROR(FoldOutputConversions(gte));
    }
  }

  return Status::OK();
}

StatusOr<bool> BFloat16ConversionFolding::Run(HloModule* module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_conversion_foldingDTcc mht_9(mht_9_v, 451, "", "./tensorflow/compiler/xla/service/bfloat16_conversion_folding.cc", "BFloat16ConversionFolding::Run");

  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (BFloat16ConversionFoldingVisitor::Run(comp, bfloat16_support_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
