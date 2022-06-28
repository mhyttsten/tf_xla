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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

// Returns true if all instructions are supported operations.
static bool AreInstructionSupported(HloComputation* comp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.cc", "AreInstructionSupported");

  for (HloInstruction* instr : comp->instructions()) {
    bool supported =
        HloInstruction::IsOpElementwise(instr->opcode()) ||
        instr->opcode() == HloOpcode::kConstant ||
        // We only support reduction when they are at the root or when
        // in a MOF, at the end. This should always be true for now,
        // but if we implement reduction epilog fusion in the future,
        // this optimization need to be updated. So disable it just for
        // future safety.
        (instr->opcode() == HloOpcode::kReduce &&
         (comp->root_instruction() == instr ||
          (instr->users().size() == 1 &&
           instr->users()[0]->opcode() == HloOpcode::kTuple))) ||
        instr->opcode() == HloOpcode::kTuple ||
        instr->opcode() == HloOpcode::kParameter ||
        (instr->opcode() == HloOpcode::kBitcast &&
         instr->shape().rank() < instr->operand(0)->shape().rank()) ||
        (instr->opcode() == HloOpcode::kBroadcast &&
         (instr->dimensions().empty() ||       // scalar broadcasting
          (instr->dimensions().size() == 1 &&  // row broadcasting
           instr->dimensions()[0] == (instr->shape().rank() - 1))));
    if (!supported) {
      VLOG(2) << "NOT SUPPORTED: " << instr->ToString();
      return false;
    }

    // If there is an instruction that change the layout, we do not do
    // anything.
    if (HloInstruction::IsOpElementwise(instr->opcode()) &&
        !absl::c_all_of(instr->operands(), [&](HloInstruction* input) {
          return ShapeUtil::EqualIgnoringElementType(input->shape(),
                                                     instr->shape());
        })) {
      VLOG(2) << "NOT SUPPORTED (instruction change the layout): "
              << instr->ToString();
      return false;
    }
  }
  return true;
}

StatusOr<bool> FusionBitcastLift::Run(HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.cc", "FusionBitcastLift::Run");

  XLA_VLOG_LINES(2, "FusionBitcastLift::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    // Copy the instruction list as we modify the HloComputation.
    std::vector<HloInstruction*> comp_instruction(comp->instructions().begin(),
                                                  comp->instructions().end());
    for (HloInstruction* instr : comp_instruction) {
      // 1) Is this a fusion that we want to modify.
      if (auto* fusion = DynCast<HloFusionInstruction>(instr)) {
        // 1.1) We only support kInput fusion and some operations.
        if (fusion->fusion_kind() != HloInstruction::FusionKind::kInput ||
            !AreInstructionSupported(
                fusion->fused_instructions_computation())) {
          continue;
        }
        // 1.2) Check if there is a bitcast that we lift. Currently
        //      we do not lift(merge) bitcast above(with) broadcast.
        if (!std::any_of(
                fusion->fused_instructions().begin(),
                fusion->fused_instructions().end(), [](HloInstruction* inner) {
                  return inner->opcode() == HloOpcode::kBitcast &&
                         inner->operand(0)->opcode() != HloOpcode::kBroadcast;
                })) {
          continue;
        }

        // 1.3) Check that all the bitcast have the same shape pattern.
        //      Multiple bitcast pattern isn't supported/tested.
        HloInstruction* bitcast = nullptr;
        bool same_shape_pattern = true;
        for (HloInstruction* fused_instr : fusion->fused_instructions()) {
          if (fused_instr->opcode() == HloOpcode::kBitcast &&
              fused_instr->shape().rank() <
                  fused_instr->operand(0)->shape().rank()) {
            if (bitcast != nullptr &&
                (!ShapeUtil::Equal(fused_instr->shape(), bitcast->shape()) ||
                 !ShapeUtil::Equal(bitcast->operand(0)->shape(),
                                   fused_instr->operand(0)->shape()))) {
              same_shape_pattern = false;
              break;
            }
            bitcast = fused_instr;
          }
        }
        if (bitcast == nullptr || !same_shape_pattern) {
          VLOG(2) << "NOT SUPPORTED: Multiple rank-reducing bitcast pattern.";
          continue;
        }

        // 2) Now that we have found a fusion that we want to modify,
        //    create the new fusion. We do so by:
        //    a) Cloning the old fusion.
        //    b) Recursively walk the graph from the root and lift the
        //       bitcast up across one instruction at a time.
        std::unique_ptr<HloInstruction> cloned_fusion =
            fusion->Clone("bitcast");
        // The following stack and set always contain the same data.
        // The stack is used for the order of traversal.
        // The set is used only as an optimization to search in the set.
        std::vector<HloInstruction*> stack(
            {cloned_fusion->fused_expression_root()});
        absl::flat_hash_set<HloInstruction*> set(
            {cloned_fusion->fused_expression_root()});
        bool clone_changed = false;
        while (!stack.empty()) {
          HloInstruction* i = stack.back();
          stack.pop_back();
          set.erase(i);
          if (i->opcode() == HloOpcode::kTuple) {
            stack.insert(stack.end(), i->operands().begin(),
                         i->operands().end());
            set.insert(i->operands().begin(), i->operands().end());
            VLOG(3) << "kTuple: " << i->ToString();
          } else if (i->opcode() == HloOpcode::kParameter &&
                     absl::c_all_of(i->users(), [](HloInstruction* u) {
                       return u->opcode() == HloOpcode::kBitcast;
                     })) {
            VLOG(3) << "kParameter: " << i->ToString();
            // Replace the parameter inside the fusion.
            Shape new_shape = i->users()[0]->shape();
            int64_t parameter_number = i->parameter_number();
            std::string name = i->name();
            auto n = HloInstruction::CreateParameter(parameter_number,
                                                     new_shape, name);
            HloInstruction* new_parameter =
                i->parent()->ReplaceParameter(parameter_number, std::move(n));
            // Remove the old inner bitcast.
            auto old_users = new_parameter->users();
            for (HloInstruction* param_user : old_users) {
              DCHECK(param_user->opcode() == HloOpcode::kBitcast)
                  << "Expected a bitcast";
              TF_RETURN_IF_ERROR(
                  param_user->parent()->ReplaceInstructionWithDifferentShape(
                      param_user, new_parameter));
            }
            // Replace the corresponding fusion operands with a new bitcast.
            HloInstruction* old_outer_parameter =
                cloned_fusion->mutable_operand(parameter_number);
            HloInstruction* new_op =
                old_outer_parameter->parent()->AddInstruction(
                    HloInstruction::CreateBitcast(new_shape,
                                                  old_outer_parameter));
            TF_RETURN_IF_ERROR(cloned_fusion->ReplaceOperandWithDifferentShape(
                parameter_number, new_op));
            clone_changed = true;
            changed = true;
          } else if (i->opcode() == HloOpcode::kBroadcast) {
            VLOG(3) << "kBroadcast: " << i->ToString();
            // For now, do nothing. Later we can merge the broadcast
            // and the bitcast, but this doesn't bring benefit in my
            // current case.
            if (set.insert(i->mutable_operand(0)).second) {
              stack.push_back(i->mutable_operand(0));
            }
          } else if (i->opcode() == HloOpcode::kConstant &&
                     !i->users().empty() &&
                     absl::c_all_of(i->users(), [](HloInstruction* u) {
                       return u->opcode() == HloOpcode::kBitcast;
                     })) {
            // Handling this case is optional for correctness, but
            // handling it clean up the graph.
            VLOG(3) << "kConstant: " << i->ToString();
            Shape new_shape = i->users()[0]->shape();
            TF_RETURN_IF_ERROR(i->parent()->ReplaceWithNewInstruction(
                i, i->CloneWithNewOperands(new_shape, {})));
            clone_changed = true;
            changed = true;
          } else if (!i->users().empty() &&
                     // If 0 operands, we can't lift the bitcast.  It
                     // must be handled manually as kConstant and
                     // kParameter.
                     !i->operands().empty() &&
                     absl::c_all_of(i->users(), [](HloInstruction* u) {
                       return u->opcode() == HloOpcode::kBitcast;
                     })) {
            VLOG(3) << "All User bitcast: " << i->ToString();
            // All users are bitcast, so lift the bitcast.
            Shape new_shape = i->users()[0]->shape();
            std::vector<HloInstruction*> new_operands;
            for (HloInstruction* opnd : i->operands()) {
              Shape dtyped_new_shape = ShapeUtil::ChangeElementType(
                  new_shape, opnd->shape().element_type());
              HloInstruction* new_opnd = opnd->parent()->AddInstruction(
                  HloInstruction::CreateBitcast(dtyped_new_shape, opnd));
              new_operands.push_back(new_opnd);
              // Handle the operand right before the inserted bitcast now.
              if (set.insert(opnd).second) {
                stack.push_back(opnd);
              }
            }
            Shape dtyped_new_shape = ShapeUtil::ChangeElementType(
                new_shape, i->shape().element_type());
            HloInstruction* cloned_i = i->parent()->AddInstruction(
                i->CloneWithNewOperands(dtyped_new_shape, new_operands));
            // Replace the old bitcasts with the new instruction to
            // remove it.
            // Copy the vector as it will be modified while we iterate on it.
            const std::vector<HloInstruction*> users = i->users();
            for (HloInstruction* user : users) {
              TF_RETURN_IF_ERROR(
                  i->parent()->ReplaceInstructionWithDifferentShape(user,
                                                                    cloned_i));
            }
            clone_changed = true;
            changed = true;
          } else {
            VLOG(3) << "Else: " << i->ToString();
            for (auto* opnd : i->operands()) {
              if (set.insert(opnd).second) {
                stack.push_back(opnd);
              }
            }
          }
        }  // while
        DCHECK(clone_changed) << "We should have changed the fusion!";
        std::function<int64_t(const Shape&)> shape_size_func =
            [](const Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfusion_bitcast_liftDTcc mht_2(mht_2_v, 422, "", "./tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.cc", "lambda");
 return ShapeUtil::ByteSizeOf(shape); };
        auto shape_verifier = absl::make_unique<ShapeVerifier>(
            /*layout_sensitive=*/true,
            /*allow_mixed_precision=*/false, shape_size_func);
        if (clone_changed) {
          Status status =
              cloned_fusion->fused_instructions_computation()->Accept(
                  shape_verifier.get());
          if (status.ok()) {
            // 3) Replace the old fusion with the new fusion.
            TF_RETURN_IF_ERROR(fusion->parent()->ReplaceWithNewInstruction(
                fusion, std::move(cloned_fusion)));
          } else {
            VLOG(2) << "Not lifting due to shape problem: "
                    << cloned_fusion->ToString();
          }
        }
      }  // if fusion
    }
  }
  XLA_VLOG_LINES(2, "FusionBitcastLift::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
