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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_dimension_simplifierDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_dimension_simplifierDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_dimension_simplifierDTcc() {
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

#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

// Concat(Concat(A, B), C) => Concat(A, B, C)
StatusOr<bool> ConcatForwarding(HloInstruction* concat) {
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }
  bool changed = false;

  auto parent = concat->parent();
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : concat->operands()) {
    if (operand->opcode() != HloOpcode::kConcatenate ||
        operand->concatenate_dimension() != concat->concatenate_dimension()) {
      new_operands.push_back(operand);
    } else {
      changed = true;
      for (HloInstruction* operand_operand : operand->operands()) {
        new_operands.push_back(operand_operand);
      }
    }
  }
  if (changed) {
    auto new_concat = parent->AddInstruction(HloInstruction::CreateConcatenate(
        concat->shape(), new_operands, concat->concatenate_dimension()));
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(concat, new_concat));
  }
  return changed;
}

// Slice(Concat(A1, A2, ..., An, ...), [n:n+1]) => An
StatusOr<bool> SliceConcatForwarding(HloInstruction* slice) {
  if (slice->opcode() != HloOpcode::kSlice) {
    return false;
  }
  auto concat = slice->mutable_operand(0);
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }

  if (slice->shape().rank() != 1) {
    // Slice concat forwarding only work for size 1 tensor.
    return false;
  }

  int64_t concat_dim = concat->concatenate_dimension();

  std::vector<HloInstruction*> new_operands;
  int64_t size_so_far = 0;
  int64_t slice_size = slice->shape().dimensions(concat_dim);
  if (slice_size != slice->slice_limits(0) - slice->slice_starts(0)) {
    return false;
  }
  if (slice->slice_strides(0) != 1) {
    return false;
  }
  for (HloInstruction* operand : concat->operands()) {
    if (size_so_far == slice->slice_starts(0) &&
        operand->shape().dimensions(0) == slice_size) {
      // Found an operand that can be forwarded.
      TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(operand));
      return true;
    }
    size_so_far += operand->shape().dimensions(concat_dim);
  }

  return false;
}

// Reshape(Broadcast(A, []->[1]), [1]->[]) ==> A
StatusOr<bool> ReshapeBroadcastForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto broadcast = reshape->mutable_operand(0);
  if (broadcast->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  if (reshape->shape().rank() != 0) {
    return false;
  }

  if (broadcast->shape().rank() != 1) {
    return false;
  }

  if (broadcast->mutable_operand(0)->shape().rank() != 0) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(broadcast->mutable_operand(0)));

  return true;
}

// Reshape(Reshape(A, []->[1]), [1]->[]) ==> A
StatusOr<bool> ReshapeReshapeForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto reshape_2 = reshape->mutable_operand(0);
  if (reshape_2->opcode() != HloOpcode::kReshape) {
    return false;
  }

  if (!Shape::Equal()(reshape->shape(), reshape_2->operand(0)->shape())) {
    return false;
  }
  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(reshape_2->mutable_operand(0)));

  return true;
}

// Convert(A, T->T) ==> A
StatusOr<bool> IdentityConvertRemoving(HloInstruction* convert) {
  if (convert->opcode() != HloOpcode::kConvert) {
    return false;
  }
  auto operand = convert->mutable_operand(0);
  if (Shape::Equal()(convert->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(convert->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}

// Reshape(A, S->S) ==> A
StatusOr<bool> IdentityReshapeRemoving(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto operand = reshape->mutable_operand(0);
  if (Shape::Equal()(reshape->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(reshape->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}

}  // namespace

StatusOr<bool> DynamicDimensionSimplifier::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_dimension_simplifierDTcc mht_0(mht_0_v, 336, "", "./tensorflow/compiler/xla/service/dynamic_dimension_simplifier.cc", "DynamicDimensionSimplifier::Run");

  XLA_VLOG_LINES(
      2, "DynamicDimensionSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ConcatForwarding(inst));
      changed |= local_changed;
    }
  }

  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, SliceConcatForwarding(inst));
      changed |= local_changed;
    }
  }

  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ReshapeBroadcastForwarding(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ReshapeReshapeForwarding(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, IdentityConvertRemoving(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, IdentityReshapeRemoving(inst));
      changed |= local_changed;
    }
  }
  XLA_VLOG_LINES(
      2, "DynamicDimensionSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}
}  // namespace xla
